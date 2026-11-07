#pragma once

#include "duckdb.hpp"
#include "duckdb/function/aggregate_state.hpp"
#include "duckdb_compat.hpp"
#include "aggregate_combine.hpp"

#include <algorithm>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace duckdb {

//! A group label of a two-sample test: either an integer or a string.
//!
//! DuckDB relocates aggregate states with a raw memory copy, so nothing inside a
//! state may point into itself. std::string's small-string buffer does, which is
//! why the string label lives behind a pointer.
struct GroupLabel {
	bool is_string = false;
	int64_t int_value = 0;
	shared_ptr<const std::string> str_value;

	static GroupLabel FromInt(int64_t v) {
		GroupLabel l;
		l.int_value = v;
		return l;
	}
	static GroupLabel FromString(std::string v) {
		GroupLabel l;
		l.is_string = true;
		l.str_value = make_shared_ptr<const std::string>(std::move(v));
		return l;
	}

	bool operator==(const GroupLabel &o) const {
		return is_string == o.is_string && (is_string ? *str_value == *o.str_value : int_value == o.int_value);
	}
	bool operator<(const GroupLabel &o) const {
		if (is_string != o.is_string) {
			return !is_string;
		}
		return is_string ? *str_value < *o.str_value : int_value < o.int_value;
	}
	std::string ToString() const {
		return is_string ? "'" + *str_value + "'" : std::to_string(int_value);
	}
};

//! Samples of a two-sample test, keyed by whatever two distinct labels the data
//! uses. Earlier versions treated label 0 as group 1 and every other value as
//! group 2, so data coded 1/2 or with a third group was silently mis-assigned.
//! Group 1 is the sample with the smaller label; a third label is an error.
struct TwoGroupSamples {
	GroupLabel labels[2];
	idx_t n_labels = 0;
	std::vector<double> values[2];

	void Clear() {
		n_labels = 0;
		values[0].clear();
		values[1].clear();
	}

	void Add(const GroupLabel &label, double value, const char *function_name) {
		values[Slot(label, function_name)].push_back(value);
	}

	void Merge(TwoGroupSamples &source, const AggregateInputData &aggr_input_data, const char *function_name) {
		for (idx_t k = 0; k < source.n_labels; k++) {
			auto slot = Slot(source.labels[k], function_name);
			auto &target_values = values[slot];
			if (target_values.empty()) {
				target_values = CombineTake(source.values[k], aggr_input_data);
			} else {
				target_values.insert(target_values.end(), source.values[k].begin(), source.values[k].end());
			}
		}
	}

	//! Sort both samples. The finalizers call this before handing the samples to
	//! the core, so the result (including a seeded resampling draw) is bitwise
	//! independent of the order in which parallel threads delivered the rows.
	void SortValues() {
		std::sort(values[0].begin(), values[0].end());
		std::sort(values[1].begin(), values[1].end());
	}

	//! Sample with the smaller label (empty when fewer than two labels were seen).
	const std::vector<double> &Group1() const {
		return values[FirstSlot()];
	}
	//! Sample with the larger label.
	const std::vector<double> &Group2() const {
		return values[1 - FirstSlot()];
	}

private:
	idx_t FirstSlot() const {
		return (n_labels == 2 && labels[1] < labels[0]) ? 1 : 0;
	}

	idx_t Slot(const GroupLabel &label, const char *function_name) {
		for (idx_t k = 0; k < n_labels; k++) {
			if (labels[k] == label) {
				return k;
			}
		}
		if (n_labels == 2) {
			throw InvalidInputException("%s compares exactly two groups, but found a third group label %s (already "
			                            "saw %s and %s)",
			                            function_name, label.ToString(), labels[0].ToString(), labels[1].ToString());
		}
		labels[n_labels] = label;
		return n_labels++;
	}
};

//! Read a group label from a unified vector of BIGINT or VARCHAR.
//! Dense numeric codes for arbitrary group labels (k-sample tests, #186). The
//! codes are what the FFI groups by. They are handed out in first-seen order,
//! which depends on the row order; Canonicalize renumbers them by label before
//! the FFI call. Relocatable (a vector plus a heap-held index).
//!
//! Few labels are found by a linear scan. Past kIndexThreshold labels a hash
//! index takes over, so coding many distinct keys (e.g. query ids) stays linear
//! in the row count. The index lazily catches up with `labels`, which callers may
//! assign directly (Combine into an empty target), and is rebuilt when `labels`
//! shrinks or is reordered.
struct GroupCoder {
	vector<GroupLabel> labels;

	GroupCoder() = default;
	GroupCoder(const GroupCoder &other) : labels(other.labels) {
	}
	GroupCoder &operator=(const GroupCoder &other) {
		labels = other.labels;
		index.reset();
		return *this;
	}
	GroupCoder(GroupCoder &&) = default;
	GroupCoder &operator=(GroupCoder &&) = default;

	//! Replace or drop all labels. Use these instead of writing `labels` directly,
	//! so the index never describes a stale label set.
	void Assign(vector<GroupLabel> new_labels) {
		labels = std::move(new_labels);
		index.reset();
	}
	void Clear() {
		labels.clear();
		index.reset();
	}

	double Code(const GroupLabel &label) {
		if (labels.size() < kIndexThreshold) {
			for (idx_t k = 0; k < labels.size(); k++) {
				if (labels[k] == label) {
					return static_cast<double>(k);
				}
			}
		} else {
			SyncIndex();
			auto found = index->Find(label);
			if (found != DConstants::INVALID_INDEX) {
				return static_cast<double>(found);
			}
		}
		labels.push_back(label);
		if (index) {
			index->Insert(labels.back(), labels.size() - 1);
		}
		return static_cast<double>(labels.size() - 1);
	}

	//! This coder's code for each of source's codes (adding unseen labels).
	vector<double> Remap(const GroupCoder &source) {
		vector<double> map(source.labels.size());
		for (idx_t k = 0; k < source.labels.size(); k++) {
			map[k] = Code(source.labels[k]);
		}
		return map;
	}

	//! Renumber the labels and `codes` so code k is the k-th smallest label. The
	//! FFI orders the groups by code (and sorts the values within each group), so
	//! the result is then bitwise independent of the row order (#214).
	void Canonicalize(vector<double> &codes) {
		vector<idx_t> order(labels.size());
		std::iota(order.begin(), order.end(), idx_t(0));
		std::sort(order.begin(), order.end(), [&](idx_t a, idx_t b) { return labels[a] < labels[b]; });
		vector<double> rank(labels.size());
		vector<GroupLabel> sorted(labels.size());
		for (idx_t r = 0; r < order.size(); r++) {
			rank[order[r]] = static_cast<double>(r);
			sorted[r] = labels[order[r]];
		}
		for (auto &code : codes) {
			code = rank[static_cast<idx_t>(code)];
		}
		labels = std::move(sorted);
		index.reset();
	}

private:
	static constexpr idx_t kIndexThreshold = 32;

	struct Index {
		std::unordered_map<int64_t, idx_t> ints;
		std::unordered_map<std::string, idx_t> strings;
		idx_t covered = 0; //!< labels[0, covered) are indexed

		idx_t Find(const GroupLabel &label) const {
			if (label.is_string) {
				auto it = strings.find(*label.str_value);
				return it == strings.end() ? DConstants::INVALID_INDEX : it->second;
			}
			auto it = ints.find(label.int_value);
			return it == ints.end() ? DConstants::INVALID_INDEX : it->second;
		}
		void Insert(const GroupLabel &label, idx_t k) {
			if (label.is_string) {
				strings.emplace(*label.str_value, k);
			} else {
				ints.emplace(label.int_value, k);
			}
			covered = k + 1;
		}
	};
	unique_ptr<Index> index;

	void SyncIndex() {
		if (!index || index->covered > labels.size()) {
			index = make_uniq<Index>();
		}
		for (idx_t k = index->covered; k < labels.size(); k++) {
			index->Insert(labels[k], k);
		}
	}
};

inline GroupLabel ReadGroupLabel(const UnifiedVectorFormat &group_data, idx_t idx, bool is_string) {
	if (is_string) {
		return GroupLabel::FromString(UnifiedVectorFormat::GetData<string_t>(group_data)[idx].GetString());
	}
	return GroupLabel::FromInt(UnifiedVectorFormat::GetData<int64_t>(group_data)[idx]);
}

} // namespace duckdb
