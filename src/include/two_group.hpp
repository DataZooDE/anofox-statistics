#pragma once

#include "duckdb.hpp"
#include "duckdb/function/aggregate_state.hpp"
#include "aggregate_combine.hpp"

#include <algorithm>
#include <string>
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

	//! Sort both samples. Seeded resampling tests call this before handing the
	//! samples to the core, so the result does not depend on the order in which
	//! parallel threads delivered the rows.
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
inline GroupLabel ReadGroupLabel(const UnifiedVectorFormat &group_data, idx_t idx, bool is_string) {
	if (is_string) {
		return GroupLabel::FromString(UnifiedVectorFormat::GetData<string_t>(group_data)[idx].GetString());
	}
	return GroupLabel::FromInt(UnifiedVectorFormat::GetData<int64_t>(group_data)[idx]);
}

} // namespace duckdb
