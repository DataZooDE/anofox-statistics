#pragma once

// Cluster ids of the cluster-robust (sandwich) covariance forms:
//   ols_fit_agg(y, x, cluster, options) / wls_fit_agg(y, x, w, cluster, options)   (#309)
//   poisson / binomial / logistic / negbinom / gamma / tweedie_fit_agg(y, x, cluster, {'vcov': 'cluster', ...})
//   aft_fit_agg(time, x, event, cluster, options)
//
// The cluster id is per-row data, so it is an aggregate argument. A scalar of any
// type is one clustering dimension; a LIST or STRUCT of ids (e.g. [firm, year])
// gives multiway clustering, one dimension per element. Labels are
// dictionary-encoded here; the covariance itself is upstream's cluster_vcov.
//
// The GLM forms need 'vcov': 'cluster' because their 4-argument position is also
// the prior weights of poisson / gamma / tweedie_fit_agg (y, x, w, options), and a
// cluster id is as often an integer as a weight is.

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb_compat.hpp"
#include "canonical_order.hpp"
#include "map_options_parser.hpp"

namespace duckdb {

//! One clustering dimension: dense ids per row and their labels.
struct ClusterDim {
	vector<int32_t> ids;
	vector<string> labels;
	unique_ptr<unordered_map<string, int32_t>> index;

	int32_t Intern(const string &key) {
		if (!index) {
			index = make_uniq<unordered_map<string, int32_t>>();
		}
		auto it = index->find(key);
		if (it != index->end()) {
			return it->second;
		}
		auto id = static_cast<int32_t>(labels.size());
		index->emplace(key, id);
		labels.push_back(key);
		return id;
	}

	//! Renumber by sorted label, so the codes do not depend on the row order.
	void Canonicalize() {
		const idx_t g = labels.size();
		vector<idx_t> order(g);
		std::iota(order.begin(), order.end(), 0);
		std::sort(order.begin(), order.end(), [&](idx_t a, idx_t b) { return labels[a] < labels[b]; });
		vector<int32_t> rank(g);
		vector<string> sorted(g);
		for (idx_t r = 0; r < g; r++) {
			rank[order[r]] = static_cast<int32_t>(r);
			sorted[r] = labels[order[r]];
		}
		for (auto &id : ids) {
			id = rank[id];
		}
		labels = std::move(sorted);
		index = make_uniq<unordered_map<string, int32_t>>();
		for (idx_t r = 0; r < g; r++) {
			index->emplace(labels[r], static_cast<int32_t>(r));
		}
	}
};

//! The per-row ids of the cluster argument: one per dimension (LIST / STRUCT
//! elements, or the scalar itself). False when the row has a NULL id.
inline bool ReadClusterIds(const Value &v, vector<string> &out) {
	out.clear();
	if (v.IsNull()) {
		return false;
	}
	const auto type_id = v.type().id();
	if (type_id == LogicalTypeId::LIST || type_id == LogicalTypeId::STRUCT) {
		auto &children = type_id == LogicalTypeId::LIST ? ListValue::GetChildren(v) : StructValue::GetChildren(v);
		for (auto &c : children) {
			if (c.IsNull()) {
				return false;
			}
			out.push_back(c.ToString());
		}
		if (out.empty()) {
			throw InvalidInputException("cluster: a LIST / STRUCT of cluster ids must not be empty");
		}
		return true;
	}
	out.push_back(v.ToString());
	return true;
}

//! The cluster ids of an aggregate state's rows, one ClusterDim per dimension,
//! kept in step with the state's other columns.
struct ClusterRows {
	vector<ClusterDim> dims;
	//! The canonical codes as sort keys (alive until the rows are sorted).
	vector<vector<double>> keys;

	//! Append one row's ids. The first row fixes the number of dimensions.
	void Push(const vector<string> &ids, const char *name) {
		if (dims.empty()) {
			dims.resize(ids.size());
		}
		if (ids.size() != dims.size()) {
			throw InvalidInputException("%s: inconsistent number of cluster dimensions: expected %lu, got %lu", name,
			                            dims.size(), ids.size());
		}
		for (idx_t d = 0; d < ids.size(); d++) {
			dims[d].ids.push_back(dims[d].Intern(ids[d]));
		}
	}

	//! Append the rows of another state (whose labels were interned independently).
	void Append(const ClusterRows &source, const char *name) {
		if (source.dims.empty()) {
			return;
		}
		if (dims.empty()) {
			dims.resize(source.dims.size());
		}
		if (source.dims.size() != dims.size()) {
			throw InvalidInputException("%s: inconsistent number of cluster dimensions: expected %lu, got %lu", name,
			                            dims.size(), source.dims.size());
		}
		for (idx_t d = 0; d < dims.size(); d++) {
			auto &sd = source.dims[d];
			for (auto id : sd.ids) {
				dims[d].ids.push_back(dims[d].Intern(sd.labels[id]));
			}
		}
	}

	void Clear() {
		dims.clear();
		keys.clear();
	}

	//! Renumber the codes by label and add them to `rows` (as keys after the
	//! caller's, and carried along), for a canonical row order (#226).
	void AddTo(CanonicalRows &rows) {
		keys.clear();
		for (auto &d : dims) {
			d.Canonicalize();
			keys.emplace_back(d.ids.begin(), d.ids.end());
		}
		rows.Keys(keys);
		for (auto &d : dims) {
			rows.Carry(d.ids);
		}
	}

	vector<const int32_t *> Pointers() const {
		vector<const int32_t *> out;
		for (auto &d : dims) {
			out.push_back(d.ids.data());
		}
		return out;
	}
};

//! The cluster options of a GLM / AFT fit's bind data.
struct ClusterBind {
	bool active = false;
	HcType hc_type = HcType::NONE; // NONE: HC0, vcovCL's default for GLMs and survreg's
	bool cadjust = true;

	bool Equals(const ClusterBind &o) const {
		return active == o.active && hc_type == o.hc_type && cadjust == o.cadjust;
	}

	//! Read the cluster options of a fit whose options were parsed into `opts`.
	//! `cluster_form`: the call has the cluster argument (GLM: the 4-argument
	//! form with 'vcov': 'cluster'; AFT: the 5-argument form). Resets the prior
	//! covariance type (the clustered covariance replaces it). `default_cadjust`
	//! is vcovCL's (true) for GLMs and survreg's (false) for AFT.
	void Load(RegressionMapOptions &opts, bool cluster_form, bool default_cadjust, const char *name,
	          const char *form) {
		const bool asked = opts.vcov.has_value() && opts.vcov.value() == VcovTypeOpt::CLUSTER;
		if (asked && !cluster_form) {
			throw InvalidInputException("%s: 'vcov': 'cluster' needs the cluster ids: %s", name, form);
		}
		active = cluster_form;
		if (!active) {
			if (opts.hc_type.has_value() || opts.cadjust.has_value()) {
				throw InvalidInputException("%s: 'hc_type' and 'cadjust' apply to the cluster-robust covariance: %s",
				                            name, form);
			}
			return;
		}
		if (opts.vcov.has_value() && !asked) {
			throw InvalidInputException("%s: the cluster form reports the cluster-robust covariance; 'vcov' must be "
			                            "'cluster' or left out",
			                            name);
		}
		opts.vcov.reset();
		hc_type = opts.hc_type.value_or(HcType::NONE);
		if (hc_type == HcType::HC2 || hc_type == HcType::HC3) {
			throw InvalidInputException("%s: hc_type HC2 / HC3 of the cluster-robust covariance are available for "
			                            "linear models only; use 'hc0' (default) or 'hc1'",
			                            name);
		}
		cadjust = opts.cadjust.value_or(default_cadjust);
	}

	const char *VcovTypeName() const {
		return hc_type == HcType::HC1 ? "cluster_HC1" : "cluster_HC0";
	}
};

//! Whether a GLM call is the cluster form f(y, x, cluster, {'vcov': 'cluster', ...}).
inline bool GlmClusterForm(const RegressionMapOptions &opts, const vector<unique_ptr<Expression>> &arguments) {
	return arguments.size() == 4 && opts.vcov.has_value() && opts.vcov.value() == VcovTypeOpt::CLUSTER;
}

//! vcov_type VARCHAR and n_clusters BIGINT[] of the cluster forms, appended last.
inline void AppendClusterFields(child_list_t<LogicalType> &children) {
	children.push_back(MakeChild("vcov_type", LogicalType::VARCHAR));
	children.push_back(MakeChild("n_clusters", LogicalType::LIST(LogicalType::BIGINT)));
}

//! Fill vcov_type and n_clusters (the last two fields) of row `row`; NULL
//! without inference.
inline void WriteClusterFields(Vector &result, idx_t row, bool compute_inference, const char *vcov_type,
                               const vector<int64_t> &n_clusters) {
	auto &entries = StructVector::GetEntries(result);
	const idx_t n = entries.size();
	auto &type_vec = StructEntry(entries[n - 2]);
	auto &count_vec = StructEntry(entries[n - 1]);
	if (!compute_inference) {
		FlatVector::SetNull(type_vec, row, true);
		FlatVector::SetNull(count_vec, row, true);
		return;
	}
	FlatDataMut<string_t>(type_vec)[row] = StringVector::AddString(type_vec, vcov_type);
	auto list_offset = ListVector::GetListSize(count_vec);
	ListVector::Reserve(count_vec, list_offset + n_clusters.size());
	ListVector::SetListSize(count_vec, list_offset + n_clusters.size());
	auto child = FlatDataMut<int64_t>(ListVector::GetEntry(count_vec));
	for (idx_t d = 0; d < n_clusters.size(); d++) {
		child[list_offset + d] = n_clusters[d];
	}
	ListVector::GetData(count_vec)[row] = {list_offset, (idx_t)n_clusters.size()};
}

//! The FunctionDescription of a GLM's cluster form f(y, x, cluster, options).
inline FunctionDescription GlmClusterDescription(const char *name) {
	FunctionDescription d;
	d.description = string("Fits the same model with a cluster-robust (sandwich::vcovCL(glm)) coefficient "
	                       "covariance: std_errors, z tests and vcov are clustered, plus vcov_type and "
	                       "n_clusters. cluster: one id per row of any type, or a LIST / STRUCT of ids for "
	                       "multiway clustering; needs 'vcov': 'cluster' in the options. 'hc_type': 'hc0' "
	                       "(default) or 'hc1', 'cadjust' (default true); with priors or glm_lambda the "
	                       "penalized bread is used. compute_inference defaults to true.");
	d.examples = {string(name) + "(y, [x1, x2], firm_id, {'vcov': 'cluster'})"};
	d.categories = {"regression"};
	d.parameter_names = {"y", "x", "cluster", "options"};
	d.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY,
	                     LogicalType::ANY};
	return d;
}

} // namespace duckdb
