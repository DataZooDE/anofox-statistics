// Cluster-robust (sandwich) covariance of OLS / WLS fits (#309).
//
//   ols_fit_agg(y DOUBLE, x LIST(DOUBLE), cluster ANY, options)
//   wls_fit_agg(y DOUBLE, x LIST(DOUBLE), weight DOUBLE, cluster ANY, options)
//
// The cluster id is per-row data, so it is an aggregate argument (as wls_fit_agg's
// weight), not a constant option. A scalar of any type is one clustering
// dimension; a LIST or STRUCT of ids (e.g. [firm, year] or {'f': firm, 'y': year})
// gives multiway clustering, one dimension per element. Labels are
// dictionary-encoded here as glmm_fit_agg's group; the covariance itself is
// upstream's cluster_vcov (sandwich::vcovCL), reached through
// anofox_ols_cluster_fit / anofox_wls_cluster_fit.
//
// These overloads are added to the ols_fit_agg / wls_fit_agg function sets by their
// Register*AggregateFunction (AddClusterFitOverload), and return the sibling's model STRUCT with
// the cluster-robust std_errors / tests / F test / vcov plus vcov_type and
// n_clusters. The options argument is required to tell the cluster apart from
// the options of the 3-argument ols_fit_agg(y, x, options); compute_inference
// defaults to true here.

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "../include/ffi_enum_converters.hpp"
#include "../include/map_options_parser.hpp"
#include "../include/model_struct.hpp"
#include "../include/cluster_fit.hpp"
#include "telemetry.hpp"
#include "list_input.hpp"
#include "../include/min_obs_guard.hpp"
#include "aggregate_finalize_guard.hpp"
#include "../include/canonical_order.hpp"

namespace duckdb {

namespace {

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

struct ClusterFitState {
	vector<double> y_values;
	vector<vector<double>> x_columns;
	vector<double> weights;
	vector<ClusterDim> dims;
	idx_t n_features;
	bool initialized;

	ClusterFitState() : n_features(0), initialized(false) {
	}

	void Reset() {
		y_values.clear();
		x_columns.clear();
		weights.clear();
		dims.clear();
		n_features = 0;
		initialized = false;
	}
};

struct ClusterFitBindData : public FunctionData {
	bool weighted = false;
	bool fit_intercept = true;
	bool compute_inference = true;
	double confidence_level = 0.95;
	SolverType solver = SolverType::SVD;
	HcType hc_type = HcType::NONE; // NONE = HC1, vcovCL's default for linear models
	bool cadjust = true;
	double alias_tolerance = 0.0;
	bool compute_loo = false;

	unique_ptr<FunctionData> Copy() const override {
		auto r = make_uniq<ClusterFitBindData>();
		r->weighted = weighted;
		r->fit_intercept = fit_intercept;
		r->compute_inference = compute_inference;
		r->confidence_level = confidence_level;
		r->solver = solver;
		r->hc_type = hc_type;
		r->cadjust = cadjust;
		r->alias_tolerance = alias_tolerance;
		r->compute_loo = compute_loo;
		return std::move(r);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<ClusterFitBindData>();
		return weighted == o.weighted && fit_intercept == o.fit_intercept &&
		       compute_inference == o.compute_inference && confidence_level == o.confidence_level &&
		       solver == o.solver && hc_type == o.hc_type && cadjust == o.cadjust &&
		       alias_tolerance == o.alias_tolerance && compute_loo == o.compute_loo;
	}

	const char *Name() const {
		return weighted ? "wls_fit_agg" : "ols_fit_agg";
	}
};

const char *VcovTypeName(HcType hc) {
	switch (hc) {
	case HcType::HC0:
		return "cluster_HC0";
	case HcType::HC2:
		return "cluster_HC2";
	case HcType::HC3:
		return "cluster_HC3";
	default:
		return "cluster_HC1";
	}
}

LogicalType GetClusterFitResultType() {
	child_list_t<LogicalType> children;
	children.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
	children.push_back(make_pair("intercept", LogicalType::DOUBLE));
	children.push_back(make_pair("r_squared", LogicalType::DOUBLE));
	children.push_back(make_pair("adj_r_squared", LogicalType::DOUBLE));
	children.push_back(make_pair("residual_std_error", LogicalType::DOUBLE));
	children.push_back(make_pair("n_observations", LogicalType::BIGINT));
	children.push_back(make_pair("n_features", LogicalType::BIGINT));
	AppendCoefficientInferenceFields(children);
	AppendModelSummaryFields(children);
	AppendCovarianceFields(children);
	AppendLooFields(children);
	// What the std_errors / vcov are, and the clusters per dimension among the fitted rows.
	children.push_back(make_pair("vcov_type", LogicalType::VARCHAR));
	children.push_back(make_pair("n_clusters", LogicalType::LIST(LogicalType::BIGINT)));
	return LogicalType::STRUCT(std::move(children));
}

//! The per-row ids of the cluster argument: one per dimension (LIST / STRUCT
//! elements, or the scalar itself). False when the row has a NULL id.
bool ReadClusterIds(const Value &v, vector<string> &out) {
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

void ClusterFitInitialize(data_ptr_t state_p) {
	new (state_p) ClusterFitState();
}

void ClusterFitDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (ClusterFitState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~ClusterFitState();
	}
}

void ClusterFitUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, Vector &state_vector,
                      idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<ClusterFitBindData>();
	const idx_t cluster_arg = bind.weighted ? 3 : 2;

	UnifiedVectorFormat y_data, x_data, w_data;
	inputs[0].ToUnifiedFormat(count, y_data);
	inputs[1].ToUnifiedFormat(count, x_data);
	if (bind.weighted) {
		inputs[2].ToUnifiedFormat(count, w_data);
	}
	auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
	auto x_list_data = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &x_child = ListVector::GetEntry(inputs[1]);
	auto x_child_data = FlatDataMut<double>(x_child);
	auto &x_child_validity = FlatValidityMut(x_child);
	// Any id type: read through GetValue and rendered to a string, as glmm_fit_agg's group.
	auto &cluster_vec = inputs[cluster_arg];

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (ClusterFitState **)sdata.data;

	vector<string> ids;
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];

		auto y_idx = y_data.sel->get_index(i);
		auto x_idx = x_data.sel->get_index(i);
		if (!y_data.validity.RowIsValid(y_idx) || !x_data.validity.RowIsValid(x_idx)) {
			continue;
		}
		double w_val = 1.0;
		if (bind.weighted) {
			auto w_idx = w_data.sel->get_index(i);
			if (!w_data.validity.RowIsValid(w_idx)) {
				continue;
			}
			w_val = UnifiedVectorFormat::GetData<double>(w_data)[w_idx];
		}
		auto list_entry = x_list_data[x_idx];
		if (ListHasNullElement(x_child_validity, list_entry)) {
			continue;
		}
		if (!ReadClusterIds(cluster_vec.GetValue(i), ids)) {
			continue;
		}

		idx_t n_features = list_entry.length;
		if (!state.initialized) {
			state.n_features = n_features;
			state.x_columns.resize(n_features);
			state.dims.resize(ids.size());
			state.initialized = true;
		}
		if (n_features != state.n_features) {
			throw InvalidInputException("Inconsistent feature count: expected %lu, got %lu", state.n_features,
			                            n_features);
		}
		if (ids.size() != state.dims.size()) {
			throw InvalidInputException("%s: inconsistent number of cluster dimensions: expected %lu, got %lu",
			                            bind.Name(), state.dims.size(), ids.size());
		}

		state.y_values.push_back(y_values[y_idx]);
		if (bind.weighted) {
			state.weights.push_back(w_val);
		}
		for (idx_t d = 0; d < ids.size(); d++) {
			state.dims[d].ids.push_back(state.dims[d].Intern(ids[d]));
		}
		for (idx_t j = 0; j < n_features; j++) {
			state.x_columns[j].push_back(x_child_data[list_entry.offset + j]);
		}
	}
}

void ClusterFitCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                       idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<ClusterFitBindData>();
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (ClusterFitState **)source_data.data;
	auto targets = (ClusterFitState **)target_data.data;

	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (!source.initialized) {
			continue;
		}
		if (!target.initialized) {
			target.n_features = source.n_features;
			target.x_columns.resize(source.n_features);
			target.dims.resize(source.dims.size());
			target.initialized = true;
		}
		if (source.n_features != target.n_features) {
			throw InvalidInputException("Cannot combine states with different feature counts: %lu vs %lu",
			                            source.n_features, target.n_features);
		}
		if (source.dims.size() != target.dims.size()) {
			throw InvalidInputException("%s: inconsistent number of cluster dimensions: expected %lu, got %lu",
			                            bind.Name(), target.dims.size(), source.dims.size());
		}
		// The states interned labels independently: remap the source's ids.
		for (idx_t r = 0; r < source.y_values.size(); r++) {
			target.y_values.push_back(source.y_values[r]);
			if (!source.weights.empty()) {
				target.weights.push_back(source.weights[r]);
			}
			for (idx_t d = 0; d < source.dims.size(); d++) {
				auto &sd = source.dims[d];
				target.dims[d].ids.push_back(target.dims[d].Intern(sd.labels[sd.ids[r]]));
			}
			for (idx_t j = 0; j < target.n_features; j++) {
				target.x_columns[j].push_back(source.x_columns[j][r]);
			}
		}
	}
}

void SetListInResult(Vector &list_vec, idx_t row, const double *data, size_t len) {
	auto &child = ListVector::GetEntry(list_vec);
	auto offset = ListVector::GetListSize(list_vec);
	ListVector::Reserve(list_vec, offset + len);
	ListVector::SetListSize(list_vec, offset + len);
	auto vec_data = FlatDataMut<double>(child);
	for (size_t i = 0; i < len; i++) {
		vec_data[offset + i] = data[i];
	}
	ListVector::GetData(list_vec)[row] = {offset, (idx_t)len};
}

idx_t FieldIndex(const LogicalType &type, const char *name) {
	auto &children = StructType::GetChildTypes(type);
	for (idx_t k = 0; k < children.size(); k++) {
		if (NameString(children[k].first) == name) {
			return k;
		}
	}
	throw InternalException("cluster fit: result field %s missing", name);
}

void ClusterFitFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                        idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<ClusterFitBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (ClusterFitState **)sdata.data;

	auto &struct_entries = StructVector::GetEntries(result);
	ModelStructWriter writer(result);
	auto &vcov_type_vec = StructEntry(struct_entries[FieldIndex(result.GetType(), "vcov_type")]);
	auto &n_clusters_vec = StructEntry(struct_entries[FieldIndex(result.GetType(), "n_clusters")]);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t result_idx = i + offset;

		// Same minimum as the sibling: OLS needs more rows than estimable
		// parameters, WLS at least two rows.
		if (!state.initialized ||
		    (bind.weighted ? state.y_values.size() < 2
		                   : state.y_values.size() <= MinObsForFit(state.x_columns, bind.fit_intercept))) {
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}

		// Canonical row order (#226): codes renumbered by label, then sorted rows.
		vector<vector<double>> cluster_keys;
		for (auto &d : state.dims) {
			d.Canonicalize();
			cluster_keys.emplace_back(d.ids.begin(), d.ids.end());
		}
		CanonicalRows rows;
		rows.Keys(state.x_columns).Key(state.y_values);
		if (bind.weighted) {
			rows.Key(state.weights);
		}
		rows.Keys(cluster_keys);
		for (auto &d : state.dims) {
			rows.Carry(d.ids);
		}
		rows.Sort();

		AnofoxDataArray y_array {state.y_values.data(), nullptr, state.y_values.size()};
		AnofoxDataArray w_array {state.weights.data(), nullptr, state.weights.size()};
		vector<AnofoxDataArray> x_arrays;
		for (auto &col : state.x_columns) {
			x_arrays.push_back(AnofoxDataArray {col.data(), nullptr, col.size()});
		}
		vector<const int32_t *> cluster_ptrs;
		for (auto &d : state.dims) {
			cluster_ptrs.push_back(d.ids.data());
		}
		vector<int64_t> n_clusters(state.dims.size(), 0);

		AnofoxFitResultCore core;
		AnofoxFitResultInference inference;
		AnofoxError error;
		double *vcov = nullptr;
		size_t dim = 0;
		bool success;
		if (bind.weighted) {
			AnofoxWlsOptions options {};
			options.fit_intercept = bind.fit_intercept;
			options.compute_inference = bind.compute_inference;
			options.confidence_level = bind.confidence_level;
			options.solver = ConvertSolverType(bind.solver);
			options.hc_type = ConvertHcType(bind.hc_type);
			options.alias_tolerance = bind.alias_tolerance;
			options.compute_loo = bind.compute_loo;
			success = anofox_wls_cluster_fit(y_array, x_arrays.data(), x_arrays.size(), w_array,
			                                 cluster_ptrs.data(), cluster_ptrs.size(), bind.cadjust, options, &core,
			                                 bind.compute_inference ? &inference : nullptr, &vcov, &dim,
			                                 n_clusters.data(), &error);
		} else {
			AnofoxOlsOptions options {};
			options.fit_intercept = bind.fit_intercept;
			options.compute_inference = bind.compute_inference;
			options.confidence_level = bind.confidence_level;
			options.solver = ConvertSolverType(bind.solver);
			options.hc_type = ConvertHcType(bind.hc_type);
			options.alias_tolerance = bind.alias_tolerance;
			options.compute_loo = bind.compute_loo;
			success = anofox_ols_cluster_fit(y_array, x_arrays.data(), x_arrays.size(), cluster_ptrs.data(),
			                                 cluster_ptrs.size(), bind.cadjust, options, &core,
			                                 bind.compute_inference ? &inference : nullptr, &vcov, &dim,
			                                 n_clusters.data(), &error);
		}
		if (!success) {
			ThrowUnlessDegenerate(bind.Name(), error);
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}

		idx_t k = 0;
		SetListInResult(StructEntry(struct_entries[k++]), result_idx, core.coefficients, core.coefficients_len);
		FlatDataMut<double>(StructEntry(struct_entries[k++]))[result_idx] = core.intercept;
		FlatDataMut<double>(StructEntry(struct_entries[k++]))[result_idx] = core.r_squared;
		FlatDataMut<double>(StructEntry(struct_entries[k++]))[result_idx] = core.adj_r_squared;
		FlatDataMut<double>(StructEntry(struct_entries[k++]))[result_idx] = core.residual_std_error;
		FlatDataMut<int64_t>(StructEntry(struct_entries[k++]))[result_idx] = core.n_observations;
		FlatDataMut<int64_t>(StructEntry(struct_entries[k++]))[result_idx] = core.n_features;

		writer.WriteInference(result_idx, bind.compute_inference ? &inference : nullptr);
		writer.WriteSummary(result_idx, core.summary);
		writer.WriteCovariance(result_idx, vcov, dim,
		                       ResidualDf(core.n_observations, core.coefficients, core.coefficients_len,
		                                  bind.fit_intercept));

		if (bind.compute_inference) {
			FlatDataMut<string_t>(vcov_type_vec)[result_idx] =
			    StringVector::AddString(vcov_type_vec, VcovTypeName(bind.hc_type));
			auto list_offset = ListVector::GetListSize(n_clusters_vec);
			ListVector::Reserve(n_clusters_vec, list_offset + n_clusters.size());
			ListVector::SetListSize(n_clusters_vec, list_offset + n_clusters.size());
			auto child = FlatDataMut<int64_t>(ListVector::GetEntry(n_clusters_vec));
			for (idx_t d = 0; d < n_clusters.size(); d++) {
				child[list_offset + d] = n_clusters[d];
			}
			ListVector::GetData(n_clusters_vec)[result_idx] = {list_offset, (idx_t)n_clusters.size()};
			anofox_free_result_inference(&inference);
		} else {
			FlatVector::SetNull(vcov_type_vec, result_idx, true);
			FlatVector::SetNull(n_clusters_vec, result_idx, true);
		}
		if (vcov) {
			anofox_free_interval_matrix(vcov);
		}
		anofox_free_result_core(&core);
		state.Reset();
	}
}

unique_ptr<FunctionData> ClusterFitBind(ClientContext &context, AggregateBindFunction &function,
                                        vector<unique_ptr<Expression>> &arguments, bool weighted) {
	auto result = make_uniq<ClusterFitBindData>();
	result->weighted = weighted;
	const idx_t options_arg = weighted ? 4 : 3;
	auto opts = RegressionMapOptions::ParseFromExpression(
	    context, *arguments[options_arg], result->Name(),
	    {"fit_intercept", "compute_inference", "confidence_level", "solver", "hc_type", "cadjust", "alias_tolerance",
	     "compute_loo"});
	if (opts.fit_intercept.has_value()) {
		result->fit_intercept = opts.fit_intercept.value();
	}
	if (opts.compute_inference.has_value()) {
		result->compute_inference = opts.compute_inference.value();
	}
	if (opts.confidence_level.has_value()) {
		result->confidence_level = opts.confidence_level.value();
	}
	if (opts.solver.has_value()) {
		result->solver = opts.solver.value();
	}
	if (opts.hc_type.has_value()) {
		result->hc_type = opts.hc_type.value();
	}
	if (opts.cadjust.has_value()) {
		result->cadjust = opts.cadjust.value();
	}
	if (opts.alias_tolerance.has_value()) {
		result->alias_tolerance = opts.alias_tolerance.value();
	}
	if (opts.compute_loo.has_value()) {
		result->compute_loo = opts.compute_loo.value();
	}
	SetReturnTypeOf(function, GetClusterFitResultType());
	PostHogTelemetry::Instance().RecordFunctionCall(weighted ? "wls_fit_agg_cluster" : "ols_fit_agg_cluster");
	return std::move(result);
}

unique_ptr<FunctionData> OlsClusterFitBind(ClientContext &context, AggregateBindFunction &function,
                                           vector<unique_ptr<Expression>> &arguments) {
	return ClusterFitBind(context, function, arguments, false);
}

unique_ptr<FunctionData> WlsClusterFitBind(ClientContext &context, AggregateBindFunction &function,
                                           vector<unique_ptr<Expression>> &arguments) {
	return ClusterFitBind(context, function, arguments, true);
}

} // namespace

void AddClusterFitOverload(AggregateFunctionSet &set, bool weighted) {
	const auto list_double = LogicalType::LIST(LogicalType::DOUBLE);
	vector<LogicalType> args = {LogicalType::DOUBLE, list_double};
	if (weighted) {
		args.push_back(LogicalType::DOUBLE);
	}
	args.push_back(LogicalType::ANY); // cluster
	args.push_back(LogicalType::ANY); // options
	set.AddFunction(AggregateFunction(
	    weighted ? "wls_fit_agg" : "ols_fit_agg", args, LogicalType::ANY, AggregateFunction::StateSize<ClusterFitState>,
	    ANOFOX_AGG_INIT(ClusterFitInitialize),
	    ANOFOX_GUARDED_UPDATE(ClusterFitUpdate, ClusterFitDestroy, ClusterFitInitialize), ClusterFitCombine,
	    ANOFOX_GUARDED_FINALIZE(ClusterFitFinalize, ClusterFitDestroy, ClusterFitInitialize), nullptr,
	    weighted ? ANOFOX_AGG_BIND(WlsClusterFitBind) : ANOFOX_AGG_BIND(OlsClusterFitBind), ClusterFitDestroy));
}

FunctionDescription ClusterFitDescription(bool weighted) {
	const auto list_double = LogicalType::LIST(LogicalType::DOUBLE);
	FunctionDescription d;
	d.description = string(weighted ? "Fits WLS" : "Fits OLS") +
	                " with a cluster-robust (sandwich::vcovCL) coefficient covariance: std_errors, tests, the F test "
	                "and vcov are clustered. cluster: one id per row of any type, or a LIST / STRUCT of ids for "
	                "multiway clustering. Options as the sibling plus 'cadjust' (default true); 'hc_type' picks "
	                "vcovCL's type (default HC1, i.e. Stata's CR1); compute_inference defaults to true.";
	d.categories = {"regression"};
	if (weighted) {
		d.examples = {"wls_fit_agg(y, [x1, x2], w, [firm_id, year], MAP {})"};
		d.parameter_names = {"y", "x", "weight", "cluster", "options"};
		d.parameter_types = {LogicalType::DOUBLE, list_double, LogicalType::DOUBLE, LogicalType::ANY,
		                     LogicalType::ANY};
	} else {
		d.examples = {"ols_fit_agg(y, [x1, x2], firm_id, {'hc_type': 'hc1'})"};
		d.parameter_names = {"y", "x", "cluster", "options"};
		d.parameter_types = {LogicalType::DOUBLE, list_double, LogicalType::ANY, LogicalType::ANY};
	}
	return d;
}

} // namespace duckdb
