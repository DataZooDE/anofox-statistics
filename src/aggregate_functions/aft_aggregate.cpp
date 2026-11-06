#include <cmath>
#include <limits>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "../include/glm_prior_options.hpp"
#include "../include/cluster_ids.hpp"
#include "../include/ffi_enum_converters.hpp"
#include "../include/map_options_parser.hpp"
#include "../include/model_struct.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "list_input.hpp"
#include "aggregate_finalize_guard.hpp"
#include "../include/canonical_order.hpp"

namespace duckdb {

// Must match AftAugmentResultFFI (crates/anofox-stats-ffi/src/survival_extras_ffi.rs): five pointers, two sizes.
static_assert(offsetof(AnofoxAftAugmentResult, n_rows) == 5 * sizeof(double *),
              "AnofoxAftAugmentResult must match AftAugmentResultFFI");

//===--------------------------------------------------------------------===//
// AFT (accelerated failure time) survival regression with right censoring.
//
//   anofox_stats_aft_fit_agg(time DOUBLE, x LIST(DOUBLE), event DOUBLE [, options])
//   aft_fit_agg(time, x, event, cluster, options): the cluster-robust covariance
//     (survreg(..., cluster = id); upstream AftResult::cluster_vcov)
//
// `event` is 1 when the event was observed and 0 when the row is still censored.
// Coefficients are on the log-time scale.
//===--------------------------------------------------------------------===//
struct AftAggregateState {
	GlmPriorState prior_state;
	vector<double> time_values;
	vector<double> event_values;
	vector<vector<double>> x_columns;
	ClusterRows cluster; // cluster ids of the cluster form; empty otherwise
	idx_t n_features;
	bool initialized;

	AnofoxAftDistribution dist;
	bool fit_intercept;
	uint32_t max_iterations;
	double tolerance;
	bool compute_inference;
	double confidence_level;

	AftAggregateState()
	    : n_features(0), initialized(false), dist(ANOFOX_AFT_WEIBULL), fit_intercept(true), max_iterations(100),
	      tolerance(1e-9), compute_inference(false), confidence_level(0.95) {
	}

	void Reset() {
		prior_state.Clear();
		time_values.clear();
		event_values.clear();
		x_columns.clear();
		cluster.Clear();
		n_features = 0;
		initialized = false;
	}
};

struct AftAggregateBindData : public FunctionData {
	GlmPriorBindData prior_opts;
	AnofoxAftDistribution dist = ANOFOX_AFT_WEIBULL;
	bool fit_intercept = true;
	uint32_t max_iterations = 100;
	double tolerance = 1e-9;
	bool compute_inference = false;
	double confidence_level = 0.95;
	ClusterBind cluster; // aft_fit_agg(time, x, event, cluster, options)

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<AftAggregateBindData>();
		result->cluster = cluster;
		result->prior_opts = prior_opts;
		result->dist = dist;
		result->fit_intercept = fit_intercept;
		result->max_iterations = max_iterations;
		result->tolerance = tolerance;
		result->compute_inference = compute_inference;
		result->confidence_level = confidence_level;
		return std::move(result);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<AftAggregateBindData>();
		return dist == o.dist && fit_intercept == o.fit_intercept && max_iterations == o.max_iterations &&
		       tolerance == o.tolerance && compute_inference == o.compute_inference &&
		       confidence_level == o.confidence_level && prior_opts.Equals(o.prior_opts) && cluster.Equals(o.cluster);
	}
};

static LogicalType GetAftAggResultType(bool clustered = false) {
	child_list_t<LogicalType> children;

	children.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
	children.push_back(make_pair("intercept", LogicalType::DOUBLE));
	children.push_back(make_pair("scale", LogicalType::DOUBLE));
	children.push_back(make_pair("log_likelihood", LogicalType::DOUBLE));
	children.push_back(make_pair("null_log_likelihood", LogicalType::DOUBLE));
	children.push_back(make_pair("aic", LogicalType::DOUBLE));
	children.push_back(make_pair("bic", LogicalType::DOUBLE));
	children.push_back(make_pair("n_observations", LogicalType::BIGINT));
	children.push_back(make_pair("n_events", LogicalType::BIGINT));
	children.push_back(make_pair("n_censored", LogicalType::BIGINT));
	children.push_back(make_pair("n_features", LogicalType::BIGINT));
	children.push_back(make_pair("iterations", LogicalType::INTEGER));
	children.push_back(make_pair("converged", LogicalType::BOOLEAN));

	// Stable shape (#152): the inference fields are always present, NULL
	// without compute_inference.
	AppendCoefficientInferenceFields(children, "z_values", false);
	children.push_back(make_pair("intercept_std_error", LogicalType::DOUBLE));
	children.push_back(make_pair("log_scale_std_error", LogicalType::DOUBLE));
	AppendModelSummaryFields(children);
	if (clustered) {
		// The clustered covariance of (intercept, coefficients, log(scale)), row-major; then
		// vcov_type and n_clusters.
		children.push_back(make_pair("vcov", LogicalType::LIST(LogicalType::DOUBLE)));
		AppendClusterFields(children);
	}

	return LogicalType::STRUCT(std::move(children));
}

static void AftAggInitialize(data_ptr_t state_p) {
	new (state_p) AftAggregateState();
}

static void AftAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (AftAggregateState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~AftAggregateState();
	}
}

static void AftAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &state_vector,
                         idx_t count) {
	auto &bind_data = aggr_input_data.bind_data->Cast<AftAggregateBindData>();

	UnifiedVectorFormat time_data, x_data, event_data;
	inputs[0].ToUnifiedFormat(count, time_data);
	inputs[1].ToUnifiedFormat(count, x_data);
	inputs[2].ToUnifiedFormat(count, event_data);

	auto time_values = UnifiedVectorFormat::GetData<double>(time_data);
	auto event_values = UnifiedVectorFormat::GetData<double>(event_data);
	auto x_list_data = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &x_child = ListVector::GetEntry(inputs[1]);
	auto x_child_data = FlatDataMut<double>(x_child);
	auto &x_child_validity = FlatValidityMut(x_child);

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (AftAggregateState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];

		state.dist = bind_data.dist;
		state.fit_intercept = bind_data.fit_intercept;
		state.max_iterations = bind_data.max_iterations;
		state.tolerance = bind_data.tolerance;
		state.compute_inference = bind_data.compute_inference;
		state.confidence_level = bind_data.confidence_level;

		auto t_idx = time_data.sel->get_index(i);
		auto e_idx = event_data.sel->get_index(i);
		auto x_idx = x_data.sel->get_index(i);
		if (!time_data.validity.RowIsValid(t_idx) || !event_data.validity.RowIsValid(e_idx) ||
		    !x_data.validity.RowIsValid(x_idx)) {
			continue;
		}

		auto list_entry = x_list_data[x_idx];
		// A LIST holding a NULL element is itself valid; skip the row like a NULL list.
		if (ListHasNullElement(x_child_validity, list_entry)) {
			continue;
		}
		vector<string> cluster_ids;
		if (bind_data.cluster.active && !ReadClusterIds(inputs[3].GetValue(i), cluster_ids)) {
			continue; // NULL cluster id: the row is left out
		}
		idx_t n_features = list_entry.length;

		if (!state.initialized) {
			state.n_features = n_features;
			state.prior_state.Materialize(bind_data.prior_opts, n_features, state.fit_intercept);
			state.x_columns.resize(n_features);
			state.initialized = true;
		}
		if (n_features != state.n_features) {
			throw InvalidInputException("Inconsistent feature count: expected %lu, got %lu", state.n_features,
			                            n_features);
		}

		if (bind_data.cluster.active) {
			state.cluster.Push(cluster_ids, "aft_fit_agg");
		}
		state.time_values.push_back(time_values[t_idx]);
		state.event_values.push_back(event_values[e_idx]);
		for (idx_t j = 0; j < n_features; j++) {
			state.x_columns[j].push_back(x_child_data[list_entry.offset + j]);
		}
	}
}

static void AftAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);

	auto sources = (AftAggregateState **)source_data.data;
	auto targets = (AftAggregateState **)target_data.data;

	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];

		if (!source.initialized) {
			continue;
		}

		target.cluster.Append(source.cluster, "aft_fit_agg");
		if (!target.initialized) {
			target.time_values = CombineTake(source.time_values, aggr_input_data);
			target.event_values = CombineTake(source.event_values, aggr_input_data);
			target.x_columns = CombineTake(source.x_columns, aggr_input_data);
			target.n_features = source.n_features;
			target.initialized = true;
			// Options travel with the data, priors included.
			target.prior_state = CombineTake(source.prior_state, aggr_input_data);
			target.dist = source.dist;
			target.fit_intercept = source.fit_intercept;
			target.max_iterations = source.max_iterations;
			target.tolerance = source.tolerance;
			target.compute_inference = source.compute_inference;
			target.confidence_level = source.confidence_level;
			continue;
		}

		if (source.n_features != target.n_features) {
			throw InvalidInputException("Inconsistent feature count during combine");
		}
		target.time_values.insert(target.time_values.end(), source.time_values.begin(), source.time_values.end());
		target.event_values.insert(target.event_values.end(), source.event_values.begin(), source.event_values.end());
		for (idx_t j = 0; j < target.n_features; j++) {
			target.x_columns[j].insert(target.x_columns[j].end(), source.x_columns[j].begin(),
			                           source.x_columns[j].end());
		}
	}
}

//! Copy a double array into a LIST child of the result STRUCT.
static void SetDoubleList(Vector &target, idx_t row, const double *values, idx_t len) {
	auto list_data = FlatDataMut<list_entry_t>(target);
	auto child_offset = ListVector::GetListSize(target);
	ListVector::Reserve(target, child_offset + len);
	auto child_values = FlatDataMut<double>(ListVector::GetEntry(target));
	for (idx_t j = 0; j < len; j++) {
		child_values[child_offset + j] = values[j];
	}
	ListVector::SetListSize(target, child_offset + len);
	list_data[row].offset = child_offset;
	list_data[row].length = len;
}

static void AftAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                           idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<AftAggregateBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (AftAggregateState **)sdata.data;
	auto &struct_entries = StructVector::GetEntries(result);
	ModelStructWriter writer(result);
	Vector *log_scale_se = nullptr;
	Vector *vcov_vec = nullptr; // cluster form only
	{
		auto &types = StructType::GetChildTypes(result.GetType());
		for (idx_t k = 0; k < types.size(); k++) {
			if (types[k].first == "log_scale_std_error") {
				log_scale_se = &StructEntry(struct_entries[k]);
			} else if (types[k].first == "vcov") {
				vcov_vec = &StructEntry(struct_entries[k]);
			}
		}
	}

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		const idx_t row = i + offset;

		if (!state.initialized || state.time_values.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}

		// Canonical row order: bitwise-identical results whatever the thread count (#226).
		const bool clustered = !state.cluster.dims.empty();
		{
			CanonicalRows rows;
			rows.Keys(state.x_columns).Key(state.time_values).Key(state.event_values);
			if (clustered) {
				state.cluster.AddTo(rows);
			}
			rows.Sort();
		}
		AnofoxDataArray time_array {state.time_values.data(), nullptr, state.time_values.size()};
		AnofoxDataArray event_array {state.event_values.data(), nullptr, state.event_values.size()};
		vector<AnofoxDataArray> x_arrays;
		x_arrays.reserve(state.n_features);
		for (idx_t j = 0; j < state.n_features; j++) {
			x_arrays.push_back(AnofoxDataArray {state.x_columns[j].data(), nullptr, state.x_columns[j].size()});
		}

		AnofoxAftOptions options {};
		options.dist = state.dist;
		options.fit_intercept = state.fit_intercept;
		options.max_iterations = state.max_iterations;
		options.tolerance = state.tolerance;
		options.compute_inference = state.compute_inference;
		options.confidence_level = state.confidence_level;
		state.prior_state.Apply(options);

		AnofoxAftFitResultCore core {};
		AnofoxAftInference inference {};
		AnofoxError error;

		auto cluster_ptrs = state.cluster.Pointers();
		vector<int64_t> n_clusters(cluster_ptrs.size(), 0);
		double *cluster_vcov = nullptr;
		size_t cluster_dim = 0;
		bool success =
		    clustered ? anofox_aft_cluster_fit(time_array, x_arrays.data(), x_arrays.size(), event_array, options,
		                                       cluster_ptrs.data(), cluster_ptrs.size(),
		                                       ConvertHcType(bind.cluster.hc_type), bind.cluster.cadjust, &core,
		                                       state.compute_inference ? &inference : nullptr, &cluster_vcov,
		                                       &cluster_dim, n_clusters.data(), &error)
		              : anofox_aft_fit(time_array, x_arrays.data(), x_arrays.size(), event_array, options, &core,
		                               state.compute_inference ? &inference : nullptr, &error);
		if (!success) {
			ThrowUnlessDegenerate("aft_fit_agg", error);
			FlatVector::SetNull(result, row, true);
			state.Reset();
			continue;
		}

		idx_t c = 0;
		SetDoubleList(StructEntry(struct_entries[c++]), row, core.coefficients, core.coefficients_len);
		FlatDataMut<double>(StructEntry(struct_entries[c++]))[row] = core.intercept;
		FlatDataMut<double>(StructEntry(struct_entries[c++]))[row] = core.scale;
		FlatDataMut<double>(StructEntry(struct_entries[c++]))[row] = core.log_likelihood;
		FlatDataMut<double>(StructEntry(struct_entries[c++]))[row] = core.null_log_likelihood;
		FlatDataMut<double>(StructEntry(struct_entries[c++]))[row] = core.aic;
		FlatDataMut<double>(StructEntry(struct_entries[c++]))[row] = core.bic;
		FlatDataMut<int64_t>(StructEntry(struct_entries[c++]))[row] = (int64_t)core.n_observations;
		FlatDataMut<int64_t>(StructEntry(struct_entries[c++]))[row] = (int64_t)core.n_events;
		FlatDataMut<int64_t>(StructEntry(struct_entries[c++]))[row] = (int64_t)core.n_censored;
		FlatDataMut<int64_t>(StructEntry(struct_entries[c++]))[row] = (int64_t)core.n_features;
		FlatDataMut<int32_t>(StructEntry(struct_entries[c++]))[row] = (int32_t)core.iterations;
		FlatDataMut<bool>(StructEntry(struct_entries[c++]))[row] = core.converged;

		// Inference (NULL unless requested), CI fields and the model summary
		if (state.compute_inference) {
			// AFT has its own inference struct; view it in the shared layout.
			AnofoxFitResultInference view = {inference.std_errors, inference.z_values, inference.p_values,
			                                  inference.ci_lower,  inference.ci_upper, inference.len,
			                                  inference.confidence_level, NAN, NAN};
			writer.WriteInference(row, &view);
			FlatDataMut<double>(*log_scale_se)[row] = inference.log_scale_std_error;
		} else {
			writer.WriteInference(row, nullptr);
			FlatVector::SetNull(*log_scale_se, row, true);
		}
		writer.WriteSummary(row, core.summary);
		if (bind.cluster.active) {
			if (cluster_vcov && vcov_vec) {
				auto list_offset = ListVector::GetListSize(*vcov_vec);
				const idx_t len = cluster_dim * cluster_dim;
				ListVector::Reserve(*vcov_vec, list_offset + len);
				auto child = FlatDataMut<double>(ListVector::GetEntry(*vcov_vec));
				for (idx_t k = 0; k < len; k++) {
					child[list_offset + k] = cluster_vcov[k];
				}
				ListVector::SetListSize(*vcov_vec, list_offset + len);
				ListVector::GetData(*vcov_vec)[row] = {list_offset, len};
			} else if (vcov_vec) {
				FlatVector::SetNull(*vcov_vec, row, true);
			}
			WriteClusterFields(result, row, state.compute_inference, bind.cluster.VcovTypeName(), n_clusters);
		}
		if (cluster_vcov) {
			anofox_free_interval_matrix(cluster_vcov);
		}
		if (state.compute_inference) {
			anofox_free_aft_inference(&inference);
		}

		anofox_free_aft_result(&core);
		state.Reset();
	}
}

static unique_ptr<AftAggregateBindData> AftParseBind(ClientContext &context,
                                                     vector<unique_ptr<Expression>> &arguments, const char *name) {
	auto result = make_uniq<AftAggregateBindData>();

	if (arguments.size() >= 4) {
		// aft_fit_agg(time, x, event, cluster, options): the cluster-robust covariance.
		const bool cluster_form = arguments.size() == 5;
		auto opts = RegressionMapOptions::ParseFromExpression(
            context, *arguments[arguments.size() - 1], name,
            {"fit_intercept", "compute_inference", "confidence_level", "max_iterations", "tolerance", "distribution", "feature_names", "prior", "vcov", "hc_type", "cadjust"});
		// survreg(cluster =) / rms::robcov: HC0 without the G / (G - 1) adjustment.
		result->cluster.Load(opts, cluster_form, false, name, "aft_fit_agg(time, x, event, cluster, options)");
		if (result->cluster.active) {
			result->compute_inference = true; // the default of the cluster form, as ols_fit_agg's
		}
		result->prior_opts.LoadFrom(opts);
		if (opts.aft_dist.has_value()) {
			result->dist = (AnofoxAftDistribution)opts.aft_dist.value();
		}
		if (opts.fit_intercept.has_value()) {
			result->fit_intercept = opts.fit_intercept.value();
		}
		if (opts.compute_inference.has_value()) {
			result->compute_inference = opts.compute_inference.value();
		}
		if (opts.confidence_level.has_value()) {
			result->confidence_level = opts.confidence_level.value();
		}
		if (opts.max_iterations.has_value()) {
			result->max_iterations = opts.max_iterations.value();
		}
		if (opts.tolerance.has_value()) {
			result->tolerance = opts.tolerance.value();
		}
	}
	return result;
}

static unique_ptr<FunctionData> AftAggBind(ClientContext &context, AggregateBindFunction &function,
                                           vector<unique_ptr<Expression>> &arguments) {
	auto result = AftParseBind(context, arguments, "aft_fit_agg");
	SetReturnTypeOf(function, GetAftAggResultType(result->cluster.active));
	PostHogTelemetry::Instance().RecordFunctionCall(result->cluster.active ? "aft_fit_agg_cluster" : "aft_fit_agg");
	return std::move(result);
}

//===--------------------------------------------------------------------===//
// aft_augment_agg (#322): one STRUCT per input row, in frame order
//
//   aft_augment_agg(time DOUBLE, x LIST(DOUBLE), event DOUBLE [, options])
//
// Fits the same model as aft_fit_agg over the window frame and returns, per
// row, the linear predictor, the standardized residual (log t - eta) / sigma,
// the Cox-Snell residual -log S(t | x) and the score contributions
// (FittedAft::{standardized_residuals, cox_snell_residuals,
// score_contributions} of anofox-regression). Rows the fit left out get NULLs.
//===--------------------------------------------------------------------===//
struct AftAugmentState {
	vector<double> time;           // NaN for a NULL
	vector<double> event;          // NaN for a NULL
	vector<vector<double>> x_rows; // empty for a NULL x list
	idx_t n_features = 0;
	bool initialized = false;
};

static void AftAugmentInitialize(data_ptr_t state_p) {
	new (state_p) AftAugmentState();
}

static void AftAugmentDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (AftAugmentState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~AftAugmentState();
	}
}

static void AftAugmentUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat time_data, x_data, event_data, sdata;
	inputs[0].ToUnifiedFormat(count, time_data);
	inputs[1].ToUnifiedFormat(count, x_data);
	inputs[2].ToUnifiedFormat(count, event_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto times = UnifiedVectorFormat::GetData<double>(time_data);
	auto events = UnifiedVectorFormat::GetData<double>(event_data);
	auto x_list_data = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &x_child = ListVector::GetEntry(inputs[1]);
	auto x_child_data = FlatDataMut<double>(x_child);
	auto &x_child_validity = FlatValidityMut(x_child);
	auto states = (AftAugmentState **)sdata.data;
	const double nan = std::numeric_limits<double>::quiet_NaN();
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto t_idx = time_data.sel->get_index(i);
		auto e_idx = event_data.sel->get_index(i);
		auto x_idx = x_data.sel->get_index(i);
		vector<double> x_row;
		if (x_data.validity.RowIsValid(x_idx)) {
			auto entry = x_list_data[x_idx];
			if (!state.initialized) {
				state.n_features = entry.length;
				state.initialized = true;
			} else if (entry.length != state.n_features) {
				throw InvalidInputException("aft_augment_agg: inconsistent feature count: expected %llu, got %llu",
				                            (unsigned long long)state.n_features, (unsigned long long)entry.length);
			}
			x_row.resize(entry.length);
			for (idx_t j = 0; j < entry.length; j++) {
				auto c = entry.offset + j;
				// A NULL element makes the row non-finite: it is left out of the fit.
				x_row[j] = x_child_validity.RowIsValid(c) ? x_child_data[c] : nan;
			}
		}
		state.time.push_back(time_data.validity.RowIsValid(t_idx) ? times[t_idx] : nan);
		state.event.push_back(event_data.validity.RowIsValid(e_idx) ? events[e_idx] : nan);
		state.x_rows.push_back(std::move(x_row));
	}
}

static void AftAugmentCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (AftAugmentState **)source_data.data;
	auto targets = (AftAugmentState **)target_data.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (source.initialized && target.initialized && source.n_features != target.n_features) {
			throw InvalidInputException("Inconsistent feature count during combine");
		}
		if (source.initialized && !target.initialized) {
			target.n_features = source.n_features;
			target.initialized = true;
		}
		// Appended in order (the window frame order).
		target.time.insert(target.time.end(), source.time.begin(), source.time.end());
		target.event.insert(target.event.end(), source.event.begin(), source.event.end());
		for (auto &row : source.x_rows) {
			target.x_rows.push_back(row);
		}
	}
}

static LogicalType AftAugmentResultType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("linear_predictor", LogicalType::DOUBLE));
	c.push_back(make_pair("standardized_residual", LogicalType::DOUBLE));
	c.push_back(make_pair("cox_snell_residual", LogicalType::DOUBLE));
	c.push_back(make_pair("score_contributions", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(make_pair("is_training", LogicalType::BOOLEAN));
	return LogicalType::LIST(LogicalType::STRUCT(std::move(c)));
}

static void SetAugmentDouble(Vector &v, idx_t row, double value) {
	if (std::isfinite(value)) {
		FlatDataMut<double>(v)[row] = value;
	} else {
		FlatVector::SetNull(v, row, true);
	}
}

static void AftAugmentFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                               idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<AftAggregateBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (AftAugmentState **)sdata.data;
	const double nan = std::numeric_limits<double>::quiet_NaN();
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		const idx_t row = i + offset;
		if (!state.initialized || state.time.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		const idx_t n = state.time.size();
		const idx_t p = state.n_features;
		vector<vector<double>> columns(p, vector<double>(n));
		for (idx_t r = 0; r < n; r++) {
			for (idx_t j = 0; j < p; j++) {
				columns[j][r] = state.x_rows[r].empty() ? nan : state.x_rows[r][j];
			}
		}
		vector<double> time = state.time;
		vector<double> event = state.event;
		// Canonical row order: bitwise-identical results whatever the thread count (#226).
		auto order = CanonicalRows().Keys(columns).Key(time).Key(event).Sort();

		AnofoxDataArray time_array {time.data(), nullptr, static_cast<size_t>(n)};
		AnofoxDataArray event_array {event.data(), nullptr, static_cast<size_t>(n)};
		vector<AnofoxDataArray> x_arrays;
		for (idx_t j = 0; j < p; j++) {
			x_arrays.push_back(AnofoxDataArray {columns[j].data(), nullptr, static_cast<size_t>(n)});
		}
		GlmPriorState priors;
		priors.Materialize(bind.prior_opts, p, bind.fit_intercept);
		AnofoxAftOptions options {};
		options.dist = bind.dist;
		options.fit_intercept = bind.fit_intercept;
		options.max_iterations = bind.max_iterations;
		options.tolerance = bind.tolerance;
		options.compute_inference = false;
		options.confidence_level = bind.confidence_level;
		priors.Apply(options);

		AnofoxAftAugmentResult aug {};
		AnofoxError error;
		if (!anofox_aft_augment(time_array, x_arrays.data(), x_arrays.size(), event_array, options, &aug, &error)) {
			ThrowUnlessDegenerate("aft_augment_agg", error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		if (aug.n_rows != n) {
			anofox_free_aft_augment_result(&aug);
			throw InternalException("aft_augment_agg: returned %llu rows for %llu inputs", (uint64_t)aug.n_rows,
			                        (uint64_t)n);
		}
		const idx_t k = aug.n_params;
		auto list_offset = ListVector::GetListSize(result);
		ListVector::Reserve(result, list_offset + n);
		ListVector::SetListSize(result, list_offset + n);
		FlatDataMut<list_entry_t>(result)[row] = list_entry_t(list_offset, n);
		auto &e = StructVector::GetEntries(ListVector::GetEntry(result));
		auto &score = StructEntry(e[3]);
		for (idx_t r = 0; r < n; r++) {
			// Sorted row r belongs to input (frame) row order[r].
			const idx_t out = list_offset + (order.empty() ? r : order[r]);
			SetAugmentDouble(StructEntry(e[0]), out, aug.linear_predictor[r]);
			SetAugmentDouble(StructEntry(e[1]), out, aug.standardized_residual[r]);
			SetAugmentDouble(StructEntry(e[2]), out, aug.cox_snell_residual[r]);
			if (!aug.is_training[r]) {
				FlatVector::SetNull(score, out, true);
				FlatDataMut<list_entry_t>(score)[out] = list_entry_t(ListVector::GetListSize(score), 0);
			} else {
				auto score_offset = ListVector::GetListSize(score);
				ListVector::Reserve(score, score_offset + k);
				auto &child = ListVector::GetEntry(score);
				for (idx_t j = 0; j < k; j++) {
					SetAugmentDouble(child, score_offset + j, aug.score_mat[r * k + j]);
				}
				ListVector::SetListSize(score, score_offset + k);
				FlatDataMut<list_entry_t>(score)[out] = list_entry_t(score_offset, k);
			}
			FlatDataMut<bool>(StructEntry(e[4]))[out] = aug.is_training[r];
		}
		anofox_free_aft_augment_result(&aug);
	}
}

static unique_ptr<FunctionData> AftAugmentBind(ClientContext &context, AggregateBindFunction &function,
                                               vector<unique_ptr<Expression>> &arguments) {
	auto result = AftParseBind(context, arguments, "aft_augment_agg");
	SetReturnTypeOf(function, AftAugmentResultType());
	PostHogTelemetry::Instance().RecordFunctionCall("aft_augment_agg");
	return std::move(result);
}

void RegisterAftAggregateFunction(ExtensionLoader &loader) {
	AggregateFunctionSet func_set("aft_fit_agg");

	auto basic = AggregateFunction("aft_fit_agg",
	                               {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE},
	                               LogicalType::ANY, AggregateFunction::StateSize<AftAggregateState>, ANOFOX_AGG_INIT(AftAggInitialize),
	                               ANOFOX_GUARDED_UPDATE(AftAggUpdate, AftAggDestroy, AftAggInitialize), AftAggCombine, ANOFOX_GUARDED_FINALIZE(AftAggFinalize, AftAggDestroy, AftAggInitialize), nullptr, ANOFOX_AGG_BIND(AftAggBind), AftAggDestroy);
	func_set.AddFunction(basic);

	auto with_opts = AggregateFunction(
	    "aft_fit_agg",
	    {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE, LogicalType::ANY},
	    LogicalType::ANY, AggregateFunction::StateSize<AftAggregateState>, ANOFOX_AGG_INIT(AftAggInitialize), ANOFOX_GUARDED_UPDATE(AftAggUpdate, AftAggDestroy, AftAggInitialize),
	    AftAggCombine, ANOFOX_GUARDED_FINALIZE(AftAggFinalize, AftAggDestroy, AftAggInitialize), nullptr, ANOFOX_AGG_BIND(AftAggBind), AftAggDestroy);
	func_set.AddFunction(with_opts);

	// aft_fit_agg(time, x, event, cluster, options): the cluster-robust covariance.
	func_set.AddFunction(AggregateFunction(
	    "aft_fit_agg",
	    {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE, LogicalType::ANY,
	     LogicalType::ANY},
	    LogicalType::ANY, AggregateFunction::StateSize<AftAggregateState>, ANOFOX_AGG_INIT(AftAggInitialize),
	    ANOFOX_GUARDED_UPDATE(AftAggUpdate, AftAggDestroy, AftAggInitialize), AftAggCombine,
	    ANOFOX_GUARDED_FINALIZE(AftAggFinalize, AftAggDestroy, AftAggInitialize), nullptr, ANOFOX_AGG_BIND(AftAggBind),
	    AftAggDestroy));

	CreateAggregateFunctionInfo info(func_set);
	FunctionDescription d1;
	d1.description = "Fits an accelerated failure time (AFT) survival model to right-censored durations (event = 1 observed, 0 censored); coefficients are on the log-time scale.";
	d1.examples = {"aft_fit_agg(duration, [x1, x2], event)"};
	d1.categories = {"survival", "regression"};
	d1.parameter_names = {"time", "x", "event"};
	d1.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE};
	info.descriptions.push_back(std::move(d1));
	FunctionDescription d2;
	d2.description = "Fits an accelerated failure time (AFT) survival model with a MAP of options (distribution, fit_intercept, compute_inference, confidence_level, ...).";
	d2.examples = {"aft_fit_agg(duration, [x1, x2], event, {'distribution': 'lognormal'})"};
	d2.categories = {"survival", "regression"};
	d2.parameter_names = {"time", "x", "event", "options"};
	d2.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE, LogicalType::ANY};
	info.descriptions.push_back(std::move(d2));
	FunctionDescription d3;
	d3.description = "Fits the same AFT model with a cluster-robust coefficient covariance (survreg(..., cluster = "
	                 "id), rms::robcov): std_errors, z tests, intercept_std_error and log_scale_std_error are "
	                 "clustered, plus vcov (intercept, coefficients, log(scale)), vcov_type and n_clusters. cluster: "
	                 "one id per row of any type, or a LIST / STRUCT of ids for multiway clustering. 'hc_type': "
	                 "'hc0' (default) or 'hc1', 'cadjust' (default false, as survreg); with priors the penalized "
	                 "bread is used. compute_inference defaults to true.";
	d3.examples = {"aft_fit_agg(duration, [x1, x2], event, patient_id, MAP {})"};
	d3.categories = {"survival", "regression"};
	d3.parameter_names = {"time", "x", "event", "cluster", "options"};
	d3.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE,
	                      LogicalType::ANY, LogicalType::ANY};
	info.descriptions.push_back(std::move(d3));
	loader.RegisterFunction(info);

	AggregateFunctionSet augment_set("aft_augment_agg");
	auto dlist = LogicalType::LIST(LogicalType::DOUBLE);
	for (bool with_options : {false, true}) {
		vector<LogicalType> args = {LogicalType::DOUBLE, dlist, LogicalType::DOUBLE};
		if (with_options) {
			args.push_back(LogicalType::ANY);
		}
		augment_set.AddFunction(AggregateFunction(
		    "aft_augment_agg", args, LogicalType::ANY, AggregateFunction::StateSize<AftAugmentState>,
		    ANOFOX_AGG_INIT(AftAugmentInitialize),
		    ANOFOX_GUARDED_UPDATE(AftAugmentUpdate, AftAugmentDestroy, AftAugmentInitialize), AftAugmentCombine,
		    ANOFOX_GUARDED_FINALIZE(AftAugmentFinalize, AftAugmentDestroy, AftAugmentInitialize), nullptr,
		    ANOFOX_AGG_BIND(AftAugmentBind), AftAugmentDestroy));
	}
	CreateAggregateFunctionInfo augment_info(augment_set);
	FunctionDescription a1;
	a1.description = "Window aggregate: fits the same AFT model as aft_fit_agg over the frame and returns one STRUCT "
	                 "per input row (in frame order): linear_predictor, standardized_residual ((log t - eta) / "
	                 "scale), cox_snell_residual (-log S(t | x)), score_contributions (intercept, coefficients, "
	                 "log(scale)) and is_training. Use it over a whole partition, as aft_augment_by does.";
	a1.examples = {"aft_augment_agg(duration, [x1, x2], event) OVER ()"};
	a1.categories = {"survival", "regression"};
	a1.parameter_names = {"time", "x", "event"};
	a1.parameter_types = {LogicalType::DOUBLE, dlist, LogicalType::DOUBLE};
	augment_info.descriptions.push_back(a1);
	FunctionDescription a2 = a1;
	a2.description = a1.description + " With the options of aft_fit_agg.";
	a2.examples = {"aft_augment_agg(duration, [x1, x2], event, {'distribution': 'lognormal'}) OVER ()"};
	a2.parameter_names = {"time", "x", "event", "options"};
	a2.parameter_types = {LogicalType::DOUBLE, dlist, LogicalType::DOUBLE, LogicalType::ANY};
	augment_info.descriptions.push_back(a2);
	loader.RegisterFunction(augment_info);
}

//===--------------------------------------------------------------------===//
// Stateless survival / quantile helpers.
//
// These mirror anofox_stats_predict: they take a fitted model's pieces rather
// than the model itself, so they compose with it in plain SQL.
//===--------------------------------------------------------------------===//
static AnofoxAftDistribution ParseAftDistName(const string &raw) {
	string v = StringUtil::Lower(raw);
	if (v == "weibull") {
		return ANOFOX_AFT_WEIBULL;
	}
	if (v == "lognormal" || v == "log_normal" || v == "log-normal") {
		return ANOFOX_AFT_LOGNORMAL;
	}
	if (v == "loglogistic" || v == "log_logistic" || v == "log-logistic") {
		return ANOFOX_AFT_LOGLOGISTIC;
	}
	if (v == "exponential" || v == "exp") {
		return ANOFOX_AFT_EXPONENTIAL;
	}
	throw InvalidInputException("Unknown AFT distribution '%s'. Expected 'weibull', 'lognormal', "
	                            "'loglogistic' or 'exponential'.",
	                            raw);
}

//! Shared driver for the two 4-argument scalar helpers. DuckDB's executor
//! helpers stop at three arguments, so the vectors are walked directly.
template <typename FN>
static void AftScalarDriver(DataChunk &args, Vector &result, FN &&fn) {
	const idx_t count = args.size();
	UnifiedVectorFormat a0, a1, a2, a3;
	args.data[0].ToUnifiedFormat(count, a0);
	args.data[1].ToUnifiedFormat(count, a1);
	args.data[2].ToUnifiedFormat(count, a2);
	args.data[3].ToUnifiedFormat(count, a3);

	auto v0 = UnifiedVectorFormat::GetData<double>(a0);
	auto v1 = UnifiedVectorFormat::GetData<double>(a1);
	auto v2 = UnifiedVectorFormat::GetData<double>(a2);
	auto v3 = UnifiedVectorFormat::GetData<string_t>(a3);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatDataMut<double>(result);
	auto &mask = FlatValidityMut(result);

	for (idx_t i = 0; i < count; i++) {
		auto i0 = a0.sel->get_index(i);
		auto i1 = a1.sel->get_index(i);
		auto i2 = a2.sel->get_index(i);
		auto i3 = a3.sel->get_index(i);
		if (!a0.validity.RowIsValid(i0) || !a1.validity.RowIsValid(i1) || !a2.validity.RowIsValid(i2) ||
		    !a3.validity.RowIsValid(i3)) {
			mask.SetInvalid(i);
			continue;
		}
		out[i] = fn(v0[i0], v1[i1], v2[i2], ParseAftDistName(v3[i3].GetString()));
	}
}

static void AftCdfFunctionImpl(DataChunk &args, ExpressionState &state, Vector &result);

// Constant inputs must yield a CONSTANT_VECTOR (DuckDB constant folding
// asserts this in debug builds).
static void AftCdfFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	// Check before the call: implementations may Flatten() args in place.
	const bool all_constant = args.AllConstant();
	AftCdfFunctionImpl(args, state, result);
	if (all_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void AftCdfFunctionImpl(DataChunk &args, ExpressionState &state, Vector &result) {
	AftScalarDriver(args, result, [](double t, double eta, double scale, AnofoxAftDistribution d) {
		return anofox_aft_cdf(t, eta, scale, d);
	});
}

static void AftQuantileFunctionImpl(DataChunk &args, ExpressionState &state, Vector &result);

// Constant inputs must yield a CONSTANT_VECTOR (DuckDB constant folding
// asserts this in debug builds).
static void AftQuantileFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	// Check before the call: implementations may Flatten() args in place.
	const bool all_constant = args.AllConstant();
	AftQuantileFunctionImpl(args, state, result);
	if (all_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void AftQuantileFunctionImpl(DataChunk &args, ExpressionState &state, Vector &result) {
	AftScalarDriver(args, result, [](double p, double eta, double scale, AnofoxAftDistribution d) {
		return anofox_aft_quantile(p, eta, scale, d);
	});
}

//===--------------------------------------------------------------------===//
// aft_survival(model, x, times) -> [{time, surv}] (#156)
//
// S(t | x) of the fitted model that aft_fit_agg returns, at each of `times`, for
// the covariate row `x` (R psurvreg(t, eta, scale, dist, lower.tail = FALSE)).
//===--------------------------------------------------------------------===//
static LogicalType AftSurvivalResultType() {
	child_list_t<LogicalType> children;
	children.push_back(make_pair("time", LogicalType::DOUBLE));
	children.push_back(make_pair("surv", LogicalType::DOUBLE));
	return LogicalType::LIST(LogicalType::STRUCT(std::move(children)));
}

static const Value *AftModelField(const child_list_t<Value> &fields, const char *name) {
	for (auto &f : fields) {
		if (f.first == name) {
			return &f.second;
		}
	}
	return nullptr;
}

static unique_ptr<FunctionData> AftSurvivalBind(ClientContext &, ScalarBindFunction &bound_function,
                                                vector<unique_ptr<Expression>> &arguments) {
	auto &type = ReturnTypeOf(*arguments[0]);
	if (type.id() != LogicalTypeId::STRUCT) {
		throw InvalidInputException("aft_survival: model must be the STRUCT returned by aft_fit_agg");
	}
	for (auto name : {"coefficients", "intercept", "scale", "family"}) {
		bool found = false;
		for (auto &child : StructType::GetChildTypes(type)) {
			found = found || child.first == name;
		}
		if (!found) {
			throw InvalidInputException("aft_survival: model has no '%s' field; pass the STRUCT returned by "
			                            "aft_fit_agg",
			                            string(name));
		}
	}
	SetArgumentTypeOf(bound_function, 0, type);
	return nullptr;
}

static vector<double> DoubleListValue(const Value &v) {
	vector<double> out;
	for (auto &child : ListValue::GetChildren(v)) {
		out.push_back(child.IsNull() ? NAN : child.GetValue<double>());
	}
	return out;
}

static void AftSurvivalFunctionImpl(DataChunk &args, ExpressionState &state, Vector &result) {
	const idx_t count = args.size();
	result.SetVectorType(VectorType::FLAT_VECTOR);
	ListVector::SetListSize(result, 0);
	auto list_data = FlatDataMut<list_entry_t>(result);
	auto &mask = FlatValidityMut(result);

	for (idx_t i = 0; i < count; i++) {
		auto model = args.data[0].GetValue(i);
		auto x_value = args.data[1].GetValue(i);
		auto times_value = args.data[2].GetValue(i);
		if (model.IsNull() || x_value.IsNull() || times_value.IsNull()) {
			mask.SetInvalid(i);
			continue;
		}
		auto &fields = StructValue::GetChildren(model);
		auto &model_type = model.type();
		child_list_t<Value> named;
		for (idx_t f = 0; f < fields.size(); f++) {
			named.push_back(make_pair(StructType::GetChildName(model_type, f), fields[f]));
		}
		auto coef_v = AftModelField(named, "coefficients");
		auto scale_v = AftModelField(named, "scale");
		auto family_v = AftModelField(named, "family");
		auto intercept_v = AftModelField(named, "intercept");
		if (coef_v->IsNull() || scale_v->IsNull() || family_v->IsNull()) {
			mask.SetInvalid(i);
			continue;
		}
		auto coefficients = DoubleListValue(*coef_v);
		auto x = DoubleListValue(x_value);
		auto times = DoubleListValue(times_value);
		if (x.size() != coefficients.size()) {
			throw InvalidInputException("aft_survival: x has %llu values but the model has %llu coefficients",
			                            (unsigned long long)x.size(), (unsigned long long)coefficients.size());
		}
		double intercept = intercept_v->IsNull() ? NAN : intercept_v->GetValue<double>();
		vector<double> surv(times.size());
		anofox_aft_survival(coefficients.data(), coefficients.size(), intercept, scale_v->GetValue<double>(),
		                    ParseAftDistName(family_v->ToString()), x.data(), times.data(), times.size(),
		                    surv.data());

		auto n = static_cast<idx_t>(times.size());
		auto offset = ListVector::GetListSize(result);
		ListVector::Reserve(result, offset + n);
		auto &entries = StructVector::GetEntries(ListVector::GetEntry(result));
		for (idx_t k = 0; k < n; k++) {
			auto out = offset + k;
			if (std::isnan(times[k])) {
				FlatVector::SetNull(StructEntry(entries[0]), out, true);
			} else {
				FlatDataMut<double>(StructEntry(entries[0]))[out] = times[k];
			}
			if (std::isnan(surv[k])) {
				FlatVector::SetNull(StructEntry(entries[1]), out, true);
			} else {
				FlatDataMut<double>(StructEntry(entries[1]))[out] = surv[k];
			}
		}
		ListVector::SetListSize(result, offset + n);
		list_data[i] = list_entry_t(offset, n);
	}
}

// Constant inputs must yield a CONSTANT_VECTOR (DuckDB constant folding
// asserts this in debug builds).
static void AftSurvivalFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const bool all_constant = args.AllConstant();
	AftSurvivalFunctionImpl(args, state, result);
	if (all_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

void RegisterAftScalarFunctions(ExtensionLoader &loader) {
	ScalarFunction survival("aft_survival",
	                        {LogicalType::ANY, LogicalType::LIST(LogicalType::DOUBLE),
	                         LogicalType::LIST(LogicalType::DOUBLE)},
	                        AftSurvivalResultType(), AftSurvivalFunction, ANOFOX_SCALAR_BIND(AftSurvivalBind));
	MarkFallible(survival);
	{
		CreateScalarFunctionInfo info(survival);
		FunctionDescription d;
		d.description = "Model-based survival curve S(t | x) of a fitted AFT model (the STRUCT returned by "
		                "aft_fit_agg) for the covariate row x, at each of times (R psurvreg(t, eta, scale, dist, "
		                "lower.tail = FALSE)): one {time, surv} per time.";
		d.examples = {"aft_survival(m, [1.0, 0.5], [1, 5, 10, 20])"};
		d.categories = {"survival"};
		d.parameter_names = {"model", "x", "times"};
		d.parameter_types = {LogicalType::ANY, LogicalType::LIST(LogicalType::DOUBLE),
		                     LogicalType::LIST(LogicalType::DOUBLE)};
		info.descriptions.push_back(std::move(d));
		loader.RegisterFunction(std::move(info));
	}

	ScalarFunction cdf("aft_cdf",
	                   {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::VARCHAR},
	                   LogicalType::DOUBLE, AftCdfFunction);
	MarkFallible(cdf);
	{
		CreateScalarFunctionInfo info(cdf);
		FunctionDescription d;
		d.description = "Cumulative distribution P(T <= t) of an AFT survival time with linear predictor eta, scale and distribution (weibull, lognormal, loglogistic, exponential).";
		d.examples = {"aft_cdf(5.0, f.intercept + f.coefficients[1] * x1, f.scale, 'weibull')"};
		d.categories = {"survival"};
		d.parameter_names = {"t", "eta", "scale", "distribution"};
		d.parameter_types = {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::VARCHAR};
		info.descriptions.push_back(std::move(d));
		loader.RegisterFunction(std::move(info));
	}

	ScalarFunction quantile("aft_quantile",
	                        {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::VARCHAR},
	                        LogicalType::DOUBLE, AftQuantileFunction);
	MarkFallible(quantile);
	{
		CreateScalarFunctionInfo info(quantile);
		FunctionDescription d;
		d.description = "p-quantile of an AFT survival time with linear predictor eta, scale and distribution (weibull, lognormal, loglogistic, exponential).";
		d.examples = {"aft_quantile(0.5, f.intercept + f.coefficients[1] * x1, f.scale, 'weibull')"};
		d.categories = {"survival"};
		d.parameter_names = {"p", "eta", "scale", "distribution"};
		d.parameter_types = {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::VARCHAR};
		info.descriptions.push_back(std::move(d));
		loader.RegisterFunction(std::move(info));
	}


}

} // namespace duckdb
