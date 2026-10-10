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
#include "../include/map_options_parser.hpp"
#include "../include/model_struct.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "list_input.hpp"
#include "aggregate_finalize_guard.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// AFT (accelerated failure time) survival regression with right censoring.
//
//   anofox_stats_aft_fit_agg(time DOUBLE, x LIST(DOUBLE), event DOUBLE [, options])
//
// `event` is 1 when the event was observed and 0 when the row is still censored.
// Coefficients are on the log-time scale.
//===--------------------------------------------------------------------===//
struct AftAggregateState {
	GlmPriorState prior_state;
	vector<double> time_values;
	vector<double> event_values;
	vector<vector<double>> x_columns;
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

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<AftAggregateBindData>();
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
		       confidence_level == o.confidence_level && prior_opts.Equals(o.prior_opts);
	}
};

static LogicalType GetAftAggResultType() {
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

	return LogicalType::STRUCT(std::move(children));
}

static void AftAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
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
	auto x_list_data = ListVector::GetData(inputs[1]);
	auto &x_child = ListVector::GetEntry(inputs[1]);
	auto x_child_data = FlatVector::GetData<double>(x_child);
	auto &x_child_validity = FlatVector::Validity(x_child);

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
	auto list_data = FlatVector::GetData<list_entry_t>(target);
	auto child_offset = ListVector::GetListSize(target);
	ListVector::Reserve(target, child_offset + len);
	auto child_values = FlatVector::GetData<double>(ListVector::GetEntry(target));
	for (idx_t j = 0; j < len; j++) {
		child_values[child_offset + j] = values[j];
	}
	ListVector::SetListSize(target, child_offset + len);
	list_data[row].offset = child_offset;
	list_data[row].length = len;
}

static void AftAggFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (AftAggregateState **)sdata.data;
	auto &struct_entries = StructVector::GetEntries(result);
	ModelStructWriter writer(result);
	Vector *log_scale_se = nullptr;
	{
		auto &types = StructType::GetChildTypes(result.GetType());
		for (idx_t k = 0; k < types.size(); k++) {
			if (types[k].first == "log_scale_std_error") {
				log_scale_se = struct_entries[k].get();
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

		bool success = anofox_aft_fit(time_array, x_arrays.data(), x_arrays.size(), event_array, options, &core,
		                              state.compute_inference ? &inference : nullptr, &error);
		if (!success) {
			ThrowUnlessDegenerate("aft_fit_agg", error);
			FlatVector::SetNull(result, row, true);
			state.Reset();
			continue;
		}

		idx_t c = 0;
		SetDoubleList(*struct_entries[c++], row, core.coefficients, core.coefficients_len);
		FlatVector::GetData<double>(*struct_entries[c++])[row] = core.intercept;
		FlatVector::GetData<double>(*struct_entries[c++])[row] = core.scale;
		FlatVector::GetData<double>(*struct_entries[c++])[row] = core.log_likelihood;
		FlatVector::GetData<double>(*struct_entries[c++])[row] = core.null_log_likelihood;
		FlatVector::GetData<double>(*struct_entries[c++])[row] = core.aic;
		FlatVector::GetData<double>(*struct_entries[c++])[row] = core.bic;
		FlatVector::GetData<int64_t>(*struct_entries[c++])[row] = (int64_t)core.n_observations;
		FlatVector::GetData<int64_t>(*struct_entries[c++])[row] = (int64_t)core.n_events;
		FlatVector::GetData<int64_t>(*struct_entries[c++])[row] = (int64_t)core.n_censored;
		FlatVector::GetData<int64_t>(*struct_entries[c++])[row] = (int64_t)core.n_features;
		FlatVector::GetData<int32_t>(*struct_entries[c++])[row] = (int32_t)core.iterations;
		FlatVector::GetData<bool>(*struct_entries[c++])[row] = core.converged;

		// Inference (NULL unless requested), CI fields and the model summary
		if (state.compute_inference) {
			// AFT has its own inference struct; view it in the shared layout.
			AnofoxFitResultInference view = {inference.std_errors, inference.z_values, inference.p_values,
			                                  inference.ci_lower,  inference.ci_upper, inference.len,
			                                  inference.confidence_level, NAN, NAN};
			writer.WriteInference(row, &view);
			FlatVector::GetData<double>(*log_scale_se)[row] = inference.log_scale_std_error;
		} else {
			writer.WriteInference(row, nullptr);
			FlatVector::SetNull(*log_scale_se, row, true);
		}
		writer.WriteSummary(row, core.summary);
		if (state.compute_inference) {
			anofox_free_aft_inference(&inference);
		}

		anofox_free_aft_result(&core);
		state.Reset();
	}
}

static unique_ptr<FunctionData> AftAggBind(ClientContext &context, AggregateFunction &function,
                                           vector<unique_ptr<Expression>> &arguments) {
	auto result = make_uniq<AftAggregateBindData>();

	if (arguments.size() >= 4) {
		auto opts = RegressionMapOptions::ParseFromExpression(
            context, *arguments[3], "aft_fit_agg",
            {"fit_intercept", "compute_inference", "confidence_level", "max_iterations", "tolerance", "distribution", "feature_names", "prior", "vcov"});
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

	function.return_type = GetAftAggResultType();
	PostHogTelemetry::Instance().RecordFunctionCall("aft_fit_agg");
	return std::move(result);
}

void RegisterAftAggregateFunction(ExtensionLoader &loader) {
	AggregateFunctionSet func_set("aft_fit_agg");

	auto basic = AggregateFunction("aft_fit_agg",
	                               {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE},
	                               LogicalType::ANY, AggregateFunction::StateSize<AftAggregateState>, AftAggInitialize,
	                               ANOFOX_GUARDED_UPDATE(AftAggUpdate, AftAggDestroy, AftAggInitialize), AftAggCombine, ANOFOX_GUARDED_FINALIZE(AftAggFinalize, AftAggDestroy, AftAggInitialize), nullptr, AftAggBind, AftAggDestroy);
	func_set.AddFunction(basic);

	auto with_opts = AggregateFunction(
	    "aft_fit_agg",
	    {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE, LogicalType::ANY},
	    LogicalType::ANY, AggregateFunction::StateSize<AftAggregateState>, AftAggInitialize, ANOFOX_GUARDED_UPDATE(AftAggUpdate, AftAggDestroy, AftAggInitialize),
	    AftAggCombine, ANOFOX_GUARDED_FINALIZE(AftAggFinalize, AftAggDestroy, AftAggInitialize), nullptr, AftAggBind, AftAggDestroy);
	func_set.AddFunction(with_opts);

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
	loader.RegisterFunction(info);

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
	auto out = FlatVector::GetData<double>(result);
	auto &mask = FlatVector::Validity(result);

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

void RegisterAftScalarFunctions(ExtensionLoader &loader) {
	ScalarFunction cdf("aft_cdf",
	                   {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::VARCHAR},
	                   LogicalType::DOUBLE, AftCdfFunction);
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
