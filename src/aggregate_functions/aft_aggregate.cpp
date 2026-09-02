#include <limits>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/glm_prior_options.hpp"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"

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

static LogicalType GetAftAggResultType(bool compute_inference) {
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

	if (compute_inference) {
		children.push_back(make_pair("std_errors", LogicalType::LIST(LogicalType::DOUBLE)));
		children.push_back(make_pair("z_values", LogicalType::LIST(LogicalType::DOUBLE)));
		children.push_back(make_pair("p_values", LogicalType::LIST(LogicalType::DOUBLE)));
		children.push_back(make_pair("ci_lower", LogicalType::LIST(LogicalType::DOUBLE)));
		children.push_back(make_pair("ci_upper", LogicalType::LIST(LogicalType::DOUBLE)));
		children.push_back(make_pair("intercept_std_error", LogicalType::DOUBLE));
		children.push_back(make_pair("log_scale_std_error", LogicalType::DOUBLE));
	}

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

static void AftAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
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
			target.time_values = std::move(source.time_values);
			target.event_values = std::move(source.event_values);
			target.x_columns = std::move(source.x_columns);
			target.n_features = source.n_features;
			target.initialized = true;
			// Options travel with the data, priors included.
			target.prior_state = std::move(source.prior_state);
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

		if (state.compute_inference) {
			SetDoubleList(*struct_entries[c++], row, inference.std_errors, inference.len);
			SetDoubleList(*struct_entries[c++], row, inference.z_values, inference.len);
			SetDoubleList(*struct_entries[c++], row, inference.p_values, inference.len);
			SetDoubleList(*struct_entries[c++], row, inference.ci_lower, inference.len);
			SetDoubleList(*struct_entries[c++], row, inference.ci_upper, inference.len);
			FlatVector::GetData<double>(*struct_entries[c++])[row] = inference.intercept_std_error;
			FlatVector::GetData<double>(*struct_entries[c++])[row] = inference.log_scale_std_error;
			anofox_free_aft_inference(&inference);
		}

		anofox_free_aft_result(&core);
		state.Reset();
	}
}

static unique_ptr<FunctionData> AftAggBind(ClientContext &context, AggregateFunction &function,
                                           vector<unique_ptr<Expression>> &arguments) {
	auto result = make_uniq<AftAggregateBindData>();

	if (arguments.size() >= 4 && arguments[3]->IsFoldable()) {
		auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[3]);
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

	function.return_type = GetAftAggResultType(result->compute_inference);
	PostHogTelemetry::Instance().RecordFunctionCall("aft_fit_agg");
	return std::move(result);
}

void RegisterAftAggregateFunction(ExtensionLoader &loader) {
	AggregateFunctionSet func_set("aft_fit_agg");

	auto basic = AggregateFunction("aft_fit_agg",
	                               {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE},
	                               LogicalType::ANY, AggregateFunction::StateSize<AftAggregateState>, AftAggInitialize,
	                               AftAggUpdate, AftAggCombine, AftAggFinalize, nullptr, AftAggBind, AftAggDestroy);
	func_set.AddFunction(basic);

	auto with_opts = AggregateFunction(
	    "aft_fit_agg",
	    {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE, LogicalType::ANY},
	    LogicalType::ANY, AggregateFunction::StateSize<AftAggregateState>, AftAggInitialize, AftAggUpdate,
	    AftAggCombine, AftAggFinalize, nullptr, AftAggBind, AftAggDestroy);
	func_set.AddFunction(with_opts);

	CreateAggregateFunctionInfo info(func_set);
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

static void AftCdfFunction(DataChunk &args, ExpressionState &, Vector &result) {
	AftScalarDriver(args, result, [](double t, double eta, double scale, AnofoxAftDistribution d) {
		return anofox_aft_cdf(t, eta, scale, d);
	});
}

static void AftQuantileFunction(DataChunk &args, ExpressionState &, Vector &result) {
	AftScalarDriver(args, result, [](double p, double eta, double scale, AnofoxAftDistribution d) {
		return anofox_aft_quantile(p, eta, scale, d);
	});
}

void RegisterAftScalarFunctions(ExtensionLoader &loader) {
	ScalarFunction cdf("aft_cdf",
	                   {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::VARCHAR},
	                   LogicalType::DOUBLE, AftCdfFunction);
	loader.RegisterFunction(cdf);

	ScalarFunction quantile("aft_quantile",
	                        {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::VARCHAR},
	                        LogicalType::DOUBLE, AftQuantileFunction);
	loader.RegisterFunction(quantile);


}

} // namespace duckdb
