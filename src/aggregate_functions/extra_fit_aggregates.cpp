// Coefficient-returning aggregates for PLS, quantile and isotonic regression.
//
// These methods previously only had *_fit_predict_agg / *_fit_predict_by forms, so
// the fitted model itself could not be retrieved. The aggregates here mirror the
// other *_fit_agg functions: (y, x[, options]) per group, returning a STRUCT.

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/planner/expression.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"
#include "../include/model_struct.hpp"
#include "aggregate_combine.hpp"
#include "error_dispatch.hpp"
#include "list_input.hpp"
#include "xy_rows_aggregate.hpp"
#include "canonical_order.hpp"
#include "telemetry.hpp"

namespace duckdb {

namespace {

// Shared state: complete (y, x) rows of one group (xy_rows_aggregate.hpp).
using ExtraFitState = XyRowsState;
using StatePointers = XyRowsStatePointers;

struct ExtraFitBindData : public FunctionData {
	bool fit_intercept = true;
	size_t n_components = 1;
	double tau = 0.5;
	uint32_t max_iterations = 1000;
	double tolerance = 1e-6;
	bool increasing = true;
	// Quantile inference (#157)
	bool compute_inference = false;
	int32_t se_type = ANOFOX_QUANTILE_SE_NID;
	double confidence_level = 0.95;
	uint32_t n_boot = 200;
	uint64_t seed = 42;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ExtraFitBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<ExtraFitBindData>();
		return fit_intercept == o.fit_intercept && n_components == o.n_components && tau == o.tau &&
		       max_iterations == o.max_iterations && tolerance == o.tolerance && increasing == o.increasing &&
		       compute_inference == o.compute_inference && se_type == o.se_type &&
		       confidence_level == o.confidence_level && n_boot == o.n_boot && seed == o.seed;
	}
};

//! Update for (y DOUBLE, x DOUBLE) — isotonic regression has a single regressor.
void ExtraFitScalarUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat y_data, x_data, sdata;
	inputs[0].ToUnifiedFormat(count, y_data);
	inputs[1].ToUnifiedFormat(count, x_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
	auto x_values = UnifiedVectorFormat::GetData<double>(x_data);
	auto states = (ExtraFitState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto y_idx = y_data.sel->get_index(i);
		auto x_idx = x_data.sel->get_index(i);
		if (!y_data.validity.RowIsValid(y_idx) || !x_data.validity.RowIsValid(x_idx)) {
			continue;
		}
		double y = y_values[y_idx];
		double x = x_values[x_idx];
		if (!std::isfinite(y) || !std::isfinite(x)) {
			continue;
		}
		if (!state.initialized) {
			state.n_features = 1;
			state.x_columns.resize(1);
			state.initialized = true;
		}
		state.y_values.push_back(y);
		state.x_columns[0].push_back(x);
	}
}

const auto ColumnArrays = XyRowsColumnArrays;
const auto SetDoubleList = XyRowsSetDoubleList;
const auto SetDouble = XyRowsSetDouble;

//===--------------------------------------------------------------------===//
// pls_fit_agg
//===--------------------------------------------------------------------===//
LogicalType PlsFitAggResultType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(make_pair("intercept", LogicalType::DOUBLE));
	c.push_back(make_pair("r_squared", LogicalType::DOUBLE));
	c.push_back(make_pair("n_components", LogicalType::BIGINT));
	c.push_back(make_pair("n_observations", LogicalType::BIGINT));
	c.push_back(make_pair("n_features", LogicalType::BIGINT));
	// Stable shape (#152): no coefficient inference for this model, the
	// inference fields are NULL.
	AppendCoefficientInferenceFields(c, "t_values", false, false);
	AppendModelSummaryFields(c);
	return LogicalType::STRUCT(std::move(c));
}

void PlsFitAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                       idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<ExtraFitBindData>();
	StatePointers states(state_vector, count);
	auto &entries = StructVector::GetEntries(result);
	ModelStructWriter writer(result);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		idx_t row = i + offset;
		if (!state.initialized || state.y_values.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		AnofoxDataArray y = {state.y_values.data(), nullptr, state.y_values.size()};
		auto x = ColumnArrays(state);
		AnofoxPlsOptions options;
		options.n_components = bind.n_components;
		options.fit_intercept = bind.fit_intercept;
		AnofoxPlsFitResultCore fit;
		AnofoxError error;
		if (!anofox_pls_fit(y, x.data(), x.size(), options, &fit, &error)) {
			ThrowUnlessDegenerate("pls_fit_agg", error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		SetDoubleList(StructEntry(entries[0]), row, fit.coefficients, fit.coefficients_len);
		SetDouble(StructEntry(entries[1]), row, bind.fit_intercept ? fit.intercept : 0.0);
		SetDouble(StructEntry(entries[2]), row, fit.r_squared);
		FlatDataMut<int64_t>(StructEntry(entries[3]))[row] = (int64_t)fit.n_components;
		FlatDataMut<int64_t>(StructEntry(entries[4]))[row] = (int64_t)fit.n_observations;
		FlatDataMut<int64_t>(StructEntry(entries[5]))[row] = (int64_t)fit.n_features;
		writer.WriteInference(row, nullptr);
		writer.WriteSummary(row, fit.summary);
		anofox_free_pls_result(&fit);
	}
}

//===--------------------------------------------------------------------===//
// quantile_fit_agg
//===--------------------------------------------------------------------===//
LogicalType QuantileFitAggResultType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(make_pair("intercept", LogicalType::DOUBLE));
	c.push_back(make_pair("tau", LogicalType::DOUBLE));
	c.push_back(make_pair("n_observations", LogicalType::BIGINT));
	c.push_back(make_pair("n_features", LogicalType::BIGINT));
	// Stable shape (#152): no coefficient inference for this model, the
	// inference fields are NULL.
	AppendCoefficientInferenceFields(c, "t_values", false, false);
	AppendModelSummaryFields(c);
	return LogicalType::STRUCT(std::move(c));
}

void QuantileFitAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                            idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<ExtraFitBindData>();
	StatePointers states(state_vector, count);
	auto &entries = StructVector::GetEntries(result);
	ModelStructWriter writer(result);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		idx_t row = i + offset;
		if (!state.initialized || state.y_values.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		if (bind.compute_inference && bind.se_type == ANOFOX_QUANTILE_SE_BOOT) {
			// The seeded bootstrap draws rows by position: fix their order.
			SortRowsCanonically(state.y_values, state.x_columns);
		}
		AnofoxDataArray y = {state.y_values.data(), nullptr, state.y_values.size()};
		auto x = ColumnArrays(state);
		AnofoxQuantileOptions options;
		options.tau = bind.tau;
		options.fit_intercept = bind.fit_intercept;
		options.max_iterations = bind.max_iterations;
		options.tolerance = bind.tolerance;
		AnofoxQuantileFitResultCore fit;
		AnofoxError error;
		if (!anofox_quantile_fit(y, x.data(), x.size(), options, &fit, &error)) {
			ThrowUnlessDegenerate("quantile_fit_agg", error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		SetDoubleList(StructEntry(entries[0]), row, fit.coefficients, fit.coefficients_len);
		SetDouble(StructEntry(entries[1]), row, bind.fit_intercept ? fit.intercept : 0.0);
		SetDouble(StructEntry(entries[2]), row, fit.tau);
		FlatDataMut<int64_t>(StructEntry(entries[3]))[row] = (int64_t)fit.n_observations;
		FlatDataMut<int64_t>(StructEntry(entries[4]))[row] = (int64_t)fit.n_features;
		// Coefficient inference (#157): quantreg::summary.rq(se = iid | nid | boot)
		// from upstream's quantile_inference, slopes in the lists and the
		// intercept in the intercept_* fields.
		AnofoxQuantileProcessResult inf;
		bool has_inference = false;
		if (bind.compute_inference) {
			has_inference = anofox_quantile_process(y, x.data(), x.size(), &bind.tau, 1, bind.fit_intercept,
			                                        bind.se_type, bind.seed, bind.n_boot, bind.confidence_level,
			                                        &inf, &error);
			if (!has_inference) {
				ThrowUnlessDegenerate("quantile_fit_agg", error);
			}
		}
		if (has_inference && inf.k == fit.coefficients_len + (bind.fit_intercept ? 1 : 0)) {
			idx_t shift = bind.fit_intercept ? 1 : 0;
			AnofoxFitResultInference slopes;
			slopes.std_errors = inf.std_error + shift;
			slopes.t_values = inf.statistic + shift;
			slopes.p_values = inf.p_value + shift;
			slopes.ci_lower = inf.conf_low + shift;
			slopes.ci_upper = inf.conf_high + shift;
			slopes.len = fit.coefficients_len;
			slopes.confidence_level = inf.conf_level;
			slopes.f_statistic = std::numeric_limits<double>::quiet_NaN();
			slopes.f_pvalue = std::numeric_limits<double>::quiet_NaN();
			if (bind.fit_intercept) {
				fit.summary.intercept_std_error = inf.std_error[0];
				fit.summary.intercept_statistic = inf.statistic[0];
				fit.summary.intercept_p_value = inf.p_value[0];
				fit.summary.intercept_conf_low = inf.conf_low[0];
				fit.summary.intercept_conf_high = inf.conf_high[0];
			}
			writer.WriteInference(row, &slopes);
		} else {
			writer.WriteInference(row, nullptr);
		}
		if (has_inference) {
			anofox_free_quantile_process_result(&inf);
		}
		writer.WriteSummary(row, fit.summary);
		anofox_free_quantile_result(&fit);
	}
}

//===--------------------------------------------------------------------===//
// isotonic_fit_agg
//===--------------------------------------------------------------------===//
// The fitted model is a step function: sorted distinct x values with the fitted
// value at each. Interpolate between knots to predict.
LogicalType IsotonicFitAggResultType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("x", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(make_pair("fitted", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(make_pair("increasing", LogicalType::BOOLEAN));
	c.push_back(make_pair("r_squared", LogicalType::DOUBLE));
	c.push_back(make_pair("n_observations", LogicalType::BIGINT));
	AppendModelSummaryFields(c);
	return LogicalType::STRUCT(std::move(c));
}

void IsotonicFitAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                            idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<ExtraFitBindData>();
	StatePointers states(state_vector, count);
	auto &entries = StructVector::GetEntries(result);
	ModelStructWriter writer(result);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		idx_t row = i + offset;
		if (!state.initialized || state.y_values.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto &xs = state.x_columns[0];
		AnofoxDataArray x = {xs.data(), nullptr, xs.size()};
		AnofoxDataArray y = {state.y_values.data(), nullptr, state.y_values.size()};
		AnofoxIsotonicOptions options;
		options.increasing = bind.increasing;
		AnofoxIsotonicFitResultCore fit;
		AnofoxError error;
		if (!anofox_isotonic_fit(x, y, options, &fit, &error)) {
			ThrowUnlessDegenerate("isotonic_fit_agg", error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		// fitted_values align with the input rows; collapse to one knot per distinct x.
		vector<idx_t> order(xs.size());
		std::iota(order.begin(), order.end(), 0);
		std::stable_sort(order.begin(), order.end(), [&](idx_t a, idx_t b) { return xs[a] < xs[b]; });
		vector<double> knots, fitted;
		for (auto k : order) {
			if (k >= fit.fitted_values_len) {
				continue;
			}
			if (knots.empty() || xs[k] != knots.back()) {
				knots.push_back(xs[k]);
				fitted.push_back(fit.fitted_values[k]);
			}
		}
		SetDoubleList(StructEntry(entries[0]), row, knots.data(), knots.size());
		SetDoubleList(StructEntry(entries[1]), row, fitted.data(), fitted.size());
		FlatDataMut<bool>(StructEntry(entries[2]))[row] = fit.increasing;
		SetDouble(StructEntry(entries[3]), row, fit.r_squared);
		FlatDataMut<int64_t>(StructEntry(entries[4]))[row] = (int64_t)fit.n_observations;
		writer.WriteInference(row, nullptr);
		writer.WriteSummary(row, fit.summary);
		anofox_free_isotonic_result(&fit);
	}
}

//===--------------------------------------------------------------------===//
// Bind
//===--------------------------------------------------------------------===//
//! 'se' (iid | nid | boot) turns inference on; 'compute_inference' alone uses 'nid'
//! (quantreg's default for large n; its rank-inversion default for small n is not
//! available upstream).
void ApplyQuantileInferenceOptions(const RegressionMapOptions &opts, ExtraFitBindData &bind) {
	if (opts.compute_inference.has_value()) {
		bind.compute_inference = opts.compute_inference.value();
	}
	if (opts.se_type.has_value()) {
		bind.compute_inference = true;
		const auto &se = opts.se_type.value();
		bind.se_type = se == "iid" ? ANOFOX_QUANTILE_SE_IID
		               : se == "boot" ? ANOFOX_QUANTILE_SE_BOOT
		                              : ANOFOX_QUANTILE_SE_NID;
	}
	if (opts.confidence_level.has_value()) {
		bind.confidence_level = opts.confidence_level.value();
	}
	if (opts.n_boot.has_value()) {
		bind.n_boot = opts.n_boot.value();
	}
	if (opts.random_state.has_value()) {
		bind.seed = opts.random_state.value();
	}
}

template <const char *NAME>
unique_ptr<FunctionData> ExtraFitBind(ClientContext &context, AggregateBindFunction &, vector<unique_ptr<Expression>> &args) {
	// Format messages with this string, never with NAME itself: MSVC miscompiles
	// a `const char *` template argument bound to the formatter's `const T &`
	// and dereferences the name's characters as a pointer (#162).
	const string name(NAME);
	auto bind = make_uniq<ExtraFitBindData>();
	if (args.size() >= 3) {
		// Each function declares the keys it reads; anything else is rejected.
		// 'quantile' resolves to 'tau' for quantile_fit_agg (contextual alias).
		static const vector<string> pls_keys = {"fit_intercept", "n_components"};
		static const vector<string> quantile_keys = {"fit_intercept", "tau",    "max_iterations", "tolerance",
		                                             "compute_inference", "confidence_level", "se", "n_boot",
		                                             "random_state"};
		static const vector<string> isotonic_keys = {"increasing"};
		const vector<string> &keys =
		    name == "pls_fit_agg" ? pls_keys : (name == "quantile_fit_agg" ? quantile_keys : isotonic_keys);
		auto opts = RegressionMapOptions::ParseFromExpression(context, *args[2], name, keys);
		if (opts.fit_intercept.has_value()) {
			bind->fit_intercept = opts.fit_intercept.value();
		}
		if (opts.n_components.has_value()) {
			bind->n_components = opts.n_components.value();
		}
		if (opts.tau.has_value()) {
			bind->tau = opts.tau.value();
		} else if (opts.quantile.has_value()) {
			bind->tau = opts.quantile.value();
		}
		if (opts.max_iterations.has_value()) {
			bind->max_iterations = opts.max_iterations.value();
		}
		if (opts.tolerance.has_value()) {
			bind->tolerance = opts.tolerance.value();
		}
		if (opts.increasing.has_value()) {
			bind->increasing = opts.increasing.value();
		}
		ApplyQuantileInferenceOptions(opts, *bind);
	}
	if (!(bind->tau > 0.0 && bind->tau < 1.0)) {
		throw InvalidInputException("%s: tau must be in (0, 1), got %g", name, bind->tau);
	}
	if (bind->n_components == 0) {
		throw InvalidInputException("%s: n_components must be at least 1", name);
	}
	PostHogTelemetry::Instance().RecordFunctionCall(name);
	return std::move(bind);
}

constexpr char kPlsName[] = "pls_fit_agg";
constexpr char kQuantileName[] = "quantile_fit_agg";
constexpr char kIsotonicName[] = "isotonic_fit_agg";

void RegisterExtraFit(ExtensionLoader &loader, const char *name, const LogicalType &x_type, aggregate_update_t update,
                      aggregate_finalize_t finalize, bind_aggregate_function_t bind, const LogicalType &result_type,
                      const string &description, const string &example, const string &options_example) {
	AggregateFunctionSet set(AsDuckName(name));
	for (bool with_options : {false, true}) {
		vector<LogicalType> args = {LogicalType::DOUBLE, x_type};
		if (with_options) {
			args.push_back(LogicalType::ANY);
		}
		AggregateFunction fn(AsDuckName(name), args, result_type, AggregateFunction::StateSize<ExtraFitState>, ANOFOX_AGG_INIT(XyRowsInitialize),
		                     update, XyRowsCombine, finalize, nullptr, bind, XyRowsDestroy);
		set.AddFunction(fn);
	}
	CreateAggregateFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	FunctionDescription d1;
	d1.description = description;
	d1.examples = {example};
	d1.categories = {"regression"};
	d1.parameter_names = {"y", "x"};
	d1.parameter_types = {LogicalType::DOUBLE, x_type};
	info.descriptions.push_back(std::move(d1));
	FunctionDescription d2;
	d2.description = description;
	d2.examples = {options_example};
	d2.categories = {"regression"};
	d2.parameter_names = {"y", "x", "options"};
	d2.parameter_types = {LogicalType::DOUBLE, x_type, LogicalType::ANY};
	info.descriptions.push_back(std::move(d2));
	loader.RegisterFunction(std::move(info));
}

} // namespace

void RegisterExtraFitAggregateFunctions(ExtensionLoader &loader) {
	RegisterExtraFit(loader, kPlsName, LogicalType::LIST(LogicalType::DOUBLE), XyRowsUpdate, ANOFOX_AGG_FINALIZE(PlsFitAggFinalize),
	                 ANOFOX_AGG_BIND(ExtraFitBind<kPlsName>), PlsFitAggResultType(),
	                 "Fits a Partial Least Squares regression per group and returns the coefficients, intercept, R^2 "
	                 "and number of components.",
	                 "pls_fit_agg(y, [x1, x2, x3])", "pls_fit_agg(y, [x1, x2, x3], {'n_components': 2})");
	RegisterExtraFit(loader, kQuantileName, LogicalType::LIST(LogicalType::DOUBLE), XyRowsUpdate,
	                 ANOFOX_AGG_FINALIZE(QuantileFitAggFinalize), ANOFOX_AGG_BIND(ExtraFitBind<kQuantileName>), QuantileFitAggResultType(),
	                 "Fits a linear quantile regression per group and returns the coefficients and intercept for the "
	                 "requested quantile tau. With {'compute_inference': true} or {'se': 'iid' | 'nid' | 'boot'} it "
	                 "adds standard errors, t statistics, p-values and conf_low / conf_high (quantreg::summary.rq).",
	                 "quantile_fit_agg(y, [x1, x2])", "quantile_fit_agg(y, [x1, x2], {'tau': 0.9, 'se': 'nid'})");
	RegisterExtraFit(loader, kIsotonicName, LogicalType::DOUBLE, ExtraFitScalarUpdate, ANOFOX_AGG_FINALIZE(IsotonicFitAggFinalize),
	                 ANOFOX_AGG_BIND(ExtraFitBind<kIsotonicName>), IsotonicFitAggResultType(),
	                 "Fits a monotonic (isotonic) regression of y on a single x per group and returns the step "
	                 "function as sorted knots x with fitted values.",
	                 "isotonic_fit_agg(y, x)", "isotonic_fit_agg(y, x, {'increasing': false})");
}

} // namespace duckdb
