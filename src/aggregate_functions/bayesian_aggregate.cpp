// Bayesian Ridge and ARD regression (#61):
//
//   bayesian_ridge_fit_agg(y, x[, options])          -> model STRUCT
//   ard_fit_agg(y, x[, options])                     -> model STRUCT
//   bayesian_ridge_fit_predict_agg(y, x[, options])  -> LIST of per-row predictions
//   ard_fit_predict_agg(y, x[, options])             -> LIST of per-row predictions
//
// Both models are upstream's BayesianRidge / ArdRegression (scikit-learn's
// BayesianRidge / ARDRegression). The fit_predict aggregates bound each
// prediction with the posterior predictive interval at confidence_level.

#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/planner/expression.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/fit_predict_state.hpp"
#include "../include/map_options_parser.hpp"
#include "../include/model_struct.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"
#include "error_dispatch.hpp"
#include "list_input.hpp"
#include "prediction_interval.hpp"
#include "telemetry.hpp"
#include "xy_rows_aggregate.hpp"

namespace duckdb {

namespace {

struct BayesianBindData : public FitPredictRowsBindBase {
	bool ard = false;
	AnofoxBayesianOptions options = {true, 300, 1e-3, 1e-6, 1e-6, 1e-6, 1e-6, 10000.0, 0.95};
	double confidence_level = 0.95;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<BayesianBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<BayesianBindData>();
		return ard == o.ard && options.fit_intercept == o.options.fit_intercept &&
		       options.max_iterations == o.options.max_iterations && options.tolerance == o.options.tolerance &&
		       options.alpha_1 == o.options.alpha_1 && options.alpha_2 == o.options.alpha_2 &&
		       options.lambda_1 == o.options.lambda_1 && options.lambda_2 == o.options.lambda_2 &&
		       options.threshold_lambda == o.options.threshold_lambda && confidence_level == o.confidence_level &&
		       use_split_col == o.use_split_col && null_policy == o.null_policy;
	}
};

const char *FitName(bool ard) {
	return ard ? "ard_fit_agg" : "bayesian_ridge_fit_agg";
}

const char *FitPredictName(bool ard) {
	return ard ? "ard_fit_predict_agg" : "bayesian_ridge_fit_predict_agg";
}

//! Parse the options MAP (each function declares the keys it reads).
void ParseBayesianOptions(ClientContext &context, Expression &expr, BayesianBindData &bind, const string &name,
                          bool fit_predict) {
	vector<string> keys = {"fit_intercept", "max_iterations", "tolerance", "alpha_1",
	                       "alpha_2",       "lambda_1",       "lambda_2"};
	if (bind.ard) {
		keys.push_back("threshold_lambda");
	}
	// The level of the coefficient credible intervals (fit) or of the
	// posterior predictive interval (fit-predict).
	keys.push_back("confidence_level");
	if (fit_predict) {
		keys.push_back("null_policy");
	}
	auto opts = RegressionMapOptions::ParseFromExpression(context, expr, name, keys);
	auto &o = bind.options;
	if (opts.fit_intercept.has_value()) {
		o.fit_intercept = opts.fit_intercept.value();
	}
	if (opts.max_iterations.has_value()) {
		o.max_iterations = opts.max_iterations.value();
	}
	if (opts.tolerance.has_value()) {
		o.tolerance = opts.tolerance.value();
	}
	if (opts.alpha_1.has_value()) {
		o.alpha_1 = opts.alpha_1.value();
	}
	if (opts.alpha_2.has_value()) {
		o.alpha_2 = opts.alpha_2.value();
	}
	if (opts.lambda_1.has_value()) {
		o.lambda_1 = opts.lambda_1.value();
	}
	if (opts.lambda_2.has_value()) {
		o.lambda_2 = opts.lambda_2.value();
	}
	if (opts.threshold_lambda.has_value()) {
		o.threshold_lambda = opts.threshold_lambda.value();
	}
	if (opts.null_policy.has_value()) {
		bind.null_policy = opts.null_policy.value();
	}
	if (opts.confidence_level.has_value()) {
		bind.confidence_level = opts.confidence_level.value();
		o.confidence_level = bind.confidence_level;
	}
}

void ValidateBayesianOptions(const BayesianBindData &bind, const string &name) {
	auto &o = bind.options;
	for (double v : {o.alpha_1, o.alpha_2, o.lambda_1, o.lambda_2}) {
		if (!(std::isfinite(v) && v >= 0.0)) {
			throw InvalidInputException("%s: alpha_1, alpha_2, lambda_1 and lambda_2 must be finite and >= 0", name);
		}
	}
	if (!(std::isfinite(o.tolerance) && o.tolerance > 0.0)) {
		throw InvalidInputException("%s: tolerance must be > 0", name);
	}
	if (o.max_iterations == 0) {
		throw InvalidInputException("%s: max_iterations must be at least 1", name);
	}
	if (!(o.threshold_lambda > 0.0)) {
		throw InvalidInputException("%s: threshold_lambda must be > 0", name);
	}
	if (!(bind.confidence_level > 0.0 && bind.confidence_level < 1.0)) {
		throw InvalidInputException("%s: confidence_level must be in (0, 1)", name);
	}
}

//===--------------------------------------------------------------------===//
// bayesian_ridge_fit_agg / ard_fit_agg
//===--------------------------------------------------------------------===//
LogicalType BayesianFitResultType(bool ard) {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(make_pair("intercept", LogicalType::DOUBLE));
	c.push_back(make_pair("r_squared", LogicalType::DOUBLE));
	c.push_back(make_pair("n_observations", LogicalType::BIGINT));
	c.push_back(make_pair("n_features", LogicalType::BIGINT));
	c.push_back(make_pair("noise_precision", LogicalType::DOUBLE));
	if (ard) {
		c.push_back(make_pair("weight_precisions", LogicalType::LIST(LogicalType::DOUBLE)));
	} else {
		c.push_back(make_pair("weight_precision", LogicalType::DOUBLE));
	}
	c.push_back(make_pair("posterior_sd", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(make_pair("iterations", LogicalType::BIGINT));
	if (!ard) {
		c.push_back(make_pair("log_marginal_likelihood", LogicalType::DOUBLE));
	}
	// Stable shape (#152). The inference fields carry the posterior (#222):
	// std_errors = posterior_sd, conf_low / conf_high the central credible
	// interval at conf_level; no test, so t_values / p_values are NULL.
	AppendCoefficientInferenceFields(c, "t_values", false, false);
	AppendModelSummaryFields(c);
	return LogicalType::STRUCT(std::move(c));
}

void BayesianFitFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                         idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<BayesianBindData>();
	XyRowsStatePointers states(state_vector, count);
	auto &entries = StructVector::GetEntries(result);
	ModelStructWriter writer(result);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		idx_t row = i + offset;
		if (!state.initialized || state.y_values.size() < 2) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		AnofoxDataArray y = {state.y_values.data(), nullptr, state.y_values.size()};
		auto x = XyRowsColumnArrays(state);
		AnofoxBayesianFitResult fit;
		AnofoxError error;
		if (!anofox_bayesian_fit(y, x.data(), x.size(), bind.options, bind.ard, &fit, &error)) {
			ThrowUnlessDegenerate(FitName(bind.ard), error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		idx_t k = 0;
		XyRowsSetDoubleList(StructEntry(entries[k++]), row, fit.coefficients, static_cast<idx_t>(fit.coefficients_len));
		XyRowsSetDouble(StructEntry(entries[k++]), row, bind.options.fit_intercept ? fit.intercept : 0.0);
		XyRowsSetDouble(StructEntry(entries[k++]), row, fit.r_squared);
		FlatDataMut<int64_t>(StructEntry(entries[k++]))[row] = static_cast<int64_t>(fit.n_observations);
		FlatDataMut<int64_t>(StructEntry(entries[k++]))[row] = static_cast<int64_t>(fit.n_features);
		XyRowsSetDouble(StructEntry(entries[k++]), row, fit.noise_precision);
		if (bind.ard) {
			XyRowsSetDoubleList(StructEntry(entries[k++]), row, fit.weight_precisions,
			                    static_cast<idx_t>(fit.weight_precisions_len));
		} else {
			XyRowsSetDouble(StructEntry(entries[k++]), row, fit.weight_precision);
		}
		XyRowsSetDoubleList(StructEntry(entries[k++]), row, fit.posterior_sd, static_cast<idx_t>(fit.coefficients_len));
		FlatDataMut<int64_t>(StructEntry(entries[k++]))[row] = static_cast<int64_t>(fit.iterations);
		if (!bind.ard) {
			XyRowsSetDouble(StructEntry(entries[k++]), row, fit.log_marginal_likelihood);
		}
		// Credible intervals, not a test: NULL t_values / p_values.
		AnofoxFitResultInference credible = {};
		credible.std_errors = fit.posterior_sd;
		credible.ci_lower = fit.credible_lower;
		credible.ci_upper = fit.credible_upper;
		credible.len = fit.coefficients_len;
		credible.confidence_level = fit.confidence_level;
		credible.f_statistic = std::nan("");
		credible.f_pvalue = std::nan("");
		writer.WriteInference(row, &credible);
		writer.WriteSummary(row, fit.summary);
		anofox_free_bayesian_result(&fit);
	}
}

template <bool ARD>
unique_ptr<FunctionData> BayesianFitBind(ClientContext &context, AggregateBindFunction &function,
                                         vector<unique_ptr<Expression>> &arguments) {
	const string name(FitName(ARD));
	auto bind = make_uniq<BayesianBindData>();
	bind->ard = ARD;
	if (arguments.size() >= 3) {
		ParseBayesianOptions(context, *arguments[2], *bind, name, false);
	}
	ValidateBayesianOptions(*bind, name);
	SetReturnTypeOf(function, BayesianFitResultType(ARD));
	PostHogTelemetry::Instance().RecordFunctionCall(name);
	return std::move(bind);
}

//===--------------------------------------------------------------------===//
// bayesian_ridge_fit_predict_agg / ard_fit_predict_agg
//===--------------------------------------------------------------------===//
void BayesianPredictFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                             idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<BayesianBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (FitPredictRowsState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t result_idx = i + offset;
		if (!state.initialized || state.y_train.size() < 2) {
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}
		PadNullXRows(state);
		const idx_t n_rows = state.y_all.size();
		const idx_t p = state.n_features;
		vector<double> rows(n_rows * p);
		for (idx_t r = 0; r < n_rows; r++) {
			for (idx_t j = 0; j < p; j++) {
				rows[r * p + j] = state.x_all[r][j];
			}
		}
		AnofoxDataArray y = {state.y_train.data(), nullptr, state.y_train.size()};
		auto x = FitPredictTrainColumns(state);
		vector<double> yhat(n_rows), lower(n_rows), upper(n_rows);
		AnofoxError error;
		if (!anofox_bayesian_fit_predict(y, x.data(), x.size(), rows.data(), static_cast<size_t>(n_rows),
		                                 bind.options, bind.ard, bind.confidence_level, yhat.data(), lower.data(),
		                                 upper.data(), &error)) {
			ThrowUnlessDegenerate(FitPredictName(bind.ard), error);
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}

		WriteFitPredictRows(result, result_idx, state, yhat.data(), lower.data(), upper.data());
	}
}

template <bool ARD, bool SPLIT>
unique_ptr<FunctionData> BayesianPredictBind(ClientContext &context, AggregateBindFunction &function,
                                             vector<unique_ptr<Expression>> &arguments) {
	const string name(FitPredictName(ARD));
	auto bind = make_uniq<BayesianBindData>();
	bind->ard = ARD;
	bind->use_split_col = SPLIT;
	const idx_t options_idx = SPLIT ? 3 : 2;
	if (arguments.size() > options_idx) {
		ParseBayesianOptions(context, *arguments[options_idx], *bind, name, true);
	}
	ValidateBayesianOptions(*bind, name);
	SetReturnTypeOf(function, FitPredictRowsResultType());
	PostHogTelemetry::Instance().RecordFunctionCall(name);
	return std::move(bind);
}

template <bool ARD>
void RegisterBayesianModel(ExtensionLoader &loader, const string &label, const string &fit_description) {
	const auto x_type = LogicalType::LIST(LogicalType::DOUBLE);
	const string fit_name = FitName(ARD);
	const string predict_name = FitPredictName(ARD);

	AggregateFunctionSet fit_set(AsDuckName(fit_name));
	for (bool with_options : {false, true}) {
		vector<LogicalType> args = {LogicalType::DOUBLE, x_type};
		if (with_options) {
			args.push_back(LogicalType::ANY);
		}
		AggregateFunction fn(AsDuckName(fit_name), args, LogicalType::ANY, AggregateFunction::StateSize<XyRowsState>,
		                     ANOFOX_AGG_INIT(XyRowsInitialize), XyRowsUpdate, XyRowsCombine,
		                     ANOFOX_GUARDED_FINALIZE(BayesianFitFinalize, XyRowsDestroy, XyRowsInitialize), nullptr,
		                     ANOFOX_AGG_BIND(BayesianFitBind<ARD>), XyRowsDestroy);
		fit_set.AddFunction(fn);
	}
	CreateAggregateFunctionInfo fit_info(std::move(fit_set));
	fit_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	FunctionDescription f1;
	f1.description = fit_description;
	f1.examples = {fit_name + "(y, [x1, x2, x3])"};
	f1.categories = {"regression", "bayesian"};
	f1.parameter_names = {"y", "x"};
	f1.parameter_types = {LogicalType::DOUBLE, x_type};
	fit_info.descriptions.push_back(std::move(f1));
	FunctionDescription f2;
	f2.description = fit_description;
	f2.examples = {fit_name + (ARD ? "(y, [x1, x2, x3], {'threshold_lambda': 1000})"
	                               : "(y, [x1, x2, x3], {'max_iterations': 500})")};
	f2.categories = {"regression", "bayesian"};
	f2.parameter_names = {"y", "x", "options"};
	f2.parameter_types = {LogicalType::DOUBLE, x_type, LogicalType::ANY};
	fit_info.descriptions.push_back(std::move(f2));
	loader.RegisterFunction(std::move(fit_info));

	AggregateFunctionSet predict_set(AsDuckName(predict_name));
	const vector<vector<LogicalType>> signatures = {{LogicalType::DOUBLE, x_type},
	                                                {LogicalType::DOUBLE, x_type, LogicalType::ANY},
	                                                {LogicalType::DOUBLE, x_type, LogicalType::VARCHAR},
	                                                {LogicalType::DOUBLE, x_type, LogicalType::VARCHAR, LogicalType::ANY}};
	for (auto &args : signatures) {
		bool split = args.size() >= 3 && args[2].id() == LogicalTypeId::VARCHAR;
		AggregateFunction fn(
		    AsDuckName(predict_name), args, LogicalType::ANY, AggregateFunction::StateSize<FitPredictRowsState>,
		    ANOFOX_AGG_INIT(FitPredictRowsInitialize),
		    ANOFOX_GUARDED_UPDATE(FitPredictRowsUpdate, FitPredictRowsDestroy, FitPredictRowsInitialize),
		    FitPredictRowsCombine,
		    ANOFOX_GUARDED_FINALIZE(BayesianPredictFinalize, FitPredictRowsDestroy, FitPredictRowsInitialize),
		    nullptr, split ? ANOFOX_AGG_BIND(BayesianPredictBind<ARD, true>) : ANOFOX_AGG_BIND(BayesianPredictBind<ARD, false>), FitPredictRowsDestroy);
		predict_set.AddFunction(fn);
	}
	CreateAggregateFunctionInfo predict_info(std::move(predict_set));
	predict_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	const string predict_description =
	    "Fits " + label +
	    " over a partition (training rows: y NOT NULL, or split = 'train') and returns one STRUCT per row: y, yhat, "
	    "yhat_lower / yhat_upper (posterior predictive interval at confidence_level, default 0.95) and is_training.";
	const vector<vector<string>> names = {
	    {"y", "x"}, {"y", "x", "options"}, {"y", "x", "split_col"}, {"y", "x", "split_col", "options"}};
	const vector<string> examples = {predict_name + "(y, x)", predict_name + "(y, x, {'confidence_level': 0.9})",
	                                 predict_name + "(y, x, split)", predict_name + "(y, x, split, {'fit_intercept': false})"};
	for (idx_t s = 0; s < signatures.size(); s++) {
		FunctionDescription d;
		d.description = predict_description;
		d.examples = {examples[s]};
		d.categories = {"regression", "bayesian"};
		d.parameter_names = names[s];
		d.parameter_types = signatures[s];
		predict_info.descriptions.push_back(std::move(d));
	}
	loader.RegisterFunction(std::move(predict_info));
}

} // namespace

void RegisterBayesianAggregateFunctions(ExtensionLoader &loader) {
	RegisterBayesianModel<false>(
	    loader, "a Bayesian Ridge regression",
	    "Fits a Bayesian Ridge regression per group (scikit-learn BayesianRidge: Gaussian prior on the coefficients, "
	    "noise and weight precisions estimated by evidence maximisation) and returns the posterior mean coefficients, "
	    "intercept, R^2, noise_precision (alpha_), weight_precision (lambda_), the posterior standard deviation of each "
	    "coefficient, the iteration count and the log marginal likelihood.");
	RegisterBayesianModel<true>(
	    loader, "an ARD (automatic relevance determination) regression",
	    "Fits an ARD regression per group (scikit-learn ARDRegression: one Gaussian prior precision per coefficient, "
	    "features whose precision reaches threshold_lambda are pruned to 0) and returns the posterior mean "
	    "coefficients, intercept, R^2, noise_precision (alpha_), weight_precisions (lambda_, one per feature), the "
	    "posterior standard deviation of each coefficient and the iteration count.");
}

} // namespace duckdb
