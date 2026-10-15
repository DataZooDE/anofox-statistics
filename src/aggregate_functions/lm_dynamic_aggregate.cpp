// lmDynamic, time-varying coefficients (#63):
//
//   lm_dynamic_fit_agg(y, x[, options] ORDER BY t)             -> model STRUCT with per-row coefficients
//   lm_dynamic_fit_predict_agg(y, x[, split][, options] ORDER BY t) -> LIST of per-row fits
//
// Upstream's LmDynamicRegressor reproduces greybox::lmDynamic(bruteforce = TRUE)
// for the normal distribution: every subset of the regressors is fitted, each
// observation weights the subsets by their pointwise information criterion,
// optionally smoothed over time with lowess, and the coefficients of row t are
// the weighted average. The row order is the time index, so give it with
// ORDER BY.

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
#include "telemetry.hpp"
#include "xy_rows_aggregate.hpp"

namespace duckdb {

namespace {

constexpr const char *kFitName = "lm_dynamic_fit_agg";
constexpr const char *kFitPredictName = "lm_dynamic_fit_predict_agg";

struct LmDynamicBindData : public FitPredictRowsBindBase {
	bool fit_intercept = true;
	int ic = ANOFOX_IC_AICC;
	double span = 0.3;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<LmDynamicBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<LmDynamicBindData>();
		return fit_intercept == o.fit_intercept && ic == o.ic && span == o.span && use_split_col == o.use_split_col;
	}

	AnofoxLmDynamicOptions Options() const {
		AnofoxLmDynamicOptions o;
		o.fit_intercept = fit_intercept;
		o.ic = static_cast<AnofoxInformationCriterion>(ic);
		o.distribution = ANOFOX_ALM_DIST_NORMAL;
		o.lowess_span = span;
		o.max_models = 0; // every regressor subset
		o.confidence_level = 0.95;
		return o;
	}
};

void ParseLmDynamicOptions(ClientContext &context, Expression &expr, LmDynamicBindData &bind, const string &name) {
	auto opts = RegressionMapOptions::ParseFromExpression(context, expr, name, {"fit_intercept", "ic", "span"});
	if (opts.fit_intercept.has_value()) {
		bind.fit_intercept = opts.fit_intercept.value();
	}
	if (opts.lm_dynamic_ic.has_value()) {
		bind.ic = opts.lm_dynamic_ic.value();
	}
	if (opts.lowess_span.has_value()) {
		bind.span = opts.lowess_span.value();
	}
	if (!(bind.span >= 0.0 && bind.span <= 1.0)) {
		throw InvalidInputException("%s: span must be in [0, 1] (0 = no smoothing of the weights)", name);
	}
}

//! Fit on the given rows; false (after throwing for input errors) when no model.
bool FitLmDynamic(const char *name, vector<double> &y, vector<vector<double>> &x_cols, const LmDynamicBindData &bind,
                  AnofoxLmDynamicFit &fit) {
	AnofoxDataArray y_arr = {y.data(), nullptr, y.size()};
	vector<AnofoxDataArray> x;
	for (auto &col : x_cols) {
		x.push_back({col.data(), nullptr, col.size()});
	}
	AnofoxError error;
	if (!anofox_lm_dynamic_fit(y_arr, x.data(), x.size(), bind.Options(), &fit, &error)) {
		ThrowUnlessDegenerate(name, error);
		return false;
	}
	return true;
}

//===--------------------------------------------------------------------===//
// lm_dynamic_fit_agg
//===--------------------------------------------------------------------===//
LogicalType LmDynamicFitResultType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(make_pair("intercept", LogicalType::DOUBLE));
	c.push_back(make_pair("r_squared", LogicalType::DOUBLE));
	c.push_back(make_pair("adj_r_squared", LogicalType::DOUBLE));
	c.push_back(make_pair("residual_std_error", LogicalType::DOUBLE));
	c.push_back(make_pair("n_observations", LogicalType::BIGINT));
	c.push_back(make_pair("n_features", LogicalType::BIGINT));
	c.push_back(make_pair("n_models", LogicalType::BIGINT));
	c.push_back(make_pair("dynamic_intercept", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(make_pair("dynamic_coefficients", LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE))));
	c.push_back(make_pair("fitted_values", LogicalType::LIST(LogicalType::DOUBLE)));
	// Stable shape (#152): no coefficient inference for this model.
	AppendCoefficientInferenceFields(c, "t_values", false, false);
	AppendModelSummaryFields(c);
	return LogicalType::STRUCT(std::move(c));
}

//! Write `rows` lists of `width` values each (row-major `values`) as LIST(LIST(DOUBLE)).
void SetNestedList(Vector &vec, idx_t row, const double *values, idx_t rows, idx_t width, idx_t skip) {
	auto outer = FlatVector::GetData<list_entry_t>(vec);
	auto outer_offset = ListVector::GetListSize(vec);
	ListVector::Reserve(vec, outer_offset + rows);
	auto &inner = ListVector::GetEntry(vec);
	for (idx_t r = 0; r < rows; r++) {
		XyRowsSetDoubleList(inner, outer_offset + r, values + r * width + skip, width - skip);
	}
	outer[row].offset = outer_offset;
	outer[row].length = rows;
	ListVector::SetListSize(vec, outer_offset + rows);
}

void LmDynamicFitFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                          idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<LmDynamicBindData>();
	XyRowsStatePointers states(state_vector, count);
	auto &entries = StructVector::GetEntries(result);
	ModelStructWriter writer(result);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		idx_t row = i + offset;
		AnofoxLmDynamicFit fit;
		if (!state.initialized || state.y_values.empty() ||
		    !FitLmDynamic(kFitName, state.y_values, state.x_columns, bind, fit)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		const idx_t n_rows = static_cast<idx_t>(fit.n_rows);
		const idx_t width = static_cast<idx_t>(fit.n_coefs_per_row);
		XyRowsSetDoubleList(*entries[0], row, fit.coefficients, static_cast<idx_t>(fit.n_features));
		XyRowsSetDouble(*entries[1], row, bind.fit_intercept ? fit.intercept : 0.0);
		XyRowsSetDouble(*entries[2], row, fit.r_squared);
		XyRowsSetDouble(*entries[3], row, fit.adj_r_squared);
		XyRowsSetDouble(*entries[4], row, fit.residual_std_error);
		FlatVector::GetData<int64_t>(*entries[5])[row] = static_cast<int64_t>(fit.n_observations);
		FlatVector::GetData<int64_t>(*entries[6])[row] = static_cast<int64_t>(fit.n_features);
		FlatVector::GetData<int64_t>(*entries[7])[row] = static_cast<int64_t>(fit.n_models);
		vector<double> intercepts(n_rows);
		for (idx_t r = 0; r < n_rows; r++) {
			intercepts[r] = bind.fit_intercept ? fit.dynamic_coefficients[r * width] : 0.0;
		}
		XyRowsSetDoubleList(*entries[8], row, intercepts.data(), n_rows);
		SetNestedList(*entries[9], row, fit.dynamic_coefficients, n_rows, width, 1);
		XyRowsSetDoubleList(*entries[10], row, fit.fitted_values, n_rows);
		writer.WriteInference(row, nullptr);
		writer.WriteSummary(row, fit.summary);
		anofox_free_lm_dynamic_fit_result(&fit);
	}
}

unique_ptr<FunctionData> LmDynamicFitBind(ClientContext &context, AggregateFunction &function,
                                          vector<unique_ptr<Expression>> &arguments) {
	auto bind = make_uniq<LmDynamicBindData>();
	if (arguments.size() >= 3) {
		ParseLmDynamicOptions(context, *arguments[2], *bind, kFitName);
	}
	function.return_type = LmDynamicFitResultType();
	PostHogTelemetry::Instance().RecordFunctionCall(kFitName);
	return std::move(bind);
}

//===--------------------------------------------------------------------===//
// lm_dynamic_fit_predict_agg
//===--------------------------------------------------------------------===//
// The rows of FitPredictRowsResultType plus the row's dynamic intercept and
// coefficients (NULL for rows not used in the fit).
LogicalType LmDynamicPredictResultType() {
	child_list_t<LogicalType> row;
	row.push_back(make_pair("y", LogicalType::DOUBLE));
	row.push_back(make_pair("yhat", LogicalType::DOUBLE));
	row.push_back(make_pair("yhat_lower", LogicalType::DOUBLE));
	row.push_back(make_pair("yhat_upper", LogicalType::DOUBLE));
	row.push_back(make_pair("is_training", LogicalType::BOOLEAN));
	row.push_back(make_pair("intercept", LogicalType::DOUBLE));
	row.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
	return LogicalType::LIST(LogicalType::STRUCT(std::move(row)));
}

void LmDynamicPredictFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                              idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<LmDynamicBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (FitPredictRowsState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t result_idx = i + offset;
		AnofoxLmDynamicFit fit;
		if (!state.initialized || state.y_train.empty() ||
		    !FitLmDynamic(kFitPredictName, state.y_train, state.x_train, bind, fit)) {
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}
		PadNullXRows(state);
		const idx_t n_rows = state.y_all.size();
		const idx_t width = static_cast<idx_t>(fit.n_coefs_per_row);
		const idx_t p = state.n_features;

		auto list_data = ListVector::GetData(result);
		auto list_offset = ListVector::GetListSize(result);
		ListVector::Reserve(result, list_offset + n_rows);
		list_data[result_idx].offset = list_offset;
		list_data[result_idx].length = n_rows;
		auto &entries = StructVector::GetEntries(ListVector::GetEntry(result));
		idx_t train_row = 0;
		for (idx_t r = 0; r < n_rows; r++) {
			idx_t child = list_offset + r;
			if (state.y_is_null[r]) {
				FlatVector::SetNull(*entries[0], child, true);
			} else {
				FlatVector::GetData<double>(*entries[0])[child] = state.y_all[r];
			}
			double yhat;
			if (state.is_training[r]) {
				// The row's own time-varying fit.
				const double *coefs = fit.dynamic_coefficients + train_row * width;
				yhat = fit.fitted_values[train_row];
				XyRowsSetDouble(*entries[5], child, bind.fit_intercept ? coefs[0] : 0.0);
				XyRowsSetDoubleList(*entries[6], child, coefs + 1, width - 1);
				train_row++;
			} else {
				// Rows outside the fit: the time-averaged coefficients (greybox's mean model).
				yhat = std::isfinite(fit.intercept) ? fit.intercept : 0.0;
				for (idx_t j = 0; j < p; j++) {
					if (std::isfinite(fit.coefficients[j])) {
						yhat += fit.coefficients[j] * state.x_all[r][j];
					}
				}
				FlatVector::SetNull(*entries[5], child, true);
				FlatVector::SetNull(*entries[6], child, true);
			}
			XyRowsSetDouble(*entries[1], child, yhat);
			FlatVector::SetNull(*entries[2], child, true);
			FlatVector::SetNull(*entries[3], child, true);
			FlatVector::GetData<bool>(*entries[4])[child] = state.is_training[r];
		}
		ListVector::SetListSize(result, list_offset + n_rows);
		anofox_free_lm_dynamic_fit_result(&fit);
	}
}

template <bool SPLIT>
unique_ptr<FunctionData> LmDynamicPredictBind(ClientContext &context, AggregateFunction &function,
                                              vector<unique_ptr<Expression>> &arguments) {
	auto bind = make_uniq<LmDynamicBindData>();
	bind->use_split_col = SPLIT;
	const idx_t options_idx = SPLIT ? 3 : 2;
	if (arguments.size() > options_idx) {
		ParseLmDynamicOptions(context, *arguments[options_idx], *bind, kFitPredictName);
	}
	function.return_type = LmDynamicPredictResultType();
	PostHogTelemetry::Instance().RecordFunctionCall(kFitPredictName);
	return std::move(bind);
}

void AddDescription(CreateAggregateFunctionInfo &info, const string &description, const string &example,
                    const vector<string> &names, const vector<LogicalType> &types) {
	FunctionDescription d;
	d.description = description;
	d.examples = {example};
	d.categories = {"regression", "time-series"};
	d.parameter_names = names;
	d.parameter_types = types;
	info.descriptions.push_back(std::move(d));
}

} // namespace

void RegisterLmDynamicFunctions(ExtensionLoader &loader) {
	const auto x_type = LogicalType::LIST(LogicalType::DOUBLE);
	const string options_note =
	    " Options: ic ('aicc' default | 'aic' | 'bic'), span (lowess f smoothing the model weights over time, default "
	    "0.3; 0 = no smoothing), fit_intercept.";

	AggregateFunctionSet fit_set(kFitName);
	for (bool with_options : {false, true}) {
		vector<LogicalType> args = {LogicalType::DOUBLE, x_type};
		if (with_options) {
			args.push_back(LogicalType::ANY);
		}
		AggregateFunction fn(kFitName, args, LogicalType::ANY, AggregateFunction::StateSize<XyRowsState>,
		                     XyRowsInitialize, XyRowsUpdate, XyRowsCombine,
		                     ANOFOX_GUARDED_FINALIZE(LmDynamicFitFinalize, XyRowsDestroy, XyRowsInitialize), nullptr,
		                     LmDynamicFitBind, XyRowsDestroy);
		// The row order is the time index.
		fn.order_dependent = AggregateOrderDependent::ORDER_DEPENDENT;
		fit_set.AddFunction(fn);
	}
	CreateAggregateFunctionInfo fit_info(std::move(fit_set));
	fit_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	const string fit_description =
	    "Fits a dynamic linear model with time-varying coefficients (greybox lmDynamic, bruteforce): every subset of "
	    "the regressors is fitted, each row weights them by its pointwise information criterion and gets the "
	    "weighted coefficients. Rows are in input order (give it with ORDER BY). Returns the time-averaged "
	    "coefficients and intercept, fit statistics, n_models and, per row used, dynamic_intercept, "
	    "dynamic_coefficients and fitted_values." +
	    options_note;
	AddDescription(fit_info, fit_description, "lm_dynamic_fit_agg(y, [x1, x2] ORDER BY t)", {"y", "x"},
	               {LogicalType::DOUBLE, x_type});
	AddDescription(fit_info, fit_description, "lm_dynamic_fit_agg(y, [x1, x2], {'ic': 'bic', 'span': 0.5} ORDER BY t)",
	               {"y", "x", "options"}, {LogicalType::DOUBLE, x_type, LogicalType::ANY});
	loader.RegisterFunction(std::move(fit_info));

	AggregateFunctionSet predict_set(kFitPredictName);
	const vector<vector<LogicalType>> signatures = {{LogicalType::DOUBLE, x_type},
	                                                {LogicalType::DOUBLE, x_type, LogicalType::ANY},
	                                                {LogicalType::DOUBLE, x_type, LogicalType::VARCHAR},
	                                                {LogicalType::DOUBLE, x_type, LogicalType::VARCHAR, LogicalType::ANY}};
	for (auto &args : signatures) {
		bool split = args.size() >= 3 && args[2].id() == LogicalTypeId::VARCHAR;
		AggregateFunction fn(
		    kFitPredictName, args, LogicalType::ANY, AggregateFunction::StateSize<FitPredictRowsState>,
		    FitPredictRowsInitialize,
		    ANOFOX_GUARDED_UPDATE(FitPredictRowsUpdate, FitPredictRowsDestroy, FitPredictRowsInitialize),
		    FitPredictRowsCombine,
		    ANOFOX_GUARDED_FINALIZE(LmDynamicPredictFinalize, FitPredictRowsDestroy, FitPredictRowsInitialize),
		    nullptr, split ? LmDynamicPredictBind<true> : LmDynamicPredictBind<false>, FitPredictRowsDestroy);
		fn.order_dependent = AggregateOrderDependent::ORDER_DEPENDENT;
		predict_set.AddFunction(fn);
	}
	CreateAggregateFunctionInfo predict_info(std::move(predict_set));
	predict_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	const string predict_description =
	    "Fits a dynamic linear model (greybox lmDynamic) on the training rows of a partition (y NOT NULL, or split = "
	    "'train') in input order and returns one STRUCT per row: y, yhat (the row's own time-varying fit; rows "
	    "outside the fit use the time-averaged coefficients), yhat_lower / yhat_upper (NULL), is_training, and the "
	    "row's dynamic intercept and coefficients (NULL outside the fit)." +
	    options_note;
	const vector<vector<string>> names = {
	    {"y", "x"}, {"y", "x", "options"}, {"y", "x", "split_col"}, {"y", "x", "split_col", "options"}};
	const vector<string> examples = {"lm_dynamic_fit_predict_agg(y, x ORDER BY t)",
	                                 "lm_dynamic_fit_predict_agg(y, x, {'span': 0} ORDER BY t)",
	                                 "lm_dynamic_fit_predict_agg(y, x, split ORDER BY t)",
	                                 "lm_dynamic_fit_predict_agg(y, x, split, {'ic': 'aic'} ORDER BY t)"};
	for (idx_t s = 0; s < signatures.size(); s++) {
		AddDescription(predict_info, predict_description, examples[s], names[s], signatures[s]);
	}
	loader.RegisterFunction(std::move(predict_info));
}

} // namespace duckdb
