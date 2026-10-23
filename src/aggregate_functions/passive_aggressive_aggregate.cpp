// Passive-Aggressive online regression (#63):
//
//   passive_aggressive_fit_agg(y, x[, options] ORDER BY t)      -> model STRUCT
//   passive_aggressive_fit_predict_agg(y, x[, split][, options]) -> LIST of per-row predictions
//   passive_aggressive_fit_predict(y, x[, options]) OVER (...)  -> prediction for the frame's last row
//
// Upstream's PassiveAggressiveRegressor (scikit-learn PassiveAggressiveRegressor)
// processes the rows in input order without shuffling: the default single pass
// is the online update (scikit-learn's partial_fit), so the aggregates are
// order dependent; give the order with ORDER BY. The model has no prediction
// interval: yhat_lower / yhat_upper are NULL.

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

constexpr const char *kFitName = "passive_aggressive_fit_agg";
constexpr const char *kFitPredictName = "passive_aggressive_fit_predict_agg";
constexpr const char *kWindowName = "passive_aggressive_fit_predict";

struct PaBindData : public FitPredictRowsBindBase {
	AnofoxPassiveAggressiveOptions options = {true, false, 1, 1.0, 0.1, 1e-3};

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<PaBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<PaBindData>();
		return options.fit_intercept == o.options.fit_intercept && options.squared_loss == o.options.squared_loss &&
		       options.max_epochs == o.options.max_epochs && options.c == o.options.c &&
		       options.epsilon == o.options.epsilon && options.tolerance == o.options.tolerance &&
		       use_split_col == o.use_split_col && null_policy == o.null_policy;
	}
};

void ParsePaOptions(ClientContext &context, Expression &expr, PaBindData &bind, const string &name) {
	vector<string> keys = {"fit_intercept", "c", "epsilon", "variant", "max_iterations", "tolerance"};
	if (name == kFitPredictName) {
		keys.push_back("null_policy");
	}
	auto opts = RegressionMapOptions::ParseFromExpression(context, expr, name, keys);
	if (opts.null_policy.has_value()) {
		bind.null_policy = opts.null_policy.value();
	}
	auto &o = bind.options;
	if (opts.fit_intercept.has_value()) {
		o.fit_intercept = opts.fit_intercept.value();
	}
	if (opts.pa_c.has_value()) {
		o.c = opts.pa_c.value();
	}
	if (opts.epsilon.has_value()) {
		o.epsilon = opts.epsilon.value();
	}
	if (opts.pa_squared_loss.has_value()) {
		o.squared_loss = opts.pa_squared_loss.value();
	}
	if (opts.max_iterations.has_value()) {
		o.max_epochs = opts.max_iterations.value();
	}
	if (opts.tolerance.has_value()) {
		o.tolerance = opts.tolerance.value();
	}
	if (!(std::isfinite(o.c) && o.c > 0.0)) {
		throw InvalidInputException("%s: c must be finite and > 0", name);
	}
	if (!(std::isfinite(o.epsilon) && o.epsilon >= 0.0)) {
		throw InvalidInputException("%s: epsilon must be finite and >= 0", name);
	}
	if (o.max_epochs == 0) {
		throw InvalidInputException("%s: max_iterations (passes over the rows) must be at least 1", name);
	}
	if (!(std::isfinite(o.tolerance) && o.tolerance > 0.0)) {
		throw InvalidInputException("%s: tolerance must be > 0", name);
	}
}

//! Fit on the given training rows; false (after throwing for input errors) when no model.
bool FitPa(const char *name, vector<double> &y, vector<vector<double>> &x_cols,
           const AnofoxPassiveAggressiveOptions &options, AnofoxPassiveAggressiveResult &fit) {
	AnofoxDataArray y_arr = {y.data(), nullptr, y.size()};
	vector<AnofoxDataArray> x;
	for (auto &col : x_cols) {
		x.push_back({col.data(), nullptr, col.size()});
	}
	AnofoxError error;
	if (!anofox_passive_aggressive_fit(y_arr, x.data(), x.size(), options, &fit, &error)) {
		ThrowUnlessDegenerate(name, error);
		return false;
	}
	return true;
}

//! intercept + sum(coefficients * x); NaN when x is incomplete.
double PaPredict(const AnofoxPassiveAggressiveResult &fit, const double *x, idx_t n) {
	double yhat = std::isfinite(fit.intercept) ? fit.intercept : 0.0;
	for (idx_t j = 0; j < n; j++) {
		yhat += fit.coefficients[j] * x[j];
	}
	return yhat;
}

//===--------------------------------------------------------------------===//
// passive_aggressive_fit_agg
//===--------------------------------------------------------------------===//
LogicalType PaFitResultType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(make_pair("intercept", LogicalType::DOUBLE));
	c.push_back(make_pair("r_squared", LogicalType::DOUBLE));
	c.push_back(make_pair("n_observations", LogicalType::BIGINT));
	c.push_back(make_pair("n_features", LogicalType::BIGINT));
	c.push_back(make_pair("iterations", LogicalType::BIGINT));
	// Stable shape (#152): no coefficient inference for this model.
	AppendCoefficientInferenceFields(c, "t_values", false, false);
	AppendModelSummaryFields(c);
	return LogicalType::STRUCT(std::move(c));
}

void PaFitFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                   idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<PaBindData>();
	XyRowsStatePointers states(state_vector, count);
	auto &entries = StructVector::GetEntries(result);
	ModelStructWriter writer(result);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		idx_t row = i + offset;
		AnofoxPassiveAggressiveResult fit;
		if (!state.initialized || state.y_values.empty() ||
		    !FitPa(kFitName, state.y_values, state.x_columns, bind.options, fit)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		XyRowsSetDoubleList(StructEntry(entries[0]), row, fit.coefficients, static_cast<idx_t>(fit.n_features));
		XyRowsSetDouble(StructEntry(entries[1]), row, bind.options.fit_intercept ? fit.intercept : 0.0);
		XyRowsSetDouble(StructEntry(entries[2]), row, fit.r_squared);
		FlatDataMut<int64_t>(StructEntry(entries[3]))[row] = static_cast<int64_t>(fit.n_observations);
		FlatDataMut<int64_t>(StructEntry(entries[4]))[row] = static_cast<int64_t>(fit.n_features);
		FlatDataMut<int64_t>(StructEntry(entries[5]))[row] = static_cast<int64_t>(fit.epochs);
		writer.WriteInference(row, nullptr);
		writer.WriteSummary(row, fit.summary);
		anofox_free_passive_aggressive_result(&fit);
	}
}

unique_ptr<FunctionData> PaFitBind(ClientContext &context, AggregateBindFunction &function,
                                   vector<unique_ptr<Expression>> &arguments) {
	auto bind = make_uniq<PaBindData>();
	if (arguments.size() >= 3) {
		ParsePaOptions(context, *arguments[2], *bind, kFitName);
	}
	SetReturnTypeOf(function, PaFitResultType());
	PostHogTelemetry::Instance().RecordFunctionCall(kFitName);
	return std::move(bind);
}

//===--------------------------------------------------------------------===//
// passive_aggressive_fit_predict_agg
//===--------------------------------------------------------------------===//
void PaPredictFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                       idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<PaBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (FitPredictRowsState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t result_idx = i + offset;
		AnofoxPassiveAggressiveResult fit;
		if (!state.initialized || state.y_train.empty() ||
		    !FitPa(kFitPredictName, state.y_train, state.x_train, bind.options, fit)) {
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}
		PadNullXRows(state);
		vector<double> yhat(state.x_all.size());
		for (idx_t r = 0; r < state.x_all.size(); r++) {
			yhat[r] = PaPredict(fit, state.x_all[r].data(), state.n_features);
		}
		anofox_free_passive_aggressive_result(&fit);
		WriteFitPredictRows(result, result_idx, state, yhat.data(), nullptr, nullptr);
	}
}

template <bool SPLIT>
unique_ptr<FunctionData> PaPredictBind(ClientContext &context, AggregateBindFunction &function,
                                       vector<unique_ptr<Expression>> &arguments) {
	auto bind = make_uniq<PaBindData>();
	bind->use_split_col = SPLIT;
	const idx_t options_idx = SPLIT ? 3 : 2;
	if (arguments.size() > options_idx) {
		ParsePaOptions(context, *arguments[options_idx], *bind, kFitPredictName);
	}
	SetReturnTypeOf(function, FitPredictRowsResultType());
	PostHogTelemetry::Instance().RecordFunctionCall(kFitPredictName);
	return std::move(bind);
}

//===--------------------------------------------------------------------===//
// passive_aggressive_fit_predict (window)
//===--------------------------------------------------------------------===//
// The frame's training rows in order plus the last row's x, as rls_fit_predict.
struct PaWindowState {
	vector<double> y_values;
	vector<vector<double>> x_columns;
	vector<double> current_x;
	bool has_current_x = false;
	// Saw any row, NULL x included: such a state ends a combined frame (#208)
	bool saw_row = false;
	idx_t n_features = 0;
	bool initialized = false;
};

void PaWindowInitialize(data_ptr_t state_p) {
	new (state_p) PaWindowState();
}

void PaWindowDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (PaWindowState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~PaWindowState();
	}
}

void PaWindowUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat y_data, x_data, sdata;
	inputs[0].ToUnifiedFormat(count, y_data);
	inputs[1].ToUnifiedFormat(count, x_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
	auto x_lists = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &x_child = ListVector::GetEntry(inputs[1]);
	auto x_child_data = FlatDataMut<double>(x_child);
	auto &x_child_validity = FlatValidityMut(x_child);
	auto states = (PaWindowState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto x_idx = x_data.sel->get_index(i);
		if (!x_data.validity.RowIsValid(x_idx)) {
			state.has_current_x = false;
			state.saw_row = true;
			continue;
		}
		auto entry = x_lists[x_idx];
		if (!state.initialized) {
			state.n_features = entry.length;
			state.x_columns.resize(state.n_features);
			state.initialized = true;
		}
		if (entry.length != state.n_features) {
			throw InvalidInputException("Inconsistent feature count");
		}
		state.current_x.resize(state.n_features);
		bool complete = true;
		for (idx_t j = 0; j < state.n_features; j++) {
			state.current_x[j] = ListChildValue(x_child_data, x_child_validity, entry.offset + j);
			complete = complete && std::isfinite(state.current_x[j]);
		}
		state.has_current_x = true;
		state.saw_row = true;
		auto y_idx = y_data.sel->get_index(i);
		if (y_data.validity.RowIsValid(y_idx) && std::isfinite(y_values[y_idx]) && complete) {
			state.y_values.push_back(y_values[y_idx]);
			for (idx_t j = 0; j < state.n_features; j++) {
				state.x_columns[j].push_back(state.current_x[j]);
			}
		}
	}
}

void PaWindowCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (PaWindowState **)source_data.data;
	auto targets = (PaWindowState **)target_data.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		// The source holds the later rows, so its last row is the frame's last row
		target.saw_row = target.saw_row || source.saw_row;
		if (!source.initialized) {
			if (source.saw_row) {
				target.has_current_x = false;
			}
			continue;
		}
		if (!target.initialized) {
			target.y_values = CombineTake(source.y_values, aggr_input_data);
			target.x_columns = CombineTake(source.x_columns, aggr_input_data);
			target.current_x = CombineTake(source.current_x, aggr_input_data);
			target.has_current_x = source.has_current_x;
			target.n_features = source.n_features;
			target.initialized = true;
			continue;
		}
		if (source.n_features != target.n_features) {
			throw InvalidInputException("Cannot combine states with different feature counts");
		}
		AppendTo(target.y_values, source.y_values);
		for (idx_t j = 0; j < target.n_features; j++) {
			AppendTo(target.x_columns[j], source.x_columns[j]);
		}
		if (source.saw_row) {
			target.current_x = CombineTake(source.current_x, aggr_input_data);
			target.has_current_x = source.has_current_x;
		}
	}
}

LogicalType PaWindowResultType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("yhat", LogicalType::DOUBLE));
	c.push_back(make_pair("yhat_lower", LogicalType::DOUBLE));
	c.push_back(make_pair("yhat_upper", LogicalType::DOUBLE));
	return LogicalType::STRUCT(std::move(c));
}

void PaWindowFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                      idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<PaBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (PaWindowState **)sdata.data;
	auto &entries = StructVector::GetEntries(result);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		AnofoxPassiveAggressiveResult fit;
		// A NULL or incomplete current x has no prediction (#208)
		if (!state.initialized || !state.has_current_x || !AllFinite(state.current_x) || state.y_values.empty() ||
		    !FitPa(kWindowName, state.y_values, state.x_columns, bind.options, fit)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		double yhat = PaPredict(fit, state.current_x.data(), state.n_features);
		anofox_free_passive_aggressive_result(&fit);
		XyRowsSetDouble(StructEntry(entries[0]), row, yhat);
		FlatVector::SetNull(StructEntry(entries[1]), row, true);
		FlatVector::SetNull(StructEntry(entries[2]), row, true);
	}
}

unique_ptr<FunctionData> PaWindowBind(ClientContext &context, AggregateBindFunction &function,
                                      vector<unique_ptr<Expression>> &arguments) {
	auto bind = make_uniq<PaBindData>();
	if (arguments.size() >= 3) {
		ParsePaOptions(context, *arguments[2], *bind, kWindowName);
	}
	SetReturnTypeOf(function, PaWindowResultType());
	PostHogTelemetry::Instance().RecordFunctionCall(kWindowName);
	return std::move(bind);
}

void AddDescription(CreateAggregateFunctionInfo &info, const string &description, const string &example,
                    const vector<string> &names, const vector<LogicalType> &types) {
	FunctionDescription d;
	d.description = description;
	d.examples = {example};
	d.categories = {"regression", "online"};
	d.parameter_names = names;
	d.parameter_types = types;
	info.descriptions.push_back(std::move(d));
}

} // namespace

void RegisterPassiveAggressiveFunctions(ExtensionLoader &loader) {
	const auto x_type = LogicalType::LIST(LogicalType::DOUBLE);
	const string options_note =
	    " Options: c (aggressiveness, default 1), epsilon (insensitive zone, 0.1), variant ('pa1' default | 'pa2'), "
	    "fit_intercept, max_iterations (passes over the rows, default 1 = online) and tolerance (early stopping of "
	    "multi-pass fits).";

	// passive_aggressive_fit_agg
	AggregateFunctionSet fit_set(kFitName);
	for (bool with_options : {false, true}) {
		vector<LogicalType> args = {LogicalType::DOUBLE, x_type};
		if (with_options) {
			args.push_back(LogicalType::ANY);
		}
		AggregateFunction fn(kFitName, args, LogicalType::ANY, AggregateFunction::StateSize<XyRowsState>,
		                     ANOFOX_AGG_INIT(XyRowsInitialize), XyRowsUpdate, XyRowsCombine,
		                     ANOFOX_GUARDED_FINALIZE(PaFitFinalize, XyRowsDestroy, XyRowsInitialize), nullptr,
		                     ANOFOX_AGG_BIND(PaFitBind), XyRowsDestroy);
		// Sequential estimator: the row order is part of the input.
		SetOrderDependentOf(fn, AggregateOrderDependent::ORDER_DEPENDENT);
		fit_set.AddFunction(fn);
	}
	CreateAggregateFunctionInfo fit_info(std::move(fit_set));
	fit_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	const string fit_description =
	    "Fits a Passive-Aggressive online regression (scikit-learn PassiveAggressiveRegressor) by updating the "
	    "coefficients row by row in input order, without shuffling; give the order with ORDER BY. Returns the "
	    "coefficients, intercept, R^2 and the number of passes." +
	    options_note;
	AddDescription(fit_info, fit_description, "passive_aggressive_fit_agg(y, [x1, x2] ORDER BY t)", {"y", "x"},
	               {LogicalType::DOUBLE, x_type});
	AddDescription(fit_info, fit_description,
	               "passive_aggressive_fit_agg(y, [x1, x2], {'c': 0.5, 'epsilon': 0.05, 'variant': 'pa2'} ORDER BY t)",
	               {"y", "x", "options"}, {LogicalType::DOUBLE, x_type, LogicalType::ANY});
	loader.RegisterFunction(std::move(fit_info));

	// passive_aggressive_fit_predict_agg
	AggregateFunctionSet predict_set(kFitPredictName);
	const vector<vector<LogicalType>> signatures = {{LogicalType::DOUBLE, x_type},
	                                                {LogicalType::DOUBLE, x_type, LogicalType::ANY},
	                                                {LogicalType::DOUBLE, x_type, LogicalType::VARCHAR},
	                                                {LogicalType::DOUBLE, x_type, LogicalType::VARCHAR, LogicalType::ANY}};
	for (auto &args : signatures) {
		bool split = args.size() >= 3 && args[2].id() == LogicalTypeId::VARCHAR;
		AggregateFunction fn(
		    kFitPredictName, args, LogicalType::ANY, AggregateFunction::StateSize<FitPredictRowsState>,
		    ANOFOX_AGG_INIT(FitPredictRowsInitialize),
		    ANOFOX_GUARDED_UPDATE(FitPredictRowsUpdate, FitPredictRowsDestroy, FitPredictRowsInitialize),
		    FitPredictRowsCombine,
		    ANOFOX_GUARDED_FINALIZE(PaPredictFinalize, FitPredictRowsDestroy, FitPredictRowsInitialize), nullptr,
		    split ? ANOFOX_AGG_BIND(PaPredictBind<true>) : ANOFOX_AGG_BIND(PaPredictBind<false>), FitPredictRowsDestroy);
		SetOrderDependentOf(fn, AggregateOrderDependent::ORDER_DEPENDENT);
		predict_set.AddFunction(fn);
	}
	CreateAggregateFunctionInfo predict_info(std::move(predict_set));
	predict_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	const string predict_description =
	    "Fits a Passive-Aggressive online regression on the training rows of a partition (y NOT NULL and, with "
	    "split_col, split 'train' / 'training' in any case) in input order and returns one STRUCT per row: y, yhat "
	    "(from the final coefficients), yhat_lower / yhat_upper (NULL: the model has no interval) and is_training." +
	    options_note;
	const vector<vector<string>> names = {
	    {"y", "x"}, {"y", "x", "options"}, {"y", "x", "split_col"}, {"y", "x", "split_col", "options"}};
	const vector<string> examples = {"passive_aggressive_fit_predict_agg(y, x ORDER BY t)",
	                                 "passive_aggressive_fit_predict_agg(y, x, {'c': 0.5} ORDER BY t)",
	                                 "passive_aggressive_fit_predict_agg(y, x, split ORDER BY t)",
	                                 "passive_aggressive_fit_predict_agg(y, x, split, {'variant': 'pa2'} ORDER BY t)"};
	for (idx_t s = 0; s < signatures.size(); s++) {
		AddDescription(predict_info, predict_description, examples[s], names[s], signatures[s]);
	}
	loader.RegisterFunction(std::move(predict_info));

	// passive_aggressive_fit_predict (window)
	AggregateFunctionSet window_set(kWindowName);
	for (bool with_options : {false, true}) {
		vector<LogicalType> args = {LogicalType::DOUBLE, x_type};
		if (with_options) {
			args.push_back(LogicalType::ANY);
		}
		AggregateFunction fn(kWindowName, args, PaWindowResultType(), AggregateFunction::StateSize<PaWindowState>,
		                     ANOFOX_AGG_INIT(PaWindowInitialize),
		                     ANOFOX_GUARDED_UPDATE(PaWindowUpdate, PaWindowDestroy, PaWindowInitialize),
		                     PaWindowCombine,
		                     ANOFOX_GUARDED_FINALIZE(PaWindowFinalize, PaWindowDestroy, PaWindowInitialize), nullptr,
		                     ANOFOX_AGG_BIND(PaWindowBind), PaWindowDestroy);
		window_set.AddFunction(fn);
	}
	CreateAggregateFunctionInfo window_info(std::move(window_set));
	window_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	const string window_description =
	    "Window aggregate: runs the Passive-Aggressive online update over the rows of the window frame in frame order "
	    "and returns the prediction for the LAST row of the frame (yhat; yhat_lower / yhat_upper are NULL). Use "
	    "frames ending at CURRENT ROW over a unique ordering." +
	    options_note;
	AddDescription(window_info, window_description,
	               "passive_aggressive_fit_predict(y, [x1, x2]) OVER (ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND "
	               "CURRENT ROW)",
	               {"y", "x"}, {LogicalType::DOUBLE, x_type});
	AddDescription(window_info, window_description,
	               "passive_aggressive_fit_predict(y, [x1, x2], {'c': 0.5}) OVER (ORDER BY t ROWS BETWEEN 49 "
	               "PRECEDING AND CURRENT ROW)",
	               {"y", "x", "options"}, {LogicalType::DOUBLE, x_type, LogicalType::ANY});
	loader.RegisterFunction(std::move(window_info));
}

} // namespace duckdb
