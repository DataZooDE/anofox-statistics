// P-spline smoother (#316):
//
//   pspline_fit_agg(y, x[, options])                 -> STRUCT (the smooth at the distinct x, lambda, edf, ...)
//   pspline_fit_predict_agg(y, x[, split][, options]) -> LIST(STRUCT(y, yhat, yhat_lower, yhat_upper, is_training))
//
// Upstream's PSplineRegressor, mgcv::gam(y ~ s(x, bs = "ps", k = n_basis, m = c(2, penalty_order)),
// method = "GCV.Cp"): cubic B-splines on evenly spaced knots, a difference penalty, the
// smoothing parameter fixed (lambda) or chosen by GCV, Bayesian standard errors of the mean
// and a linear extension outside the data range.
//
// Upstream keeps the fitted basis private, so a stored model cannot be evaluated at new x:
// predictions come from fitting and predicting in one aggregate (as isotonic_fit_predict_agg).
//
// Options: n_basis (alias k; 0 = automatic, the default), penalty_order (default 2),
// lambda (default: GCV); fit_predict also confidence_level (default 0.95) and interval
// ('prediction' (default) | 'confidence' | 'none').

#include <algorithm>
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
#include "telemetry.hpp"
#include "xy_rows_aggregate.hpp"
#include "../include/canonical_order.hpp"

namespace duckdb {

namespace {

constexpr const char *kFitName = "pspline_fit_agg";
constexpr const char *kFitPredictName = "pspline_fit_predict_agg";

enum PSplineInterval : int32_t { PSPLINE_NONE = -1, PSPLINE_PREDICTION = 0, PSPLINE_CONFIDENCE = 1 };

struct PSplineBindData : public FitPredictRowsBindBase {
	AnofoxPSplineOptions options = {0, 2, std::nan("")};
	double confidence_level = 0.95;
	int32_t interval = PSPLINE_PREDICTION;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<PSplineBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<PSplineBindData>();
		auto same_lambda = (std::isnan(options.lambda) && std::isnan(o.options.lambda)) ||
		                   options.lambda == o.options.lambda;
		return options.n_basis == o.options.n_basis && options.penalty_order == o.options.penalty_order &&
		       same_lambda && confidence_level == o.confidence_level && interval == o.interval &&
		       use_split_col == o.use_split_col;
	}
};

//! Walk a MAP / STRUCT options value as (lower-case key, value) pairs.
template <class FN>
void VisitPSplineOptions(const Value &options, FN &&fn) {
	const Value v = UnwrapVariant(options);
	if (v.IsNull()) {
		return;
	}
	if (v.type().id() == LogicalTypeId::MAP) {
		for (auto &entry : MapValue::GetChildren(v)) {
			auto &kv = StructValue::GetChildren(entry);
			if (kv.size() != 2 || kv[0].IsNull()) {
				throw InvalidInputException("Invalid MAP option entry");
			}
			fn(StringUtil::Lower(kv[0].ToString()), UnwrapVariant(kv[1]));
		}
	} else if (v.type().id() == LogicalTypeId::STRUCT) {
		auto &children = StructValue::GetChildren(v);
		auto &types = StructType::GetChildTypes(v.type());
		for (idx_t i = 0; i < types.size(); i++) {
			fn(StringUtil::Lower(NameString(types[i].first)), UnwrapVariant(children[i]));
		}
	} else {
		throw InvalidInputException("Expected MAP or STRUCT type for options, got %s", v.type().ToString());
	}
}

void ParsePSplineOptions(ClientContext &context, Expression &expr, PSplineBindData &bind, const string &fn,
                         bool fit_predict) {
	auto as_double = [&](const string &key, const Value &v) {
		try {
			return v.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
		} catch (std::exception &) {
			throw InvalidInputException("%s: option '%s' must be a number, got '%s'", fn, key, v.ToString());
		}
	};
	auto as_count = [&](const string &key, const Value &v, double lo, double hi) {
		double d = as_double(key, v);
		if (!(d >= lo && d <= hi) || d != std::floor(d)) {
			throw InvalidInputException("%s: %s must be an integer in [%g, %g], got '%s'", fn, key, lo, hi,
			                            v.ToString());
		}
		return static_cast<uint32_t>(d);
	};
	VisitPSplineOptions(EvaluateConstantOptions(context, expr, fn), [&](const string &key, const Value &v) {
		if (v.IsNull()) {
			return;
		}
		if (key == "n_basis" || key == "k") {
			bind.options.n_basis = as_count("n_basis", v, 0, 1000);
		} else if (key == "penalty_order") {
			bind.options.penalty_order = as_count(key, v, 1, 10);
		} else if (key == "lambda") {
			double d = as_double(key, v);
			if (!(std::isfinite(d) && d >= 0.0)) {
				throw InvalidInputException("%s: lambda must be finite and >= 0, got %g", fn, d);
			}
			bind.options.lambda = d;
		} else if (fit_predict && key == "confidence_level") {
			double d = as_double(key, v);
			if (!(d > 0.0 && d < 1.0)) {
				throw InvalidInputException("%s: confidence_level must be in (0, 1), got %g", fn, d);
			}
			bind.confidence_level = d;
		} else if (fit_predict && key == "interval") {
			auto s = StringUtil::Lower(v.ToString());
			if (s == "prediction") {
				bind.interval = PSPLINE_PREDICTION;
			} else if (s == "confidence") {
				bind.interval = PSPLINE_CONFIDENCE;
			} else if (s == "none") {
				bind.interval = PSPLINE_NONE;
			} else {
				throw InvalidInputException(
				    "%s: unknown interval '%s'. Expected 'prediction', 'confidence' or 'none'.", fn, v.ToString());
			}
		} else {
			throw InvalidInputException("%s: unknown option '%s'. Supported options: n_basis (alias k), "
			                            "penalty_order, lambda%s",
			                            fn, key, fit_predict ? ", confidence_level, interval" : "");
		}
	});
}

//===--------------------------------------------------------------------===//
// pspline_fit_agg
//===--------------------------------------------------------------------===//
struct PSplineFitState {
	vector<double> x;
	vector<double> y;
};

void PSplineFitInitialize(data_ptr_t state_p) {
	new (state_p) PSplineFitState();
}

void PSplineFitDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (PSplineFitState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~PSplineFitState();
	}
}

//! Keeps the complete rows: y and x non-NULL and finite.
void PSplineFitUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat y_data, x_data, sdata;
	inputs[0].ToUnifiedFormat(count, y_data);
	inputs[1].ToUnifiedFormat(count, x_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
	auto x_values = UnifiedVectorFormat::GetData<double>(x_data);
	auto states = (PSplineFitState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
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
		auto &state = *states[sdata.sel->get_index(i)];
		state.x.push_back(x);
		state.y.push_back(y);
	}
}

void PSplineFitCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                       idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (PSplineFitState **)source_data.data;
	auto targets = (PSplineFitState **)target_data.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (target.x.empty()) {
			target.x = CombineTake(source.x, aggr_input_data);
			target.y = CombineTake(source.y, aggr_input_data);
			continue;
		}
		target.x.insert(target.x.end(), source.x.begin(), source.x.end());
		target.y.insert(target.y.end(), source.y.begin(), source.y.end());
	}
}

LogicalType PSplineFitResultType() {
	auto list = LogicalType::LIST(LogicalType::DOUBLE);
	child_list_t<LogicalType> c;
	c.push_back(make_pair("x", list));
	c.push_back(make_pair("fitted", list));
	c.push_back(make_pair("se_fit", list));
	c.push_back(make_pair("coefficients", list));
	c.push_back(make_pair("lambda", LogicalType::DOUBLE));
	c.push_back(make_pair("edf", LogicalType::DOUBLE));
	c.push_back(make_pair("sigma2", LogicalType::DOUBLE));
	c.push_back(make_pair("r_squared", LogicalType::DOUBLE));
	c.push_back(make_pair("n_basis", LogicalType::BIGINT));
	c.push_back(make_pair("penalty_order", LogicalType::BIGINT));
	c.push_back(make_pair("n_observations", LogicalType::BIGINT));
	// Stable shape (#152): no coefficient inference, those fields are NULL.
	AppendModelSummaryFields(c);
	return LogicalType::STRUCT(std::move(c));
}

void PSplineFitFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                        idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<PSplineBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (PSplineFitState **)sdata.data;
	auto &entries = StructVector::GetEntries(result);
	ModelStructWriter writer(result);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		if (state.x.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		// Canonical row order: bitwise-identical results whatever the thread count (#226).
		CanonicalRows().Key(state.x).Key(state.y).Sort();
		// The smooth is reported at the distinct x values, increasing.
		vector<double> grid = state.x;
		std::sort(grid.begin(), grid.end());
		grid.erase(std::unique(grid.begin(), grid.end()), grid.end());
		AnofoxDataArray x = {state.x.data(), nullptr, state.x.size()};
		AnofoxDataArray y = {state.y.data(), nullptr, state.y.size()};
		AnofoxPSplineResult fit;
		AnofoxError error;
		if (!anofox_pspline_fit_predict(x, y, grid.data(), grid.size(), bind.options, PSPLINE_NONE, 0.95, &fit,
		                                &error)) {
			ThrowUnlessDegenerate(kFitName, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		idx_t k = 0;
		XyRowsSetDoubleList(StructEntry(entries[k++]), row, grid.data(), grid.size());
		XyRowsSetDoubleList(StructEntry(entries[k++]), row, fit.fitted, static_cast<idx_t>(fit.n_points));
		XyRowsSetDoubleList(StructEntry(entries[k++]), row, fit.se_fit, static_cast<idx_t>(fit.n_points));
		XyRowsSetDoubleList(StructEntry(entries[k++]), row, fit.coefficients,
		                    static_cast<idx_t>(fit.coefficients_len));
		XyRowsSetDouble(StructEntry(entries[k++]), row, fit.lambda);
		XyRowsSetDouble(StructEntry(entries[k++]), row, fit.edf);
		XyRowsSetDouble(StructEntry(entries[k++]), row, fit.sigma2);
		XyRowsSetDouble(StructEntry(entries[k++]), row, fit.r_squared);
		FlatDataMut<int64_t>(StructEntry(entries[k++]))[row] = static_cast<int64_t>(fit.coefficients_len);
		FlatDataMut<int64_t>(StructEntry(entries[k++]))[row] = static_cast<int64_t>(fit.penalty_order);
		FlatDataMut<int64_t>(StructEntry(entries[k++]))[row] = static_cast<int64_t>(fit.n_observations);
		writer.WriteInference(row, nullptr);
		writer.WriteSummary(row, fit.summary);
		anofox_free_pspline_result(&fit);
	}
}

unique_ptr<FunctionData> PSplineFitBind(ClientContext &context, AggregateBindFunction &function,
                                        vector<unique_ptr<Expression>> &arguments) {
	const string name(kFitName);
	auto bind = make_uniq<PSplineBindData>();
	if (arguments.size() >= 3) {
		ParsePSplineOptions(context, *arguments[2], *bind, name, false);
	}
	SetReturnTypeOf(function, PSplineFitResultType());
	PostHogTelemetry::Instance().RecordFunctionCall(name);
	return std::move(bind);
}

//===--------------------------------------------------------------------===//
// pspline_fit_predict_agg
//===--------------------------------------------------------------------===//
//! Update for (y DOUBLE, x DOUBLE[, split VARCHAR][, options]) into the shared
//! fit-predict state with one feature. A training row has a finite y and x and,
//! with a split column, split 'train' / 'training'; a NULL x is kept as a
//! non-training row with a NULL prediction.
void PSplinePredictUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                          Vector &state_vector, idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<PSplineBindData>();
	UnifiedVectorFormat y_data, x_data, split_data, sdata;
	inputs[0].ToUnifiedFormat(count, y_data);
	inputs[1].ToUnifiedFormat(count, x_data);
	const bool with_split = bind.use_split_col && input_count >= 3;
	if (with_split) {
		inputs[2].ToUnifiedFormat(count, split_data);
	}
	state_vector.ToUnifiedFormat(count, sdata);
	auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
	auto x_values = UnifiedVectorFormat::GetData<double>(x_data);
	auto states = (FitPredictRowsState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto y_idx = y_data.sel->get_index(i);
		bool y_valid = y_data.validity.RowIsValid(y_idx);
		double y_val = y_valid ? y_values[y_idx] : std::nan("");
		auto x_idx = x_data.sel->get_index(i);
		if (!x_data.validity.RowIsValid(x_idx)) {
			PushNullXRow(state, y_valid, y_val);
			continue;
		}
		if (!state.initialized) {
			state.n_features = 1;
			state.x_train.resize(1);
			state.initialized = true;
		}
		double x_val = x_values[x_idx];
		bool training = y_valid && std::isfinite(y_val) && std::isfinite(x_val);
		if (with_split) {
			auto s_idx = split_data.sel->get_index(i);
			training = training && split_data.validity.RowIsValid(s_idx) &&
			           FitPredictIsTrainSplit(UnifiedVectorFormat::GetData<string_t>(split_data)[s_idx]);
		}
		state.y_all.push_back(y_val);
		state.y_is_null.push_back(!y_valid);
		state.is_training.push_back(training);
		state.x_all.push_back({x_val});
		if (training) {
			state.y_train.push_back(y_val);
			state.x_train[0].push_back(x_val);
		}
	}
}

void PSplinePredictFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                            idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<PSplineBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (FitPredictRowsState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t result_idx = i + offset;
		if (!state.initialized || state.y_train.empty()) {
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}
		PadNullXRows(state);
		const idx_t n_rows = state.y_all.size();
		vector<double> points(n_rows);
		for (idx_t r = 0; r < n_rows; r++) {
			points[r] = state.x_all[r][0];
		}
		// Canonical row order: bitwise-identical results whatever the thread count (#226).
		SortRowsCanonically(state.y_train, state.x_train);
		AnofoxDataArray x = {state.x_train[0].data(), nullptr, state.x_train[0].size()};
		AnofoxDataArray y = {state.y_train.data(), nullptr, state.y_train.size()};
		AnofoxPSplineResult fit;
		AnofoxError error;
		if (!anofox_pspline_fit_predict(x, y, points.data(), points.size(), bind.options, bind.interval,
		                                bind.confidence_level, &fit, &error)) {
			ThrowUnlessDegenerate(kFitPredictName, error);
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}
		const bool band = bind.interval != PSPLINE_NONE;
		WriteFitPredictRows(result, result_idx, state, fit.fitted, band ? fit.lower : nullptr,
		                    band ? fit.upper : nullptr);
		anofox_free_pspline_result(&fit);
	}
}

template <bool SPLIT>
unique_ptr<FunctionData> PSplinePredictBind(ClientContext &context, AggregateBindFunction &function,
                                            vector<unique_ptr<Expression>> &arguments) {
	const string name(kFitPredictName);
	auto bind = make_uniq<PSplineBindData>();
	bind->use_split_col = SPLIT;
	const idx_t options_idx = SPLIT ? 3 : 2;
	if (arguments.size() > options_idx) {
		ParsePSplineOptions(context, *arguments[options_idx], *bind, name, true);
	}
	SetReturnTypeOf(function, FitPredictRowsResultType());
	PostHogTelemetry::Instance().RecordFunctionCall(name);
	return std::move(bind);
}

} // namespace

void RegisterPSplineAggregateFunctions(ExtensionLoader &loader) {
	const vector<string> categories = {"regression", "smoothing"};
	{
		AggregateFunctionSet set(kFitName);
		for (bool with_options : {false, true}) {
			vector<LogicalType> args = {LogicalType::DOUBLE, LogicalType::DOUBLE};
			if (with_options) {
				args.push_back(LogicalType::ANY);
			}
			AggregateFunction fn(kFitName, args, LogicalType::ANY, AggregateFunction::StateSize<PSplineFitState>,
			                     ANOFOX_AGG_INIT(PSplineFitInitialize),
			                     ANOFOX_GUARDED_UPDATE(PSplineFitUpdate, PSplineFitDestroy, PSplineFitInitialize),
			                     PSplineFitCombine,
			                     ANOFOX_GUARDED_FINALIZE(PSplineFitFinalize, PSplineFitDestroy, PSplineFitInitialize),
			                     nullptr, ANOFOX_AGG_BIND(PSplineFitBind), PSplineFitDestroy);
			set.AddFunction(fn);
		}
		CreateAggregateFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		const string description =
		    "Fits a P-spline smoother of y on x per group (mgcv s(x, bs = \"ps\"): cubic B-splines with a "
		    "difference penalty, smoothing parameter by GCV unless lambda is given) and returns the smooth at the "
		    "distinct x values (x increasing, fitted, se_fit: Bayesian standard error of the mean), the B-spline "
		    "coefficients, lambda, edf (effective degrees of freedom), sigma2 (residual variance), r_squared, "
		    "n_basis, penalty_order and n_observations. Options: n_basis (alias k, 0 = automatic), penalty_order "
		    "(default 2), lambda. Rows with a NULL or non-finite x or y are skipped.";
		const vector<vector<string>> names = {{"y", "x"}, {"y", "x", "options"}};
		const vector<string> examples = {"pspline_fit_agg(y, x)",
		                                 "pspline_fit_agg(y, x, {'n_basis': 20, 'penalty_order': 2})"};
		for (idx_t s = 0; s < names.size(); s++) {
			FunctionDescription d;
			d.description = description;
			d.examples = {examples[s]};
			d.categories = categories;
			d.parameter_names = names[s];
			d.parameter_types = s == 0 ? vector<LogicalType> {LogicalType::DOUBLE, LogicalType::DOUBLE}
			                           : vector<LogicalType> {LogicalType::DOUBLE, LogicalType::DOUBLE,
			                                                  LogicalType::ANY};
			info.descriptions.push_back(std::move(d));
		}
		loader.RegisterFunction(std::move(info));
	}
	{
		AggregateFunctionSet set(kFitPredictName);
		const vector<vector<LogicalType>> signatures = {
		    {LogicalType::DOUBLE, LogicalType::DOUBLE},
		    {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::ANY},
		    {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::VARCHAR},
		    {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::VARCHAR, LogicalType::ANY}};
		for (auto &args : signatures) {
			bool split = args.size() >= 3 && args[2].id() == LogicalTypeId::VARCHAR;
			AggregateFunction fn(
			    kFitPredictName, args, LogicalType::ANY, AggregateFunction::StateSize<FitPredictRowsState>,
			    ANOFOX_AGG_INIT(FitPredictRowsInitialize),
			    ANOFOX_GUARDED_UPDATE(PSplinePredictUpdate, FitPredictRowsDestroy, FitPredictRowsInitialize),
			    FitPredictRowsCombine,
			    ANOFOX_GUARDED_FINALIZE(PSplinePredictFinalize, FitPredictRowsDestroy, FitPredictRowsInitialize),
			    nullptr, split ? ANOFOX_AGG_BIND(PSplinePredictBind<true>) : ANOFOX_AGG_BIND(PSplinePredictBind<false>),
			    FitPredictRowsDestroy);
			set.AddFunction(fn);
		}
		CreateAggregateFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		const string description =
		    "Fits a P-spline smoother of y on x over a partition (training rows: y and x non-NULL and finite and, "
		    "with split_col, split 'train' / 'training' in any case) and returns one STRUCT per row: y, yhat, "
		    "yhat_lower / yhat_upper (interval 'prediction' (default), 'confidence' or 'none', at "
		    "confidence_level, default 0.95) and is_training. Outside the training range the smooth is extended "
		    "linearly. Options as pspline_fit_agg plus confidence_level and interval.";
		const vector<vector<string>> names = {
		    {"y", "x"}, {"y", "x", "options"}, {"y", "x", "split_col"}, {"y", "x", "split_col", "options"}};
		const vector<string> examples = {
		    "pspline_fit_predict_agg(y, x)", "pspline_fit_predict_agg(y, x, {'interval': 'confidence'})",
		    "pspline_fit_predict_agg(y, x, split)", "pspline_fit_predict_agg(y, x, split, {'n_basis': 20})"};
		for (idx_t s = 0; s < signatures.size(); s++) {
			FunctionDescription d;
			d.description = description;
			d.examples = {examples[s]};
			d.categories = categories;
			d.parameter_names = names[s];
			d.parameter_types = signatures[s];
			info.descriptions.push_back(std::move(d));
		}
		loader.RegisterFunction(std::move(info));
	}
}

} // namespace duckdb
