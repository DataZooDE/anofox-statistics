// Coefficient-returning aggregates for PLS, quantile and isotonic regression.
//
// These methods previously only had *_fit_predict_agg / *_fit_predict_by forms, so
// the fitted model itself could not be retrieved. The aggregates here mirror the
// other *_fit_agg functions: (y, x[, options]) per group, returning a STRUCT.

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/planner/expression.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"
#include "aggregate_combine.hpp"
#include "error_dispatch.hpp"
#include "list_input.hpp"
#include "telemetry.hpp"

namespace duckdb {

namespace {

//===--------------------------------------------------------------------===//
// Shared state: complete (y, x) rows of one group
//===--------------------------------------------------------------------===//
struct ExtraFitState {
	vector<double> y_values;
	vector<vector<double>> x_columns;
	idx_t n_features;
	bool initialized;

	ExtraFitState() : n_features(0), initialized(false) {
	}
};

//! State pointers of a state vector, whatever its vector type.
struct StatePointers {
	UnifiedVectorFormat format;
	StatePointers(Vector &v, idx_t count) {
		v.ToUnifiedFormat(count, format);
	}
	ExtraFitState &operator[](idx_t i) const {
		return *reinterpret_cast<ExtraFitState *const *>(format.data)[format.sel->get_index(i)];
	}
};

struct ExtraFitBindData : public FunctionData {
	bool fit_intercept = true;
	size_t n_components = 1;
	double tau = 0.5;
	uint32_t max_iterations = 1000;
	double tolerance = 1e-6;
	bool increasing = true;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ExtraFitBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<ExtraFitBindData>();
		return fit_intercept == o.fit_intercept && n_components == o.n_components && tau == o.tau &&
		       max_iterations == o.max_iterations && tolerance == o.tolerance && increasing == o.increasing;
	}
};

void ExtraFitInitialize(const AggregateFunction &, data_ptr_t state_p) {
	new (state_p) ExtraFitState();
}

void ExtraFitDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	StatePointers states(state_vector, count);
	for (idx_t i = 0; i < count; i++) {
		states[i].~ExtraFitState();
	}
}

//! Update for (y DOUBLE, x LIST(DOUBLE)); rows with any NULL/NaN are skipped later by Rust,
//! rows with a NULL y, NULL list or NULL list element are skipped here.
void ExtraFitListUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat y_data, x_data, sdata;
	inputs[0].ToUnifiedFormat(count, y_data);
	inputs[1].ToUnifiedFormat(count, x_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
	auto x_list_data = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &x_child = ListVector::GetEntry(inputs[1]);
	auto x_child_data = FlatVector::GetData<double>(x_child);
	auto &x_child_validity = FlatVector::Validity(x_child);
	auto states = (ExtraFitState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto y_idx = y_data.sel->get_index(i);
		auto x_idx = x_data.sel->get_index(i);
		if (!y_data.validity.RowIsValid(y_idx) || !x_data.validity.RowIsValid(x_idx)) {
			continue;
		}
		auto list_entry = x_list_data[x_idx];
		if (ListHasNullElement(x_child_validity, list_entry)) {
			continue;
		}
		if (!state.initialized) {
			state.n_features = list_entry.length;
			state.x_columns.resize(state.n_features);
			state.initialized = true;
		}
		if (list_entry.length != state.n_features) {
			throw InvalidInputException("Inconsistent feature count: expected %lu, got %lu", state.n_features,
			                            list_entry.length);
		}
		state.y_values.push_back(y_values[y_idx]);
		for (idx_t j = 0; j < state.n_features; j++) {
			state.x_columns[j].push_back(x_child_data[list_entry.offset + j]);
		}
	}
}

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

void ExtraFitCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data, idx_t count) {
	StatePointers sources(source_vector, count);
	StatePointers targets(target_vector, count);
	for (idx_t i = 0; i < count; i++) {
		auto &source = sources[i];
		auto &target = targets[i];
		if (!source.initialized) {
			continue;
		}
		if (!target.initialized) {
			target.y_values = CombineTake(source.y_values, aggr_input_data);
			target.x_columns = CombineTake(source.x_columns, aggr_input_data);
			target.n_features = source.n_features;
			target.initialized = true;
			continue;
		}
		if (source.n_features != target.n_features) {
			throw InvalidInputException("Inconsistent feature count: expected %lu, got %lu", target.n_features,
			                            source.n_features);
		}
		target.y_values.insert(target.y_values.end(), source.y_values.begin(), source.y_values.end());
		for (idx_t j = 0; j < target.n_features; j++) {
			target.x_columns[j].insert(target.x_columns[j].end(), source.x_columns[j].begin(),
			                           source.x_columns[j].end());
		}
	}
}

vector<AnofoxDataArray> ColumnArrays(ExtraFitState &state) {
	vector<AnofoxDataArray> arrays;
	for (auto &col : state.x_columns) {
		arrays.push_back({col.data(), nullptr, col.size()});
	}
	return arrays;
}

void SetDoubleList(Vector &list_vec, idx_t row, const double *values, idx_t n) {
	auto list_data = FlatVector::GetData<list_entry_t>(list_vec);
	auto offset = ListVector::GetListSize(list_vec);
	ListVector::Reserve(list_vec, offset + n);
	auto child_data = FlatVector::GetData<double>(ListVector::GetEntry(list_vec));
	auto &child_validity = FlatVector::Validity(ListVector::GetEntry(list_vec));
	for (idx_t j = 0; j < n; j++) {
		if (std::isfinite(values[j])) {
			child_data[offset + j] = values[j];
		} else {
			child_validity.SetInvalid(offset + j);
		}
	}
	list_data[row].offset = offset;
	list_data[row].length = n;
	ListVector::SetListSize(list_vec, offset + n);
}

void SetDouble(Vector &vec, idx_t row, double value) {
	if (std::isfinite(value)) {
		FlatVector::GetData<double>(vec)[row] = value;
	} else {
		FlatVector::SetNull(vec, row, true);
	}
}

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
	return LogicalType::STRUCT(std::move(c));
}

void PlsFitAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                       idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<ExtraFitBindData>();
	StatePointers states(state_vector, count);
	auto &entries = StructVector::GetEntries(result);
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
		SetDoubleList(*entries[0], row, fit.coefficients, fit.coefficients_len);
		SetDouble(*entries[1], row, bind.fit_intercept ? fit.intercept : 0.0);
		SetDouble(*entries[2], row, fit.r_squared);
		FlatVector::GetData<int64_t>(*entries[3])[row] = (int64_t)fit.n_components;
		FlatVector::GetData<int64_t>(*entries[4])[row] = (int64_t)fit.n_observations;
		FlatVector::GetData<int64_t>(*entries[5])[row] = (int64_t)fit.n_features;
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
	return LogicalType::STRUCT(std::move(c));
}

void QuantileFitAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                            idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<ExtraFitBindData>();
	StatePointers states(state_vector, count);
	auto &entries = StructVector::GetEntries(result);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		idx_t row = i + offset;
		if (!state.initialized || state.y_values.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
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
		SetDoubleList(*entries[0], row, fit.coefficients, fit.coefficients_len);
		SetDouble(*entries[1], row, bind.fit_intercept ? fit.intercept : 0.0);
		SetDouble(*entries[2], row, fit.tau);
		FlatVector::GetData<int64_t>(*entries[3])[row] = (int64_t)fit.n_observations;
		FlatVector::GetData<int64_t>(*entries[4])[row] = (int64_t)fit.n_features;
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
	return LogicalType::STRUCT(std::move(c));
}

void IsotonicFitAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                            idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<ExtraFitBindData>();
	StatePointers states(state_vector, count);
	auto &entries = StructVector::GetEntries(result);
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
		SetDoubleList(*entries[0], row, knots.data(), knots.size());
		SetDoubleList(*entries[1], row, fitted.data(), fitted.size());
		FlatVector::GetData<bool>(*entries[2])[row] = fit.increasing;
		SetDouble(*entries[3], row, fit.r_squared);
		FlatVector::GetData<int64_t>(*entries[4])[row] = (int64_t)fit.n_observations;
		anofox_free_isotonic_result(&fit);
	}
}

//===--------------------------------------------------------------------===//
// Bind
//===--------------------------------------------------------------------===//
template <const char *NAME>
unique_ptr<FunctionData> ExtraFitBind(ClientContext &context, AggregateFunction &, vector<unique_ptr<Expression>> &args) {
	auto bind = make_uniq<ExtraFitBindData>();
	if (args.size() >= 3) {
		// Each function declares the keys it reads; anything else is rejected.
		// 'quantile' resolves to 'tau' for quantile_fit_agg (contextual alias).
		static const vector<string> pls_keys = {"fit_intercept", "n_components"};
		static const vector<string> quantile_keys = {"fit_intercept", "tau", "max_iterations", "tolerance"};
		static const vector<string> isotonic_keys = {"increasing"};
		const string name(NAME);
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
	}
	if (!(bind->tau > 0.0 && bind->tau < 1.0)) {
		throw InvalidInputException("%s: tau must be in (0, 1), got %g", NAME, bind->tau);
	}
	if (bind->n_components == 0) {
		throw InvalidInputException("%s: n_components must be at least 1", NAME);
	}
	PostHogTelemetry::Instance().RecordFunctionCall(NAME);
	return std::move(bind);
}

constexpr char kPlsName[] = "pls_fit_agg";
constexpr char kQuantileName[] = "quantile_fit_agg";
constexpr char kIsotonicName[] = "isotonic_fit_agg";

void RegisterExtraFit(ExtensionLoader &loader, const char *name, const LogicalType &x_type, aggregate_update_t update,
                      aggregate_finalize_t finalize, bind_aggregate_function_t bind, const LogicalType &result_type,
                      const string &description, const string &example, const string &options_example) {
	AggregateFunctionSet set(name);
	for (bool with_options : {false, true}) {
		vector<LogicalType> args = {LogicalType::DOUBLE, x_type};
		if (with_options) {
			args.push_back(LogicalType::ANY);
		}
		AggregateFunction fn(name, args, result_type, AggregateFunction::StateSize<ExtraFitState>, ExtraFitInitialize,
		                     update, ExtraFitCombine, finalize, nullptr, bind, ExtraFitDestroy);
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
	RegisterExtraFit(loader, kPlsName, LogicalType::LIST(LogicalType::DOUBLE), ExtraFitListUpdate, PlsFitAggFinalize,
	                 ExtraFitBind<kPlsName>, PlsFitAggResultType(),
	                 "Fits a Partial Least Squares regression per group and returns the coefficients, intercept, R^2 "
	                 "and number of components.",
	                 "pls_fit_agg(y, [x1, x2, x3])", "pls_fit_agg(y, [x1, x2, x3], {'n_components': 2})");
	RegisterExtraFit(loader, kQuantileName, LogicalType::LIST(LogicalType::DOUBLE), ExtraFitListUpdate,
	                 QuantileFitAggFinalize, ExtraFitBind<kQuantileName>, QuantileFitAggResultType(),
	                 "Fits a linear quantile regression per group and returns the coefficients and intercept for the "
	                 "requested quantile tau.",
	                 "quantile_fit_agg(y, [x1, x2])", "quantile_fit_agg(y, [x1, x2], {'tau': 0.9})");
	RegisterExtraFit(loader, kIsotonicName, LogicalType::DOUBLE, ExtraFitScalarUpdate, IsotonicFitAggFinalize,
	                 ExtraFitBind<kIsotonicName>, IsotonicFitAggResultType(),
	                 "Fits a monotonic (isotonic) regression of y on a single x per group and returns the step "
	                 "function as sorted knots x with fitted values.",
	                 "isotonic_fit_agg(y, x)", "isotonic_fit_agg(y, x, {'increasing': false})");
}

} // namespace duckdb
