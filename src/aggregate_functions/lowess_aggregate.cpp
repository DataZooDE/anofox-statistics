// LOWESS smoother (#63):
//
//   lowess_agg(y, x[, frac[, iterations]]) -> LIST(STRUCT(x DOUBLE, yhat DOUBLE)), x increasing
//
// Cleveland's robust locally weighted scatterplot smoother, upstream's port of
// R's stats::lowess(x, y, f = frac, iter = iterations) (delta at R's default,
// 1% of the range of x). Rows with a NULL or non-finite x or y are skipped.

#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"
#include "error_dispatch.hpp"
#include "telemetry.hpp"
#include "../include/canonical_order.hpp"

namespace duckdb {

namespace {

constexpr const char *kLowessName = "lowess_agg";

struct LowessState {
	vector<double> x;
	vector<double> y;
};

struct LowessBindData : public FunctionData {
	double frac = 2.0 / 3.0;
	uint32_t iterations = 3;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<LowessBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<LowessBindData>();
		return frac == o.frac && iterations == o.iterations;
	}
};

void LowessInitialize(data_ptr_t state_p) {
	new (state_p) LowessState();
}

void LowessDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (LowessState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~LowessState();
	}
}

void LowessUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat y_data, x_data, sdata;
	inputs[0].ToUnifiedFormat(count, y_data);
	inputs[1].ToUnifiedFormat(count, x_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
	auto x_values = UnifiedVectorFormat::GetData<double>(x_data);
	auto states = (LowessState **)sdata.data;
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

void LowessCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (LowessState **)source_data.data;
	auto targets = (LowessState **)target_data.data;
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

LogicalType LowessResultType() {
	child_list_t<LogicalType> point;
	point.push_back(make_pair("x", LogicalType::DOUBLE));
	point.push_back(make_pair("yhat", LogicalType::DOUBLE));
	return LogicalType::LIST(LogicalType::STRUCT(std::move(point)));
}

void LowessFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                    idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<LowessBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (LowessState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		if (state.x.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		// Canonical row order: bitwise-identical results whatever the thread count (#226).
		CanonicalRows().Key(state.x).Key(state.y).Sort();
		AnofoxDataArray x = {state.x.data(), nullptr, state.x.size()};
		AnofoxDataArray y = {state.y.data(), nullptr, state.y.size()};
		AnofoxLowessResult curve;
		AnofoxError error;
		if (!anofox_lowess(x, y, bind.frac, bind.iterations, std::nan(""), &curve, &error)) {
			ThrowUnlessDegenerate(kLowessName, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		const idx_t n = static_cast<idx_t>(curve.len);
		auto list_data = ListVector::GetData(result);
		auto list_offset = ListVector::GetListSize(result);
		ListVector::Reserve(result, list_offset + n);
		list_data[row].offset = list_offset;
		list_data[row].length = n;
		auto &entries = StructVector::GetEntries(ListVector::GetEntry(result));
		auto x_out = FlatDataMut<double>(StructEntry(entries[0]));
		auto y_out = FlatDataMut<double>(StructEntry(entries[1]));
		for (idx_t k = 0; k < n; k++) {
			x_out[list_offset + k] = curve.x[k];
			y_out[list_offset + k] = curve.fitted[k];
		}
		ListVector::SetListSize(result, list_offset + n);
		anofox_free_lowess_result(&curve);
	}
}

//! The value of a constant argument; a non-constant one is an error naming it.
Value ConstantArgument(ClientContext &context, Expression &expr, const string &name, const char *argument) {
	RequireResolvedParameters(expr);
	if (!expr.IsFoldable()) {
		throw InvalidInputException("%s: %s must be a constant", name, argument);
	}
	return ExpressionExecutor::EvaluateScalar(context, expr);
}

unique_ptr<FunctionData> LowessBind(ClientContext &context, AggregateBindFunction &function,
                                    vector<unique_ptr<Expression>> &arguments) {
	const string name(kLowessName);
	auto bind = make_uniq<LowessBindData>();
	if (arguments.size() >= 3) {
		auto frac = ConstantArgument(context, *arguments[2], name, "frac");
		if (!frac.IsNull()) {
			bind->frac = frac.GetValue<double>();
		}
	}
	if (arguments.size() >= 4) {
		auto iterations = ConstantArgument(context, *arguments[3], name, "iterations");
		if (!iterations.IsNull()) {
			auto v = iterations.GetValue<int64_t>();
			if (v < 0 || v > 1000) {
				throw InvalidInputException("%s: iterations must be in [0, 1000]", name);
			}
			bind->iterations = static_cast<uint32_t>(v);
		}
	}
	if (!(std::isfinite(bind->frac) && bind->frac > 0.0)) {
		throw InvalidInputException("%s: frac must be finite and > 0", name);
	}
	SetReturnTypeOf(function, LowessResultType());
	PostHogTelemetry::Instance().RecordFunctionCall(name);
	return std::move(bind);
}

} // namespace

void RegisterLowessFunction(ExtensionLoader &loader) {
	AggregateFunctionSet set(kLowessName);
	const vector<vector<LogicalType>> signatures = {
	    {LogicalType::DOUBLE, LogicalType::DOUBLE},
	    {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE},
	    {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::INTEGER}};
	for (auto &args : signatures) {
		AggregateFunction fn(kLowessName, args, LogicalType::ANY, AggregateFunction::StateSize<LowessState>,
		                     ANOFOX_AGG_INIT(LowessInitialize), ANOFOX_GUARDED_UPDATE(LowessUpdate, LowessDestroy, LowessInitialize),
		                     LowessCombine, ANOFOX_GUARDED_FINALIZE(LowessFinalize, LowessDestroy, LowessInitialize),
		                     nullptr, ANOFOX_AGG_BIND(LowessBind), LowessDestroy);
		set.AddFunction(fn);
	}
	CreateAggregateFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	const string description =
	    "LOWESS scatterplot smoother of y on x per group (R stats::lowess: local linear fits with tricube weights "
	    "over the frac * n nearest points, iterations robustifying passes; delta at R's default). Returns the curve "
	    "as a LIST of {x, yhat}, x increasing, one point per complete row. frac defaults to 2/3, iterations to 3.";
	const vector<vector<string>> names = {{"y", "x"}, {"y", "x", "frac"}, {"y", "x", "frac", "iterations"}};
	const vector<string> examples = {"lowess_agg(y, x)", "lowess_agg(y, x, 0.3)", "lowess_agg(y, x, 0.3, 0)"};
	for (idx_t s = 0; s < signatures.size(); s++) {
		FunctionDescription d;
		d.description = description;
		d.examples = {examples[s]};
		d.categories = {"regression", "smoothing"};
		d.parameter_names = names[s];
		d.parameter_types = signatures[s];
		info.descriptions.push_back(std::move(d));
	}
	loader.RegisterFunction(std::move(info));
}

} // namespace duckdb
