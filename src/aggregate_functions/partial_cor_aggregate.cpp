#include <cmath>
#include <cstddef>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "../include/result_fields.hpp"
#include "../include/test_struct.hpp"
#include "aggregate_finalize_guard.hpp"
#include "canonical_order.hpp"
#include "list_input.hpp"
#include "telemetry.hpp"
#include "xy_rows_aggregate.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Partial and semi-partial correlation (issue #296).
//
//   partial_cor_agg(x DOUBLE, y DOUBLE, controls DOUBLE[] [, options])
//   semi_partial_cor_agg(x DOUBLE, y DOUBLE, controls DOUBLE[] [, options])
//
// Upstream anofox-statistics correlation::partial_cor / semi_partial_cor
// (R ppcor::pcor.test / spcor.test). options: {'method': 'pearson' | 'spearman'}.
// A row with a NULL or NaN in x, y, the control list or one of its elements is
// skipped (as pearson_agg / spearman_agg skip incomplete pairs). Too few rows
// (n < n_controls + 3) or a constant column give NULL.
//
// The rows live in an XyRowsState: y_values = y, x_columns = [x, controls...].
//===--------------------------------------------------------------------===//

static_assert(offsetof(AnofoxPartialCorResult, n) == 4 * sizeof(double) &&
                  sizeof(AnofoxPartialCorResult) ==
                      offsetof(AnofoxPartialCorResult, unified) + sizeof(AnofoxTestSummaryFFI),
              "AnofoxPartialCorResult must match PartialCorResultFFI in crates/anofox-stats-ffi/src/partial_cor_ffi.rs");

namespace {

constexpr int32_t kPartial = 0;
constexpr int32_t kSemiPartial = 1;

struct PartialCorBindData : public FunctionData {
	int32_t kind = kPartial;
	bool spearman = false;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<PartialCorBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<PartialCorBindData>();
		return kind == o.kind && spearman == o.spearman;
	}
};

const char *FunctionName(int32_t kind) {
	return kind == kSemiPartial ? "semi_partial_cor_agg" : "partial_cor_agg";
}

LogicalType PartialCorResultType() {
	child_list_t<LogicalType> children;
	children.push_back(make_pair("r", LogicalType::DOUBLE));
	children.push_back(make_pair("statistic", LogicalType::DOUBLE));
	children.push_back(make_pair("p_value", LogicalType::DOUBLE));
	children.push_back(make_pair("df", LogicalType::DOUBLE));
	children.push_back(make_pair("n", LogicalType::BIGINT));
	children.push_back(make_pair("n_controls", LogicalType::BIGINT));
	children.push_back(make_pair("method", LogicalType::VARCHAR));
	children.push_back(make_pair("alternative", LogicalType::VARCHAR));
	AppendTestResultFields(children);
	return LogicalType::STRUCT(std::move(children));
}

void PartialCorUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat x_data, y_data, z_data, sdata;
	inputs[0].ToUnifiedFormat(count, x_data);
	inputs[1].ToUnifiedFormat(count, y_data);
	inputs[2].ToUnifiedFormat(count, z_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto x_values = UnifiedVectorFormat::GetData<double>(x_data);
	auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
	auto z_lists = UnifiedVectorFormat::GetData<list_entry_t>(z_data);
	auto &z_child = ListVector::GetEntry(inputs[2]);
	auto z_child_data = FlatDataMut<double>(z_child);
	auto &z_child_validity = FlatValidityMut(z_child);
	auto states = (XyRowsState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto x_idx = x_data.sel->get_index(i);
		auto y_idx = y_data.sel->get_index(i);
		auto z_idx = z_data.sel->get_index(i);
		if (!x_data.validity.RowIsValid(x_idx) || !y_data.validity.RowIsValid(y_idx) ||
		    !z_data.validity.RowIsValid(z_idx)) {
			continue;
		}
		const auto controls = z_lists[z_idx];
		if (ListHasNullElement(z_child_validity, controls)) {
			continue;
		}
		const double x = x_values[x_idx];
		const double y = y_values[y_idx];
		if (std::isnan(x) || std::isnan(y)) {
			continue;
		}
		bool has_nan = false;
		for (idx_t j = 0; j < controls.length; j++) {
			has_nan = has_nan || std::isnan(z_child_data[controls.offset + j]);
		}
		if (has_nan) {
			continue;
		}
		if (!state.initialized) {
			state.n_features = 1 + controls.length;
			state.x_columns.resize(state.n_features);
			state.initialized = true;
		}
		if (1 + controls.length != state.n_features) {
			throw InvalidInputException("Inconsistent control count: expected %lu, got %lu", state.n_features - 1,
			                            controls.length);
		}
		state.y_values.push_back(y);
		state.x_columns[0].push_back(x);
		for (idx_t j = 0; j < controls.length; j++) {
			state.x_columns[1 + j].push_back(z_child_data[controls.offset + j]);
		}
	}
}

void WriteDouble(Vector &vec, idx_t row, double value) {
	if (std::isnan(value)) {
		FlatVector::SetNull(vec, row, true);
	} else {
		FlatDataMut<double>(vec)[row] = value;
	}
}

void PartialCorFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                        idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<PartialCorBindData>();
	const char *fn_name = FunctionName(bind.kind);
	XyRowsStatePointers states(state_vector, count);
	auto &fields = StructVector::GetEntries(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		const idx_t row = i + offset;
		if (!state.initialized || state.y_values.size() < 3) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		// Canonical row order: bitwise-identical results whatever the thread count (#226).
		SortRowsCanonically(state.y_values, state.x_columns);

		const size_t n = state.y_values.size(); // size_t: AnofoxDataArray::len (32-bit on wasm)
		AnofoxDataArray x_arr {state.x_columns[0].data(), nullptr, n};
		AnofoxDataArray y_arr {state.y_values.data(), nullptr, n};
		vector<AnofoxDataArray> controls;
		for (idx_t j = 1; j < state.n_features; j++) {
			controls.push_back({state.x_columns[j].data(), nullptr, n});
		}

		AnofoxPartialCorOptions options {};
		options.kind = bind.kind;
		options.method = bind.spearman ? 1 : 0;

		AnofoxPartialCorResult res {};
		AnofoxError error;
		if (!anofox_partial_cor(x_arr, y_arr, controls.empty() ? nullptr : controls.data(), controls.size(), options,
		                        &res, &error)) {
			ThrowUnlessDegenerate(fn_name, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}

		idx_t f = 0;
		WriteDouble(StructEntry(fields[f++]), row, res.r);
		WriteDouble(StructEntry(fields[f++]), row, res.statistic);
		WriteDouble(StructEntry(fields[f++]), row, res.p_value);
		WriteDouble(StructEntry(fields[f++]), row, res.df);
		FlatDataMut<int64_t>(StructEntry(fields[f++]))[row] = static_cast<int64_t>(res.n);
		FlatDataMut<int64_t>(StructEntry(fields[f++]))[row] = static_cast<int64_t>(res.n_controls);
		SetResultString(StructEntry(fields[f++]), row, res.unified.method);
		SetResultString(StructEntry(fields[f++]), row, "two_sided");
		WriteTestResultFields(result, row, f, res.unified);
	}
}

unique_ptr<FunctionData> PartialCorBindImpl(ClientContext &context, AggregateBindFunction &function,
                                            vector<unique_ptr<Expression>> &arguments, int32_t kind) {
	const char *fn_name = FunctionName(kind);
	auto bind = make_uniq<PartialCorBindData>();
	bind->kind = kind;
	if (arguments.size() >= 4) {
		Value options_val = EvaluateConstantOptions(context, *arguments[3], fn_name);
		auto opts = PartialCorMapOptions::ParseFromValue(options_val, fn_name);
		bind->spearman = opts.spearman.value_or(false);
	}
	SetReturnTypeOf(function, PartialCorResultType());
	PostHogTelemetry::Instance().RecordFunctionCall(fn_name);
	return std::move(bind);
}

unique_ptr<FunctionData> PartialCorBind(ClientContext &context, AggregateBindFunction &function,
                                        vector<unique_ptr<Expression>> &arguments) {
	return PartialCorBindImpl(context, function, arguments, kPartial);
}

unique_ptr<FunctionData> SemiPartialCorBind(ClientContext &context, AggregateBindFunction &function,
                                            vector<unique_ptr<Expression>> &arguments) {
	return PartialCorBindImpl(context, function, arguments, kSemiPartial);
}

void RegisterOne(ExtensionLoader &loader, int32_t kind) {
	const char *fn_name = FunctionName(kind);
	AggregateFunctionSet func_set(fn_name);
	const auto list_double = LogicalType::LIST(LogicalType::DOUBLE);
	const vector<vector<LogicalType>> signatures = {
	    {LogicalType::DOUBLE, LogicalType::DOUBLE, list_double},
	    {LogicalType::DOUBLE, LogicalType::DOUBLE, list_double, LogicalType::ANY}};
	for (auto &args : signatures) {
		AggregateFunction fn(fn_name, args, LogicalType::ANY, AggregateFunction::StateSize<XyRowsState>,
		                     ANOFOX_AGG_INIT(XyRowsInitialize),
		                     ANOFOX_GUARDED_UPDATE(PartialCorUpdate, XyRowsDestroy, XyRowsInitialize), XyRowsCombine,
		                     ANOFOX_GUARDED_FINALIZE(PartialCorFinalize, XyRowsDestroy, XyRowsInitialize), nullptr,
		                     kind == kSemiPartial ? ANOFOX_AGG_BIND(SemiPartialCorBind) : ANOFOX_AGG_BIND(PartialCorBind),
		                     XyRowsDestroy);
		func_set.AddFunction(fn);
	}

	CreateAggregateFunctionInfo info(std::move(func_set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	const string description =
	    kind == kSemiPartial
	        ? "Semi-partial (part) correlation of x and y with the controls removed from y only (R "
	          "ppcor::spcor.test), with its t test. At most one control variable. options: {'method': 'pearson' | "
	          "'spearman'}."
	        : "Partial correlation of x and y controlling for the controls (R ppcor::pcor.test), with its t test "
	          "(df = n - 2 - n_controls). options: {'method': 'pearson' | 'spearman'}.";
	const vector<vector<string>> names = {{"x", "y", "controls"}, {"x", "y", "controls", "options"}};
	const string base = string(fn_name) + (kind == kSemiPartial ? "(x, y, [z]" : "(x, y, [z1, z2]");
	const vector<string> examples = {base + ")", base + ", {'method': 'spearman'})"};
	for (idx_t k = 0; k < signatures.size(); k++) {
		FunctionDescription d;
		d.description = description;
		d.examples = {examples[k]};
		d.categories = {"correlation"};
		d.parameter_names = names[k];
		d.parameter_types = signatures[k];
		info.descriptions.push_back(std::move(d));
	}
	loader.RegisterFunction(std::move(info));
}

} // namespace

void RegisterPartialCorAggregateFunctions(ExtensionLoader &loader) {
	RegisterOne(loader, kPartial);
	RegisterOne(loader, kSemiPartial);
}

} // namespace duckdb
