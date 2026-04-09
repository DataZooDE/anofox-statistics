#include <vector>
#include <map>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Chi-Square Test Aggregate State
//===--------------------------------------------------------------------===//
struct ChiSquareAggregateState {
    vector<double> row_var;
    vector<double> col_var;
    bool initialized;

    ChiSquareAggregateState() : initialized(false) {}

    void Reset() {
        row_var.clear();
        col_var.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetChiSquareAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("statistic", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("df", LogicalType::BIGINT));
    children.push_back(make_pair("method", LogicalType::VARCHAR));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Bind data
//===--------------------------------------------------------------------===//
struct ChiSquareBindData : public FunctionData {
    ChiSquareMapOptions options;

    ChiSquareBindData() {}

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<ChiSquareBindData>();
        copy->options = options;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<ChiSquareBindData>();
        return options.continuity_correction == other.options.continuity_correction;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void ChiSquareAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) ChiSquareAggregateState();
}

static void ChiSquareAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ChiSquareAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~ChiSquareAggregateState();
    }
}

static void ChiSquareAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                Vector &state_vector, idx_t count) {
    UnifiedVectorFormat row_data, col_data;
    inputs[0].ToUnifiedFormat(count, row_data);
    inputs[1].ToUnifiedFormat(count, col_data);
    auto row_values = UnifiedVectorFormat::GetData<int32_t>(row_data);
    auto col_values = UnifiedVectorFormat::GetData<int32_t>(col_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ChiSquareAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto row_idx = row_data.sel->get_index(i);
        auto col_idx = col_data.sel->get_index(i);

        if (!row_data.validity.RowIsValid(row_idx) || !col_data.validity.RowIsValid(col_idx)) {
            continue;
        }

        int32_t row = row_values[row_idx];
        int32_t col = col_values[col_idx];

        state.row_var.push_back(static_cast<double>(row));
        state.col_var.push_back(static_cast<double>(col));
    }
}

static void ChiSquareAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (ChiSquareAggregateState **)source_data.data;
    auto targets = (ChiSquareAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.row_var = std::move(source.row_var);
            target.col_var = std::move(source.col_var);
            target.initialized = true;
            continue;
        }

        target.row_var.insert(target.row_var.end(), source.row_var.begin(), source.row_var.end());
        target.col_var.insert(target.col_var.end(), source.col_var.begin(), source.col_var.end());
    }
}

static void ChiSquareAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                  idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ChiSquareAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);
    auto &bind_data = aggr_input_data.bind_data->Cast<ChiSquareBindData>();

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.row_var.size() < 1) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        AnofoxDataArray row_array;
        row_array.data = state.row_var.data();
        row_array.validity = nullptr;
        row_array.len = state.row_var.size();

        AnofoxDataArray col_array;
        col_array.data = state.col_var.data();
        col_array.validity = nullptr;
        col_array.len = state.col_var.size();

        AnofoxChiSquareOptions options;
        options.correction = bind_data.options.continuity_correction.value_or(false);

        AnofoxChiSquareResult chisq_result;
        AnofoxError error;

        bool success = anofox_chisq_test(row_array, col_array, options, &chisq_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = chisq_result.statistic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = chisq_result.p_value;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(chisq_result.df);
        auto& method_vector = *struct_entries[struct_idx++];
        FlatVector::GetData<string_t>(method_vector)[result_idx] =
            StringVector::AddString(method_vector, chisq_result.method ? chisq_result.method : "Chi-Square");

        anofox_free_chisq_result(&chisq_result);
        state.Reset();
    }
}

static unique_ptr<FunctionData> ChiSquareAggBind(ClientContext &context, AggregateFunction &function,
                                                  vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetChiSquareAggResultType();
    auto bind_data = make_uniq<ChiSquareBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        Value options_val = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
        bind_data->options = ChiSquareMapOptions::ParseFromValue(options_val);
    }

    PostHogTelemetry::Instance().CaptureFunctionExecution("chisq_test_agg");
    return bind_data;
}

void RegisterChiSquareAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_chisq_test_agg");

    auto func_with_opts = AggregateFunction(
        "anofox_stats_chisq_test_agg", {LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::ANY},
        LogicalType::ANY,
        AggregateFunction::StateSize<ChiSquareAggregateState>, ChiSquareAggInitialize,
        ChiSquareAggUpdate, ChiSquareAggCombine, ChiSquareAggFinalize,
        nullptr, ChiSquareAggBind, ChiSquareAggDestroy);
    func_set.AddFunction(func_with_opts);

    auto func_no_opts = AggregateFunction(
        "anofox_stats_chisq_test_agg", {LogicalType::INTEGER, LogicalType::INTEGER},
        LogicalType::ANY,
        AggregateFunction::StateSize<ChiSquareAggregateState>, ChiSquareAggInitialize,
        ChiSquareAggUpdate, ChiSquareAggCombine, ChiSquareAggFinalize,
        nullptr, ChiSquareAggBind, ChiSquareAggDestroy);
    func_set.AddFunction(func_no_opts);

    CreateAggregateFunctionInfo info(std::move(func_set));
    info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
    FunctionDescription d1;
    d1.description     = "Performs a chi-squared test of independence on a 2×2 contingency table from two categorical columns.";
    d1.examples        = {"anofox_stats_chisq_test_agg(row_var, col_var, {'correction': true})"};
    d1.categories      = {"hypothesis-testing", "categorical"};
    d1.parameter_names = {"row_var", "col_var", "options"};
    d1.parameter_types = {LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::ANY};
    info.descriptions.push_back(std::move(d1));
    FunctionDescription d2;
    d2.description     = "Performs a chi-squared test of independence on a 2×2 contingency table from two categorical columns.";
    d2.examples        = {"anofox_stats_chisq_test_agg(row_var, col_var)"};
    d2.categories      = {"hypothesis-testing", "categorical"};
    d2.parameter_names = {"row_var", "col_var"};
    d2.parameter_types = {LogicalType::INTEGER, LogicalType::INTEGER};
    info.descriptions.push_back(std::move(d2));
    loader.RegisterFunction(std::move(info));

    // Short alias
    {
        AggregateFunctionSet alias_set("chisq_test_agg");
        alias_set.AddFunction(func_with_opts);
        alias_set.AddFunction(func_no_opts);
        CreateAggregateFunctionInfo alias_info(std::move(alias_set));
        alias_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        alias_info.alias_of = "anofox_stats_chisq_test_agg";
        loader.RegisterFunction(std::move(alias_info));
    }
}

} // namespace duckdb
