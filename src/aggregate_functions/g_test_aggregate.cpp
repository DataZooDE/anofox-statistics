#include <vector>
#include <map>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// G-Test Aggregate State
//===--------------------------------------------------------------------===//
struct GTestAggregateState {
    vector<int64_t> row_values;
    vector<int64_t> col_values;
    bool initialized;

    GTestAggregateState() : initialized(false) {}

    void Reset() {
        row_values.clear();
        col_values.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetGTestAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("statistic", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("df", LogicalType::BIGINT));
    children.push_back(make_pair("method", LogicalType::VARCHAR));

    return LogicalType::STRUCT(std::move(children));
}


//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void GTestAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) GTestAggregateState();
}

static void GTestAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (GTestAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~GTestAggregateState();
    }
}

static void GTestAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                            Vector &state_vector, idx_t count) {
    UnifiedVectorFormat row_data, col_data;
    inputs[0].ToUnifiedFormat(count, row_data);
    inputs[1].ToUnifiedFormat(count, col_data);
    auto row_vals = UnifiedVectorFormat::GetData<int64_t>(row_data);
    auto col_vals = UnifiedVectorFormat::GetData<int64_t>(col_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (GTestAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto row_idx = row_data.sel->get_index(i);
        auto col_idx = col_data.sel->get_index(i);

        if (!row_data.validity.RowIsValid(row_idx) || !col_data.validity.RowIsValid(col_idx)) {
            continue;
        }

        state.row_values.push_back(row_vals[row_idx]);
        state.col_values.push_back(col_vals[col_idx]);
    }
}

static void GTestAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (GTestAggregateState **)source_data.data;
    auto targets = (GTestAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.row_values = std::move(source.row_values);
            target.col_values = std::move(source.col_values);
            target.initialized = true;
            continue;
        }

        target.row_values.insert(target.row_values.end(), source.row_values.begin(), source.row_values.end());
        target.col_values.insert(target.col_values.end(), source.col_values.begin(), source.col_values.end());
    }
}

static void GTestAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                              idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (GTestAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.row_values.size() < 4) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Build contingency table from row/col values
        std::map<int64_t, size_t> row_map, col_map;
        for (auto r : state.row_values) {
            if (row_map.find(r) == row_map.end()) {
                row_map[r] = row_map.size();
            }
        }
        for (auto c : state.col_values) {
            if (col_map.find(c) == col_map.end()) {
                col_map[c] = col_map.size();
            }
        }

        size_t n_rows = row_map.size();
        size_t n_cols = col_map.size();

        if (n_rows < 2 || n_cols < 2) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Build the contingency table (row-major)
        vector<size_t> table(n_rows * n_cols, 0);
        for (size_t j = 0; j < state.row_values.size(); j++) {
            size_t r = row_map[state.row_values[j]];
            size_t c = col_map[state.col_values[j]];
            table[r * n_cols + c]++;
        }

        // Build row lengths array
        vector<size_t> row_lengths(n_rows, n_cols);

        AnofoxChiSquareResult g_result;
        AnofoxError error;

        bool success = anofox_g_test(table.data(), row_lengths.data(), n_rows,
                                      &g_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = g_result.statistic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = g_result.p_value;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(g_result.df);
        FlatVector::GetData<string_t>(*struct_entries[struct_idx++])[result_idx] =
            StringVector::AddString(*struct_entries[struct_idx - 1], g_result.method ? g_result.method : "G-test");

        anofox_free_chisq_result(&g_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> GTestAggBind(ClientContext &context, AggregateFunction &function,
                                              vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetGTestAggResultType();
    PostHogTelemetry::Instance().CaptureFunctionExecution("g_test_agg");
    return nullptr;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterGTestAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_g_test_agg");

    // (row_var BIGINT, col_var BIGINT)
    auto func = AggregateFunction(
        "anofox_stats_g_test_agg", {LogicalType::BIGINT, LogicalType::BIGINT},
        LogicalType::ANY,
        AggregateFunction::StateSize<GTestAggregateState>, GTestAggInitialize,
        GTestAggUpdate, GTestAggCombine, GTestAggFinalize,
        nullptr, GTestAggBind, GTestAggDestroy);
    func_set.AddFunction(func);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("g_test_agg");
    alias_set.AddFunction(func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
