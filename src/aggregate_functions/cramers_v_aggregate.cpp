#include <vector>
#include <map>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Cramer's V Aggregate State
//===--------------------------------------------------------------------===//
struct CramersVAggregateState {
    // Store row/col values to build contingency table
    vector<int64_t> row_values;
    vector<int64_t> col_values;
    bool initialized;

    CramersVAggregateState() : initialized(false) {}

    void Reset() {
        row_values.clear();
        col_values.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetCramersVAggResultType() {
    return LogicalType::DOUBLE;
}

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void CramersVAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) CramersVAggregateState();
}

static void CramersVAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (CramersVAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~CramersVAggregateState();
    }
}

static void CramersVAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                               Vector &state_vector, idx_t count) {
    UnifiedVectorFormat row_data, col_data;
    inputs[0].ToUnifiedFormat(count, row_data);
    inputs[1].ToUnifiedFormat(count, col_data);
    auto row_vals = UnifiedVectorFormat::GetData<int64_t>(row_data);
    auto col_vals = UnifiedVectorFormat::GetData<int64_t>(col_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (CramersVAggregateState **)sdata.data;

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

static void CramersVAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (CramersVAggregateState **)source_data.data;
    auto targets = (CramersVAggregateState **)target_data.data;

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

static void CramersVAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                 idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (CramersVAggregateState **)sdata.data;

    auto result_data = FlatVector::GetData<double>(result);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.row_values.size() < 2) {
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

        // Build row lengths array (all rows have same length = n_cols)
        vector<size_t> row_lengths(n_rows, n_cols);

        double cramers_v;
        AnofoxError error;

        bool success = anofox_cramers_v(table.data(), row_lengths.data(), n_rows,
                                         &cramers_v, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        result_data[result_idx] = cramers_v;
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> CramersVAggBind(ClientContext &context, AggregateFunction &function,
                                                 vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetCramersVAggResultType();
    return nullptr;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterCramersVAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_cramers_v_agg");

    // (row_var BIGINT, col_var BIGINT)
    auto func = AggregateFunction(
        "anofox_stats_cramers_v_agg", {LogicalType::BIGINT, LogicalType::BIGINT},
        LogicalType::DOUBLE,
        AggregateFunction::StateSize<CramersVAggregateState>, CramersVAggInitialize,
        CramersVAggUpdate, CramersVAggCombine, CramersVAggFinalize,
        nullptr, CramersVAggBind, CramersVAggDestroy);
    func_set.AddFunction(func);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("cramers_v_agg");
    alias_set.AddFunction(func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
