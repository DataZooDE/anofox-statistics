#include <vector>
#include <map>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Phi Coefficient Aggregate State
//===--------------------------------------------------------------------===//
struct PhiCoefficientAggregateState {
    vector<int64_t> row_values;
    vector<int64_t> col_values;
    bool initialized;

    PhiCoefficientAggregateState() : initialized(false) {}

    void Reset() {
        row_values.clear();
        col_values.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void PhiCoefficientAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) PhiCoefficientAggregateState();
}

static void PhiCoefficientAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (PhiCoefficientAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~PhiCoefficientAggregateState();
    }
}

static void PhiCoefficientAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                     Vector &state_vector, idx_t count) {
    UnifiedVectorFormat row_data, col_data;
    inputs[0].ToUnifiedFormat(count, row_data);
    inputs[1].ToUnifiedFormat(count, col_data);
    auto row_vals = UnifiedVectorFormat::GetData<int64_t>(row_data);
    auto col_vals = UnifiedVectorFormat::GetData<int64_t>(col_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (PhiCoefficientAggregateState **)sdata.data;

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

static void PhiCoefficientAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (PhiCoefficientAggregateState **)source_data.data;
    auto targets = (PhiCoefficientAggregateState **)target_data.data;

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

static void PhiCoefficientAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                       idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (PhiCoefficientAggregateState **)sdata.data;

    auto result_data = FlatVector::GetData<double>(result);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.row_values.size() < 4) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Build 2x2 contingency table from binary row/col values
        // Assumes binary values (0/1 or similar)
        // Table: [[a, b], [c, d]] where:
        // a = row=0, col=0; b = row=0, col=1
        // c = row=1, col=0; d = row=1, col=1
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

        size_t n_rows_cat = row_map.size();
        size_t n_cols_cat = col_map.size();

        // Phi coefficient requires 2x2 table
        if (n_rows_cat != 2 || n_cols_cat != 2) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Build the 2x2 contingency table
        size_t table[2][2] = {{0, 0}, {0, 0}};
        for (size_t j = 0; j < state.row_values.size(); j++) {
            size_t r = row_map[state.row_values[j]];
            size_t c = col_map[state.col_values[j]];
            table[r][c]++;
        }

        double phi;
        AnofoxError error;

        bool success = anofox_phi_coefficient(table[0][0], table[0][1], table[1][0], table[1][1],
                                               &phi, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        result_data[result_idx] = phi;
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> PhiCoefficientAggBind(ClientContext &context, AggregateFunction &function,
                                                       vector<unique_ptr<Expression>> &arguments) {
    function.return_type = LogicalType::DOUBLE;
    return nullptr;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterPhiCoefficientAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_phi_coefficient_agg");

    // (row_var BIGINT, col_var BIGINT)
    auto func = AggregateFunction(
        "anofox_stats_phi_coefficient_agg", {LogicalType::BIGINT, LogicalType::BIGINT},
        LogicalType::DOUBLE,
        AggregateFunction::StateSize<PhiCoefficientAggregateState>, PhiCoefficientAggInitialize,
        PhiCoefficientAggUpdate, PhiCoefficientAggCombine, PhiCoefficientAggFinalize,
        nullptr, PhiCoefficientAggBind, PhiCoefficientAggDestroy);
    func_set.AddFunction(func);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("phi_coefficient_agg");
    alias_set.AddFunction(func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
