#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/common/types/data_chunk.hpp"

#include "../include/anofox_stats_ffi.h"

#include <vector>

namespace duckdb {

//===--------------------------------------------------------------------===//
// VIF Aggregate State - accumulates feature columns
//===--------------------------------------------------------------------===//
struct VifAggregateState {
    vector<vector<double>> x_columns;
    idx_t n_features;
    bool initialized;

    VifAggregateState() : n_features(0), initialized(false) {}

    void Reset() {
        x_columns.clear();
        n_features = 0;
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void VifAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) VifAggregateState();
}

static void VifAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (VifAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~VifAggregateState();
    }
}

static void VifAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                          Vector &state_vector, idx_t count) {
    UnifiedVectorFormat x_data;
    inputs[0].ToUnifiedFormat(count, x_data);  // x: LIST(DOUBLE)

    auto x_list_data = ListVector::GetData(inputs[0]);
    auto &x_child = ListVector::GetEntry(inputs[0]);
    auto x_child_data = FlatVector::GetData<double>(x_child);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (VifAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];

        auto x_idx = x_data.sel->get_index(i);
        if (!x_data.validity.RowIsValid(x_idx)) {
            continue;  // Skip NULL x values
        }

        auto list_entry = x_list_data[x_idx];
        idx_t n_features = list_entry.length;

        // Initialize x_columns on first valid row
        if (!state.initialized) {
            state.n_features = n_features;
            state.x_columns.resize(n_features);
            state.initialized = true;
        }

        // Validate consistent feature count
        if (n_features != state.n_features) {
            throw InvalidInputException(
                "Inconsistent feature count: expected %lu, got %lu",
                state.n_features, n_features);
        }

        // Accumulate x values
        for (idx_t j = 0; j < n_features; j++) {
            double x_val = x_child_data[list_entry.offset + j];
            if (!std::isnan(x_val)) {
                state.x_columns[j].push_back(x_val);
            }
        }
    }
}

static void VifAggCombine(Vector &source_vector, Vector &target_vector,
                           AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (VifAggregateState **)source_data.data;
    auto targets = (VifAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.x_columns = std::move(source.x_columns);
            target.n_features = source.n_features;
            target.initialized = true;
            continue;
        }

        if (source.n_features != target.n_features) {
            throw InvalidInputException(
                "Cannot combine states with different feature counts: %lu vs %lu",
                source.n_features, target.n_features);
        }

        for (idx_t j = 0; j < target.n_features; j++) {
            target.x_columns[j].insert(target.x_columns[j].end(),
                                        source.x_columns[j].begin(),
                                        source.x_columns[j].end());
        }
    }
}

// Helper to set a list in result
static void SetListInResult(Vector &list_vec, idx_t row, double *data, size_t len) {
    auto &child = ListVector::GetEntry(list_vec);
    auto offset = ListVector::GetListSize(list_vec);
    ListVector::SetListSize(list_vec, offset + len);
    auto vec_data = FlatVector::GetData<double>(child);
    for (size_t i = 0; i < len; i++) {
        vec_data[offset + i] = data[i];
    }
    ListVector::GetData(list_vec)[row] = {offset, (idx_t)len};
}

static void VifAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data,
                            Vector &result, idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (VifAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.n_features < 2 || state.x_columns[0].size() < 3) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Prepare FFI data
        vector<AnofoxDataArray> x_arrays;
        for (auto &col : state.x_columns) {
            AnofoxDataArray arr;
            arr.data = col.data();
            arr.validity = nullptr;
            arr.len = col.size();
            x_arrays.push_back(arr);
        }

        double *vif_values = nullptr;
        size_t vif_len = 0;
        AnofoxError error;

        bool success = anofox_compute_vif(
            x_arrays.data(),
            x_arrays.size(),
            &vif_values,
            &vif_len,
            &error
        );

        if (!success || vif_values == nullptr) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Set list result
        SetListInResult(result, result_idx, vif_values, vif_len);

        anofox_free_vif(vif_values);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> VifAggBind(ClientContext &context, AggregateFunction &function,
                                            vector<unique_ptr<Expression>> &arguments) {
    function.return_type = LogicalType::LIST(LogicalType::DOUBLE);
    return nullptr;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterVifAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_vif_agg");

    auto func = AggregateFunction(
        "anofox_stats_vif_agg",
        {LogicalType::LIST(LogicalType::DOUBLE)},
        LogicalType::ANY,  // Set in bind
        AggregateFunction::StateSize<VifAggregateState>,
        VifAggInitialize,
        VifAggUpdate,
        VifAggCombine,
        VifAggFinalize,
        nullptr,  // simple_update
        VifAggBind,
        VifAggDestroy
    );
    func_set.AddFunction(func);

    loader.RegisterFunction(func_set);

    // Also register short alias
    AggregateFunctionSet alias_set("vif_agg");
    alias_set.AddFunction(func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
