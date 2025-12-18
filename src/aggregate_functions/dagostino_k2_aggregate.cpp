#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"

namespace duckdb {

//===--------------------------------------------------------------------===//
// D'Agostino K² Aggregate State
//===--------------------------------------------------------------------===//
struct DAgostinoK2AggregateState {
    vector<double> values;
    bool initialized;

    DAgostinoK2AggregateState() : initialized(false) {}

    void Reset() {
        values.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetDAgostinoK2AggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("statistic", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("n", LogicalType::BIGINT));
    children.push_back(make_pair("method", LogicalType::VARCHAR));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void DAgostinoK2AggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) DAgostinoK2AggregateState();
}

static void DAgostinoK2AggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (DAgostinoK2AggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~DAgostinoK2AggregateState();
    }
}

static void DAgostinoK2AggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                  Vector &state_vector, idx_t count) {
    UnifiedVectorFormat input_data;
    inputs[0].ToUnifiedFormat(count, input_data);
    auto input_values = UnifiedVectorFormat::GetData<double>(input_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (DAgostinoK2AggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto val_idx = input_data.sel->get_index(i);
        if (!input_data.validity.RowIsValid(val_idx)) {
            continue;
        }

        double val = input_values[val_idx];
        if (!std::isnan(val)) {
            state.values.push_back(val);
        }
    }
}

static void DAgostinoK2AggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (DAgostinoK2AggregateState **)source_data.data;
    auto targets = (DAgostinoK2AggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.values = std::move(source.values);
            target.initialized = true;
            continue;
        }

        target.values.insert(target.values.end(), source.values.begin(), source.values.end());
    }
}

static void DAgostinoK2AggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                    idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (DAgostinoK2AggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        // D'Agostino test requires n >= 20
        if (!state.initialized || state.values.size() < 20) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Prepare FFI data
        AnofoxDataArray data_array;
        data_array.data = state.values.data();
        data_array.validity = nullptr;
        data_array.len = state.values.size();

        AnofoxTestResult test_result;
        AnofoxError error;

        bool success = anofox_dagostino_k2(data_array, &test_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Fill STRUCT result
        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.statistic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.p_value;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(test_result.n);
        FlatVector::GetData<string_t>(*struct_entries[struct_idx++])[result_idx] =
            StringVector::AddString(*struct_entries[struct_idx - 1], test_result.method ? test_result.method : "D'Agostino-Pearson K²");

        anofox_free_test_result(&test_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> DAgostinoK2AggBind(ClientContext &context, AggregateFunction &function,
                                                    vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetDAgostinoK2AggResultType();
    return nullptr;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterDAgostinoK2AggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_dagostino_k2_agg");

    auto func = AggregateFunction("anofox_stats_dagostino_k2_agg", {LogicalType::DOUBLE},
                                  LogicalType::ANY,
                                  AggregateFunction::StateSize<DAgostinoK2AggregateState>, DAgostinoK2AggInitialize,
                                  DAgostinoK2AggUpdate, DAgostinoK2AggCombine, DAgostinoK2AggFinalize,
                                  nullptr, DAgostinoK2AggBind, DAgostinoK2AggDestroy);
    func_set.AddFunction(func);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("dagostino_k2_agg");
    alias_set.AddFunction(func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
