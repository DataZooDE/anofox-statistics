#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "telemetry.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Jarque-Bera Aggregate State
//===--------------------------------------------------------------------===//
struct JarqueBeraAggregateState {
    vector<double> values;
    bool initialized;

    JarqueBeraAggregateState() : initialized(false) {}

    void Reset() {
        values.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetJarqueBeraAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("statistic", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("skewness", LogicalType::DOUBLE));
    children.push_back(make_pair("kurtosis", LogicalType::DOUBLE));
    children.push_back(make_pair("n", LogicalType::BIGINT));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void JarqueBeraAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) JarqueBeraAggregateState();
}

static void JarqueBeraAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (JarqueBeraAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~JarqueBeraAggregateState();
    }
}

static void JarqueBeraAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                Vector &state_vector, idx_t count) {
    UnifiedVectorFormat input_data;
    inputs[0].ToUnifiedFormat(count, input_data);
    auto input_values = UnifiedVectorFormat::GetData<double>(input_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (JarqueBeraAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto val_idx = input_data.sel->get_index(i);
        if (!input_data.validity.RowIsValid(val_idx)) {
            continue; // Skip NULL values
        }

        double val = input_values[val_idx];
        if (!std::isnan(val)) {
            state.values.push_back(val);
        }
    }
}

static void JarqueBeraAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (JarqueBeraAggregateState **)source_data.data;
    auto targets = (JarqueBeraAggregateState **)target_data.data;

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

static void JarqueBeraAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                  idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (JarqueBeraAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.values.size() < 3) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Prepare FFI data
        AnofoxDataArray data_array;
        data_array.data = state.values.data();
        data_array.validity = nullptr;
        data_array.len = state.values.size();

        AnofoxJarqueBeraResult jb_result;
        AnofoxError error;

        bool success = anofox_jarque_bera(data_array, &jb_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Fill STRUCT result
        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = jb_result.statistic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = jb_result.p_value;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = jb_result.skewness;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = jb_result.kurtosis;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = jb_result.n;

        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> JarqueBeraAggBind(ClientContext &context, AggregateFunction &function,
                                                  vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetJarqueBeraAggResultType();
    PostHogTelemetry::Instance().CaptureFunctionExecution("jarque_bera_agg");
    return nullptr;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterJarqueBeraAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_jarque_bera_agg");

    auto func = AggregateFunction("anofox_stats_jarque_bera_agg", {LogicalType::DOUBLE},
                                  LogicalType::ANY, // Set in bind
                                  AggregateFunction::StateSize<JarqueBeraAggregateState>, JarqueBeraAggInitialize,
                                  JarqueBeraAggUpdate, JarqueBeraAggCombine, JarqueBeraAggFinalize,
                                  nullptr, // simple_update
                                  JarqueBeraAggBind, JarqueBeraAggDestroy);
    func_set.AddFunction(func);

    loader.RegisterFunction(func_set);

    // Also register short alias
    AggregateFunctionSet alias_set("jarque_bera_agg");
    alias_set.AddFunction(func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
