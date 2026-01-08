#include <vector>
#include <map>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "telemetry.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// One-Way ANOVA Aggregate State
//===--------------------------------------------------------------------===//
struct AnovaAggregateState {
    vector<double> values;
    vector<double> groups;  // Stored as double for FFI
    bool initialized;

    AnovaAggregateState() : initialized(false) {}

    void Reset() {
        values.clear();
        groups.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetAnovaAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("f_statistic", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("df_between", LogicalType::BIGINT));
    children.push_back(make_pair("df_within", LogicalType::BIGINT));
    children.push_back(make_pair("ss_between", LogicalType::DOUBLE));
    children.push_back(make_pair("ss_within", LogicalType::DOUBLE));
    children.push_back(make_pair("n_groups", LogicalType::BIGINT));
    children.push_back(make_pair("n", LogicalType::BIGINT));
    children.push_back(make_pair("method", LogicalType::VARCHAR));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void AnovaAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) AnovaAggregateState();
}

static void AnovaAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (AnovaAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~AnovaAggregateState();
    }
}

static void AnovaAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                           Vector &state_vector, idx_t count) {
    UnifiedVectorFormat value_data, group_data;
    inputs[0].ToUnifiedFormat(count, value_data);
    inputs[1].ToUnifiedFormat(count, group_data);
    auto values = UnifiedVectorFormat::GetData<double>(value_data);
    auto groups = UnifiedVectorFormat::GetData<int32_t>(group_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (AnovaAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto val_idx = value_data.sel->get_index(i);
        auto grp_idx = group_data.sel->get_index(i);

        if (!value_data.validity.RowIsValid(val_idx) || !group_data.validity.RowIsValid(grp_idx)) {
            continue;
        }

        double val = values[val_idx];
        int32_t group = groups[grp_idx];

        if (std::isnan(val)) {
            continue;
        }

        state.values.push_back(val);
        state.groups.push_back(static_cast<double>(group));
    }
}

static void AnovaAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (AnovaAggregateState **)source_data.data;
    auto targets = (AnovaAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.values = std::move(source.values);
            target.groups = std::move(source.groups);
            target.initialized = true;
            continue;
        }

        target.values.insert(target.values.end(), source.values.begin(), source.values.end());
        target.groups.insert(target.groups.end(), source.groups.begin(), source.groups.end());
    }
}

static void AnovaAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                              idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (AnovaAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.values.size() < 3) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        AnofoxDataArray values_array;
        values_array.data = state.values.data();
        values_array.validity = nullptr;
        values_array.len = state.values.size();

        AnofoxDataArray groups_array;
        groups_array.data = state.groups.data();
        groups_array.validity = nullptr;
        groups_array.len = state.groups.size();

        AnofoxAnovaResult anova_result;
        AnofoxError error;

        bool success = anofox_one_way_anova(values_array, groups_array, &anova_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = anova_result.f_statistic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = anova_result.p_value;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(anova_result.df_between);
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(anova_result.df_within);
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = anova_result.ss_between;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = anova_result.ss_within;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(anova_result.n_groups);
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(anova_result.n);
        auto& method_vector = *struct_entries[struct_idx++];
        FlatVector::GetData<string_t>(method_vector)[result_idx] =
            StringVector::AddString(method_vector, anova_result.method ? anova_result.method : "One-Way ANOVA");

        anofox_free_anova_result(&anova_result);
        state.Reset();
    }
}

static unique_ptr<FunctionData> AnovaAggBind(ClientContext &context, AggregateFunction &function,
                                              vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetAnovaAggResultType();
    PostHogTelemetry::Instance().CaptureFunctionExecution("one_way_anova_agg");
    return nullptr;
}

void RegisterAnovaAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_one_way_anova_agg");

    auto func = AggregateFunction(
        "anofox_stats_one_way_anova_agg", {LogicalType::DOUBLE, LogicalType::INTEGER},
        LogicalType::ANY,
        AggregateFunction::StateSize<AnovaAggregateState>, AnovaAggInitialize,
        AnovaAggUpdate, AnovaAggCombine, AnovaAggFinalize,
        nullptr, AnovaAggBind, AnovaAggDestroy);
    func_set.AddFunction(func);

    loader.RegisterFunction(func_set);

    AggregateFunctionSet alias_set("one_way_anova_agg");
    alias_set.AddFunction(func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
