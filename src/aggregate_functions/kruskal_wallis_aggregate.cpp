#include <vector>

#include "duckdb.hpp"
#include "../include/two_group.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/test_struct.hpp"
#include "../include/anofox_stats_ffi.h"
#include "../include/result_fields.hpp"
#include "../include/error_dispatch.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Kruskal-Wallis Aggregate State
//===--------------------------------------------------------------------===//
struct KruskalWallisAggregateState {
    vector<double> values;
    vector<double> groups;
    GroupCoder coder; // label -> code (#186)
    bool initialized;

    KruskalWallisAggregateState() : initialized(false) {}

    void Reset() {
        values.clear();
        groups.clear();
        coder.labels.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetKruskalWallisAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("statistic", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("df", LogicalType::DOUBLE));
    children.push_back(make_pair("n", LogicalType::BIGINT));
    children.push_back(make_pair("method", LogicalType::VARCHAR));
    children.push_back(make_pair("alternative", LogicalType::VARCHAR));

    AppendTestResultFields(children);
    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void KruskalWallisAggInitialize(data_ptr_t state_p) {
    new (state_p) KruskalWallisAggregateState();
}

static void KruskalWallisAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (KruskalWallisAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~KruskalWallisAggregateState();
    }
}

static void KruskalWallisAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                    Vector &state_vector, idx_t count) {
    UnifiedVectorFormat value_data, group_data;
    inputs[0].ToUnifiedFormat(count, value_data);
    inputs[1].ToUnifiedFormat(count, group_data);
    auto values = UnifiedVectorFormat::GetData<double>(value_data);
    const bool group_is_string = inputs[1].GetType().id() == LogicalTypeId::VARCHAR;

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (KruskalWallisAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto val_idx = value_data.sel->get_index(i);
        auto grp_idx = group_data.sel->get_index(i);

        if (!value_data.validity.RowIsValid(val_idx) || !group_data.validity.RowIsValid(grp_idx)) {
            continue;
        }

        double val = values[val_idx];

        if (std::isnan(val)) {
            continue;
        }

        state.values.push_back(val);
        state.groups.push_back(state.coder.Code(ReadGroupLabel(group_data, grp_idx, group_is_string)));
    }
}

static void KruskalWallisAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (KruskalWallisAggregateState **)source_data.data;
    auto targets = (KruskalWallisAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.values = CombineTake(source.values, aggr_input_data);
            target.groups = CombineTake(source.groups, aggr_input_data);
            target.coder.labels = CombineTake(source.coder.labels, aggr_input_data);
            target.initialized = true;
            continue;
        }

        target.values.insert(target.values.end(), source.values.begin(), source.values.end());
        auto remap = target.coder.Remap(source.coder);
        for (auto code : source.groups) {
            target.groups.push_back(remap[static_cast<idx_t>(code)]);
        }
    }
}

static void KruskalWallisAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                      idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (KruskalWallisAggregateState **)sdata.data;

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

        AnofoxTestResult test_result;
        AnofoxError error;

        bool success = anofox_kruskal_wallis(values_array, groups_array, &test_result, &error);

        if (!success) {
            ThrowUnlessDegenerate("kruskal_wallis_agg", error);
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t struct_idx = 0;
        FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = test_result.statistic;
        FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = test_result.p_value;
        FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = test_result.df;
        FlatDataMut<int64_t>(StructEntry(struct_entries[struct_idx++]))[result_idx] = static_cast<int64_t>(test_result.n);
        auto& method_vector = StructEntry(struct_entries[struct_idx++]);
        FlatDataMut<string_t>(method_vector)[result_idx] =
            StringVector::AddString(method_vector, test_result.method ? test_result.method : "Kruskal-Wallis");
        SetResultNull(StructEntry(struct_entries[struct_idx++]), result_idx);
        WriteTestResultFields(result, result_idx, struct_idx, test_result.unified); // alternative: not applicable

        anofox_free_test_result(&test_result);
        state.Reset();
    }
}

static unique_ptr<FunctionData> KruskalWallisAggBind(ClientContext &context, AggregateBindFunction &function,
                                                      vector<unique_ptr<Expression>> &arguments) {
    SetReturnTypeOf(function, GetKruskalWallisAggResultType());
    PostHogTelemetry::Instance().RecordFunctionCall("kruskal_wallis_agg");
    return nullptr;
}

void RegisterKruskalWallisAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("kruskal_wallis_agg");

    auto func = AggregateFunction(
        "kruskal_wallis_agg", {LogicalType::DOUBLE, LogicalType::BIGINT},
        LogicalType::ANY,
        AggregateFunction::StateSize<KruskalWallisAggregateState>, ANOFOX_AGG_INIT(KruskalWallisAggInitialize),
        ANOFOX_GUARDED_UPDATE(KruskalWallisAggUpdate, KruskalWallisAggDestroy, KruskalWallisAggInitialize), KruskalWallisAggCombine, ANOFOX_GUARDED_FINALIZE(KruskalWallisAggFinalize, KruskalWallisAggDestroy, KruskalWallisAggInitialize),
        nullptr, ANOFOX_AGG_BIND(KruskalWallisAggBind), KruskalWallisAggDestroy);
    func_set.AddFunction(func);
    SetArgumentTypeOf(func, 1, LogicalType::VARCHAR); // any group label (#186)
    func_set.AddFunction(func);

    CreateAggregateFunctionInfo info(std::move(func_set));
    info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
    FunctionDescription d1;
    d1.description     = "Performs the Kruskal-Wallis H-test, a nonparametric alternative to one-way ANOVA.";
    d1.examples        = {"kruskal_wallis_agg(value, group_id)"};
    d1.categories      = {"hypothesis-testing", "nonparametric"};
    d1.parameter_names = {"value", "group_id"};
    d1.parameter_types = {LogicalType::DOUBLE, LogicalType::BIGINT};
    info.descriptions.push_back(std::move(d1));
    // The VARCHAR-label overload (#186): same description, VARCHAR group.
    FunctionDescription d2 = info.descriptions.back();
    d2.parameter_types = {LogicalType::DOUBLE, LogicalType::VARCHAR};
    info.descriptions.push_back(std::move(d2));
    loader.RegisterFunction(std::move(info));

}

} // namespace duckdb
