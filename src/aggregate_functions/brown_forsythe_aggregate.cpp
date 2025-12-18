#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Brown-Forsythe Test Aggregate State
//===--------------------------------------------------------------------===//
struct BrownForsytheAggregateState {
    vector<double> values;
    vector<double> groups;
    bool initialized;

    BrownForsytheAggregateState() : initialized(false) {}

    void Reset() {
        values.clear();
        groups.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetBrownForsytheAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("statistic", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("df", LogicalType::DOUBLE));
    children.push_back(make_pair("n", LogicalType::BIGINT));
    children.push_back(make_pair("method", LogicalType::VARCHAR));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Bind data for options
//===--------------------------------------------------------------------===//
struct BrownForsytheBindData : public FunctionData {
    BrownForsytheBindData() {}

    unique_ptr<FunctionData> Copy() const override {
        return make_uniq<BrownForsytheBindData>();
    }

    bool Equals(const FunctionData &other_p) const override {
        return true;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void BrownForsytheAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) BrownForsytheAggregateState();
}

static void BrownForsytheAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (BrownForsytheAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~BrownForsytheAggregateState();
    }
}

static void BrownForsytheAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                    Vector &state_vector, idx_t count) {
    UnifiedVectorFormat value_data, group_data;
    inputs[0].ToUnifiedFormat(count, value_data);
    inputs[1].ToUnifiedFormat(count, group_data);
    auto values = UnifiedVectorFormat::GetData<double>(value_data);
    auto groups = UnifiedVectorFormat::GetData<int32_t>(group_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (BrownForsytheAggregateState **)sdata.data;

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

static void BrownForsytheAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (BrownForsytheAggregateState **)source_data.data;
    auto targets = (BrownForsytheAggregateState **)target_data.data;

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

static void BrownForsytheAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                      idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (BrownForsytheAggregateState **)sdata.data;

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

        bool success = anofox_brown_forsythe(values_array, groups_array, &test_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.statistic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.p_value;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.df;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(test_result.n);
        FlatVector::GetData<string_t>(*struct_entries[struct_idx++])[result_idx] =
            StringVector::AddString(*struct_entries[struct_idx - 1], test_result.method ? test_result.method : "Brown-Forsythe test");

        anofox_free_test_result(&test_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> BrownForsytheAggBind(ClientContext &context, AggregateFunction &function,
                                                       vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetBrownForsytheAggResultType();
    return make_uniq<BrownForsytheBindData>();
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterBrownForsytheAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_brown_forsythe_agg");

    // brown_forsythe_agg(value, group_id)
    auto func = AggregateFunction(
        "anofox_stats_brown_forsythe_agg", {LogicalType::DOUBLE, LogicalType::INTEGER},
        LogicalType::ANY,
        AggregateFunction::StateSize<BrownForsytheAggregateState>, BrownForsytheAggInitialize,
        BrownForsytheAggUpdate, BrownForsytheAggCombine, BrownForsytheAggFinalize,
        nullptr, BrownForsytheAggBind, BrownForsytheAggDestroy);
    func_set.AddFunction(func);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("brown_forsythe_agg");
    alias_set.AddFunction(func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
