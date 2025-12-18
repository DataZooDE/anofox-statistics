#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Clark-West Test Aggregate State
//===--------------------------------------------------------------------===//
struct ClarkWestAggregateState {
    vector<double> actual;
    vector<double> forecast_restricted;
    vector<double> forecast_unrestricted;
    bool initialized;

    ClarkWestAggregateState() : initialized(false) {}

    void Reset() {
        actual.clear();
        forecast_restricted.clear();
        forecast_unrestricted.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetClarkWestAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("statistic", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("n", LogicalType::BIGINT));
    children.push_back(make_pair("method", LogicalType::VARCHAR));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Bind data for options
//===--------------------------------------------------------------------===//
struct ClarkWestBindData : public FunctionData {
    size_t horizon;

    ClarkWestBindData() : horizon(1) {}

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<ClarkWestBindData>();
        copy->horizon = horizon;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<ClarkWestBindData>();
        return horizon == other.horizon;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void ClarkWestAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) ClarkWestAggregateState();
}

static void ClarkWestAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ClarkWestAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~ClarkWestAggregateState();
    }
}

static void ClarkWestAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                Vector &state_vector, idx_t count) {
    UnifiedVectorFormat actual_data, fr_data, fu_data;
    inputs[0].ToUnifiedFormat(count, actual_data);
    inputs[1].ToUnifiedFormat(count, fr_data);
    inputs[2].ToUnifiedFormat(count, fu_data);
    auto actuals = UnifiedVectorFormat::GetData<double>(actual_data);
    auto frs = UnifiedVectorFormat::GetData<double>(fr_data);
    auto fus = UnifiedVectorFormat::GetData<double>(fu_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ClarkWestAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto act_idx = actual_data.sel->get_index(i);
        auto fr_idx = fr_data.sel->get_index(i);
        auto fu_idx = fu_data.sel->get_index(i);

        if (!actual_data.validity.RowIsValid(act_idx) ||
            !fr_data.validity.RowIsValid(fr_idx) ||
            !fu_data.validity.RowIsValid(fu_idx)) {
            continue;
        }

        double act = actuals[act_idx];
        double fr = frs[fr_idx];
        double fu = fus[fu_idx];

        if (std::isnan(act) || std::isnan(fr) || std::isnan(fu)) {
            continue;
        }

        state.actual.push_back(act);
        state.forecast_restricted.push_back(fr);
        state.forecast_unrestricted.push_back(fu);
    }
}

static void ClarkWestAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (ClarkWestAggregateState **)source_data.data;
    auto targets = (ClarkWestAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.actual = std::move(source.actual);
            target.forecast_restricted = std::move(source.forecast_restricted);
            target.forecast_unrestricted = std::move(source.forecast_unrestricted);
            target.initialized = true;
            continue;
        }

        target.actual.insert(target.actual.end(), source.actual.begin(), source.actual.end());
        target.forecast_restricted.insert(target.forecast_restricted.end(),
                                          source.forecast_restricted.begin(), source.forecast_restricted.end());
        target.forecast_unrestricted.insert(target.forecast_unrestricted.end(),
                                            source.forecast_unrestricted.begin(), source.forecast_unrestricted.end());
    }
}

static void ClarkWestAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                  idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ClarkWestAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);
    auto &bind_data = aggr_input_data.bind_data->Cast<ClarkWestBindData>();

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.actual.size() < 3) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        AnofoxDataArray actual_array;
        actual_array.data = state.actual.data();
        actual_array.validity = nullptr;
        actual_array.len = state.actual.size();

        AnofoxDataArray fr_array;
        fr_array.data = state.forecast_restricted.data();
        fr_array.validity = nullptr;
        fr_array.len = state.forecast_restricted.size();

        AnofoxDataArray fu_array;
        fu_array.data = state.forecast_unrestricted.data();
        fu_array.validity = nullptr;
        fu_array.len = state.forecast_unrestricted.size();

        AnofoxTestResult test_result;
        AnofoxError error;

        bool success = anofox_clark_west(actual_array, fr_array, fu_array,
                                          bind_data.horizon, &test_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.statistic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.p_value;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(test_result.n);
        FlatVector::GetData<string_t>(*struct_entries[struct_idx++])[result_idx] =
            StringVector::AddString(*struct_entries[struct_idx - 1], test_result.method ? test_result.method : "Clark-West test");

        anofox_free_test_result(&test_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> ClarkWestAggBind(ClientContext &context, AggregateFunction &function,
                                                   vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetClarkWestAggResultType();
    auto bind_data = make_uniq<ClarkWestBindData>();

    if (arguments.size() >= 4 && arguments[3]->IsFoldable()) {
        Value options_val = ExpressionExecutor::EvaluateScalar(context, *arguments[3]);
        if (options_val.type().id() == LogicalTypeId::MAP) {
            auto &map_children = MapValue::GetChildren(options_val);
            for (auto &entry : map_children) {
                auto &key_list = StructValue::GetChildren(entry);
                if (key_list.size() >= 2) {
                    auto key = StringValue::Get(key_list[0]).c_str();
                    if (strcasecmp(key, "horizon") == 0) {
                        bind_data->horizon = static_cast<size_t>(key_list[1].GetValue<int64_t>());
                    }
                }
            }
        }
    }

    return bind_data;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterClarkWestAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_clark_west_agg");

    // With options: (actual, forecast_restricted, forecast_unrestricted, options)
    auto func_with_opts = AggregateFunction(
        "anofox_stats_clark_west_agg",
        {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::ANY},
        LogicalType::ANY,
        AggregateFunction::StateSize<ClarkWestAggregateState>, ClarkWestAggInitialize,
        ClarkWestAggUpdate, ClarkWestAggCombine, ClarkWestAggFinalize,
        nullptr, ClarkWestAggBind, ClarkWestAggDestroy);
    func_set.AddFunction(func_with_opts);

    // Without options: (actual, forecast_restricted, forecast_unrestricted)
    auto func_no_opts = AggregateFunction(
        "anofox_stats_clark_west_agg",
        {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE},
        LogicalType::ANY,
        AggregateFunction::StateSize<ClarkWestAggregateState>, ClarkWestAggInitialize,
        ClarkWestAggUpdate, ClarkWestAggCombine, ClarkWestAggFinalize,
        nullptr, ClarkWestAggBind, ClarkWestAggDestroy);
    func_set.AddFunction(func_no_opts);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("clark_west_agg");
    alias_set.AddFunction(func_with_opts);
    alias_set.AddFunction(func_no_opts);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
