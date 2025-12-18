#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Two-Sample Proportion Test Aggregate State
//===--------------------------------------------------------------------===//
struct PropTestTwoAggregateState {
    size_t successes1;
    size_t trials1;
    size_t successes2;
    size_t trials2;
    bool initialized;

    PropTestTwoAggregateState() : successes1(0), trials1(0), successes2(0), trials2(0), initialized(false) {}

    void Reset() {
        successes1 = 0;
        trials1 = 0;
        successes2 = 0;
        trials2 = 0;
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetPropTestTwoAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("statistic", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("estimate", LogicalType::DOUBLE));
    children.push_back(make_pair("ci_lower", LogicalType::DOUBLE));
    children.push_back(make_pair("ci_upper", LogicalType::DOUBLE));
    children.push_back(make_pair("n", LogicalType::BIGINT));
    children.push_back(make_pair("method", LogicalType::VARCHAR));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Bind data for options
//===--------------------------------------------------------------------===//
struct PropTestTwoBindData : public FunctionData {
    AnofoxAlternative alternative;
    bool correction;

    PropTestTwoBindData() : alternative(ANOFOX_ALTERNATIVE_TWO_SIDED), correction(true) {}

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<PropTestTwoBindData>();
        copy->alternative = alternative;
        copy->correction = correction;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<PropTestTwoBindData>();
        return alternative == other.alternative && correction == other.correction;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void PropTestTwoAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) PropTestTwoAggregateState();
}

static void PropTestTwoAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (PropTestTwoAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~PropTestTwoAggregateState();
    }
}

static void PropTestTwoAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                  Vector &state_vector, idx_t count) {
    UnifiedVectorFormat val_data, group_data;
    inputs[0].ToUnifiedFormat(count, val_data);
    inputs[1].ToUnifiedFormat(count, group_data);
    auto vals = UnifiedVectorFormat::GetData<int64_t>(val_data);
    auto groups = UnifiedVectorFormat::GetData<int64_t>(group_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (PropTestTwoAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto val_idx = val_data.sel->get_index(i);
        auto group_idx = group_data.sel->get_index(i);

        if (!val_data.validity.RowIsValid(val_idx) || !group_data.validity.RowIsValid(group_idx)) {
            continue;
        }

        int64_t val = vals[val_idx];
        int64_t group = groups[group_idx];

        // Group 0 or 1 expected
        if (group == 0) {
            if (val != 0) {
                state.successes1++;
            }
            state.trials1++;
        } else {
            if (val != 0) {
                state.successes2++;
            }
            state.trials2++;
        }
    }
}

static void PropTestTwoAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (PropTestTwoAggregateState **)source_data.data;
    auto targets = (PropTestTwoAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        target.successes1 += source.successes1;
        target.trials1 += source.trials1;
        target.successes2 += source.successes2;
        target.trials2 += source.trials2;
        target.initialized = true;
    }
}

static void PropTestTwoAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                    idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (PropTestTwoAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);
    auto &bind_data = aggr_input_data.bind_data->Cast<PropTestTwoBindData>();

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.trials1 < 1 || state.trials2 < 1) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        AnofoxPropTestResult prop_result;
        AnofoxError error;

        bool success = anofox_prop_test_two(state.successes1, state.trials1,
                                             state.successes2, state.trials2,
                                             bind_data.alternative, bind_data.correction,
                                             &prop_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = prop_result.statistic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = prop_result.p_value;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = prop_result.estimate;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = prop_result.ci_lower;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = prop_result.ci_upper;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(prop_result.n);
        FlatVector::GetData<string_t>(*struct_entries[struct_idx++])[result_idx] =
            StringVector::AddString(*struct_entries[struct_idx - 1], prop_result.method ? prop_result.method : "Two-sample proportion test");

        anofox_free_prop_test_result(&prop_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> PropTestTwoAggBind(ClientContext &context, AggregateFunction &function,
                                                    vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetPropTestTwoAggResultType();
    auto bind_data = make_uniq<PropTestTwoBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        Value options_val = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
        if (options_val.type().id() == LogicalTypeId::MAP) {
            auto &map_children = MapValue::GetChildren(options_val);
            for (auto &entry : map_children) {
                auto &key_list = StructValue::GetChildren(entry);
                if (key_list.size() >= 2) {
                    auto key = StringValue::Get(key_list[0]).c_str();
                    if (strcasecmp(key, "alternative") == 0) {
                        auto alt_str = StringValue::Get(key_list[1]);
                        if (strcasecmp(alt_str.c_str(), "less") == 0) {
                            bind_data->alternative = ANOFOX_ALTERNATIVE_LESS;
                        } else if (strcasecmp(alt_str.c_str(), "greater") == 0) {
                            bind_data->alternative = ANOFOX_ALTERNATIVE_GREATER;
                        } else {
                            bind_data->alternative = ANOFOX_ALTERNATIVE_TWO_SIDED;
                        }
                    } else if (strcasecmp(key, "correction") == 0) {
                        bind_data->correction = key_list[1].GetValue<bool>();
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
void RegisterPropTestTwoAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_prop_test_two_agg");

    // With options: (value BIGINT, group_id BIGINT, options)
    auto func_with_opts = AggregateFunction(
        "anofox_stats_prop_test_two_agg", {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::ANY},
        LogicalType::ANY,
        AggregateFunction::StateSize<PropTestTwoAggregateState>, PropTestTwoAggInitialize,
        PropTestTwoAggUpdate, PropTestTwoAggCombine, PropTestTwoAggFinalize,
        nullptr, PropTestTwoAggBind, PropTestTwoAggDestroy);
    func_set.AddFunction(func_with_opts);

    // Without options: (value BIGINT, group_id BIGINT)
    auto func_no_opts = AggregateFunction(
        "anofox_stats_prop_test_two_agg", {LogicalType::BIGINT, LogicalType::BIGINT},
        LogicalType::ANY,
        AggregateFunction::StateSize<PropTestTwoAggregateState>, PropTestTwoAggInitialize,
        PropTestTwoAggUpdate, PropTestTwoAggCombine, PropTestTwoAggFinalize,
        nullptr, PropTestTwoAggBind, PropTestTwoAggDestroy);
    func_set.AddFunction(func_no_opts);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("prop_test_two_agg");
    alias_set.AddFunction(func_with_opts);
    alias_set.AddFunction(func_no_opts);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
