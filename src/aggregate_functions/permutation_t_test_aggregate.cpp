#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Permutation T-Test Aggregate State
//===--------------------------------------------------------------------===//
struct PermutationTTestAggregateState {
    vector<double> group1;
    vector<double> group2;
    bool initialized;

    PermutationTTestAggregateState() : initialized(false) {}

    void Reset() {
        group1.clear();
        group2.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetPermutationTTestAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("statistic", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("n1", LogicalType::BIGINT));
    children.push_back(make_pair("n2", LogicalType::BIGINT));
    children.push_back(make_pair("method", LogicalType::VARCHAR));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Bind data for options
//===--------------------------------------------------------------------===//
struct PermutationTTestBindData : public FunctionData {
    AnofoxAlternative alternative;
    size_t n_permutations;

    PermutationTTestBindData() : alternative(ANOFOX_ALTERNATIVE_TWO_SIDED), n_permutations(10000) {}

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<PermutationTTestBindData>();
        copy->alternative = alternative;
        copy->n_permutations = n_permutations;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<PermutationTTestBindData>();
        return alternative == other.alternative && n_permutations == other.n_permutations;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void PermutationTTestAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) PermutationTTestAggregateState();
}

static void PermutationTTestAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (PermutationTTestAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~PermutationTTestAggregateState();
    }
}

static void PermutationTTestAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                       Vector &state_vector, idx_t count) {
    UnifiedVectorFormat value_data, group_data;
    inputs[0].ToUnifiedFormat(count, value_data);
    inputs[1].ToUnifiedFormat(count, group_data);
    auto values = UnifiedVectorFormat::GetData<double>(value_data);
    auto groups = UnifiedVectorFormat::GetData<int32_t>(group_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (PermutationTTestAggregateState **)sdata.data;

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

        if (group == 0) {
            state.group1.push_back(val);
        } else {
            state.group2.push_back(val);
        }
    }
}

static void PermutationTTestAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (PermutationTTestAggregateState **)source_data.data;
    auto targets = (PermutationTTestAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.group1 = std::move(source.group1);
            target.group2 = std::move(source.group2);
            target.initialized = true;
            continue;
        }

        target.group1.insert(target.group1.end(), source.group1.begin(), source.group1.end());
        target.group2.insert(target.group2.end(), source.group2.begin(), source.group2.end());
    }
}

static void PermutationTTestAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                         idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (PermutationTTestAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);
    auto &bind_data = aggr_input_data.bind_data->Cast<PermutationTTestBindData>();

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.group1.size() < 2 || state.group2.size() < 2) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        AnofoxDataArray group1_array;
        group1_array.data = state.group1.data();
        group1_array.validity = nullptr;
        group1_array.len = state.group1.size();

        AnofoxDataArray group2_array;
        group2_array.data = state.group2.data();
        group2_array.validity = nullptr;
        group2_array.len = state.group2.size();

        AnofoxTestResult test_result;
        AnofoxError error;

        bool success = anofox_permutation_t_test(group1_array, group2_array,
                                                  bind_data.alternative, bind_data.n_permutations,
                                                  0, false,  // No seed
                                                  &test_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.statistic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.p_value;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(test_result.n1);
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(test_result.n2);
        FlatVector::GetData<string_t>(*struct_entries[struct_idx++])[result_idx] =
            StringVector::AddString(*struct_entries[struct_idx - 1], test_result.method ? test_result.method : "Permutation t-test");

        anofox_free_test_result(&test_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> PermutationTTestAggBind(ClientContext &context, AggregateFunction &function,
                                                          vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetPermutationTTestAggResultType();
    auto bind_data = make_uniq<PermutationTTestBindData>();

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
                        }
                    } else if (strcasecmp(key, "n_permutations") == 0 || strcasecmp(key, "permutations") == 0) {
                        bind_data->n_permutations = static_cast<size_t>(key_list[1].GetValue<int64_t>());
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
void RegisterPermutationTTestAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_permutation_t_test_agg");

    // With options: (value, group_id, options)
    auto func_with_opts = AggregateFunction(
        "anofox_stats_permutation_t_test_agg", {LogicalType::DOUBLE, LogicalType::INTEGER, LogicalType::ANY},
        LogicalType::ANY,
        AggregateFunction::StateSize<PermutationTTestAggregateState>, PermutationTTestAggInitialize,
        PermutationTTestAggUpdate, PermutationTTestAggCombine, PermutationTTestAggFinalize,
        nullptr, PermutationTTestAggBind, PermutationTTestAggDestroy);
    func_set.AddFunction(func_with_opts);

    // Without options: (value, group_id)
    auto func_no_opts = AggregateFunction(
        "anofox_stats_permutation_t_test_agg", {LogicalType::DOUBLE, LogicalType::INTEGER},
        LogicalType::ANY,
        AggregateFunction::StateSize<PermutationTTestAggregateState>, PermutationTTestAggInitialize,
        PermutationTTestAggUpdate, PermutationTTestAggCombine, PermutationTTestAggFinalize,
        nullptr, PermutationTTestAggBind, PermutationTTestAggDestroy);
    func_set.AddFunction(func_no_opts);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("permutation_t_test_agg");
    alias_set.AddFunction(func_with_opts);
    alias_set.AddFunction(func_no_opts);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
