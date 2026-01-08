#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"

#ifdef _WIN32
#define strcasecmp _stricmp
#endif

namespace duckdb {

//===--------------------------------------------------------------------===//
// Yuen's Trimmed Mean Test Aggregate State
//===--------------------------------------------------------------------===//
struct YuenAggregateState {
    vector<double> group1;
    vector<double> group2;
    bool initialized;

    YuenAggregateState() : initialized(false) {}

    void Reset() {
        group1.clear();
        group2.clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetYuenAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("statistic", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("df", LogicalType::DOUBLE));
    children.push_back(make_pair("effect_size", LogicalType::DOUBLE));
    children.push_back(make_pair("ci_lower", LogicalType::DOUBLE));
    children.push_back(make_pair("ci_upper", LogicalType::DOUBLE));
    children.push_back(make_pair("n1", LogicalType::BIGINT));
    children.push_back(make_pair("n2", LogicalType::BIGINT));
    children.push_back(make_pair("method", LogicalType::VARCHAR));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Bind data for options
//===--------------------------------------------------------------------===//
struct YuenBindData : public FunctionData {
    double trim;
    AnofoxAlternative alternative;
    double confidence_level;

    YuenBindData() : trim(0.2), alternative(ANOFOX_ALTERNATIVE_TWO_SIDED), confidence_level(0.95) {}

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<YuenBindData>();
        copy->trim = trim;
        copy->alternative = alternative;
        copy->confidence_level = confidence_level;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<YuenBindData>();
        return trim == other.trim &&
               alternative == other.alternative &&
               confidence_level == other.confidence_level;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void YuenAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) YuenAggregateState();
}

static void YuenAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (YuenAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~YuenAggregateState();
    }
}

static void YuenAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                           Vector &state_vector, idx_t count) {
    UnifiedVectorFormat value_data, group_data;
    inputs[0].ToUnifiedFormat(count, value_data);
    inputs[1].ToUnifiedFormat(count, group_data);
    auto values = UnifiedVectorFormat::GetData<double>(value_data);
    auto groups = UnifiedVectorFormat::GetData<int32_t>(group_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (YuenAggregateState **)sdata.data;

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

static void YuenAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (YuenAggregateState **)source_data.data;
    auto targets = (YuenAggregateState **)target_data.data;

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

static void YuenAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                             idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (YuenAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);
    auto &bind_data = aggr_input_data.bind_data->Cast<YuenBindData>();

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

        bool success = anofox_yuen_test(group1_array, group2_array, bind_data.trim,
                                         bind_data.alternative, bind_data.confidence_level,
                                         &test_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.statistic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.p_value;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.df;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.effect_size;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.ci_lower;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.ci_upper;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(test_result.n1);
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(test_result.n2);
        auto& method_vector = *struct_entries[struct_idx++];
        FlatVector::GetData<string_t>(method_vector)[result_idx] =
            StringVector::AddString(method_vector, test_result.method ? test_result.method : "Yuen's trimmed mean test");

        anofox_free_test_result(&test_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> YuenAggBind(ClientContext &context, AggregateFunction &function,
                                              vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetYuenAggResultType();
    auto bind_data = make_uniq<YuenBindData>();

    // Parse options if provided (3rd argument)
    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        Value options_val = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
        if (options_val.type().id() == LogicalTypeId::MAP) {
            auto &map_children = MapValue::GetChildren(options_val);
            for (auto &entry : map_children) {
                auto &key_list = StructValue::GetChildren(entry);
                if (key_list.size() >= 2) {
                    auto key = StringValue::Get(key_list[0]).c_str();
                    if (strcasecmp(key, "trim") == 0) {
                        bind_data->trim = key_list[1].GetValue<double>();
                    } else if (strcasecmp(key, "alternative") == 0) {
                        auto alt_str = StringValue::Get(key_list[1]);
                        if (strcasecmp(alt_str.c_str(), "less") == 0) {
                            bind_data->alternative = ANOFOX_ALTERNATIVE_LESS;
                        } else if (strcasecmp(alt_str.c_str(), "greater") == 0) {
                            bind_data->alternative = ANOFOX_ALTERNATIVE_GREATER;
                        } else {
                            bind_data->alternative = ANOFOX_ALTERNATIVE_TWO_SIDED;
                        }
                    } else if (strcasecmp(key, "confidence_level") == 0) {
                        bind_data->confidence_level = key_list[1].GetValue<double>();
                    }
                }
            }
        }
    }

    PostHogTelemetry::Instance().CaptureFunctionExecution("yuen_agg");
    return bind_data;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterYuenAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_yuen_agg");

    // Version with options: yuen_agg(value, group_id, {'trim': 0.2})
    auto func_with_opts = AggregateFunction(
        "anofox_stats_yuen_agg", {LogicalType::DOUBLE, LogicalType::INTEGER, LogicalType::ANY},
        LogicalType::ANY,
        AggregateFunction::StateSize<YuenAggregateState>, YuenAggInitialize,
        YuenAggUpdate, YuenAggCombine, YuenAggFinalize,
        nullptr, YuenAggBind, YuenAggDestroy);
    func_set.AddFunction(func_with_opts);

    // Version without options: yuen_agg(value, group_id)
    auto func_no_opts = AggregateFunction(
        "anofox_stats_yuen_agg", {LogicalType::DOUBLE, LogicalType::INTEGER},
        LogicalType::ANY,
        AggregateFunction::StateSize<YuenAggregateState>, YuenAggInitialize,
        YuenAggUpdate, YuenAggCombine, YuenAggFinalize,
        nullptr, YuenAggBind, YuenAggDestroy);
    func_set.AddFunction(func_no_opts);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("yuen_agg");
    alias_set.AddFunction(func_with_opts);
    alias_set.AddFunction(func_no_opts);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
