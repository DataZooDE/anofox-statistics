#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/ffi_enum_converters.hpp"
#include "../include/result_fields.hpp"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "two_group.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Mann-Whitney U Aggregate State
//===--------------------------------------------------------------------===//
struct MannWhitneyAggregateState {
    TwoGroupSamples samples;
    bool initialized;

    MannWhitneyAggregateState() : initialized(false) {}

    void Reset() {
        samples.Clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetMannWhitneyAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("statistic", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("effect_size", LogicalType::DOUBLE));
    children.push_back(make_pair("ci_lower", LogicalType::DOUBLE));
    children.push_back(make_pair("ci_upper", LogicalType::DOUBLE));
    children.push_back(make_pair("n1", LogicalType::BIGINT));
    children.push_back(make_pair("n2", LogicalType::BIGINT));
    children.push_back(make_pair("method", LogicalType::VARCHAR));
    children.push_back(make_pair("n", LogicalType::BIGINT));
    children.push_back(make_pair("alternative", LogicalType::VARCHAR));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Bind data
//===--------------------------------------------------------------------===//
struct MannWhitneyBindData : public FunctionData {
    MannWhitneyMapOptions options;

    MannWhitneyBindData() {}

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<MannWhitneyBindData>();
        copy->options = options;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<MannWhitneyBindData>();
        return options.alternative == other.options.alternative &&
               options.continuity_correction == other.options.continuity_correction &&
               options.confidence_level == other.options.confidence_level && options.exact == other.options.exact &&
               options.mu == other.options.mu;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void MannWhitneyAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) MannWhitneyAggregateState();
}

static void MannWhitneyAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (MannWhitneyAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~MannWhitneyAggregateState();
    }
}

static void MannWhitneyAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                  Vector &state_vector, idx_t count) {
    UnifiedVectorFormat value_data, group_data;
    inputs[0].ToUnifiedFormat(count, value_data);
    inputs[1].ToUnifiedFormat(count, group_data);
    auto values = UnifiedVectorFormat::GetData<double>(value_data);
    const bool group_is_string = inputs[1].GetType().id() == LogicalTypeId::VARCHAR;

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (MannWhitneyAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto val_idx = value_data.sel->get_index(i);
        auto grp_idx = group_data.sel->get_index(i);

        if (!value_data.validity.RowIsValid(val_idx) || !group_data.validity.RowIsValid(grp_idx)) {
            continue;
        }

        double val = values[val_idx];
        auto group = ReadGroupLabel(group_data, grp_idx, group_is_string);

        if (std::isnan(val)) {
            continue;
        }

        state.samples.Add(group, val, "mann_whitney_u_agg");
    }
}

static void MannWhitneyAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (MannWhitneyAggregateState **)source_data.data;
    auto targets = (MannWhitneyAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.samples.Merge(source.samples, aggr_input_data, "mann_whitney_u_agg");
            target.initialized = true;
            continue;
        }

        target.samples.Merge(source.samples, aggr_input_data, "mann_whitney_u_agg");
    }
}

static void MannWhitneyAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                    idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (MannWhitneyAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);
    auto &bind_data = aggr_input_data.bind_data->Cast<MannWhitneyBindData>();

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.samples.Group1().size() < 1 || state.samples.Group2().size() < 1) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        AnofoxDataArray group1_array;
        group1_array.data = state.samples.Group1().data();
        group1_array.validity = nullptr;
        group1_array.len = state.samples.Group1().size();

        AnofoxDataArray group2_array;
        group2_array.data = state.samples.Group2().data();
        group2_array.validity = nullptr;
        group2_array.len = state.samples.Group2().size();

        AnofoxMannWhitneyOptions options;
        options.alternative = bind_data.options.alternative.value_or(Alternative::TWO_SIDED) == Alternative::TWO_SIDED
                                  ? ANOFOX_ALTERNATIVE_TWO_SIDED
                                  : (bind_data.options.alternative.value_or(Alternative::TWO_SIDED) == Alternative::LESS
                                         ? ANOFOX_ALTERNATIVE_LESS
                                         : ANOFOX_ALTERNATIVE_GREATER);
        options.exact = bind_data.options.exact.value_or(false);
        options.continuity_correction = bind_data.options.continuity_correction.value_or(true);
        options.confidence_level = bind_data.options.confidence_level.value_or(0.95);
        options.mu = bind_data.options.mu.value_or(0.0);

        AnofoxTestResult test_result;
        AnofoxError error;

        bool success = anofox_mann_whitney_u(group1_array, group2_array, options, &test_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.statistic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.p_value;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.effect_size;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.ci_lower;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = test_result.ci_upper;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(test_result.n1);
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(test_result.n2);
        auto& method_vector = *struct_entries[struct_idx++];
        FlatVector::GetData<string_t>(method_vector)[result_idx] =
            StringVector::AddString(method_vector, test_result.method ? test_result.method : "Mann-Whitney U");
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = static_cast<int64_t>(test_result.n1 + test_result.n2);
        SetResultString(*struct_entries[struct_idx++], result_idx, AlternativeName(options.alternative));

        anofox_free_test_result(&test_result);
        state.Reset();
    }
}

static unique_ptr<FunctionData> MannWhitneyAggBind(ClientContext &context, AggregateFunction &function,
                                                    vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetMannWhitneyAggResultType();
    auto bind_data = make_uniq<MannWhitneyBindData>();

    if (arguments.size() >= 3) {
        Value options_val = EvaluateConstantOptions(context, *arguments[2], "mann_whitney_u_agg");
        bind_data->options = MannWhitneyMapOptions::ParseFromValue(options_val, "mann_whitney_u_agg");
    }

    PostHogTelemetry::Instance().RecordFunctionCall("mann_whitney_u_agg");
    return bind_data;
}

void RegisterMannWhitneyAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("mann_whitney_u_agg");

    auto func_with_opts = AggregateFunction(
        "mann_whitney_u_agg", {LogicalType::DOUBLE, LogicalType::BIGINT, LogicalType::ANY},
        LogicalType::ANY,
        AggregateFunction::StateSize<MannWhitneyAggregateState>, MannWhitneyAggInitialize,
        MannWhitneyAggUpdate, MannWhitneyAggCombine, MannWhitneyAggFinalize,
        nullptr, MannWhitneyAggBind, MannWhitneyAggDestroy);
    func_set.AddFunction(func_with_opts);
    func_with_opts.arguments[1] = LogicalType::VARCHAR;
    func_set.AddFunction(func_with_opts);

    auto func_no_opts = AggregateFunction(
        "mann_whitney_u_agg", {LogicalType::DOUBLE, LogicalType::BIGINT},
        LogicalType::ANY,
        AggregateFunction::StateSize<MannWhitneyAggregateState>, MannWhitneyAggInitialize,
        MannWhitneyAggUpdate, MannWhitneyAggCombine, MannWhitneyAggFinalize,
        nullptr, MannWhitneyAggBind, MannWhitneyAggDestroy);
    func_set.AddFunction(func_no_opts);
    func_no_opts.arguments[1] = LogicalType::VARCHAR;
    func_set.AddFunction(func_no_opts);

    CreateAggregateFunctionInfo info(std::move(func_set));
    info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
    FunctionDescription d1;
    d1.description     = "Performs the Mann-Whitney U test (Wilcoxon rank-sum) for two independent samples.";
    d1.examples        = {"mann_whitney_u_agg(value, group_id, {'alternative': 'two_sided'})"};
    d1.categories      = {"hypothesis-testing", "nonparametric"};
    d1.parameter_names = {"value", "group_id", "options"};
    d1.parameter_types = {LogicalType::DOUBLE, LogicalType::BIGINT, LogicalType::ANY};
    info.descriptions.push_back(std::move(d1));
    FunctionDescription d2;
    d2.description     = "Performs the Mann-Whitney U test (Wilcoxon rank-sum) for two independent samples, using default options.";
    d2.examples        = {"mann_whitney_u_agg(value, group_id)"};
    d2.categories      = {"hypothesis-testing", "nonparametric"};
    d2.parameter_names = {"value", "group_id"};
    d2.parameter_types = {LogicalType::DOUBLE, LogicalType::BIGINT};
    info.descriptions.push_back(std::move(d2));
    loader.RegisterFunction(std::move(info));

}

} // namespace duckdb
