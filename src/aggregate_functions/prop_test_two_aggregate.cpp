#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/result_fields.hpp"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "../include/ffi_enum_converters.hpp"
#include "telemetry.hpp"
#include "aggregate_finalize_guard.hpp"


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
    children.push_back(make_pair("alternative", LogicalType::VARCHAR));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Bind data for options
//===--------------------------------------------------------------------===//
struct PropTestTwoBindData : public FunctionData {
    AnofoxAlternative alternative;
    bool correction;
    double confidence_level;

    PropTestTwoBindData() : alternative(ANOFOX_ALTERNATIVE_TWO_SIDED), correction(true), confidence_level(0.95) {}

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<PropTestTwoBindData>();
        copy->alternative = alternative;
        copy->correction = correction;
        copy->confidence_level = confidence_level;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<PropTestTwoBindData>();
        return alternative == other.alternative && correction == other.correction &&
               confidence_level == other.confidence_level;
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

        bool success = anofox_prop_test_two_with_conf_level(state.successes1, state.trials1, state.successes2,
                                                            state.trials2, bind_data.alternative,
                                                            bind_data.correction, bind_data.confidence_level,
                                                            &prop_result, &error);

        if (!success) {
            ThrowUnlessDegenerate("prop_test_two_agg", error);
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
        auto& method_vector = *struct_entries[struct_idx++];
        FlatVector::GetData<string_t>(method_vector)[result_idx] =
            StringVector::AddString(method_vector, prop_result.method ? prop_result.method : "Two-sample proportion test");
        SetResultString(*struct_entries[struct_idx++], result_idx, AlternativeName(bind_data.alternative));

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

    if (arguments.size() >= 3) {
        Value options_val = EvaluateConstantOptions(context, *arguments[2], "prop_test_two_agg");
        auto opts = PropTestTwoMapOptions::ParseFromValue(options_val, "prop_test_two_agg");
        if (opts.alternative.has_value()) {
            bind_data->alternative = ConvertAlternative(opts.alternative.value());
        }
        if (opts.correction.has_value()) {
            bind_data->correction = opts.correction.value();
        }
        if (opts.confidence_level.has_value()) {
            bind_data->confidence_level = opts.confidence_level.value();
        }
    }

    PostHogTelemetry::Instance().RecordFunctionCall("prop_test_two_agg");
    return bind_data;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterPropTestTwoAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("prop_test_two_agg");

    // With options: (value BIGINT, group_id BIGINT, options)
    auto func_with_opts = AggregateFunction(
        "prop_test_two_agg", {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::ANY},
        LogicalType::ANY,
        AggregateFunction::StateSize<PropTestTwoAggregateState>, PropTestTwoAggInitialize,
        PropTestTwoAggUpdate, PropTestTwoAggCombine, ANOFOX_GUARDED_FINALIZE(PropTestTwoAggFinalize, PropTestTwoAggDestroy, PropTestTwoAggInitialize),
        nullptr, PropTestTwoAggBind, PropTestTwoAggDestroy);
    func_set.AddFunction(func_with_opts);

    // Without options: (value BIGINT, group_id BIGINT)
    auto func_no_opts = AggregateFunction(
        "prop_test_two_agg", {LogicalType::BIGINT, LogicalType::BIGINT},
        LogicalType::ANY,
        AggregateFunction::StateSize<PropTestTwoAggregateState>, PropTestTwoAggInitialize,
        PropTestTwoAggUpdate, PropTestTwoAggCombine, ANOFOX_GUARDED_FINALIZE(PropTestTwoAggFinalize, PropTestTwoAggDestroy, PropTestTwoAggInitialize),
        nullptr, PropTestTwoAggBind, PropTestTwoAggDestroy);
    func_set.AddFunction(func_no_opts);

    CreateAggregateFunctionInfo info(std::move(func_set));
    info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
    FunctionDescription d1;
    d1.description     = "Tests whether two observed proportions are equal (two-sample proportion test).";
    d1.examples        = {"prop_test_two_agg(value, group_id, {'alternative': 'two_sided'})"};
    d1.categories      = {"hypothesis-testing", "proportion"};
    d1.parameter_names = {"value", "group_id", "options"};
    d1.parameter_types = {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::ANY};
    info.descriptions.push_back(std::move(d1));
    FunctionDescription d2;
    d2.description     = "Tests whether two observed proportions are equal (two-sample proportion test), using default options.";
    d2.examples        = {"prop_test_two_agg(value, group_id)"};
    d2.categories      = {"hypothesis-testing", "proportion"};
    d2.parameter_names = {"value", "group_id"};
    d2.parameter_types = {LogicalType::BIGINT, LogicalType::BIGINT};
    info.descriptions.push_back(std::move(d2));
    loader.RegisterFunction(std::move(info));

}

} // namespace duckdb
