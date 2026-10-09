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


namespace duckdb {

//===--------------------------------------------------------------------===//
// One-Sample Proportion Test Aggregate State
//===--------------------------------------------------------------------===//
struct PropTestOneAggregateState {
    size_t successes;
    size_t trials;
    bool initialized;

    PropTestOneAggregateState() : successes(0), trials(0), initialized(false) {}

    void Reset() {
        successes = 0;
        trials = 0;
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetPropTestOneAggResultType() {
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
struct PropTestOneBindData : public FunctionData {
    double p0;
    AnofoxAlternative alternative;
    double confidence_level;

    PropTestOneBindData() : p0(0.5), alternative(ANOFOX_ALTERNATIVE_TWO_SIDED), confidence_level(0.95) {}

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<PropTestOneBindData>();
        copy->p0 = p0;
        copy->alternative = alternative;
        copy->confidence_level = confidence_level;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<PropTestOneBindData>();
        return p0 == other.p0 && alternative == other.alternative && confidence_level == other.confidence_level;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void PropTestOneAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) PropTestOneAggregateState();
}

static void PropTestOneAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (PropTestOneAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~PropTestOneAggregateState();
    }
}

static void PropTestOneAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                  Vector &state_vector, idx_t count) {
    UnifiedVectorFormat val_data;
    inputs[0].ToUnifiedFormat(count, val_data);
    auto vals = UnifiedVectorFormat::GetData<int64_t>(val_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (PropTestOneAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto val_idx = val_data.sel->get_index(i);

        if (!val_data.validity.RowIsValid(val_idx)) {
            continue;
        }

        int64_t val = vals[val_idx];

        // Count successes (non-zero values treated as success)
        if (val != 0) {
            state.successes++;
        }
        state.trials++;
    }
}

static void PropTestOneAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (PropTestOneAggregateState **)source_data.data;
    auto targets = (PropTestOneAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        target.successes += source.successes;
        target.trials += source.trials;
        target.initialized = true;
    }
}

static void PropTestOneAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                    idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (PropTestOneAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);
    auto &bind_data = aggr_input_data.bind_data->Cast<PropTestOneBindData>();

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.trials < 1) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        AnofoxPropTestResult prop_result;
        AnofoxError error;

        bool success = anofox_prop_test_one_with_conf_level(state.successes, state.trials, bind_data.p0,
                                                            bind_data.alternative, bind_data.confidence_level,
                                                            &prop_result, &error);

        if (!success) {
            ThrowUnlessDegenerate("prop_test_one_agg", error);
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
            StringVector::AddString(method_vector, prop_result.method ? prop_result.method : "One-sample proportion test");
        SetResultString(*struct_entries[struct_idx++], result_idx, AlternativeName(bind_data.alternative));

        anofox_free_prop_test_result(&prop_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> PropTestOneAggBind(ClientContext &context, AggregateFunction &function,
                                                    vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetPropTestOneAggResultType();
    auto bind_data = make_uniq<PropTestOneBindData>();

    if (arguments.size() >= 2) {
        Value options_val = EvaluateConstantOptions(context, *arguments[1], "prop_test_one_agg");
        auto opts = ProportionMapOptions::ParseFromValue(options_val, "prop_test_one_agg");
        if (opts.p0.has_value()) {
            bind_data->p0 = opts.p0.value();
        }
        if (opts.alternative.has_value()) {
            bind_data->alternative = ConvertAlternative(opts.alternative.value());
        }
        if (opts.confidence_level.has_value()) {
            bind_data->confidence_level = opts.confidence_level.value();
        }
    }

    PostHogTelemetry::Instance().RecordFunctionCall("prop_test_one_agg");
    return bind_data;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterPropTestOneAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("prop_test_one_agg");

    // With options: (value BIGINT, options)
    auto func_with_opts = AggregateFunction(
        "prop_test_one_agg", {LogicalType::BIGINT, LogicalType::ANY},
        LogicalType::ANY,
        AggregateFunction::StateSize<PropTestOneAggregateState>, PropTestOneAggInitialize,
        PropTestOneAggUpdate, PropTestOneAggCombine, PropTestOneAggFinalize,
        nullptr, PropTestOneAggBind, PropTestOneAggDestroy);
    func_set.AddFunction(func_with_opts);

    // Without options: (value BIGINT)
    auto func_no_opts = AggregateFunction(
        "prop_test_one_agg", {LogicalType::BIGINT},
        LogicalType::ANY,
        AggregateFunction::StateSize<PropTestOneAggregateState>, PropTestOneAggInitialize,
        PropTestOneAggUpdate, PropTestOneAggCombine, PropTestOneAggFinalize,
        nullptr, PropTestOneAggBind, PropTestOneAggDestroy);
    func_set.AddFunction(func_no_opts);

    CreateAggregateFunctionInfo info(std::move(func_set));
    info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
    FunctionDescription d1;
    d1.description     = "Tests whether an observed proportion differs from a hypothesized value (one-sample proportion test).";
    d1.examples        = {"prop_test_one_agg(value, {'p0': 0.5, 'alternative': 'two_sided'})"};
    d1.categories      = {"hypothesis-testing", "proportion"};
    d1.parameter_names = {"value", "options"};
    d1.parameter_types = {LogicalType::BIGINT, LogicalType::ANY};
    info.descriptions.push_back(std::move(d1));
    FunctionDescription d2;
    d2.description     = "Tests whether an observed proportion differs from a hypothesized value (one-sample proportion test), using default options.";
    d2.examples        = {"prop_test_one_agg(value)"};
    d2.categories      = {"hypothesis-testing", "proportion"};
    d2.parameter_names = {"value"};
    d2.parameter_types = {LogicalType::BIGINT};
    info.descriptions.push_back(std::move(d2));
    loader.RegisterFunction(std::move(info));

}

} // namespace duckdb
