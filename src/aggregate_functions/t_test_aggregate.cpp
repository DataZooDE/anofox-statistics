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
// T-Test Aggregate State
//===--------------------------------------------------------------------===//
struct TTestAggregateState {
    TwoGroupSamples samples;
    bool initialized;

    TTestAggregateState() : initialized(false) {}

    void Reset() {
        samples.Clear();
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetTTestAggResultType() {
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
struct TTestBindData : public FunctionData {
    TTestMapOptions options;

    TTestBindData() {}

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<TTestBindData>();
        copy->options = options;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<TTestBindData>();
        return options.alternative == other.options.alternative &&
               options.confidence_level == other.options.confidence_level &&
               options.kind == other.options.kind && options.mu == other.options.mu;
    }
};

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void TTestAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) TTestAggregateState();
}

static void TTestAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (TTestAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~TTestAggregateState();
    }
}

static void TTestAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                           Vector &state_vector, idx_t count) {
    UnifiedVectorFormat value_data, group_data;
    inputs[0].ToUnifiedFormat(count, value_data);
    inputs[1].ToUnifiedFormat(count, group_data);
    auto values = UnifiedVectorFormat::GetData<double>(value_data);
    const bool group_is_string = inputs[1].GetType().id() == LogicalTypeId::VARCHAR;

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (TTestAggregateState **)sdata.data;

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

        state.samples.Add(group, val, "t_test_agg");
    }
}

static void TTestAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (TTestAggregateState **)source_data.data;
    auto targets = (TTestAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.samples.Merge(source.samples, aggr_input_data, "t_test_agg");
            target.initialized = true;
            continue;
        }

        target.samples.Merge(source.samples, aggr_input_data, "t_test_agg");
    }
}

static void TTestAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                             idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (TTestAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);

    // Get options from bind data
    auto &bind_data = aggr_input_data.bind_data->Cast<TTestBindData>();

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.samples.Group1().size() < 2 || state.samples.Group2().size() < 2) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Prepare FFI data
        AnofoxDataArray group1_array;
        group1_array.data = state.samples.Group1().data();
        group1_array.validity = nullptr;
        group1_array.len = state.samples.Group1().size();

        AnofoxDataArray group2_array;
        group2_array.data = state.samples.Group2().data();
        group2_array.validity = nullptr;
        group2_array.len = state.samples.Group2().size();

        // Set options
        AnofoxTTestOptions options;
        options.alternative = bind_data.options.alternative.value_or(Alternative::TWO_SIDED) == Alternative::TWO_SIDED
                                  ? ANOFOX_ALTERNATIVE_TWO_SIDED
                                  : (bind_data.options.alternative.value_or(Alternative::TWO_SIDED) == Alternative::LESS
                                         ? ANOFOX_ALTERNATIVE_LESS
                                         : ANOFOX_ALTERNATIVE_GREATER);
        options.confidence_level = bind_data.options.confidence_level.value_or(0.95);
        options.var_equal = bind_data.options.kind.value_or(TTestKind::WELCH) == TTestKind::STUDENT;
        options.mu = bind_data.options.mu.value_or(0.0);

        AnofoxTestResult test_result;
        AnofoxError error;

        bool success = anofox_t_test(group1_array, group2_array, options, &test_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Fill STRUCT result
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
            StringVector::AddString(method_vector, test_result.method ? test_result.method : "t-test");

        anofox_free_test_result(&test_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> TTestAggBind(ClientContext &context, AggregateFunction &function,
                                              vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetTTestAggResultType();
    auto bind_data = make_uniq<TTestBindData>();

    // Parse options if provided (3rd argument)
    if (arguments.size() >= 3) {
        Value options_val = EvaluateConstantOptions(context, *arguments[2], "t_test_agg");
        bind_data->options = TTestMapOptions::ParseFromValue(options_val, "t_test_agg");
        // The (value, group) layout carries two independent samples; there is no
        // pairing information, so a paired test cannot be computed here. The key
        // used to be accepted and silently ignored (running a two-sample test).
        if (bind_data->options.paired.value_or(false)) {
            throw InvalidInputException(
                "t_test_agg: 'paired': true is not supported -- t_test_agg(value, group) compares two "
                "independent samples and has no pairing information. For paired data use "
                "tost_paired_agg(x, y, ...) (equivalence) or wilcoxon_signed_rank_agg(x, y, ...) (difference).");
        }
    }

    PostHogTelemetry::Instance().RecordFunctionCall("t_test_agg");
    return bind_data;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterTTestAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("t_test_agg");

    // Version with options: t_test_agg(value, group_id, {'alternative': 'two_sided'})
    auto func_with_opts = AggregateFunction(
        "t_test_agg", {LogicalType::DOUBLE, LogicalType::BIGINT, LogicalType::ANY},
        LogicalType::ANY,
        AggregateFunction::StateSize<TTestAggregateState>, TTestAggInitialize,
        TTestAggUpdate, TTestAggCombine, TTestAggFinalize,
        nullptr, TTestAggBind, TTestAggDestroy);
    func_set.AddFunction(func_with_opts);
    func_with_opts.arguments[1] = LogicalType::VARCHAR;
    func_set.AddFunction(func_with_opts);

    // Version without options: t_test_agg(value, group_id)
    auto func_no_opts = AggregateFunction(
        "t_test_agg", {LogicalType::DOUBLE, LogicalType::BIGINT},
        LogicalType::ANY,
        AggregateFunction::StateSize<TTestAggregateState>, TTestAggInitialize,
        TTestAggUpdate, TTestAggCombine, TTestAggFinalize,
        nullptr, TTestAggBind, TTestAggDestroy);
    func_set.AddFunction(func_no_opts);
    func_no_opts.arguments[1] = LogicalType::VARCHAR;
    func_set.AddFunction(func_no_opts);

    CreateAggregateFunctionInfo info(std::move(func_set));
    info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
    FunctionDescription d1;
    d1.description     = "Performs a two-sample t-test (Welch or Student) comparing values between two groups.";
    d1.examples        = {"t_test_agg(value, group_id, {'alternative': 'two_sided'})"};
    d1.categories      = {"hypothesis-testing"};
    d1.parameter_names = {"value", "group_id", "options"};
    d1.parameter_types = {LogicalType::DOUBLE, LogicalType::BIGINT, LogicalType::ANY};
    info.descriptions.push_back(std::move(d1));
    FunctionDescription d2;
    d2.description     = "Performs a two-sample t-test (Welch or Student) comparing values between two groups, using default options.";
    d2.examples        = {"t_test_agg(value, group_id)"};
    d2.categories      = {"hypothesis-testing"};
    d2.parameter_names = {"value", "group_id"};
    d2.parameter_types = {LogicalType::DOUBLE, LogicalType::BIGINT};
    info.descriptions.push_back(std::move(d2));
    loader.RegisterFunction(std::move(info));

}

} // namespace duckdb
