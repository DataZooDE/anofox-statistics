// t_test_one_sample_agg(x [, options]) and t_test_paired_agg(x, y [, options]) (#299).
//
// Both marshal to the upstream t-test (anofox-statistics `t_test` with the
// paired kind; Cohen's d from `cohens_d_one_sample` / `cohens_d`). The result
// STRUCT mirrors t_test_agg without its two-group counts n1 / n2.

#include <algorithm>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/test_struct.hpp"
#include "../include/anofox_stats_ffi.h"
#include "../include/ffi_enum_converters.hpp"
#include "../include/result_fields.hpp"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"
#include "../include/canonical_order.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Shared result type, options and result writer
//===--------------------------------------------------------------------===//
static LogicalType GetOneSamplePairedTTestResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("statistic", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("df", LogicalType::DOUBLE));
    children.push_back(make_pair("effect_size", LogicalType::DOUBLE));
    children.push_back(make_pair("ci_lower", LogicalType::DOUBLE));
    children.push_back(make_pair("ci_upper", LogicalType::DOUBLE));
    children.push_back(make_pair("n", LogicalType::BIGINT));
    children.push_back(make_pair("method", LogicalType::VARCHAR));
    children.push_back(make_pair("alternative", LogicalType::VARCHAR));

    AppendTestResultFields(children);
    return LogicalType::STRUCT(std::move(children));
}

struct OneSamplePairedTTestBindData : public FunctionData {
    TTestMapOptions options;

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<OneSamplePairedTTestBindData>();
        copy->options = options;
        return copy;
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<OneSamplePairedTTestBindData>();
        return options.alternative == other.options.alternative &&
               options.confidence_level == other.options.confidence_level && options.mu == other.options.mu;
    }
};

static unique_ptr<FunctionData> BindOneSamplePairedTTest(ClientContext &context, AggregateBindFunction &function,
                                                         vector<unique_ptr<Expression>> &arguments,
                                                         idx_t options_index, const char *fn_name) {
    SetReturnTypeOf(function, GetOneSamplePairedTTestResultType());
    auto bind_data = make_uniq<OneSamplePairedTTestBindData>();

    if (arguments.size() > options_index) {
        Value options_val = EvaluateConstantOptions(context, *arguments[options_index], fn_name);
        bind_data->options = TTestMapOptions::ParseFromValue(options_val, fn_name);
        // 'kind' (welch / student) and 'paired' describe two independent samples
        // and do not apply here; reject them rather than silently ignoring them.
        if (bind_data->options.kind.has_value()) {
            throw InvalidInputException("%s: option 'kind' does not apply (valid options: alternative, "
                                        "confidence_level, mu)",
                                        fn_name);
        }
        if (bind_data->options.paired.has_value()) {
            throw InvalidInputException("%s: option 'paired' does not apply (valid options: alternative, "
                                        "confidence_level, mu)",
                                        fn_name);
        }
    }

    PostHogTelemetry::Instance().RecordFunctionCall(fn_name);
    return bind_data;
}

static AnofoxTTestOptions ToFfiOptions(const TTestMapOptions &opts) {
    AnofoxTTestOptions options;
    switch (opts.alternative.value_or(Alternative::TWO_SIDED)) {
    case Alternative::LESS:
        options.alternative = ANOFOX_ALTERNATIVE_LESS;
        break;
    case Alternative::GREATER:
        options.alternative = ANOFOX_ALTERNATIVE_GREATER;
        break;
    default:
        options.alternative = ANOFOX_ALTERNATIVE_TWO_SIDED;
    }
    options.confidence_level = opts.confidence_level.value_or(0.95);
    options.var_equal = false; // ignored by the one-sample / paired tests
    options.mu = opts.mu.value_or(0.0);
    return options;
}

static void WriteOneSamplePairedTTestResult(Vector &result, idx_t result_idx, const AnofoxTestResult &test_result,
                                            const AnofoxTTestOptions &options, const char *default_method) {
    auto &struct_entries = StructVector::GetEntries(result);
    idx_t struct_idx = 0;
    FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = test_result.statistic;
    FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = test_result.p_value;
    FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = test_result.df;
    FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = test_result.effect_size;
    FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = test_result.ci_lower;
    FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = test_result.ci_upper;
    FlatDataMut<int64_t>(StructEntry(struct_entries[struct_idx++]))[result_idx] =
        static_cast<int64_t>(test_result.n);
    SetResultString(StructEntry(struct_entries[struct_idx++]), result_idx,
                    test_result.method ? test_result.method : default_method);
    SetResultString(StructEntry(struct_entries[struct_idx++]), result_idx, AlternativeName(options.alternative));
    WriteTestResultFields(result, result_idx, struct_idx, test_result.unified);
}

//===--------------------------------------------------------------------===//
// t_test_one_sample_agg(x [, options])
//===--------------------------------------------------------------------===//
struct TTestOneSampleAggregateState {
    vector<double> values;
    bool initialized;

    TTestOneSampleAggregateState() : initialized(false) {}

    void Reset() {
        values.clear();
        initialized = false;
    }
};

static void TTestOneSampleInitialize(data_ptr_t state_p) {
    new (state_p) TTestOneSampleAggregateState();
}

static void TTestOneSampleDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (TTestOneSampleAggregateState **)sdata.data;
    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~TTestOneSampleAggregateState();
    }
}

static void TTestOneSampleUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
    UnifiedVectorFormat x_data;
    inputs[0].ToUnifiedFormat(count, x_data);
    auto x_vals = UnifiedVectorFormat::GetData<double>(x_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (TTestOneSampleAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto x_idx = x_data.sel->get_index(i);
        if (!x_data.validity.RowIsValid(x_idx)) {
            continue;
        }
        double x_val = x_vals[x_idx];
        if (std::isnan(x_val)) {
            continue;
        }
        state.values.push_back(x_val);
    }
}

static void TTestOneSampleCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                                  idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (TTestOneSampleAggregateState **)source_data.data;
    auto targets = (TTestOneSampleAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }
        if (!target.initialized) {
            target.values = CombineTake(source.values, aggr_input_data);
            target.initialized = true;
            continue;
        }
        target.values.insert(target.values.end(), source.values.begin(), source.values.end());
    }
}

static void TTestOneSampleFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                   idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (TTestOneSampleAggregateState **)sdata.data;
    auto &bind_data = aggr_input_data.bind_data->Cast<OneSamplePairedTTestBindData>();

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.values.size() < 2) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Bitwise-reproducible result regardless of the row order (#214).
        std::sort(state.values.begin(), state.values.end());

        AnofoxDataArray x_array;
        x_array.data = state.values.data();
        x_array.validity = nullptr;
        x_array.len = state.values.size();

        auto options = ToFfiOptions(bind_data.options);
        AnofoxTestResult test_result;
        AnofoxError error;

        if (!anofox_t_test_one_sample(x_array, options, &test_result, &error)) {
            ThrowUnlessDegenerate("t_test_one_sample_agg", error);
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        WriteOneSamplePairedTTestResult(result, result_idx, test_result, options, "One Sample t-test");
        anofox_free_test_result(&test_result);
        state.Reset();
    }
}

static unique_ptr<FunctionData> TTestOneSampleBind(ClientContext &context, AggregateBindFunction &function,
                                                   vector<unique_ptr<Expression>> &arguments) {
    return BindOneSamplePairedTTest(context, function, arguments, 1, "t_test_one_sample_agg");
}

//===--------------------------------------------------------------------===//
// t_test_paired_agg(x, y [, options])
//===--------------------------------------------------------------------===//
struct TTestPairedAggregateState {
    vector<double> x_values;
    vector<double> y_values;
    bool initialized;

    TTestPairedAggregateState() : initialized(false) {}

    void Reset() {
        x_values.clear();
        y_values.clear();
        initialized = false;
    }
};

static void TTestPairedInitialize(data_ptr_t state_p) {
    new (state_p) TTestPairedAggregateState();
}

static void TTestPairedDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (TTestPairedAggregateState **)sdata.data;
    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~TTestPairedAggregateState();
    }
}

static void TTestPairedUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
    UnifiedVectorFormat x_data, y_data;
    inputs[0].ToUnifiedFormat(count, x_data);
    inputs[1].ToUnifiedFormat(count, y_data);
    auto x_vals = UnifiedVectorFormat::GetData<double>(x_data);
    auto y_vals = UnifiedVectorFormat::GetData<double>(y_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (TTestPairedAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto x_idx = x_data.sel->get_index(i);
        auto y_idx = y_data.sel->get_index(i);
        if (!x_data.validity.RowIsValid(x_idx) || !y_data.validity.RowIsValid(y_idx)) {
            continue;
        }
        double x_val = x_vals[x_idx];
        double y_val = y_vals[y_idx];
        if (std::isnan(x_val) || std::isnan(y_val)) {
            continue;
        }
        state.x_values.push_back(x_val);
        state.y_values.push_back(y_val);
    }
}

static void TTestPairedCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                               idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (TTestPairedAggregateState **)source_data.data;
    auto targets = (TTestPairedAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }
        if (!target.initialized) {
            target.x_values = CombineTake(source.x_values, aggr_input_data);
            target.y_values = CombineTake(source.y_values, aggr_input_data);
            target.initialized = true;
            continue;
        }
        target.x_values.insert(target.x_values.end(), source.x_values.begin(), source.x_values.end());
        target.y_values.insert(target.y_values.end(), source.y_values.begin(), source.y_values.end());
    }
}

static void TTestPairedFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (TTestPairedAggregateState **)sdata.data;
    auto &bind_data = aggr_input_data.bind_data->Cast<OneSamplePairedTTestBindData>();

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.x_values.size() < 2) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Canonical row order: bitwise-identical results whatever the thread count (#226).
        CanonicalRows().Key(state.x_values).Key(state.y_values).Sort();

        AnofoxDataArray x_array;
        x_array.data = state.x_values.data();
        x_array.validity = nullptr;
        x_array.len = state.x_values.size();

        AnofoxDataArray y_array;
        y_array.data = state.y_values.data();
        y_array.validity = nullptr;
        y_array.len = state.y_values.size();

        auto options = ToFfiOptions(bind_data.options);
        AnofoxTestResult test_result;
        AnofoxError error;

        if (!anofox_t_test_paired(x_array, y_array, options, &test_result, &error)) {
            ThrowUnlessDegenerate("t_test_paired_agg", error);
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        WriteOneSamplePairedTTestResult(result, result_idx, test_result, options, "Paired t-test");
        anofox_free_test_result(&test_result);
        state.Reset();
    }
}

static unique_ptr<FunctionData> TTestPairedBind(ClientContext &context, AggregateBindFunction &function,
                                                vector<unique_ptr<Expression>> &arguments) {
    return BindOneSamplePairedTTest(context, function, arguments, 2, "t_test_paired_agg");
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
static void RegisterTTestOneSample(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("t_test_one_sample_agg");

    func_set.AddFunction(AggregateFunction(
        "t_test_one_sample_agg", {LogicalType::DOUBLE, LogicalType::ANY}, LogicalType::ANY,
        AggregateFunction::StateSize<TTestOneSampleAggregateState>, ANOFOX_AGG_INIT(TTestOneSampleInitialize),
        ANOFOX_GUARDED_UPDATE(TTestOneSampleUpdate, TTestOneSampleDestroy, TTestOneSampleInitialize),
        TTestOneSampleCombine,
        ANOFOX_GUARDED_FINALIZE(TTestOneSampleFinalize, TTestOneSampleDestroy, TTestOneSampleInitialize), nullptr,
        ANOFOX_AGG_BIND(TTestOneSampleBind), TTestOneSampleDestroy));
    func_set.AddFunction(AggregateFunction(
        "t_test_one_sample_agg", {LogicalType::DOUBLE}, LogicalType::ANY,
        AggregateFunction::StateSize<TTestOneSampleAggregateState>, ANOFOX_AGG_INIT(TTestOneSampleInitialize),
        ANOFOX_GUARDED_UPDATE(TTestOneSampleUpdate, TTestOneSampleDestroy, TTestOneSampleInitialize),
        TTestOneSampleCombine,
        ANOFOX_GUARDED_FINALIZE(TTestOneSampleFinalize, TTestOneSampleDestroy, TTestOneSampleInitialize), nullptr,
        ANOFOX_AGG_BIND(TTestOneSampleBind), TTestOneSampleDestroy));

    CreateAggregateFunctionInfo info(std::move(func_set));
    info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
    FunctionDescription d1;
    d1.description = "Performs a one-sample t-test of H0: mean(x) = mu, with Cohen's d.";
    d1.examples = {"t_test_one_sample_agg(x, {'mu': 5})"};
    d1.categories = {"hypothesis-testing"};
    d1.parameter_names = {"x", "options"};
    d1.parameter_types = {LogicalType::DOUBLE, LogicalType::ANY};
    info.descriptions.push_back(std::move(d1));
    FunctionDescription d2;
    d2.description = "Performs a one-sample t-test of H0: mean(x) = 0, with Cohen's d, using default options.";
    d2.examples = {"t_test_one_sample_agg(x)"};
    d2.categories = {"hypothesis-testing"};
    d2.parameter_names = {"x"};
    d2.parameter_types = {LogicalType::DOUBLE};
    info.descriptions.push_back(std::move(d2));
    loader.RegisterFunction(std::move(info));
}

static void RegisterTTestPaired(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("t_test_paired_agg");

    func_set.AddFunction(AggregateFunction(
        "t_test_paired_agg", {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::ANY}, LogicalType::ANY,
        AggregateFunction::StateSize<TTestPairedAggregateState>, ANOFOX_AGG_INIT(TTestPairedInitialize),
        ANOFOX_GUARDED_UPDATE(TTestPairedUpdate, TTestPairedDestroy, TTestPairedInitialize), TTestPairedCombine,
        ANOFOX_GUARDED_FINALIZE(TTestPairedFinalize, TTestPairedDestroy, TTestPairedInitialize), nullptr,
        ANOFOX_AGG_BIND(TTestPairedBind), TTestPairedDestroy));
    func_set.AddFunction(AggregateFunction(
        "t_test_paired_agg", {LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::ANY,
        AggregateFunction::StateSize<TTestPairedAggregateState>, ANOFOX_AGG_INIT(TTestPairedInitialize),
        ANOFOX_GUARDED_UPDATE(TTestPairedUpdate, TTestPairedDestroy, TTestPairedInitialize), TTestPairedCombine,
        ANOFOX_GUARDED_FINALIZE(TTestPairedFinalize, TTestPairedDestroy, TTestPairedInitialize), nullptr,
        ANOFOX_AGG_BIND(TTestPairedBind), TTestPairedDestroy));

    CreateAggregateFunctionInfo info(std::move(func_set));
    info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
    FunctionDescription d1;
    d1.description = "Performs a paired t-test of H0: mean(x - y) = mu, with Cohen's d_z.";
    d1.examples = {"t_test_paired_agg(before, after, {'alternative': 'two_sided'})"};
    d1.categories = {"hypothesis-testing"};
    d1.parameter_names = {"x", "y", "options"};
    d1.parameter_types = {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::ANY};
    info.descriptions.push_back(std::move(d1));
    FunctionDescription d2;
    d2.description = "Performs a paired t-test of H0: mean(x - y) = 0, with Cohen's d_z, using default options.";
    d2.examples = {"t_test_paired_agg(before, after)"};
    d2.categories = {"hypothesis-testing"};
    d2.parameter_names = {"x", "y"};
    d2.parameter_types = {LogicalType::DOUBLE, LogicalType::DOUBLE};
    info.descriptions.push_back(std::move(d2));
    loader.RegisterFunction(std::move(info));
}

void RegisterTTestOneSamplePairedAggregateFunctions(ExtensionLoader &loader) {
    RegisterTTestOneSample(loader);
    RegisterTTestPaired(loader);
}

} // namespace duckdb
