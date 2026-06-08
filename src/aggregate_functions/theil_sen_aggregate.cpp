#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/ffi_enum_converters.hpp"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Theil-Sen aggregate state. The single Option<usize> knob (n_subsamples)
// is stored as a have/value pair that the FFI flattens back at Finalize.
//===--------------------------------------------------------------------===//
struct TheilSenAggregateState {
    vector<double> y_values;
    vector<vector<double>> x_columns;
    idx_t n_features;
    bool initialized;

    bool fit_intercept;
    bool compute_inference;
    double confidence_level;
    uint32_t max_subpopulation;
    uint32_t max_iterations;
    double tolerance;
    uint64_t random_state;

    bool n_subsamples_set;
    uint32_t n_subsamples_value;

    TheilSenAggregateState()
        : n_features(0), initialized(false), fit_intercept(true), compute_inference(false),
          confidence_level(0.95), max_subpopulation(10000), max_iterations(300), tolerance(1e-3), random_state(0),
          n_subsamples_set(false), n_subsamples_value(0) {}

    void Reset() {
        y_values.clear();
        x_columns.clear();
        n_features = 0;
        initialized = false;
    }
};

struct TheilSenAggregateBindData : public FunctionData {
    bool fit_intercept = true;
    bool compute_inference = false;
    double confidence_level = 0.95;
    uint32_t max_subpopulation = 10000;
    uint32_t max_iterations = 300;
    double tolerance = 1e-3;
    uint64_t random_state = 0;
    bool n_subsamples_set = false;
    uint32_t n_subsamples_value = 0;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<TheilSenAggregateBindData>();
        result->fit_intercept = fit_intercept;
        result->compute_inference = compute_inference;
        result->confidence_level = confidence_level;
        result->max_subpopulation = max_subpopulation;
        result->max_iterations = max_iterations;
        result->tolerance = tolerance;
        result->random_state = random_state;
        result->n_subsamples_set = n_subsamples_set;
        result->n_subsamples_value = n_subsamples_value;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &o = other_p.Cast<TheilSenAggregateBindData>();
        return fit_intercept == o.fit_intercept && compute_inference == o.compute_inference &&
               confidence_level == o.confidence_level && max_subpopulation == o.max_subpopulation &&
               max_iterations == o.max_iterations && tolerance == o.tolerance &&
               random_state == o.random_state && n_subsamples_set == o.n_subsamples_set &&
               n_subsamples_value == o.n_subsamples_value;
    }
};

// Result shape: standard fit fields, no Theil-Sen-specific extras (no inlier
// mask / trial count to surface).
static LogicalType GetTheilSenAggResultType(bool compute_inference) {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
    children.push_back(make_pair("intercept", LogicalType::DOUBLE));
    children.push_back(make_pair("r_squared", LogicalType::DOUBLE));
    children.push_back(make_pair("adj_r_squared", LogicalType::DOUBLE));
    children.push_back(make_pair("residual_std_error", LogicalType::DOUBLE));
    children.push_back(make_pair("n_observations", LogicalType::BIGINT));
    children.push_back(make_pair("n_features", LogicalType::BIGINT));

    if (compute_inference) {
        children.push_back(make_pair("std_errors", LogicalType::LIST(LogicalType::DOUBLE)));
        children.push_back(make_pair("t_values", LogicalType::LIST(LogicalType::DOUBLE)));
        children.push_back(make_pair("p_values", LogicalType::LIST(LogicalType::DOUBLE)));
        children.push_back(make_pair("ci_lower", LogicalType::LIST(LogicalType::DOUBLE)));
        children.push_back(make_pair("ci_upper", LogicalType::LIST(LogicalType::DOUBLE)));
        children.push_back(make_pair("f_statistic", LogicalType::DOUBLE));
        children.push_back(make_pair("f_pvalue", LogicalType::DOUBLE));
    }

    return LogicalType::STRUCT(std::move(children));
}

static void TheilSenAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) TheilSenAggregateState();
}

static void TheilSenAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (TheilSenAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~TheilSenAggregateState();
    }
}

static void TheilSenAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                              Vector &state_vector, idx_t count) {
    auto &bind_data = aggr_input_data.bind_data->Cast<TheilSenAggregateBindData>();

    UnifiedVectorFormat y_data;
    UnifiedVectorFormat x_data;
    inputs[0].ToUnifiedFormat(count, y_data);
    inputs[1].ToUnifiedFormat(count, x_data);

    auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
    auto x_list_data = ListVector::GetData(inputs[1]);
    auto &x_child = ListVector::GetEntry(inputs[1]);
    auto x_child_data = FlatVector::GetData<double>(x_child);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (TheilSenAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];

        state.fit_intercept = bind_data.fit_intercept;
        state.compute_inference = bind_data.compute_inference;
        state.confidence_level = bind_data.confidence_level;
        state.max_subpopulation = bind_data.max_subpopulation;
        state.max_iterations = bind_data.max_iterations;
        state.tolerance = bind_data.tolerance;
        state.random_state = bind_data.random_state;
        state.n_subsamples_set = bind_data.n_subsamples_set;
        state.n_subsamples_value = bind_data.n_subsamples_value;

        auto y_idx = y_data.sel->get_index(i);
        if (!y_data.validity.RowIsValid(y_idx)) {
            continue;
        }
        double y_val = y_values[y_idx];

        auto x_idx = x_data.sel->get_index(i);
        if (!x_data.validity.RowIsValid(x_idx)) {
            continue;
        }

        auto list_entry = x_list_data[x_idx];
        idx_t n_features = list_entry.length;

        if (!state.initialized) {
            state.n_features = n_features;
            state.x_columns.resize(n_features);
            state.initialized = true;
        }

        if (n_features != state.n_features) {
            throw InvalidInputException("Inconsistent feature count: expected %lu, got %lu", state.n_features,
                                        n_features);
        }

        state.y_values.push_back(y_val);
        for (idx_t j = 0; j < n_features; j++) {
            double x_val = x_child_data[list_entry.offset + j];
            state.x_columns[j].push_back(x_val);
        }
    }
}

static void TheilSenAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (TheilSenAggregateState **)source_data.data;
    auto targets = (TheilSenAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.y_values = std::move(source.y_values);
            target.x_columns = std::move(source.x_columns);
            target.n_features = source.n_features;
            target.initialized = true;
            target.fit_intercept = source.fit_intercept;
            target.compute_inference = source.compute_inference;
            target.confidence_level = source.confidence_level;
            target.max_subpopulation = source.max_subpopulation;
            target.max_iterations = source.max_iterations;
            target.tolerance = source.tolerance;
            target.random_state = source.random_state;
            target.n_subsamples_set = source.n_subsamples_set;
            target.n_subsamples_value = source.n_subsamples_value;
            continue;
        }

        if (source.n_features != target.n_features) {
            throw InvalidInputException("Cannot combine states with different feature counts: %lu vs %lu",
                                        source.n_features, target.n_features);
        }

        target.y_values.insert(target.y_values.end(), source.y_values.begin(), source.y_values.end());
        for (idx_t j = 0; j < target.n_features; j++) {
            target.x_columns[j].insert(target.x_columns[j].end(), source.x_columns[j].begin(),
                                       source.x_columns[j].end());
        }
    }
}

static void SetListInResult(Vector &list_vec, idx_t row, double *data, size_t len) {
    auto &child = ListVector::GetEntry(list_vec);
    auto offset = ListVector::GetListSize(list_vec);
    ListVector::SetListSize(list_vec, offset + len);
    auto vec_data = FlatVector::GetData<double>(child);
    for (size_t i = 0; i < len; i++) {
        vec_data[offset + i] = data[i];
    }
    ListVector::GetData(list_vec)[row] = {offset, (idx_t)len};
}

static void TheilSenAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (TheilSenAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.y_values.size() < 2) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        AnofoxDataArray y_array;
        y_array.data = state.y_values.data();
        y_array.validity = nullptr;
        y_array.len = state.y_values.size();

        vector<AnofoxDataArray> x_arrays;
        for (auto &col : state.x_columns) {
            AnofoxDataArray arr;
            arr.data = col.data();
            arr.validity = nullptr;
            arr.len = col.size();
            x_arrays.push_back(arr);
        }

        AnofoxTheilSenOptions options;
        options.fit_intercept = state.fit_intercept;
        options.compute_inference = state.compute_inference;
        options.confidence_level = state.confidence_level;
        options.max_subpopulation = state.max_subpopulation;
        options.max_iterations = state.max_iterations;
        options.tolerance = state.tolerance;
        options.random_state = state.random_state;
        options.n_subsamples_set = state.n_subsamples_set;
        options.n_subsamples_value = state.n_subsamples_value;

        AnofoxFitResultCore core_result;
        AnofoxFitResultInference inference_result;
        AnofoxError error;

        bool success = anofox_theilsen_fit(y_array, x_arrays.data(), x_arrays.size(), options, &core_result,
                                           state.compute_inference ? &inference_result : nullptr, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t struct_idx = 0;

        SetListInResult(*struct_entries[struct_idx++], result_idx, core_result.coefficients,
                        core_result.coefficients_len);

        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.intercept;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.r_squared;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.adj_r_squared;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.residual_std_error;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = core_result.n_observations;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = core_result.n_features;

        if (state.compute_inference) {
            SetListInResult(*struct_entries[struct_idx++], result_idx, inference_result.std_errors,
                            inference_result.len);
            SetListInResult(*struct_entries[struct_idx++], result_idx, inference_result.t_values,
                            inference_result.len);
            SetListInResult(*struct_entries[struct_idx++], result_idx, inference_result.p_values,
                            inference_result.len);
            SetListInResult(*struct_entries[struct_idx++], result_idx, inference_result.ci_lower,
                            inference_result.len);
            SetListInResult(*struct_entries[struct_idx++], result_idx, inference_result.ci_upper,
                            inference_result.len);

            FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = inference_result.f_statistic;
            FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = inference_result.f_pvalue;

            anofox_free_result_inference(&inference_result);
        }

        anofox_free_result_core(&core_result);
        state.Reset();
    }
}

static void ExtractTheilSenOptions(ClientContext &context, Expression &opts_expr,
                                   TheilSenAggregateBindData &result) {
    auto opts = RegressionMapOptions::ParseFromExpression(context, opts_expr);
    if (opts.fit_intercept.has_value()) {
        result.fit_intercept = opts.fit_intercept.value();
    }
    if (opts.compute_inference.has_value()) {
        result.compute_inference = opts.compute_inference.value();
    }
    if (opts.confidence_level.has_value()) {
        result.confidence_level = opts.confidence_level.value();
    }
    if (opts.max_subpopulation.has_value()) {
        result.max_subpopulation = opts.max_subpopulation.value();
    }
    if (opts.max_iterations.has_value()) {
        result.max_iterations = opts.max_iterations.value();
    }
    if (opts.tolerance.has_value()) {
        result.tolerance = opts.tolerance.value();
    }
    if (opts.random_state.has_value()) {
        result.random_state = opts.random_state.value();
    }
    if (opts.n_subsamples.has_value()) {
        result.n_subsamples_set = true;
        result.n_subsamples_value = opts.n_subsamples.value();
    }
}

static unique_ptr<FunctionData> TheilSenAggBind(ClientContext &context, AggregateFunction &function,
                                                vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<TheilSenAggregateBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        ExtractTheilSenOptions(context, *arguments[2], *result);
    }

    function.return_type = GetTheilSenAggResultType(result->compute_inference);

    PostHogTelemetry::Instance().CaptureFunctionExecution("theilsen_fit_agg");
    return std::move(result);
}

void RegisterTheilSenAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_theilsen_fit_agg");

    auto basic_func = AggregateFunction(
        "anofox_stats_theilsen_fit_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)},
        LogicalType::ANY, AggregateFunction::StateSize<TheilSenAggregateState>, TheilSenAggInitialize,
        TheilSenAggUpdate, TheilSenAggCombine, TheilSenAggFinalize, nullptr, TheilSenAggBind, TheilSenAggDestroy);
    func_set.AddFunction(basic_func);

    auto map_func = AggregateFunction(
        "anofox_stats_theilsen_fit_agg",
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY}, LogicalType::ANY,
        AggregateFunction::StateSize<TheilSenAggregateState>, TheilSenAggInitialize, TheilSenAggUpdate,
        TheilSenAggCombine, TheilSenAggFinalize, nullptr, TheilSenAggBind, TheilSenAggDestroy);
    func_set.AddFunction(map_func);

    CreateAggregateFunctionInfo info(std::move(func_set));
    info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
    FunctionDescription d1;
    d1.description =
        "Fits a Theil-Sen robust regression model and returns coefficients and fit statistics as a struct.";
    d1.examples = {"anofox_stats_theilsen_fit_agg(y, x, {'random_state': 42, 'max_subpopulation': 5000})"};
    d1.categories = {"regression"};
    d1.parameter_names = {"y", "x", "options"};
    d1.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY};
    info.descriptions.push_back(std::move(d1));
    FunctionDescription d2;
    d2.description =
        "Fits a Theil-Sen robust regression model and returns coefficients and fit statistics as a struct.";
    d2.examples = {"anofox_stats_theilsen_fit_agg(y, x)"};
    d2.categories = {"regression"};
    d2.parameter_names = {"y", "x"};
    d2.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)};
    info.descriptions.push_back(std::move(d2));
    loader.RegisterFunction(std::move(info));

    {
        AggregateFunctionSet alias_set("theilsen_fit_agg");
        alias_set.AddFunction(basic_func);
        alias_set.AddFunction(map_func);
        CreateAggregateFunctionInfo alias_info(std::move(alias_set));
        alias_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        alias_info.alias_of = "anofox_stats_theilsen_fit_agg";
        loader.RegisterFunction(std::move(alias_info));
    }
}

} // namespace duckdb
