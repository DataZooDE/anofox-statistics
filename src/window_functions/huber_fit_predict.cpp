#include <cmath>
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
// Huber fit-predict window state — accumulates training rows in the current
// frame and stores the latest row's x for prediction. Mirrors
// OlsFitPredictState; swaps options for Huber's epsilon/alpha/etc.
//===--------------------------------------------------------------------===//
struct HuberFitPredictState {
    vector<double> y_values;
    vector<vector<double>> x_columns;
    idx_t n_features;
    bool initialized;

    vector<double> current_x;
    bool has_current_x;

    bool fit_intercept;
    double confidence_level;
    double epsilon;
    double alpha;
    uint32_t max_iterations;
    double tolerance;
    NullPolicy null_policy;

    HuberFitPredictState()
        : n_features(0), initialized(false), has_current_x(false), fit_intercept(true), confidence_level(0.95),
          epsilon(1.35), alpha(0.0001), max_iterations(100), tolerance(1e-5), null_policy(NullPolicy::DROP) {}

    void Reset() {
        y_values.clear();
        x_columns.clear();
        current_x.clear();
        n_features = 0;
        initialized = false;
        has_current_x = false;
    }
};

struct HuberFitPredictBindData : public FunctionData {
    bool fit_intercept = true;
    double confidence_level = 0.95;
    double epsilon = 1.35;
    double alpha = 0.0001;
    uint32_t max_iterations = 100;
    double tolerance = 1e-5;
    NullPolicy null_policy = NullPolicy::DROP;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<HuberFitPredictBindData>();
        result->fit_intercept = fit_intercept;
        result->confidence_level = confidence_level;
        result->epsilon = epsilon;
        result->alpha = alpha;
        result->max_iterations = max_iterations;
        result->tolerance = tolerance;
        result->null_policy = null_policy;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<HuberFitPredictBindData>();
        return fit_intercept == other.fit_intercept && confidence_level == other.confidence_level &&
               epsilon == other.epsilon && alpha == other.alpha && max_iterations == other.max_iterations &&
               tolerance == other.tolerance && null_policy == other.null_policy;
    }
};

// Result type: STRUCT(yhat, yhat_lower, yhat_upper) — identical to OLS so
// downstream macros / per-row consumers stay drop-in compatible.
static LogicalType GetHuberFitPredictResultType() {
    child_list_t<LogicalType> children;
    children.push_back(make_pair("yhat", LogicalType::DOUBLE));
    children.push_back(make_pair("yhat_lower", LogicalType::DOUBLE));
    children.push_back(make_pair("yhat_upper", LogicalType::DOUBLE));
    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// State machine
//===--------------------------------------------------------------------===//

static void HuberFitPredictInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) HuberFitPredictState();
}

static void HuberFitPredictDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (HuberFitPredictState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~HuberFitPredictState();
    }
}

static void HuberFitPredictUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                  Vector &state_vector, idx_t count) {
    auto &bind_data = aggr_input_data.bind_data->Cast<HuberFitPredictBindData>();

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
    auto states = (HuberFitPredictState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];

        state.fit_intercept = bind_data.fit_intercept;
        state.confidence_level = bind_data.confidence_level;
        state.epsilon = bind_data.epsilon;
        state.alpha = bind_data.alpha;
        state.max_iterations = bind_data.max_iterations;
        state.tolerance = bind_data.tolerance;
        state.null_policy = bind_data.null_policy;

        auto x_idx = x_data.sel->get_index(i);
        if (!x_data.validity.RowIsValid(x_idx)) {
            state.has_current_x = false;
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

        // The current row's x — what we'll predict for in Finalize.
        state.current_x.resize(n_features);
        for (idx_t j = 0; j < n_features; j++) {
            state.current_x[j] = x_child_data[list_entry.offset + j];
        }
        state.has_current_x = true;

        auto y_idx = y_data.sel->get_index(i);
        bool y_valid = y_data.validity.RowIsValid(y_idx);
        bool use_for_training = y_valid;

        if (use_for_training && state.null_policy == NullPolicy::DROP_Y_ZERO_X) {
            for (idx_t j = 0; j < n_features; j++) {
                if (state.current_x[j] == 0.0) {
                    use_for_training = false;
                    break;
                }
            }
        }

        if (use_for_training) {
            double y_val = y_values[y_idx];
            state.y_values.push_back(y_val);

            for (idx_t j = 0; j < n_features; j++) {
                state.x_columns[j].push_back(x_child_data[list_entry.offset + j]);
            }
        }
    }
}

static void HuberFitPredictCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (HuberFitPredictState **)source_data.data;
    auto targets = (HuberFitPredictState **)target_data.data;

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
            target.current_x = std::move(source.current_x);
            target.has_current_x = source.has_current_x;
            target.fit_intercept = source.fit_intercept;
            target.confidence_level = source.confidence_level;
            target.epsilon = source.epsilon;
            target.alpha = source.alpha;
            target.max_iterations = source.max_iterations;
            target.tolerance = source.tolerance;
            target.null_policy = source.null_policy;
            continue;
        }

        if (source.n_features != target.n_features) {
            throw InvalidInputException("Cannot combine states with different feature counts");
        }

        target.y_values.insert(target.y_values.end(), source.y_values.begin(), source.y_values.end());
        for (idx_t j = 0; j < target.n_features; j++) {
            target.x_columns[j].insert(target.x_columns[j].end(), source.x_columns[j].begin(),
                                       source.x_columns[j].end());
        }

        if (source.has_current_x) {
            target.current_x = std::move(source.current_x);
            target.has_current_x = true;
        }
    }
}

static void HuberFitPredictFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count,
                                    idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (HuberFitPredictState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || !state.has_current_x) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t min_obs = state.fit_intercept ? state.n_features + 1 : state.n_features;
        if (state.y_values.size() <= min_obs) {
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

        AnofoxHuberOptions options;
        options.epsilon = state.epsilon;
        options.alpha = state.alpha;
        options.fit_intercept = state.fit_intercept;
        options.compute_inference = false;
        options.confidence_level = state.confidence_level;
        options.max_iterations = state.max_iterations;
        options.tolerance = state.tolerance;

        AnofoxFitResultCore core_result;
        AnofoxError error;

        // Predict-only — skip inference and Huber-specific extras here; the
        // window result is just yhat/yhat_lower/yhat_upper.
        bool success = anofox_huber_fit(y_array, x_arrays.data(), x_arrays.size(), options, &core_result,
                                        nullptr, nullptr, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        AnofoxPredictionResult pred_result;
        bool pred_success =
            anofox_predict_with_interval(core_result.coefficients, core_result.coefficients_len, core_result.intercept,
                                         state.current_x.data(), state.current_x.size(),
                                         core_result.residual_std_error, core_result.n_observations,
                                         state.confidence_level, &pred_result);

        anofox_free_result_core(&core_result);

        if (!pred_success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        FlatVector::GetData<double>(*struct_entries[0])[result_idx] = pred_result.yhat;
        FlatVector::GetData<double>(*struct_entries[1])[result_idx] = pred_result.yhat_lower;
        FlatVector::GetData<double>(*struct_entries[2])[result_idx] = pred_result.yhat_upper;

        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind + registration
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> HuberFitPredictBind(ClientContext &context, AggregateFunction &function,
                                                     vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<HuberFitPredictBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[2]);
        if (opts.fit_intercept.has_value()) {
            result->fit_intercept = opts.fit_intercept.value();
        }
        if (opts.confidence_level.has_value()) {
            result->confidence_level = opts.confidence_level.value();
        }
        if (opts.epsilon.has_value()) {
            result->epsilon = opts.epsilon.value();
        }
        if (opts.alpha.has_value()) {
            result->alpha = opts.alpha.value();
        }
        if (opts.max_iterations.has_value()) {
            result->max_iterations = opts.max_iterations.value();
        }
        if (opts.tolerance.has_value()) {
            result->tolerance = opts.tolerance.value();
        }
        if (opts.null_policy.has_value()) {
            result->null_policy = opts.null_policy.value();
        }
    }

    function.return_type = GetHuberFitPredictResultType();
    PostHogTelemetry::Instance().CaptureFunctionExecution("huber_fit_predict");
    return std::move(result);
}

void RegisterHuberFitPredictFunction(ExtensionLoader &loader) {
    auto basic_func = AggregateFunction(
        "anofox_stats_huber_fit_predict", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)},
        GetHuberFitPredictResultType(), AggregateFunction::StateSize<HuberFitPredictState>,
        HuberFitPredictInitialize, HuberFitPredictUpdate, HuberFitPredictCombine, HuberFitPredictFinalize, nullptr,
        HuberFitPredictBind, HuberFitPredictDestroy);

    auto map_func = AggregateFunction(
        "anofox_stats_huber_fit_predict",
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY},
        GetHuberFitPredictResultType(), AggregateFunction::StateSize<HuberFitPredictState>,
        HuberFitPredictInitialize, HuberFitPredictUpdate, HuberFitPredictCombine, HuberFitPredictFinalize, nullptr,
        HuberFitPredictBind, HuberFitPredictDestroy);

    {
        AggregateFunctionSet func_set("anofox_stats_huber_fit_predict");
        func_set.AddFunction(basic_func);
        func_set.AddFunction(map_func);
        CreateAggregateFunctionInfo info(std::move(func_set));
        info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;

        FunctionDescription d1;
        d1.description = "Fits a Huber M-estimator robust regression over a window partition and returns the "
                         "prediction for the current row with confidence intervals.";
        d1.examples = {"anofox_stats_huber_fit_predict(y, x) OVER (PARTITION BY g ORDER BY t)"};
        d1.categories = {"regression", "prediction"};
        d1.parameter_names = {"y", "x"};
        d1.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)};
        info.descriptions.push_back(std::move(d1));

        FunctionDescription d2;
        d2.description = "Fits a Huber M-estimator robust regression over a window with a MAP of options.";
        d2.examples = {"anofox_stats_huber_fit_predict(y, x, {'epsilon': 1.5}) OVER (...)"};
        d2.categories = {"regression", "prediction"};
        d2.parameter_names = {"y", "x", "options"};
        d2.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY};
        info.descriptions.push_back(std::move(d2));

        loader.RegisterFunction(std::move(info));
    }

    {
        AggregateFunctionSet alias_set("huber_fit_predict");
        alias_set.AddFunction(basic_func);
        alias_set.AddFunction(map_func);
        CreateAggregateFunctionInfo alias_info(std::move(alias_set));
        alias_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        alias_info.alias_of = "anofox_stats_huber_fit_predict";
        loader.RegisterFunction(std::move(alias_info));
    }
}

} // namespace duckdb
