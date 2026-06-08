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

struct RansacFitPredictState {
    vector<double> y_values;
    vector<vector<double>> x_columns;
    idx_t n_features;
    bool initialized;

    vector<double> current_x;
    bool has_current_x;

    bool fit_intercept;
    double confidence_level;
    uint32_t max_trials;
    double stop_probability;
    uint64_t random_state;
    bool min_samples_set;
    uint32_t min_samples_value;
    bool residual_threshold_set;
    double residual_threshold_value;
    bool stop_n_inliers_set;
    uint32_t stop_n_inliers_value;
    NullPolicy null_policy;

    RansacFitPredictState()
        : n_features(0), initialized(false), has_current_x(false), fit_intercept(true), confidence_level(0.95),
          max_trials(100), stop_probability(0.99), random_state(0), min_samples_set(false), min_samples_value(0),
          residual_threshold_set(false), residual_threshold_value(0.0), stop_n_inliers_set(false),
          stop_n_inliers_value(0), null_policy(NullPolicy::DROP) {}

    void Reset() {
        y_values.clear();
        x_columns.clear();
        current_x.clear();
        n_features = 0;
        initialized = false;
        has_current_x = false;
    }
};

struct RansacFitPredictBindData : public FunctionData {
    bool fit_intercept = true;
    double confidence_level = 0.95;
    uint32_t max_trials = 100;
    double stop_probability = 0.99;
    uint64_t random_state = 0;
    bool min_samples_set = false;
    uint32_t min_samples_value = 0;
    bool residual_threshold_set = false;
    double residual_threshold_value = 0.0;
    bool stop_n_inliers_set = false;
    uint32_t stop_n_inliers_value = 0;
    NullPolicy null_policy = NullPolicy::DROP;

    unique_ptr<FunctionData> Copy() const override {
        auto r = make_uniq<RansacFitPredictBindData>();
        r->fit_intercept = fit_intercept;
        r->confidence_level = confidence_level;
        r->max_trials = max_trials;
        r->stop_probability = stop_probability;
        r->random_state = random_state;
        r->min_samples_set = min_samples_set;
        r->min_samples_value = min_samples_value;
        r->residual_threshold_set = residual_threshold_set;
        r->residual_threshold_value = residual_threshold_value;
        r->stop_n_inliers_set = stop_n_inliers_set;
        r->stop_n_inliers_value = stop_n_inliers_value;
        r->null_policy = null_policy;
        return std::move(r);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &o = other_p.Cast<RansacFitPredictBindData>();
        return fit_intercept == o.fit_intercept && confidence_level == o.confidence_level &&
               max_trials == o.max_trials && stop_probability == o.stop_probability &&
               random_state == o.random_state && min_samples_set == o.min_samples_set &&
               min_samples_value == o.min_samples_value && residual_threshold_set == o.residual_threshold_set &&
               residual_threshold_value == o.residual_threshold_value &&
               stop_n_inliers_set == o.stop_n_inliers_set && stop_n_inliers_value == o.stop_n_inliers_value &&
               null_policy == o.null_policy;
    }
};

static LogicalType GetRansacFitPredictResultType() {
    child_list_t<LogicalType> children;
    children.push_back(make_pair("yhat", LogicalType::DOUBLE));
    children.push_back(make_pair("yhat_lower", LogicalType::DOUBLE));
    children.push_back(make_pair("yhat_upper", LogicalType::DOUBLE));
    return LogicalType::STRUCT(std::move(children));
}

static void RansacFitPredictInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) RansacFitPredictState();
}

static void RansacFitPredictDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (RansacFitPredictState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~RansacFitPredictState();
    }
}

static void RansacFitPredictUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                   Vector &state_vector, idx_t count) {
    auto &bind_data = aggr_input_data.bind_data->Cast<RansacFitPredictBindData>();

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
    auto states = (RansacFitPredictState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];

        state.fit_intercept = bind_data.fit_intercept;
        state.confidence_level = bind_data.confidence_level;
        state.max_trials = bind_data.max_trials;
        state.stop_probability = bind_data.stop_probability;
        state.random_state = bind_data.random_state;
        state.min_samples_set = bind_data.min_samples_set;
        state.min_samples_value = bind_data.min_samples_value;
        state.residual_threshold_set = bind_data.residual_threshold_set;
        state.residual_threshold_value = bind_data.residual_threshold_value;
        state.stop_n_inliers_set = bind_data.stop_n_inliers_set;
        state.stop_n_inliers_value = bind_data.stop_n_inliers_value;
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

static void RansacFitPredictCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (RansacFitPredictState **)source_data.data;
    auto targets = (RansacFitPredictState **)target_data.data;

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
            target.max_trials = source.max_trials;
            target.stop_probability = source.stop_probability;
            target.random_state = source.random_state;
            target.min_samples_set = source.min_samples_set;
            target.min_samples_value = source.min_samples_value;
            target.residual_threshold_set = source.residual_threshold_set;
            target.residual_threshold_value = source.residual_threshold_value;
            target.stop_n_inliers_set = source.stop_n_inliers_set;
            target.stop_n_inliers_value = source.stop_n_inliers_value;
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

static void RansacFitPredictFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count,
                                     idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (RansacFitPredictState **)sdata.data;

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

        AnofoxRansacOptions options;
        options.fit_intercept = state.fit_intercept;
        options.compute_inference = false;
        options.confidence_level = state.confidence_level;
        options.max_trials = state.max_trials;
        options.stop_probability = state.stop_probability;
        options.random_state = state.random_state;
        options.min_samples_set = state.min_samples_set;
        options.min_samples_value = state.min_samples_value;
        options.residual_threshold_set = state.residual_threshold_set;
        options.residual_threshold_value = state.residual_threshold_value;
        options.stop_n_inliers_set = state.stop_n_inliers_set;
        options.stop_n_inliers_value = state.stop_n_inliers_value;

        AnofoxFitResultCore core_result;
        AnofoxError error;

        bool success = anofox_ransac_fit(y_array, x_arrays.data(), x_arrays.size(), options, &core_result,
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

static unique_ptr<FunctionData> RansacFitPredictBind(ClientContext &context, AggregateFunction &function,
                                                     vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<RansacFitPredictBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[2]);
        if (opts.fit_intercept.has_value()) {
            result->fit_intercept = opts.fit_intercept.value();
        }
        if (opts.confidence_level.has_value()) {
            result->confidence_level = opts.confidence_level.value();
        }
        if (opts.max_trials.has_value()) {
            result->max_trials = opts.max_trials.value();
        }
        if (opts.stop_probability.has_value()) {
            result->stop_probability = opts.stop_probability.value();
        }
        if (opts.random_state.has_value()) {
            result->random_state = opts.random_state.value();
        }
        if (opts.min_samples.has_value()) {
            result->min_samples_set = true;
            result->min_samples_value = opts.min_samples.value();
        }
        if (opts.residual_threshold.has_value()) {
            result->residual_threshold_set = true;
            result->residual_threshold_value = opts.residual_threshold.value();
        }
        if (opts.stop_n_inliers.has_value()) {
            result->stop_n_inliers_set = true;
            result->stop_n_inliers_value = opts.stop_n_inliers.value();
        }
        if (opts.null_policy.has_value()) {
            result->null_policy = opts.null_policy.value();
        }
    }

    function.return_type = GetRansacFitPredictResultType();
    PostHogTelemetry::Instance().CaptureFunctionExecution("ransac_fit_predict");
    return std::move(result);
}

void RegisterRansacFitPredictFunction(ExtensionLoader &loader) {
    auto basic_func = AggregateFunction(
        "anofox_stats_ransac_fit_predict", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)},
        GetRansacFitPredictResultType(), AggregateFunction::StateSize<RansacFitPredictState>,
        RansacFitPredictInitialize, RansacFitPredictUpdate, RansacFitPredictCombine, RansacFitPredictFinalize,
        nullptr, RansacFitPredictBind, RansacFitPredictDestroy);

    auto map_func = AggregateFunction(
        "anofox_stats_ransac_fit_predict",
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY},
        GetRansacFitPredictResultType(), AggregateFunction::StateSize<RansacFitPredictState>,
        RansacFitPredictInitialize, RansacFitPredictUpdate, RansacFitPredictCombine, RansacFitPredictFinalize,
        nullptr, RansacFitPredictBind, RansacFitPredictDestroy);

    {
        AggregateFunctionSet func_set("anofox_stats_ransac_fit_predict");
        func_set.AddFunction(basic_func);
        func_set.AddFunction(map_func);
        CreateAggregateFunctionInfo info(std::move(func_set));
        info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;

        FunctionDescription d1;
        d1.description = "Fits a RANSAC robust regression over a window partition and returns the prediction for the "
                         "current row.";
        d1.examples = {"anofox_stats_ransac_fit_predict(y, x) OVER (PARTITION BY g ORDER BY t)"};
        d1.categories = {"regression", "prediction"};
        d1.parameter_names = {"y", "x"};
        d1.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)};
        info.descriptions.push_back(std::move(d1));

        FunctionDescription d2;
        d2.description = "Fits a RANSAC regression over a window with a MAP of options.";
        d2.examples = {"anofox_stats_ransac_fit_predict(y, x, {'residual_threshold': 0.5}) OVER (...)"};
        d2.categories = {"regression", "prediction"};
        d2.parameter_names = {"y", "x", "options"};
        d2.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY};
        info.descriptions.push_back(std::move(d2));

        loader.RegisterFunction(std::move(info));
    }

    {
        AggregateFunctionSet alias_set("ransac_fit_predict");
        alias_set.AddFunction(basic_func);
        alias_set.AddFunction(map_func);
        CreateAggregateFunctionInfo alias_info(std::move(alias_set));
        alias_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        alias_info.alias_of = "anofox_stats_ransac_fit_predict";
        loader.RegisterFunction(std::move(alias_info));
    }
}

} // namespace duckdb
