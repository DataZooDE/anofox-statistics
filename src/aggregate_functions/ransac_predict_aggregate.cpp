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
// RANSAC predict-aggregate state — same training-rows + all-rows split as
// the OLS/Huber predict aggregates. Options carry the full RANSAC knob set.
//===--------------------------------------------------------------------===//
struct RansacPredictAggState {
    vector<double> y_train;
    vector<vector<double>> x_train;

    vector<double> y_all;
    vector<bool> y_is_null;
    vector<bool> is_training;
    vector<vector<double>> x_all;

    idx_t n_features;
    bool initialized;

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
    bool use_split_col;

    RansacPredictAggState()
        : n_features(0), initialized(false), fit_intercept(true), confidence_level(0.95), max_trials(100),
          stop_probability(0.99), random_state(0), min_samples_set(false), min_samples_value(0),
          residual_threshold_set(false), residual_threshold_value(0.0), stop_n_inliers_set(false),
          stop_n_inliers_value(0), null_policy(NullPolicy::DROP), use_split_col(false) {}

    void Reset() {
        y_train.clear();
        x_train.clear();
        y_all.clear();
        y_is_null.clear();
        is_training.clear();
        x_all.clear();
        n_features = 0;
        initialized = false;
    }
};

struct RansacPredictAggBindData : public FunctionData {
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
    bool use_split_col = false;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<RansacPredictAggBindData>();
        result->fit_intercept = fit_intercept;
        result->confidence_level = confidence_level;
        result->max_trials = max_trials;
        result->stop_probability = stop_probability;
        result->random_state = random_state;
        result->min_samples_set = min_samples_set;
        result->min_samples_value = min_samples_value;
        result->residual_threshold_set = residual_threshold_set;
        result->residual_threshold_value = residual_threshold_value;
        result->stop_n_inliers_set = stop_n_inliers_set;
        result->stop_n_inliers_value = stop_n_inliers_value;
        result->null_policy = null_policy;
        result->use_split_col = use_split_col;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &o = other_p.Cast<RansacPredictAggBindData>();
        return fit_intercept == o.fit_intercept && confidence_level == o.confidence_level &&
               max_trials == o.max_trials && stop_probability == o.stop_probability &&
               random_state == o.random_state && min_samples_set == o.min_samples_set &&
               min_samples_value == o.min_samples_value && residual_threshold_set == o.residual_threshold_set &&
               residual_threshold_value == o.residual_threshold_value &&
               stop_n_inliers_set == o.stop_n_inliers_set && stop_n_inliers_value == o.stop_n_inliers_value &&
               null_policy == o.null_policy && use_split_col == o.use_split_col;
    }
};

// Result shape identical to ols_fit_predict_agg / huber_fit_predict_agg so
// the *_fit_predict_by macros and UNNEST patterns stay drop-in.
static LogicalType GetRansacPredictAggResultType() {
    child_list_t<LogicalType> row_children;
    row_children.push_back(make_pair("y", LogicalType::DOUBLE));
    row_children.push_back(make_pair("yhat", LogicalType::DOUBLE));
    row_children.push_back(make_pair("yhat_lower", LogicalType::DOUBLE));
    row_children.push_back(make_pair("yhat_upper", LogicalType::DOUBLE));
    row_children.push_back(make_pair("is_training", LogicalType::BOOLEAN));
    return LogicalType::LIST(LogicalType::STRUCT(std::move(row_children)));
}

static void RansacPredictAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) RansacPredictAggState();
}

static void RansacPredictAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (RansacPredictAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~RansacPredictAggState();
    }
}

static bool IsSplitTraining(const string_t &split_val) {
    string val = split_val.GetString();
    for (auto &c : val) {
        c = std::tolower(c);
    }
    return val == "train" || val == "training";
}

static void RansacPredictAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                   Vector &state_vector, idx_t count) {
    auto &bind_data = aggr_input_data.bind_data->Cast<RansacPredictAggBindData>();

    UnifiedVectorFormat y_data;
    UnifiedVectorFormat x_data;
    inputs[0].ToUnifiedFormat(count, y_data);
    inputs[1].ToUnifiedFormat(count, x_data);

    auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
    auto x_list_data = ListVector::GetData(inputs[1]);
    auto &x_child = ListVector::GetEntry(inputs[1]);
    auto x_child_data = FlatVector::GetData<double>(x_child);
    auto &x_child_validity = FlatVector::Validity(x_child);

    UnifiedVectorFormat split_data;
    const string_t *split_values = nullptr;
    if (bind_data.use_split_col && input_count >= 3) {
        inputs[2].ToUnifiedFormat(count, split_data);
        split_values = UnifiedVectorFormat::GetData<string_t>(split_data);
    }

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (RansacPredictAggState **)sdata.data;

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
        state.use_split_col = bind_data.use_split_col;

        auto x_idx = x_data.sel->get_index(i);
        if (!x_data.validity.RowIsValid(x_idx)) {
            continue;
        }

        auto list_entry = x_list_data[x_idx];
        idx_t n_features = list_entry.length;

        if (!state.initialized) {
            state.n_features = n_features;
            state.x_train.resize(n_features);
            state.initialized = true;
        }

        if (n_features != state.n_features) {
            throw InvalidInputException("Inconsistent feature count: expected %lu, got %lu", state.n_features,
                                        n_features);
        }

        vector<double> x_row(n_features);
        for (idx_t j = 0; j < n_features; j++) {
            idx_t child_pos = list_entry.offset + j;
            // List elements can be NULL; the flat-vector slot for a NULL is
            // uninitialized and must not be read (returned garbage that poisoned
            // predictions, #95). Substitute NaN so a missing feature yields a
            // NaN/NULL prediction instead of garbage.
            x_row[j] = x_child_validity.RowIsValid(child_pos) ? x_child_data[child_pos] : std::nan("");
        }

        auto y_idx = y_data.sel->get_index(i);
        bool y_valid = y_data.validity.RowIsValid(y_idx);
        double y_val = y_valid ? y_values[y_idx] : std::nan("");

        bool row_is_training;
        if (bind_data.use_split_col && split_values) {
            auto split_idx = split_data.sel->get_index(i);
            if (split_data.validity.RowIsValid(split_idx)) {
                row_is_training = IsSplitTraining(split_values[split_idx]);
            } else {
                row_is_training = false;
            }
            if (row_is_training && !y_valid) {
                row_is_training = false;
            }
        } else {
            row_is_training = y_valid;
        }

        if (row_is_training && state.null_policy == NullPolicy::DROP_Y_ZERO_X) {
            for (idx_t j = 0; j < n_features; j++) {
                if (x_row[j] == 0.0) {
                    row_is_training = false;
                    break;
                }
            }
        }

        state.y_all.push_back(y_val);
        state.y_is_null.push_back(!y_valid);
        state.is_training.push_back(row_is_training);
        state.x_all.push_back(x_row);

        if (row_is_training) {
            state.y_train.push_back(y_val);
            for (idx_t j = 0; j < n_features; j++) {
                state.x_train[j].push_back(x_row[j]);
            }
        }
    }
}

static void RansacPredictAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (RansacPredictAggState **)source_data.data;
    auto targets = (RansacPredictAggState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.y_train = std::move(source.y_train);
            target.x_train = std::move(source.x_train);
            target.y_all = std::move(source.y_all);
            target.y_is_null = std::move(source.y_is_null);
            target.is_training = std::move(source.is_training);
            target.x_all = std::move(source.x_all);
            target.n_features = source.n_features;
            target.initialized = true;
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
            target.use_split_col = source.use_split_col;
            continue;
        }

        if (source.n_features != target.n_features) {
            throw InvalidInputException("Cannot combine states with different feature counts: %lu vs %lu",
                                        source.n_features, target.n_features);
        }

        target.y_train.insert(target.y_train.end(), source.y_train.begin(), source.y_train.end());
        for (idx_t j = 0; j < target.n_features; j++) {
            target.x_train[j].insert(target.x_train[j].end(), source.x_train[j].begin(), source.x_train[j].end());
        }
        target.y_all.insert(target.y_all.end(), source.y_all.begin(), source.y_all.end());
        target.y_is_null.insert(target.y_is_null.end(), source.y_is_null.begin(), source.y_is_null.end());
        target.is_training.insert(target.is_training.end(), source.is_training.begin(), source.is_training.end());
        target.x_all.insert(target.x_all.end(), source.x_all.begin(), source.x_all.end());
    }
}

static void RansacPredictAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                     idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (RansacPredictAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.y_train.size() < 2) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        AnofoxDataArray y_array;
        y_array.data = state.y_train.data();
        y_array.validity = nullptr;
        y_array.len = state.y_train.size();

        vector<AnofoxDataArray> x_arrays;
        for (auto &col : state.x_train) {
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

        // Predict-only flow — no inference, no extras (per-fit consensus
        // diagnostics belong in ransac_fit_agg).
        bool success = anofox_ransac_fit(y_array, x_arrays.data(), x_arrays.size(), options, &core_result,
                                         nullptr, nullptr, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t n_rows = state.y_all.size();
        auto *list_data = ListVector::GetData(result);
        auto list_offset = ListVector::GetListSize(result);

        list_data[result_idx].offset = list_offset;
        list_data[result_idx].length = n_rows;

        ListVector::Reserve(result, list_offset + n_rows);
        ListVector::SetListSize(result, list_offset + n_rows);

        auto &child_struct = ListVector::GetEntry(result);
        auto &struct_entries = StructVector::GetEntries(child_struct);

        auto &y_vec = *struct_entries[0];
        auto &yhat_vec = *struct_entries[1];
        auto &yhat_lower_vec = *struct_entries[2];
        auto &yhat_upper_vec = *struct_entries[3];
        auto &is_training_vec = *struct_entries[4];

        for (idx_t row = 0; row < n_rows; row++) {
            idx_t child_idx = list_offset + row;

            if (state.y_is_null[row]) {
                FlatVector::SetNull(y_vec, child_idx, true);
            } else {
                FlatVector::GetData<double>(y_vec)[child_idx] = state.y_all[row];
            }

            AnofoxPredictionResult pred;
            bool pred_success = anofox_predict_with_interval(
                core_result.coefficients, core_result.coefficients_len, core_result.intercept,
                state.x_all[row].data(), state.n_features, core_result.residual_std_error,
                core_result.n_observations, state.confidence_level, &pred);

            if (pred_success && std::isfinite(pred.yhat)) {
                FlatVector::GetData<double>(yhat_vec)[child_idx] = pred.yhat;
                FlatVector::GetData<double>(yhat_lower_vec)[child_idx] = pred.yhat_lower;
                FlatVector::GetData<double>(yhat_upper_vec)[child_idx] = pred.yhat_upper;
            } else {
                FlatVector::SetNull(yhat_vec, child_idx, true);
                FlatVector::SetNull(yhat_lower_vec, child_idx, true);
                FlatVector::SetNull(yhat_upper_vec, child_idx, true);
            }

            FlatVector::GetData<bool>(is_training_vec)[child_idx] = state.is_training[row];
        }

        anofox_free_result_core(&core_result);
        state.Reset();
    }
}

// Shared option extraction.
static void ExtractRansacPredictOptions(ClientContext &context, Expression &opts_expr,
                                        RansacPredictAggBindData &result) {
    auto opts = RegressionMapOptions::ParseFromExpression(context, opts_expr);
    if (opts.fit_intercept.has_value()) {
        result.fit_intercept = opts.fit_intercept.value();
    }
    if (opts.confidence_level.has_value()) {
        result.confidence_level = opts.confidence_level.value();
    }
    if (opts.max_trials.has_value()) {
        result.max_trials = opts.max_trials.value();
    }
    if (opts.stop_probability.has_value()) {
        result.stop_probability = opts.stop_probability.value();
    }
    if (opts.random_state.has_value()) {
        result.random_state = opts.random_state.value();
    }
    if (opts.min_samples.has_value()) {
        result.min_samples_set = true;
        result.min_samples_value = opts.min_samples.value();
    }
    if (opts.residual_threshold.has_value()) {
        result.residual_threshold_set = true;
        result.residual_threshold_value = opts.residual_threshold.value();
    }
    if (opts.stop_n_inliers.has_value()) {
        result.stop_n_inliers_set = true;
        result.stop_n_inliers_value = opts.stop_n_inliers.value();
    }
    if (opts.null_policy.has_value()) {
        result.null_policy = opts.null_policy.value();
    }
}

static unique_ptr<FunctionData> RansacPredictAggBind(ClientContext &context, AggregateFunction &function,
                                                    vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<RansacPredictAggBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        ExtractRansacPredictOptions(context, *arguments[2], *result);
    }

    function.return_type = GetRansacPredictAggResultType();
    PostHogTelemetry::Instance().CaptureFunctionExecution("ransac_fit_predict_agg");
    return std::move(result);
}

static unique_ptr<FunctionData> RansacPredictAggBindWithSplit(ClientContext &context, AggregateFunction &function,
                                                                vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<RansacPredictAggBindData>();
    result->use_split_col = true;

    if (arguments.size() >= 4 && arguments[3]->IsFoldable()) {
        ExtractRansacPredictOptions(context, *arguments[3], *result);
    }

    function.return_type = GetRansacPredictAggResultType();
    PostHogTelemetry::Instance().CaptureFunctionExecution("ransac_fit_predict_agg");
    return std::move(result);
}

void RegisterRansacFitPredictAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_ransac_fit_predict_agg");

    auto basic_func = AggregateFunction(
        "anofox_stats_ransac_fit_predict_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)},
        LogicalType::ANY, AggregateFunction::StateSize<RansacPredictAggState>, RansacPredictAggInitialize,
        RansacPredictAggUpdate, RansacPredictAggCombine, RansacPredictAggFinalize, nullptr,
        RansacPredictAggBind, RansacPredictAggDestroy);
    func_set.AddFunction(basic_func);

    auto map_func = AggregateFunction(
        "anofox_stats_ransac_fit_predict_agg",
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY}, LogicalType::ANY,
        AggregateFunction::StateSize<RansacPredictAggState>, RansacPredictAggInitialize, RansacPredictAggUpdate,
        RansacPredictAggCombine, RansacPredictAggFinalize, nullptr, RansacPredictAggBind, RansacPredictAggDestroy);
    func_set.AddFunction(map_func);

    auto split_func = AggregateFunction(
        "anofox_stats_ransac_fit_predict_agg",
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::VARCHAR}, LogicalType::ANY,
        AggregateFunction::StateSize<RansacPredictAggState>, RansacPredictAggInitialize, RansacPredictAggUpdate,
        RansacPredictAggCombine, RansacPredictAggFinalize, nullptr, RansacPredictAggBindWithSplit,
        RansacPredictAggDestroy);
    func_set.AddFunction(split_func);

    auto split_opts_func = AggregateFunction(
        "anofox_stats_ransac_fit_predict_agg",
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::VARCHAR, LogicalType::ANY},
        LogicalType::ANY, AggregateFunction::StateSize<RansacPredictAggState>, RansacPredictAggInitialize,
        RansacPredictAggUpdate, RansacPredictAggCombine, RansacPredictAggFinalize, nullptr,
        RansacPredictAggBindWithSplit, RansacPredictAggDestroy);
    func_set.AddFunction(split_opts_func);

    CreateAggregateFunctionInfo info(std::move(func_set));
    info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;

    FunctionDescription d1;
    d1.description = "Fits a RANSAC robust regression over a partition and returns per-row predictions with "
                     "confidence intervals.";
    d1.examples = {"anofox_stats_ransac_fit_predict_agg(y, x)"};
    d1.categories = {"regression", "prediction"};
    d1.parameter_names = {"y", "x"};
    d1.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)};
    info.descriptions.push_back(std::move(d1));

    FunctionDescription d2;
    d2.description = "Fits RANSAC over a partition with a MAP of options and returns per-row predictions.";
    d2.examples = {"anofox_stats_ransac_fit_predict_agg(y, x, {'residual_threshold': 0.5, 'random_state': 42})"};
    d2.categories = {"regression", "prediction"};
    d2.parameter_names = {"y", "x", "options"};
    d2.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY};
    info.descriptions.push_back(std::move(d2));

    FunctionDescription d3;
    d3.description = "Fits RANSAC using only training rows (split_col='train') and predicts all rows.";
    d3.examples = {"anofox_stats_ransac_fit_predict_agg(y, x, split_col)"};
    d3.categories = {"regression", "prediction"};
    d3.parameter_names = {"y", "x", "split_col"};
    d3.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::VARCHAR};
    info.descriptions.push_back(std::move(d3));

    FunctionDescription d4;
    d4.description = "Fits RANSAC on training rows with a MAP of options and predicts all rows.";
    d4.examples = {"anofox_stats_ransac_fit_predict_agg(y, x, split_col, {'residual_threshold': 0.5})"};
    d4.categories = {"regression", "prediction"};
    d4.parameter_names = {"y", "x", "split_col", "options"};
    d4.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::VARCHAR,
                          LogicalType::ANY};
    info.descriptions.push_back(std::move(d4));

    loader.RegisterFunction(std::move(info));

    {
        AggregateFunctionSet alias_set("ransac_fit_predict_agg");
        alias_set.AddFunction(basic_func);
        alias_set.AddFunction(map_func);
        alias_set.AddFunction(split_func);
        alias_set.AddFunction(split_opts_func);
        CreateAggregateFunctionInfo alias_info(std::move(alias_set));
        alias_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        alias_info.alias_of = "anofox_stats_ransac_fit_predict_agg";
        loader.RegisterFunction(std::move(alias_info));
    }
}

} // namespace duckdb
