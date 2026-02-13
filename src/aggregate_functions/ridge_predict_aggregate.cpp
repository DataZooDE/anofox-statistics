#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Ridge Predict Aggregate State
//===--------------------------------------------------------------------===//
struct RidgePredictAggState {
    vector<double> y_train;
    vector<vector<double>> x_train;
    vector<double> y_all;
    vector<bool> y_is_null;
    vector<bool> is_training;
    vector<vector<double>> x_all;

    idx_t n_features;
    bool initialized;

    // Options
    double alpha;
    bool fit_intercept;
    double confidence_level;
    NullPolicy null_policy;
    bool use_split_col;  // True if using split column instead of y NULL

    RidgePredictAggState()
        : n_features(0), initialized(false), alpha(1.0), fit_intercept(true), confidence_level(0.95),
          null_policy(NullPolicy::DROP), use_split_col(false) {}

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

//===--------------------------------------------------------------------===//
// Bind Data
//===--------------------------------------------------------------------===//
struct RidgePredictAggBindData : public FunctionData {
    double alpha = 1.0;
    bool fit_intercept = true;
    double confidence_level = 0.95;
    NullPolicy null_policy = NullPolicy::DROP;
    bool use_split_col = false;  // True if split column is provided

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<RidgePredictAggBindData>();
        result->alpha = alpha;
        result->fit_intercept = fit_intercept;
        result->confidence_level = confidence_level;
        result->null_policy = null_policy;
        result->use_split_col = use_split_col;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<RidgePredictAggBindData>();
        return alpha == other.alpha && fit_intercept == other.fit_intercept &&
               confidence_level == other.confidence_level && null_policy == other.null_policy &&
               use_split_col == other.use_split_col;
    }
};

//===--------------------------------------------------------------------===//
// Result type
//===--------------------------------------------------------------------===//
static LogicalType GetRidgePredictAggResultType() {
    child_list_t<LogicalType> row_children;
    row_children.push_back(make_pair("y", LogicalType::DOUBLE));
    row_children.push_back(make_pair("yhat", LogicalType::DOUBLE));
    row_children.push_back(make_pair("yhat_lower", LogicalType::DOUBLE));
    row_children.push_back(make_pair("yhat_upper", LogicalType::DOUBLE));
    row_children.push_back(make_pair("is_training", LogicalType::BOOLEAN));

    auto row_struct = LogicalType::STRUCT(std::move(row_children));
    return LogicalType::LIST(row_struct);
}

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void RidgePredictAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) RidgePredictAggState();
}

static void RidgePredictAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (RidgePredictAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~RidgePredictAggState();
    }
}

// Helper: check if split value indicates training (case-insensitive)
static bool RidgeIsSplitTraining(const string_t &split_val) {
    string val = split_val.GetString();
    for (auto &c : val) {
        c = std::tolower(c);
    }
    return val == "train" || val == "training";
}

static void RidgePredictAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                   Vector &state_vector, idx_t count) {
    auto &bind_data = aggr_input_data.bind_data->Cast<RidgePredictAggBindData>();

    UnifiedVectorFormat y_data;
    UnifiedVectorFormat x_data;
    inputs[0].ToUnifiedFormat(count, y_data);
    inputs[1].ToUnifiedFormat(count, x_data);

    auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
    auto x_list_data = ListVector::GetData(inputs[1]);
    auto &x_child = ListVector::GetEntry(inputs[1]);
    auto x_child_data = FlatVector::GetData<double>(x_child);

    // Optional split column (when use_split_col is true, it's at index 2)
    UnifiedVectorFormat split_data;
    const string_t *split_values = nullptr;
    if (bind_data.use_split_col && input_count >= 3) {
        inputs[2].ToUnifiedFormat(count, split_data);
        split_values = UnifiedVectorFormat::GetData<string_t>(split_data);
    }

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (RidgePredictAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];

        state.alpha = bind_data.alpha;
        state.fit_intercept = bind_data.fit_intercept;
        state.confidence_level = bind_data.confidence_level;
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
            x_row[j] = x_child_data[list_entry.offset + j];
        }

        auto y_idx = y_data.sel->get_index(i);
        bool y_valid = y_data.validity.RowIsValid(y_idx);
        double y_val = y_valid ? y_values[y_idx] : std::nan("");

        // Determine if this row is for training
        bool row_is_training;
        if (bind_data.use_split_col && split_values) {
            auto split_idx = split_data.sel->get_index(i);
            if (split_data.validity.RowIsValid(split_idx)) {
                row_is_training = RidgeIsSplitTraining(split_values[split_idx]);
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

static void RidgePredictAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (RidgePredictAggState **)source_data.data;
    auto targets = (RidgePredictAggState **)target_data.data;

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
            target.alpha = source.alpha;
            target.fit_intercept = source.fit_intercept;
            target.confidence_level = source.confidence_level;
            target.null_policy = source.null_policy;
            target.use_split_col = source.use_split_col;
            continue;
        }

        if (source.n_features != target.n_features) {
            throw InvalidInputException("Cannot combine states with different feature counts");
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

static void RidgePredictAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                     idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (RidgePredictAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.y_train.size() < 2) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Note: Detailed min_obs validation including zero-variance column handling is done in Rust
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

        AnofoxRidgeOptions options;
        options.alpha = state.alpha;
        options.fit_intercept = state.fit_intercept;
        options.compute_inference = false;
        options.confidence_level = state.confidence_level;

        AnofoxFitResultCore core_result;
        AnofoxError error;

        bool success = anofox_ridge_fit(y_array, x_arrays.data(), x_arrays.size(), options, &core_result, nullptr, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t n_rows = state.y_all.size();
        auto list_data = ListVector::GetData(result);
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
                core_result.coefficients, core_result.coefficients_len, core_result.intercept, state.x_all[row].data(),
                state.n_features, core_result.residual_std_error, core_result.n_observations, state.confidence_level,
                &pred);

            if (pred_success) {
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

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> RidgePredictAggBind(ClientContext &context, AggregateFunction &function,
                                                     vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<RidgePredictAggBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[2]);
        if (opts.alpha.has_value()) {
            result->alpha = opts.alpha.value();
        }
        if (opts.fit_intercept.has_value()) {
            result->fit_intercept = opts.fit_intercept.value();
        }
        if (opts.confidence_level.has_value()) {
            result->confidence_level = opts.confidence_level.value();
        }
        if (opts.null_policy.has_value()) {
            result->null_policy = opts.null_policy.value();
        }
    }

    function.return_type = GetRidgePredictAggResultType();
    PostHogTelemetry::Instance().CaptureFunctionExecution("ridge_fit_predict_agg");
    return std::move(result);
}

// Bind function for versions with split column
static unique_ptr<FunctionData> RidgePredictAggBindWithSplit(ClientContext &context, AggregateFunction &function,
                                                              vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<RidgePredictAggBindData>();
    result->use_split_col = true;

    // Parse MAP options if provided as 4th argument
    if (arguments.size() >= 4 && arguments[3]->IsFoldable()) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[3]);
        if (opts.alpha.has_value()) {
            result->alpha = opts.alpha.value();
        }
        if (opts.fit_intercept.has_value()) {
            result->fit_intercept = opts.fit_intercept.value();
        }
        if (opts.confidence_level.has_value()) {
            result->confidence_level = opts.confidence_level.value();
        }
        if (opts.null_policy.has_value()) {
            result->null_policy = opts.null_policy.value();
        }
    }

    function.return_type = GetRidgePredictAggResultType();
    PostHogTelemetry::Instance().CaptureFunctionExecution("ridge_fit_predict_agg");
    return std::move(result);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterRidgeFitPredictAggregateFunction(ExtensionLoader &loader) {
    // Primary name (new)
    AggregateFunctionSet func_set("anofox_stats_ridge_fit_predict_agg");

    auto basic_func = AggregateFunction(
        "anofox_stats_ridge_fit_predict_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)},
        LogicalType::ANY, AggregateFunction::StateSize<RidgePredictAggState>, RidgePredictAggInitialize,
        RidgePredictAggUpdate, RidgePredictAggCombine, RidgePredictAggFinalize, nullptr, RidgePredictAggBind,
        RidgePredictAggDestroy);
    func_set.AddFunction(basic_func);

    auto map_func = AggregateFunction(
        "anofox_stats_ridge_fit_predict_agg",
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY}, LogicalType::ANY,
        AggregateFunction::StateSize<RidgePredictAggState>, RidgePredictAggInitialize, RidgePredictAggUpdate,
        RidgePredictAggCombine, RidgePredictAggFinalize, nullptr, RidgePredictAggBind, RidgePredictAggDestroy);
    func_set.AddFunction(map_func);

    // Version with split column: ridge_fit_predict_agg(y, x, split_col)
    auto split_func = AggregateFunction(
        "anofox_stats_ridge_fit_predict_agg",
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::VARCHAR}, LogicalType::ANY,
        AggregateFunction::StateSize<RidgePredictAggState>, RidgePredictAggInitialize, RidgePredictAggUpdate,
        RidgePredictAggCombine, RidgePredictAggFinalize, nullptr, RidgePredictAggBindWithSplit, RidgePredictAggDestroy);
    func_set.AddFunction(split_func);

    // Version with split column and options: ridge_fit_predict_agg(y, x, split_col, options)
    auto split_opts_func = AggregateFunction(
        "anofox_stats_ridge_fit_predict_agg",
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::VARCHAR, LogicalType::ANY},
        LogicalType::ANY, AggregateFunction::StateSize<RidgePredictAggState>, RidgePredictAggInitialize,
        RidgePredictAggUpdate, RidgePredictAggCombine, RidgePredictAggFinalize, nullptr, RidgePredictAggBindWithSplit,
        RidgePredictAggDestroy);
    func_set.AddFunction(split_opts_func);

    loader.RegisterFunction(func_set);

    // Short alias (new)
    AggregateFunctionSet alias_set("ridge_fit_predict_agg");
    alias_set.AddFunction(basic_func);
    alias_set.AddFunction(map_func);
    alias_set.AddFunction(split_func);
    alias_set.AddFunction(split_opts_func);
    loader.RegisterFunction(alias_set);

    // Deprecated aliases (old names for backwards compatibility)
    AggregateFunctionSet deprecated_set("ridge_predict_agg");
    deprecated_set.AddFunction(basic_func);
    deprecated_set.AddFunction(map_func);
    deprecated_set.AddFunction(split_func);
    deprecated_set.AddFunction(split_opts_func);
    loader.RegisterFunction(deprecated_set);

    AggregateFunctionSet deprecated_full_set("anofox_stats_ridge_predict_agg");
    deprecated_full_set.AddFunction(basic_func);
    deprecated_full_set.AddFunction(map_func);
    deprecated_full_set.AddFunction(split_func);
    deprecated_full_set.AddFunction(split_opts_func);
    loader.RegisterFunction(deprecated_full_set);
}

} // namespace duckdb
