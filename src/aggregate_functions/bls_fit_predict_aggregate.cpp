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
// BLS Fit Predict Aggregate State
// Accumulates ALL rows for output, tracks which are training vs prediction
//===--------------------------------------------------------------------===//
struct BlsFitPredictAggState {
    // Training data (rows where y is NOT NULL)
    vector<double> y_train;
    vector<vector<double>> x_train;

    // ALL rows' data for output
    vector<double> y_all;       // Original y values (NaN for NULL)
    vector<bool> y_is_null;     // True if y was NULL in input
    vector<bool> is_training;   // True if row was used for training
    vector<vector<double>> x_all;  // x values for ALL rows

    idx_t n_features;
    bool initialized;

    // Options
    bool fit_intercept;
    double lower_bound;
    double upper_bound;
    bool has_lower_bound;
    bool has_upper_bound;
    uint32_t max_iterations;
    double tolerance;
    double confidence_level;
    NullPolicy null_policy;
    bool use_split_col;

    BlsFitPredictAggState()
        : n_features(0), initialized(false), fit_intercept(false), lower_bound(0.0), upper_bound(0.0),
          has_lower_bound(false), has_upper_bound(false), max_iterations(1000), tolerance(1e-10),
          confidence_level(0.95), null_policy(NullPolicy::DROP), use_split_col(false) {}

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
struct BlsFitPredictAggBindData : public FunctionData {
    bool fit_intercept = false;
    double lower_bound = 0.0;
    double upper_bound = 0.0;
    bool has_lower_bound = false;
    bool has_upper_bound = false;
    uint32_t max_iterations = 1000;
    double tolerance = 1e-10;
    double confidence_level = 0.95;
    NullPolicy null_policy = NullPolicy::DROP;
    bool use_split_col = false;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<BlsFitPredictAggBindData>();
        result->fit_intercept = fit_intercept;
        result->lower_bound = lower_bound;
        result->upper_bound = upper_bound;
        result->has_lower_bound = has_lower_bound;
        result->has_upper_bound = has_upper_bound;
        result->max_iterations = max_iterations;
        result->tolerance = tolerance;
        result->confidence_level = confidence_level;
        result->null_policy = null_policy;
        result->use_split_col = use_split_col;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<BlsFitPredictAggBindData>();
        return fit_intercept == other.fit_intercept && lower_bound == other.lower_bound &&
               upper_bound == other.upper_bound && has_lower_bound == other.has_lower_bound &&
               has_upper_bound == other.has_upper_bound && max_iterations == other.max_iterations &&
               tolerance == other.tolerance && confidence_level == other.confidence_level &&
               null_policy == other.null_policy && use_split_col == other.use_split_col;
    }
};

//===--------------------------------------------------------------------===//
// Result type: LIST(STRUCT(y, yhat, yhat_lower, yhat_upper, is_training))
//===--------------------------------------------------------------------===//
static LogicalType GetBlsFitPredictAggResultType() {
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

static void BlsFitPredictAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) BlsFitPredictAggState();
}

static void BlsFitPredictAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (BlsFitPredictAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~BlsFitPredictAggState();
    }
}

static bool BlsIsSplitTraining(const string_t &split_val) {
    string val = split_val.GetString();
    for (auto &c : val) {
        c = std::tolower(c);
    }
    return val == "train" || val == "training";
}

// Update: accumulate ALL rows, track which are training vs prediction
static void BlsFitPredictAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                 Vector &state_vector, idx_t count) {
    auto &bind_data = aggr_input_data.bind_data->Cast<BlsFitPredictAggBindData>();

    UnifiedVectorFormat y_data;
    UnifiedVectorFormat x_data;
    inputs[0].ToUnifiedFormat(count, y_data);
    inputs[1].ToUnifiedFormat(count, x_data);

    auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
    auto x_list_data = ListVector::GetData(inputs[1]);
    auto &x_child = ListVector::GetEntry(inputs[1]);
    auto x_child_data = FlatVector::GetData<double>(x_child);

    UnifiedVectorFormat split_data;
    const string_t *split_values = nullptr;
    if (bind_data.use_split_col && input_count >= 3) {
        inputs[2].ToUnifiedFormat(count, split_data);
        split_values = UnifiedVectorFormat::GetData<string_t>(split_data);
    }

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (BlsFitPredictAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];

        // Copy options from bind data
        state.fit_intercept = bind_data.fit_intercept;
        state.lower_bound = bind_data.lower_bound;
        state.upper_bound = bind_data.upper_bound;
        state.has_lower_bound = bind_data.has_lower_bound;
        state.has_upper_bound = bind_data.has_upper_bound;
        state.max_iterations = bind_data.max_iterations;
        state.tolerance = bind_data.tolerance;
        state.confidence_level = bind_data.confidence_level;
        state.null_policy = bind_data.null_policy;
        state.use_split_col = bind_data.use_split_col;

        // Get x values
        auto x_idx = x_data.sel->get_index(i);
        if (!x_data.validity.RowIsValid(x_idx)) {
            continue; // Skip rows with NULL x
        }

        auto list_entry = x_list_data[x_idx];
        idx_t n_features = list_entry.length;

        // Initialize on first valid row
        if (!state.initialized) {
            state.n_features = n_features;
            state.x_train.resize(n_features);
            state.initialized = true;
        }

        // Validate consistent feature count
        if (n_features != state.n_features) {
            throw InvalidInputException("Inconsistent feature count: expected %lu, got %lu", state.n_features,
                                        n_features);
        }

        // Extract x values for this row
        vector<double> x_row(n_features);
        for (idx_t j = 0; j < n_features; j++) {
            x_row[j] = x_child_data[list_entry.offset + j];
        }

        // Check y validity
        auto y_idx = y_data.sel->get_index(i);
        bool y_valid = y_data.validity.RowIsValid(y_idx);
        double y_val = y_valid ? y_values[y_idx] : std::nan("");

        // Determine if this row is for training
        bool row_is_training;
        if (bind_data.use_split_col && split_values) {
            auto split_idx = split_data.sel->get_index(i);
            if (split_data.validity.RowIsValid(split_idx)) {
                row_is_training = BlsIsSplitTraining(split_values[split_idx]);
            } else {
                row_is_training = false;
            }
            if (row_is_training && !y_valid) {
                row_is_training = false;
            }
        } else {
            row_is_training = y_valid;
        }

        // Apply null_policy for drop_y_zero_x
        if (row_is_training && state.null_policy == NullPolicy::DROP_Y_ZERO_X) {
            for (idx_t j = 0; j < n_features; j++) {
                if (x_row[j] == 0.0) {
                    row_is_training = false;
                    break;
                }
            }
        }

        // Store ALL row data for output (including training flag)
        state.y_all.push_back(y_val);
        state.y_is_null.push_back(!y_valid);
        state.is_training.push_back(row_is_training);
        state.x_all.push_back(x_row);

        // Add to training data if training row
        if (row_is_training) {
            state.y_train.push_back(y_val);
            for (idx_t j = 0; j < n_features; j++) {
                state.x_train[j].push_back(x_row[j]);
            }
        }
    }
}

// Combine: merge two states
static void BlsFitPredictAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (BlsFitPredictAggState **)source_data.data;
    auto targets = (BlsFitPredictAggState **)target_data.data;

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
            target.lower_bound = source.lower_bound;
            target.upper_bound = source.upper_bound;
            target.has_lower_bound = source.has_lower_bound;
            target.has_upper_bound = source.has_upper_bound;
            target.max_iterations = source.max_iterations;
            target.tolerance = source.tolerance;
            target.confidence_level = source.confidence_level;
            target.null_policy = source.null_policy;
            target.use_split_col = source.use_split_col;
            continue;
        }

        if (source.n_features != target.n_features) {
            throw InvalidInputException("Cannot combine states with different feature counts: %lu vs %lu",
                                        source.n_features, target.n_features);
        }

        // Merge training data
        target.y_train.insert(target.y_train.end(), source.y_train.begin(), source.y_train.end());
        for (idx_t j = 0; j < target.n_features; j++) {
            target.x_train[j].insert(target.x_train[j].end(), source.x_train[j].begin(), source.x_train[j].end());
        }

        // Merge all data
        target.y_all.insert(target.y_all.end(), source.y_all.begin(), source.y_all.end());
        target.y_is_null.insert(target.y_is_null.end(), source.y_is_null.begin(), source.y_is_null.end());
        target.is_training.insert(target.is_training.end(), source.is_training.begin(), source.is_training.end());
        target.x_all.insert(target.x_all.end(), source.x_all.begin(), source.x_all.end());
    }
}

// Finalize: fit model and return predictions for all rows
static void BlsFitPredictAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                   idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (BlsFitPredictAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        // Check if we have enough training data
        if (!state.initialized || state.y_train.size() < 2) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Note: Detailed min_obs validation including zero-variance column handling is done in Rust
        // Prepare FFI data for fitting
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

        AnofoxBlsOptions options;
        options.fit_intercept = state.fit_intercept;
        options.max_iterations = state.max_iterations;
        options.tolerance = state.tolerance;

        // Set bounds
        if (state.has_lower_bound) {
            options.lower_bounds = &state.lower_bound;
            options.lower_bounds_len = 1;
        } else {
            options.lower_bounds = nullptr;
            options.lower_bounds_len = 0;
        }

        if (state.has_upper_bound) {
            options.upper_bounds = &state.upper_bound;
            options.upper_bounds_len = 1;
        } else {
            options.upper_bounds = nullptr;
            options.upper_bounds_len = 0;
        }

        AnofoxBlsFitResultCore core_result;
        AnofoxError error;

        bool success = anofox_bls_fit(y_array, x_arrays.data(), x_arrays.size(), options, &core_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Calculate residual_std_error from SSR for prediction intervals
        double residual_std_error = std::nan("");
        size_t df = core_result.n_observations - core_result.n_features - (state.fit_intercept ? 1 : 0);
        if (df > 0 && core_result.ssr >= 0) {
            residual_std_error = std::sqrt(core_result.ssr / df);
        }

        // Build LIST result with predictions for ALL rows
        idx_t n_rows = state.y_all.size();
        auto *list_data = ListVector::GetData(result);
        auto list_offset = ListVector::GetListSize(result);

        list_data[result_idx].offset = list_offset;
        list_data[result_idx].length = n_rows;

        ListVector::Reserve(result, list_offset + n_rows);
        ListVector::SetListSize(result, list_offset + n_rows);

        auto &child_struct = ListVector::GetEntry(result);
        auto &struct_entries = StructVector::GetEntries(child_struct);

        // struct_entries: [y, yhat, yhat_lower, yhat_upper, is_training]
        auto &y_vec = *struct_entries[0];
        auto &yhat_vec = *struct_entries[1];
        auto &yhat_lower_vec = *struct_entries[2];
        auto &yhat_upper_vec = *struct_entries[3];
        auto &is_training_vec = *struct_entries[4];

        for (idx_t row = 0; row < n_rows; row++) {
            idx_t child_idx = list_offset + row;

            // Set y (NULL if it was NULL in input)
            if (state.y_is_null[row]) {
                FlatVector::SetNull(y_vec, child_idx, true);
            } else {
                FlatVector::GetData<double>(y_vec)[child_idx] = state.y_all[row];
            }

            // Compute prediction for this row
            AnofoxPredictionResult pred;
            bool pred_success = anofox_predict_with_interval(
                core_result.coefficients, core_result.coefficients_len, core_result.intercept, state.x_all[row].data(),
                state.n_features, residual_std_error, core_result.n_observations, state.confidence_level,
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

            // Set is_training flag
            FlatVector::GetData<bool>(is_training_vec)[child_idx] = state.is_training[row];
        }

        anofox_free_bls_result(&core_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> BlsFitPredictAggBind(ClientContext &context, AggregateFunction &function,
                                                   vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<BlsFitPredictAggBindData>();

    // Parse MAP options if provided as 3rd argument
    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[2]);
        if (opts.fit_intercept.has_value()) {
            result->fit_intercept = opts.fit_intercept.value();
        }
        if (opts.lower_bound.has_value()) {
            result->lower_bound = opts.lower_bound.value();
            result->has_lower_bound = true;
        }
        if (opts.upper_bound.has_value()) {
            result->upper_bound = opts.upper_bound.value();
            result->has_upper_bound = true;
        }
        if (opts.max_iterations.has_value()) {
            result->max_iterations = opts.max_iterations.value();
        }
        if (opts.tolerance.has_value()) {
            result->tolerance = opts.tolerance.value();
        }
        if (opts.confidence_level.has_value()) {
            result->confidence_level = opts.confidence_level.value();
        }
        if (opts.null_policy.has_value()) {
            result->null_policy = opts.null_policy.value();
        }
    }

    function.return_type = GetBlsFitPredictAggResultType();
    PostHogTelemetry::Instance().CaptureFunctionExecution("bls_fit_predict_agg");
    return std::move(result);
}

static unique_ptr<FunctionData> BlsFitPredictAggBindWithSplit(ClientContext &context, AggregateFunction &function,
                                                               vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<BlsFitPredictAggBindData>();
    result->use_split_col = true;

    if (arguments.size() >= 4 && arguments[3]->IsFoldable()) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[3]);
        if (opts.fit_intercept.has_value()) {
            result->fit_intercept = opts.fit_intercept.value();
        }
        if (opts.lower_bound.has_value()) {
            result->lower_bound = opts.lower_bound.value();
            result->has_lower_bound = true;
        }
        if (opts.upper_bound.has_value()) {
            result->upper_bound = opts.upper_bound.value();
            result->has_upper_bound = true;
        }
        if (opts.max_iterations.has_value()) {
            result->max_iterations = opts.max_iterations.value();
        }
        if (opts.tolerance.has_value()) {
            result->tolerance = opts.tolerance.value();
        }
        if (opts.confidence_level.has_value()) {
            result->confidence_level = opts.confidence_level.value();
        }
        if (opts.null_policy.has_value()) {
            result->null_policy = opts.null_policy.value();
        }
    }

    function.return_type = GetBlsFitPredictAggResultType();
    PostHogTelemetry::Instance().CaptureFunctionExecution("bls_fit_predict_agg");
    return std::move(result);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterBlsFitPredictAggregateFunction(ExtensionLoader &loader) {
    // Primary name
    AggregateFunctionSet func_set("anofox_stats_bls_fit_predict_agg");

    // Basic version: bls_fit_predict_agg(y, x)
    auto basic_func =
        AggregateFunction("anofox_stats_bls_fit_predict_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)},
                          LogicalType::ANY, AggregateFunction::StateSize<BlsFitPredictAggState>, BlsFitPredictAggInitialize,
                          BlsFitPredictAggUpdate, BlsFitPredictAggCombine, BlsFitPredictAggFinalize, nullptr, BlsFitPredictAggBind,
                          BlsFitPredictAggDestroy);
    func_set.AddFunction(basic_func);

    // Version with MAP options: bls_fit_predict_agg(y, x, {'null_policy': 'drop', ...})
    auto map_func = AggregateFunction(
        "anofox_stats_bls_fit_predict_agg",
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY}, LogicalType::ANY,
        AggregateFunction::StateSize<BlsFitPredictAggState>, BlsFitPredictAggInitialize, BlsFitPredictAggUpdate,
        BlsFitPredictAggCombine, BlsFitPredictAggFinalize, nullptr, BlsFitPredictAggBind, BlsFitPredictAggDestroy);
    func_set.AddFunction(map_func);

    auto split_func = AggregateFunction(
        "anofox_stats_bls_fit_predict_agg",
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::VARCHAR}, LogicalType::ANY,
        AggregateFunction::StateSize<BlsFitPredictAggState>, BlsFitPredictAggInitialize, BlsFitPredictAggUpdate,
        BlsFitPredictAggCombine, BlsFitPredictAggFinalize, nullptr, BlsFitPredictAggBindWithSplit, BlsFitPredictAggDestroy);
    func_set.AddFunction(split_func);

    auto split_opts_func = AggregateFunction(
        "anofox_stats_bls_fit_predict_agg",
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::VARCHAR, LogicalType::ANY},
        LogicalType::ANY, AggregateFunction::StateSize<BlsFitPredictAggState>, BlsFitPredictAggInitialize,
        BlsFitPredictAggUpdate, BlsFitPredictAggCombine, BlsFitPredictAggFinalize, nullptr, BlsFitPredictAggBindWithSplit,
        BlsFitPredictAggDestroy);
    func_set.AddFunction(split_opts_func);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("bls_fit_predict_agg");
    alias_set.AddFunction(basic_func);
    alias_set.AddFunction(map_func);
    alias_set.AddFunction(split_func);
    alias_set.AddFunction(split_opts_func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
