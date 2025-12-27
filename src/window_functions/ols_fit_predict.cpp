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
// OLS Fit-Predict State
// Accumulates training data and stores current row's x for prediction
//===--------------------------------------------------------------------===//
struct OlsFitPredictState {
    // Training data (rows where y IS NOT NULL)
    vector<double> y_values;
    vector<vector<double>> x_columns;
    idx_t n_features;
    bool initialized;

    // Current row's x values for prediction (stored from last Update call)
    vector<double> current_x;
    bool has_current_x;

    // Options
    bool fit_intercept;
    double confidence_level;
    NullPolicy null_policy;

    OlsFitPredictState()
        : n_features(0), initialized(false), has_current_x(false), fit_intercept(true), confidence_level(0.95),
          null_policy(NullPolicy::DROP) {}

    void Reset() {
        y_values.clear();
        x_columns.clear();
        current_x.clear();
        n_features = 0;
        initialized = false;
        has_current_x = false;
    }
};

//===--------------------------------------------------------------------===//
// Bind Data for options
//===--------------------------------------------------------------------===//
struct OlsFitPredictBindData : public FunctionData {
    bool fit_intercept = true;
    double confidence_level = 0.95;
    NullPolicy null_policy = NullPolicy::DROP;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<OlsFitPredictBindData>();
        result->fit_intercept = fit_intercept;
        result->confidence_level = confidence_level;
        result->null_policy = null_policy;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<OlsFitPredictBindData>();
        return fit_intercept == other.fit_intercept && confidence_level == other.confidence_level &&
               null_policy == other.null_policy;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetOlsFitPredictResultType() {
    child_list_t<LogicalType> children;
    children.push_back(make_pair("yhat", LogicalType::DOUBLE));
    children.push_back(make_pair("yhat_lower", LogicalType::DOUBLE));
    children.push_back(make_pair("yhat_upper", LogicalType::DOUBLE));
    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void OlsFitPredictInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) OlsFitPredictState();
}

static void OlsFitPredictDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (OlsFitPredictState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~OlsFitPredictState();
    }
}

// Update: accumulate training data and store current x
static void OlsFitPredictUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                 Vector &state_vector, idx_t count) {
    auto &bind_data = aggr_input_data.bind_data->Cast<OlsFitPredictBindData>();

    UnifiedVectorFormat y_data;
    UnifiedVectorFormat x_data;
    inputs[0].ToUnifiedFormat(count, y_data); // y: DOUBLE (nullable)
    inputs[1].ToUnifiedFormat(count, x_data); // x: LIST(DOUBLE)

    auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
    auto x_list_data = ListVector::GetData(inputs[1]);
    auto &x_child = ListVector::GetEntry(inputs[1]);
    auto x_child_data = FlatVector::GetData<double>(x_child);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (OlsFitPredictState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];

        // Copy options
        state.fit_intercept = bind_data.fit_intercept;
        state.confidence_level = bind_data.confidence_level;
        state.null_policy = bind_data.null_policy;

        // Get x values
        auto x_idx = x_data.sel->get_index(i);
        if (!x_data.validity.RowIsValid(x_idx)) {
            state.has_current_x = false;
            continue;
        }

        auto list_entry = x_list_data[x_idx];
        idx_t n_features = list_entry.length;

        // Initialize on first valid row
        if (!state.initialized) {
            state.n_features = n_features;
            state.x_columns.resize(n_features);
            state.initialized = true;
        }

        if (n_features != state.n_features) {
            throw InvalidInputException("Inconsistent feature count: expected %lu, got %lu", state.n_features,
                                        n_features);
        }

        // Store current x for prediction
        state.current_x.resize(n_features);
        for (idx_t j = 0; j < n_features; j++) {
            state.current_x[j] = x_child_data[list_entry.offset + j];
        }
        state.has_current_x = true;

        // Determine if this row should be used for training
        auto y_idx = y_data.sel->get_index(i);
        bool y_valid = y_data.validity.RowIsValid(y_idx);
        bool use_for_training = y_valid;

        // Apply null_policy for drop_y_zero_x
        if (use_for_training && state.null_policy == NullPolicy::DROP_Y_ZERO_X) {
            for (idx_t j = 0; j < n_features; j++) {
                if (state.current_x[j] == 0.0) {
                    use_for_training = false;
                    break;
                }
            }
        }

        // Add to training data if valid
        if (use_for_training) {
            double y_val = y_values[y_idx];
            state.y_values.push_back(y_val);

            for (idx_t j = 0; j < n_features; j++) {
                state.x_columns[j].push_back(x_child_data[list_entry.offset + j]);
            }
        }
    }
}

// Combine: merge two states
static void OlsFitPredictCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (OlsFitPredictState **)source_data.data;
    auto targets = (OlsFitPredictState **)target_data.data;

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

        // Keep current_x from source if it has one
        if (source.has_current_x) {
            target.current_x = std::move(source.current_x);
            target.has_current_x = true;
        }
    }
}

// Finalize: fit model and make prediction
static void OlsFitPredictFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count,
                                   idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (OlsFitPredictState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        // Check if we can make a prediction
        if (!state.initialized || !state.has_current_x) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Need minimum data to fit
        idx_t min_obs = state.fit_intercept ? state.n_features + 1 : state.n_features;
        if (state.y_values.size() <= min_obs) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Fit model
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

        AnofoxOlsOptions options;
        options.fit_intercept = state.fit_intercept;
        options.compute_inference = false;
        options.confidence_level = state.confidence_level;

        AnofoxFitResultCore core_result;
        AnofoxError error;

        bool success = anofox_ols_fit(y_array, x_arrays.data(), x_arrays.size(), options, &core_result, nullptr, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Make prediction with interval
        AnofoxPredictionResult pred_result;
        bool pred_success =
            anofox_predict_with_interval(core_result.coefficients, core_result.coefficients_len, core_result.intercept,
                                         state.current_x.data(), state.current_x.size(), core_result.residual_std_error,
                                         core_result.n_observations, state.confidence_level, &pred_result);

        anofox_free_result_core(&core_result);

        if (!pred_success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Fill result
        FlatVector::GetData<double>(*struct_entries[0])[result_idx] = pred_result.yhat;
        FlatVector::GetData<double>(*struct_entries[1])[result_idx] = pred_result.yhat_lower;
        FlatVector::GetData<double>(*struct_entries[2])[result_idx] = pred_result.yhat_upper;

        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> OlsFitPredictBind(ClientContext &context, AggregateFunction &function,
                                                   vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<OlsFitPredictBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[2]);
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

    function.return_type = GetOlsFitPredictResultType();
    PostHogTelemetry::Instance().CaptureFunctionExecution("ols_fit_predict");
    return std::move(result);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterOlsFitPredictFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_ols_fit_predict");

    auto basic_func =
        AggregateFunction("anofox_stats_ols_fit_predict",
                          {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)}, GetOlsFitPredictResultType(),
                          AggregateFunction::StateSize<OlsFitPredictState>, OlsFitPredictInitialize,
                          OlsFitPredictUpdate, OlsFitPredictCombine, OlsFitPredictFinalize,
                          nullptr, // simple_update
                          OlsFitPredictBind, OlsFitPredictDestroy);
    func_set.AddFunction(basic_func);

    auto map_func = AggregateFunction("anofox_stats_ols_fit_predict",
                                      {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY},
                                      GetOlsFitPredictResultType(), AggregateFunction::StateSize<OlsFitPredictState>,
                                      OlsFitPredictInitialize, OlsFitPredictUpdate, OlsFitPredictCombine,
                                      OlsFitPredictFinalize, nullptr, OlsFitPredictBind, OlsFitPredictDestroy);
    func_set.AddFunction(map_func);

    loader.RegisterFunction(func_set);

    // Register short alias
    AggregateFunctionSet alias_set("ols_fit_predict");
    alias_set.AddFunction(basic_func);
    alias_set.AddFunction(map_func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
