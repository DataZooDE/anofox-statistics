#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"

namespace duckdb {

//===--------------------------------------------------------------------===//
// WLS Aggregate State - accumulates y, x, and weight values for each group
//===--------------------------------------------------------------------===//
struct WlsAggregateState {
    vector<double> y_values;
    vector<vector<double>> x_columns;
    vector<double> weights;
    idx_t n_features;
    bool initialized;

    // Options
    bool fit_intercept;
    bool compute_inference;
    double confidence_level;

    WlsAggregateState()
        : n_features(0), initialized(false), fit_intercept(true), compute_inference(false), confidence_level(0.95) {}

    void Reset() {
        y_values.clear();
        x_columns.clear();
        weights.clear();
        n_features = 0;
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Bind Data for options
//===--------------------------------------------------------------------===//
struct WlsAggregateBindData : public FunctionData {
    bool fit_intercept = true;
    bool compute_inference = false;
    double confidence_level = 0.95;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<WlsAggregateBindData>();
        result->fit_intercept = fit_intercept;
        result->compute_inference = compute_inference;
        result->confidence_level = confidence_level;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<WlsAggregateBindData>();
        return fit_intercept == other.fit_intercept && compute_inference == other.compute_inference &&
               confidence_level == other.confidence_level;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetWlsAggResultType(bool compute_inference) {
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

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

// Initialize aggregate state
static void WlsAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) WlsAggregateState();
}

// Destroy aggregate state
static void WlsAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (WlsAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~WlsAggregateState();
    }
}

// Update: accumulate values from input rows
static void WlsAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &state_vector,
                         idx_t count) {
    auto &bind_data = aggr_input_data.bind_data->Cast<WlsAggregateBindData>();

    UnifiedVectorFormat y_data;
    UnifiedVectorFormat x_data;
    UnifiedVectorFormat w_data;
    inputs[0].ToUnifiedFormat(count, y_data); // y: DOUBLE
    inputs[1].ToUnifiedFormat(count, x_data); // x: LIST(DOUBLE)
    inputs[2].ToUnifiedFormat(count, w_data); // weight: DOUBLE

    auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
    auto w_values = UnifiedVectorFormat::GetData<double>(w_data);
    auto x_list_data = ListVector::GetData(inputs[1]);
    auto &x_child = ListVector::GetEntry(inputs[1]);
    auto x_child_data = FlatVector::GetData<double>(x_child);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (WlsAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];

        // Copy options from bind data
        state.fit_intercept = bind_data.fit_intercept;
        state.compute_inference = bind_data.compute_inference;
        state.confidence_level = bind_data.confidence_level;

        // Get y value
        auto y_idx = y_data.sel->get_index(i);
        if (!y_data.validity.RowIsValid(y_idx)) {
            continue; // Skip NULL y values
        }
        double y_val = y_values[y_idx];

        // Get weight value
        auto w_idx = w_data.sel->get_index(i);
        if (!w_data.validity.RowIsValid(w_idx)) {
            continue; // Skip NULL weight values
        }
        double w_val = w_values[w_idx];

        // Get x values (LIST(DOUBLE))
        auto x_idx = x_data.sel->get_index(i);
        if (!x_data.validity.RowIsValid(x_idx)) {
            continue; // Skip NULL x values
        }

        auto list_entry = x_list_data[x_idx];
        idx_t n_features = list_entry.length;

        // Initialize x_columns on first valid row
        if (!state.initialized) {
            state.n_features = n_features;
            state.x_columns.resize(n_features);
            state.initialized = true;
        }

        // Validate consistent feature count
        if (n_features != state.n_features) {
            throw InvalidInputException("Inconsistent feature count: expected %lu, got %lu", state.n_features,
                                        n_features);
        }

        // Accumulate y value
        state.y_values.push_back(y_val);

        // Accumulate weight value
        state.weights.push_back(w_val);

        // Accumulate x values
        for (idx_t j = 0; j < n_features; j++) {
            double x_val = x_child_data[list_entry.offset + j];
            state.x_columns[j].push_back(x_val);
        }
    }
}

// Combine: merge two states
static void WlsAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (WlsAggregateState **)source_data.data;
    auto targets = (WlsAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue; // Nothing to combine
        }

        if (!target.initialized) {
            // Copy source to target
            target.y_values = std::move(source.y_values);
            target.x_columns = std::move(source.x_columns);
            target.weights = std::move(source.weights);
            target.n_features = source.n_features;
            target.initialized = true;
            target.fit_intercept = source.fit_intercept;
            target.compute_inference = source.compute_inference;
            target.confidence_level = source.confidence_level;
            continue;
        }

        // Validate same feature count
        if (source.n_features != target.n_features) {
            throw InvalidInputException("Cannot combine states with different feature counts: %lu vs %lu",
                                        source.n_features, target.n_features);
        }

        // Merge y values
        target.y_values.insert(target.y_values.end(), source.y_values.begin(), source.y_values.end());

        // Merge weights
        target.weights.insert(target.weights.end(), source.weights.begin(), source.weights.end());

        // Merge x columns
        for (idx_t j = 0; j < target.n_features; j++) {
            target.x_columns[j].insert(target.x_columns[j].end(), source.x_columns[j].begin(),
                                       source.x_columns[j].end());
        }
    }
}

// Helper to set a list in STRUCT result
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

// Finalize: compute WLS for accumulated data
static void WlsAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                           idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (WlsAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        // Check if we have enough data
        if (!state.initialized || state.y_values.size() < 2) {
            // Set NULL result
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Minimum observations check
        idx_t min_obs = state.fit_intercept ? state.n_features + 1 : state.n_features;
        if (state.y_values.size() <= min_obs) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Prepare FFI data
        AnofoxDataArray y_array;
        y_array.data = state.y_values.data();
        y_array.validity = nullptr;
        y_array.len = state.y_values.size();

        AnofoxDataArray weights_array;
        weights_array.data = state.weights.data();
        weights_array.validity = nullptr;
        weights_array.len = state.weights.size();

        vector<AnofoxDataArray> x_arrays;
        for (auto &col : state.x_columns) {
            AnofoxDataArray arr;
            arr.data = col.data();
            arr.validity = nullptr;
            arr.len = col.size();
            x_arrays.push_back(arr);
        }

        AnofoxWlsOptions options;
        options.fit_intercept = state.fit_intercept;
        options.compute_inference = state.compute_inference;
        options.confidence_level = state.confidence_level;

        AnofoxFitResultCore core_result;
        AnofoxFitResultInference inference_result;
        AnofoxError error;

        bool success = anofox_wls_fit(y_array, x_arrays.data(), x_arrays.size(), weights_array, options, &core_result,
                                      state.compute_inference ? &inference_result : nullptr, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Fill STRUCT result
        idx_t struct_idx = 0;

        // Coefficients
        SetListInResult(*struct_entries[struct_idx++], result_idx, core_result.coefficients,
                        core_result.coefficients_len);

        // Scalars
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.intercept;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.r_squared;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.adj_r_squared;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.residual_std_error;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = core_result.n_observations;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = core_result.n_features;

        // Inference results
        if (state.compute_inference) {
            SetListInResult(*struct_entries[struct_idx++], result_idx, inference_result.std_errors,
                            inference_result.len);
            SetListInResult(*struct_entries[struct_idx++], result_idx, inference_result.t_values, inference_result.len);
            SetListInResult(*struct_entries[struct_idx++], result_idx, inference_result.p_values, inference_result.len);
            SetListInResult(*struct_entries[struct_idx++], result_idx, inference_result.ci_lower, inference_result.len);
            SetListInResult(*struct_entries[struct_idx++], result_idx, inference_result.ci_upper, inference_result.len);

            FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = inference_result.f_statistic;
            FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = inference_result.f_pvalue;

            anofox_free_result_inference(&inference_result);
        }

        anofox_free_result_core(&core_result);

        // Reset state for next use
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> WlsAggBind(ClientContext &context, AggregateFunction &function,
                                           vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<WlsAggregateBindData>();

    // Extract options if provided (4th, 5th, 6th arguments)
    if (arguments.size() >= 4 && arguments[3]->IsFoldable()) {
        result->fit_intercept = BooleanValue::Get(ExpressionExecutor::EvaluateScalar(context, *arguments[3]));
    }
    if (arguments.size() >= 5 && arguments[4]->IsFoldable()) {
        result->compute_inference = BooleanValue::Get(ExpressionExecutor::EvaluateScalar(context, *arguments[4]));
    }
    if (arguments.size() >= 6 && arguments[5]->IsFoldable()) {
        result->confidence_level = DoubleValue::Get(ExpressionExecutor::EvaluateScalar(context, *arguments[5]));
    }

    // Set return type based on options
    function.return_type = GetWlsAggResultType(result->compute_inference);

    return std::move(result);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterWlsAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_wls_fit_agg");

    // Basic version: anofox_stats_wls_fit_agg(y, x, weight)
    auto basic_func = AggregateFunction(
        "anofox_stats_wls_fit_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE},
        LogicalType::ANY, // Set in bind
        AggregateFunction::StateSize<WlsAggregateState>, WlsAggInitialize, WlsAggUpdate, WlsAggCombine, WlsAggFinalize,
        nullptr, // simple_update
        WlsAggBind, WlsAggDestroy);
    func_set.AddFunction(basic_func);

    // Version with options: anofox_stats_wls_fit_agg(y, x, weight, fit_intercept, compute_inference, confidence_level)
    auto full_func =
        AggregateFunction("anofox_stats_wls_fit_agg",
                          {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE,
                           LogicalType::BOOLEAN, LogicalType::BOOLEAN, LogicalType::DOUBLE},
                          LogicalType::ANY, AggregateFunction::StateSize<WlsAggregateState>, WlsAggInitialize,
                          WlsAggUpdate, WlsAggCombine, WlsAggFinalize, nullptr, WlsAggBind, WlsAggDestroy);
    func_set.AddFunction(full_func);

    loader.RegisterFunction(func_set);
}

} // namespace duckdb
