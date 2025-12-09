#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/common/types/data_chunk.hpp"

#include "../include/anofox_stats_ffi.h"

#include <vector>

namespace duckdb {

//===--------------------------------------------------------------------===//
// OLS Aggregate State - accumulates y and x values for each group
//===--------------------------------------------------------------------===//
struct OlsAggregateState {
    vector<double> y_values;
    vector<vector<double>> x_columns;
    idx_t n_features;
    bool initialized;

    // Options
    bool fit_intercept;
    bool compute_inference;
    double confidence_level;

    OlsAggregateState()
        : n_features(0), initialized(false), fit_intercept(true),
          compute_inference(false), confidence_level(0.95) {}

    void Reset() {
        y_values.clear();
        x_columns.clear();
        n_features = 0;
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Bind Data for options
//===--------------------------------------------------------------------===//
struct OlsAggregateBindData : public FunctionData {
    bool fit_intercept = true;
    bool compute_inference = false;
    double confidence_level = 0.95;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<OlsAggregateBindData>();
        result->fit_intercept = fit_intercept;
        result->compute_inference = compute_inference;
        result->confidence_level = confidence_level;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<OlsAggregateBindData>();
        return fit_intercept == other.fit_intercept &&
               compute_inference == other.compute_inference &&
               confidence_level == other.confidence_level;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetOlsAggResultType(bool compute_inference) {
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

// Initialize aggregate state (const version for new API)
static void OlsAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) OlsAggregateState();
}

// Destroy aggregate state (vectorized version for new API)
static void OlsAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (OlsAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~OlsAggregateState();
    }
}

// Update: accumulate values from input rows
static void OlsAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                         Vector &state_vector, idx_t count) {
    auto &bind_data = aggr_input_data.bind_data->Cast<OlsAggregateBindData>();

    UnifiedVectorFormat y_data;
    UnifiedVectorFormat x_data;
    inputs[0].ToUnifiedFormat(count, y_data);  // y: DOUBLE
    inputs[1].ToUnifiedFormat(count, x_data);  // x: LIST(DOUBLE)

    auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
    auto x_list_data = ListVector::GetData(inputs[1]);
    auto &x_child = ListVector::GetEntry(inputs[1]);
    auto x_child_data = FlatVector::GetData<double>(x_child);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (OlsAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];

        // Copy options from bind data
        state.fit_intercept = bind_data.fit_intercept;
        state.compute_inference = bind_data.compute_inference;
        state.confidence_level = bind_data.confidence_level;

        // Get y value
        auto y_idx = y_data.sel->get_index(i);
        if (!y_data.validity.RowIsValid(y_idx)) {
            continue;  // Skip NULL y values
        }
        double y_val = y_values[y_idx];

        // Get x values (LIST(DOUBLE))
        auto x_idx = x_data.sel->get_index(i);
        if (!x_data.validity.RowIsValid(x_idx)) {
            continue;  // Skip NULL x values
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
            throw InvalidInputException(
                "Inconsistent feature count: expected %lu, got %lu",
                state.n_features, n_features);
        }

        // Accumulate y value
        state.y_values.push_back(y_val);

        // Accumulate x values
        for (idx_t j = 0; j < n_features; j++) {
            double x_val = x_child_data[list_entry.offset + j];
            state.x_columns[j].push_back(x_val);
        }
    }
}

// Combine: merge two states (vectorized version for new API)
static void OlsAggCombine(Vector &source_vector, Vector &target_vector,
                          AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (OlsAggregateState **)source_data.data;
    auto targets = (OlsAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;  // Nothing to combine
        }

        if (!target.initialized) {
            // Copy source to target
            target.y_values = std::move(source.y_values);
            target.x_columns = std::move(source.x_columns);
            target.n_features = source.n_features;
            target.initialized = true;
            target.fit_intercept = source.fit_intercept;
            target.compute_inference = source.compute_inference;
            target.confidence_level = source.confidence_level;
            continue;
        }

        // Validate same feature count
        if (source.n_features != target.n_features) {
            throw InvalidInputException(
                "Cannot combine states with different feature counts: %lu vs %lu",
                source.n_features, target.n_features);
        }

        // Merge y values
        target.y_values.insert(target.y_values.end(), source.y_values.begin(), source.y_values.end());

        // Merge x columns
        for (idx_t j = 0; j < target.n_features; j++) {
            target.x_columns[j].insert(target.x_columns[j].end(),
                                        source.x_columns[j].begin(),
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

// Finalize: compute OLS for accumulated data
static void OlsAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data,
                           Vector &result, idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (OlsAggregateState **)sdata.data;

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
        options.compute_inference = state.compute_inference;
        options.confidence_level = state.confidence_level;

        AnofoxFitResultCore core_result;
        AnofoxFitResultInference inference_result;
        AnofoxError error;

        bool success = anofox_ols_fit(
            y_array,
            x_arrays.data(),
            x_arrays.size(),
            options,
            &core_result,
            state.compute_inference ? &inference_result : nullptr,
            &error
        );

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Fill STRUCT result
        idx_t struct_idx = 0;

        // Coefficients
        SetListInResult(*struct_entries[struct_idx++], result_idx,
                       core_result.coefficients, core_result.coefficients_len);

        // Scalars
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.intercept;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.r_squared;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.adj_r_squared;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.residual_std_error;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = core_result.n_observations;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = core_result.n_features;

        // Inference results
        if (state.compute_inference) {
            SetListInResult(*struct_entries[struct_idx++], result_idx,
                           inference_result.std_errors, inference_result.len);
            SetListInResult(*struct_entries[struct_idx++], result_idx,
                           inference_result.t_values, inference_result.len);
            SetListInResult(*struct_entries[struct_idx++], result_idx,
                           inference_result.p_values, inference_result.len);
            SetListInResult(*struct_entries[struct_idx++], result_idx,
                           inference_result.ci_lower, inference_result.len);
            SetListInResult(*struct_entries[struct_idx++], result_idx,
                           inference_result.ci_upper, inference_result.len);

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
static unique_ptr<FunctionData> OlsAggBind(ClientContext &context, AggregateFunction &function,
                                           vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<OlsAggregateBindData>();

    // Extract options if provided (3rd, 4th, 5th arguments)
    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        result->fit_intercept = BooleanValue::Get(ExpressionExecutor::EvaluateScalar(context, *arguments[2]));
    }
    if (arguments.size() >= 4 && arguments[3]->IsFoldable()) {
        result->compute_inference = BooleanValue::Get(ExpressionExecutor::EvaluateScalar(context, *arguments[3]));
    }
    if (arguments.size() >= 5 && arguments[4]->IsFoldable()) {
        result->confidence_level = DoubleValue::Get(ExpressionExecutor::EvaluateScalar(context, *arguments[4]));
    }

    // Set return type based on options
    function.return_type = GetOlsAggResultType(result->compute_inference);

    return std::move(result);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterOlsAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_ols_fit_agg");

    // Basic version: anofox_stats_ols_fit_agg(y, x)
    auto basic_func = AggregateFunction(
        "anofox_stats_ols_fit_agg",
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)},
        LogicalType::ANY,  // Set in bind
        AggregateFunction::StateSize<OlsAggregateState>,
        OlsAggInitialize,
        OlsAggUpdate,
        OlsAggCombine,
        OlsAggFinalize,
        nullptr,  // simple_update
        OlsAggBind,
        OlsAggDestroy
    );
    func_set.AddFunction(basic_func);

    // Version with options: anofox_stats_ols_fit_agg(y, x, fit_intercept, compute_inference, confidence_level)
    auto full_func = AggregateFunction(
        "anofox_stats_ols_fit_agg",
        {LogicalType::DOUBLE,
         LogicalType::LIST(LogicalType::DOUBLE),
         LogicalType::BOOLEAN,
         LogicalType::BOOLEAN,
         LogicalType::DOUBLE},
        LogicalType::ANY,
        AggregateFunction::StateSize<OlsAggregateState>,
        OlsAggInitialize,
        OlsAggUpdate,
        OlsAggCombine,
        OlsAggFinalize,
        nullptr,
        OlsAggBind,
        OlsAggDestroy
    );
    func_set.AddFunction(full_func);

    loader.RegisterFunction(func_set);
}

} // namespace duckdb
