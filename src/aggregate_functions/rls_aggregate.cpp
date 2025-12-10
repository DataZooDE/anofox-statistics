#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"

namespace duckdb {

//===--------------------------------------------------------------------===//
// RLS Aggregate State - accumulates y and x values for each group
//===--------------------------------------------------------------------===//
struct RlsAggregateState {
    vector<double> y_values;
    vector<vector<double>> x_columns;
    idx_t n_features;
    bool initialized;

    // Options
    double forgetting_factor;
    bool fit_intercept;
    double initial_p_diagonal;

    RlsAggregateState()
        : n_features(0), initialized(false), forgetting_factor(1.0), fit_intercept(true), initial_p_diagonal(100.0) {}

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
struct RlsAggregateBindData : public FunctionData {
    double forgetting_factor = 1.0;
    bool fit_intercept = true;
    double initial_p_diagonal = 100.0;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<RlsAggregateBindData>();
        result->forgetting_factor = forgetting_factor;
        result->fit_intercept = fit_intercept;
        result->initial_p_diagonal = initial_p_diagonal;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<RlsAggregateBindData>();
        return forgetting_factor == other.forgetting_factor && fit_intercept == other.fit_intercept &&
               initial_p_diagonal == other.initial_p_diagonal;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetRlsAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
    children.push_back(make_pair("intercept", LogicalType::DOUBLE));
    children.push_back(make_pair("r_squared", LogicalType::DOUBLE));
    children.push_back(make_pair("adj_r_squared", LogicalType::DOUBLE));
    children.push_back(make_pair("residual_std_error", LogicalType::DOUBLE));
    children.push_back(make_pair("n_observations", LogicalType::BIGINT));
    children.push_back(make_pair("n_features", LogicalType::BIGINT));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

// Initialize aggregate state
static void RlsAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) RlsAggregateState();
}

// Destroy aggregate state
static void RlsAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (RlsAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~RlsAggregateState();
    }
}

// Update: accumulate values from input rows
static void RlsAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &state_vector,
                         idx_t count) {
    auto &bind_data = aggr_input_data.bind_data->Cast<RlsAggregateBindData>();

    UnifiedVectorFormat y_data;
    UnifiedVectorFormat x_data;
    inputs[0].ToUnifiedFormat(count, y_data); // y: DOUBLE
    inputs[1].ToUnifiedFormat(count, x_data); // x: LIST(DOUBLE)

    auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
    auto x_list_data = ListVector::GetData(inputs[1]);
    auto &x_child = ListVector::GetEntry(inputs[1]);
    auto x_child_data = FlatVector::GetData<double>(x_child);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (RlsAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];

        // Copy options from bind data
        state.forgetting_factor = bind_data.forgetting_factor;
        state.fit_intercept = bind_data.fit_intercept;
        state.initial_p_diagonal = bind_data.initial_p_diagonal;

        // Get y value
        auto y_idx = y_data.sel->get_index(i);
        if (!y_data.validity.RowIsValid(y_idx)) {
            continue; // Skip NULL y values
        }
        double y_val = y_values[y_idx];

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

        // Accumulate x values
        for (idx_t j = 0; j < n_features; j++) {
            double x_val = x_child_data[list_entry.offset + j];
            state.x_columns[j].push_back(x_val);
        }
    }
}

// Combine: merge two states
static void RlsAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (RlsAggregateState **)source_data.data;
    auto targets = (RlsAggregateState **)target_data.data;

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
            target.n_features = source.n_features;
            target.initialized = true;
            target.forgetting_factor = source.forgetting_factor;
            target.fit_intercept = source.fit_intercept;
            target.initial_p_diagonal = source.initial_p_diagonal;
            continue;
        }

        // Validate same feature count
        if (source.n_features != target.n_features) {
            throw InvalidInputException("Cannot combine states with different feature counts: %lu vs %lu",
                                        source.n_features, target.n_features);
        }

        // Merge y values
        target.y_values.insert(target.y_values.end(), source.y_values.begin(), source.y_values.end());

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

// Finalize: compute RLS for accumulated data
static void RlsAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                           idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (RlsAggregateState **)sdata.data;

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

        AnofoxRlsOptions options;
        options.forgetting_factor = state.forgetting_factor;
        options.fit_intercept = state.fit_intercept;
        options.initial_p_diagonal = state.initial_p_diagonal;

        AnofoxFitResultCore core_result;
        AnofoxError error;

        bool success = anofox_rls_fit(y_array, x_arrays.data(), x_arrays.size(), options, &core_result, &error);

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

        anofox_free_result_core(&core_result);

        // Reset state for next use
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> RlsAggBind(ClientContext &context, AggregateFunction &function,
                                           vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<RlsAggregateBindData>();

    // Extract options if provided (3rd, 4th, 5th arguments)
    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        result->forgetting_factor = DoubleValue::Get(ExpressionExecutor::EvaluateScalar(context, *arguments[2]));
    }
    if (arguments.size() >= 4 && arguments[3]->IsFoldable()) {
        result->fit_intercept = BooleanValue::Get(ExpressionExecutor::EvaluateScalar(context, *arguments[3]));
    }
    if (arguments.size() >= 5 && arguments[4]->IsFoldable()) {
        result->initial_p_diagonal = DoubleValue::Get(ExpressionExecutor::EvaluateScalar(context, *arguments[4]));
    }

    // Set return type
    function.return_type = GetRlsAggResultType();

    return std::move(result);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterRlsAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_rls_fit_agg");

    // Basic version: anofox_stats_rls_fit_agg(y, x)
    auto basic_func = AggregateFunction(
        "anofox_stats_rls_fit_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)},
        LogicalType::ANY, // Set in bind
        AggregateFunction::StateSize<RlsAggregateState>, RlsAggInitialize, RlsAggUpdate, RlsAggCombine, RlsAggFinalize,
        nullptr, // simple_update
        RlsAggBind, RlsAggDestroy);
    func_set.AddFunction(basic_func);

    // Version with options: anofox_stats_rls_fit_agg(y, x, forgetting_factor, fit_intercept, initial_p_diagonal)
    auto full_func =
        AggregateFunction("anofox_stats_rls_fit_agg",
                          {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE,
                           LogicalType::BOOLEAN, LogicalType::DOUBLE},
                          LogicalType::ANY, AggregateFunction::StateSize<RlsAggregateState>, RlsAggInitialize,
                          RlsAggUpdate, RlsAggCombine, RlsAggFinalize, nullptr, RlsAggBind, RlsAggDestroy);
    func_set.AddFunction(full_func);

    loader.RegisterFunction(func_set);
}

} // namespace duckdb
