#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Elastic Net Aggregate State - accumulates y and x values for each group
//===--------------------------------------------------------------------===//
struct ElasticNetAggregateState {
    vector<double> y_values;
    vector<vector<double>> x_columns;
    idx_t n_features;
    bool initialized;

    // Options
    double alpha;
    double l1_ratio;
    bool fit_intercept;
    uint32_t max_iterations;
    double tolerance;

    ElasticNetAggregateState()
        : n_features(0), initialized(false), alpha(1.0), l1_ratio(0.5), fit_intercept(true), max_iterations(1000),
          tolerance(1e-6) {}

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
struct ElasticNetAggregateBindData : public FunctionData {
    double alpha = 1.0;
    double l1_ratio = 0.5;
    bool fit_intercept = true;
    uint32_t max_iterations = 1000;
    double tolerance = 1e-6;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<ElasticNetAggregateBindData>();
        result->alpha = alpha;
        result->l1_ratio = l1_ratio;
        result->fit_intercept = fit_intercept;
        result->max_iterations = max_iterations;
        result->tolerance = tolerance;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<ElasticNetAggregateBindData>();
        return alpha == other.alpha && l1_ratio == other.l1_ratio && fit_intercept == other.fit_intercept &&
               max_iterations == other.max_iterations && tolerance == other.tolerance;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition (no inference for Elastic Net)
//===--------------------------------------------------------------------===//
static LogicalType GetElasticNetAggResultType() {
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
static void ElasticNetAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) ElasticNetAggregateState();
}

// Destroy aggregate state
static void ElasticNetAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ElasticNetAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~ElasticNetAggregateState();
    }
}

// Update: accumulate values from input rows
static void ElasticNetAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                Vector &state_vector, idx_t count) {
    auto &bind_data = aggr_input_data.bind_data->Cast<ElasticNetAggregateBindData>();

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
    auto states = (ElasticNetAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];

        // Copy options from bind data
        state.alpha = bind_data.alpha;
        state.l1_ratio = bind_data.l1_ratio;
        state.fit_intercept = bind_data.fit_intercept;
        state.max_iterations = bind_data.max_iterations;
        state.tolerance = bind_data.tolerance;

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
static void ElasticNetAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (ElasticNetAggregateState **)source_data.data;
    auto targets = (ElasticNetAggregateState **)target_data.data;

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
            target.alpha = source.alpha;
            target.l1_ratio = source.l1_ratio;
            target.fit_intercept = source.fit_intercept;
            target.max_iterations = source.max_iterations;
            target.tolerance = source.tolerance;
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

// Finalize: compute Elastic Net for accumulated data
static void ElasticNetAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                  idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ElasticNetAggregateState **)sdata.data;

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

        AnofoxElasticNetOptions options;
        options.alpha = state.alpha;
        options.l1_ratio = state.l1_ratio;
        options.fit_intercept = state.fit_intercept;
        options.max_iterations = state.max_iterations;
        options.tolerance = state.tolerance;

        AnofoxFitResultCore core_result;
        AnofoxError error;

        bool success = anofox_elasticnet_fit(y_array, x_arrays.data(), x_arrays.size(), options, &core_result, &error);

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
static unique_ptr<FunctionData> ElasticNetAggBind(ClientContext &context, AggregateFunction &function,
                                                  vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<ElasticNetAggregateBindData>();

    // Parse MAP options if provided as 3rd argument
    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[2]);
        if (opts.fit_intercept.has_value()) {
            result->fit_intercept = opts.fit_intercept.value();
        }
        auto reg_strength = opts.GetRegularizationStrength();
        if (reg_strength.has_value()) {
            result->alpha = reg_strength.value();
        }
        if (opts.l1_ratio.has_value()) {
            result->l1_ratio = opts.l1_ratio.value();
        }
        if (opts.max_iterations.has_value()) {
            result->max_iterations = opts.max_iterations.value();
        }
        if (opts.tolerance.has_value()) {
            result->tolerance = opts.tolerance.value();
        }
    }

    // Set return type
    function.return_type = GetElasticNetAggResultType();

    return std::move(result);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterElasticNetAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_elasticnet_fit_agg");

    // Basic version: anofox_stats_elasticnet_fit_agg(y, x) - uses default alpha=1.0, l1_ratio=0.5
    auto basic_func = AggregateFunction(
        "anofox_stats_elasticnet_fit_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)},
        LogicalType::ANY, // Set in bind
        AggregateFunction::StateSize<ElasticNetAggregateState>, ElasticNetAggInitialize, ElasticNetAggUpdate,
        ElasticNetAggCombine, ElasticNetAggFinalize,
        nullptr, // simple_update
        ElasticNetAggBind, ElasticNetAggDestroy);
    func_set.AddFunction(basic_func);

    // Version with MAP options: anofox_stats_elasticnet_fit_agg(y, x, {'alpha': 1.0, 'l1_ratio': 0.5, ...})
    auto map_func = AggregateFunction("anofox_stats_elasticnet_fit_agg",
                                      {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE),
                                       LogicalType::ANY}, // MAP or STRUCT for options
                                      LogicalType::ANY, AggregateFunction::StateSize<ElasticNetAggregateState>,
                                      ElasticNetAggInitialize, ElasticNetAggUpdate, ElasticNetAggCombine,
                                      ElasticNetAggFinalize, nullptr, ElasticNetAggBind, ElasticNetAggDestroy);
    func_set.AddFunction(map_func);

    loader.RegisterFunction(func_set);

    // Register short alias
    AggregateFunctionSet alias_set("elasticnet_fit_agg");
    alias_set.AddFunction(basic_func);
    alias_set.AddFunction(map_func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
