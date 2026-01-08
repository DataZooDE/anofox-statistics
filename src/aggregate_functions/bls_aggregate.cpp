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
// BLS Aggregate State - accumulates y and x values for each group
//===--------------------------------------------------------------------===//
struct BlsAggregateState {
    vector<double> y_values;
    vector<vector<double>> x_columns;
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

    BlsAggregateState()
        : n_features(0), initialized(false), fit_intercept(false), lower_bound(0.0), upper_bound(0.0),
          has_lower_bound(false), has_upper_bound(false), max_iterations(1000), tolerance(1e-10) {}

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
struct BlsAggregateBindData : public FunctionData {
    bool fit_intercept = false;
    double lower_bound = 0.0;
    double upper_bound = 0.0;
    bool has_lower_bound = false;
    bool has_upper_bound = false;
    uint32_t max_iterations = 1000;
    double tolerance = 1e-10;
    bool is_nnls = false; // Non-negative least squares mode

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<BlsAggregateBindData>();
        result->fit_intercept = fit_intercept;
        result->lower_bound = lower_bound;
        result->upper_bound = upper_bound;
        result->has_lower_bound = has_lower_bound;
        result->has_upper_bound = has_upper_bound;
        result->max_iterations = max_iterations;
        result->tolerance = tolerance;
        result->is_nnls = is_nnls;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<BlsAggregateBindData>();
        return fit_intercept == other.fit_intercept && lower_bound == other.lower_bound &&
               upper_bound == other.upper_bound && has_lower_bound == other.has_lower_bound &&
               has_upper_bound == other.has_upper_bound && max_iterations == other.max_iterations &&
               tolerance == other.tolerance && is_nnls == other.is_nnls;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetBlsAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
    children.push_back(make_pair("intercept", LogicalType::DOUBLE));
    children.push_back(make_pair("ssr", LogicalType::DOUBLE));
    children.push_back(make_pair("r_squared", LogicalType::DOUBLE));
    children.push_back(make_pair("n_observations", LogicalType::BIGINT));
    children.push_back(make_pair("n_features", LogicalType::BIGINT));
    children.push_back(make_pair("n_active_constraints", LogicalType::BIGINT));
    children.push_back(make_pair("at_lower_bound", LogicalType::LIST(LogicalType::BOOLEAN)));
    children.push_back(make_pair("at_upper_bound", LogicalType::LIST(LogicalType::BOOLEAN)));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void BlsAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) BlsAggregateState();
}

static void BlsAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (BlsAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~BlsAggregateState();
    }
}

static void BlsAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &state_vector,
                         idx_t count) {
    auto &bind_data = aggr_input_data.bind_data->Cast<BlsAggregateBindData>();

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
    auto states = (BlsAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];

        state.fit_intercept = bind_data.fit_intercept;
        state.lower_bound = bind_data.lower_bound;
        state.upper_bound = bind_data.upper_bound;
        state.has_lower_bound = bind_data.has_lower_bound;
        state.has_upper_bound = bind_data.has_upper_bound;
        state.max_iterations = bind_data.max_iterations;
        state.tolerance = bind_data.tolerance;

        auto y_idx = y_data.sel->get_index(i);
        if (!y_data.validity.RowIsValid(y_idx)) {
            continue;
        }
        double y_val = y_values[y_idx];

        auto x_idx = x_data.sel->get_index(i);
        if (!x_data.validity.RowIsValid(x_idx)) {
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

        state.y_values.push_back(y_val);

        for (idx_t j = 0; j < n_features; j++) {
            double x_val = x_child_data[list_entry.offset + j];
            state.x_columns[j].push_back(x_val);
        }
    }
}

static void BlsAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (BlsAggregateState **)source_data.data;
    auto targets = (BlsAggregateState **)target_data.data;

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
            target.fit_intercept = source.fit_intercept;
            target.lower_bound = source.lower_bound;
            target.upper_bound = source.upper_bound;
            target.has_lower_bound = source.has_lower_bound;
            target.has_upper_bound = source.has_upper_bound;
            target.max_iterations = source.max_iterations;
            target.tolerance = source.tolerance;
            continue;
        }

        if (source.n_features != target.n_features) {
            throw InvalidInputException("Cannot combine states with different feature counts: %lu vs %lu",
                                        source.n_features, target.n_features);
        }

        target.y_values.insert(target.y_values.end(), source.y_values.begin(), source.y_values.end());

        for (idx_t j = 0; j < target.n_features; j++) {
            target.x_columns[j].insert(target.x_columns[j].end(), source.x_columns[j].begin(),
                                       source.x_columns[j].end());
        }
    }
}

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

static void SetBoolListInResult(Vector &list_vec, idx_t row, bool *data, size_t len) {
    auto &child = ListVector::GetEntry(list_vec);
    auto offset = ListVector::GetListSize(list_vec);
    ListVector::SetListSize(list_vec, offset + len);
    auto vec_data = FlatVector::GetData<bool>(child);
    for (size_t i = 0; i < len; i++) {
        vec_data[offset + i] = data[i];
    }
    ListVector::GetData(list_vec)[row] = {offset, (idx_t)len};
}

static void BlsAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                           idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (BlsAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.y_values.size() < 2) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Note: Detailed min_obs validation including zero-variance column handling is done in Rust
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

        idx_t struct_idx = 0;

        SetListInResult(*struct_entries[struct_idx++], result_idx, core_result.coefficients,
                        core_result.coefficients_len);

        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.intercept;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.ssr;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.r_squared;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = core_result.n_observations;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = core_result.n_features;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = core_result.n_active_constraints;

        SetBoolListInResult(*struct_entries[struct_idx++], result_idx, core_result.at_lower_bound,
                            core_result.coefficients_len);
        SetBoolListInResult(*struct_entries[struct_idx++], result_idx, core_result.at_upper_bound,
                            core_result.coefficients_len);

        anofox_free_bls_result(&core_result);

        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> BlsAggBind(ClientContext &context, AggregateFunction &function,
                                           vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<BlsAggregateBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[2]);
        if (opts.fit_intercept.has_value()) {
            result->fit_intercept = opts.fit_intercept.value();
        }
        if (opts.max_iterations.has_value()) {
            result->max_iterations = opts.max_iterations.value();
        }
        if (opts.tolerance.has_value()) {
            result->tolerance = opts.tolerance.value();
        }
        if (opts.lower_bound.has_value()) {
            result->lower_bound = opts.lower_bound.value();
            result->has_lower_bound = true;
        }
        if (opts.upper_bound.has_value()) {
            result->upper_bound = opts.upper_bound.value();
            result->has_upper_bound = true;
        }
    }

    function.return_type = GetBlsAggResultType();

    PostHogTelemetry::Instance().CaptureFunctionExecution("bls_fit_agg");
    return std::move(result);
}

// Bind for NNLS (Non-Negative Least Squares)
static unique_ptr<FunctionData> NnlsAggBind(ClientContext &context, AggregateFunction &function,
                                            vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<BlsAggregateBindData>();

    // NNLS: lower bound is 0, no upper bound
    result->is_nnls = true;
    result->has_lower_bound = false; // FFI will use NNLS defaults (0 lower bound)
    result->has_upper_bound = false;

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[2]);
        if (opts.fit_intercept.has_value()) {
            result->fit_intercept = opts.fit_intercept.value();
        }
        if (opts.max_iterations.has_value()) {
            result->max_iterations = opts.max_iterations.value();
        }
        if (opts.tolerance.has_value()) {
            result->tolerance = opts.tolerance.value();
        }
    }

    function.return_type = GetBlsAggResultType();

    PostHogTelemetry::Instance().CaptureFunctionExecution("nnls_fit_agg");
    return std::move(result);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterBlsAggregateFunction(ExtensionLoader &loader) {
    // BLS (Bounded Least Squares)
    AggregateFunctionSet bls_set("anofox_stats_bls_fit_agg");

    auto bls_basic = AggregateFunction(
        "anofox_stats_bls_fit_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)}, LogicalType::ANY,
        AggregateFunction::StateSize<BlsAggregateState>, BlsAggInitialize, BlsAggUpdate, BlsAggCombine, BlsAggFinalize,
        nullptr, BlsAggBind, BlsAggDestroy);
    bls_set.AddFunction(bls_basic);

    auto bls_map = AggregateFunction(
        "anofox_stats_bls_fit_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY},
        LogicalType::ANY, AggregateFunction::StateSize<BlsAggregateState>, BlsAggInitialize, BlsAggUpdate,
        BlsAggCombine, BlsAggFinalize, nullptr, BlsAggBind, BlsAggDestroy);
    bls_set.AddFunction(bls_map);

    loader.RegisterFunction(bls_set);

    // Short alias
    AggregateFunctionSet bls_alias("bls_fit_agg");
    bls_alias.AddFunction(bls_basic);
    bls_alias.AddFunction(bls_map);
    loader.RegisterFunction(bls_alias);

    // NNLS (Non-Negative Least Squares)
    AggregateFunctionSet nnls_set("anofox_stats_nnls_fit_agg");

    auto nnls_basic = AggregateFunction(
        "anofox_stats_nnls_fit_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)}, LogicalType::ANY,
        AggregateFunction::StateSize<BlsAggregateState>, BlsAggInitialize, BlsAggUpdate, BlsAggCombine, BlsAggFinalize,
        nullptr, NnlsAggBind, BlsAggDestroy);
    nnls_set.AddFunction(nnls_basic);

    auto nnls_map = AggregateFunction(
        "anofox_stats_nnls_fit_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY},
        LogicalType::ANY, AggregateFunction::StateSize<BlsAggregateState>, BlsAggInitialize, BlsAggUpdate,
        BlsAggCombine, BlsAggFinalize, nullptr, NnlsAggBind, BlsAggDestroy);
    nnls_set.AddFunction(nnls_map);

    loader.RegisterFunction(nnls_set);

    // Short alias for NNLS
    AggregateFunctionSet nnls_alias("nnls_fit_agg");
    nnls_alias.AddFunction(nnls_basic);
    nnls_alias.AddFunction(nnls_map);
    loader.RegisterFunction(nnls_alias);
}

} // namespace duckdb
