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
// LARS Aggregate State - accumulates y and x values for each group
//===--------------------------------------------------------------------===//
struct LarsAggregateState {
    vector<double> y_values;
    vector<vector<double>> x_columns;
    idx_t n_features;
    bool initialized;

    // Options
    bool method_lasso;
    bool fit_intercept;
    double alpha;
    int64_t n_nonzero_coefs;
    bool standardize;

    LarsAggregateState()
        : n_features(0), initialized(false), method_lasso(false), fit_intercept(true), alpha(0.0),
          n_nonzero_coefs(0), standardize(true) {}

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
struct LarsAggregateBindData : public FunctionData {
    bool method_lasso = false;
    bool fit_intercept = true;
    double alpha = 0.0;
    int64_t n_nonzero_coefs = 0;
    bool standardize = true;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<LarsAggregateBindData>();
        result->method_lasso = method_lasso;
        result->fit_intercept = fit_intercept;
        result->alpha = alpha;
        result->n_nonzero_coefs = n_nonzero_coefs;
        result->standardize = standardize;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<LarsAggregateBindData>();
        return method_lasso == other.method_lasso && fit_intercept == other.fit_intercept &&
               alpha == other.alpha && n_nonzero_coefs == other.n_nonzero_coefs &&
               standardize == other.standardize;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition (no inference for LARS)
//===--------------------------------------------------------------------===//
static LogicalType GetLarsAggResultType() {
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

static void LarsAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) LarsAggregateState();
}

static void LarsAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (LarsAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~LarsAggregateState();
    }
}

static void LarsAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                          Vector &state_vector, idx_t count) {
    auto &bind_data = aggr_input_data.bind_data->Cast<LarsAggregateBindData>();

    UnifiedVectorFormat y_data;
    UnifiedVectorFormat x_data;
    inputs[0].ToUnifiedFormat(count, y_data); // y: DOUBLE
    inputs[1].ToUnifiedFormat(count, x_data); // x: LIST(DOUBLE)

    auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
    auto x_list_data = ListVector::GetData(inputs[1]);
    auto &x_child = ListVector::GetEntry(inputs[1]);
    auto x_child_data = FlatVector::GetData<double>(x_child);
    auto &x_child_validity = FlatVector::Validity(x_child);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (LarsAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];

        // Copy options from bind data
        state.method_lasso = bind_data.method_lasso;
        state.fit_intercept = bind_data.fit_intercept;
        state.alpha = bind_data.alpha;
        state.n_nonzero_coefs = bind_data.n_nonzero_coefs;
        state.standardize = bind_data.standardize;

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

        // Accumulate x values; a NULL list element's slot is uninitialized, so
        // substitute NaN rather than reading it (the Rust fit drops NaN rows).
        for (idx_t j = 0; j < n_features; j++) {
            idx_t child_pos = list_entry.offset + j;
            double x_val = x_child_validity.RowIsValid(child_pos) ? x_child_data[child_pos] : std::nan("");
            state.x_columns[j].push_back(x_val);
        }
    }
}

static void LarsAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (LarsAggregateState **)source_data.data;
    auto targets = (LarsAggregateState **)target_data.data;

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
            target.method_lasso = source.method_lasso;
            target.fit_intercept = source.fit_intercept;
            target.alpha = source.alpha;
            target.n_nonzero_coefs = source.n_nonzero_coefs;
            target.standardize = source.standardize;
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

static void LarsAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                            idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (LarsAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.y_values.size() < 2) {
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

        AnofoxLarsOptions options;
        options.method_lasso = state.method_lasso;
        options.fit_intercept = state.fit_intercept;
        options.alpha = state.alpha;
        options.n_nonzero_coefs = state.n_nonzero_coefs;
        options.standardize = state.standardize;

        AnofoxFitResultCore core_result;
        AnofoxError error;

        bool success = anofox_lars_fit(y_array, x_arrays.data(), x_arrays.size(), options, &core_result, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t struct_idx = 0;
        SetListInResult(*struct_entries[struct_idx++], result_idx, core_result.coefficients,
                        core_result.coefficients_len);
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.intercept;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.r_squared;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.adj_r_squared;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.residual_std_error;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = core_result.n_observations;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = core_result.n_features;

        anofox_free_result_core(&core_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> LarsAggBind(ClientContext &context, AggregateFunction &function,
                                            vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<LarsAggregateBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[2]);
        if (opts.fit_intercept.has_value()) {
            result->fit_intercept = opts.fit_intercept.value();
        }
        auto reg_strength = opts.GetRegularizationStrength();
        if (reg_strength.has_value()) {
            result->alpha = reg_strength.value();
        }
    }

    function.return_type = GetLarsAggResultType();

    PostHogTelemetry::Instance().CaptureFunctionExecution("lars_fit_agg");
    return std::move(result);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterLarsAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_lars_fit_agg");

    auto basic_func = AggregateFunction(
        "anofox_stats_lars_fit_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)},
        LogicalType::ANY, AggregateFunction::StateSize<LarsAggregateState>, LarsAggInitialize, LarsAggUpdate,
        LarsAggCombine, LarsAggFinalize, nullptr, LarsAggBind, LarsAggDestroy);
    func_set.AddFunction(basic_func);

    auto map_func = AggregateFunction(
        "anofox_stats_lars_fit_agg",
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY}, LogicalType::ANY,
        AggregateFunction::StateSize<LarsAggregateState>, LarsAggInitialize, LarsAggUpdate, LarsAggCombine,
        LarsAggFinalize, nullptr, LarsAggBind, LarsAggDestroy);
    func_set.AddFunction(map_func);

    CreateAggregateFunctionInfo info(std::move(func_set));
    info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
    FunctionDescription d1;
    d1.description = "Fits a Least Angle Regression (LARS / LassoLars) model and returns coefficients and fit "
                     "statistics.";
    d1.examples = {"anofox_stats_lars_fit_agg(y, x, {'fit_intercept': true, 'alpha': 0.0})"};
    d1.categories = {"regression"};
    d1.parameter_names = {"y", "x", "options"};
    d1.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY};
    info.descriptions.push_back(std::move(d1));
    FunctionDescription d2;
    d2.description = "Fits a Least Angle Regression (LARS) model and returns coefficients and fit statistics.";
    d2.examples = {"anofox_stats_lars_fit_agg(y, x)"};
    d2.categories = {"regression"};
    d2.parameter_names = {"y", "x"};
    d2.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)};
    info.descriptions.push_back(std::move(d2));
    loader.RegisterFunction(std::move(info));

    // Register short alias
    {
        AggregateFunctionSet alias_set("lars_fit_agg");
        alias_set.AddFunction(basic_func);
        alias_set.AddFunction(map_func);
        CreateAggregateFunctionInfo alias_info(std::move(alias_set));
        alias_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        alias_info.alias_of = "anofox_stats_lars_fit_agg";
        loader.RegisterFunction(std::move(alias_info));
    }
}

} // namespace duckdb
