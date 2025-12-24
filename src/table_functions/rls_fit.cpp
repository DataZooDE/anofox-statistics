#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/vector_operations/generic_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"

namespace duckdb {

// Result struct type for RLS (same as core fit result)
static LogicalType GetRlsResultType() {
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

// Bind data for RLS function
struct RlsFitBindData : public FunctionData {
    double forgetting_factor = 1.0;
    bool fit_intercept = true;
    double initial_p_diagonal = 100.0;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<RlsFitBindData>();
        result->forgetting_factor = forgetting_factor;
        result->fit_intercept = fit_intercept;
        result->initial_p_diagonal = initial_p_diagonal;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<RlsFitBindData>();
        return forgetting_factor == other.forgetting_factor && fit_intercept == other.fit_intercept &&
               initial_p_diagonal == other.initial_p_diagonal;
    }
};

// Bind function
static unique_ptr<FunctionData> RlsFitBind(ClientContext &context, ScalarFunction &bound_function,
                                           vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<RlsFitBindData>();

    // Parse MAP options if provided as 3rd argument
    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[2]);
        if (opts.fit_intercept.has_value()) {
            result->fit_intercept = opts.fit_intercept.value();
        }
        if (opts.forgetting_factor.has_value()) {
            result->forgetting_factor = opts.forgetting_factor.value();
        }
        if (opts.initial_p_diagonal.has_value()) {
            result->initial_p_diagonal = opts.initial_p_diagonal.value();
        }
    }

    // Set return type
    bound_function.return_type = GetRlsResultType();

    PostHogTelemetry::Instance().CaptureFunctionExecution("rls_fit");
    return std::move(result);
}

// Extract list of doubles from a DuckDB list value
static vector<double> ExtractDoubleList(Vector &vec, idx_t row_idx) {
    auto list_data = ListVector::GetData(vec);
    auto &child = ListVector::GetEntry(vec);
    auto child_data = FlatVector::GetData<double>(child);

    vector<double> result;
    auto offset = list_data[row_idx].offset;
    auto length = list_data[row_idx].length;

    for (idx_t i = 0; i < length; i++) {
        result.push_back(child_data[offset + i]);
    }

    return result;
}

// Helper to set list in result
static void SetListResult(Vector &list_vec, idx_t row, double *data, size_t len) {
    auto &child = ListVector::GetEntry(list_vec);
    auto offset = ListVector::GetListSize(list_vec);
    ListVector::SetListSize(list_vec, offset + len);
    auto vec_data = FlatVector::GetData<double>(child);
    for (size_t i = 0; i < len; i++) {
        vec_data[offset + i] = data[i];
    }
    ListVector::GetData(list_vec)[row] = {offset, (idx_t)len};
}

// Main RLS fit function
static void RlsFitFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &bind_data = state.expr.Cast<BoundFunctionExpression>().bind_info->Cast<RlsFitBindData>();

    auto &y_vec = args.data[0]; // LIST(DOUBLE)
    auto &x_vec = args.data[1]; // LIST(LIST(DOUBLE)) - list of feature columns

    idx_t count = args.size();
    auto &struct_entries = StructVector::GetEntries(result);

    // Process each row
    for (idx_t row = 0; row < count; row++) {
        // Extract y values
        vector<double> y_data = ExtractDoubleList(y_vec, row);

        // Extract x values (list of columns)
        auto x_list_data = ListVector::GetData(x_vec);
        auto &x_child = ListVector::GetEntry(x_vec);

        vector<vector<double>> x_cols;
        auto x_offset = x_list_data[row].offset;
        auto x_length = x_list_data[row].length;

        for (idx_t col = 0; col < x_length; col++) {
            x_cols.push_back(ExtractDoubleList(x_child, x_offset + col));
        }

        // Prepare FFI data
        AnofoxDataArray y_array;
        y_array.data = y_data.data();
        y_array.validity = nullptr;
        y_array.len = y_data.size();

        vector<AnofoxDataArray> x_arrays;
        for (auto &col : x_cols) {
            AnofoxDataArray arr;
            arr.data = col.data();
            arr.validity = nullptr;
            arr.len = col.size();
            x_arrays.push_back(arr);
        }

        // Set options
        AnofoxRlsOptions options;
        options.forgetting_factor = bind_data.forgetting_factor;
        options.fit_intercept = bind_data.fit_intercept;
        options.initial_p_diagonal = bind_data.initial_p_diagonal;

        // Call Rust FFI
        AnofoxFitResultCore core_result;
        AnofoxError error;

        bool success = anofox_rls_fit(y_array, x_arrays.data(), x_arrays.size(), options, &core_result, &error);

        if (!success) {
            FlatVector::SetNull(result, row, true);
            continue;
        }

        // Fill result struct
        idx_t struct_idx = 0;

        // Coefficients
        SetListResult(*struct_entries[struct_idx++], row, core_result.coefficients, core_result.coefficients_len);

        // Scalars
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[row] = core_result.intercept;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[row] = core_result.r_squared;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[row] = core_result.adj_r_squared;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[row] = core_result.residual_std_error;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[row] = core_result.n_observations;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[row] = core_result.n_features;

        anofox_free_result_core(&core_result);
    }
}

// Register the function
void RegisterRlsFitFunction(ExtensionLoader &loader) {
    ScalarFunctionSet func_set("anofox_stats_rls_fit");

    // Basic version: anofox_stats_rls_fit(y, x) - uses defaults
    ScalarFunction basic_func(
        {LogicalType::LIST(LogicalType::DOUBLE), LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE))},
        LogicalType::ANY, // Set in bind
        RlsFitFunction, RlsFitBind);
    func_set.AddFunction(basic_func);

    // Version with MAP options: anofox_stats_rls_fit(y, x, {'forgetting_factor': 0.99, ...})
    ScalarFunction map_func({LogicalType::LIST(LogicalType::DOUBLE),
                             LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)),
                             LogicalType::ANY}, // MAP or STRUCT for options
                            LogicalType::ANY, RlsFitFunction, RlsFitBind);
    func_set.AddFunction(map_func);

    loader.RegisterFunction(func_set);

    // Register short alias
    ScalarFunctionSet alias_set("rls_fit");
    alias_set.AddFunction(basic_func);
    alias_set.AddFunction(map_func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
