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

// Result struct type for Elastic Net (no inference since L1 penalty makes standard errors non-standard)
static LogicalType GetElasticNetResultType() {
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

// Bind data for Elastic Net function
struct ElasticNetFitBindData : public FunctionData {
    double alpha = 1.0;
    double l1_ratio = 0.5;
    bool fit_intercept = true;
    uint32_t max_iterations = 1000;
    double tolerance = 1e-6;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<ElasticNetFitBindData>();
        result->alpha = alpha;
        result->l1_ratio = l1_ratio;
        result->fit_intercept = fit_intercept;
        result->max_iterations = max_iterations;
        result->tolerance = tolerance;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<ElasticNetFitBindData>();
        return alpha == other.alpha && l1_ratio == other.l1_ratio && fit_intercept == other.fit_intercept &&
               max_iterations == other.max_iterations && tolerance == other.tolerance;
    }
};

// Bind function
static unique_ptr<FunctionData> ElasticNetFitBind(ClientContext &context, ScalarFunction &bound_function,
                                                  vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<ElasticNetFitBindData>();

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
    bound_function.return_type = GetElasticNetResultType();

    PostHogTelemetry::Instance().CaptureFunctionExecution("elasticnet_fit");
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

// Main Elastic Net fit function
static void ElasticNetFitFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &bind_data = state.expr.Cast<BoundFunctionExpression>().bind_info->Cast<ElasticNetFitBindData>();

    auto &y_vec = args.data[0]; // LIST(DOUBLE)
    auto &x_vec = args.data[1]; // LIST(LIST(DOUBLE)) - list of feature columns

    idx_t count = args.size();

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
        AnofoxElasticNetOptions options;
        options.alpha = bind_data.alpha;
        options.l1_ratio = bind_data.l1_ratio;
        options.fit_intercept = bind_data.fit_intercept;
        options.max_iterations = bind_data.max_iterations;
        options.tolerance = bind_data.tolerance;

        // Call Rust FFI
        AnofoxFitResultCore core_result;
        AnofoxError error;

        bool success = anofox_elasticnet_fit(y_array, x_arrays.data(), x_arrays.size(), options, &core_result, &error);

        if (!success) {
            throw InvalidInputException("Elastic Net fit failed: " + string(error.message));
        }

        // Build result struct
        auto &struct_vec = StructVector::GetEntries(result);
        idx_t struct_idx = 0;

        // Coefficients
        auto &coef_list = *struct_vec[struct_idx++];
        auto &coef_child = ListVector::GetEntry(coef_list);
        auto coef_offset = ListVector::GetListSize(coef_list);
        ListVector::SetListSize(coef_list, coef_offset + core_result.coefficients_len);
        auto coef_data = FlatVector::GetData<double>(coef_child);
        for (size_t i = 0; i < core_result.coefficients_len; i++) {
            coef_data[coef_offset + i] = core_result.coefficients[i];
        }
        ListVector::GetData(coef_list)[row] = {coef_offset, core_result.coefficients_len};

        // Scalars
        FlatVector::GetData<double>(*struct_vec[struct_idx++])[row] = core_result.intercept;
        FlatVector::GetData<double>(*struct_vec[struct_idx++])[row] = core_result.r_squared;
        FlatVector::GetData<double>(*struct_vec[struct_idx++])[row] = core_result.adj_r_squared;
        FlatVector::GetData<double>(*struct_vec[struct_idx++])[row] = core_result.residual_std_error;
        FlatVector::GetData<int64_t>(*struct_vec[struct_idx++])[row] = core_result.n_observations;
        FlatVector::GetData<int64_t>(*struct_vec[struct_idx++])[row] = core_result.n_features;

        // Free core result
        anofox_free_result_core(&core_result);
    }

    result.SetVectorType(VectorType::FLAT_VECTOR);
}

// Register the function
void RegisterElasticNetFitFunction(ExtensionLoader &loader) {
    ScalarFunctionSet func_set("anofox_stats_elasticnet_fit");

    // Basic version: anofox_stats_elasticnet_fit(y, x) - uses default alpha=1.0, l1_ratio=0.5
    ScalarFunction basic_func(
        {LogicalType::LIST(LogicalType::DOUBLE), LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE))},
        LogicalType::ANY, // Will be set in bind
        ElasticNetFitFunction, ElasticNetFitBind);
    func_set.AddFunction(basic_func);

    // Version with MAP options: anofox_stats_elasticnet_fit(y, x, {'alpha': 1.0, 'l1_ratio': 0.5, ...})
    ScalarFunction map_func({LogicalType::LIST(LogicalType::DOUBLE),
                             LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)),
                             LogicalType::ANY}, // MAP or STRUCT for options
                            LogicalType::ANY, ElasticNetFitFunction, ElasticNetFitBind);
    func_set.AddFunction(map_func);

    loader.RegisterFunction(func_set);

    // Register short alias
    ScalarFunctionSet alias_set("elasticnet_fit");
    alias_set.AddFunction(basic_func);
    alias_set.AddFunction(map_func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
