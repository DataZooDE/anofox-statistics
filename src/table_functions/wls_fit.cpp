#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/vector_operations/generic_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "../include/anofox_stats_ffi.h"

namespace duckdb {

// Result struct type for WLS
static LogicalType GetWlsResultType(bool compute_inference) {
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

// Bind data for WLS function
struct WlsFitBindData : public FunctionData {
    bool fit_intercept = true;
    bool compute_inference = false;
    double confidence_level = 0.95;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<WlsFitBindData>();
        result->fit_intercept = fit_intercept;
        result->compute_inference = compute_inference;
        result->confidence_level = confidence_level;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<WlsFitBindData>();
        return fit_intercept == other.fit_intercept && compute_inference == other.compute_inference &&
               confidence_level == other.confidence_level;
    }
};

// Bind function
static unique_ptr<FunctionData> WlsFitBind(ClientContext &context, ScalarFunction &bound_function,
                                           vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<WlsFitBindData>();

    // Check for constant arguments for options (if provided as 4th, 5th, 6th args)
    if (arguments.size() >= 4 && arguments[3]->IsFoldable()) {
        result->fit_intercept = BooleanValue::Get(ExpressionExecutor::EvaluateScalar(context, *arguments[3]));
    }
    if (arguments.size() >= 5 && arguments[4]->IsFoldable()) {
        result->compute_inference = BooleanValue::Get(ExpressionExecutor::EvaluateScalar(context, *arguments[4]));
    }
    if (arguments.size() >= 6 && arguments[5]->IsFoldable()) {
        result->confidence_level = DoubleValue::Get(ExpressionExecutor::EvaluateScalar(context, *arguments[5]));
    }

    // Set return type
    bound_function.return_type = GetWlsResultType(result->compute_inference);

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

// Main WLS fit function
static void WlsFitFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &bind_data = state.expr.Cast<BoundFunctionExpression>().bind_info->Cast<WlsFitBindData>();

    auto &y_vec = args.data[0];       // LIST(DOUBLE)
    auto &x_vec = args.data[1];       // LIST(LIST(DOUBLE)) - list of feature columns
    auto &weights_vec = args.data[2]; // LIST(DOUBLE) - weights

    idx_t count = args.size();

    // Process each row
    for (idx_t row = 0; row < count; row++) {
        // Extract y values
        vector<double> y_data = ExtractDoubleList(y_vec, row);

        // Extract weights
        vector<double> weights_data = ExtractDoubleList(weights_vec, row);

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

        AnofoxDataArray weights_array;
        weights_array.data = weights_data.data();
        weights_array.validity = nullptr;
        weights_array.len = weights_data.size();

        vector<AnofoxDataArray> x_arrays;
        for (auto &col : x_cols) {
            AnofoxDataArray arr;
            arr.data = col.data();
            arr.validity = nullptr;
            arr.len = col.size();
            x_arrays.push_back(arr);
        }

        // Set options
        AnofoxWlsOptions options;
        options.fit_intercept = bind_data.fit_intercept;
        options.compute_inference = bind_data.compute_inference;
        options.confidence_level = bind_data.confidence_level;

        // Call Rust FFI
        AnofoxFitResultCore core_result;
        AnofoxFitResultInference inference_result;
        AnofoxError error;

        bool success = anofox_wls_fit(y_array, x_arrays.data(), x_arrays.size(), weights_array, options, &core_result,
                                      bind_data.compute_inference ? &inference_result : nullptr, &error);

        if (!success) {
            throw InvalidInputException("WLS fit failed: " + string(error.message));
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

        // Inference results if computed
        if (bind_data.compute_inference) {
            auto set_list = [&](Vector &list_vec, double *data, size_t len) {
                auto &child = ListVector::GetEntry(list_vec);
                auto offset = ListVector::GetListSize(list_vec);
                ListVector::SetListSize(list_vec, offset + len);
                auto vec_data = FlatVector::GetData<double>(child);
                for (size_t i = 0; i < len; i++) {
                    vec_data[offset + i] = data[i];
                }
                ListVector::GetData(list_vec)[row] = {offset, len};
            };

            set_list(*struct_vec[struct_idx++], inference_result.std_errors, inference_result.len);
            set_list(*struct_vec[struct_idx++], inference_result.t_values, inference_result.len);
            set_list(*struct_vec[struct_idx++], inference_result.p_values, inference_result.len);
            set_list(*struct_vec[struct_idx++], inference_result.ci_lower, inference_result.len);
            set_list(*struct_vec[struct_idx++], inference_result.ci_upper, inference_result.len);

            FlatVector::GetData<double>(*struct_vec[struct_idx++])[row] = inference_result.f_statistic;
            FlatVector::GetData<double>(*struct_vec[struct_idx++])[row] = inference_result.f_pvalue;

            anofox_free_result_inference(&inference_result);
        }

        // Free core result
        anofox_free_result_core(&core_result);
    }

    result.SetVectorType(VectorType::FLAT_VECTOR);
}

// Register the function
void RegisterWlsFitFunction(ExtensionLoader &loader) {
    // Basic version: anofox_stats_wls_fit(y, x, weights)
    ScalarFunctionSet func_set("anofox_stats_wls_fit");

    // Version with just y, x, and weights
    ScalarFunction basic_func({LogicalType::LIST(LogicalType::DOUBLE),
                               LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)),
                               LogicalType::LIST(LogicalType::DOUBLE)},
                              LogicalType::ANY, // Will be set in bind
                              WlsFitFunction, WlsFitBind);
    func_set.AddFunction(basic_func);

    // Version with options: anofox_stats_wls_fit(y, x, weights, fit_intercept, compute_inference, confidence_level)
    ScalarFunction full_func({LogicalType::LIST(LogicalType::DOUBLE),
                              LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)),
                              LogicalType::LIST(LogicalType::DOUBLE),
                              LogicalType::BOOLEAN, // fit_intercept
                              LogicalType::BOOLEAN, // compute_inference
                              LogicalType::DOUBLE}, // confidence_level
                             LogicalType::ANY, WlsFitFunction, WlsFitBind);
    func_set.AddFunction(full_func);

    loader.RegisterFunction(func_set);
}

} // namespace duckdb
