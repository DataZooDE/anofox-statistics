#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/vector_operations/generic_executor.hpp"

#include "../include/anofox_stats_ffi.h"

#include <vector>

namespace duckdb {

// Result struct type for residuals diagnostics
static LogicalType GetResidualsDiagnosticsResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("raw", LogicalType::LIST(LogicalType::DOUBLE)));
    children.push_back(make_pair("standardized", LogicalType::LIST(LogicalType::DOUBLE)));
    children.push_back(make_pair("studentized", LogicalType::LIST(LogicalType::DOUBLE)));
    children.push_back(make_pair("leverage", LogicalType::LIST(LogicalType::DOUBLE)));

    return LogicalType::STRUCT(std::move(children));
}

// Bind function
static unique_ptr<FunctionData> ResidualsDiagnosticsBind(ClientContext &context, ScalarFunction &bound_function,
                                                          vector<unique_ptr<Expression>> &arguments) {
    bound_function.return_type = GetResidualsDiagnosticsResultType();
    return nullptr;
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

// Helper to set a list in STRUCT result
static void SetListInResult(Vector &list_vec, idx_t row, double *data, size_t len) {
    if (data == nullptr || len == 0) {
        FlatVector::SetNull(list_vec, row, true);
        return;
    }
    auto &child = ListVector::GetEntry(list_vec);
    auto offset = ListVector::GetListSize(list_vec);
    ListVector::SetListSize(list_vec, offset + len);
    auto vec_data = FlatVector::GetData<double>(child);
    for (size_t i = 0; i < len; i++) {
        vec_data[offset + i] = data[i];
    }
    ListVector::GetData(list_vec)[row] = {offset, (idx_t)len};
}

// Main residuals diagnostics function
// Arguments: y, y_hat, [x, residual_std_error, include_studentized]
static void ResidualsDiagnosticsFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &y_vec = args.data[0];      // LIST(DOUBLE)
    auto &y_hat_vec = args.data[1];  // LIST(DOUBLE)

    idx_t count = args.size();
    auto &struct_entries = StructVector::GetEntries(result);

    // Check for optional arguments
    bool has_x = args.ColumnCount() >= 3;
    bool has_rse = args.ColumnCount() >= 4;
    bool has_studentized_flag = args.ColumnCount() >= 5;

    for (idx_t row = 0; row < count; row++) {
        // Extract y and y_hat values
        vector<double> y_data = ExtractDoubleList(y_vec, row);
        vector<double> y_hat_data = ExtractDoubleList(y_hat_vec, row);

        if (y_data.size() < 3 || y_data.size() != y_hat_data.size()) {
            FlatVector::SetNull(result, row, true);
            continue;
        }

        // Prepare FFI data
        AnofoxDataArray y_array;
        y_array.data = y_data.data();
        y_array.validity = nullptr;
        y_array.len = y_data.size();

        AnofoxDataArray y_hat_array;
        y_hat_array.data = y_hat_data.data();
        y_hat_array.validity = nullptr;
        y_hat_array.len = y_hat_data.size();

        // Handle optional x parameter
        vector<vector<double>> x_cols;
        vector<AnofoxDataArray> x_arrays;
        const AnofoxDataArray* x_ptr = nullptr;
        size_t x_count = 0;

        if (has_x) {
            auto &x_vec = args.data[2];  // LIST(LIST(DOUBLE))
            auto x_list_data = ListVector::GetData(x_vec);
            auto &x_child = ListVector::GetEntry(x_vec);

            auto x_offset = x_list_data[row].offset;
            auto x_length = x_list_data[row].length;

            for (idx_t col = 0; col < x_length; col++) {
                x_cols.push_back(ExtractDoubleList(x_child, x_offset + col));
            }

            for (auto &col : x_cols) {
                AnofoxDataArray arr;
                arr.data = col.data();
                arr.validity = nullptr;
                arr.len = col.size();
                x_arrays.push_back(arr);
            }

            if (!x_arrays.empty()) {
                x_ptr = x_arrays.data();
                x_count = x_arrays.size();
            }
        }

        // Get residual_std_error if provided
        double rse = std::nan("");
        if (has_rse) {
            auto &rse_vec = args.data[3];
            auto rse_data = FlatVector::GetData<double>(rse_vec);
            rse = rse_data[row];
        }

        // Get include_studentized flag
        bool include_studentized = true;
        if (has_studentized_flag) {
            auto &flag_vec = args.data[4];
            auto flag_data = FlatVector::GetData<bool>(flag_vec);
            include_studentized = flag_data[row];
        }

        // Call Rust FFI
        AnofoxResidualsResult resid_result;
        AnofoxError error;

        bool success = anofox_compute_residuals(
            y_array,
            y_hat_array,
            x_ptr,
            x_count,
            rse,
            include_studentized,
            &resid_result,
            &error
        );

        if (!success) {
            FlatVector::SetNull(result, row, true);
            continue;
        }

        // Fill result struct
        idx_t struct_idx = 0;

        // Raw residuals
        SetListInResult(*struct_entries[struct_idx++], row, resid_result.raw, resid_result.len);

        // Standardized residuals
        if (resid_result.has_standardized) {
            SetListInResult(*struct_entries[struct_idx++], row, resid_result.standardized, resid_result.len);
        } else {
            FlatVector::SetNull(*struct_entries[struct_idx++], row, true);
        }

        // Studentized residuals
        if (resid_result.has_studentized) {
            SetListInResult(*struct_entries[struct_idx++], row, resid_result.studentized, resid_result.len);
        } else {
            FlatVector::SetNull(*struct_entries[struct_idx++], row, true);
        }

        // Leverage
        if (resid_result.has_leverage) {
            SetListInResult(*struct_entries[struct_idx++], row, resid_result.leverage, resid_result.len);
        } else {
            FlatVector::SetNull(*struct_entries[struct_idx++], row, true);
        }

        anofox_free_residuals(&resid_result);
    }
}

// Register the function
void RegisterResidualsDiagnosticsFunction(ExtensionLoader &loader) {
    ScalarFunctionSet func_set("anofox_stats_residuals_diagnostics");

    // Basic version: anofox_stats_residuals_diagnostics(y, y_hat)
    auto basic_func = ScalarFunction(
        {LogicalType::LIST(LogicalType::DOUBLE), LogicalType::LIST(LogicalType::DOUBLE)},
        LogicalType::ANY,  // Set in bind
        ResidualsDiagnosticsFunction,
        ResidualsDiagnosticsBind
    );
    func_set.AddFunction(basic_func);

    // Full version: anofox_stats_residuals_diagnostics(y, y_hat, x, residual_std_error, include_studentized)
    auto full_func = ScalarFunction(
        {LogicalType::LIST(LogicalType::DOUBLE),
         LogicalType::LIST(LogicalType::DOUBLE),
         LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)),
         LogicalType::DOUBLE,
         LogicalType::BOOLEAN},
        LogicalType::ANY,
        ResidualsDiagnosticsFunction,
        ResidualsDiagnosticsBind
    );
    func_set.AddFunction(full_func);

    loader.RegisterFunction(func_set);

    // Also register short alias
    ScalarFunctionSet alias_set("residuals_diagnostics");
    alias_set.AddFunction(basic_func);
    alias_set.AddFunction(full_func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
