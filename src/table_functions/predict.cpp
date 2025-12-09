#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "../include/anofox_stats_ffi.h"

namespace duckdb {

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

// Predict function: anofox_stats_predict(x, coefficients, intercept) -> LIST(DOUBLE)
static void PredictFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &x_vec = args.data[0];         // LIST(LIST(DOUBLE)) - new feature data
    auto &coef_vec = args.data[1];      // LIST(DOUBLE) - coefficients
    auto &intercept_vec = args.data[2]; // DOUBLE - intercept (can be NULL)

    idx_t count = args.size();

    // Flatten intercept if needed
    UnifiedVectorFormat intercept_data;
    intercept_vec.ToUnifiedFormat(count, intercept_data);
    auto intercept_values = UnifiedVectorFormat::GetData<double>(intercept_data);

    // Process each row
    for (idx_t row = 0; row < count; row++) {
        // Extract coefficients
        vector<double> coefficients = ExtractDoubleList(coef_vec, row);

        // Extract intercept (use NaN if NULL)
        auto intercept_idx = intercept_data.sel->get_index(row);
        double intercept =
            intercept_data.validity.RowIsValid(intercept_idx) ? intercept_values[intercept_idx] : std::nan("");

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
        vector<AnofoxDataArray> x_arrays;
        for (auto &col : x_cols) {
            AnofoxDataArray arr;
            arr.data = col.data();
            arr.validity = nullptr;
            arr.len = col.size();
            x_arrays.push_back(arr);
        }

        // Call Rust FFI
        double *predictions = nullptr;
        size_t predictions_len = 0;
        AnofoxError error;

        bool success = anofox_predict(x_arrays.data(), x_arrays.size(), coefficients.data(), coefficients.size(),
                                      intercept, &predictions, &predictions_len, &error);

        if (!success) {
            throw InvalidInputException("Predict failed: %s", error.message);
        }

        // Build result list
        auto &result_child = ListVector::GetEntry(result);
        auto result_offset = ListVector::GetListSize(result);
        ListVector::SetListSize(result, result_offset + predictions_len);
        auto result_data = FlatVector::GetData<double>(result_child);

        for (size_t i = 0; i < predictions_len; i++) {
            result_data[result_offset + i] = predictions[i];
        }
        ListVector::GetData(result)[row] = {result_offset, predictions_len};

        // Free predictions
        anofox_free_predictions(predictions);
    }

    result.SetVectorType(VectorType::FLAT_VECTOR);
}

// Register the function
void RegisterPredictFunction(ExtensionLoader &loader) {
    ScalarFunctionSet func_set("anofox_stats_predict");

    // anofox_stats_predict(x, coefficients, intercept)
    // x: LIST(LIST(DOUBLE)) - feature data (list of feature columns, each a list of values)
    // coefficients: LIST(DOUBLE) - fitted coefficients
    // intercept: DOUBLE - intercept value (can be NULL)
    ScalarFunction func({LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)),
                         LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE},
                        LogicalType::LIST(LogicalType::DOUBLE), PredictFunction);
    func_set.AddFunction(func);

    loader.RegisterFunction(func_set);
}

} // namespace duckdb
