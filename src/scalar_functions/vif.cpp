#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/types/data_chunk.hpp"

#include "../include/anofox_stats_ffi.h"

#include <vector>

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

// VIF function: anofox_stats_vif(x) -> LIST(DOUBLE)
// x: LIST(LIST(DOUBLE)) - feature data (list of feature columns)
static void VifFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &x_vec = args.data[0];  // LIST(LIST(DOUBLE))

    idx_t count = args.size();

    for (idx_t row = 0; row < count; row++) {
        // Extract x values (list of feature columns)
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
        double *vif_values = nullptr;
        size_t vif_len = 0;
        AnofoxError error;

        bool success = anofox_compute_vif(
            x_arrays.data(),
            x_arrays.size(),
            &vif_values,
            &vif_len,
            &error
        );

        if (!success) {
            throw InvalidInputException("VIF computation failed: %s", error.message);
        }

        // Build result list
        auto &result_child = ListVector::GetEntry(result);
        auto result_offset = ListVector::GetListSize(result);
        ListVector::SetListSize(result, result_offset + vif_len);
        auto result_data = FlatVector::GetData<double>(result_child);

        for (size_t i = 0; i < vif_len; i++) {
            result_data[result_offset + i] = vif_values[i];
        }
        ListVector::GetData(result)[row] = {result_offset, vif_len};

        // Free VIF values
        anofox_free_vif(vif_values);
    }

    result.SetVectorType(VectorType::FLAT_VECTOR);
}

// Register the function
void RegisterVifFunction(ExtensionLoader &loader) {
    ScalarFunctionSet func_set("anofox_stats_vif");

    // anofox_stats_vif(x) -> LIST(DOUBLE)
    ScalarFunction func(
        {LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE))},
        LogicalType::LIST(LogicalType::DOUBLE),
        VifFunction
    );
    func_set.AddFunction(func);

    loader.RegisterFunction(func_set);

    // Also register short alias
    ScalarFunctionSet alias_set("vif");
    alias_set.AddFunction(func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
