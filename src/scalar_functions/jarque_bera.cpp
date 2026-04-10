#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/vector_operations/generic_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "telemetry.hpp"

namespace duckdb {

// Result struct type for Jarque-Bera
static LogicalType GetJarqueBeraResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("statistic", LogicalType::DOUBLE));
    children.push_back(make_pair("p_value", LogicalType::DOUBLE));
    children.push_back(make_pair("skewness", LogicalType::DOUBLE));
    children.push_back(make_pair("kurtosis", LogicalType::DOUBLE));
    children.push_back(make_pair("n", LogicalType::BIGINT));

    return LogicalType::STRUCT(std::move(children));
}

// Bind function
static unique_ptr<FunctionData> JarqueBeraBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {
    bound_function.return_type = GetJarqueBeraResultType();
    PostHogTelemetry::Instance().CaptureFunctionExecution("jarque_bera");
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

// Main Jarque-Bera function
static void JarqueBeraFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &data_vec = args.data[0]; // LIST(DOUBLE)

    idx_t count = args.size();
    auto &struct_entries = StructVector::GetEntries(result);

    // Process each row
    for (idx_t row = 0; row < count; row++) {
        // Extract data values
        vector<double> data = ExtractDoubleList(data_vec, row);

        if (data.size() < 3) {
            FlatVector::SetNull(result, row, true);
            continue;
        }

        // Prepare FFI data
        AnofoxDataArray data_array;
        data_array.data = data.data();
        data_array.validity = nullptr;
        data_array.len = data.size();

        // Call Rust FFI
        AnofoxJarqueBeraResult jb_result;
        AnofoxError error;

        bool success = anofox_jarque_bera(data_array, &jb_result, &error);

        if (!success) {
            FlatVector::SetNull(result, row, true);
            continue;
        }

        // Fill result struct
        idx_t struct_idx = 0;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[row] = jb_result.statistic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[row] = jb_result.p_value;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[row] = jb_result.skewness;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[row] = jb_result.kurtosis;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[row] = jb_result.n;
    }
}

// Register the function
void RegisterJarqueBeraFunction(ExtensionLoader &loader) {
    auto func = ScalarFunction({LogicalType::LIST(LogicalType::DOUBLE)},
                               LogicalType::ANY,
                               JarqueBeraFunction, JarqueBeraBind);

    // Primary
    {
        ScalarFunctionSet func_set("anofox_stats_jarque_bera");
        func_set.AddFunction(func);
        CreateScalarFunctionInfo info(std::move(func_set));
        info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        FunctionDescription desc;
        desc.description     = "Tests whether a sample has skewness and kurtosis consistent with a normal distribution (Jarque-Bera test).";
        desc.examples        = {"anofox_stats_jarque_bera(values)"};
        desc.categories      = {"normality-test"};
        desc.parameter_names = {"values"};
        desc.parameter_types = {LogicalType::LIST(LogicalType::DOUBLE)};
        info.descriptions.push_back(std::move(desc));
        loader.RegisterFunction(std::move(info));
    }
    // Alias
    {
        ScalarFunctionSet alias_set("jarque_bera");
        alias_set.AddFunction(func);
        CreateScalarFunctionInfo alias_info(std::move(alias_set));
        alias_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        alias_info.alias_of = "anofox_stats_jarque_bera";
        loader.RegisterFunction(std::move(alias_info));
    }
}

} // namespace duckdb
