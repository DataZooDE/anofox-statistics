#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "telemetry.hpp"

namespace duckdb {

// AIC function: anofox_stats_aic(rss, n, k) -> DOUBLE
static void AicFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    PostHogTelemetry::Instance().CaptureFunctionExecution("aic");
    auto &rss_vec = args.data[0];
    auto &n_vec = args.data[1];
    auto &k_vec = args.data[2];

    idx_t count = args.size();

    UnifiedVectorFormat rss_data, n_data, k_data;
    rss_vec.ToUnifiedFormat(count, rss_data);
    n_vec.ToUnifiedFormat(count, n_data);
    k_vec.ToUnifiedFormat(count, k_data);

    auto rss_values = UnifiedVectorFormat::GetData<double>(rss_data);
    auto n_values = UnifiedVectorFormat::GetData<int64_t>(n_data);
    auto k_values = UnifiedVectorFormat::GetData<int64_t>(k_data);

    auto result_data = FlatVector::GetData<double>(result);

    for (idx_t row = 0; row < count; row++) {
        auto rss_idx = rss_data.sel->get_index(row);
        auto n_idx = n_data.sel->get_index(row);
        auto k_idx = k_data.sel->get_index(row);

        if (!rss_data.validity.RowIsValid(rss_idx) || !n_data.validity.RowIsValid(n_idx) ||
            !k_data.validity.RowIsValid(k_idx)) {
            FlatVector::SetNull(result, row, true);
            continue;
        }

        double rss = rss_values[rss_idx];
        size_t n = static_cast<size_t>(n_values[n_idx]);
        size_t k = static_cast<size_t>(k_values[k_idx]);

        double aic;
        AnofoxError error;

        bool success = anofox_compute_aic(rss, n, k, &aic, &error);

        if (!success) {
            throw InvalidInputException("AIC computation failed: " + string(error.message));
        }

        result_data[row] = aic;
    }

    result.SetVectorType(VectorType::FLAT_VECTOR);
}

// BIC function: anofox_stats_bic(rss, n, k) -> DOUBLE
static void BicFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    PostHogTelemetry::Instance().CaptureFunctionExecution("bic");
    auto &rss_vec = args.data[0];
    auto &n_vec = args.data[1];
    auto &k_vec = args.data[2];

    idx_t count = args.size();

    UnifiedVectorFormat rss_data, n_data, k_data;
    rss_vec.ToUnifiedFormat(count, rss_data);
    n_vec.ToUnifiedFormat(count, n_data);
    k_vec.ToUnifiedFormat(count, k_data);

    auto rss_values = UnifiedVectorFormat::GetData<double>(rss_data);
    auto n_values = UnifiedVectorFormat::GetData<int64_t>(n_data);
    auto k_values = UnifiedVectorFormat::GetData<int64_t>(k_data);

    auto result_data = FlatVector::GetData<double>(result);

    for (idx_t row = 0; row < count; row++) {
        auto rss_idx = rss_data.sel->get_index(row);
        auto n_idx = n_data.sel->get_index(row);
        auto k_idx = k_data.sel->get_index(row);

        if (!rss_data.validity.RowIsValid(rss_idx) || !n_data.validity.RowIsValid(n_idx) ||
            !k_data.validity.RowIsValid(k_idx)) {
            FlatVector::SetNull(result, row, true);
            continue;
        }

        double rss = rss_values[rss_idx];
        size_t n = static_cast<size_t>(n_values[n_idx]);
        size_t k = static_cast<size_t>(k_values[k_idx]);

        double bic;
        AnofoxError error;

        bool success = anofox_compute_bic(rss, n, k, &bic, &error);

        if (!success) {
            throw InvalidInputException("BIC computation failed: " + string(error.message));
        }

        result_data[row] = bic;
    }

    result.SetVectorType(VectorType::FLAT_VECTOR);
}

// Register the functions
void RegisterAicBicFunctions(ExtensionLoader &loader) {
    ScalarFunction aic_func({LogicalType::DOUBLE, LogicalType::BIGINT, LogicalType::BIGINT}, LogicalType::DOUBLE, AicFunction);
    ScalarFunction bic_func({LogicalType::DOUBLE, LogicalType::BIGINT, LogicalType::BIGINT}, LogicalType::DOUBLE, BicFunction);

    // AIC primary
    {
        ScalarFunctionSet aic_set("anofox_stats_aic");
        aic_set.AddFunction(aic_func);
        CreateScalarFunctionInfo info(std::move(aic_set));
        info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        FunctionDescription desc;
        desc.description     = "Computes Akaike Information Criterion (AIC) from residual sum of squares, number of observations, and number of parameters.";
        desc.examples        = {"anofox_stats_aic(rss, n, k)"};
        desc.categories      = {"model-selection"};
        desc.parameter_names = {"rss", "n", "k"};
        desc.parameter_types = {LogicalType::DOUBLE, LogicalType::BIGINT, LogicalType::BIGINT};
        info.descriptions.push_back(std::move(desc));
        loader.RegisterFunction(std::move(info));
    }
    // AIC alias
    {
        ScalarFunctionSet alias_set("aic");
        alias_set.AddFunction(aic_func);
        CreateScalarFunctionInfo alias_info(std::move(alias_set));
        alias_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        alias_info.alias_of = "anofox_stats_aic";
        loader.RegisterFunction(std::move(alias_info));
    }

    // BIC primary
    {
        ScalarFunctionSet bic_set("anofox_stats_bic");
        bic_set.AddFunction(bic_func);
        CreateScalarFunctionInfo info(std::move(bic_set));
        info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        FunctionDescription desc;
        desc.description     = "Computes Bayesian Information Criterion (BIC) from residual sum of squares, number of observations, and number of parameters.";
        desc.examples        = {"anofox_stats_bic(rss, n, k)"};
        desc.categories      = {"model-selection"};
        desc.parameter_names = {"rss", "n", "k"};
        desc.parameter_types = {LogicalType::DOUBLE, LogicalType::BIGINT, LogicalType::BIGINT};
        info.descriptions.push_back(std::move(desc));
        loader.RegisterFunction(std::move(info));
    }
    // BIC alias
    {
        ScalarFunctionSet alias_set("bic");
        alias_set.AddFunction(bic_func);
        CreateScalarFunctionInfo alias_info(std::move(alias_set));
        alias_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        alias_info.alias_of = "anofox_stats_bic";
        loader.RegisterFunction(std::move(alias_info));
    }
}

} // namespace duckdb
