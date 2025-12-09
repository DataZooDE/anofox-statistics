#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/types/data_chunk.hpp"

#include "../include/anofox_stats_ffi.h"

namespace duckdb {

// AIC function: anofox_stats_aic(rss, n, k) -> DOUBLE
static void AicFunction(DataChunk &args, ExpressionState &state, Vector &result) {
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

        if (!rss_data.validity.RowIsValid(rss_idx) ||
            !n_data.validity.RowIsValid(n_idx) ||
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
            throw InvalidInputException("AIC computation failed: %s", error.message);
        }

        result_data[row] = aic;
    }

    result.SetVectorType(VectorType::FLAT_VECTOR);
}

// BIC function: anofox_stats_bic(rss, n, k) -> DOUBLE
static void BicFunction(DataChunk &args, ExpressionState &state, Vector &result) {
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

        if (!rss_data.validity.RowIsValid(rss_idx) ||
            !n_data.validity.RowIsValid(n_idx) ||
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
            throw InvalidInputException("BIC computation failed: %s", error.message);
        }

        result_data[row] = bic;
    }

    result.SetVectorType(VectorType::FLAT_VECTOR);
}

// Register the functions
void RegisterAicBicFunctions(ExtensionLoader &loader) {
    // AIC function
    ScalarFunctionSet aic_set("anofox_stats_aic");
    ScalarFunction aic_func(
        {LogicalType::DOUBLE, LogicalType::BIGINT, LogicalType::BIGINT},
        LogicalType::DOUBLE,
        AicFunction
    );
    aic_set.AddFunction(aic_func);
    loader.RegisterFunction(aic_set);

    // AIC short alias
    ScalarFunctionSet aic_alias("aic");
    aic_alias.AddFunction(aic_func);
    loader.RegisterFunction(aic_alias);

    // BIC function
    ScalarFunctionSet bic_set("anofox_stats_bic");
    ScalarFunction bic_func(
        {LogicalType::DOUBLE, LogicalType::BIGINT, LogicalType::BIGINT},
        LogicalType::DOUBLE,
        BicFunction
    );
    bic_set.AddFunction(bic_func);
    loader.RegisterFunction(bic_set);

    // BIC short alias
    ScalarFunctionSet bic_alias("bic");
    bic_alias.AddFunction(bic_func);
    loader.RegisterFunction(bic_alias);
}

} // namespace duckdb
