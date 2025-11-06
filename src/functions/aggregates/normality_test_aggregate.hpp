#pragma once

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * Normality Test Aggregate Function
 *
 * Performs Jarque-Bera normality test on grouped residuals.
 *
 * Usage:
 *   SELECT category,
 *          anofox_statistics_normality_test_agg(residual, MAP{'alpha': 0.05})
 *   FROM residual_data
 *   GROUP BY category;
 */
struct NormalityTestAggregateFunction {
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
