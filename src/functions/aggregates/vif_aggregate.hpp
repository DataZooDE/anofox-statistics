#pragma once

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * VIF (Variance Inflation Factor) Aggregate Function
 *
 * Computes VIF for each feature to detect multicollinearity on grouped data.
 *
 * Usage:
 *   SELECT category,
 *          anofox_statistics_vif_agg([x1, x2, x3, x4])
 *   FROM data
 *   GROUP BY category;
 */
struct VifAggregateFunction {
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
