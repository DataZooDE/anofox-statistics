#pragma once

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * Residual Diagnostics Aggregate Function
 *
 * Computes diagnostic statistics for residuals on grouped data.
 *
 * Usage:
 *   -- Summary mode (default)
 *   SELECT category,
 *          anofox_statistics_residual_diagnostics_agg(y_actual, y_predicted, MAP{'outlier_threshold': 2.5})
 *   FROM predictions
 *   GROUP BY category;
 *
 *   -- Detailed mode (returns arrays)
 *   SELECT category,
 *          anofox_statistics_residual_diagnostics_agg(y_actual, y_predicted, MAP{'detailed': true})
 *   FROM predictions
 *   GROUP BY category;
 */
struct ResidualDiagnosticsAggregateFunction {
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
