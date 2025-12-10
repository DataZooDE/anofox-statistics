#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Residual Diagnostics - Comprehensive residual analysis
 *
 * Computes diagnostic metrics for each observation:
 * - Residuals (actual - predicted)
 * - Standardized residuals
 * - Studentized residuals
 * - Leverage (hat values)
 * - Cook's Distance (influence measure)
 * - DFFITS (change in fitted value)
 * - Outlier flags
 * - Influential point flags
 *
 * Usage:
 *   SELECT * FROM residual_diagnostics(
 *       y := [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
 *       x := [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],
 *       add_intercept := true,
 *       outlier_threshold := 2.5,
 *       influence_threshold := 0.5
 *   );
 *
 * Returns one row per observation with:
 * - obs_id: Observation ID (1-indexed)
 * - residual: Raw residual
 * - std_residual: Standardized residual
 * - studentized_residual: Studentized residual
 * - leverage: Hat value (diagonal of hat matrix)
 * - cooks_distance: Cook's D (influence measure)
 * - dffits: Change in fitted value when removing point
 * - is_outlier: Flag for outlier (|studentized| > threshold)
 * - is_influential: Flag for influential point (Cook's D > threshold)
 */
class ResidualDiagnosticsFunction {
public:
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
