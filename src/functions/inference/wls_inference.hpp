#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief WLS Inference - Statistical inference for Weighted Least Squares regression
 *
 * Returns comprehensive inference for WLS regression including:
 * - Coefficient estimates
 * - Heteroscedasticity-consistent standard errors
 * - t-statistics
 * - p-values (two-tailed)
 * - Confidence intervals
 * - Significance flags
 *
 * Note: Standard errors account for heteroscedasticity via observation weights.
 *
 * Usage:
 *   SELECT * FROM anofox_statistics_wls_inference(
 *       y := [1.0, 2.0, 3.0]::DOUBLE[],
 *       x := [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]::DOUBLE[][],
 *       weights := [1.0, 2.0, 1.5]::DOUBLE[],
 *       options := MAP{'confidence_level': 0.95, 'intercept': true}
 *   );
 */
class WlsInferenceFunction {
public:
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
