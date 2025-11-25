#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief RLS Inference - Statistical inference for Recursive Least Squares
 *
 * Returns comprehensive inference for RLS regression including:
 * - Coefficient estimates (from final iteration)
 * - Standard errors
 * - t-statistics
 * - p-values (two-tailed)
 * - Confidence intervals
 * - Significance flags
 *
 * Note: Tests coefficients from the final iteration of recursive updates.
 *
 * Usage:
 *   SELECT * FROM anofox_statistics_rls_inference(
 *       y := [1.0, 2.0, 3.0, 4.0]::DOUBLE[],
 *       x := [[1.0], [2.0], [3.0], [4.0]]::DOUBLE[][],
 *       options := MAP{'confidence_level': 0.95, 'intercept': true, 'forgetting_factor': 0.99}
 *   );
 */
class RlsInferenceFunction {
public:
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
