#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Ridge Inference - Statistical inference for Ridge regression
 *
 * Returns comprehensive inference for Ridge regression including:
 * - Coefficient estimates (biased by regularization)
 * - Standard errors (accounting for ridge penalty)
 * - t-statistics
 * - p-values (two-tailed)
 * - Confidence intervals
 * - Significance flags
 *
 * Note: Ridge coefficients are biased estimates due to L2 regularization.
 * Standard errors account for the bias introduced by the penalty term.
 *
 * Usage:
 *   SELECT * FROM anofox_statistics_ridge_inference(
 *       y := [1.0, 2.0, 3.0]::DOUBLE[],
 *       x := [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]::DOUBLE[][],
 *       options := MAP{'lambda': 1.0, 'confidence_level': 0.95, 'intercept': true}
 *   );
 *
 * Returns one row per coefficient with:
 * - variable: Variable name ('intercept', 'x1', 'x2', ...)
 * - estimate: Ridge coefficient estimate
 * - std_error: Standard error of coefficient (ridge-adjusted)
 * - t_statistic: t-statistic for H0: β = 0
 * - p_value: Two-tailed p-value
 * - ci_lower: Lower confidence interval
 * - ci_upper: Upper confidence interval
 * - significant: Boolean flag (p < alpha)
 */
class RidgeInferenceFunction {
public:
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
