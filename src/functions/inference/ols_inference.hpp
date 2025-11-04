#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief OLS Inference - Statistical inference for OLS regression
 *
 * Returns comprehensive inference including:
 * - Coefficient estimates
 * - Standard errors
 * - t-statistics
 * - p-values (two-tailed)
 * - Confidence intervals
 * - Significance flags
 *
 * Usage:
 *   SELECT * FROM ols_inference(
 *       y := [1.0, 2.0, 3.0]::DOUBLE[],
 *       x := [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]::DOUBLE[][],
 *       confidence_level := 0.95,
 *       add_intercept := true
 *   );
 *
 * Returns one row per coefficient with:
 * - variable: Variable name ('intercept', 'x1', 'x2', ...)
 * - estimate: Coefficient estimate
 * - std_error: Standard error of coefficient
 * - t_statistic: t-statistic for H0: β = 0
 * - p_value: Two-tailed p-value
 * - ci_lower: Lower confidence interval
 * - ci_upper: Upper confidence interval
 * - significant: Boolean flag (p < 0.05)
 */
class OlsInferenceFunction {
public:
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
