#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Information Criteria - Model selection metrics
 *
 * Computes various information criteria and metrics for model comparison:
 * - AIC (Akaike Information Criterion)
 * - BIC (Bayesian Information Criterion)
 * - AICc (Corrected AIC for small samples)
 * - Adjusted R²
 *
 * Usage:
 *   SELECT * FROM information_criteria(
 *       y := [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
 *       x := [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],
 *       add_intercept := true
 *   );
 *
 * Returns single row with:
 * - n_obs: Number of observations
 * - n_params: Number of parameters (including intercept)
 * - rss: Residual sum of squares
 * - r_squared: R²
 * - adj_r_squared: Adjusted R²
 * - aic: Akaike Information Criterion
 * - bic: Bayesian Information Criterion
 * - aicc: Corrected AIC (small sample)
 * - log_likelihood: Log-likelihood
 *
 * Lower AIC/BIC indicates better model fit (penalizes complexity)
 */
class InformationCriteriaFunction {
public:
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
