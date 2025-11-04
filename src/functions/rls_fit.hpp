#pragma once

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include <vector>
#include <string>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Recursive Least Squares (RLS) regression - Online learning algorithm
 *
 * Performs sequential/online linear regression that updates coefficients
 * as each new observation arrives. Useful for streaming data and time-series
 * with changing relationships.
 *
 * Signature:
 *   SELECT * FROM anofox_statistics_rls_fit(
 *       y := <DOUBLE[]>,
 *       x1 := <DOUBLE[]>,
 *       [x2 := <DOUBLE[]>, ...],
 *       [lambda := <DOUBLE>],           -- Forgetting factor (default: 1.0)
 *       [add_intercept := <BOOLEAN>]    -- Default: true
 *   )
 *
 * Algorithm:
 *   β_t = β_{t-1} + K_t * (y_t - x_t'β_{t-1})
 *   K_t = P_{t-1}x_t / (λ + x_t'P_{t-1}x_t)
 *   P_t = (1/λ) * (P_{t-1} - K_t x_t' P_{t-1})
 *
 * Where:
 *   - λ is the forgetting factor (0 < λ ≤ 1)
 *   - λ = 1.0: All observations equally weighted (standard RLS)
 *   - λ < 1.0: Exponential weighting (recent data more important)
 *   - P_t is the inverse covariance matrix
 *
 * Returns:
 *   - coefficients: DOUBLE[] - Final RLS regression coefficients
 *   - intercept: DOUBLE - Model intercept (if add_intercept=true)
 *   - r_squared: DOUBLE - R² on final predictions
 *   - adj_r_squared: DOUBLE - Adjusted R²
 *   - mse: DOUBLE - Mean Squared Error
 *   - rmse: DOUBLE - Root Mean Squared Error
 *   - lambda: DOUBLE - Forgetting factor used
 *   - n_obs: BIGINT - Number of observations
 *   - n_features: BIGINT - Number of features
 *
 * Examples:
 *   -- Standard RLS (λ=1.0, all data equally weighted)
 *   SELECT * FROM anofox_statistics_rls_fit(
 *       [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
 *       [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[]
 *   );
 *
 *   -- RLS with forgetting factor (λ=0.95, recent data more important)
 *   SELECT * FROM anofox_statistics_rls_fit(
 *       [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
 *       [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
 *       0.95
 *   );
 *
 *   -- Multi-variable RLS
 *   SELECT * FROM anofox_statistics_rls_fit(
 *       [10.0, 20.0, 30.0, 40.0, 50.0]::DOUBLE[],
 *       [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
 *       [5.0, 10.0, 15.0, 20.0, 25.0]::DOUBLE[],
 *       0.98
 *   );
 */
class RlsFitFunction {
public:
	/**
	 * @brief Register the anofox_statistics_rls_fit table function
	 *
	 * @param loader Extension loader context
	 */
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
