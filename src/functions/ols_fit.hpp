#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Ordinary Least Squares (OLS) regression table function
 *
 * Performs standard linear regression without regularization (lambda=0).
 * OLS finds coefficients that minimize the sum of squared residuals.
 *
 * Signature:
 *   SELECT * FROM anofox_statistics_ols(
 *       y := [1.0, 2.0, 3.0, 4.0],
 *       x := [[1.1, 2.1, 2.9, 4.2], [0.5, 1.5, 2.5, 3.5]],
 *       options := MAP{'intercept': true}
 *   )
 *
 * Model: min ||y - Xβ||²
 *   Solution: β = (X'X)^(-1) X'y
 *
 * Returns:
 *   - coefficients: DOUBLE[] - OLS regression coefficients
 *   - intercept: DOUBLE - Model intercept
 *   - r2: DOUBLE - R² coefficient of determination
 *   - adj_r2: DOUBLE - Adjusted R²
 *   - mse: DOUBLE - Mean Squared Error
 *   - rmse: DOUBLE - Root Mean Squared Error
 *   - n_obs: BIGINT - Number of observations
 *   - n_features: BIGINT - Number of features
 */
class OlsFitFunction {
public:
	/**
	 * @brief Register the anofox_statistics_ols table function
	 *
	 * @param loader Extension loader context
	 */
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
