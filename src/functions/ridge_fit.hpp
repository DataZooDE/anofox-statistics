#pragma once

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include <vector>
#include <string>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Ridge regression (L2 regularization) table function
 *
 * Performs linear regression with L2 penalty (Tikhonov regularization).
 * Ridge regression adds a penalty term to OLS to reduce model complexity and
 * improve generalization, especially useful when features are correlated.
 *
 * Signature:
 *   SELECT * FROM anofox_ridge_fit(
 *       data := <table>,
 *       x_cols := [<col_name>, ...],
 *       y_col := <col_name>,
 *       lambda := <DOUBLE>,        -- Regularization strength (default 0.01)
 *       add_intercept := <BOOLEAN> -- Include intercept (default true)
 *   )
 *
 * Model: min ||y - Xβ||² + λ||β||²
 *   where λ is the regularization parameter
 *
 * Returns:
 *   - coefficients: DOUBLE[] - Ridge regression coefficients
 *   - intercept: DOUBLE - Model intercept
 *   - r2: DOUBLE - R² on training data
 *   - adj_r2: DOUBLE - Adjusted R²
 *   - mse: DOUBLE - Mean Squared Error
 *   - rmse: DOUBLE - Root Mean Squared Error
 *   - n_obs: BIGINT - Number of observations
 *   - n_features: BIGINT - Number of features
 *   - lambda: DOUBLE - Regularization parameter used
 *   - lambda_optimal: DOUBLE - Cross-validation optimal lambda (if computed)
 *
 * Examples:
 *   -- Basic Ridge with default lambda
 *   SELECT * FROM anofox_ridge_fit(
 *       (SELECT x1, x2, y FROM my_data),
 *       ['x1', 'x2'],
 *       'y',
 *       0.1
 *   );
 *
 *   -- Ridge with custom lambda
 *   SELECT * FROM anofox_ridge_fit(
 *       (SELECT x1, x2, y FROM my_data),
 *       ['x1', 'x2'],
 *       'y',
 *       lambda := 1.0,
 *       add_intercept := true
 *   );
 */
class RidgeFitFunction {
public:
	/**
	 * @brief Register the anofox_ridge_fit table function
	 *
	 * @param loader Extension loader context
	 */
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
