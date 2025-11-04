#pragma once

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include <vector>
#include <string>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Weighted Least Squares (WLS) regression table function
 *
 * Performs linear regression with case weights (heteroscedasticity-consistent).
 * Useful when observations have different reliabilities or variances.
 *
 * Signature:
 *   SELECT * FROM anofox_wls_fit(
 *       data := <table>,
 *       x_cols := [<col_name>, ...],
 *       y_col := <col_name>,
 *       weights_col := <col_name>,
 *       add_intercept := <BOOLEAN> (default true)
 *   )
 *
 * Model: min Σ(w_i * (y_i - Σ(β_j * x_ij))²)
 *   where w_i are the weights for each observation
 *
 * Returns:
 *   - coefficients: DOUBLE[] - WLS regression coefficients
 *   - intercept: DOUBLE - Model intercept
 *   - r_squared: DOUBLE - Weighted R²
 *   - adj_r_squared: DOUBLE - Adjusted weighted R²
 *   - mse: DOUBLE - Weighted Mean Squared Error
 *   - rmse: DOUBLE - Weighted Root Mean Squared Error
 *   - n_obs: BIGINT - Number of observations
 *   - n_features: BIGINT - Number of features
 *   - sum_weights: DOUBLE - Sum of all weights used
 *
 * Examples:
 *   -- Basic WLS with weights column
 *   SELECT * FROM anofox_wls_fit(
 *       (SELECT x1, x2, y, weight FROM my_data),
 *       ['x1', 'x2'],
 *       'y',
 *       'weight'
 *   );
 *
 *   -- WLS without intercept
 *   SELECT * FROM anofox_wls_fit(
 *       (SELECT x1, x2, y, 1.0/std_error FROM my_data),
 *       ['x1', 'x2'],
 *       'y',
 *       weights_col := 'weight',
 *       add_intercept := false
 *   );
 */
class WlsFitFunction {
public:
	/**
	 * @brief Register the anofox_statistics_wls_fit table function
	 *
	 * @param loader Extension loader context
	 */
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
