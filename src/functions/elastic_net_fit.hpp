#pragma once

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Elastic Net regression table function (combined L1 + L2 regularization)
 *
 * Array-based API with MAP options.
 * Performs linear regression with both L1 (Lasso) and L2 (Ridge) penalties using coordinate descent.
 *
 * Signature:
 *   SELECT * FROM anofox_statistics_elastic_net(
 *       y := [1.0, 2.0, 3.0, 4.0],
 *       x := [[1.1, 2.1, 2.9, 4.2], [0.5, 1.5, 2.5, 3.5]],
 *       options := MAP{'intercept': true, 'alpha': 0.5, 'lambda': 0.01}
 *   )
 *
 * Parameters:
 *   - y: DOUBLE[] - Response variable
 *   - x: DOUBLE[][] - Feature matrix (each inner array is one feature)
 *   - options: MAP - Optional configuration:
 *     - 'intercept': BOOLEAN (default true)
 *     - 'alpha': DOUBLE in [0,1] (default 0.5) - L1 vs L2 mix (0=Ridge, 1=Lasso)
 *     - 'lambda': DOUBLE >= 0 (default 0.01) - Regularization strength
 *
 * Model: min ||y - Xβ||² + λ(α||β||₁ + (1-α)||β||₂²)
 *
 * Returns:
 *   - coefficients: DOUBLE[] - Fitted coefficients
 *   - intercept: DOUBLE - Model intercept
 *   - r2: DOUBLE - R² statistic
 *   - adj_r2: DOUBLE - Adjusted R²
 *   - mse: DOUBLE - Mean Squared Error
 *   - rmse: DOUBLE - Root Mean Squared Error
 *   - n_obs: BIGINT - Number of observations
 *   - n_features: BIGINT - Number of features
 *   - alpha: DOUBLE - Alpha parameter used
 *   - lambda: DOUBLE - Lambda parameter used
 *   - n_nonzero: BIGINT - Number of non-zero coefficients (sparsity)
 *
 * Examples:
 *   -- Balanced Elastic Net (alpha=0.5)
 *   SELECT * FROM anofox_statistics_elastic_net(
 *       [10.0, 20.0, 30.0, 40.0],
 *       [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]],
 *       MAP{'intercept': true, 'alpha': 0.5, 'lambda': 0.1}
 *   );
 *
 *   -- Lasso-like (alpha=0.9 for sparsity)
 *   SELECT * FROM anofox_statistics_elastic_net(
 *       y_array,
 *       x_matrix,
 *       MAP{'alpha': 0.9, 'lambda': 0.05}
 *   );
 */
class ElasticNetFitFunction {
public:
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
