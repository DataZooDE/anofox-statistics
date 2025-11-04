#pragma once

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include <vector>
#include <string>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief ElasticNet regression (combined L1 + L2 regularization) table function
 *
 * Performs linear regression with both L1 (Lasso) and L2 (Ridge) penalties.
 * Combines the benefits of both Ridge (handles multicollinearity) and Lasso
 * (performs feature selection).
 *
 * Signature:
 *   SELECT * FROM anofox_elastic_net_fit(
 *       data := <table>,
 *       x_cols := [<col_name>, ...],
 *       y_col := <col_name>,
 *       alpha := <DOUBLE>,        -- Weight of L1 vs L2 (0=Ridge, 1=Lasso)
 *       lambda := <DOUBLE>,       -- Regularization strength
 *       add_intercept := <BOOLEAN> (default true)
 *   )
 *
 * Model: min ||y - Xβ||² + λ(α||β||₁ + (1-α)||β||₂²)
 *   where:
 *     α ∈ [0, 1] controls L1 (α=0) vs L2 (α=1) trade-off
 *     λ > 0 controls regularization strength
 *
 * Returns:
 *   - coefficients: DOUBLE[] - ElasticNet coefficients
 *   - intercept: DOUBLE - Model intercept
 *   - r_squared: DOUBLE - R² on training data
 *   - adj_r_squared: DOUBLE - Adjusted R²
 *   - mse: DOUBLE - Mean Squared Error
 *   - rmse: DOUBLE - Root Mean Squared Error
 *   - n_obs: BIGINT - Number of observations
 *   - n_features: BIGINT - Number of features
 *   - alpha: DOUBLE - L1 weight parameter used
 *   - lambda: DOUBLE - Regularization strength used
 *   - n_nonzero: BIGINT - Number of non-zero coefficients (for sparsity)
 *
 * Examples:
 *   -- Balanced ElasticNet (alpha=0.5)
 *   SELECT * FROM anofox_elastic_net_fit(
 *       (SELECT x1, x2, x3, y FROM my_data),
 *       ['x1', 'x2', 'x3'],
 *       'y',
 *       0.5,
 *       0.1
 *   );
 *
 *   -- Closer to Lasso (alpha=0.9)
 *   SELECT * FROM anofox_elastic_net_fit(
 *       (SELECT x1, x2, x3, y FROM my_data),
 *       ['x1', 'x2', 'x3'],
 *       'y',
 *       alpha := 0.9,
 *       lambda := 0.05,
 *       add_intercept := true
 *   );
 *
 *   -- Closer to Ridge (alpha=0.1)
 *   SELECT * FROM anofox_elastic_net_fit(
 *       (SELECT x1, x2, x3, y FROM my_data),
 *       ['x1', 'x2', 'x3'],
 *       'y',
 *       alpha := 0.1,
 *       lambda := 0.5
 *   );
 */
class ElasticNetFitFunction {
public:
	/**
	 * @brief Register the anofox_elastic_net_fit table function
	 *
	 * @param loader Extension loader context
	 */
	static void Register(ExtensionLoader &loader);

private:
	// Bind phase - collect data and validate inputs
	static unique_ptr<FunctionData> ElasticNetFitBind(ClientContext &context, TableFunctionBindInput &input);

	// Execute phase - stream results back
	static void ElasticNetFitExecute(ClientContext &context, TableFunctionInput &data, DataChunk &output);
};

} // namespace anofox_statistics
} // namespace duckdb
