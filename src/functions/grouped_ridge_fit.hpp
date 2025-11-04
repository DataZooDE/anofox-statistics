#pragma once

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include <vector>
#include <string>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Grouped Ridge regression function
 *
 * Performs Ridge regression (L2-regularized linear regression) independently for each group/partition.
 * Ridge regression addresses multicollinearity issues by adding L2 penalty: min ||y - Xβ||² + λ||β||²
 *
 * Signature:
 *   SELECT * FROM anofox_grouped_ridge_fit(
 *       data := <table>,
 *       group_cols := [<col_name>, ...],     -- GROUP BY columns
 *       x_cols := [<col_name>, ...],         -- Feature columns
 *       y_col := <col_name>,                 -- Target column
 *       lambda := <DOUBLE>,                  -- Regularization strength (λ ≥ 0)
 *       add_intercept := <BOOLEAN>           -- Default true
 *   )
 *
 * Returns for each group:
 *   - group_id: BIGINT - Unique group identifier
 *   - group_key: VARCHAR[] - Values of grouping columns
 *   - coefficients: DOUBLE[] - Ridge coefficient estimates
 *   - intercept: DOUBLE - Ridge intercept estimate
 *   - r_squared: DOUBLE - R² (coefficient of determination)
 *   - adj_r_squared: DOUBLE - Adjusted R²
 *   - mse: DOUBLE - Mean squared error
 *   - rmse: DOUBLE - Root mean squared error
 *   - lambda: DOUBLE - Regularization parameter used
 *   - n_obs: BIGINT - Observations in group
 *   - n_features: BIGINT - Number of features
 *
 * Parameters:
 *   - lambda: Regularization strength
 *     * λ = 0.0: Standard OLS (no regularization)
 *     * λ = 0.01: Weak regularization
 *     * λ = 1.0: Moderate regularization
 *     * λ = 10.0: Strong regularization
 *
 * Examples:
 *   -- Ridge regression for each customer with λ=1.0
 *   SELECT group_id, group_key, coefficients, r_squared
 *   FROM anofox_grouped_ridge_fit(
 *       (SELECT customer_id, x1, x2, y FROM sales_data),
 *       ['customer_id'],
 *       ['x1', 'x2'],
 *       'y',
 *       lambda := 1.0
 *   );
 *
 *   -- Ridge with different lambdas for each region
 *   SELECT group_key[1] as region, lambda, mse
 *   FROM anofox_grouped_ridge_fit(
 *       (SELECT region, feature1, feature2, target FROM metrics),
 *       ['region'],
 *       ['feature1', 'feature2'],
 *       'target',
 *       lambda := 0.5
 *   );
 */
class GroupedRidgeFitFunction {
public:
	/**
	 * @brief Register the anofox_grouped_ridge_fit table function
	 *
	 * @param loader Extension loader context
	 */
	static void Register(ExtensionLoader &loader);

private:
	// Bind phase - collect data and validate inputs
	static unique_ptr<FunctionData> GroupedRidgeFitBind(ClientContext &context, TableFunctionBindInput &input);

	// Execute phase - stream results back
	static void GroupedRidgeFitExecute(ClientContext &context, TableFunctionInput &data, DataChunk &output);
};

} // namespace anofox_statistics
} // namespace duckdb
