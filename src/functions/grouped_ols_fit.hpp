#pragma once

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include <vector>
#include <string>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Grouped OLS (Ordinary Least Squares) regression function
 *
 * Performs OLS regression independently for each group/partition in the data.
 * Useful for analyzing relationships within distinct subsets of data (e.g., per customer, per region, per time period).
 *
 * The grouped OLS function:
 * - Partitions data by one or more grouping columns
 * - Fits a separate OLS model for each partition
 * - Returns regression parameters and metrics for each group
 *
 * Signature:
 *   SELECT * FROM anofox_grouped_ols_fit(
 *       data := <table>,
 *       group_cols := [<col_name>, ...],     -- GROUP BY columns
 *       x_cols := [<col_name>, ...],         -- Feature columns
 *       y_col := <col_name>,                 -- Target column
 *       add_intercept := <BOOLEAN>           -- Default true
 *   )
 *
 * Returns for each group:
 *   - group_id: BIGINT - Unique group identifier
 *   - group_key: VARCHAR[] - Values of grouping columns
 *   - coefficients: DOUBLE[] - OLS coefficient estimates
 *   - intercept: DOUBLE - OLS intercept estimate
 *   - r_squared: DOUBLE - R² (coefficient of determination)
 *   - adj_r_squared: DOUBLE - Adjusted R²
 *   - mse: DOUBLE - Mean squared error
 *   - rmse: DOUBLE - Root mean squared error
 *   - n_obs: BIGINT - Observations in group
 *   - n_features: BIGINT - Number of features
 *
 * Examples:
 *   -- Fit OLS for each customer
 *   SELECT group_id, group_key, coefficients, r_squared
 *   FROM anofox_grouped_ols_fit(
 *       (SELECT customer_id, x1, x2, y FROM sales_data),
 *       ['customer_id'],
 *       ['x1', 'x2'],
 *       'y'
 *   );
 *
 *   -- Fit OLS for each region and product
 *   SELECT group_key[1] as region, group_key[2] as product,
 *          coefficients, mse
 *   FROM anofox_grouped_ols_fit(
 *       (SELECT region, product, feature1, feature2, target FROM metrics),
 *       ['region', 'product'],
 *       ['feature1', 'feature2'],
 *       'target'
 *   );
 */
class GroupedOlsFitFunction {
public:
	/**
	 * @brief Register the anofox_grouped_ols_fit table function
	 *
	 * @param loader Extension loader context
	 */
	static void Register(ExtensionLoader &loader);

private:
	// Bind phase - collect data and validate inputs
	static unique_ptr<FunctionData> GroupedOlsFitBind(ClientContext &context, TableFunctionBindInput &input);

	// Execute phase - stream results back
	static void GroupedOlsFitExecute(ClientContext &context, TableFunctionInput &data, DataChunk &output);
};

} // namespace anofox_statistics
} // namespace duckdb
