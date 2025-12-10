#pragma once

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include <vector>
#include <string>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Grouped aggregate metrics function
 *
 * Computes regression metrics (R², MSE, RMSE, MAE) independently for each group/partition.
 * Useful for evaluating model performance within different segments of data.
 *
 * Signature:
 *   SELECT * FROM anofox_grouped_metrics(
 *       data := <table>,
 *       group_cols := [<col_name>, ...],     -- GROUP BY columns
 *       y_actual_col := <col_name>,          -- Actual target values
 *       y_pred_col := <col_name>             -- Predicted values
 *   )
 *
 * Computed Metrics:
 *   - r_squared: R² (coefficient of determination)
 *     * 1.0 = perfect fit
 *     * 0.5 = explains 50% of variance
 *     * 0.0 = explains none of variance
 *     * <0 = worse than horizontal line
 *
 *   - mse: Mean Squared Error = Σ(y_actual - y_pred)² / n
 *
 *   - rmse: Root Mean Squared Error = sqrt(MSE)
 *
 *   - mae: Mean Absolute Error = Σ|y_actual - y_pred| / n
 *
 * Returns for each group:
 *   - group_id: BIGINT - Unique group identifier
 *   - group_key: VARCHAR[] - Values of grouping columns
 *   - r_squared: DOUBLE - R² metric
 *   - mse: DOUBLE - Mean squared error
 *   - rmse: DOUBLE - Root mean squared error
 *   - mae: DOUBLE - Mean absolute error
 *   - n_obs: BIGINT - Observations in group
 *
 * Examples:
 *   -- Evaluate model performance per customer
 *   SELECT group_id, group_key, r_squared, rmse
 *   FROM anofox_grouped_metrics(
 *       (SELECT customer_id, actual_sales, predicted_sales FROM predictions),
 *       ['customer_id'],
 *       'actual_sales',
 *       'predicted_sales'
 *   )
 *   ORDER BY r_squared DESC;
 *
 *   -- Compare metrics across regions and products
 *   SELECT group_key[1] as region, group_key[2] as product,
 *          r_squared, mae
 *   FROM anofox_grouped_metrics(
 *       (SELECT region, product, target, pred FROM forecast_eval),
 *       ['region', 'product'],
 *       'target',
 *       'pred'
 *   );
 */
class GroupedMetricsFunction {
public:
	/**
	 * @brief Register the anofox_grouped_metrics table function
	 *
	 * @param loader Extension loader context
	 */
	static void Register(ExtensionLoader &loader);

private:
	// Bind phase - collect data and validate inputs
	static unique_ptr<FunctionData> GroupedMetricsBind(ClientContext &context, TableFunctionBindInput &input);

	// Execute phase - stream results back
	static void GroupedMetricsExecute(ClientContext &context, TableFunctionInput &data, DataChunk &output);
};

} // namespace anofox_statistics
} // namespace duckdb
