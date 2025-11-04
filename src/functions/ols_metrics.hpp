#pragma once

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include <vector>
#include <string>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief OLS regression metrics functions
 *
 * Compute various performance metrics for regression models.
 *
 * Functions:
 * - anofox_ols_r_squared(actual, predicted) -> DOUBLE
 *   Coefficient of determination (R²)
 *
 * - anofox_ols_mse(actual, predicted) -> DOUBLE
 *   Mean Squared Error
 *
 * - anofox_ols_rmse(actual, predicted) -> DOUBLE
 *   Root Mean Squared Error
 *
 * - anofox_ols_mae(actual, predicted) -> DOUBLE
 *   Mean Absolute Error
 *
 * Examples:
 *   -- Compute metrics for predictions
 *   SELECT
 *       anofox_ols_r_squared(actual_y, predicted_y) as r_squared,
 *       anofox_ols_mse(actual_y, predicted_y) as mse,
 *       anofox_ols_rmse(actual_y, predicted_y) as rmse,
 *       anofox_ols_mae(actual_y, predicted_y) as mae
 *   FROM test_data;
 *
 *   -- Aggregate metrics across multiple folds
 *   SELECT
 *       anofox_ols_mse(actual_y, predicted_y) as fold_mse
 *   FROM cv_results
 *   GROUP BY fold_id;
 */
class OlsMetricsFunction {
public:
	/**
	 * @brief Register all metric functions
	 *
	 * @param loader Extension loader context
	 */
	static void Register(ExtensionLoader &loader);

private:
	/**
	 * @brief Compute R² (coefficient of determination)
	 * R² = 1 - (SS_res / SS_tot)
	 */
	static void ComputeRSquared(DataChunk &args, ExpressionState &state, Vector &result);

	/**
	 * @brief Compute Mean Squared Error
	 * MSE = mean((y_actual - y_predicted)²)
	 */
	static void ComputeMSE(DataChunk &args, ExpressionState &state, Vector &result);

	/**
	 * @brief Compute Root Mean Squared Error
	 * RMSE = sqrt(MSE)
	 */
	static void ComputeRMSE(DataChunk &args, ExpressionState &state, Vector &result);

	/**
	 * @brief Compute Mean Absolute Error
	 * MAE = mean(|y_actual - y_predicted|)
	 */
	static void ComputeMAE(DataChunk &args, ExpressionState &state, Vector &result);
};

} // namespace anofox_statistics
} // namespace duckdb
