#pragma once

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include <vector>
#include <string>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Expanding window OLS regression function
 *
 * Computes OLS regression over an expanding window, starting from an initial
 * window and including all subsequent observations. Useful for detecting how
 * model stability and fit change as more historical data becomes available.
 *
 * Signature:
 *   SELECT * FROM anofox_expanding_ols_fit(
 *       data := <table>,
 *       x_cols := [<col_name>, ...],
 *       y_col := <col_name>,
 *       min_obs := <INTEGER>,             -- Minimum observations to start
 *       partition_col := <col_name>,      -- Optional: partition by column
 *       order_col := <col_name>,          -- Optional: sort order
 *       add_intercept := <BOOLEAN>        -- Default true
 *   )
 *
 * Returns for each point in the expansion:
 *   - window_id: BIGINT - Window number (starts at 0)
 *   - coefficients: DOUBLE[] - OLS coefficients for that window
 *   - intercept: DOUBLE - Window intercept
 *   - r_squared: DOUBLE - Window R²
 *   - adj_r_squared: DOUBLE - Adjusted R²
 *   - mse: DOUBLE - Window MSE
 *   - rmse: DOUBLE - Window RMSE
 *   - n_obs: BIGINT - Observations in window (increasing)
 *   - n_features: BIGINT - Number of features
 *   - window_start_idx: BIGINT - Always 0 (starts from beginning)
 *   - window_end_idx: BIGINT - Current observation index
 *
 * Examples:
 *   -- Expanding window from 20 observations
 *   SELECT window_id, n_obs, r_squared, coefficients
 *   FROM anofox_expanding_ols_fit(
 *       (SELECT x1, x2, y FROM timeseries_data ORDER BY date),
 *       ['x1', 'x2'],
 *       'y',
 *       min_obs := 20
 *   );
 *
 *   -- Expanding window with partition (multiple time series)
 *   SELECT stock_id, window_id, n_obs, r_squared
 *   FROM anofox_expanding_ols_fit(
 *       (SELECT stock_id, x1, x2, y FROM stocks ORDER BY stock_id, date),
 *       ['x1', 'x2'],
 *       'y',
 *       min_obs := 30,
 *       partition_col := 'stock_id',
 *       order_col := 'date'
 *   );
 */
class ExpandingOlsFitFunction {
public:
	/**
	 * @brief Register the anofox_expanding_ols_fit table function
	 *
	 * @param loader Extension loader context
	 */
	static void Register(ExtensionLoader &loader);

private:
	// Bind phase - collect data and validate inputs
	static unique_ptr<FunctionData> ExpandingOlsFitBind(ClientContext &context, TableFunctionBindInput &input);

	// Execute phase - stream results back
	static void ExpandingOlsFitExecute(ClientContext &context, TableFunctionInput &data, DataChunk &output);
};

} // namespace anofox_statistics
} // namespace duckdb
