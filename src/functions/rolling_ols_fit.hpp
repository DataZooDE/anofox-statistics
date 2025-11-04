#pragma once

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include <vector>
#include <string>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Rolling window OLS regression function
 *
 * Computes OLS regression over a fixed-size sliding window of sorted data.
 * Useful for time-series analysis where you want to see how model parameters
 * change as new observations enter and old ones leave the window.
 *
 * Signature:
 *   SELECT * FROM anofox_rolling_ols_fit(
 *       data := <table>,
 *       x_cols := [<col_name>, ...],
 *       y_col := <col_name>,
 *       window_size := <INTEGER>,
 *       partition_col := <col_name>,      -- Optional: partition by column
 *       order_col := <col_name>,          -- Optional: sort order
 *       add_intercept := <BOOLEAN>        -- Default true
 *   )
 *
 * Returns for each point in the window:
 *   - window_id: BIGINT - Window number (starts at 0)
 *   - coefficients: DOUBLE[] - OLS coefficients for that window
 *   - intercept: DOUBLE - Window intercept
 *   - r_squared: DOUBLE - Window R²
 *   - adj_r_squared: DOUBLE - Adjusted R²
 *   - mse: DOUBLE - Window MSE
 *   - rmse: DOUBLE - Window RMSE
 *   - n_obs: BIGINT - Observations in window
 *   - n_features: BIGINT - Number of features
 *   - window_start_idx: BIGINT - Start index in original data
 *   - window_end_idx: BIGINT - End index in original data
 *
 * Examples:
 *   -- Simple rolling window (50 observations)
 *   SELECT window_id, coefficients, r_squared
 *   FROM anofox_rolling_ols_fit(
 *       (SELECT x1, x2, y FROM timeseries_data ORDER BY date),
 *       ['x1', 'x2'],
 *       'y',
 *       window_size := 50
 *   );
 *
 *   -- Rolling window with overlapping output
 *   SELECT window_id, coefficients[1] as coeff_x1, r_squared
 *   FROM anofox_rolling_ols_fit(
 *       (SELECT x1, x2, y FROM stocks ORDER BY stock_id, date),
 *       ['x1', 'x2'],
 *       'y',
 *       window_size := 30,
 *       partition_col := 'stock_id',
 *       order_col := 'date'
 *   );
 */
class RollingOlsFitFunction {
public:
	/**
	 * @brief Register the anofox_rolling_ols_fit table function
	 *
	 * @param loader Extension loader context
	 */
	static void Register(ExtensionLoader &loader);

private:
	// Bind phase - collect data and validate inputs
	static unique_ptr<FunctionData> RollingOlsFitBind(ClientContext &context, TableFunctionBindInput &input);

	// Execute phase - stream results back
	static void RollingOlsFitExecute(ClientContext &context, TableFunctionInput &data, DataChunk &output);
};

} // namespace anofox_statistics
} // namespace duckdb
