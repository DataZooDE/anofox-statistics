#pragma once

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include <vector>
#include <string>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Rolling Window OLS regression - Time-series regression on moving windows
 *
 * Computes OLS regression on sliding windows of data. Useful for detecting
 * changing trends and relationships in time-series data.
 *
 * Signature:
 *   SELECT * FROM anofox_statistics_rolling_ols(
 *       y := <DOUBLE[]>,
 *       x1 := <DOUBLE[]>,
 *       [x2 := <DOUBLE[]>, ...],
 *       window_size := <BIGINT>,        -- Window size (must be > n_features + 1)
 *       [add_intercept := <BOOLEAN>]    -- Default: true
 *   )
 *
 * Algorithm:
 *   For each position i from 0 to (n - window_size):
 *     - Extract window: observations [i, i+window_size)
 *     - Compute OLS on window data
 *     - Return coefficients and statistics for this window
 *
 * Returns (one row per window):
 *   - window_start: BIGINT - Starting index of window
 *   - window_end: BIGINT - Ending index of window (exclusive)
 *   - coefficients: DOUBLE[] - OLS coefficients for this window
 *   - intercept: DOUBLE - Model intercept for this window
 *   - r_squared: DOUBLE - R² for this window
 *   - mse: DOUBLE - Mean Squared Error for this window
 *   - n_obs: BIGINT - Number of observations in window
 *   - n_features: BIGINT - Number of features
 *
 * Examples:
 *   -- 3-observation rolling window
 *   SELECT * FROM anofox_statistics_rolling_ols(
 *       [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
 *       [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
 *       3  -- Window size
 *   );
 *   -- Returns 3 rows: windows [0,3), [1,4), [2,5)
 *
 *   -- Multi-variable rolling OLS with 4-observation window
 *   SELECT * FROM anofox_statistics_rolling_ols(
 *       [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]::DOUBLE[],
 *       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]::DOUBLE[],
 *       [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]::DOUBLE[],
 *       4  -- Window size
 *   );
 *   -- Returns 3 rows: windows [0,4), [1,5), [2,6)
 */
class RollingOlsFunction {
public:
	/**
	 * @brief Register the anofox_statistics_rolling_ols table function
	 *
	 * @param loader Extension loader context
	 */
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
