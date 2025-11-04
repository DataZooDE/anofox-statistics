#pragma once

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include <vector>
#include <string>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Expanding Window OLS regression - Cumulative time-series regression
 *
 * Computes OLS regression on expanding windows of data. Each window starts
 * at a fixed position and grows to include more observations. Useful for
 * analyzing how model parameters evolve as more data is accumulated.
 *
 * Signature:
 *   SELECT * FROM anofox_statistics_expanding_ols(
 *       y := <DOUBLE[]>,
 *       x1 := <DOUBLE[]>,
 *       [x2 := <DOUBLE[]>, ...],
 *       min_periods := <BIGINT>,       -- Minimum observations (must be > n_features + 1)
 *       [add_intercept := <BOOLEAN>]   -- Default: true
 *   )
 *
 * Algorithm:
 *   For each position i from min_periods to n:
 *     - Extract window: observations [0, i)
 *     - Compute OLS on cumulative data
 *     - Return coefficients and statistics for this window
 *
 * Returns (one row per window):
 *   - window_start: BIGINT - Starting index of window (always 0)
 *   - window_end: BIGINT - Ending index of window (exclusive)
 *   - coefficients: DOUBLE[] - OLS coefficients for this window
 *   - intercept: DOUBLE - Model intercept for this window
 *   - r_squared: DOUBLE - R² for this window
 *   - mse: DOUBLE - Mean Squared Error for this window
 *   - n_obs: BIGINT - Number of observations in window
 *   - n_features: BIGINT - Number of features
 *
 * Examples:
 *   -- Expanding window starting at 3 observations
 *   SELECT * FROM anofox_statistics_expanding_ols(
 *       [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
 *       [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
 *       3  -- Minimum periods
 *   );
 *   -- Returns 3 rows: windows [0,3), [0,4), [0,5)
 *
 *   -- Multi-variable expanding OLS
 *   SELECT * FROM anofox_statistics_expanding_ols(
 *       [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]::DOUBLE[],
 *       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]::DOUBLE[],
 *       [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]::DOUBLE[],
 *       4  -- Minimum periods
 *   );
 *   -- Returns 3 rows: windows [0,4), [0,5), [0,6)
 */
class ExpandingOlsFunction {
public:
	/**
	 * @brief Register the anofox_statistics_expanding_ols table function
	 *
	 * @param loader Extension loader context
	 */
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
