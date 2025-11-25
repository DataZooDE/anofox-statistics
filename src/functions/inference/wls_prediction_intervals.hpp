#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief WLS Prediction Intervals
 *
 * Fits WLS regression on training data and returns predictions with
 * confidence or prediction intervals for new observations.
 *
 * Note: Accounts for heteroscedasticity via observation weights.
 * Usage:
 *   SELECT * FROM anofox_statistics_wls_predict_interval(
 *       y_train := [1.0, 2.0, 3.0]::DOUBLE[],
 *       x_train := [[1.0], [2.0], [3.0]]::DOUBLE[][],
 *       weights := [1.0, 2.0, 1.5]::DOUBLE[],
 *       x_new := [[4.0], [5.0]]::DOUBLE[][],
 *       options := MAP{'confidence_level': 0.95, 'interval_type': 'prediction'}
 *   );
 *
 * Returns one row per new observation with:
 * - observation_id: Row number (1-indexed)
 * - predicted: Point prediction
 * - ci_lower: Lower interval bound
 * - ci_upper: Upper interval bound
 * - se: Standard error of prediction
 */
class WLSPredictIntervalFunction {
public:
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
