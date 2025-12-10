#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief OLS Prediction with Intervals
 *
 * Predicts values for new observations with confidence or prediction intervals.
 *
 * Two types of intervals:
 * - Confidence Interval: Uncertainty in mean prediction (narrower)
 * - Prediction Interval: Uncertainty for individual prediction (wider, includes residual variance)
 *
 * Usage:
 *   SELECT * FROM ols_predict_interval(
 *       y_train := [1.0, 2.0, 3.0]::DOUBLE[],
 *       x_train := [[1.0], [2.0], [3.0]]::DOUBLE[][],
 *       x_new := [[4.0], [5.0]]::DOUBLE[][],
 *       confidence_level := 0.95,
 *       interval_type := 'prediction',  -- or 'confidence'
 *       add_intercept := true
 *   );
 *
 * Returns:
 * - observation_id: Row number (1-indexed)
 * - predicted: Point prediction
 * - ci_lower: Lower bound of interval
 * - ci_upper: Upper bound of interval
 * - se: Standard error of prediction
 */
class OlsPredictIntervalFunction {
public:
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
