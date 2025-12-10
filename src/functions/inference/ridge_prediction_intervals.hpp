#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Ridge Prediction Intervals - Predictions with intervals for Ridge regression
 *
 * Fits Ridge regression on training data and returns predictions with
 * confidence or prediction intervals for new observations.
 *
 * Usage:
 *   SELECT * FROM anofox_statistics_ridge_predict_interval(
 *       y_train := [1.0, 2.0, 3.0]::DOUBLE[],
 *       x_train := [[1.0], [2.0], [3.0]]::DOUBLE[][],
 *       x_new := [[4.0], [5.0]]::DOUBLE[][],
 *       options := MAP{'lambda': 1.0, 'confidence_level': 0.95, 'interval_type': 'prediction'}
 *   );
 */
class RidgePredictIntervalFunction {
public:
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
