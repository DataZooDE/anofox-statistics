#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Elastic Net Prediction Intervals
 *
 * Fits Elastic Net regression on training data and returns predictions with
 * confidence or prediction intervals for new observations.
 *
 * Note: Combines L1 and L2 regularization for sparse solutions.
 * Usage:
 *   SELECT * FROM anofox_statistics_elastic_net_predict_interval(
 *       y_train := [1.0, 2.0, 3.0]::DOUBLE[],
 *       x_train := [[1.0], [2.0], [3.0]]::DOUBLE[][],
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
class ElasticNetPredictIntervalFunction {
public:
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
