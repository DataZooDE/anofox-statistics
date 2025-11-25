#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Elastic-Net Predict Aggregate - Apply pre-fitted model to new data
 *
 * Aggregate function that takes a pre-fitted Elastic-Net model and applies it
 * to new observations within GROUP BY or window contexts.
 *
 * Usage:
 *   SELECT category,
 *          anofox_statistics_predict_elastic_net_agg(
 *              model.coefficients,
 *              model.intercept,
 *              model.mse,
 *              model.x_train_means,
 *              model.coefficient_std_errors,
 *              model.intercept_std_error,
 *              model.df_residual,
 *              [x1, x2],
 *              MAP{'confidence_level': 0.95, 'interval_type': 'prediction'}
 *          ) as prediction
 *   FROM test_data
 *   JOIN models ON test_data.category = models.category
 *   GROUP BY category;
 *
 * Returns STRUCT(yhat, yhat_lower, yhat_upper, std_error)
 */
class ElasticNetPredictAggregateFunction {
public:
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
