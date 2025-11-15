#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Elastic Net Fit-Predict Aggregate Function
 *
 * Elastic Net (combined L1 + L2 regularization) fit-predict function that:
 * 1. Trains Elastic Net model on rows where y IS NOT NULL
 * 2. Predicts for ALL rows (returns yhat column)
 * 3. Supports PARTITION BY (one model per partition)
 * 4. Supports rolling windows with ORDER BY
 * 5. Returns prediction intervals (approximate due to regularization bias)
 * 6. Performs feature selection via L1 penalty
 *
 * Signature:
 *   anofox_statistics_fit_predict_elastic_net(
 *       y DOUBLE,
 *       x DOUBLE[],          -- Use COLUMNS([x1, x2, x3])
 *       options MAP          -- Must include 'alpha' (0-1) and 'lambda'
 *   ) OVER (PARTITION BY ... ORDER BY ...)
 *   RETURNS STRUCT(yhat DOUBLE, yhat_lower DOUBLE, yhat_upper DOUBLE, std_error DOUBLE)
 *
 * Examples:
 *   -- Elastic Net with alpha=0.5 (equal L1/L2 mix) and lambda=1.0
 *   SELECT
 *       *,
 *       (pred).yhat
 *   FROM (
 *       SELECT
 *           *,
 *           anofox_statistics_fit_predict_elastic_net(
 *               y,
 *               COLUMNS([x1, x2, x3]),
 *               MAP{'alpha': 0.5, 'lambda': 1.0, 'intercept': true}
 *           ) OVER () as pred
 *       FROM my_table
 *   );
 */
class ElasticNetFitPredictFunction {
public:
    static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
