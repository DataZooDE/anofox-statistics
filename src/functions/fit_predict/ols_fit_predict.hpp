#pragma once

#include "duckdb.hpp"
#include "fit_predict_base.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief OLS Fit-Predict Aggregate Function
 *
 * Unified fit-predict function that:
 * 1. Trains OLS model on rows where y IS NOT NULL
 * 2. Predicts for ALL rows (returns yhat column)
 * 3. Supports PARTITION BY (one model per partition)
 * 4. Supports rolling windows with ORDER BY
 * 5. Returns prediction intervals
 *
 * Signature:
 *   anofox_statistics_fit_predict_ols(
 *       y DOUBLE,
 *       x DOUBLE[],          -- Use COLUMNS([x1, x2, x3])
 *       options MAP
 *   ) OVER (PARTITION BY ... ORDER BY ...)
 *   RETURNS STRUCT(yhat DOUBLE, yhat_lower DOUBLE, yhat_upper DOUBLE, std_error DOUBLE)
 *
 * Examples:
 *   -- Fit on training data, predict for all rows
 *   SELECT
 *       *,
 *       (pred).yhat,
 *       (pred).yhat_lower,
 *       (pred).yhat_upper
 *   FROM (
 *       SELECT
 *           *,
 *           anofox_statistics_fit_predict_ols(
 *               y,
 *               COLUMNS([x1, x2, x3]),
 *               MAP{'confidence_level': 0.95}
 *           ) OVER () as pred
 *       FROM my_table
 *   );
 *
 *   -- With partitioning (one model per group)
 *   SELECT
 *       *,
 *       (pred).yhat
 *   FROM (
 *       SELECT
 *           *,
 *           anofox_statistics_fit_predict_ols(
 *               y,
 *               COLUMNS([x1, x2]),
 *               MAP{'intercept': true}
 *           ) OVER (PARTITION BY category) as pred
 *       FROM sales_data
 *   );
 *
 *   -- Rolling window prediction
 *   SELECT
 *       *,
 *       (pred).yhat
 *   FROM (
 *       SELECT
 *           *,
 *           anofox_statistics_fit_predict_ols(
 *               y,
 *               [x],
 *               MAP{}
 *           ) OVER (ORDER BY date ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) as pred
 *       FROM time_series
 *   );
 */
class OlsFitPredictFunction {
public:
    /**
     * @brief Register OLS fit-predict aggregate function
     *
     * @param loader Extension loader context
     */
    static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
