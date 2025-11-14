#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Ridge Fit-Predict Aggregate Function
 *
 * Ridge regression (L2 regularization) fit-predict function that:
 * 1. Trains Ridge model on rows where y IS NOT NULL
 * 2. Predicts for ALL rows (returns yhat column)
 * 3. Supports PARTITION BY (one model per partition)
 * 4. Supports rolling windows with ORDER BY
 * 5. Returns prediction intervals
 *
 * Signature:
 *   anofox_statistics_fit_predict_ridge(
 *       y DOUBLE,
 *       x DOUBLE[],          -- Use COLUMNS([x1, x2, x3])
 *       options MAP          -- Must include 'lambda' for regularization
 *   ) OVER (PARTITION BY ... ORDER BY ...)
 *   RETURNS STRUCT(yhat DOUBLE, yhat_lower DOUBLE, yhat_upper DOUBLE, std_error DOUBLE)
 *
 * Examples:
 *   -- Ridge regression with lambda=1.0
 *   SELECT
 *       *,
 *       (pred).yhat
 *   FROM (
 *       SELECT
 *           *,
 *           anofox_statistics_fit_predict_ridge(
 *               y,
 *               COLUMNS([x1, x2, x3]),
 *               MAP{'lambda': 1.0, 'intercept': true}
 *           ) OVER () as pred
 *       FROM my_table
 *   );
 */
class RidgeFitPredictFunction {
public:
    static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
