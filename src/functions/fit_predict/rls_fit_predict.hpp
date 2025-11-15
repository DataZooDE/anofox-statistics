#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief RLS (Recursive Least Squares) Fit-Predict Aggregate Function
 *
 * Recursive Least Squares fit-predict function that:
 * 1. Trains RLS model on rows where y IS NOT NULL
 * 2. Predicts for ALL rows (returns yhat column)
 * 3. Supports PARTITION BY (one model per partition)
 * 4. Supports rolling windows with ORDER BY
 * 5. Returns prediction intervals
 * 6. Uses sequential Kalman updates for online learning
 * 7. Supports forgetting factor for adaptive tracking
 *
 * Signature:
 *   anofox_statistics_fit_predict_rls(
 *       y DOUBLE,
 *       x DOUBLE[],          -- Use COLUMNS([x1, x2, x3])
 *       options MAP          -- Options like 'intercept' and 'forgetting_factor'
 *   ) OVER (PARTITION BY ... ORDER BY ...)
 *   RETURNS STRUCT(yhat DOUBLE, yhat_lower DOUBLE, yhat_upper DOUBLE, std_error DOUBLE)
 *
 * Algorithm:
 *   Sequential Kalman update for each observation:
 *   β_t = β_{t-1} + K_t * (y_t - x_t'β_{t-1})
 *   K_t = P_{t-1}x_t / (λ + x_t'P_{t-1}x_t)
 *   P_t = (1/λ) * (P_{t-1} - K_t x_t' P_{t-1})
 *
 *   Where:
 *   - λ is the forgetting factor (0 < λ ≤ 1, default 1.0)
 *   - P_t is the inverse covariance matrix
 *   - K_t is the Kalman gain vector
 *
 * Examples:
 *   -- RLS regression with forgetting factor
 *   SELECT
 *       *,
 *       (pred).yhat
 *   FROM (
 *       SELECT
 *           *,
 *           anofox_statistics_fit_predict_rls(
 *               y,
 *               COLUMNS([x1, x2, x3]),
 *               MAP{'intercept': true, 'forgetting_factor': 0.99}
 *           ) OVER () as pred
 *       FROM my_table
 *   );
 */
class RlsFitPredictFunction {
public:
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
