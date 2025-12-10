#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief WLS (Weighted Least Squares) Fit-Predict Aggregate Function
 *
 * Weighted Least Squares fit-predict function that:
 * 1. Trains WLS model on rows where y IS NOT NULL
 * 2. Predicts for ALL rows (returns yhat column)
 * 3. Supports PARTITION BY (one model per partition)
 * 4. Supports rolling windows with ORDER BY
 * 5. Returns prediction intervals
 * 6. Uses weights to adjust influence of each observation
 *
 * Signature:
 *   anofox_statistics_fit_predict_wls(
 *       y DOUBLE,
 *       x DOUBLE[],          -- Use COLUMNS([x1, x2, x3])
 *       weight DOUBLE,       -- Weight for this observation (must be positive)
 *       options MAP          -- Options like 'intercept'
 *   ) OVER (PARTITION BY ... ORDER BY ...)
 *   RETURNS STRUCT(yhat DOUBLE, yhat_lower DOUBLE, yhat_upper DOUBLE, std_error DOUBLE)
 *
 * Examples:
 *   -- WLS regression with custom weights
 *   SELECT
 *       *,
 *       (pred).yhat
 *   FROM (
 *       SELECT
 *           *,
 *           anofox_statistics_fit_predict_wls(
 *               y,
 *               COLUMNS([x1, x2, x3]),
 *               weight,
 *               MAP{'intercept': true}
 *           ) OVER () as pred
 *       FROM my_table
 *   );
 */
class WlsFitPredictFunction {
public:
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
