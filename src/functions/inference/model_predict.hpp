#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief OLS Model Prediction with Intervals (using pre-fitted model)
 *
 * Makes predictions using a pre-fitted OLS model (from full_output=true).
 * Much more efficient than ols_predict_interval as it doesn't refit the model.
 *
 * Workflow:
 *   1. Fit model with full_output=true:
 *      CREATE TABLE model AS SELECT * FROM anofox_statistics_ols(y, x, true, {'full_output': true});
 *
 *   2. Make predictions:
 *      SELECT p.* FROM model m,
 *      LATERAL anofox_statistics_model_predict(
 *          m.intercept, m.coefficients, m.mse, m.x_train_means,
 *          m.coefficient_std_errors, m.intercept_std_error, m.df_residual,
 *          [6.0, 7.0]::DOUBLE[],  -- x_new (single feature)
 *          0.95,                   -- confidence_level
 *          'prediction'            -- interval_type: 'prediction', 'confidence', or 'none'
 *      ) p;
 *
 * For multiple features:
 *      SELECT p.* FROM model m,
 *      LATERAL anofox_statistics_model_predict(
 *          m.intercept, m.coefficients, m.mse, m.x_train_means,
 *          m.coefficient_std_errors, m.intercept_std_error, m.df_residual,
 *          [[6.0, 2.0], [7.0, 3.0]]::DOUBLE[][],  -- x_new (multiple features)
 *          0.95,
 *          'prediction'
 *      ) p;
 *
 * Returns for each new observation:
 * - observation_id: Row number (1-indexed)
 * - predicted: Point prediction (ŷ = intercept + β₁x₁ + β₂x₂ + ...)
 * - ci_lower: Lower bound of interval (NULL if interval_type='none')
 * - ci_upper: Upper bound of interval (NULL if interval_type='none')
 * - se: Standard error of prediction (NULL if interval_type='none')
 *
 * Interval Types:
 * - 'confidence': Confidence interval for mean prediction (narrower)
 *                 SE(ŷ) = √(MSE * h) where h = x'(X'X)⁻¹x
 * - 'prediction': Prediction interval for individual prediction (wider)
 *                 SE(pred) = √(MSE * (1 + h))
 * - 'none': Only return point predictions (faster)
 */
class AnofoxStatisticsModelPredictFunction {
public:
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
