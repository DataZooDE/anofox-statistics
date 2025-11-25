#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <limits>

namespace libanostat {
namespace core {

/**
 * Statistical inference results for regression coefficients
 *
 * This structure contains all statistical inference outputs including
 * t-statistics, p-values, confidence intervals, and prediction intervals.
 *
 * Design notes:
 * - All vectors use Eigen types for consistency
 * - NaN values indicate unavailable/invalid results (e.g., for aliased coefficients)
 * - Separate structures for coefficient-level and prediction-level inference
 */
struct CoefficientInference {
	// ========================================================================
	// Coefficient-level inference
	// ========================================================================

	/// Standard errors of coefficients (length = n_params)
	/// Aliased coefficients have NaN std errors
	Eigen::VectorXd std_errors;

	/// t-statistics: coef / std_error (length = n_params)
	/// Aliased coefficients have NaN t-statistics
	Eigen::VectorXd t_statistics;

	/// Two-tailed p-values for H0: coef = 0 (length = n_params)
	/// Aliased coefficients have NaN p-values
	Eigen::VectorXd p_values;

	/// Lower bounds of confidence intervals (length = n_params)
	/// Computed as: coef - t_critical * std_error
	/// Aliased coefficients have NaN bounds
	Eigen::VectorXd ci_lower;

	/// Upper bounds of confidence intervals (length = n_params)
	/// Computed as: coef + t_critical * std_error
	/// Aliased coefficients have NaN bounds
	Eigen::VectorXd ci_upper;

	/// Confidence level used (e.g., 0.95 for 95% CI)
	double confidence_level = 0.95;

	/// Degrees of freedom used for t-distribution
	size_t degrees_of_freedom = 0;

	// ========================================================================
	// Constructors
	// ========================================================================

	/// Default constructor
	CoefficientInference() = default;

	/// Convenience constructor with dimensions
	CoefficientInference(size_t n_params, double conf_level = 0.95)
		: confidence_level(conf_level) {
		std_errors = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(n_params),
		                                       std::numeric_limits<double>::quiet_NaN());
		t_statistics = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(n_params),
		                                         std::numeric_limits<double>::quiet_NaN());
		p_values = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(n_params),
		                                     std::numeric_limits<double>::quiet_NaN());
		ci_lower = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(n_params),
		                                     std::numeric_limits<double>::quiet_NaN());
		ci_upper = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(n_params),
		                                     std::numeric_limits<double>::quiet_NaN());
	}
};

/**
 * Prediction intervals for new observations
 *
 * Used when making predictions with uncertainty quantification.
 * Prediction intervals are wider than confidence intervals because they
 * account for both estimation uncertainty AND residual variance.
 */
struct PredictionIntervals {
	// ========================================================================
	// Prediction interval outputs
	// ========================================================================

	/// Predicted values (point estimates)
	Eigen::VectorXd predictions;

	/// Lower bounds of prediction intervals
	/// Computed as: yhat - t_critical * sqrt(mse * (1 + leverage))
	Eigen::VectorXd lower_bounds;

	/// Upper bounds of prediction intervals
	/// Computed as: yhat + t_critical * sqrt(mse * (1 + leverage))
	Eigen::VectorXd upper_bounds;

	/// Standard errors of predictions (optional)
	/// sqrt(mse * (1 + leverage)) for each observation
	Eigen::VectorXd std_errors;

	/// Confidence level used (e.g., 0.95 for 95% PI)
	double confidence_level = 0.95;

	/// Degrees of freedom used for t-distribution
	size_t degrees_of_freedom = 0;

	// ========================================================================
	// Constructors
	// ========================================================================

	/// Default constructor
	PredictionIntervals() = default;

	/// Convenience constructor with dimensions
	PredictionIntervals(size_t n_predictions, double conf_level = 0.95)
		: confidence_level(conf_level) {
		predictions = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_predictions));
		lower_bounds = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(n_predictions),
		                                         std::numeric_limits<double>::quiet_NaN());
		upper_bounds = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(n_predictions),
		                                         std::numeric_limits<double>::quiet_NaN());
		std_errors = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(n_predictions),
		                                       std::numeric_limits<double>::quiet_NaN());
	}
};

/**
 * Complete inference result combining coefficient and prediction inference
 *
 * This is the full output from statistical inference functions.
 */
struct InferenceResult {
	/// Coefficient-level inference (t-stats, p-values, CIs)
	CoefficientInference coefficient_inference;

	/// Prediction-level inference (prediction intervals)
	/// Only populated when making predictions
	PredictionIntervals prediction_intervals;

	/// Flag indicating if coefficient inference was computed
	bool has_coefficient_inference = false;

	/// Flag indicating if prediction intervals were computed
	bool has_prediction_intervals = false;

	// ========================================================================
	// Constructors
	// ========================================================================

	/// Default constructor
	InferenceResult() = default;

	/// Constructor for coefficient inference only
	static InferenceResult WithCoefficientInference(size_t n_params, double conf_level = 0.95) {
		InferenceResult result;
		result.coefficient_inference = CoefficientInference(n_params, conf_level);
		result.has_coefficient_inference = true;
		return result;
	}

	/// Constructor for prediction intervals only
	static InferenceResult WithPredictionIntervals(size_t n_predictions, double conf_level = 0.95) {
		InferenceResult result;
		result.prediction_intervals = PredictionIntervals(n_predictions, conf_level);
		result.has_prediction_intervals = true;
		return result;
	}

	/// Constructor for both coefficient and prediction inference
	static InferenceResult WithBoth(size_t n_params, size_t n_predictions, double conf_level = 0.95) {
		InferenceResult result;
		result.coefficient_inference = CoefficientInference(n_params, conf_level);
		result.prediction_intervals = PredictionIntervals(n_predictions, conf_level);
		result.has_coefficient_inference = true;
		result.has_prediction_intervals = true;
		return result;
	}
};

} // namespace core
} // namespace libanostat
