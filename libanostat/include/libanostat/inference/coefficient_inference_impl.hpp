#pragma once

#include "coefficient_inference.hpp"
#include <stdexcept>

namespace libanostat {
namespace inference {

// Implementation of CoefficientInference methods

inline Eigen::MatrixXd CoefficientInference::ExtractValidColumns(
    const Eigen::MatrixXd &X,
    const std::vector<bool> &is_aliased) {

	size_t n = X.rows();
	size_t p = X.cols();

	// Count non-aliased features
	size_t n_valid = 0;
	for (size_t j = 0; j < p; j++) {
		if (!is_aliased[j]) {
			n_valid++;
		}
	}

	if (n_valid == 0) {
		return Eigen::MatrixXd(n, 0);
	}

	// Extract non-aliased columns
	Eigen::MatrixXd X_valid(n, n_valid);
	size_t valid_idx = 0;
	for (size_t j = 0; j < p; j++) {
		if (!is_aliased[j]) {
			X_valid.col(valid_idx) = X.col(j);
			valid_idx++;
		}
	}

	return X_valid;
}

inline Eigen::MatrixXd CoefficientInference::ComputeXtXInverse(
    const Eigen::MatrixXd &X,
    const std::vector<bool> &is_aliased) {

	// Extract non-aliased columns
	Eigen::MatrixXd X_valid = ExtractValidColumns(X, is_aliased);

	if (X_valid.cols() == 0) {
		return Eigen::MatrixXd(0, 0);
	}

	// Compute (X'X)^-1
	Eigen::MatrixXd XtX = X_valid.transpose() * X_valid;
	return XtX.inverse();
}

inline Eigen::VectorXd CoefficientInference::ComputeStdErrors(
    double mse,
    const Eigen::MatrixXd &X,
    const std::vector<bool> &is_aliased) {

	size_t p = X.cols();
	Eigen::VectorXd std_errors = Eigen::VectorXd::Constant(p, std::numeric_limits<double>::quiet_NaN());

	try {
		Eigen::MatrixXd XtX_inv = ComputeXtXInverse(X, is_aliased);

		if (XtX_inv.rows() == 0) {
			return std_errors;
		}

		// Compute standard errors for non-aliased coefficients
		size_t valid_idx = 0;
		for (size_t j = 0; j < p; j++) {
			if (!is_aliased[j]) {
				std_errors(j) = std::sqrt(mse * XtX_inv(valid_idx, valid_idx));
				valid_idx++;
			}
		}
	} catch (...) {
		// If computation fails, leave as NaN
	}

	return std_errors;
}

inline double CoefficientInference::ComputeInterceptStdError(
    double mse,
    size_t n_obs,
    const Eigen::MatrixXd &X_centered,
    const Eigen::VectorXd &x_means,
    const std::vector<bool> &is_aliased) {

	try {
		size_t p = X_centered.cols();

		// Extract non-aliased features
		Eigen::MatrixXd X_valid = ExtractValidColumns(X_centered, is_aliased);

		if (X_valid.cols() == 0) {
			// No valid features -> SE(intercept) = sqrt(MSE/n)
			return std::sqrt(mse / n_obs);
		}

		// Extract means for non-aliased features
		Eigen::VectorXd x_means_valid(X_valid.cols());
		size_t valid_idx = 0;
		for (size_t j = 0; j < p; j++) {
			if (!is_aliased[j]) {
				x_means_valid(valid_idx) = x_means(j);
				valid_idx++;
			}
		}

		// Compute (X'X)^-1
		Eigen::MatrixXd XtX = X_valid.transpose() * X_valid;
		Eigen::MatrixXd XtX_inv = XtX.inverse();

		// SE(intercept) = sqrt(MSE * (1/n + x_mean' * (X'X)^-1 * x_mean))
		double variance_component = x_means_valid.transpose() * XtX_inv * x_means_valid;
		return std::sqrt(mse * (1.0 / n_obs + variance_component));

	} catch (...) {
		return std::numeric_limits<double>::quiet_NaN();
	}
}

inline Eigen::VectorXd CoefficientInference::ComputeTStatistics(
    const Eigen::VectorXd &coefficients,
    const Eigen::VectorXd &std_errors) {

	size_t p = coefficients.size();
	Eigen::VectorXd t_stats(p);

	for (size_t j = 0; j < p; j++) {
		if (std::isnan(coefficients(j)) || std::isnan(std_errors(j)) || std_errors(j) == 0.0) {
			t_stats(j) = std::numeric_limits<double>::quiet_NaN();
		} else {
			t_stats(j) = coefficients(j) / std_errors(j);
		}
	}

	return t_stats;
}

inline Eigen::VectorXd CoefficientInference::ComputePValues(
    const Eigen::VectorXd &t_statistics,
    size_t df) {

	size_t p = t_statistics.size();
	Eigen::VectorXd p_values(p);

	for (size_t j = 0; j < p; j++) {
		if (std::isnan(t_statistics(j))) {
			p_values(j) = std::numeric_limits<double>::quiet_NaN();
		} else {
			p_values(j) = utils::student_t_pvalue(t_statistics(j), df);
		}
	}

	return p_values;
}

inline std::pair<Eigen::VectorXd, Eigen::VectorXd> CoefficientInference::ComputeConfidenceIntervals(
    const Eigen::VectorXd &coefficients,
    const Eigen::VectorXd &std_errors,
    size_t df,
    double confidence_level) {

	size_t p = coefficients.size();

	double alpha = 1.0 - confidence_level;
	double t_crit = utils::student_t_critical(alpha / 2.0, df);

	Eigen::VectorXd ci_lower(p);
	Eigen::VectorXd ci_upper(p);

	for (size_t j = 0; j < p; j++) {
		if (std::isnan(coefficients(j)) || std::isnan(std_errors(j))) {
			ci_lower(j) = std::numeric_limits<double>::quiet_NaN();
			ci_upper(j) = std::numeric_limits<double>::quiet_NaN();
		} else {
			ci_lower(j) = coefficients(j) - t_crit * std_errors(j);
			ci_upper(j) = coefficients(j) + t_crit * std_errors(j);
		}
	}

	return {ci_lower, ci_upper};
}

inline core::InferenceResult CoefficientInference::ComputeInference(
    const core::RegressionResult &result,
    const Eigen::MatrixXd &X,
    const Eigen::VectorXd &y,
    size_t n_obs,
    bool intercept,
    double confidence_level) {

	core::InferenceResult inference;

	size_t p = result.coefficients.size();

	// Compute degrees of freedom: rank now includes intercept if fitted
	size_t n_params_fitted = result.rank;

	if (n_obs <= n_params_fitted) {
		throw std::invalid_argument("Insufficient observations for inference: n <= p");
	}

	size_t df = n_obs - n_params_fitted;

	// Compute MSE from residuals (should be on original scale)
	double mse = result.mse;

	// Compute standard errors for slopes
	Eigen::VectorXd slope_std_errors = ComputeStdErrors(mse, X, result.is_aliased);

	// Compute t-statistics
	Eigen::VectorXd t_stats = ComputeTStatistics(result.coefficients, slope_std_errors);

	// Compute p-values
	Eigen::VectorXd p_vals = ComputePValues(t_stats, df);

	// Compute confidence intervals
	auto [ci_lower, ci_upper] = ComputeConfidenceIntervals(
		result.coefficients, slope_std_errors, df, confidence_level);

	// Store results
	inference.std_errors = slope_std_errors;
	inference.t_statistics = t_stats;
	inference.p_values = p_vals;
	inference.ci_lower = ci_lower;
	inference.ci_upper = ci_upper;
	inference.degrees_of_freedom = df;
	inference.confidence_level = confidence_level;
	inference.has_inference = true;

	return inference;
}

} // namespace inference
} // namespace libanostat
