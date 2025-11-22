#pragma once

#include "libanostat/core/regression_result.hpp"
#include "libanostat/core/regression_options.hpp"
#include "libanostat/solvers/ols_solver.hpp"
#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <cmath>

namespace libanostat {
namespace solvers {

/**
 * Weighted Least Squares (WLS) Regression Solver
 *
 * WLS extends OLS by allowing different weights for observations:
 *   minimize: Σ w_i * (y_i - x_i'β)²
 *
 * Where:
 * - w_i > 0 is the weight for observation i
 * - Higher weights give more importance to certain observations
 * - Useful for handling heteroscedasticity (non-constant variance)
 * - When all weights = 1, reduces to standard OLS
 *
 * Algorithm: Transform to weighted OLS
 * 1. Compute sqrt(W) where W is diagonal matrix of weights
 * 2. Transform: X_weighted = sqrt(W) * X, y_weighted = sqrt(W) * y
 * 3. Solve OLS on weighted matrices using rank-deficient solver
 * 4. Coefficients from weighted problem are correct for original X, y
 *
 * Design notes:
 * - Header-only for performance
 * - No DuckDB dependencies
 * - Leverages OLSSolver for actual solving (code reuse)
 * - Stateless design (all methods are static)
 */
class WLSSolver {
public:
	/**
	 * Fit Weighted Least Squares regression
	 *
	 * @param y Response vector (length n)
	 * @param X Design matrix (n × p)
	 * @param weights Weight vector (length n), all weights must be > 0
	 * @param options Regression options (intercept, tolerance, etc.)
	 * @return RegressionResult with coefficients and fit statistics
	 */
	static core::RegressionResult Fit(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
	                                   const Eigen::VectorXd &weights,
	                                   const core::RegressionOptions &options = core::RegressionOptions::OLS());

	/**
	 * Fit WLS regression with standard errors for statistical inference
	 *
	 * Computes standard errors for non-aliased coefficients.
	 * Standard errors account for the weighting.
	 *
	 * @param y Response vector (length n)
	 * @param X Design matrix (n × p)
	 * @param weights Weight vector (length n), all weights must be > 0
	 * @param options Regression options
	 * @return RegressionResult with coefficients and std_errors
	 */
	static core::RegressionResult FitWithStdErrors(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
	                                                const Eigen::VectorXd &weights,
	                                                const core::RegressionOptions &options =
	                                                    core::RegressionOptions::OLS());

private:
	/**
	 * Compute fit quality statistics using weighted residuals
	 *
	 * For WLS, R² is computed using weighted sums of squares:
	 * R² = 1 - (Σ w_i * residual_i²) / (Σ w_i * (y_i - ȳ_weighted)²)
	 */
	static void ComputeWeightedStatistics(const Eigen::VectorXd &y, const Eigen::VectorXd &y_pred,
	                                      const Eigen::VectorXd &residuals, const Eigen::VectorXd &weights,
	                                      size_t rank, size_t n, core::RegressionResult &result);
};

// ============================================================================
// Implementation (header-only for performance)
// ============================================================================

inline core::RegressionResult WLSSolver::Fit(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
                                               const Eigen::VectorXd &weights,
                                               const core::RegressionOptions &options) {
	const size_t n = static_cast<size_t>(X.rows());
	const size_t p = static_cast<size_t>(X.cols());

	// Validate options
	options.Validate();

	// Validate weights
	if (static_cast<size_t>(weights.size()) != n) {
		throw std::invalid_argument("Weights vector must have same length as y");
	}

	for (size_t i = 0; i < n; i++) {
		auto i_idx = static_cast<Eigen::Index>(i);
		if (weights(i_idx) <= 0.0) {
			throw std::invalid_argument("All weights must be positive");
		}
	}

	// Special case: if all weights are equal, reduce to OLS (optimization)
	bool all_equal = true;
	double first_weight = weights(0);
	for (size_t i = 1; i < n; i++) {
		if (std::abs(weights(static_cast<Eigen::Index>(i)) - first_weight) > 1e-10) {
			all_equal = false;
			break;
		}
	}
	if (all_equal) {
		return OLSSolver::Fit(y, X, options);
	}

	// Compute sqrt(weights) for transformation
	Eigen::VectorXd sqrt_w = weights.array().sqrt();

	// Compute weighted means (needed for centering if intercept=true and for R²)
	double sum_weights = weights.sum();
	double y_weighted_mean = (weights.array() * y.array()).sum() / sum_weights;
	Eigen::VectorXd x_weighted_means = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(p));
	for (size_t j = 0; j < p; j++) {
		auto j_idx = static_cast<Eigen::Index>(j);
		x_weighted_means(j_idx) = (weights.array() * X.col(j_idx).array()).sum() / sum_weights;
	}

	// Work matrices (will be centered if options.intercept=true)
	Eigen::MatrixXd X_work = X;
	Eigen::VectorXd y_work = y;

	if (options.intercept) {
		// Center the data BEFORE applying sqrt(W)
		// This ensures the regression coefficients are for centered data
		for (size_t i = 0; i < n; i++) {
			auto i_idx = static_cast<Eigen::Index>(i);
			y_work(i_idx) = y(i_idx) - y_weighted_mean;
			for (size_t j = 0; j < p; j++) {
				auto j_idx = static_cast<Eigen::Index>(j);
				X_work(i_idx, j_idx) = X(i_idx, j_idx) - x_weighted_means(j_idx);
			}
		}
	}

	// Apply sqrt(W) transformation to centered (or uncentered) data
	Eigen::MatrixXd X_weighted = sqrt_w.asDiagonal() * X_work;
	Eigen::VectorXd y_weighted = sqrt_w.asDiagonal() * y_work;

	// Use rank-deficient OLS solver on weighted, centered matrices
	auto result = OLSSolver::Fit(y_weighted, X_weighted, options);

	// Coefficients are already correct (they're the same whether computed on
	// centered or uncentered data, as long as we use the intercept correction below)

	// Compute predictions on ORIGINAL scale using coefficients from weighted problem
	Eigen::VectorXd y_pred = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n));
	for (size_t j = 0; j < p; j++) {
		auto j_idx = static_cast<Eigen::Index>(j);
		if (!result.is_aliased[j]) {
			y_pred += result.coefficients[j_idx] * X.col(j_idx);
		}
	}

	// Add intercept if needed (computed from centering relationship)
	if (options.intercept) {
		double beta_dot_xmean = 0.0;
		for (size_t j = 0; j < p; j++) {
			auto j_idx = static_cast<Eigen::Index>(j);
			if (!result.is_aliased[j]) {
				beta_dot_xmean += result.coefficients[j_idx] * x_weighted_means(j_idx);
			}
		}
		double intercept = y_weighted_mean - beta_dot_xmean;
		y_pred.array() += intercept;
	}

	// Compute residuals
	result.residuals = y - y_pred;

	// Compute weighted fit statistics
	ComputeWeightedStatistics(y, y_pred, result.residuals, weights, result.rank, n, result);

	return result;
}

inline core::RegressionResult WLSSolver::FitWithStdErrors(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
                                                            const Eigen::VectorXd &weights,
                                                            const core::RegressionOptions &options) {
	const size_t n = static_cast<size_t>(X.rows());
	const size_t p = static_cast<size_t>(X.cols());

	// Validate weights
	if (static_cast<size_t>(weights.size()) != n) {
		throw std::invalid_argument("Weights vector must have same length as y");
	}

	for (size_t i = 0; i < n; i++) {
		auto i_idx = static_cast<Eigen::Index>(i);
		if (weights(i_idx) <= 0.0) {
			throw std::invalid_argument("All weights must be positive");
		}
	}

	// Special case: if all weights are equal, reduce to OLS
	bool all_equal = true;
	double first_weight = weights(0);
	for (size_t i = 1; i < n; i++) {
		if (std::abs(weights(static_cast<Eigen::Index>(i)) - first_weight) > 1e-10) {
			all_equal = false;
			break;
		}
	}
	if (all_equal) {
		return OLSSolver::FitWithStdErrors(y, X, options);
	}

	// Compute sqrt(weights) for transformation
	Eigen::VectorXd sqrt_w = weights.array().sqrt();

	// Compute weighted means
	double sum_weights = weights.sum();
	double y_weighted_mean = (weights.array() * y.array()).sum() / sum_weights;
	Eigen::VectorXd x_weighted_means = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(p));
	for (size_t j = 0; j < p; j++) {
		auto j_idx = static_cast<Eigen::Index>(j);
		x_weighted_means(j_idx) = (weights.array() * X.col(j_idx).array()).sum() / sum_weights;
	}

	// Work matrices
	Eigen::MatrixXd X_work = X;
	Eigen::VectorXd y_work = y;

	if (options.intercept) {
		for (size_t i = 0; i < n; i++) {
			auto i_idx = static_cast<Eigen::Index>(i);
			y_work(i_idx) = y(i_idx) - y_weighted_mean;
			for (size_t j = 0; j < p; j++) {
				auto j_idx = static_cast<Eigen::Index>(j);
				X_work(i_idx, j_idx) = X(i_idx, j_idx) - x_weighted_means(j_idx);
			}
		}
	}

	// Apply sqrt(W) transformation
	Eigen::MatrixXd X_weighted = sqrt_w.asDiagonal() * X_work;
	Eigen::VectorXd y_weighted = sqrt_w.asDiagonal() * y_work;

	// Use OLS solver with standard errors on weighted matrices
	auto result = OLSSolver::FitWithStdErrors(y_weighted, X_weighted, options);

	// Compute predictions on original scale
	Eigen::VectorXd y_pred = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n));
	for (size_t j = 0; j < p; j++) {
		auto j_idx = static_cast<Eigen::Index>(j);
		if (!result.is_aliased[j]) {
			y_pred += result.coefficients[j_idx] * X.col(j_idx);
		}
	}

	if (options.intercept) {
		double beta_dot_xmean = 0.0;
		for (size_t j = 0; j < p; j++) {
			auto j_idx = static_cast<Eigen::Index>(j);
			if (!result.is_aliased[j]) {
				beta_dot_xmean += result.coefficients[j_idx] * x_weighted_means(j_idx);
			}
		}
		double intercept = y_weighted_mean - beta_dot_xmean;
		y_pred.array() += intercept;
	}

	// Compute residuals
	result.residuals = y - y_pred;

	// Compute weighted fit statistics
	ComputeWeightedStatistics(y, y_pred, result.residuals, weights, result.rank, n, result);

	return result;
}

inline void WLSSolver::ComputeWeightedStatistics(const Eigen::VectorXd &y, const Eigen::VectorXd &y_pred,
                                                  const Eigen::VectorXd &residuals, const Eigen::VectorXd &weights,
                                                  size_t rank, size_t n, core::RegressionResult &result) {
	// Weighted sum of squares
	double sum_weights = weights.sum();
	double y_weighted_mean = (weights.array() * y.array()).sum() / sum_weights;

	double ss_res_weighted = (weights.array() * residuals.array().square()).sum();
	double ss_tot_weighted = (weights.array() * (y.array() - y_weighted_mean).square()).sum();

	// Weighted R-squared
	result.r_squared = (ss_tot_weighted > 1e-10) ? (1.0 - ss_res_weighted / ss_tot_weighted) : 0.0;

	// Bound R-squared to valid range [0, 1]
	if (result.r_squared < 0.0) {
		result.r_squared = 0.0;
	} else if (result.r_squared > 1.0) {
		result.r_squared = 1.0;
	}

	// Adjusted R-squared using effective rank
	if (n > rank + 1) {
		double adj_factor = static_cast<double>(n - 1) / static_cast<double>(n - rank);
		result.adj_r_squared = 1.0 - (1.0 - result.r_squared) * adj_factor;

		// Bound adjusted R-squared as well
		if (result.adj_r_squared < 0.0) {
			result.adj_r_squared = 0.0;
		} else if (result.adj_r_squared > 1.0) {
			result.adj_r_squared = 1.0;
		}
	} else {
		result.adj_r_squared = result.r_squared;
	}

	// MSE and RMSE (using weighted residuals)
	if (n > rank) {
		result.mse = ss_res_weighted / static_cast<double>(n - rank);
	} else {
		result.mse = 0.0;
	}
	result.rmse = std::sqrt(result.mse);
}

} // namespace solvers
} // namespace libanostat
