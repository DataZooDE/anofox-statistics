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
 * Recursive Least Squares (RLS) Regression Solver
 *
 * RLS is an online learning algorithm that updates coefficients sequentially as new data arrives:
 *
 * Algorithm (for observation t):
 *   β_t = β_{t-1} + K_t * (y_t - x_t'β_{t-1})     [Update coefficients]
 *   K_t = P_{t-1}x_t / (λ + x_t'P_{t-1}x_t)       [Kalman gain]
 *   P_t = (1/λ) * (P_{t-1} - K_t x_t' P_{t-1})    [Update covariance]
 *
 * Where:
 * - λ (forgetting_factor) ∈ (0,1] controls memory:
 *   - λ = 1: All observations weighted equally (infinite memory)
 *   - λ < 1: Exponential decay, recent data weighted more (finite memory)
 * - P_t is the inverse covariance matrix (uncertainty in estimates)
 * - K_t is the Kalman gain vector (learning rate for each coefficient)
 *
 * Use cases:
 * - Online learning (data arrives sequentially)
 * - Time-varying parameters
 * - Adaptive filtering
 * - Real-time prediction systems
 *
 * Design notes:
 * - Header-only for performance
 * - No DuckDB dependencies
 * - Handles rank-deficient data (constant columns marked as aliased)
 * - Stateless design (all methods are static)
 */
class RLSSolver {
public:
	/**
	 * Fit Recursive Least Squares regression
	 *
	 * Processes observations sequentially, updating coefficients after each observation.
	 * Final coefficients reflect all observations processed.
	 *
	 * @param y Response vector (length n)
	 * @param X Design matrix (n × p)
	 * @param options Regression options (intercept, forgetting_factor in (0,1])
	 * @return RegressionResult with final coefficients and fit statistics
	 */
	static core::RegressionResult Fit(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
	                                   const core::RegressionOptions &options = core::RegressionOptions::RLS(1.0));

	/**
	 * Fit RLS regression with standard errors for statistical inference
	 *
	 * Note: Standard errors for RLS are approximate. They represent the final model state
	 * uncertainty based on the final covariance matrix P. For time-varying parameters,
	 * these may not accurately reflect the true parameter uncertainty.
	 *
	 * SE_j ≈ sqrt(MSE * P_jj) where P is the final covariance matrix
	 *
	 * @param y Response vector (length n)
	 * @param X Design matrix (n × p)
	 * @param options Regression options
	 * @return RegressionResult with coefficients and approximate std_errors
	 */
	static core::RegressionResult FitWithStdErrors(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
	                                                const core::RegressionOptions &options =
	                                                    core::RegressionOptions::RLS(1.0));

private:
	/**
	 * Compute fit quality statistics (R², adjusted R², RMSE, MSE)
	 */
	static void ComputeStatistics(const Eigen::VectorXd &y, const Eigen::VectorXd &y_pred,
	                              const Eigen::VectorXd &residuals, size_t rank, size_t n,
	                              core::RegressionResult &result);
};

// ============================================================================
// Implementation (header-only for performance)
// ============================================================================

inline core::RegressionResult RLSSolver::Fit(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
                                               const core::RegressionOptions &options) {
	const size_t n = static_cast<size_t>(X.rows());
	const size_t p = static_cast<size_t>(X.cols());

	// Validate options
	options.Validate();

	if (options.forgetting_factor <= 0.0 || options.forgetting_factor > 1.0) {
		throw std::invalid_argument("Forgetting factor must be in range (0, 1]");
	}

	if (n < p + 1) {
		throw std::invalid_argument("Insufficient observations: need at least p+1 observations for p features");
	}

	// Build working matrix (augmented with intercept if needed)
	size_t p_work = options.intercept ? (p + 1) : p;
	Eigen::MatrixXd X_work(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(p_work));

	if (options.intercept) {
		// Augment: X_work = [1, X] where 1 is column of ones
		X_work.col(0) = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(n));
		for (size_t j = 0; j < p; j++) {
			X_work.col(static_cast<Eigen::Index>(j + 1)) = X.col(static_cast<Eigen::Index>(j));
		}
	} else {
		X_work = X;
	}

	// Detect constant columns (will be aliased)
	auto constant_cols_work = OLSSolver::DetectConstantColumns(X_work);

	// Never alias the intercept column (it's constant by design)
	if (options.intercept) {
		constant_cols_work[0] = false;
	}

	// Build list of non-constant column indices
	std::vector<size_t> valid_indices;
	for (size_t j = 0; j < p_work; j++) {
		if (!constant_cols_work[j]) {
			valid_indices.push_back(j);
		}
	}

	size_t p_valid = valid_indices.size();
	if (p_valid == 0) {
		throw std::invalid_argument("All features are constant - cannot fit RLS");
	}

	// Create reduced X matrix with only non-constant columns
	Eigen::MatrixXd X_valid(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(p_valid));
	for (size_t j_new = 0; j_new < p_valid; j_new++) {
		X_valid.col(static_cast<Eigen::Index>(j_new)) =
		    X_work.col(static_cast<Eigen::Index>(valid_indices[j_new]));
	}

	// Initialize RLS state
	// β_0 = 0 (start with zero coefficients)
	// P_0 = large_value * I (large initial uncertainty)
	Eigen::VectorXd beta_valid = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(p_valid));
	double initial_p = 1000.0; // Large initial uncertainty
	Eigen::MatrixXd P = Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(p_valid),
	                                               static_cast<Eigen::Index>(p_valid)) * initial_p;

	// Sequential RLS updates for each observation
	for (size_t t = 0; t < n; t++) {
		auto t_idx = static_cast<Eigen::Index>(t);

		// Get current observation x_t (p_valid × 1 vector)
		Eigen::VectorXd x_t = X_valid.row(t_idx).transpose();
		double y_t = y(t_idx);

		// Prediction: ŷ_t = x_t' β_{t-1}
		double y_pred = x_t.dot(beta_valid);

		// Prediction error: e_t = y_t - ŷ_t
		double error = y_t - y_pred;

		// Compute denominator: λ + x_t' P_{t-1} x_t
		double denominator = options.forgetting_factor + x_t.dot(P * x_t);

		// Kalman gain: K_t = P_{t-1} x_t / (λ + x_t' P_{t-1} x_t)
		Eigen::VectorXd K = P * x_t / denominator;

		// Update coefficients: β_t = β_{t-1} + K_t * e_t
		beta_valid = beta_valid + K * error;

		// Update covariance: P_t = (1/λ) * (P_{t-1} - K_t x_t' P_{t-1})
		P = (P - K * x_t.transpose() * P) / options.forgetting_factor;
	}

	// Initialize result
	core::RegressionResult result(n, p, 0);

	// Map coefficients back to full feature space (with NaN for aliased columns)
	result.coefficients = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(p),
	                                                 std::numeric_limits<double>::quiet_NaN());

	double intercept_value = 0.0;  // Store intercept for later

	if (options.intercept) {
		// Extract intercept (should be at valid_indices position 0 in X_work)
		// Find intercept in valid set
		size_t intercept_pos = std::numeric_limits<size_t>::max();
		for (size_t k = 0; k < valid_indices.size(); k++) {
			if (valid_indices[k] == 0) {
				intercept_pos = k;
				break;
			}
		}

		if (intercept_pos == std::numeric_limits<size_t>::max()) {
			throw std::runtime_error("RLS: Intercept not found in valid indices");
		}

		// Extract intercept value (not stored in coefficients array)
		intercept_value = beta_valid(static_cast<Eigen::Index>(intercept_pos));

		// Map feature coefficients (skip intercept column in X_work)
		for (size_t j = 0; j < p; j++) {
			size_t j_work = j + 1; // Column index in X_work (offset by intercept column)

			if (constant_cols_work[j_work]) {
				// This feature is constant (aliased)
				result.coefficients(static_cast<Eigen::Index>(j)) = std::numeric_limits<double>::quiet_NaN();
				result.is_aliased[j] = true;
			} else {
				// Find j_work in valid_indices
				size_t j_valid_pos = std::numeric_limits<size_t>::max();
				for (size_t k = 0; k < valid_indices.size(); k++) {
					if (valid_indices[k] == j_work) {
						j_valid_pos = k;
						break;
					}
				}
				if (j_valid_pos == std::numeric_limits<size_t>::max()) {
					throw std::runtime_error("RLS: Feature index not found in valid indices");
				}
				result.coefficients(static_cast<Eigen::Index>(j)) = beta_valid(static_cast<Eigen::Index>(j_valid_pos));
				result.is_aliased[j] = false;
			}
		}

		// Rank is number of non-aliased features (not counting intercept)
		result.rank = p_valid - 1;

		// Compute predictions including intercept
		Eigen::VectorXd y_pred = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(n), intercept_value);
		for (size_t j = 0; j < p; j++) {
			if (!result.is_aliased[j]) {
				y_pred += result.coefficients(static_cast<Eigen::Index>(j)) * X.col(static_cast<Eigen::Index>(j));
			}
		}

		result.residuals = y - y_pred;

	} else {
		// Without intercept: map coefficients directly
		for (size_t j = 0; j < p; j++) {
			if (constant_cols_work[j]) {
				result.coefficients(static_cast<Eigen::Index>(j)) = std::numeric_limits<double>::quiet_NaN();
				result.is_aliased[j] = true;
			} else {
				// Find j in valid_indices
				size_t j_valid_pos = std::numeric_limits<size_t>::max();
				for (size_t k = 0; k < valid_indices.size(); k++) {
					if (valid_indices[k] == j) {
						j_valid_pos = k;
						break;
					}
				}
				if (j_valid_pos == std::numeric_limits<size_t>::max()) {
					throw std::runtime_error("RLS: Feature index not found in valid indices");
				}
				result.coefficients(static_cast<Eigen::Index>(j)) = beta_valid(static_cast<Eigen::Index>(j_valid_pos));
				result.is_aliased[j] = false;
			}
		}

		result.rank = p_valid;

		// NEW: Store intercept fields
		result.intercept = intercept_value;
		result.has_intercept = true;

		// Compute predictions
		Eigen::VectorXd y_pred = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n));
		for (size_t j = 0; j < p; j++) {
			if (!result.is_aliased[j]) {
				y_pred += result.coefficients(static_cast<Eigen::Index>(j)) * X.col(static_cast<Eigen::Index>(j));
			}
		}

		result.residuals = y - y_pred;
	}

	// Compute fit statistics
	Eigen::VectorXd y_pred = y - result.residuals;
	ComputeStatistics(y, y_pred, result.residuals, result.rank, n, result);

	return result;
}

inline core::RegressionResult RLSSolver::FitWithStdErrors(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
                                                            const core::RegressionOptions &options) {
	const size_t n = static_cast<size_t>(X.rows());
	const size_t p = static_cast<size_t>(X.cols());

	// Validate
	options.Validate();

	if (options.forgetting_factor <= 0.0 || options.forgetting_factor > 1.0) {
		throw std::invalid_argument("Forgetting factor must be in range (0, 1]");
	}

	// Build working matrix
	size_t p_work = options.intercept ? (p + 1) : p;
	Eigen::MatrixXd X_work(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(p_work));

	if (options.intercept) {
		X_work.col(0) = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(n));
		for (size_t j = 0; j < p; j++) {
			X_work.col(static_cast<Eigen::Index>(j + 1)) = X.col(static_cast<Eigen::Index>(j));
		}
	} else {
		X_work = X;
	}

	// Detect constant columns
	auto constant_cols_work = OLSSolver::DetectConstantColumns(X_work);
	if (options.intercept) {
		constant_cols_work[0] = false;
	}

	// Build valid indices
	std::vector<size_t> valid_indices;
	for (size_t j = 0; j < p_work; j++) {
		if (!constant_cols_work[j]) {
			valid_indices.push_back(j);
		}
	}

	size_t p_valid = valid_indices.size();
	if (p_valid == 0) {
		throw std::invalid_argument("All features are constant - cannot fit RLS");
	}

	// Create reduced matrix
	Eigen::MatrixXd X_valid(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(p_valid));
	for (size_t j_new = 0; j_new < p_valid; j_new++) {
		X_valid.col(static_cast<Eigen::Index>(j_new)) =
		    X_work.col(static_cast<Eigen::Index>(valid_indices[j_new]));
	}

	// Initialize RLS state
	Eigen::VectorXd beta_valid = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(p_valid));
	double initial_p = 1000.0;
	Eigen::MatrixXd P = Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(p_valid),
	                                               static_cast<Eigen::Index>(p_valid)) * initial_p;

	// Sequential RLS updates
	for (size_t t = 0; t < n; t++) {
		auto t_idx = static_cast<Eigen::Index>(t);
		Eigen::VectorXd x_t = X_valid.row(t_idx).transpose();
		double y_t = y(t_idx);
		double y_pred = x_t.dot(beta_valid);
		double error = y_t - y_pred;
		double denominator = options.forgetting_factor + x_t.dot(P * x_t);
		Eigen::VectorXd K = P * x_t / denominator;
		beta_valid = beta_valid + K * error;
		P = (P - K * x_t.transpose() * P) / options.forgetting_factor;
	}

	// Initialize result
	core::RegressionResult result(n, p, 0);
	result.coefficients = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(p),
	                                                 std::numeric_limits<double>::quiet_NaN());
	result.std_errors = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(p),
	                                               std::numeric_limits<double>::quiet_NaN());
	result.has_std_errors = true;

	double intercept_value = 0.0;  // Store intercept for later
	size_t intercept_pos_outer = std::numeric_limits<size_t>::max();  // Store position for later

	// Map coefficients and compute standard errors
	if (options.intercept) {
		// Find intercept position
		size_t intercept_pos = std::numeric_limits<size_t>::max();
		for (size_t k = 0; k < valid_indices.size(); k++) {
			if (valid_indices[k] == 0) {
				intercept_pos = k;
				break;
			}
		}

		intercept_pos_outer = intercept_pos;  // Store for later use
		intercept_value = beta_valid(static_cast<Eigen::Index>(intercept_pos));

		// Map feature coefficients
		for (size_t j = 0; j < p; j++) {
			size_t j_work = j + 1;

			if (constant_cols_work[j_work]) {
				result.is_aliased[j] = true;
			} else {
				size_t j_valid_pos = std::numeric_limits<size_t>::max();
				for (size_t k = 0; k < valid_indices.size(); k++) {
					if (valid_indices[k] == j_work) {
						j_valid_pos = k;
						break;
					}
				}
				result.coefficients(static_cast<Eigen::Index>(j)) = beta_valid(static_cast<Eigen::Index>(j_valid_pos));
				result.is_aliased[j] = false;
			}
		}

		result.rank = p_valid - 1;

		// Compute predictions
		Eigen::VectorXd y_pred = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(n), intercept_value);
		for (size_t j = 0; j < p; j++) {
			if (!result.is_aliased[j]) {
				y_pred += result.coefficients(static_cast<Eigen::Index>(j)) * X.col(static_cast<Eigen::Index>(j));
			}
		}
		result.residuals = y - y_pred;

		// Compute statistics first to get MSE
		ComputeStatistics(y, y_pred, result.residuals, result.rank, n, result);

		// Approximate standard errors from final covariance matrix
		// SE_j ≈ sqrt(MSE * P_jj)
		for (size_t j = 0; j < p; j++) {
			if (!result.is_aliased[j]) {
				size_t j_work = j + 1;
				size_t j_valid_pos = std::numeric_limits<size_t>::max();
				for (size_t k = 0; k < valid_indices.size(); k++) {
					if (valid_indices[k] == j_work) {
						j_valid_pos = k;
						break;
					}
				}
				auto j_idx = static_cast<Eigen::Index>(j_valid_pos);
				double var_j = result.mse * P(j_idx, j_idx);
				result.std_errors(static_cast<Eigen::Index>(j)) = std::sqrt(std::max(0.0, var_j));
			}
		}

	} else {
		// Without intercept
		for (size_t j = 0; j < p; j++) {
			if (constant_cols_work[j]) {
				result.is_aliased[j] = true;
			} else {
				size_t j_valid_pos = std::numeric_limits<size_t>::max();
				for (size_t k = 0; k < valid_indices.size(); k++) {
					if (valid_indices[k] == j) {
						j_valid_pos = k;
						break;
					}
				}
				result.coefficients(static_cast<Eigen::Index>(j)) = beta_valid(static_cast<Eigen::Index>(j_valid_pos));
				result.is_aliased[j] = false;
			}
		}

		result.rank = p_valid;

		Eigen::VectorXd y_pred = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n));
		for (size_t j = 0; j < p; j++) {
			if (!result.is_aliased[j]) {
				y_pred += result.coefficients(static_cast<Eigen::Index>(j)) * X.col(static_cast<Eigen::Index>(j));
			}
		}
		result.residuals = y - y_pred;

		ComputeStatistics(y, y_pred, result.residuals, result.rank, n, result);

		// Compute standard errors
		for (size_t j = 0; j < p; j++) {
			if (!result.is_aliased[j]) {
				size_t j_valid_pos = std::numeric_limits<size_t>::max();
				for (size_t k = 0; k < valid_indices.size(); k++) {
					if (valid_indices[k] == j) {
						j_valid_pos = k;
						break;
					}
				}
				auto j_idx = static_cast<Eigen::Index>(j_valid_pos);
				double var_j = result.mse * P(j_idx, j_idx);
				result.std_errors(static_cast<Eigen::Index>(j)) = std::sqrt(std::max(0.0, var_j));
			}
		}

		// NEW: Compute intercept standard error
		auto intercept_idx = static_cast<Eigen::Index>(intercept_pos_outer);
		double var_intercept = result.mse * P(intercept_idx, intercept_idx);
		result.intercept_std_error = std::sqrt(std::max(0.0, var_intercept));
	}

	// NEW: Store intercept fields
	result.intercept = intercept_value;
	result.has_intercept = true;

	return result;
}

inline void RLSSolver::ComputeStatistics(const Eigen::VectorXd &y, const Eigen::VectorXd &y_pred,
                                          const Eigen::VectorXd &residuals, size_t rank, size_t n,
                                          core::RegressionResult &result) {
	// Sum of squared residuals
	double ss_res = residuals.squaredNorm();

	// Total sum of squares
	double y_mean = y.mean();
	double ss_tot = (y.array() - y_mean).square().sum();

	// R-squared
	result.r_squared = (ss_tot > 1e-10) ? (1.0 - ss_res / ss_tot) : 0.0;

	// Bound R-squared to valid range [0, 1]
	if (result.r_squared < 0.0) {
		result.r_squared = 0.0;
	} else if (result.r_squared > 1.0) {
		result.r_squared = 1.0;
	}

	// Adjusted R-squared
	// df = n - rank - (intercept ? 1 : 0), but rank already accounts for this
	if (n > rank + 1) {
		double adj_factor = static_cast<double>(n - 1) / static_cast<double>(n - rank - 1);
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

	// MSE
	result.mse = ss_res / static_cast<double>(n);

	// RMSE
	result.rmse = std::sqrt(result.mse);
}

} // namespace solvers
} // namespace libanostat
