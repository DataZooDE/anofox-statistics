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
 * Ridge Regression Solver with L2 Regularization
 *
 * Ridge regression adds an L2 penalty term to the OLS objective function:
 *   minimize: ||y - Xβ||² + λ||β||²
 *
 * Solution: β = (X'X + λI)^(-1) X'y
 *
 * Where:
 * - λ (lambda) is the regularization parameter
 * - I is the identity matrix
 * - Higher λ means more regularization (coefficients shrink towards zero)
 * - When λ=0, reduces to standard OLS
 *
 * Design notes:
 * - Header-only for performance
 * - No DuckDB dependencies
 * - Leverages OLSSolver when lambda=0
 * - Stateless design (all methods are static)
 */
class RidgeSolver {
public:
	/**
	 * Fit Ridge regression with L2 regularization
	 *
	 * @param y Response vector (length n)
	 * @param X Design matrix (n × p)
	 * @param options Regression options (must include lambda > 0 for Ridge)
	 * @return RegressionResult with coefficients and fit statistics
	 */
	static core::RegressionResult Fit(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
	                                   const core::RegressionOptions &options = core::RegressionOptions::Ridge(1.0));

	/**
	 * Fit Ridge regression with standard errors for statistical inference
	 *
	 * Note: Ridge standard errors use approximation: sqrt(MSE * diag((X'X + λI)^-1))
	 * This is computationally efficient but not as accurate as bootstrap methods
	 *
	 * @param y Response vector (length n)
	 * @param X Design matrix (n × p)
	 * @param options Regression options
	 * @return RegressionResult with coefficients and std_errors
	 */
	static core::RegressionResult FitWithStdErrors(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
	                                                const core::RegressionOptions &options =
	                                                    core::RegressionOptions::Ridge(1.0));

private:
	/**
	 * Compute fit quality statistics (R², adjusted R², RMSE, MSE)
	 */
	static void ComputeStatistics(const Eigen::VectorXd &y, const Eigen::VectorXd &y_pred,
	                              const Eigen::VectorXd &residuals, size_t rank, size_t n,
	                              core::RegressionResult &result);

	/**
	 * Compute approximate standard errors for Ridge regression
	 *
	 * Uses formula: SE_j = sqrt(MSE * (X'X + λI)^-1_jj)
	 */
	static void ComputeStandardErrors(const Eigen::MatrixXd &XtX_regularized, double mse, size_t p,
	                                  const std::vector<bool> &constant_features, core::RegressionResult &result);
};

// ============================================================================
// Implementation (header-only for performance)
// ============================================================================

inline core::RegressionResult RidgeSolver::Fit(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
                                                const core::RegressionOptions &options) {
	const size_t n = static_cast<size_t>(X.rows());
	const size_t p_user = static_cast<size_t>(X.cols());

	// Validate options
	options.Validate();

	if (options.lambda < 0.0) {
		throw std::invalid_argument("Lambda must be non-negative for Ridge regression");
	}

	// Special case: λ=0 reduces to OLS
	if (options.lambda == 0.0) {
		return OLSSolver::Fit(y, X, options);
	}

	// Auto-add intercept column if requested (user-friendly API to match OLS)
	Eigen::MatrixXd X_work;
	size_t p;
	if (options.intercept) {
		// Prepend column of ones for intercept
		p = p_user + 1;
		X_work.resize(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(p));
		X_work.col(0) = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(n));
		X_work.rightCols(static_cast<Eigen::Index>(p_user)) = X;
	} else {
		// Use X as-is
		p = p_user;
		X_work = X;
	}

	// Initialize result
	core::RegressionResult result(n, p, 0);

	// Detect constant features (these cause numerical issues even with regularization)
	std::vector<bool> constant_features = OLSSolver::DetectConstantColumns(X_work);

	// Ridge regression: β = (X'X + λI)^(-1) X'y
	// NOTE: When intercept=true, X_work already contains intercept column at position 0
	Eigen::MatrixXd XtX = X_work.transpose() * X_work;
	Eigen::VectorXd Xty = X_work.transpose() * y;

	// Add regularization: X'X + λI
	// IMPORTANT: Do NOT penalize intercept (standard Ridge regression practice)
	Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(p));
	if (options.intercept) {
		// Set first diagonal element to 0 to exclude intercept from penalty
		identity(0, 0) = 0.0;
	}
	Eigen::MatrixXd XtX_regularized = XtX + options.lambda * identity;

	// Use ColPivHouseholderQR for rank-revealing solve
	Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(XtX_regularized);
	result.rank = static_cast<size_t>(qr.rank());

	// Solve for coefficients (these are coefficients for centered data if intercept=true)
	Eigen::VectorXd beta = qr.solve(Xty);

	// Store coefficients (set NaN for constant features)
	for (size_t i = 0; i < p; i++) {
		auto i_idx = static_cast<Eigen::Index>(i);
		if (constant_features[i]) {
			result.coefficients[i_idx] = std::numeric_limits<double>::quiet_NaN();
			result.is_aliased[i] = true;
		} else {
			result.coefficients[i_idx] = beta(i_idx);
			result.is_aliased[i] = false;
		}
	}

	// Compute predictions using X_work (which includes intercept column if needed)
	Eigen::VectorXd y_pred = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n));
	for (size_t j = 0; j < p; j++) {
		if (!constant_features[j] && std::isfinite(result.coefficients[static_cast<Eigen::Index>(j)])) {
			auto j_idx = static_cast<Eigen::Index>(j);
			y_pred += result.coefficients[j_idx] * X_work.col(j_idx);
		}
	}

	// Compute residuals
	result.residuals = y - y_pred;

	// Compute fit statistics
	ComputeStatistics(y, y_pred, result.residuals, result.rank, n, result);

	return result;
}

inline core::RegressionResult RidgeSolver::FitWithStdErrors(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
                                                             const core::RegressionOptions &options) {
	const size_t n = static_cast<size_t>(X.rows());
	const size_t p_user = static_cast<size_t>(X.cols());

	// Special case: λ=0 reduces to OLS
	if (options.lambda == 0.0) {
		return OLSSolver::FitWithStdErrors(y, X, options);
	}

	// First, perform basic fit
	auto result = Fit(y, X, options);

	// Auto-add intercept column if requested (match API)
	Eigen::MatrixXd X_work;
	size_t p;
	if (options.intercept) {
		p = p_user + 1;
		X_work.resize(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(p));
		X_work.col(0) = Eigen::VectorXd::Ones(static_cast<Eigen::Index>(n));
		X_work.rightCols(static_cast<Eigen::Index>(p_user)) = X;
	} else {
		p = p_user;
		X_work = X;
	}

	// Initialize std_errors with NaN
	result.std_errors =
	    Eigen::VectorXd::Constant(static_cast<Eigen::Index>(p), std::numeric_limits<double>::quiet_NaN());
	result.has_std_errors = true;

	// Detect constant features
	std::vector<bool> constant_features = OLSSolver::DetectConstantColumns(X_work);

	// Compute X'X + λI (X_work already includes intercept column if needed)
	Eigen::MatrixXd XtX = X_work.transpose() * X_work;
	Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(p));
	if (options.intercept) {
		// Do NOT penalize intercept (standard Ridge regression practice)
		identity(0, 0) = 0.0;
	}
	Eigen::MatrixXd XtX_regularized = XtX + options.lambda * identity;

	// Compute standard errors
	ComputeStandardErrors(XtX_regularized, result.mse, p, constant_features, result);

	return result;
}

inline void RidgeSolver::ComputeStatistics(const Eigen::VectorXd &y, const Eigen::VectorXd &y_pred,
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

	// MSE
	result.mse = ss_res / static_cast<double>(n);

	// RMSE
	result.rmse = std::sqrt(result.mse);
}

inline void RidgeSolver::ComputeStandardErrors(const Eigen::MatrixXd &XtX_regularized, double mse, size_t p,
                                               const std::vector<bool> &constant_features,
                                               core::RegressionResult &result) {
	try {
		// Compute (X'X + λI)^-1
		Eigen::MatrixXd XtX_reg_inv = XtX_regularized.inverse();

		// Standard errors: SE_j = sqrt(MSE * (X'X + λI)^-1_jj)
		for (size_t j = 0; j < p; j++) {
			auto j_idx = static_cast<Eigen::Index>(j);
			if (constant_features[j]) {
				result.std_errors[j_idx] = std::numeric_limits<double>::quiet_NaN();
			} else {
				double var_j = mse * XtX_reg_inv(j_idx, j_idx);
				result.std_errors[j_idx] = std::sqrt(std::max(0.0, var_j));
			}
		}
	} catch (...) {
		// If inversion fails, leave standard errors as NaN
	}
}

} // namespace solvers
} // namespace libanostat
