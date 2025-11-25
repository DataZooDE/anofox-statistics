#pragma once

#include "libanostat/core/regression_result.hpp"
#include "libanostat/core/regression_options.hpp"
#include "libanostat/solvers/ridge_solver.hpp"
#include "libanostat/utils/distributions.hpp"
#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <cmath>

namespace libanostat {
namespace solvers {

/**
 * Elastic Net Regression Solver using Coordinate Descent
 *
 * Elastic Net combines L1 (Lasso) and L2 (Ridge) penalties:
 *   minimize: ||y - Xβ||² + λ(α||β||₁ + (1-α)||β||₂²)
 *
 * Where:
 * - λ (lambda) is the regularization strength (>= 0)
 * - α (alpha) is the mixing parameter [0,1]:
 *   - α = 0: Pure Ridge regression (L2 only)
 *   - α = 1: Pure Lasso regression (L1 only)
 *   - 0 < α < 1: Elastic Net (combination of L1 and L2)
 *
 * Algorithm: Coordinate Descent
 * - Cyclically update one coefficient at a time
 * - Uses soft thresholding for L1 penalty
 * - Converges when max coefficient change < tolerance
 *
 * Design notes:
 * - Header-only for performance
 * - No DuckDB dependencies
 * - Leverages RidgeSolver when alpha=0 (code reuse)
 * - Stateless design (all methods are static)
 */
class ElasticNetSolver {
public:
	/**
	 * Fit Elastic Net regression using coordinate descent
	 *
	 * @param y Response vector (length n)
	 * @param X Design matrix (n × p), assumed already centered if intercept=true
	 * @param options Regression options (must include lambda >= 0, alpha in [0,1])
	 * @return RegressionResult with coefficients and fit statistics
	 */
	static core::RegressionResult Fit(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
	                                   const core::RegressionOptions &options = core::RegressionOptions::ElasticNet(1.0, 0.5));

	/**
	 * Fit Elastic Net regression with standard errors
	 *
	 * Note: Standard errors for Elastic Net are complex and typically require bootstrap.
	 * This method returns NaN for std_errors (not yet implemented).
	 *
	 * @param y Response vector (length n)
	 * @param X Design matrix (n × p)
	 * @param options Regression options
	 * @return RegressionResult with coefficients (std_errors are NaN)
	 */
	static core::RegressionResult FitWithStdErrors(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
	                                                const core::RegressionOptions &options =
	                                                    core::RegressionOptions::ElasticNet(1.0, 0.5));

private:
	/**
	 * Soft thresholding operator for L1 penalty
	 *
	 * S(z, γ) = sign(z) * max(|z| - γ, 0)
	 *
	 * This is the proximal operator for the L1 norm.
	 */
	static double SoftThreshold(double z, double gamma);

	/**
	 * Compute fit quality statistics (R², adjusted R², RMSE, MSE)
	 */
	static void ComputeStatistics(const Eigen::VectorXd &y, const Eigen::VectorXd &y_pred,
	                              const Eigen::VectorXd &residuals, size_t n_nonzero, size_t n,
	                              core::RegressionResult &result);
};

// ============================================================================
// Implementation (header-only for performance)
// ============================================================================

inline core::RegressionResult ElasticNetSolver::Fit(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
                                                      const core::RegressionOptions &options) {
	const size_t n = static_cast<size_t>(X.rows());
	const size_t p = static_cast<size_t>(X.cols());

	// Validate options
	options.Validate();

	if (options.lambda < 0.0) {
		throw std::invalid_argument("Lambda must be non-negative for Elastic Net");
	}
	if (options.alpha < 0.0 || options.alpha > 1.0) {
		throw std::invalid_argument("Alpha must be in [0,1] for Elastic Net");
	}

	// Special case: α=0 reduces to pure Ridge regression
	if (options.alpha <= 1e-10) {
		return RidgeSolver::Fit(y, X, options);
	}

	// Center data if intercept requested (similar to OLS)
	Eigen::VectorXd y_work;
	Eigen::MatrixXd X_work;
	Eigen::VectorXd x_means;
	double y_mean = 0.0;

	if (options.intercept) {
		// Center y and X
		y_mean = y.mean();
		x_means = X.colwise().mean();

		y_work = y.array() - y_mean;
		X_work = X;
		for (size_t j = 0; j < p; j++) {
			auto j_idx = static_cast<Eigen::Index>(j);
			X_work.col(j_idx) = X.col(j_idx).array() - x_means(j_idx);
		}
	} else {
		y_work = y;
		X_work = X;
	}

	// Initialize result
	core::RegressionResult result(n, p, 0);
	result.coefficients = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(p));

	// Precompute column norms squared (for efficiency in coordinate descent)
	Eigen::VectorXd x_norms_sq(static_cast<Eigen::Index>(p));
	for (size_t j = 0; j < p; j++) {
		auto j_idx = static_cast<Eigen::Index>(j);
		x_norms_sq(j_idx) = X_work.col(j_idx).squaredNorm();
	}

	// Initialize residuals
	Eigen::VectorXd residuals = y_work;

	// Coordinate descent iterations
	size_t iterations = 0;
	bool converged = false;

	for (size_t iter = 0; iter < options.max_iterations; iter++) {
		double max_change = 0.0;

		// Cycle through all coefficients
		for (size_t j = 0; j < p; j++) {
			auto j_idx = static_cast<Eigen::Index>(j);

			// Skip if column is constant/zero (no variance)
			if (x_norms_sq(j_idx) < 1e-10) {
				result.coefficients[j_idx] = std::numeric_limits<double>::quiet_NaN();
				result.is_aliased[j] = true;
				continue;
			}

			double beta_old = result.coefficients(j_idx);

			// Add back the contribution of feature j to residuals
			// (so we can compute the optimal coefficient for j in isolation)
			residuals += beta_old * X_work.col(j_idx);

			// Compute partial correlation: X_j' * residuals
			double rho = X_work.col(j_idx).dot(residuals);

			// Apply soft thresholding for L1 penalty
			double threshold = options.lambda * options.alpha * static_cast<double>(n);
			double z = SoftThreshold(rho, threshold);

			// Update coefficient with L2 penalty in denominator
			double denominator = x_norms_sq(j_idx) + options.lambda * (1.0 - options.alpha) * static_cast<double>(n);
			double beta_new = z / denominator;

			result.coefficients(j_idx) = beta_new;

			// Update residuals by removing new contribution
			residuals -= beta_new * X_work.col(j_idx);

			// Track maximum coefficient change for convergence
			double change = std::abs(beta_new - beta_old);
			if (change > max_change) {
				max_change = change;
			}
		}

		iterations = iter + 1;

		// Check convergence
		if (max_change < options.tolerance) {
			converged = true;
			break;
		}
	}

	// Count non-zero coefficients (for sparsity diagnostics)
	size_t n_nonzero = 0;
	for (size_t j = 0; j < p; j++) {
		auto j_idx = static_cast<Eigen::Index>(j);
		if (!result.is_aliased[j] && std::abs(result.coefficients(j_idx)) > 1e-10) {
			n_nonzero++;
		}
	}

	// Set rank to number of non-zero coefficients
	// (This is the "effective" rank for sparse models)
	result.rank = n_nonzero;

	// Compute intercept if requested (similar to OLS)
	double intercept = 0.0;
	if (options.intercept) {
		// intercept = y_mean - sum(coef[j] * x_mean[j])
		intercept = y_mean;
		for (size_t j = 0; j < p; j++) {
			auto j_idx = static_cast<Eigen::Index>(j);
			if (!result.is_aliased[j]) {
				intercept -= result.coefficients(j_idx) * x_means(j_idx);
			}
		}

		// Populate intercept fields
		result.intercept = intercept;
		result.has_intercept = true;
	}

	// Compute final predictions on ORIGINAL (uncentered) data
	Eigen::VectorXd y_pred;
	if (options.intercept) {
		y_pred = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(n), intercept) + X * result.coefficients;
	} else {
		y_pred = X * result.coefficients;
	}
	result.residuals = y - y_pred;

	// Compute fit statistics
	ComputeStatistics(y, y_pred, result.residuals, n_nonzero, n, result);

	return result;
}

inline core::RegressionResult ElasticNetSolver::FitWithStdErrors(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
                                                                   const core::RegressionOptions &options) {
	const size_t n = static_cast<size_t>(X.rows());
	const size_t p = static_cast<size_t>(X.cols());

	// Special case: α=0 reduces to Ridge with proper standard errors
	if (options.alpha <= 1e-10) {
		return RidgeSolver::FitWithStdErrors(y, X, options);
	}

	// First, perform basic fit
	auto result = Fit(y, X, options);

	// Initialize std_errors with NaN (not yet implemented for Elastic Net)
	// Standard errors for Elastic Net require bootstrap or advanced approximations
	result.std_errors =
	    Eigen::VectorXd::Constant(static_cast<Eigen::Index>(p), std::numeric_limits<double>::quiet_NaN());
	result.has_std_errors = true;

	// Compute simple intercept standard error estimate if intercept was fitted
	if (options.intercept && result.has_intercept) {
		// Simple estimate: SE(intercept) = sqrt(MSE / n)
		result.intercept_std_error = std::sqrt(result.mse / static_cast<double>(n));
	}

	return result;
}

inline double ElasticNetSolver::SoftThreshold(double z, double gamma) {
	if (z > gamma) {
		return z - gamma;
	} else if (z < -gamma) {
		return z + gamma;
	} else {
		return 0.0;
	}
}

inline void ElasticNetSolver::ComputeStatistics(const Eigen::VectorXd &y, const Eigen::VectorXd &y_pred,
                                                 const Eigen::VectorXd &residuals, size_t n_nonzero, size_t n,
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

	// Adjusted R-squared using number of non-zero coefficients
	// df = n - n_nonzero (degrees of freedom)
	size_t df = (n > n_nonzero) ? (n - n_nonzero) : 1;
	if (n > 1) {
		double adj_factor = static_cast<double>(n - 1) / static_cast<double>(df);
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
	result.mse = ss_res / static_cast<double>(df);

	// RMSE
	result.rmse = std::sqrt(result.mse);

	// Residual standard error (same as RMSE for consistency with R)
	result.residual_standard_error = result.rmse;

	// F-statistic for overall model significance
	size_t df_model = n_nonzero;
	size_t df_residual = df;

	if (df_residual > 0 && df_model > 0 && result.mse > 0.0 && ss_tot > 1e-10) {
		double explained_ss = ss_tot - ss_res;
		double mean_sq_model = explained_ss / static_cast<double>(df_model);
		result.f_statistic = mean_sq_model / result.mse;

		// Compute F-statistic p-value
		result.f_statistic_pvalue = utils::f_distribution_pvalue(result.f_statistic,
		                                                          static_cast<int>(df_model),
		                                                          static_cast<int>(df_residual));
	} else {
		result.f_statistic = std::numeric_limits<double>::quiet_NaN();
		result.f_statistic_pvalue = std::numeric_limits<double>::quiet_NaN();
	}

	// Information criteria (AIC, AICc, BIC, log-likelihood)
	if (n > 0 && n_nonzero > 0 && ss_res >= 0.0) {
		double n_dbl = static_cast<double>(n);
		double k = static_cast<double>(n_nonzero);

		// Log-likelihood under normal errors
		double sigma_sq_mle = ss_res / n_dbl;
		if (sigma_sq_mle > 1e-300) {
			const double log_2pi = 1.8378770664093453;
			result.log_likelihood = -0.5 * n_dbl * (log_2pi + std::log(sigma_sq_mle) + 1.0);

			// AIC = n*log(RSS/n) + 2*k
			result.aic = n_dbl * std::log(sigma_sq_mle) + 2.0 * k;

			// BIC = n*log(RSS/n) + k*log(n)
			result.bic = n_dbl * std::log(sigma_sq_mle) + k * std::log(n_dbl);

			// AICc = AIC + 2*k*(k+1)/(n-k-1)
			if (n > n_nonzero + 1) {
				double correction = 2.0 * k * (k + 1.0) / (n_dbl - k - 1.0);
				result.aicc = result.aic + correction;
			} else {
				result.aicc = std::numeric_limits<double>::quiet_NaN();
			}
		}
	}
}

} // namespace solvers
} // namespace libanostat
