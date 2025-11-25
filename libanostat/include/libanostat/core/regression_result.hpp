#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cstdint>
#include <limits>

namespace libanostat {
namespace core {

/**
 * Result of a regression fit operation
 *
 * This structure contains all outputs from a regression algorithm,
 * including coefficients, residuals, fit statistics, and optional
 * standard errors for inference.
 *
 * Design notes:
 * - Uses size_t instead of DuckDB's idx_t for library independence
 * - All Eigen types for vectors/matrices
 * - Optional fields (like std_errors) indicated by has_* flags
 * - Supports rank-deficient regression with aliasing information
 */
struct RegressionResult {
	// ========================================================================
	// Core regression outputs
	// ========================================================================

	/// Estimated regression coefficients (length = n_params)
	/// For rank-deficient case, aliased coefficients are set to NaN
	/// NOTE: Does NOT include intercept - see intercept field below
	Eigen::VectorXd coefficients;

	/// Intercept term (if fitted with intercept)
	/// Only valid if has_intercept is true
	double intercept = 0.0;

	/// Flag indicating if intercept was fitted
	bool has_intercept = false;

	/// Residuals: y - X*beta (length = n_obs)
	Eigen::VectorXd residuals;

	// ========================================================================
	// Rank and aliasing information (for rank-deficient regression)
	// ========================================================================

	/// Rank of the design matrix (0 < rank <= n_params)
	size_t rank;

	/// Number of parameters (columns in design matrix)
	size_t n_params;

	/// Number of observations (rows in design matrix)
	size_t n_obs;

	/// Indicator for aliased coefficients (length = n_params)
	/// True if coefficient is aliased (set to NaN), false if estimated
	std::vector<bool> is_aliased;

	/// Column permutation indices from QR decomposition (length = n_params)
	/// Maps original column positions to pivoted positions
	std::vector<size_t> permutation_indices;

	/// Tolerance used for rank determination in QR decomposition
	double tolerance_used = -1.0;

	// ========================================================================
	// Fit quality statistics
	// ========================================================================

	/// Coefficient of determination: 1 - SSE/SST
	double r_squared = std::numeric_limits<double>::quiet_NaN();

	/// Adjusted R²: 1 - (1-R²)*(n-1)/(n-p-1)
	double adj_r_squared = std::numeric_limits<double>::quiet_NaN();

	/// Root mean squared error: sqrt(MSE)
	double rmse = std::numeric_limits<double>::quiet_NaN();

	/// Mean squared error: SSE / df_residual
	double mse = std::numeric_limits<double>::quiet_NaN();

	/// Residual standard error: sqrt(MSE) = sqrt(RSS / df_residual)
	/// Same as RMSE in theory, but computed for consistency with R's lm()
	double residual_standard_error = std::numeric_limits<double>::quiet_NaN();

	/// F-statistic for overall model significance
	/// F = (TSS - RSS) / df_model / (RSS / df_residual)
	/// Tests H0: all coefficients are zero
	double f_statistic = std::numeric_limits<double>::quiet_NaN();

	/// p-value for F-statistic
	double f_statistic_pvalue = std::numeric_limits<double>::quiet_NaN();

	// ========================================================================
	// Model selection criteria
	// ========================================================================

	/// Akaike Information Criterion
	/// AIC = n*log(RSS/n) + 2*k where k = rank
	double aic = std::numeric_limits<double>::quiet_NaN();

	/// Corrected AIC for small samples
	/// AICc = AIC + 2*k*(k+1)/(n-k-1)
	double aicc = std::numeric_limits<double>::quiet_NaN();

	/// Bayesian Information Criterion
	/// BIC = n*log(RSS/n) + k*log(n)
	double bic = std::numeric_limits<double>::quiet_NaN();

	/// Log-likelihood (under normal errors assumption)
	/// log L = -n/2 * (log(2π) + log(RSS/n) + 1)
	double log_likelihood = std::numeric_limits<double>::quiet_NaN();

	// ========================================================================
	// Optional: Statistical inference outputs
	// ========================================================================

	/// Standard errors of coefficients (length = n_params)
	/// Only computed if requested; aliased coefficients have NaN std errors
	/// NOTE: Does NOT include intercept std error - see intercept_std_error field below
	Eigen::VectorXd std_errors;

	/// Standard error of intercept term (if fitted with intercept and std errors computed)
	/// Only valid if has_intercept && has_std_errors
	double intercept_std_error = std::numeric_limits<double>::quiet_NaN();

	/// t-statistics for coefficients: coef / std_error (length = n_params)
	/// Aliased coefficients have NaN t-statistics
	/// NOTE: Does NOT include intercept t-statistic
	Eigen::VectorXd t_statistics;

	/// t-statistic for intercept term
	double intercept_t_statistic = std::numeric_limits<double>::quiet_NaN();

	/// p-values for coefficients (two-tailed test, H0: coef = 0) (length = n_params)
	/// Aliased coefficients have NaN p-values
	/// NOTE: Does NOT include intercept p-value
	Eigen::VectorXd p_values;

	/// p-value for intercept term
	double intercept_p_value = std::numeric_limits<double>::quiet_NaN();

	/// Lower bounds of confidence intervals for coefficients (length = n_params)
	/// Aliased coefficients have NaN bounds
	/// NOTE: Does NOT include intercept CI
	Eigen::VectorXd ci_lower;

	/// Lower bound of confidence interval for intercept
	double intercept_ci_lower = std::numeric_limits<double>::quiet_NaN();

	/// Upper bounds of confidence intervals for coefficients (length = n_params)
	/// Aliased coefficients have NaN bounds
	/// NOTE: Does NOT include intercept CI
	Eigen::VectorXd ci_upper;

	/// Upper bound of confidence interval for intercept
	double intercept_ci_upper = std::numeric_limits<double>::quiet_NaN();

	/// Confidence level used for intervals (e.g., 0.95 for 95% CI)
	double confidence_level = 0.95;

	/// Flag indicating if standard errors have been computed
	bool has_std_errors = false;

	// ========================================================================
	// Degrees of freedom (for inference)
	// ========================================================================

	/// Degrees of freedom for model: equal to rank (includes intercept if present)
	size_t df_model() const {
		return rank;
	}

	/// Degrees of freedom for residuals: n - rank
	size_t df_residual() const {
		if (n_obs <= rank) return 0;
		return n_obs - rank;
	}

	// ========================================================================
	// Constructors
	// ========================================================================

	/// Default constructor
	RegressionResult()
		: rank(0), n_params(0), n_obs(0) {}

	/// Convenience constructor with dimensions
	RegressionResult(size_t n_obs_, size_t n_params_, size_t rank_)
		: rank(rank_), n_params(n_params_), n_obs(n_obs_) {
		coefficients = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(n_params_),
		                                         std::numeric_limits<double>::quiet_NaN());
		residuals = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n_obs_));
		is_aliased.resize(n_params_, true);
		permutation_indices.resize(n_params_);
	}

	// ========================================================================
	// Utility methods
	// ========================================================================

	/// Check if result is valid (has finite coefficients for non-aliased params)
	bool is_valid() const {
		if (rank == 0 || n_params == 0 || n_obs == 0) return false;

		for (size_t i = 0; i < n_params; i++) {
			if (!is_aliased[i] && !std::isfinite(coefficients[static_cast<Eigen::Index>(i)])) {
				return false;
			}
		}
		return true;
	}

	/// Get number of non-aliased (estimated) parameters
	size_t n_estimated_params() const {
		size_t count = 0;
		for (bool aliased : is_aliased) {
			if (!aliased) count++;
		}
		return count;
	}
};

} // namespace core
} // namespace libanostat
