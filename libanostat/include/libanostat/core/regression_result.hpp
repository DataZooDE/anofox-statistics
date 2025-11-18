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
	Eigen::VectorXd coefficients;

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

	// ========================================================================
	// Optional: Statistical inference outputs
	// ========================================================================

	/// Standard errors of coefficients (length = n_params)
	/// Only computed if requested; aliased coefficients have NaN std errors
	Eigen::VectorXd std_errors;

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
