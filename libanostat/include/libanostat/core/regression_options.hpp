#pragma once

#include <cstdint>
#include <string>
#include <stdexcept>

namespace libanostat {
namespace core {

/**
 * Configuration options for regression algorithms
 *
 * This structure provides a unified interface for configuring all
 * regression algorithms in libanostat. All options have sensible
 * defaults and can be overridden as needed.
 *
 * Design notes:
 * - No DuckDB dependencies (uses std::string instead of duckdb::string)
 * - All defaults specified in-class for clarity
 * - Validation method to check for invalid combinations
 * - Extensible for future options without breaking existing code
 */
struct RegressionOptions {
	// ========================================================================
	// Common regression options
	// ========================================================================

	/// Include intercept term in regression
	/// Default: true
	bool intercept = true;

	/// Return extended metadata (std errors, df, training stats)
	/// Default: false (minimal output)
	bool full_output = false;

	// ========================================================================
	// Regularization parameters
	// ========================================================================

	/// L2 penalty strength (Ridge regression)
	/// - lambda = 0: No regularization (OLS)
	/// - lambda > 0: Ridge regularization strength
	/// Default: 0.0 (no regularization)
	double lambda = 0.0;

	/// Elastic Net mixing parameter
	/// - alpha = 0: Pure Ridge regression (L2 only)
	/// - alpha = 1: Pure Lasso regression (L1 only)
	/// - 0 < alpha < 1: Elastic Net (L1 + L2 mix)
	/// Default: 0.0 (pure Ridge)
	double alpha = 0.0;

	// ========================================================================
	// Recursive Least Squares (RLS) specific
	// ========================================================================

	/// Forgetting factor for RLS (weight recent observations more)
	/// - forgetting_factor = 1.0: All observations equally weighted
	/// - 0 < forgetting_factor < 1.0: Exponential weighting (recent data weighted more)
	/// Default: 1.0 (no forgetting)
	double forgetting_factor = 1.0;

	// ========================================================================
	// Window function parameters (future extensibility)
	// ========================================================================

	/// Rolling window size (0 = not set, use all data)
	/// Default: 0 (use all data)
	size_t window_size = 0;

	/// Expanding minimum periods (0 = not set, require full window)
	/// Default: 0 (not set)
	size_t min_periods = 0;

	// ========================================================================
	// Statistical inference parameters
	// ========================================================================

	/// Confidence level for confidence intervals
	/// Default: 0.95 (95% confidence intervals)
	double confidence_level = 0.95;

	/// Use robust (heteroskedasticity-consistent) standard errors
	/// Default: false (classical standard errors)
	bool robust_se = false;

	// ========================================================================
	// Computational parameters
	// ========================================================================

	/// Maximum iterations for iterative algorithms (Elastic Net, etc.)
	/// Default: 1000
	size_t max_iterations = 1000;

	/// Convergence tolerance for iterative algorithms
	/// Default: 1e-6
	double tolerance = 1e-6;

	/// QR decomposition rank tolerance (-1 = auto, use Eigen default)
	/// Default: -1.0 (auto)
	double qr_tolerance = -1.0;

	/// Solver algorithm to use
	/// Options: "qr" (default), "svd", "cholesky"
	/// Default: "qr"
	std::string solver = "qr";

	// ========================================================================
	// Grouped regression (future extensibility)
	// ========================================================================

	/// Compute separate regression for each group
	/// Default: false
	bool compute_per_group = false;

	// ========================================================================
	// Constructors
	// ========================================================================

	/// Default constructor with all default values
	RegressionOptions() = default;

	/// Convenience constructor for common OLS options
	static RegressionOptions OLS(bool intercept_ = true) {
		RegressionOptions opts;
		opts.intercept = intercept_;
		opts.lambda = 0.0;
		opts.alpha = 0.0;
		return opts;
	}

	/// Convenience constructor for Ridge regression
	static RegressionOptions Ridge(double lambda_, bool intercept_ = true) {
		RegressionOptions opts;
		opts.intercept = intercept_;
		opts.lambda = lambda_;
		opts.alpha = 0.0;
		return opts;
	}

	/// Convenience constructor for Lasso regression
	static RegressionOptions Lasso(double lambda_, bool intercept_ = true) {
		RegressionOptions opts;
		opts.intercept = intercept_;
		opts.lambda = lambda_;
		opts.alpha = 1.0;
		return opts;
	}

	/// Convenience constructor for Elastic Net regression
	static RegressionOptions ElasticNet(double lambda_, double alpha_, bool intercept_ = true) {
		RegressionOptions opts;
		opts.intercept = intercept_;
		opts.lambda = lambda_;
		opts.alpha = alpha_;
		return opts;
	}

	/// Convenience constructor for RLS
	static RegressionOptions RLS(double forgetting_factor_ = 1.0, bool intercept_ = true) {
		RegressionOptions opts;
		opts.intercept = intercept_;
		opts.forgetting_factor = forgetting_factor_;
		return opts;
	}

	// ========================================================================
	// Validation
	// ========================================================================

	/**
	 * Validate option combinations
	 * Checks for incompatible options and invalid values
	 *
	 * @throws std::invalid_argument if validation fails
	 */
	void Validate() const {
		// Lambda must be non-negative
		if (lambda < 0.0) {
			throw std::invalid_argument("lambda must be non-negative (got " + std::to_string(lambda) + ")");
		}

		// Alpha must be in [0, 1]
		if (alpha < 0.0 || alpha > 1.0) {
			throw std::invalid_argument("alpha must be in [0, 1] (got " + std::to_string(alpha) + ")");
		}

		// Forgetting factor must be in (0, 1]
		if (forgetting_factor <= 0.0 || forgetting_factor > 1.0) {
			throw std::invalid_argument("forgetting_factor must be in (0, 1] (got " +
			                            std::to_string(forgetting_factor) + ")");
		}

		// Confidence level must be in (0, 1)
		if (confidence_level <= 0.0 || confidence_level >= 1.0) {
			throw std::invalid_argument("confidence_level must be in (0, 1) (got " +
			                            std::to_string(confidence_level) + ")");
		}

		// Tolerance must be positive
		if (tolerance <= 0.0) {
			throw std::invalid_argument("tolerance must be positive (got " + std::to_string(tolerance) + ")");
		}

		// Max iterations must be positive
		if (max_iterations == 0) {
			throw std::invalid_argument("max_iterations must be positive");
		}

		// Solver must be valid
		if (solver != "qr" && solver != "svd" && solver != "cholesky") {
			throw std::invalid_argument("solver must be 'qr', 'svd', or 'cholesky' (got '" + solver + "')");
		}

		// Check incompatible combinations
		if (alpha > 0.0 && lambda == 0.0) {
			throw std::invalid_argument("alpha > 0 requires lambda > 0 (Elastic Net needs regularization)");
		}
	}
};

} // namespace core
} // namespace libanostat
