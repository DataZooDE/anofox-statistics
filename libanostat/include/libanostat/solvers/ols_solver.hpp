#pragma once

#include "libanostat/core/regression_result.hpp"
#include "libanostat/core/regression_options.hpp"
#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <cmath>

namespace libanostat {
namespace solvers {

/**
 * Ordinary Least Squares (OLS) Regression Solver
 *
 * Uses Eigen's ColPivHouseholderQR decomposition to handle rank-deficient
 * design matrices. Behavior matches R's lm() function:
 * - Constant features return NaN coefficients (aliased)
 * - Perfectly collinear features return NaN coefficients (aliased)
 * - Non-aliased features compute correctly
 *
 * Algorithm:
 * 1. QR decomposition with column pivoting: X*P = Q*R
 * 2. Determine numerical rank from R diagonal
 * 3. Solve rank-r triangular system: R_r * beta_r = (Q^T * y)_r
 * 4. Map reduced coefficients back to original column order via permutation
 * 5. Compute residuals, predictions, and fit statistics
 *
 * Design notes:
 * - Header-only for performance
 * - No DuckDB dependencies
 * - Uses libanostat::core::RegressionResult instead of custom structs
 * - Stateless design (all methods are static)
 */
class OLSSolver {
public:
	/**
	 * Fit OLS regression with automatic rank-deficiency handling
	 *
	 * @param y Response vector (length n)
	 * @param X Design matrix (n × p)
	 * @param options Regression options (intercept, tolerance, etc.)
	 * @return RegressionResult with coefficients (NaN for aliased)
	 */
	static core::RegressionResult Fit(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
	                                   const core::RegressionOptions &options = core::RegressionOptions::OLS());

	/**
	 * Fit OLS regression with standard errors for statistical inference
	 *
	 * Computes standard errors for non-aliased coefficients.
	 * Aliased coefficients get NaN for standard errors.
	 *
	 * @param y Response vector (length n)
	 * @param X Design matrix (n × p)
	 * @param options Regression options
	 * @return RegressionResult with coefficients and std_errors
	 */
	static core::RegressionResult FitWithStdErrors(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
	                                                const core::RegressionOptions &options =
	                                                    core::RegressionOptions::OLS());

	/**
	 * Quick check for constant columns (optimization)
	 *
	 * Detects columns with zero variance before attempting QR.
	 * Can be used as a fast pre-check to avoid expensive decomposition.
	 *
	 * @param X Design matrix
	 * @param tol Tolerance for considering variance as zero
	 * @return Vector of bools, true if column is constant
	 */
	static std::vector<bool> DetectConstantColumns(const Eigen::MatrixXd &X, double tol = 1e-10);

	/**
	 * Check if matrix is full rank
	 *
	 * @param X Design matrix
	 * @param tolerance Threshold for rank determination (-1 = auto)
	 * @return true if rank(X) == ncol(X)
	 */
	static bool IsFullRank(const Eigen::MatrixXd &X, double tolerance = -1.0);

private:
	/**
	 * Compute fit quality statistics (R², adjusted R², RMSE, MSE)
	 */
	static void ComputeStatistics(const Eigen::VectorXd &y, const Eigen::VectorXd &y_pred,
	                              const Eigen::VectorXd &residuals, size_t rank, size_t n,
	                              core::RegressionResult &result);

	/**
	 * Compute standard errors using MSE and (X'X)^-1
	 *
	 * For rank-deficient case, only computes SE for non-aliased coefficients.
	 */
	static void ComputeStandardErrors(const Eigen::MatrixXd &X,
	                                  const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> &qr, double mse, size_t rank,
	                                  core::RegressionResult &result);
};

// ============================================================================
// Implementation (header-only for performance)
// ============================================================================

inline core::RegressionResult OLSSolver::Fit(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
                                              const core::RegressionOptions &options) {
	const size_t n = static_cast<size_t>(X.rows());
	const size_t p_user = static_cast<size_t>(X.cols());

	// Validate options
	options.Validate();

	// Auto-add intercept column if requested (user-friendly API)
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
	result.tolerance_used = options.qr_tolerance;

	// Step 1: Perform QR decomposition with column pivoting
	// This automatically detects ALL dependencies including constant columns
	Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X_work);

	// Step 2: Set tolerance if specified
	if (options.qr_tolerance > 0.0) {
		qr.setThreshold(options.qr_tolerance);
	}
	// else: use Eigen's default (epsilon * max(n, p))

	result.tolerance_used = qr.threshold();

	// Step 3: Get rank and permutation from QR decomposition
	// This is the ONLY source of truth for rank (matches R's DQRLS)
	result.rank = static_cast<size_t>(qr.rank());
	const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> &P = qr.colsPermutation();

	// Store permutation indices for later use
	for (size_t i = 0; i < p; i++) {
		result.permutation_indices[i] = static_cast<size_t>(P.indices()[static_cast<Eigen::Index>(i)]);
	}

	// Step 4: Solve least squares for rank-r subsystem only
	// CRITICAL: For rank-deficient matrices, qr.solve() gives unreliable results
	// Instead, we explicitly solve R_r * beta_r = (Q^T * y)_r for the rank-r subsystem
	// This matches R's DQRLS algorithm exactly

	Eigen::VectorXd coef_reduced;
	if (result.rank > 0) {
		// Extract Q^T * y
		Eigen::VectorXd QtY = qr.matrixQ().transpose() * y;

		// Extract upper-left rank×rank portion of R
		Eigen::MatrixXd R_reduced =
		    qr.matrixQR().topLeftCorner(static_cast<Eigen::Index>(result.rank), static_cast<Eigen::Index>(result.rank));

		// Solve the rank-r triangular system: R_r * coef_r = (Q^T y)_r
		coef_reduced =
		    R_reduced.triangularView<Eigen::Upper>().solve(QtY.head(static_cast<Eigen::Index>(result.rank)));
	}

	// Step 5: Assign coefficients based on R's algorithm
	// The first 'rank' columns in PIVOTED order are non-aliased
	// Map coefficients from reduced system back to original column positions
	// Columns at pivoted positions [rank:p] remain NaN (aliased)
	for (size_t i = 0; i < result.rank; i++) {
		auto i_idx = static_cast<Eigen::Index>(i);
		size_t original_idx = static_cast<size_t>(P.indices()[i_idx]);
		auto orig_idx = static_cast<Eigen::Index>(original_idx);

		// Assign coefficient from reduced system
		result.coefficients[orig_idx] = coef_reduced[i_idx];
		result.is_aliased[original_idx] = false;
	}

	// Coefficients at pivoted positions [rank:p] remain NaN and is_aliased=true
	// No special handling needed - they're already initialized as aliased

	// Step 6: Compute predictions using only non-aliased features
	// Skip NaN coefficients (aliased columns) to avoid numerical errors
	Eigen::VectorXd y_pred = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n));
	for (size_t j = 0; j < p; j++) {
		auto j_idx = static_cast<Eigen::Index>(j);
		// Check both is_aliased flag and NaN status for safety
		if (!result.is_aliased[j] && std::isfinite(result.coefficients[j_idx])) {
			y_pred += result.coefficients[j_idx] * X_work.col(j_idx);
		}
	}

	// Step 7: Compute residuals
	result.residuals = y - y_pred;

	// Step 8: Compute fit statistics
	ComputeStatistics(y, y_pred, result.residuals, result.rank, n, result);

	return result;
}

inline core::RegressionResult OLSSolver::FitWithStdErrors(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
                                                           const core::RegressionOptions &options) {
	const size_t n = static_cast<size_t>(X.rows());
	const size_t p_user = static_cast<size_t>(X.cols());

	// Auto-add intercept column if requested (user-friendly API)
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

	// First, perform basic fit
	auto result = Fit(y, X, options);

	// Initialize std_errors with NaN
	result.std_errors =
	    Eigen::VectorXd::Constant(static_cast<Eigen::Index>(p), std::numeric_limits<double>::quiet_NaN());
	result.has_std_errors = true;

	// Need to reconstruct QR for standard error computation
	Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X_work);
	if (options.qr_tolerance > 0.0) {
		qr.setThreshold(options.qr_tolerance);
	}

	// Compute standard errors for non-aliased coefficients
	ComputeStandardErrors(X_work, qr, result.mse, result.rank, result);

	return result;
}

inline std::vector<bool> OLSSolver::DetectConstantColumns(const Eigen::MatrixXd &X, double tol) {
	const size_t n = static_cast<size_t>(X.rows());
	const size_t p = static_cast<size_t>(X.cols());
	std::vector<bool> is_constant(p, false);

	for (size_t j = 0; j < p; j++) {
		const auto &col = X.col(static_cast<Eigen::Index>(j));

		// Compute variance
		double mean = col.mean();
		double variance = (col.array() - mean).square().sum() / static_cast<double>(n - 1);

		if (variance < tol) {
			is_constant[j] = true;
		}
	}

	return is_constant;
}

inline bool OLSSolver::IsFullRank(const Eigen::MatrixXd &X, double tolerance) {
	Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X);
	if (tolerance > 0.0) {
		qr.setThreshold(tolerance);
	}

	return qr.rank() == X.cols();
}

inline void OLSSolver::ComputeStatistics(const Eigen::VectorXd &y, const Eigen::VectorXd &y_pred,
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
	// Numerical errors in rank-deficient case can produce invalid values
	if (result.r_squared < 0.0) {
		result.r_squared = 0.0;
	} else if (result.r_squared > 1.0) {
		result.r_squared = 1.0;
	}

	// Adjusted R-squared
	// Use rank instead of n_params for degrees of freedom
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

	// MSE (using rank for degrees of freedom)
	if (n > rank) {
		result.mse = ss_res / static_cast<double>(n - rank);
	} else {
		result.mse = 0.0;
	}

	// RMSE
	result.rmse = std::sqrt(result.mse);
}

inline void OLSSolver::ComputeStandardErrors(const Eigen::MatrixXd &X,
                                             const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> &qr, double mse,
                                             size_t rank, core::RegressionResult &result) {
	// Get permutation
	const auto &P = qr.colsPermutation();

	// For rank-deficient case, we need (X'X)^-1 for the rank-r subspace
	// This is more complex - we need to work with the pivoted system

	try {
		// Extract the rank-r subsystem
		// Create X_reduced with only non-aliased columns (in pivoted order)
		Eigen::MatrixXd X_reduced(X.rows(), static_cast<Eigen::Index>(rank));
		for (size_t i = 0; i < rank; i++) {
			auto i_idx = static_cast<Eigen::Index>(i);
			size_t original_idx = static_cast<size_t>(P.indices()[i_idx]);
			auto orig_idx = static_cast<Eigen::Index>(original_idx);
			X_reduced.col(i_idx) = X.col(orig_idx);
		}

		// Compute (X_reduced' X_reduced)^-1
		Eigen::MatrixXd XtX = X_reduced.transpose() * X_reduced;
		Eigen::MatrixXd XtX_inv = XtX.inverse();

		// Standard errors: SE_j = sqrt(MSE * (X'X)^-1_jj)
		for (size_t i = 0; i < rank; i++) {
			auto i_idx = static_cast<Eigen::Index>(i);
			size_t original_idx = static_cast<size_t>(P.indices()[i_idx]);
			auto orig_idx = static_cast<Eigen::Index>(original_idx);
			double se = std::sqrt(mse * XtX_inv(i_idx, i_idx));
			result.std_errors[orig_idx] = se;
		}

	} catch (...) {
		// If inversion fails, leave std_errors as NaN
		// This shouldn't happen for rank-r subspace, but be safe
	}

	// Aliased columns remain NaN (already initialized)
}

} // namespace solvers
} // namespace libanostat
