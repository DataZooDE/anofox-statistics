#pragma once

#include "duckdb.hpp"
#include <Eigen/Dense>
#include <limits>
#include <cmath>

namespace duckdb {
namespace anofox_statistics {

/**
 * Result structure for rank-deficient OLS regression
 *
 * When rank < n_params, some coefficients will be NaN (aliased/constant features)
 */
struct RankDeficientOlsResult {
	// Coefficients (size = n_params, NaN for aliased columns)
	Eigen::VectorXd coefficients;

	// Track which columns are aliased/constant
	std::vector<bool> is_aliased;

	// Numerical rank (rank <= n_params)
	idx_t rank;

	// Total number of parameters
	idx_t n_params;

	// Sample size
	idx_t n_obs;

	// Tolerance used for rank determination
	double tolerance_used;

	// Fit quality metrics
	double r_squared;
	double adj_r_squared;
	double rmse;
	double mse; // Mean squared error (for standard errors)

	// Residuals (for diagnostics)
	Eigen::VectorXd residuals;

	// Standard errors (size = n_params, NaN for aliased columns)
	// Only computed if requested
	Eigen::VectorXd std_errors;
	bool has_std_errors = false;

	// QR decomposition info (for advanced use)
	Eigen::VectorXi permutation_indices; // Original column index for each pivoted position
};

/**
 * Rank-Deficient OLS Regression Solver
 *
 * Uses Eigen's ColPivHouseholderQR to handle rank-deficient design matrices.
 * Similar to R's lm() behavior:
 * - Constant features return NaN coefficients
 * - Aliased features (perfect collinearity) return NaN coefficients
 * - Non-aliased features compute correctly
 *
 * Example:
 *   Eigen::VectorXd y(5);
 *   Eigen::MatrixXd X(5, 2);
 *   // ... fill y and X, where X col 2 is constant ...
 *   auto result = RankDeficientOls::Fit(y, X);
 *   // result.coefficients[1] will be NaN
 *   // result.is_aliased[1] will be true
 */
class RankDeficientOls {
public:
	/**
	 * Fit OLS regression with automatic rank-deficiency handling
	 *
	 * @param y Response vector (size n)
	 * @param X Design matrix (size n × p)
	 * @param tolerance Threshold for rank determination.
	 *                  If < 0, uses Eigen's default: epsilon * max(n, p)
	 * @return RankDeficientOlsResult with coefficients (NaN for aliased)
	 */
	static RankDeficientOlsResult Fit(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, double tolerance = -1.0);

	/**
	 * Fit OLS regression with standard errors for inference
	 *
	 * Computes standard errors for non-aliased coefficients.
	 * Aliased coefficients get NaN for standard errors.
	 *
	 * @param y Response vector (size n)
	 * @param X Design matrix (size n × p)
	 * @param tolerance Threshold for rank determination
	 * @return RankDeficientOlsResult with coefficients and std_errors
	 */
	static RankDeficientOlsResult FitWithStdErrors(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
	                                               double tolerance = -1.0);

	/**
	 * Quick check for constant columns (optimization)
	 *
	 * Detects columns with zero variance before attempting QR.
	 * Can be used as a fast pre-check.
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
	 * @param tolerance Threshold for rank determination
	 * @return true if rank(X) == ncol(X)
	 */
	static bool IsFullRank(const Eigen::MatrixXd &X, double tolerance = -1.0);

private:
	/**
	 * Compute fit quality statistics (R², adjusted R², RMSE, MSE)
	 */
	static void ComputeStatistics(const Eigen::VectorXd &y, const Eigen::VectorXd &y_pred,
	                              const Eigen::VectorXd &residuals, idx_t rank, idx_t n,
	                              RankDeficientOlsResult &result);

	/**
	 * Compute standard errors using MSE and (X'X)^-1
	 *
	 * For rank-deficient case, only computes SE for non-aliased coefficients.
	 */
	static void ComputeStandardErrors(const Eigen::MatrixXd &X, const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> &qr,
	                                  double mse, idx_t rank, RankDeficientOlsResult &result);
};

// ============================================================================
// Implementation
// ============================================================================

inline RankDeficientOlsResult RankDeficientOls::Fit(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
                                                    double tolerance) {
	const idx_t n = X.rows();
	const idx_t p = X.cols();

	RankDeficientOlsResult result;
	result.n_obs = n;
	result.n_params = p;
	result.tolerance_used = tolerance;

	// Initialize output vectors
	result.coefficients =
	    Eigen::VectorXd::Constant(static_cast<Eigen::Index>(p), std::numeric_limits<double>::quiet_NaN());
	result.is_aliased.resize(p, true); // Assume aliased until proven otherwise
	result.permutation_indices.resize(static_cast<Eigen::Index>(p));

	// Step 1: Perform QR decomposition with column pivoting
	// This automatically detects ALL dependencies including constant columns
	Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X);

	// Step 2: Set tolerance if specified
	if (tolerance > 0.0) {
		qr.setThreshold(tolerance);
	}
	// else: use Eigen's default (epsilon * max(n, p))

	result.tolerance_used = qr.threshold();

	// Step 3: Get rank and permutation from QR decomposition
	// This is the ONLY source of truth for rank (matches R's DQRLS)
	result.rank = qr.rank();
	const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> &P = qr.colsPermutation();

	// Store permutation indices for later use
	for (idx_t i = 0; i < p; i++) {
		result.permutation_indices[static_cast<Eigen::Index>(i)] = P.indices()[static_cast<Eigen::Index>(i)];
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
		coef_reduced = R_reduced.triangularView<Eigen::Upper>().solve(QtY.head(static_cast<Eigen::Index>(result.rank)));
	}

	// Step 5: Assign coefficients based on R's algorithm
	// The first 'rank' columns in PIVOTED order are non-aliased
	// Map coefficients from reduced system back to original column positions
	// Columns at pivoted positions [rank:p] remain NaN (aliased)
	for (idx_t i = 0; i < result.rank; i++) {
		auto i_idx = static_cast<Eigen::Index>(i);
		idx_t original_idx = P.indices()[i_idx];
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
	for (idx_t j = 0; j < p; j++) {
		auto j_idx = static_cast<Eigen::Index>(j);
		// Check both is_aliased flag and NaN status for safety
		if (!result.is_aliased[j] && !std::isnan(result.coefficients[j_idx])) {
			y_pred += result.coefficients[j_idx] * X.col(j_idx);
		}
	}

	// Step 7: Compute residuals
	result.residuals = y - y_pred;

	// Step 8: Compute fit statistics
	ComputeStatistics(y, y_pred, result.residuals, result.rank, n, result);

	return result;
}

inline RankDeficientOlsResult RankDeficientOls::FitWithStdErrors(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
                                                                 double tolerance) {
	const idx_t n = X.rows();
	const idx_t p = X.cols();

	// First, perform basic fit
	auto result = Fit(y, X, tolerance);

	// Initialize std_errors with NaN
	result.std_errors =
	    Eigen::VectorXd::Constant(static_cast<Eigen::Index>(p), std::numeric_limits<double>::quiet_NaN());
	result.has_std_errors = true;

	// Need to reconstruct QR for standard error computation
	Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X);
	if (tolerance > 0.0) {
		qr.setThreshold(tolerance);
	}

	// Compute standard errors for non-aliased coefficients
	ComputeStandardErrors(X, qr, result.mse, result.rank, result);

	return result;
}

inline std::vector<bool> RankDeficientOls::DetectConstantColumns(const Eigen::MatrixXd &X, double tol) {
	const idx_t n = X.rows();
	const idx_t p = X.cols();
	std::vector<bool> is_constant(p, false);

	for (idx_t j = 0; j < p; j++) {
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

inline bool RankDeficientOls::IsFullRank(const Eigen::MatrixXd &X, double tolerance) {
	Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X);
	if (tolerance > 0.0) {
		qr.setThreshold(tolerance);
	}

	return qr.rank() == X.cols();
}

inline void RankDeficientOls::ComputeStatistics(const Eigen::VectorXd &y, const Eigen::VectorXd &y_pred,
                                                const Eigen::VectorXd &residuals, idx_t rank, idx_t n,
                                                RankDeficientOlsResult &result) {
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

	// MSE (using rank for degrees of freedom)
	if (n > rank) {
		result.mse = ss_res / static_cast<double>(n - rank);
	} else {
		result.mse = 0.0;
	}

	// RMSE
	result.rmse = std::sqrt(result.mse);
}

inline void RankDeficientOls::ComputeStandardErrors(const Eigen::MatrixXd &X,
                                                    const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> &qr, double mse,
                                                    idx_t rank, RankDeficientOlsResult &result) {
	const idx_t p = X.cols();

	// Get permutation
	const auto &P = qr.colsPermutation();

	// For rank-deficient case, we need (X'X)^-1 for the rank-r subspace
	// This is more complex - we need to work with the pivoted system

	try {
		// Extract the rank-r subsystem
		// Create X_reduced with only non-aliased columns (in pivoted order)
		Eigen::MatrixXd X_reduced(X.rows(), static_cast<Eigen::Index>(rank));
		for (idx_t i = 0; i < rank; i++) {
			auto i_idx = static_cast<Eigen::Index>(i);
			idx_t original_idx = P.indices()[i_idx];
			auto orig_idx = static_cast<Eigen::Index>(original_idx);
			X_reduced.col(i_idx) = X.col(orig_idx);
		}

		// Compute (X_reduced' X_reduced)^-1
		Eigen::MatrixXd XtX = X_reduced.transpose() * X_reduced;
		Eigen::MatrixXd XtX_inv = XtX.inverse();

		// Standard errors: SE_j = sqrt(MSE * (X'X)^-1_jj)
		for (idx_t i = 0; i < rank; i++) {
			auto i_idx = static_cast<Eigen::Index>(i);
			idx_t original_idx = P.indices()[i_idx];
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

} // namespace anofox_statistics
} // namespace duckdb
