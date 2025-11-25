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
	                                  core::RegressionResult &result, size_t coef_offset,
	                                  bool compute_intercept_se, size_t n_obs, const Eigen::VectorXd &x_means);
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

	// R-COMPATIBLE APPROACH: Handle intercept by centering data, NOT by augmenting design matrix
	// This ensures the intercept is NEVER marked as aliased (matches R's lm() behavior)
	Eigen::MatrixXd X_work;
	Eigen::VectorXd y_work;
	Eigen::VectorXd x_means;
	double y_mean = 0.0;
	size_t p;  // Number of columns in X_work (features only, no intercept column)

	if (options.intercept) {
		// Center data instead of adding intercept column
		p = p_user;
		y_mean = y.mean();
		x_means = X.colwise().mean();

		// Center y and X
		y_work = y.array() - y_mean;
		X_work.resize(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(p));
		for (size_t j = 0; j < p_user; j++) {
			X_work.col(static_cast<Eigen::Index>(j)) = X.col(static_cast<Eigen::Index>(j)).array() - x_means(static_cast<Eigen::Index>(j));
		}
	} else {
		// No intercept: use X and y as-is
		p = p_user;
		X_work = X;
		y_work = y;
		x_means = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(p_user));
	}

	// Initialize result for features + intercept (if requested)
	const size_t result_n_params = options.intercept ? (p_user + 1) : p_user;
	core::RegressionResult result(n, result_n_params, 0);
	result.tolerance_used = options.qr_tolerance;

	// Step 1: Perform QR decomposition with column pivoting on FEATURES ONLY
	// The intercept is NOT part of this decomposition (R-compatible)
	Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X_work);

	// Step 2: Set tolerance if specified
	if (options.qr_tolerance > 0.0) {
		qr.setThreshold(options.qr_tolerance);
	}

	result.tolerance_used = qr.threshold();

	// Step 3: Get rank and permutation from QR decomposition
	// Feature rank from QR (features only, not including intercept)
	size_t feature_rank = static_cast<size_t>(qr.rank());

	// Total model rank includes intercept if present
	result.rank = feature_rank;
	if (options.intercept) {
		// Intercept is never aliased, so total rank = feature_rank + 1
		result.rank += 1;
	}
	const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> &P = qr.colsPermutation();

	// Store permutation indices (for features, offset by +1 if intercept present)
	for (size_t i = 0; i < p; i++) {
		size_t perm_idx = static_cast<size_t>(P.indices()[static_cast<Eigen::Index>(i)]);
		// If intercept, feature permutations are at positions 1..p (position 0 is intercept)
		result.permutation_indices[options.intercept ? (i + 1) : i] = options.intercept ? (perm_idx + 1) : perm_idx;
	}
	if (options.intercept) {
		result.permutation_indices[0] = 0;  // Intercept is always at position 0, never permuted
	}

	// Step 4: Solve least squares for rank-r subsystem only (on centered data)
	Eigen::VectorXd coef_reduced;
	if (feature_rank > 0) {
		Eigen::VectorXd QtY = qr.matrixQ().transpose() * y_work;
		Eigen::MatrixXd R_reduced =
		    qr.matrixQR().topLeftCorner(static_cast<Eigen::Index>(feature_rank), static_cast<Eigen::Index>(feature_rank));
		coef_reduced =
		    R_reduced.triangularView<Eigen::Upper>().solve(QtY.head(static_cast<Eigen::Index>(feature_rank)));
	}

	// Step 5: Assign feature coefficients (at positions 1..p_user if intercept, else 0..p_user-1)
	const size_t coef_offset = options.intercept ? 1 : 0;
	for (size_t i = 0; i < feature_rank; i++) {
		auto i_idx = static_cast<Eigen::Index>(i);
		size_t original_idx = static_cast<size_t>(P.indices()[i_idx]);
		auto orig_idx = static_cast<Eigen::Index>(original_idx);

		// Assign coefficient from reduced system (offset if intercept present)
		result.coefficients[orig_idx + coef_offset] = coef_reduced[i_idx];
		result.is_aliased[original_idx + coef_offset] = false;
	}

	// Step 6: Compute intercept separately (NEVER aliased, matches R)
	if (options.intercept) {
		double intercept = y_mean;
		// Subtract contribution of non-aliased features: intercept = mean(y) - sum(coef[j] * mean(x[j]))
		for (size_t j = 0; j < p_user; j++) {
			if (!result.is_aliased[j + 1]) {  // Feature coefficients are at positions 1..p_user
				intercept -= result.coefficients[j + 1] * x_means(static_cast<Eigen::Index>(j));
			}
		}
		result.coefficients[0] = intercept;
		result.is_aliased[0] = false;  // Intercept is NEVER aliased

		// NEW: Populate intercept fields in result
		result.intercept = intercept;
		result.has_intercept = true;
	}

	// Step 7: Compute predictions using intercept + non-aliased features
	Eigen::VectorXd y_pred = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n));
	if (options.intercept) {
		y_pred = Eigen::VectorXd::Constant(static_cast<Eigen::Index>(n), result.coefficients[0]);
	}
	for (size_t j = 0; j < p_user; j++) {
		const size_t coef_idx = options.intercept ? (j + 1) : j;
		auto coef_idx_eigen = static_cast<Eigen::Index>(coef_idx);
		auto j_idx = static_cast<Eigen::Index>(j);
		if (!result.is_aliased[coef_idx] && std::isfinite(result.coefficients[coef_idx_eigen])) {
			y_pred += result.coefficients[coef_idx_eigen] * X.col(j_idx);
		}
	}

	// Step 8: Compute residuals
	result.residuals = y - y_pred;

	// Step 9: Compute fit statistics
	ComputeStatistics(y, y_pred, result.residuals, result.rank, n, result);

	return result;
}

inline core::RegressionResult OLSSolver::FitWithStdErrors(const Eigen::VectorXd &y, const Eigen::MatrixXd &X,
                                                           const core::RegressionOptions &options) {
	const size_t n = static_cast<size_t>(X.rows());
	const size_t p_user = static_cast<size_t>(X.cols());

	// First, perform basic fit using R-compatible centered data approach
	auto result = Fit(y, X, options);

	// Prepare working matrices for standard error computation
	// Follow same centering approach as Fit() to ensure consistency
	Eigen::MatrixXd X_work;
	Eigen::VectorXd y_work;
	size_t p;

	if (options.intercept) {
		// Center data (same as Fit())
		p = p_user;
		double y_mean = y.mean();
		Eigen::VectorXd x_means = X.colwise().mean();

		y_work = y.array() - y_mean;
		X_work.resize(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(p));
		for (size_t j = 0; j < p_user; j++) {
			X_work.col(static_cast<Eigen::Index>(j)) =
				X.col(static_cast<Eigen::Index>(j)).array() - x_means(static_cast<Eigen::Index>(j));
		}
	} else {
		// Use X as-is (no intercept)
		p = p_user;
		X_work = X;
		y_work = y;
	}

	// Initialize std_errors: size = p + (intercept ? 1 : 0)
	size_t n_coeffs = p + (options.intercept ? 1 : 0);
	result.std_errors =
	    Eigen::VectorXd::Constant(static_cast<Eigen::Index>(n_coeffs), std::numeric_limits<double>::quiet_NaN());
	result.has_std_errors = true;

	// Perform QR decomposition on centered/raw feature matrix (no intercept column)
	Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X_work);
	if (options.qr_tolerance > 0.0) {
		qr.setThreshold(options.qr_tolerance);
	}

	// Get feature rank from QR (rank of features only, not including intercept)
	size_t feature_rank = static_cast<size_t>(qr.rank());

	// Compute standard errors for non-aliased coefficients (and intercept if present)
	size_t coef_offset = options.intercept ? 1 : 0;

	// Pass feature means for intercept SE computation if needed
	Eigen::VectorXd x_means = options.intercept ? X.colwise().mean() : Eigen::VectorXd();

	ComputeStandardErrors(X_work, qr, result.mse, feature_rank, result, coef_offset,
	                      options.intercept, n, x_means);

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
		// For numerical stability with near-perfect fits, use a minimum threshold
		// This ensures standard errors remain finite even with tiny residuals
		constexpr double min_mse = 1e-20;
		if (result.mse < min_mse) {
			result.mse = min_mse;
		}
	} else {
		// Saturated model: no degrees of freedom for error
		// Set MSE to NaN as standard errors are undefined
		result.mse = std::numeric_limits<double>::quiet_NaN();
	}

	// RMSE
	result.rmse = std::sqrt(result.mse);
}

inline void OLSSolver::ComputeStandardErrors(const Eigen::MatrixXd &X,
                                             const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> &qr, double mse,
                                             size_t rank, core::RegressionResult &result, size_t coef_offset,
                                             bool compute_intercept_se, size_t n_obs, const Eigen::VectorXd &x_means) {
	// Get permutation
	const auto &P = qr.colsPermutation();

	// For rank-deficient case, we need (X'X)^-1 for the rank-r subspace
	// This is more complex - we need to work with the pivoted system

	try {
		if (rank == 0) {
			// No valid features - only intercept if present
			if (compute_intercept_se && n_obs > 0) {
				result.std_errors[0] = std::sqrt(mse / static_cast<double>(n_obs));
			}
			return;
		}

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

		// Standard errors for features: SE_j = sqrt(MSE * (X'X)^-1_jj)
		for (size_t i = 0; i < rank; i++) {
			auto i_idx = static_cast<Eigen::Index>(i);
			size_t original_idx = static_cast<size_t>(P.indices()[i_idx]);
			auto orig_idx = static_cast<Eigen::Index>(original_idx + coef_offset);
			double se = std::sqrt(mse * XtX_inv(i_idx, i_idx));
			result.std_errors[orig_idx] = se;
		}

		// Compute intercept standard error if requested
		if (compute_intercept_se) {
			// SE(intercept) = sqrt(MSE * (1/n + x_mean' * (X'X)^-1 * x_mean))
			// Extract means for non-aliased features (in pivoted order to match X_reduced)
			Eigen::VectorXd x_means_reduced(static_cast<Eigen::Index>(rank));
			for (size_t i = 0; i < rank; i++) {
				auto i_idx = static_cast<Eigen::Index>(i);
				size_t original_idx = static_cast<size_t>(P.indices()[i_idx]);
				auto orig_idx = static_cast<Eigen::Index>(original_idx);
				x_means_reduced(i_idx) = x_means(orig_idx);
			}

			double variance_component = x_means_reduced.transpose() * XtX_inv * x_means_reduced;
			double intercept_se = std::sqrt(mse * (1.0 / static_cast<double>(n_obs) + variance_component));
			result.std_errors[0] = intercept_se;

			// NEW: Populate intercept_std_error field
			result.intercept_std_error = intercept_se;
		}

	} catch (...) {
		// If inversion fails, leave std_errors as NaN
		// This shouldn't happen for rank-r subspace, but be safe
	}

	// Aliased columns remain NaN (already initialized)
}

} // namespace solvers
} // namespace libanostat
