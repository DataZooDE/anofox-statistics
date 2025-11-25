#pragma once

#include "../solvers/ols_solver.hpp"
#include "../core/regression_result.hpp"
#include "../core/regression_options.hpp"
#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <cmath>

namespace libanostat {
namespace diagnostics {

/**
 * Variance Inflation Factor (VIF) Calculator
 *
 * VIF measures how much the variance of a regression coefficient is inflated
 * due to collinearity with other predictors.
 *
 * VIF_j = 1 / (1 - R²_j)
 *
 * where R²_j is from regressing X_j on all other predictors (with intercept).
 *
 * Interpretation:
 * - VIF < 5: Low collinearity
 * - 5 ≤ VIF < 10: Moderate collinearity
 * - VIF ≥ 10: High collinearity (problematic)
 * - VIF = ∞: Perfect collinearity
 * - VIF = NaN: Undefined (constant feature)
 *
 * Design notes:
 * - Header-only for performance
 * - Uses OLSSolver for auxiliary regressions
 * - Handles edge cases (constant columns, perfect collinearity)
 * - Stateless design (all methods are static)
 */
class VIFCalculator {
public:
	/**
	 * Result for a single variable's VIF
	 */
	struct VIFResult {
		/// Variance Inflation Factor
		double vif;

		/// R² from auxiliary regression
		double r_squared;

		/// Index of the variable
		size_t variable_index;

		/// Whether VIF is well-defined
		bool is_defined;

		/// Reason if VIF is undefined
		enum class Status {
			OK,                  ///< VIF computed successfully
			CONSTANT_FEATURE,    ///< Feature is constant (zero variance)
			PERFECT_COLLINEARITY ///< Perfect collinearity with other features
		} status;
	};

	/**
	 * Compute VIF for all variables in design matrix
	 *
	 * For each variable j, performs auxiliary regression:
	 *   X_j ~ X_{-j} + intercept
	 *
	 * Then computes: VIF_j = 1 / (1 - R²_j)
	 *
	 * @param X Design matrix (n × p), assumed to be the raw feature matrix
	 *          (NO intercept column should be included)
	 * @param options Regression options for auxiliary regressions
	 *                (typically with intercept=true, which is the standard VIF definition)
	 * @return Vector of VIF results for each variable
	 */
	static std::vector<VIFResult> ComputeVIF(const Eigen::MatrixXd &X,
	                                          const core::RegressionOptions &options =
	                                              core::RegressionOptions::OLS());

	/**
	 * Compute VIF for a single variable
	 *
	 * @param X Design matrix (n × p)
	 * @param variable_index Index of variable to compute VIF for (0-based)
	 * @param options Regression options
	 * @return VIF result for the specified variable
	 */
	static VIFResult ComputeSingleVIF(const Eigen::MatrixXd &X,
	                                   size_t variable_index,
	                                   const core::RegressionOptions &options =
	                                       core::RegressionOptions::OLS());

	/**
	 * Check if any variable has high VIF (≥ threshold)
	 *
	 * @param X Design matrix
	 * @param threshold VIF threshold (default: 10.0)
	 * @param options Regression options
	 * @return true if any variable has VIF ≥ threshold
	 */
	static bool HasHighCollinearity(const Eigen::MatrixXd &X,
	                                 double threshold = 10.0,
	                                 const core::RegressionOptions &options =
	                                     core::RegressionOptions::OLS());

private:
	/**
	 * Check if a column is constant (zero variance)
	 */
	static bool IsConstantColumn(const Eigen::VectorXd &column, double tol = 1e-10);
};

// ============================================================================
// Implementation (header-only for performance)
// ============================================================================

inline std::vector<VIFCalculator::VIFResult> VIFCalculator::ComputeVIF(
    const Eigen::MatrixXd &X,
    const core::RegressionOptions &options) {

	const size_t n = static_cast<size_t>(X.rows());
	const size_t p = static_cast<size_t>(X.cols());

	if (p < 2) {
		throw std::invalid_argument("VIF requires at least 2 variables");
	}

	if (n <= p) {
		throw std::invalid_argument("VIF requires more observations than variables");
	}

	std::vector<VIFResult> results;
	results.reserve(p);

	// Compute VIF for each variable
	for (size_t j = 0; j < p; j++) {
		results.push_back(ComputeSingleVIF(X, j, options));
	}

	return results;
}

inline VIFCalculator::VIFResult VIFCalculator::ComputeSingleVIF(
    const Eigen::MatrixXd &X,
    size_t variable_index,
    const core::RegressionOptions &options) {

	const size_t n = static_cast<size_t>(X.rows());
	const size_t p = static_cast<size_t>(X.cols());
	const auto j_idx = static_cast<Eigen::Index>(variable_index);

	VIFResult result;
	result.variable_index = variable_index;
	result.is_defined = false;
	result.status = VIFResult::Status::OK;

	if (variable_index >= p) {
		throw std::out_of_range("Variable index out of range");
	}

	// Check if column is constant
	if (IsConstantColumn(X.col(j_idx))) {
		result.vif = std::numeric_limits<double>::quiet_NaN();
		result.r_squared = std::numeric_limits<double>::quiet_NaN();
		result.is_defined = false;
		result.status = VIFResult::Status::CONSTANT_FEATURE;
		return result;
	}

	// Extract dependent variable (column j)
	Eigen::VectorXd y = X.col(j_idx);

	// Build design matrix with all columns except j
	Eigen::MatrixXd X_reduced(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(p - 1));
	size_t col_idx = 0;
	for (size_t k = 0; k < p; k++) {
		if (k != variable_index) {
			auto k_idx = static_cast<Eigen::Index>(k);
			auto col_idx_eigen = static_cast<Eigen::Index>(col_idx);
			X_reduced.col(col_idx_eigen) = X.col(k_idx);
			col_idx++;
		}
	}

	// Perform OLS regression: y = X_reduced * beta + intercept
	// The intercept is handled by OLSSolver's centering approach
	try {
		auto reg_result = solvers::OLSSolver::Fit(y, X_reduced, options);

		// Check for perfect collinearity
		// If R² is extremely close to 1 (within numerical precision), VIF is effectively infinite
		if (reg_result.r_squared > (1.0 - 1e-10)) {
			result.vif = std::numeric_limits<double>::infinity();
			result.r_squared = reg_result.r_squared;
			result.is_defined = false;
			result.status = VIFResult::Status::PERFECT_COLLINEARITY;
			return result;
		}

		// Compute VIF = 1 / (1 - R²)
		result.r_squared = reg_result.r_squared;
		result.vif = 1.0 / (1.0 - reg_result.r_squared);
		result.is_defined = true;
		result.status = VIFResult::Status::OK;

	} catch (...) {
		// If regression fails (e.g., singular matrix), treat as perfect collinearity
		result.vif = std::numeric_limits<double>::infinity();
		result.r_squared = std::numeric_limits<double>::quiet_NaN();
		result.is_defined = false;
		result.status = VIFResult::Status::PERFECT_COLLINEARITY;
	}

	return result;
}

inline bool VIFCalculator::HasHighCollinearity(const Eigen::MatrixXd &X,
                                                 double threshold,
                                                 const core::RegressionOptions &options) {
	auto results = ComputeVIF(X, options);

	for (const auto &result : results) {
		if (result.is_defined && result.vif >= threshold) {
			return true;
		}
	}

	return false;
}

inline bool VIFCalculator::IsConstantColumn(const Eigen::VectorXd &column, double tol) {
	const size_t n = static_cast<size_t>(column.size());
	if (n == 0) return true;

	double mean = column.mean();
	double variance = (column.array() - mean).square().sum() / static_cast<double>(n - 1);

	return variance < tol;
}

} // namespace diagnostics
} // namespace libanostat
