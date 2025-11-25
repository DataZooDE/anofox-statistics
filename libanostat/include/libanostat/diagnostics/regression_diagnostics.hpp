#pragma once

#include "../core/regression_result.hpp"
#include <Eigen/Dense>
#include <vector>

namespace libanostat {
namespace diagnostics {

/**
 * RegressionDiagnostics: Diagnostic measures for regression models
 *
 * This class provides methods to compute:
 * - Leverage (hat values): h_i = x_i'(X'X)^{-1}x_i
 * - Standardized residuals: r_i / sqrt(MSE * (1 - h_i))
 * - Studentized residuals: deleted residuals
 * - Cook's distance: influence of each observation
 * - DFFITS: difference in fits
 * - DFBETAS: difference in betas
 *
 * These diagnostics help identify:
 * - Outliers (large residuals)
 * - High leverage points (unusual X values)
 * - Influential observations (large Cook's D)
 */
class RegressionDiagnostics {
public:
	/**
	 * Compute leverage (hat) values
	 *
	 * h_i = x_i'(X'X)^{-1}x_i
	 *
	 * Properties:
	 * - 0 ≤ h_i ≤ 1
	 * - Σ h_i = p + 1 (if intercept included)
	 * - High leverage: h_i > 2(p+1)/n or h_i > 3(p+1)/n
	 *
	 * @param X Design matrix (n × p, CENTERED if intercept=true)
	 * @param is_aliased Vector indicating which features are aliased
	 * @param intercept Whether model includes intercept
	 * @return Vector of leverage values (length n)
	 */
	static Eigen::VectorXd ComputeLeverage(
	    const Eigen::MatrixXd &X,
	    const std::vector<bool> &is_aliased,
	    bool intercept);

	/**
	 * Compute standardized residuals
	 *
	 * r_i^std = r_i / sqrt(MSE * (1 - h_i))
	 *
	 * Standardized residuals should follow N(0,1) approximately
	 * Outliers: |r_i^std| > 2 or 3
	 *
	 * @param residuals Raw residuals
	 * @param leverage Leverage values
	 * @param mse Mean squared error
	 * @return Vector of standardized residuals
	 */
	static Eigen::VectorXd ComputeStandardizedResiduals(
	    const Eigen::VectorXd &residuals,
	    const Eigen::VectorXd &leverage,
	    double mse);

	/**
	 * Compute Cook's distance
	 *
	 * D_i = (r_i^std)² * h_i / ((p+1) * (1 - h_i))
	 *
	 * Cook's distance measures the influence of observation i on all fitted values
	 * Influential: D_i > 4/n or D_i > 1
	 *
	 * @param standardized_residuals Standardized residuals
	 * @param leverage Leverage values
	 * @param n_params Number of parameters (p + 1 if intercept)
	 * @return Vector of Cook's distances
	 */
	static Eigen::VectorXd ComputeCooksDistance(
	    const Eigen::VectorXd &standardized_residuals,
	    const Eigen::VectorXd &leverage,
	    size_t n_params);

	/**
	 * Compute DFFITS (difference in fits)
	 *
	 * DFFITS_i = r_i^std * sqrt(h_i / (1 - h_i))
	 *
	 * DFFITS measures the influence of observation i on its own fitted value
	 * Influential: |DFFITS_i| > 2*sqrt((p+1)/n) for small n
	 *              |DFFITS_i| > 2*sqrt(p+1) for large n
	 *
	 * @param standardized_residuals Standardized residuals
	 * @param leverage Leverage values
	 * @return Vector of DFFITS values
	 */
	static Eigen::VectorXd ComputeDFFITS(
	    const Eigen::VectorXd &standardized_residuals,
	    const Eigen::VectorXd &leverage);

	/**
	 * Compute all diagnostics for a regression result
	 *
	 * @param result Regression result with residuals
	 * @param X Design matrix (CENTERED if intercept=true)
	 * @param intercept Whether intercept was included
	 * @return Struct with all diagnostic measures
	 */
	struct DiagnosticsResult {
		Eigen::VectorXd leverage;
		Eigen::VectorXd standardized_residuals;
		Eigen::VectorXd cooks_distance;
		Eigen::VectorXd dffits;
		bool has_diagnostics = false;
	};

	static DiagnosticsResult ComputeAllDiagnostics(
	    const core::RegressionResult &result,
	    const Eigen::MatrixXd &X,
	    bool intercept);

private:
	/**
	 * Extract non-aliased columns from design matrix
	 */
	static Eigen::MatrixXd ExtractValidColumns(
	    const Eigen::MatrixXd &X,
	    const std::vector<bool> &is_aliased);
};

} // namespace diagnostics
} // namespace libanostat
