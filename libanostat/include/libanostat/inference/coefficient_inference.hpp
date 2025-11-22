#pragma once

#include "../core/regression_result.hpp"
#include "../core/inference_result.hpp"
#include "../utils/distributions.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <limits>

namespace libanostat {
namespace inference {

/**
 * CoefficientInference: Statistical inference for regression coefficients
 *
 * This class provides methods to compute:
 * - Standard errors of coefficients
 * - t-statistics for hypothesis testing
 * - p-values (two-tailed tests)
 * - Confidence intervals
 *
 * The inference is based on the standard OLS theory:
 * - SE(β_j) = sqrt(σ² * (X'X)^{-1}_{jj})
 * - t_j = β_j / SE(β_j)  ~ t(n - p - 1)
 * - p_j = 2 * P(|T| > |t_j|)
 * - CI_j = β_j ± t_{α/2} * SE(β_j)
 *
 * Where:
 * - σ² = MSE = RSS / (n - p - 1)
 * - X is the design matrix (centered if intercept=true)
 * - n = number of observations
 * - p = number of features
 */
class CoefficientInference {
public:
	/**
	 * Compute coefficient inference from a regression result
	 *
	 * @param result Regression result with coefficients and residuals
	 * @param X Design matrix (n × p, column-major, CENTERED if intercept was used)
	 * @param y Response vector (length n, CENTERED if intercept was used)
	 * @param n_obs Number of observations
	 * @param intercept Whether intercept was included in the model
	 * @param confidence_level Confidence level for intervals (default: 0.95)
	 * @return InferenceResult with t-statistics, p-values, and confidence intervals
	 *
	 * Note: If intercept=true, X and y should be CENTERED (mean-subtracted)
	 *       The intercept inference will be computed separately
	 */
	static core::InferenceResult ComputeInference(
	    const core::RegressionResult &result,
	    const Eigen::MatrixXd &X,
	    const Eigen::VectorXd &y,
	    size_t n_obs,
	    bool intercept,
	    double confidence_level = 0.95);

	/**
	 * Compute intercept standard error
	 *
	 * SE(intercept) = sqrt(MSE * (1/n + x_mean' * (X'X)^{-1} * x_mean))
	 *
	 * @param mse Mean squared error
	 * @param n_obs Number of observations
	 * @param X_centered Centered design matrix
	 * @param x_means Feature means (before centering)
	 * @param is_aliased Vector indicating which features are aliased
	 * @return Standard error of intercept
	 */
	static double ComputeInterceptStdError(
	    double mse,
	    size_t n_obs,
	    const Eigen::MatrixXd &X_centered,
	    const Eigen::VectorXd &x_means,
	    const std::vector<bool> &is_aliased);

	/**
	 * Compute coefficient standard errors using (X'X)^{-1} approach
	 *
	 * SE(β_j) = sqrt(MSE * (X'X)^{-1}_{jj})
	 *
	 * @param mse Mean squared error
	 * @param X Design matrix (n × p, CENTERED if used with intercept)
	 * @param is_aliased Vector indicating which coefficients are aliased
	 * @return Vector of standard errors (NaN for aliased coefficients)
	 */
	static Eigen::VectorXd ComputeStdErrors(
	    double mse,
	    const Eigen::MatrixXd &X,
	    const std::vector<bool> &is_aliased);

	/**
	 * Compute t-statistics for coefficients
	 *
	 * t_j = β_j / SE(β_j)
	 *
	 * @param coefficients Coefficient estimates
	 * @param std_errors Standard errors
	 * @return Vector of t-statistics
	 */
	static Eigen::VectorXd ComputeTStatistics(
	    const Eigen::VectorXd &coefficients,
	    const Eigen::VectorXd &std_errors);

	/**
	 * Compute two-tailed p-values from t-statistics
	 *
	 * p_j = 2 * P(|T| > |t_j|) where T ~ t(df)
	 *
	 * @param t_statistics Vector of t-statistics
	 * @param df Degrees of freedom
	 * @return Vector of p-values
	 */
	static Eigen::VectorXd ComputePValues(
	    const Eigen::VectorXd &t_statistics,
	    size_t df);

	/**
	 * Compute confidence intervals for coefficients
	 *
	 * CI_j = β_j ± t_{α/2, df} * SE(β_j)
	 *
	 * @param coefficients Coefficient estimates
	 * @param std_errors Standard errors
	 * @param df Degrees of freedom
	 * @param confidence_level Confidence level (e.g., 0.95 for 95% CI)
	 * @return Pair of (lower_bounds, upper_bounds)
	 */
	static std::pair<Eigen::VectorXd, Eigen::VectorXd> ComputeConfidenceIntervals(
	    const Eigen::VectorXd &coefficients,
	    const Eigen::VectorXd &std_errors,
	    size_t df,
	    double confidence_level);

private:
	/**
	 * Build (X'X)^{-1} for non-aliased features
	 *
	 * @param X Design matrix
	 * @param is_aliased Vector indicating which features are aliased
	 * @return (X'X)^{-1} matrix (only for non-aliased features)
	 */
	static Eigen::MatrixXd ComputeXtXInverse(
	    const Eigen::MatrixXd &X,
	    const std::vector<bool> &is_aliased);

	/**
	 * Extract non-aliased columns from design matrix
	 *
	 * @param X Full design matrix
	 * @param is_aliased Vector indicating which features are aliased
	 * @return Matrix with only non-aliased columns
	 */
	static Eigen::MatrixXd ExtractValidColumns(
	    const Eigen::MatrixXd &X,
	    const std::vector<bool> &is_aliased);
};

} // namespace inference
} // namespace libanostat
