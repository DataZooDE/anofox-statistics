#pragma once

#include "duckdb.hpp"
#include <Eigen/Dense>
#include <vector>
#include <cmath>

namespace duckdb {
namespace anofox_statistics {

/**
 * Elastic Net solver using coordinate descent
 *
 * Solves: minimize ||y - Xβ||² + λ(α||β||₁ + (1-α)||β||²₂)
 *
 * Parameters:
 *   - alpha ∈ [0,1]: mixing parameter (0=Ridge, 1=Lasso, (0,1)=Elastic Net)
 *   - lambda >= 0: regularization strength
 */
struct ElasticNetResult {
	Eigen::VectorXd coefficients;  // Estimated coefficients
	idx_t n_nonzero;               // Number of non-zero coefficients
	idx_t n_iterations;            // Number of iterations until convergence
	bool converged;                // Whether the algorithm converged
	double r_squared;              // R-squared
	double adj_r_squared;          // Adjusted R-squared
	double mse;                    // Mean squared error
	double rmse;                   // Root mean squared error
};

class ElasticNetSolver {
public:
	/**
	 * Fit Elastic Net regression using coordinate descent
	 *
	 * @param y Response vector (n x 1)
	 * @param X Design matrix (n x p), assumed already centered if intercept=true
	 * @param alpha Mixing parameter [0,1]: 0=Ridge, 1=Lasso
	 * @param lambda Regularization strength >= 0
	 * @param max_iterations Maximum iterations for coordinate descent
	 * @param tolerance Convergence tolerance
	 * @return ElasticNetResult with coefficients and diagnostics
	 */
	static ElasticNetResult Fit(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, double alpha, double lambda,
	                             idx_t max_iterations = 1000, double tolerance = 1e-6);

private:
	/**
	 * Soft thresholding operator
	 * S(z, γ) = sign(z) * max(|z| - γ, 0)
	 */
	static double SoftThreshold(double z, double gamma);
};

} // namespace anofox_statistics
} // namespace duckdb
