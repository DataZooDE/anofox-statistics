#include "elastic_net_solver.hpp"
#include "tracing.hpp"
#include <cmath>
#include <limits>

namespace duckdb {
namespace anofox_statistics {

double ElasticNetSolver::SoftThreshold(double z, double gamma) {
	if (z > gamma) {
		return z - gamma;
	} else if (z < -gamma) {
		return z + gamma;
	} else {
		return 0.0;
	}
}

ElasticNetResult ElasticNetSolver::Fit(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, double alpha, double lambda,
                                       idx_t max_iterations, double tolerance) {

	idx_t n = static_cast<idx_t>(X.rows());
	idx_t p = static_cast<idx_t>(X.cols());

	ElasticNetResult result;
	result.coefficients = Eigen::VectorXd::Zero(p);
	result.n_iterations = 0;
	result.converged = false;

	// Handle special case: alpha = 0 (pure Ridge)
	if (alpha <= 1e-10) {
		// Use closed-form Ridge solution: β = (X'X + λI)^(-1) X'y
		Eigen::MatrixXd XtX = X.transpose() * X;
		Eigen::VectorXd Xty = X.transpose() * y;
		Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(p, p);
		Eigen::MatrixXd XtX_regularized = XtX + lambda * identity;

		Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(XtX_regularized);
		result.coefficients = qr.solve(Xty);
		result.n_iterations = 1;
		result.converged = true;

		// Count non-zero (for Ridge, typically all are non-zero)
		result.n_nonzero = 0;
		for (idx_t j = 0; j < p; j++) {
			if (std::abs(result.coefficients(j)) > 1e-10) {
				result.n_nonzero++;
			}
		}

		// Compute fit statistics
		Eigen::VectorXd y_pred = X * result.coefficients;
		Eigen::VectorXd residuals = y - y_pred;
		double ss_res = residuals.squaredNorm();
		double ss_tot = (y.array() - y.mean()).square().sum();
		result.r_squared = (ss_tot > 1e-10) ? (1.0 - ss_res / ss_tot) : 0.0;

		idx_t df = (n > result.n_nonzero) ? (n - result.n_nonzero) : 1;
		result.mse = ss_res / static_cast<double>(df);
		result.rmse = std::sqrt(result.mse);
		result.adj_r_squared = 1.0 - (1.0 - result.r_squared) * static_cast<double>(n - 1) / static_cast<double>(df);

		return result;
	}

	// Coordinate descent for Elastic Net (alpha > 0)
	// Precompute column norms squared
	Eigen::VectorXd x_norms_sq(p);
	for (idx_t j = 0; j < p; j++) {
		x_norms_sq(j) = X.col(j).squaredNorm();
	}

	// Initialize residuals
	Eigen::VectorXd residuals = y;

	// Coordinate descent iterations
	for (idx_t iter = 0; iter < max_iterations; iter++) {
		double max_change = 0.0;

		for (idx_t j = 0; j < p; j++) {
			// Skip if column is constant/zero
			if (x_norms_sq(j) < 1e-10) {
				continue;
			}

			double beta_old = result.coefficients(j);

			// Add back the contribution of feature j to residuals
			residuals += beta_old * X.col(j);

			// Compute partial correlation: X_j' * residuals
			double rho = X.col(j).dot(residuals);

			// Soft thresholding for L1 penalty
			double threshold = lambda * alpha * static_cast<double>(n);
			double z = SoftThreshold(rho, threshold);

			// Update coefficient with L2 penalty
			double denominator = x_norms_sq(j) + lambda * (1.0 - alpha) * static_cast<double>(n);
			double beta_new = z / denominator;

			result.coefficients(j) = beta_new;

			// Update residuals by removing new contribution
			residuals -= beta_new * X.col(j);

			// Track convergence
			double change = std::abs(beta_new - beta_old);
			if (change > max_change) {
				max_change = change;
			}
		}

		result.n_iterations = iter + 1;

		// Check convergence
		if (max_change < tolerance) {
			result.converged = true;
			break;
		}
	}

	// Count non-zero coefficients
	result.n_nonzero = 0;
	for (idx_t j = 0; j < p; j++) {
		if (std::abs(result.coefficients(j)) > 1e-10) {
			result.n_nonzero++;
		}
	}

	// Compute final predictions and statistics
	Eigen::VectorXd y_pred = X * result.coefficients;
	residuals = y - y_pred;
	double ss_res = residuals.squaredNorm();
	double ss_tot = (y.array() - y.mean()).square().sum();
	result.r_squared = (ss_tot > 1e-10) ? (1.0 - ss_res / ss_tot) : 0.0;

	idx_t df = (n > result.n_nonzero) ? (n - result.n_nonzero) : 1;
	result.mse = ss_res / static_cast<double>(df);
	result.rmse = std::sqrt(result.mse);
	result.adj_r_squared = 1.0 - (1.0 - result.r_squared) * static_cast<double>(n - 1) / static_cast<double>(df);

	ANOFOX_DEBUG("Elastic Net: converged=" << result.converged << ", iterations=" << result.n_iterations << ", nonzero="
	                                       << result.n_nonzero << "/" << p << ", R²=" << result.r_squared);

	return result;
}

} // namespace anofox_statistics
} // namespace duckdb
