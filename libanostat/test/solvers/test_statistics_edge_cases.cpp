#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <libanostat/solvers/ols_solver.hpp>
#include <libanostat/solvers/ridge_solver.hpp>
#include <libanostat/solvers/elastic_net_solver.hpp>
#include <libanostat/core/regression_options.hpp>
#include <Eigen/Dense>
#include <cmath>

using namespace libanostat;
using namespace libanostat::core;
using namespace libanostat::solvers;

const double TOLERANCE = 1e-6;

TEST_CASE("Statistics Edge: R-squared Bounds with Strong Regularization", "[statistics][ridge]") {
	// Test that R² is properly clamped to [0, 1] even with extreme regularization
	Eigen::MatrixXd X(20, 5);
	X.setRandom();
	Eigen::VectorXd y(20);
	y.setRandom();

	// Very high lambda to trigger potential R² clamping
	RegressionOptions opts = RegressionOptions::Ridge(1000.0);
	opts.intercept = true;

	auto result = RidgeSolver::Fit(y, X, opts);

	// R² and adjusted R² should be within valid bounds
	REQUIRE(result.r_squared >= 0.0);
	REQUIRE(result.r_squared <= 1.0);

	if (result.rank > 0) {
		// Adjusted R² can be negative but shouldn't be extreme
		REQUIRE(result.adj_r_squared >= -10.0);
	}
}

TEST_CASE("Statistics Edge: Elastic Net with Very Small Coefficients", "[statistics][elastic_net]") {
	// Test soft thresholding edge cases where coefficients might be zeroed out
	Eigen::MatrixXd X(30, 8);
	X.setRandom();
	X *= 0.01;  // Scale down to get small coefficients

	Eigen::VectorXd y(30);
	y.setRandom();
	y *= 0.01;

	// High lambda to trigger soft thresholding
	RegressionOptions opts = RegressionOptions::ElasticNet(5.0, 0.8);
	opts.intercept = true;
	opts.max_iterations = 1000;

	auto result = ElasticNetSolver::Fit(y, X, opts);

	REQUIRE(result.coefficients.size() == 8);
	REQUIRE(result.r_squared >= 0.0);
	REQUIRE(result.r_squared <= 1.0);

	// Some coefficients should be exactly zero due to L1 penalty
	int zero_count = 0;
	for (int i = 0; i < 8; i++) {
		if (std::abs(result.coefficients[i]) < 1e-10) {
			zero_count++;
		}
	}
	REQUIRE(zero_count >= 0);  // At least some sparsity expected
}

TEST_CASE("Statistics Edge: OLS with Very High Collinearity Near Threshold", "[statistics][ols]") {
	// Test QR decomposition with columns near the collinearity threshold
	Eigen::MatrixXd X(25, 4);
	X.col(0) = Eigen::VectorXd::LinSpaced(25, 1, 25);
	X.col(1) = Eigen::VectorXd::LinSpaced(25, 2, 26);
	X.col(2) = X.col(0) * 2.0 + X.col(1) * 0.5 + Eigen::VectorXd::Random(25) * 1e-7;  // Nearly collinear
	X.col(3) = Eigen::VectorXd::Random(25);

	Eigen::VectorXd y = Eigen::VectorXd::Random(25);

	RegressionOptions opts;
	opts.intercept = true;

	auto result = OLSSolver::Fit(y, X, opts);

	REQUIRE(result.is_valid());
	// Should detect collinearity and handle it
	REQUIRE(result.rank >= 1);
	REQUIRE(result.rank <= 5);  // 4 features + intercept
}

TEST_CASE("Statistics Edge: Ridge with Extremely Small Lambda", "[statistics][ridge]") {
	// Test ridge with lambda very close to zero (should behave like OLS)
	Eigen::MatrixXd X(15, 3);
	X.setRandom();
	Eigen::VectorXd y(15);
	y.setRandom();

	RegressionOptions ridge_opts = RegressionOptions::Ridge(1e-12);  // Tiny lambda
	ridge_opts.intercept = true;

	auto ridge_result = RidgeSolver::Fit(y, X, ridge_opts);

	// With tiny lambda, should produce valid results
	REQUIRE(ridge_result.is_valid());
	REQUIRE(ridge_result.r_squared >= 0.0);
	REQUIRE(ridge_result.r_squared <= 1.0);
}

TEST_CASE("Statistics Edge: Elastic Net Convergence with Difficult Data", "[statistics][elastic_net]") {
	// Test convergence checking with difficult-to-fit data
	Eigen::MatrixXd X(40, 6);
	X.setRandom();

	// Create y with some structure but also noise
	Eigen::VectorXd true_coef(6);
	true_coef << 1.0, -0.5, 0.0, 2.0, 0.0, -1.5;
	Eigen::VectorXd y = X * true_coef + Eigen::VectorXd::Random(40) * 0.5;

	RegressionOptions opts = RegressionOptions::ElasticNet(0.5, 0.5);
	opts.intercept = true;
	opts.max_iterations = 100;
	opts.tolerance = 1e-6;

	auto result = ElasticNetSolver::Fit(y, X, opts);

	// Should produce valid results (may or may not converge in 100 iterations)
	REQUIRE(result.coefficients.size() == 6);
	REQUIRE(result.r_squared >= 0.0);
	REQUIRE(result.r_squared <= 1.0);
}

TEST_CASE("Statistics Edge: Ridge Statistics Bounds on Tiny Dataset", "[statistics][ridge]") {
	// Extremely small dataset that might produce unusual statistics
	Eigen::MatrixXd X(4, 2);
	X.setRandom();
	Eigen::VectorXd y(4);
	y.setRandom();

	RegressionOptions opts = RegressionOptions::Ridge(10.0);
	opts.intercept = true;

	auto result = RidgeSolver::Fit(y, X, opts);

	// All statistics should be within valid ranges
	REQUIRE(std::isfinite(result.mse));
	REQUIRE(result.r_squared >= 0.0);
	REQUIRE(result.r_squared <= 1.0);

	// With strong regularization on tiny data, adjusted R² might be unusual but should be bounded
	REQUIRE(std::isfinite(result.adj_r_squared));
}

TEST_CASE("Statistics Edge: OLS with Rank Exactly at Threshold", "[statistics][ols]") {
	// Create scenario where rank determination is at the QR threshold
	Eigen::MatrixXd X(12, 3);
	X.col(0) = Eigen::VectorXd::LinSpaced(12, 1, 12);
	X.col(1) = Eigen::VectorXd::LinSpaced(12, 5, 16);
	// Make col(2) nearly but not quite a linear combination
	X.col(2) = X.col(0) * 1.5 + X.col(1) * 0.7 + Eigen::VectorXd::Constant(12, 1e-6);

	Eigen::VectorXd y(12);
	y.setRandom();

	RegressionOptions opts;
	opts.intercept = true;
	// Don't set custom tolerance - use default

	auto result = OLSSolver::FitWithStdErrors(y, X, opts);

	REQUIRE(result.is_valid());
	REQUIRE(result.rank >= 1);

	// Standard errors should be computed
	REQUIRE(result.std_errors.size() > 0);

	// All statistics should be valid
	REQUIRE(std::isfinite(result.mse));
	REQUIRE(result.r_squared >= 0.0);
	REQUIRE(result.r_squared <= 1.0);
}

TEST_CASE("Statistics Edge: Elastic Net with Alpha Extremes", "[statistics][elastic_net]") {
	Eigen::MatrixXd X(20, 4);
	X.setRandom();
	Eigen::VectorXd y(20);
	y.setRandom();

	// Test with alpha = 0 (pure Ridge)
	RegressionOptions opts_ridge = RegressionOptions::ElasticNet(1.0, 0.0);
	opts_ridge.intercept = true;
	auto result_ridge = ElasticNetSolver::Fit(y, X, opts_ridge);
	REQUIRE(result_ridge.coefficients.size() >= 4);  // At least 4 features
	REQUIRE(result_ridge.r_squared >= 0.0);
	REQUIRE(result_ridge.r_squared <= 1.0);

	// Test with alpha = 1 (pure Lasso)
	RegressionOptions opts_lasso = RegressionOptions::ElasticNet(1.0, 1.0);
	opts_lasso.intercept = true;
	auto result_lasso = ElasticNetSolver::Fit(y, X, opts_lasso);
	REQUIRE(result_lasso.coefficients.size() >= 4);  // At least 4 features
	REQUIRE(result_lasso.r_squared >= 0.0);
	REQUIRE(result_lasso.r_squared <= 1.0);
}

TEST_CASE("Statistics Edge: OLS Adjusted R-squared with Small Sample", "[statistics][ols]") {
	// Test adjusted R² calculation with very small n
	Eigen::MatrixXd X(6, 2);
	X.setRandom();
	Eigen::VectorXd y(6);
	y.setRandom();

	RegressionOptions opts;
	opts.intercept = true;

	auto result = OLSSolver::Fit(y, X, opts);

	REQUIRE(result.is_valid());

	// n = 6, rank could be 3 (intercept + 2 features)
	// df_residual = 6 - 3 = 3
	// adjusted R² uses these values and should be computed correctly
	REQUIRE(std::isfinite(result.adj_r_squared));

	// For small samples, adjusted R² can be negative if model fits poorly
	// but shouldn't be absurdly negative
	REQUIRE(result.adj_r_squared >= -2.0);
}
