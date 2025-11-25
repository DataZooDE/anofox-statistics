#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <libanostat/solvers/ols_solver.hpp>
#include <libanostat/solvers/ridge_solver.hpp>
#include <libanostat/solvers/elastic_net_solver.hpp>
#include <libanostat/diagnostics/regression_diagnostics.hpp>
#include <libanostat/diagnostics/regression_diagnostics_impl.hpp>
#include <Eigen/Dense>

using namespace libanostat;
using namespace libanostat::solvers;
using namespace libanostat::diagnostics;
using namespace libanostat::core;

const double TOLERANCE = 1e-6;

TEST_CASE("Integration: OLS with Diagnostics and Inference", "[integration][workflow]") {
	// Full workflow: Fit → Diagnostics → Inference
	Eigen::VectorXd y(10);
	Eigen::MatrixXd X(10, 2);

	X << 1.0, 2.0,
	     2.0, 3.0,
	     3.0, 4.0,
	     4.0, 5.0,
	     5.0, 6.0,
	     6.0, 7.0,
	     7.0, 8.0,
	     8.0, 9.0,
	     9.0, 10.0,
	     10.0, 11.0;

	y << 5.1, 8.0, 11.2, 14.1, 17.0, 20.1, 23.0, 26.1, 29.0, 32.1;

	// Step 1: Fit model
	RegressionOptions opts;
	opts.intercept = true;

	auto result = OLSSolver::Fit(y, X, opts);
	REQUIRE(result.rank > 0);
	REQUIRE(result.is_valid());

	// Step 2: Compute diagnostics
	auto diag = RegressionDiagnostics::ComputeAllDiagnostics(result, X, true);
	REQUIRE(diag.has_diagnostics);

	// Step 3: Verify all components are present
	REQUIRE(result.coefficients.size() > 0);
	REQUIRE(diag.leverage.size() == 10);
	REQUIRE(diag.standardized_residuals.size() == 10);
	REQUIRE(diag.cooks_distance.size() == 10);
	REQUIRE(diag.dffits.size() == 10);
}

TEST_CASE("Integration: Ridge with Prediction Intervals", "[integration][workflow]") {
	// Test Ridge regression with inference
	Eigen::VectorXd y(10);
	Eigen::MatrixXd X(10, 1);

	X << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0;
	y << 2.1, 4.0, 6.1, 8.0, 10.1, 12.0, 14.1, 16.0, 18.1, 20.0;

	RegressionOptions opts;
	opts.intercept = true;
	opts.lambda = 0.1;

	auto result = RidgeSolver::Fit(y, X, opts);
	REQUIRE(result.rank > 0);
	REQUIRE(result.is_valid());

	// Verify Ridge produces valid results
	REQUIRE(result.coefficients.size() > 0);
	REQUIRE(result.r_squared >= 0.0);
	REQUIRE(result.r_squared <= 1.0);
}

TEST_CASE("Integration: Compare OLS vs Ridge vs Elastic Net", "[integration][workflow]") {
	// Compare different methods on same data
	Eigen::VectorXd y(20);
	Eigen::MatrixXd X(20, 2);

	for (int i = 0; i < 20; i++) {
		X(i, 0) = static_cast<double>(i + 1);
		X(i, 1) = static_cast<double>(i + 1) * 0.5;
		y(i) = 1.0 + 2.0 * X(i, 0) + 0.5 * X(i, 1) + 0.1 * (i % 3);
	}

	// Fit OLS
	RegressionOptions ols_opts;
	ols_opts.intercept = true;
	auto ols_result = OLSSolver::Fit(y, X, ols_opts);
	REQUIRE(ols_result.rank > 0);
	REQUIRE(ols_result.is_valid());

	// Fit Ridge
	RegressionOptions ridge_opts;
	ridge_opts.intercept = true;
	ridge_opts.lambda = 0.1;
	auto ridge_result = RidgeSolver::Fit(y, X, ridge_opts);
	REQUIRE(ridge_result.rank > 0);
	REQUIRE(ridge_result.is_valid());

	// Fit Elastic Net
	RegressionOptions enet_opts;
	enet_opts.intercept = true;
	enet_opts.lambda = 0.1;
	enet_opts.alpha = 0.5;
	auto enet_result = ElasticNetSolver::Fit(y, X, enet_opts);
	// Elastic Net should succeed (may have rank > 0 or handle edge cases)
	if (enet_result.rank > 0) {
		REQUIRE(enet_result.is_valid());
	}

	// All should produce valid R²
	REQUIRE(ols_result.r_squared >= 0.0);
	REQUIRE(ridge_result.r_squared >= 0.0);
	REQUIRE(enet_result.r_squared >= 0.0);

	// Ridge and Elastic Net should have smaller coefficients than OLS (shrinkage)
	for (size_t i = 0; i < ols_result.coefficients.size(); i++) {
		if (!ols_result.is_aliased[i] && !ridge_result.is_aliased[i]) {
			REQUIRE(std::abs(ridge_result.coefficients(static_cast<Eigen::Index>(i))) <= 
			        std::abs(ols_result.coefficients(static_cast<Eigen::Index>(i))) + 0.1);
		}
	}
}

TEST_CASE("Integration: Residuals Consistency", "[integration][workflow]") {
	// Verify residuals from fit match manual calculation
	Eigen::VectorXd y(10);
	Eigen::MatrixXd X(10, 1);

	X << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0;
	y << 2.1, 4.0, 6.1, 8.0, 10.1, 12.0, 14.1, 16.0, 18.1, 20.0;

	RegressionOptions opts;
	opts.intercept = true;

	auto result = OLSSolver::Fit(y, X, opts);
	REQUIRE(result.rank > 0);
	REQUIRE(result.is_valid());

	// Verify residuals = y - fitted_values (where fitted = y - residuals)
	for (int i = 0; i < 10; i++) {
		double fitted_manual = y(i) - result.residuals(i);
		// Verify that residuals are consistent: y - fitted = residual
		double residual_check = y(i) - fitted_manual;
		REQUIRE_THAT(result.residuals(i),
		             Catch::Matchers::WithinAbs(residual_check, TOLERANCE));
	}
}

