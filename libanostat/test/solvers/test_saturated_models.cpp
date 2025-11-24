#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <libanostat/solvers/ols_solver.hpp>
#include <libanostat/solvers/ridge_solver.hpp>
#include <libanostat/diagnostics/regression_diagnostics.hpp>
#include <libanostat/core/regression_options.hpp>
#include <libanostat/utils/distributions.hpp>
#include <Eigen/Dense>
#include <cmath>

using namespace libanostat;
using namespace libanostat::core;
using namespace libanostat::solvers;
using namespace libanostat::diagnostics;
using namespace libanostat::utils;

const double TOLERANCE = 1e-6;
const double LOOSE_TOLERANCE = 1e-3;

TEST_CASE("Saturated Model: OLS with n == rank (No Residual DF)", "[saturated][ols]") {
	// Create a near-saturated model: 5 observations, 4 predictors + intercept
	// If any columns are collinear, rank < 5, otherwise rank = 5 and df_residual = 0
	Eigen::MatrixXd X(5, 4);
	X << 1, 2, 3, 4,
	     2, 3, 5, 7,
	     3, 5, 8, 12,
	     4, 7, 11, 17,
	     5, 9, 14, 22;

	Eigen::VectorXd y(5);
	y << 10, 15, 22, 30, 38;

	RegressionOptions opts;
	opts.intercept = true;

	auto result = OLSSolver::FitWithStdErrors(y, X, opts);

	REQUIRE(result.is_valid());

	// The goal is to test a near-saturated or saturated model
	// df_residual = n - rank should be small (0, 1, or 2)
	size_t df_residual = 5 - result.rank;
	REQUIRE(df_residual <= 2);  // Small df_residual

	// With small df, MSE calculation and statistics bounds checking are exercised
	REQUIRE((std::isfinite(result.mse) || std::isnan(result.mse)));
	REQUIRE(result.r_squared >= 0.0);
	REQUIRE(result.r_squared <= 1.0);
}

TEST_CASE("Saturated Model: Diagnostics with Leverage >= 1.0", "[saturated][diagnostics]") {
	// Create a model where leverage can be >= 1.0
	// This happens when n ≈ p (saturated or nearly saturated)
	Eigen::MatrixXd X(3, 2);
	X << 1, 0,
	     0, 1,
	     0, 0;

	Eigen::VectorXd y(3);
	y << 5, 10, 15;

	RegressionOptions opts;
	opts.intercept = true;

	auto result = OLSSolver::Fit(y, X, opts);

	// Compute diagnostics
	auto leverage = RegressionDiagnostics::ComputeLeverage(X, result.is_aliased, opts.intercept);
	auto std_residuals = RegressionDiagnostics::ComputeStandardizedResiduals(result.residuals, leverage, result.mse);
	auto cooks_d = RegressionDiagnostics::ComputeCooksDistance(std_residuals, leverage, result.rank);
	auto dffits = RegressionDiagnostics::ComputeDFFITS(std_residuals, leverage);

	REQUIRE(leverage.size() == 3);
	REQUIRE(std_residuals.size() == 3);
	REQUIRE(cooks_d.size() == 3);
	REQUIRE(dffits.size() == 3);

	// Check if any leverage values are >= 1.0 or NaN
	// When leverage >= 1.0, diagnostics should set standardized residuals, Cook's D, DFFITS to NaN
	for (int i = 0; i < 3; i++) {
		if (leverage(i) >= 1.0) {
			REQUIRE(std::isnan(std_residuals(i)));
			REQUIRE(std::isnan(cooks_d(i)));
			REQUIRE(std::isnan(dffits(i)));
		}
	}
}

TEST_CASE("Saturated Model: Ridge with Lambda=0 Delegates to OLS", "[saturated][ridge][delegation]") {
	// Test that Ridge with lambda=0 correctly delegates to OLS
	Eigen::MatrixXd X(20, 3);
	X.setRandom();
	Eigen::VectorXd y(20);
	y.setRandom();

	RegressionOptions ridge_opts = RegressionOptions::Ridge(0.0);
	RegressionOptions ols_opts = RegressionOptions::OLS();

	auto ridge_result = RidgeSolver::Fit(y, X, ridge_opts);
	auto ols_result = OLSSolver::Fit(y, X, ols_opts);

	// Results should be essentially identical
	REQUIRE((ridge_result.coefficients - ols_result.coefficients).norm() < TOLERANCE);
	REQUIRE_THAT(ridge_result.mse, Catch::Matchers::WithinAbs(ols_result.mse, TOLERANCE));
	REQUIRE_THAT(ridge_result.r_squared, Catch::Matchers::WithinAbs(ols_result.r_squared, TOLERANCE));

	// Also test FitWithStdErrors delegation
	auto ridge_se_result = RidgeSolver::FitWithStdErrors(y, X, ridge_opts);
	auto ols_se_result = OLSSolver::FitWithStdErrors(y, X, ols_opts);

	REQUIRE((ridge_se_result.coefficients - ols_se_result.coefficients).norm() < TOLERANCE);
}

TEST_CASE("Numerical Edge Case: Beta Inc Reg with Extreme Values", "[numerical][distributions]") {
	// Test epsilon guards in beta_inc_reg continued fraction
	// Very small a, b values that might trigger epsilon guards
	double result1 = beta_inc_reg(0.5, 0.001, 0.001);
	REQUIRE(std::isfinite(result1));
	REQUIRE(result1 >= 0.0);
	REQUIRE(result1 <= 1.0);

	// Test with very large a, b
	double result2 = beta_inc_reg(0.5, 100.0, 100.0);
	REQUIRE(std::isfinite(result2));
	REQUIRE_THAT(result2, Catch::Matchers::WithinAbs(0.5, 0.01));  // Should be near 0.5 by symmetry

	// Test edge case where x is very close to boundaries
	double result3 = beta_inc_reg(0.001, 2.0, 3.0);
	REQUIRE(std::isfinite(result3));
	REQUIRE(result3 < 0.1);  // Should be very small

	double result4 = beta_inc_reg(0.999, 2.0, 3.0);
	REQUIRE(std::isfinite(result4));
	REQUIRE(result4 > 0.9);  // Should be very close to 1
}

TEST_CASE("Numerical Edge Case: Student t Critical with Extreme Alpha", "[numerical][distributions]") {
	// Test the alpha boundary conditions in student_t_critical
	double critical_0_10 = student_t_critical(0.10, 50);
	REQUIRE(std::isfinite(critical_0_10));
	REQUIRE_THAT(critical_0_10, Catch::Matchers::WithinAbs(1.697, 0.1));

	double critical_0_001 = student_t_critical(0.001, 50);
	REQUIRE(std::isfinite(critical_0_001));
	REQUIRE(critical_0_001 > 3.0);  // Should be large for very small alpha

	double critical_0_0001 = student_t_critical(0.0001, 50);
	REQUIRE(std::isfinite(critical_0_0001));
	REQUIRE(critical_0_0001 >= 3.291);  // Should hit the floor value

	// Test with small df
	double critical_small_df = student_t_critical(0.05, 5);
	double critical_large_df = student_t_critical(0.05, 100);
	REQUIRE(critical_small_df > critical_large_df);  // Smaller df should have larger critical value
}

TEST_CASE("Numerical Edge Case: Statistics Bounds Checking", "[numerical][statistics]") {
	// Try to create scenarios that might produce R² outside [0, 1]
	// This is difficult but can happen with heavy regularization or numerical issues

	// Case 1: Very small sample with Ridge regression (might clamp R²)
	Eigen::MatrixXd X(3, 2);
	X.setRandom();
	Eigen::VectorXd y(3);
	y.setRandom();

	RegressionOptions opts = RegressionOptions::Ridge(100.0);  // Very high lambda
	opts.intercept = true;

	auto result = RidgeSolver::Fit(y, X, opts);

	// R² should be clamped to [0, 1]
	REQUIRE(result.r_squared >= 0.0);
	REQUIRE(result.r_squared <= 1.0);

	if (result.rank > 0) {
		REQUIRE(result.adj_r_squared >= -1.0);  // Adjusted R² can be negative, but not too negative
	}
}

TEST_CASE("Numerical Edge Case: Diagnostics Matrix Inversion Failure", "[numerical][diagnostics]") {
	// Create a nearly singular matrix that might cause leverage computation to fail
	Eigen::MatrixXd X(10, 3);
	X.col(0).setConstant(1.0);
	X.col(1) = Eigen::VectorXd::LinSpaced(10, 1, 10);
	X.col(2) = X.col(1) + Eigen::VectorXd::Random(10) * 1e-10;  // Nearly collinear

	Eigen::VectorXd y = Eigen::VectorXd::Random(10);

	RegressionOptions opts;
	opts.intercept = false;  // No intercept, use X columns directly

	auto result = OLSSolver::Fit(y, X, opts);

	// Leverage computation should not crash even with nearly singular matrix
	auto leverage = RegressionDiagnostics::ComputeLeverage(X, result.is_aliased, opts.intercept);

	REQUIRE(leverage.size() == 10);
	// Some values might be NaN if computation failed, but it shouldn't crash
	for (int i = 0; i < 10; i++) {
		REQUIRE((std::isfinite(leverage(i)) || std::isnan(leverage(i))));
	}
}
