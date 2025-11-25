#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <libanostat/solvers/ols_solver.hpp>

using namespace libanostat;
using namespace libanostat::solvers;

const double TOLERANCE = 1e-6;
const double LOOSE_TOLERANCE = 1e-4;

TEST_CASE("Inference: Simple OLS with Standard Errors", "[inference][ols]") {
	// Create simple test data: y = 2 + 3*x + noise
	Eigen::VectorXd y(5);
	Eigen::MatrixXd X(5, 1);

	X << 1.0, 2.0, 3.0, 4.0, 5.0;
	y << 5.1, 8.0, 11.2, 13.9, 17.1;

	// Fit OLS model with standard errors
	core::RegressionOptions opts;
	opts.intercept = true;
	opts.confidence_level = 0.95;

	auto result = OLSSolver::FitWithStdErrors(y, X, opts);

	// Check that we have coefficients (intercept + slope)
	REQUIRE(result.coefficients.size() == 2);
	REQUIRE(result.has_std_errors);

	// Intercept should be close to 2.0
	REQUIRE_THAT(result.coefficients(0), Catch::Matchers::WithinAbs(2.0, 0.5));

	// Slope should be close to 3.0
	REQUIRE_THAT(result.coefficients(1), Catch::Matchers::WithinAbs(3.0, 0.2));

	// Check that standard errors exist and are valid
	REQUIRE(result.std_errors.size() == 2);

	// All standard errors should be positive
	for (int i = 0; i < 2; i++) {
		REQUIRE(result.std_errors(i) > 0.0);
		REQUIRE_FALSE(std::isnan(result.std_errors(i)));
	}
}

TEST_CASE("Inference: Aliased Coefficients", "[inference][aliased]") {
	// Test that aliased coefficients get NaN values

	// Create data with perfect collinearity
	Eigen::VectorXd y(5);
	Eigen::MatrixXd X(5, 2);

	X << 1.0, 2.0,
	     2.0, 4.0,
	     3.0, 6.0,
	     4.0, 8.0,
	     5.0, 10.0;  // Second column is 2x first column

	y << 5.0, 8.0, 11.0, 14.0, 17.0;

	// Fit with standard errors
	core::RegressionOptions opts;
	opts.intercept = true;

	auto result = OLSSolver::FitWithStdErrors(y, X, opts);

	// With auto-intercept, we have 3 columns: intercept + X[:, 0] + X[:, 1]
	// Since X[:, 0] and X[:, 1] are collinear, rank should be 2 (intercept + one X column)
	REQUIRE(result.rank < 3);

	// At least one coefficient should be aliased
	bool has_aliased = false;
	for (bool aliased : result.is_aliased) {
		if (aliased) {
			has_aliased = true;
			break;
		}
	}
	REQUIRE(has_aliased);

	// Standard errors should exist
	REQUIRE(result.has_std_errors);

	// Aliased coefficients should have NaN values for both coefficients and std errors
	for (size_t i = 0; i < result.is_aliased.size(); i++) {
		if (result.is_aliased[i]) {
			REQUIRE(std::isnan(result.coefficients(static_cast<Eigen::Index>(i))));
			REQUIRE(std::isnan(result.std_errors(static_cast<Eigen::Index>(i))));
		}
	}
}

TEST_CASE("Inference: Intercept Standard Error", "[inference][intercept]") {
	// Test that intercept standard error is computed correctly

	// Simple data where we can verify intercept SE
	Eigen::VectorXd y(4);
	Eigen::MatrixXd X(4, 1);

	X << 1.0, 2.0, 3.0, 4.0;
	y << 3.0, 5.0, 7.0, 9.0;  // Perfect linear: y = 1 + 2*x

	// Fit with standard errors
	core::RegressionOptions opts;
	opts.intercept = true;

	auto result = OLSSolver::FitWithStdErrors(y, X, opts);

	REQUIRE(result.has_std_errors);

	// Intercept SE should be positive and finite
	REQUIRE(result.std_errors(0) > 0.0);
	REQUIRE_FALSE(std::isnan(result.std_errors(0)));
	REQUIRE_FALSE(std::isinf(result.std_errors(0)));
}

TEST_CASE("Inference: No Intercept Model", "[inference][no_intercept]") {
	// Test inference when no intercept is used

	Eigen::VectorXd y(5);
	Eigen::MatrixXd X(5, 1);

	X << 1.0, 2.0, 3.0, 4.0, 5.0;
	y << 2.5, 5.2, 7.8, 10.1, 12.7;  // y ≈ 2.5*x (no intercept)

	// Fit without intercept
	core::RegressionOptions opts;
	opts.intercept = false;

	auto result = OLSSolver::FitWithStdErrors(y, X, opts);

	REQUIRE(result.has_std_errors);

	// Should have only one coefficient (slope, no intercept)
	REQUIRE(result.coefficients.size() == 1);
	REQUIRE(result.std_errors.size() == 1);

	// Slope should be close to 2.5
	REQUIRE_THAT(result.coefficients(0), Catch::Matchers::WithinAbs(2.5, 0.2));

	// Standard error should be positive
	REQUIRE(result.std_errors(0) > 0.0);
	REQUIRE_FALSE(std::isnan(result.std_errors(0)));
}

TEST_CASE("Inference: Standard Errors Properties", "[inference][properties]") {
	// Test that standard errors have correct properties
	Eigen::VectorXd y(10);
	Eigen::MatrixXd X(10, 1);

	X << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0;
	y << 2.1, 4.0, 6.1, 8.0, 10.1, 12.0, 14.1, 16.0, 18.1, 20.0;

	core::RegressionOptions opts;
	opts.intercept = true;

	auto result = OLSSolver::FitWithStdErrors(y, X, opts);

	REQUIRE(result.has_std_errors);
	
	// Standard errors should be positive for non-aliased coefficients
	for (size_t i = 0; i < result.is_aliased.size(); i++) {
		if (!result.is_aliased[i]) {
			REQUIRE(result.std_errors(static_cast<Eigen::Index>(i)) > 0.0);
			REQUIRE_FALSE(std::isnan(result.std_errors(static_cast<Eigen::Index>(i))));
		}
	}
}

TEST_CASE("Inference: Standard Errors with Different Sample Sizes", "[inference][properties]") {
	// Test that standard errors decrease with larger sample size
	Eigen::VectorXd y_small(5);
	Eigen::MatrixXd X_small(5, 1);
	
	X_small << 1.0, 2.0, 3.0, 4.0, 5.0;
	y_small << 2.1, 4.0, 6.1, 8.0, 10.1;

	Eigen::VectorXd y_large(20);
	Eigen::MatrixXd X_large(20, 1);
	
	for (int i = 0; i < 20; i++) {
		X_large(i, 0) = static_cast<double>(i + 1);
		y_large(i) = 2.0 * X_large(i, 0) + 0.1;
	}

	core::RegressionOptions opts;
	opts.intercept = true;

	auto result_small = OLSSolver::FitWithStdErrors(y_small, X_small, opts);
	auto result_large = OLSSolver::FitWithStdErrors(y_large, X_large, opts);

	REQUIRE(result_small.has_std_errors);
	REQUIRE(result_large.has_std_errors);
	
	// Larger sample should have smaller standard errors (for slope)
	if (!result_small.is_aliased[1] && !result_large.is_aliased[1]) {
		double se_small = result_small.std_errors(1);
		double se_large = result_large.std_errors(1);
		REQUIRE(se_large < se_small);
	}
}

TEST_CASE("Inference: Aliased Coefficients with Inference", "[inference][aliased]") {
	// Test inference with aliased coefficients using inference namespace
	Eigen::VectorXd y(5);
	Eigen::MatrixXd X(5, 2);

	X << 1.0, 2.0,
	     2.0, 4.0,
	     3.0, 6.0,
	     4.0, 8.0,
	     5.0, 10.0;  // Second column is 2x first

	y << 5.0, 8.0, 11.0, 14.0, 17.0;

	core::RegressionOptions opts;
	opts.intercept = true;

	auto result = OLSSolver::FitWithStdErrors(y, X, opts);

	REQUIRE(result.has_std_errors);
	
	// Aliased coefficients should have NaN for standard errors
	for (size_t i = 0; i < result.is_aliased.size(); i++) {
		if (result.is_aliased[i]) {
			REQUIRE(std::isnan(result.std_errors(static_cast<Eigen::Index>(i))));
		}
	}
}

TEST_CASE("Inference: Very Small Sample", "[inference][edge]") {
	// Test with minimum sample size (n=3, p=1, intercept)
	Eigen::VectorXd y(3);
	Eigen::MatrixXd X(3, 1);

	X << 1.0, 2.0, 3.0;
	y << 2.0, 4.0, 6.0;

	core::RegressionOptions opts;
	opts.intercept = true;
	opts.confidence_level = 0.95;

	auto result = OLSSolver::FitWithStdErrors(y, X, opts);

	// Should still compute inference (though with large standard errors)
	if (result.has_std_errors) {
		for (int i = 0; i < result.std_errors.size(); i++) {
			REQUIRE(result.std_errors(i) > 0.0);
			REQUIRE_FALSE(std::isnan(result.std_errors(i)));
		}
	}
}

TEST_CASE("Inference: Very Strong Relationship", "[inference][edge]") {
	// Create data with very strong relationship (should have small standard errors)
	Eigen::VectorXd y(100);
	Eigen::MatrixXd X(100, 1);

	for (int i = 0; i < 100; i++) {
		X(i, 0) = static_cast<double>(i + 1);
		y(i) = 2.0 * X(i, 0) + 0.01;  // Almost perfect relationship
	}

	core::RegressionOptions opts;
	opts.intercept = true;

	auto result = OLSSolver::FitWithStdErrors(y, X, opts);

	REQUIRE(result.has_std_errors);
	
	// Strong relationship should have small standard errors
	if (!result.is_aliased[1]) {
		double se = result.std_errors(1);  // Slope
		REQUIRE(se > 0.0);
		REQUIRE(se < 1.0);  // Should be small for strong relationship
		REQUIRE_FALSE(std::isnan(se));
	}
}
