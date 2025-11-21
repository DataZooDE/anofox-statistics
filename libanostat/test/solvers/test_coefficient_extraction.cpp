#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <libanostat/solvers/ols_solver.hpp>
#include <libanostat/core/regression_options.hpp>
#include <Eigen/Dense>

using namespace libanostat;
using namespace libanostat::solvers;
using namespace libanostat::core;

const double TOLERANCE = 1e-6;

TEST_CASE("Coefficient Extraction: Intercept and Feature Coefficients", "[solvers][coefficients]") {
	// Test that when intercept=true, intercept is in coefficients array
	// and can be extracted correctly
	Eigen::VectorXd y(5);
	Eigen::MatrixXd X(5, 1);
	
	X << 1.0, 2.0, 3.0, 4.0, 5.0;
	y << 3.0, 5.0, 7.0, 9.0, 11.0;  // y = 1 + 2*x
	
	RegressionOptions opts;
	opts.intercept = true;
	
	auto result = OLSSolver::Fit(y, X, opts);
	REQUIRE(result.is_valid());
	REQUIRE(result.rank > 0);
	
	// When intercept=true, coefficients array should have size 2 (intercept + 1 feature)
	REQUIRE(result.coefficients.size() == 2);
	
	// Coefficients are stored at their ORIGINAL column positions (not pivoted positions)
	// Original column 0 is the intercept, original column 1 is the feature
	// So intercept is at coefficients[0], feature is at coefficients[1]
	double intercept = result.coefficients(0);
	double feature_coef = result.coefficients(1);
	
	// Verify values (allowing for numerical precision)
	REQUIRE_THAT(intercept, Catch::Matchers::WithinAbs(1.0, TOLERANCE));
	REQUIRE_THAT(feature_coef, Catch::Matchers::WithinAbs(2.0, TOLERANCE));
}

TEST_CASE("Coefficient Extraction: Rank Includes Intercept", "[solvers][rank]") {
	// Test that when intercept=true, rank includes intercept
	Eigen::VectorXd y(5);
	Eigen::MatrixXd X(5, 1);
	
	X << 1.0, 2.0, 3.0, 4.0, 5.0;
	y << 3.0, 5.0, 7.0, 9.0, 11.0;
	
	RegressionOptions opts;
	opts.intercept = true;
	
	auto result = OLSSolver::Fit(y, X, opts);
	REQUIRE(result.is_valid());
	
	// Rank should be 2 (intercept + 1 feature)
	REQUIRE(result.rank == 2);
	
	// df_residual should be n - rank = 5 - 2 = 3
	REQUIRE(result.df_residual() == 3);
}

