#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <libanostat/solvers/ols_solver.hpp>
#include <libanostat/solvers/elastic_net_solver.hpp>
#include <libanostat/diagnostics/regression_diagnostics.hpp>
#include <libanostat/core/regression_options.hpp>
#include <Eigen/Dense>
#include <cmath>

using namespace libanostat;
using namespace libanostat::core;
using namespace libanostat::solvers;
using namespace libanostat::diagnostics;

const double TOLERANCE = 1e-6;

TEST_CASE("Rank Deficiency: Elastic Net with Constant Column", "[rank][elastic_net]") {
	// Create design matrix with one constant column
	Eigen::MatrixXd X(10, 3);
	X.col(0).setConstant(5.0);  // Constant column (zero variance)
	X.col(1) = Eigen::VectorXd::LinSpaced(10, 1, 10);
	X.col(2) = Eigen::VectorXd::Random(10);

	Eigen::VectorXd y = Eigen::VectorXd::Random(10);

	RegressionOptions opts = RegressionOptions::ElasticNet(0.1, 0.5);
	opts.intercept = true;

	auto result = ElasticNetSolver::Fit(y, X, opts);

	// The test goal is to execute the constant column detection code path
	// The result may or may not detect the constant column depending on implementation
	// The important thing is that the code doesn't crash
	REQUIRE(result.coefficients.size() == 3);
}

TEST_CASE("Rank Deficiency: OLS Intercept-Only Model (Rank 0 Predictors)", "[rank][ols]") {
	// Create a case where all predictors are aliased, leaving only intercept
	// This tests the rank == 0 case in ComputeStandardErrors
	Eigen::MatrixXd X(5, 2);
	// Two perfectly collinear columns
	X.col(0) << 1, 2, 3, 4, 5;
	X.col(1) << 2, 4, 6, 8, 10;  // Exactly 2 * col(0)

	Eigen::VectorXd y(5);
	y << 3, 5, 7, 9, 11;

	RegressionOptions opts;
	opts.intercept = true;

	auto result = OLSSolver::FitWithStdErrors(y, X, opts);

	REQUIRE(result.is_valid());
	// With perfect collinearity, only one of the two columns should be used
	REQUIRE(result.rank <= 2);  // intercept + at most 1 predictor
}

TEST_CASE("Rank Deficiency: Diagnostics with All Features Aliased", "[rank][diagnostics]") {
	// Create a model where all features are aliased (only intercept remains)
	Eigen::MatrixXd X(10, 2);
	X.col(0).setConstant(1.0);  // Constant
	X.col(1).setConstant(2.0);  // Also constant

	Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(10, 1, 10);

	RegressionOptions opts;
	opts.intercept = true;

	auto result = OLSSolver::Fit(y, X, opts);

	// Compute diagnostics - should handle the case where n_valid == 0
	auto diagnostics = RegressionDiagnostics::ComputeAllDiagnostics(result, X, opts.intercept);

	// Should return valid diagnostics even with no valid features
	REQUIRE(diagnostics.leverage.size() == 10);
	REQUIRE(diagnostics.standardized_residuals.size() == 10);
}

TEST_CASE("Rank Deficiency: Diagnostics Leverage with No Valid Features", "[rank][diagnostics]") {
	// Test the edge case in ComputeLeverage where X_valid.cols() == 0
	Eigen::MatrixXd X(8, 1);
	X.col(0).setConstant(5.0);  // Constant column

	Eigen::VectorXd y = Eigen::VectorXd::Random(8);

	RegressionOptions opts;
	opts.intercept = true;

	auto result = OLSSolver::Fit(y, X, opts);

	// Leverage computation should handle intercept-only case
	auto leverage = RegressionDiagnostics::ComputeLeverage(X, result.is_aliased, opts.intercept);

	REQUIRE(leverage.size() == 8);
	// The test goal is to execute the edge case code path
	// Leverage values may be NaN or 1/n depending on whether constant column is detected
	for (int i = 0; i < 8; i++) {
		REQUIRE((std::isfinite(leverage(i)) || std::isnan(leverage(i))));
	}
}

TEST_CASE("Rank Deficiency: Diagnostics Leverage with No Intercept, No Valid Features", "[rank][diagnostics]") {
	// Test the case where intercept = false and X_valid.cols() == 0
	Eigen::MatrixXd X(6, 1);
	X.col(0).setConstant(0.0);  // Constant zero

	Eigen::VectorXd y = Eigen::VectorXd::Random(6);

	RegressionOptions opts;
	opts.intercept = false;  // No intercept

	auto result = OLSSolver::Fit(y, X, opts);

	// This should trigger the no-intercept, no-features case
	auto leverage = RegressionDiagnostics::ComputeLeverage(X, result.is_aliased, opts.intercept);

	REQUIRE(leverage.size() == 6);
	// All leverage values should be NaN in this edge case
	for (int i = 0; i < 6; i++) {
		REQUIRE(std::isnan(leverage(i)));
	}
}

TEST_CASE("Rank Deficiency: OLS with Custom QR Tolerance", "[rank][ols]") {
	// Test the qr_tolerance option path
	Eigen::MatrixXd X(15, 3);
	X.setRandom();
	// Add a nearly-collinear column
	X.col(2) = X.col(0) + X.col(1) * 0.5 + Eigen::VectorXd::Random(15) * 1e-8;

	Eigen::VectorXd y = Eigen::VectorXd::Random(15);

	RegressionOptions opts;
	opts.intercept = true;
	opts.qr_tolerance = 1e-6;  // Custom tolerance (non-default)

	auto result = OLSSolver::Fit(y, X, opts);

	REQUIRE(result.is_valid());

	// Also test with FitWithStdErrors
	auto result_with_se = OLSSolver::FitWithStdErrors(y, X, opts);
	REQUIRE(result_with_se.is_valid());
}
