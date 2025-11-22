#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <libanostat/diagnostics/regression_diagnostics.hpp>
#include <libanostat/diagnostics/regression_diagnostics_impl.hpp>
#include <libanostat/solvers/ols_solver.hpp>
#include <libanostat/core/regression_result.hpp>
#include <Eigen/Dense>

using namespace libanostat;
using namespace libanostat::diagnostics;
using namespace libanostat::solvers;
using namespace libanostat::core;

const double TOLERANCE = 1e-6;
const double LOOSE_TOLERANCE = 1e-4;

TEST_CASE("Diagnostics: Leverage with Intercept", "[diagnostics][leverage]") {
	// Simple linear regression: y = 1 + 2*x
	Eigen::VectorXd y(5);
	Eigen::MatrixXd X(5, 1);
	
	X << 1.0, 2.0, 3.0, 4.0, 5.0;
	y << 3.0, 5.0, 7.0, 9.0, 11.0;
	
	// Fit OLS model
	RegressionOptions opts;
	opts.intercept = true;
	
	auto result = OLSSolver::Fit(y, X, opts);
	REQUIRE(result.rank > 0);
	REQUIRE(result.is_valid());
	
	// Compute leverage
	Eigen::VectorXd leverage = RegressionDiagnostics::ComputeLeverage(
		X, result.is_aliased, true);
	
	REQUIRE(leverage.size() == 5);
	
	// Leverage values should be between 0 and 1
	for (int i = 0; i < 5; i++) {
		REQUIRE(leverage(i) >= 0.0);
		REQUIRE(leverage(i) <= 1.0);
		REQUIRE_FALSE(std::isnan(leverage(i)));
	}
	
	// Sum of leverage should equal p + 1 = 2 (intercept + 1 feature)
	double leverage_sum = leverage.sum();
	REQUIRE_THAT(leverage_sum, Catch::Matchers::WithinAbs(2.0, TOLERANCE));
}

TEST_CASE("Diagnostics: Leverage without Intercept", "[diagnostics][leverage]") {
	// Simple regression without intercept
	Eigen::VectorXd y(5);
	Eigen::MatrixXd X(5, 1);
	
	X << 1.0, 2.0, 3.0, 4.0, 5.0;
	y << 2.0, 4.0, 6.0, 8.0, 10.0;
	
	RegressionOptions opts;
	opts.intercept = false;
	
	auto result = OLSSolver::Fit(y, X, opts);
	REQUIRE(result.rank > 0);
	REQUIRE(result.is_valid());
	
	Eigen::VectorXd leverage = RegressionDiagnostics::ComputeLeverage(
		X, result.is_aliased, false);
	
	REQUIRE(leverage.size() == 5);
	
	// Sum of leverage should equal p = 1 (no intercept)
	double leverage_sum = leverage.sum();
	REQUIRE_THAT(leverage_sum, Catch::Matchers::WithinAbs(1.0, TOLERANCE));
}

TEST_CASE("Diagnostics: Leverage with Aliased Features", "[diagnostics][leverage]") {
	// Create data with perfect collinearity
	Eigen::VectorXd y(5);
	Eigen::MatrixXd X(5, 2);
	
	X << 1.0, 2.0,
	     2.0, 4.0,
	     3.0, 6.0,
	     4.0, 8.0,
	     5.0, 10.0;  // Second column is 2x first
	
	y << 3.0, 6.0, 9.0, 12.0, 15.0;
	
	RegressionOptions opts;
	opts.intercept = true;
	
	auto result = OLSSolver::Fit(y, X, opts);
	REQUIRE(result.rank > 0);
	REQUIRE(result.is_valid());
	// With intercept + 2 features, perfect collinearity should give rank < 3
	// (intercept + one of the collinear features)
	REQUIRE(result.rank < 3);
	
	// Should handle aliased features gracefully
	Eigen::VectorXd leverage = RegressionDiagnostics::ComputeLeverage(
		X, result.is_aliased, true);
	
	REQUIRE(leverage.size() == 5);
	
	// All leverage values should be valid (not NaN)
	for (int i = 0; i < 5; i++) {
		REQUIRE_FALSE(std::isnan(leverage(i)));
	}
}

TEST_CASE("Diagnostics: Standardized Residuals", "[diagnostics][residuals]") {
	Eigen::VectorXd y(5);
	Eigen::MatrixXd X(5, 1);
	
	X << 1.0, 2.0, 3.0, 4.0, 5.0;
	y << 3.1, 5.0, 6.9, 9.1, 10.9;  // Slight noise
	
	RegressionOptions opts;
	opts.intercept = true;
	
	auto result = OLSSolver::Fit(y, X, opts);
	REQUIRE(result.rank > 0);
	REQUIRE(result.is_valid());
	
	Eigen::VectorXd leverage = RegressionDiagnostics::ComputeLeverage(
		X, result.is_aliased, true);
	
	Eigen::VectorXd std_residuals = RegressionDiagnostics::ComputeStandardizedResiduals(
		result.residuals, leverage, result.mse);
	
	REQUIRE(std_residuals.size() == 5);
	
	// Standardized residuals should be approximately N(0,1)
	// Mean should be close to 0
	double mean = std_residuals.mean();
	REQUIRE_THAT(std::abs(mean), Catch::Matchers::WithinAbs(0.0, 0.5));
	
	// All should be finite
	for (int i = 0; i < 5; i++) {
		REQUIRE_FALSE(std::isnan(std_residuals(i)));
		REQUIRE_FALSE(std::isinf(std_residuals(i)));
	}
}

TEST_CASE("Diagnostics: Standardized Residuals with High Leverage", "[diagnostics][residuals]") {
	// Create data with one high leverage point
	Eigen::VectorXd y(5);
	Eigen::MatrixXd X(5, 1);
	
	X << 1.0, 2.0, 3.0, 4.0, 100.0;  // Last point is outlier in X
	y << 3.0, 5.0, 7.0, 9.0, 11.0;
	
	RegressionOptions opts;
	opts.intercept = true;
	
	auto result = OLSSolver::Fit(y, X, opts);
	REQUIRE(result.rank > 0);
	REQUIRE(result.is_valid());
	
	Eigen::VectorXd leverage = RegressionDiagnostics::ComputeLeverage(
		X, result.is_aliased, true);
	
	// Last observation should have high leverage
	REQUIRE(leverage(4) > leverage(0));
	REQUIRE(leverage(4) > leverage(1));
	
	Eigen::VectorXd std_residuals = RegressionDiagnostics::ComputeStandardizedResiduals(
		result.residuals, leverage, result.mse);
	
	REQUIRE(std_residuals.size() == 5);
	
	// All should be finite
	for (int i = 0; i < 5; i++) {
		REQUIRE_FALSE(std::isnan(std_residuals(i)));
	}
}

TEST_CASE("Diagnostics: Cook's Distance", "[diagnostics][cooks]") {
	Eigen::VectorXd y(5);
	Eigen::MatrixXd X(5, 1);
	
	X << 1.0, 2.0, 3.0, 4.0, 5.0;
	y << 3.0, 5.0, 7.0, 9.0, 11.0;
	
	RegressionOptions opts;
	opts.intercept = true;
	
	auto result = OLSSolver::Fit(y, X, opts);
	REQUIRE(result.rank > 0);
	REQUIRE(result.is_valid());
	
	Eigen::VectorXd leverage = RegressionDiagnostics::ComputeLeverage(
		X, result.is_aliased, true);
	
	Eigen::VectorXd std_residuals = RegressionDiagnostics::ComputeStandardizedResiduals(
		result.residuals, leverage, result.mse);
	
	size_t n_params = result.rank + 1;  // intercept + 1 feature
	Eigen::VectorXd cooks_d = RegressionDiagnostics::ComputeCooksDistance(
		std_residuals, leverage, n_params);
	
	REQUIRE(cooks_d.size() == 5);
	
	// Cook's D should be non-negative
	for (int i = 0; i < 5; i++) {
		REQUIRE(cooks_d(i) >= 0.0);
		REQUIRE_FALSE(std::isnan(cooks_d(i)));
		REQUIRE_FALSE(std::isinf(cooks_d(i)));
	}
}

TEST_CASE("Diagnostics: Cook's Distance with Influential Observation", "[diagnostics][cooks]") {
	// Create data with one influential observation
	Eigen::VectorXd y(5);
	Eigen::MatrixXd X(5, 1);
	
	X << 1.0, 2.0, 3.0, 4.0, 5.0;
	y << 3.0, 5.0, 7.0, 9.0, 50.0;  // Last point is outlier in y
	
	RegressionOptions opts;
	opts.intercept = true;
	
	auto result = OLSSolver::Fit(y, X, opts);
	REQUIRE(result.rank > 0);
	REQUIRE(result.is_valid());
	
	Eigen::VectorXd leverage = RegressionDiagnostics::ComputeLeverage(
		X, result.is_aliased, true);
	
	Eigen::VectorXd std_residuals = RegressionDiagnostics::ComputeStandardizedResiduals(
		result.residuals, leverage, result.mse);
	
	size_t n_params = result.rank + 1;
	Eigen::VectorXd cooks_d = RegressionDiagnostics::ComputeCooksDistance(
		std_residuals, leverage, n_params);
	
	// Last observation should have high Cook's D
	REQUIRE(cooks_d(4) > cooks_d(0));
	REQUIRE(cooks_d(4) > 0.0);
}

TEST_CASE("Diagnostics: DFFITS", "[diagnostics][dffits]") {
	Eigen::VectorXd y(5);
	Eigen::MatrixXd X(5, 1);
	
	X << 1.0, 2.0, 3.0, 4.0, 5.0;
	y << 3.0, 5.0, 7.0, 9.0, 11.0;
	
	RegressionOptions opts;
	opts.intercept = true;
	
	auto result = OLSSolver::Fit(y, X, opts);
	REQUIRE(result.rank > 0);
	REQUIRE(result.is_valid());
	
	Eigen::VectorXd leverage = RegressionDiagnostics::ComputeLeverage(
		X, result.is_aliased, true);
	
	Eigen::VectorXd std_residuals = RegressionDiagnostics::ComputeStandardizedResiduals(
		result.residuals, leverage, result.mse);
	
	Eigen::VectorXd dffits = RegressionDiagnostics::ComputeDFFITS(
		std_residuals, leverage);
	
	REQUIRE(dffits.size() == 5);
	
	// All DFFITS should be finite
	for (int i = 0; i < 5; i++) {
		REQUIRE_FALSE(std::isnan(dffits(i)));
		REQUIRE_FALSE(std::isinf(dffits(i)));
	}
}

TEST_CASE("Diagnostics: DFFITS with High Leverage and High Residual", "[diagnostics][dffits]") {
	// Create data with high leverage point that also has high residual
	Eigen::VectorXd y(5);
	Eigen::MatrixXd X(5, 1);
	
	X << 1.0, 2.0, 3.0, 4.0, 100.0;  // High leverage
	y << 3.0, 5.0, 7.0, 9.0, 50.0;   // High residual for last point
	
	RegressionOptions opts;
	opts.intercept = true;
	
	auto result = OLSSolver::Fit(y, X, opts);
	REQUIRE(result.rank > 0);
	REQUIRE(result.is_valid());
	
	Eigen::VectorXd leverage = RegressionDiagnostics::ComputeLeverage(
		X, result.is_aliased, true);
	
	Eigen::VectorXd std_residuals = RegressionDiagnostics::ComputeStandardizedResiduals(
		result.residuals, leverage, result.mse);
	
	Eigen::VectorXd dffits = RegressionDiagnostics::ComputeDFFITS(
		std_residuals, leverage);
	
	// Last observation should have high |DFFITS|
	REQUIRE(std::abs(dffits(4)) > std::abs(dffits(0)));
}

TEST_CASE("Diagnostics: ComputeAllDiagnostics", "[diagnostics][all]") {
	Eigen::VectorXd y(5);
	Eigen::MatrixXd X(5, 1);
	
	X << 1.0, 2.0, 3.0, 4.0, 5.0;
	y << 3.0, 5.0, 7.0, 9.0, 11.0;
	
	RegressionOptions opts;
	opts.intercept = true;
	
	auto result = OLSSolver::Fit(y, X, opts);
	REQUIRE(result.rank > 0);
	REQUIRE(result.is_valid());
	
	// Compute all diagnostics at once
	auto diag = RegressionDiagnostics::ComputeAllDiagnostics(result, X, true);
	
	REQUIRE(diag.has_diagnostics);
	REQUIRE(diag.leverage.size() == 5);
	REQUIRE(diag.standardized_residuals.size() == 5);
	REQUIRE(diag.cooks_distance.size() == 5);
	REQUIRE(diag.dffits.size() == 5);
	
	// Verify all diagnostics are computed correctly
	for (int i = 0; i < 5; i++) {
		REQUIRE_FALSE(std::isnan(diag.leverage(i)));
		REQUIRE_FALSE(std::isnan(diag.standardized_residuals(i)));
		REQUIRE_FALSE(std::isnan(diag.cooks_distance(i)));
		REQUIRE_FALSE(std::isnan(diag.dffits(i)));
	}
	
	// Verify leverage sum property
	double leverage_sum = diag.leverage.sum();
	REQUIRE_THAT(leverage_sum, Catch::Matchers::WithinAbs(2.0, TOLERANCE));
}

TEST_CASE("Diagnostics: Edge Case - Leverage >= 1.0", "[diagnostics][edge]") {
	// Create edge case where leverage might approach 1.0
	// This can happen with very few observations or extreme leverage
	Eigen::VectorXd y(3);
	Eigen::MatrixXd X(3, 1);
	
	X << 1.0, 1.0, 100.0;  // Two identical points + one extreme
	y << 2.0, 2.0, 200.0;
	
	RegressionOptions opts;
	opts.intercept = true;
	
	auto result = OLSSolver::Fit(y, X, opts);
	REQUIRE(result.rank > 0);
	REQUIRE(result.is_valid());
	
	Eigen::VectorXd leverage = RegressionDiagnostics::ComputeLeverage(
		X, result.is_aliased, true);
	
	// Should handle gracefully even if leverage is high
	for (int i = 0; i < 3; i++) {
		// Leverage should be <= 1.0 (or NaN if exactly 1.0)
		if (!std::isnan(leverage(i))) {
			REQUIRE(leverage(i) <= 1.0 + TOLERANCE);
		}
	}
}

TEST_CASE("Diagnostics: Edge Case - NaN Handling", "[diagnostics][edge]") {
	// Test with NaN in residuals
	Eigen::VectorXd residuals(5);
	residuals << 0.1, 0.2, std::numeric_limits<double>::quiet_NaN(), 0.4, 0.5;
	
	Eigen::VectorXd leverage(5);
	leverage << 0.2, 0.2, 0.2, 0.2, 0.2;
	
	double mse = 0.1;
	
	Eigen::VectorXd std_residuals = RegressionDiagnostics::ComputeStandardizedResiduals(
		residuals, leverage, mse);
	
	// NaN residual should produce NaN standardized residual
	REQUIRE(std::isnan(std_residuals(2)));
	
	// Other residuals should be computed
	REQUIRE_FALSE(std::isnan(std_residuals(0)));
	REQUIRE_FALSE(std::isnan(std_residuals(1)));
}

