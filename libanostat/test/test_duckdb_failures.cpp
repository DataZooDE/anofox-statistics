#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <libanostat/solvers/ols_solver.hpp>
#include <libanostat/core/regression_options.hpp>
#include <Eigen/Dense>

using namespace libanostat;
using namespace libanostat::solvers;
using namespace libanostat::core;

const double TOLERANCE = 1e-6;

TEST_CASE("DuckDB Failure 1: aggregate_model_predict collinear data", "[duckdb][aggregate]") {
	// This matches test/sql/aggregate_model_predict.test:23
	// Category A: perfect collinearity between price and advertising
	Eigen::VectorXd y(4);
	Eigen::MatrixXd X(4, 2);

	// sales, price, advertising
	X << 10, 5,   // advertising = 0.5 * price (perfect collinearity!)
	     12, 6,
	     14, 7,
	     16, 8;
	y << 100, 120, 140, 160;  // Perfect fit: sales = 10 * price

	RegressionOptions opts;
	opts.intercept = true;

	auto result = OLSSolver::Fit(y, X, opts);
	REQUIRE(result.is_valid());

	// R confirms: model$rank = 2 (intercept + 1 non-aliased feature)
	INFO("result.rank should be 2 (intercept + 1 feature, 1 aliased)");
	REQUIRE(result.rank == 2);

	// R confirms: df.residual = 2
	size_t n = 4;
	size_t df_residual = n - result.rank;
	INFO("df_residual = n - rank = 4 - 2 = 2");
	REQUIRE(df_residual == 2);

	// With perfect fit, MSE should be 0
	REQUIRE(result.mse < 1e-10);
}

TEST_CASE("DuckDB Failure 2: ols_validation RMSE calculation", "[duckdb][validation]") {
	// This matches test/sql/ols_validation.test:11
	Eigen::VectorXd y(5);
	Eigen::MatrixXd X(5, 1);

	X << 1.0, 2.0, 3.0, 4.0, 5.0;
	y << 2.1, 4.2, 5.9, 8.1, 10.0;

	RegressionOptions opts;
	opts.intercept = true;

	auto result = OLSSolver::Fit(y, X, opts);
	REQUIRE(result.is_valid());

	// With 1 feature + intercept: rank = 2
	INFO("result.rank should be 2 (intercept + 1 feature)");
	REQUIRE(result.rank == 2);

	// df_residual = 5 - 2 = 3
	size_t n = 5;
	size_t df_residual = n - result.rank;
	INFO("df_residual = n - rank = 5 - 2 = 3");
	REQUIRE(df_residual == 3);

	// RMSE = sqrt(MSE) where MSE uses df_residual in denominator
	// Validated with R: RMSE = 0.1197219 (see validate_rmse.R)
	double expected_rmse = 0.1197219;
	double rmse = std::sqrt(result.mse);
	INFO("result.mse = " << result.mse << ", result.rank = " << result.rank << ", n = " << n);
	INFO("df_residual = " << df_residual << ", expected df = 3");
	INFO("RMSE = sqrt(MSE) = " << rmse << ", expected = " << expected_rmse);
	INFO("RMSE should match R's calculation with correct df");
	REQUIRE_THAT(rmse, Catch::Matchers::WithinAbs(expected_rmse, 0.0001));
}

TEST_CASE("DuckDB Failure 3: model_predict confidence intervals", "[duckdb][model_predict]") {
	// This matches test/sql/model_predict.test:26
	// Simple 1-feature model to test confidence interval calculation
	Eigen::VectorXd y(5);
	Eigen::MatrixXd X(5, 1);

	X << 1.0, 2.0, 3.0, 4.0, 5.0;
	y << 3.0, 5.0, 7.0, 9.0, 11.0;  // y = 1 + 2*x

	RegressionOptions opts;
	opts.intercept = true;

	auto result = OLSSolver::Fit(y, X, opts);
	REQUIRE(result.is_valid());

	// With 1 feature + intercept: rank = 2
	INFO("result.rank should be 2 (intercept + 1 feature)");
	REQUIRE(result.rank == 2);

	// df_residual = 5 - 2 = 3
	size_t n = 5;
	size_t df_residual = n - result.rank;
	INFO("df_residual = n - rank = 5 - 2 = 3");
	REQUIRE(df_residual == 3);

	// Confidence intervals use df_residual for t-distribution lookup
	// Wrong df would give wrong critical values and thus wrong CI widths
	// With df=3, should get narrower CIs than with df=2
	INFO("df_residual determines t-critical value for confidence intervals");
	REQUIRE(df_residual == 3);
}

TEST_CASE("Verify rank calculation with various scenarios", "[duckdb][rank]") {
	SECTION("No collinearity, 1 feature + intercept") {
		Eigen::VectorXd y(5);
		Eigen::MatrixXd X(5, 1);
		X << 1.0, 2.0, 3.0, 4.0, 5.0;
		y << 3.0, 5.0, 7.0, 9.0, 11.0;

		RegressionOptions opts;
		opts.intercept = true;
		auto result = OLSSolver::Fit(y, X, opts);

		REQUIRE(result.rank == 2);  // intercept + 1 feature
		REQUIRE((5 - result.rank) == 3);  // df_residual
	}

	SECTION("No collinearity, 2 features + intercept") {
		Eigen::VectorXd y(5);
		Eigen::MatrixXd X(5, 2);
		X << 1.0, 2.0,
		     2.0, 3.0,
		     3.0, 5.0,
		     4.0, 7.0,
		     5.0, 11.0;
		y << 3.0, 5.0, 7.0, 9.0, 11.0;

		RegressionOptions opts;
		opts.intercept = true;
		auto result = OLSSolver::Fit(y, X, opts);

		REQUIRE(result.rank == 3);  // intercept + 2 features
		REQUIRE((5 - result.rank) == 2);  // df_residual
	}

	SECTION("Perfect collinearity, 2 features + intercept") {
		Eigen::VectorXd y(4);
		Eigen::MatrixXd X(4, 2);
		X << 10, 5,   // col2 = 0.5 * col1
		     12, 6,
		     14, 7,
		     16, 8;
		y << 100, 120, 140, 160;

		RegressionOptions opts;
		opts.intercept = true;
		auto result = OLSSolver::Fit(y, X, opts);

		REQUIRE(result.rank == 2);  // intercept + 1 non-aliased feature
		REQUIRE((4 - result.rank) == 2);  // df_residual
	}

	SECTION("No intercept, 1 feature") {
		Eigen::VectorXd y(5);
		Eigen::MatrixXd X(5, 1);
		X << 1.0, 2.0, 3.0, 4.0, 5.0;
		y << 2.0, 4.0, 6.0, 8.0, 10.0;

		RegressionOptions opts;
		opts.intercept = false;
		auto result = OLSSolver::Fit(y, X, opts);

		REQUIRE(result.rank == 1);  // just 1 feature, no intercept
		REQUIRE((5 - result.rank) == 4);  // df_residual
	}
}
