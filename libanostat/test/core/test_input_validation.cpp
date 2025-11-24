#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <libanostat/core/regression_options.hpp>
#include <libanostat/solvers/ols_solver.hpp>
#include <libanostat/solvers/ridge_solver.hpp>
#include <libanostat/solvers/elastic_net_solver.hpp>
#include <libanostat/utils/distributions.hpp>
#include <Eigen/Dense>

using namespace libanostat;
using namespace libanostat::core;
using namespace libanostat::solvers;
using namespace libanostat::utils;

TEST_CASE("Input Validation: RegressionOptions - Invalid Tolerance", "[validation][options]") {
	RegressionOptions opts;
	opts.tolerance = 0.0;

	REQUIRE_THROWS_AS(opts.Validate(), std::invalid_argument);

	opts.tolerance = -0.1;
	REQUIRE_THROWS_AS(opts.Validate(), std::invalid_argument);
}

TEST_CASE("Input Validation: RegressionOptions - Invalid Max Iterations", "[validation][options]") {
	RegressionOptions opts;
	opts.max_iterations = 0;

	REQUIRE_THROWS_AS(opts.Validate(), std::invalid_argument);
}

TEST_CASE("Input Validation: RegressionOptions - Invalid Solver", "[validation][options]") {
	RegressionOptions opts;
	opts.solver = "invalid_solver";

	REQUIRE_THROWS_AS(opts.Validate(), std::invalid_argument);

	opts.solver = "svd_wrong";
	REQUIRE_THROWS_AS(opts.Validate(), std::invalid_argument);
}

TEST_CASE("Input Validation: Ridge - Negative Lambda", "[validation][ridge]") {
	Eigen::MatrixXd X(10, 2);
	X.setRandom();
	Eigen::VectorXd y(10);
	y.setRandom();

	RegressionOptions opts = RegressionOptions::Ridge(-1.0);

	REQUIRE_THROWS_AS(RidgeSolver::Fit(y, X, opts), std::invalid_argument);
}

TEST_CASE("Input Validation: Elastic Net - Negative Lambda", "[validation][elastic_net]") {
	Eigen::MatrixXd X(10, 2);
	X.setRandom();
	Eigen::VectorXd y(10);
	y.setRandom();

	RegressionOptions opts = RegressionOptions::ElasticNet(-1.0, 0.5);

	REQUIRE_THROWS_AS(ElasticNetSolver::Fit(y, X, opts), std::invalid_argument);
}

TEST_CASE("Input Validation: Elastic Net - Invalid Alpha (Below Zero)", "[validation][elastic_net]") {
	Eigen::MatrixXd X(10, 2);
	X.setRandom();
	Eigen::VectorXd y(10);
	y.setRandom();

	RegressionOptions opts = RegressionOptions::ElasticNet(1.0, -0.1);

	REQUIRE_THROWS_AS(ElasticNetSolver::Fit(y, X, opts), std::invalid_argument);
}

TEST_CASE("Input Validation: Elastic Net - Invalid Alpha (Above One)", "[validation][elastic_net]") {
	Eigen::MatrixXd X(10, 2);
	X.setRandom();
	Eigen::VectorXd y(10);
	y.setRandom();

	RegressionOptions opts = RegressionOptions::ElasticNet(1.0, 1.5);

	REQUIRE_THROWS_AS(ElasticNetSolver::Fit(y, X, opts), std::invalid_argument);
}

TEST_CASE("Input Validation: Student t CDF - Invalid Degrees of Freedom", "[validation][distributions]") {
	// df <= 0 should return 0.5 (not throw)
	double result = student_t_cdf(1.0, 0);
	REQUIRE(result == 0.5);

	result = student_t_cdf(1.0, -5);
	REQUIRE(result == 0.5);
}

TEST_CASE("Input Validation: Valid Parameters Should Pass", "[validation][positive]") {
	// Ensure valid parameters don't throw
	RegressionOptions opts;
	opts.tolerance = 1e-6;
	opts.max_iterations = 1000;
	opts.solver = "qr";

	REQUIRE_NOTHROW(opts.Validate());

	opts.solver = "svd";
	REQUIRE_NOTHROW(opts.Validate());

	opts.solver = "cholesky";
	REQUIRE_NOTHROW(opts.Validate());
}
