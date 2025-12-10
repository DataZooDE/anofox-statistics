#include <catch2/catch_all.hpp>

// Test that all library headers compile correctly together
#include "libanostat/core/regression_result.hpp"
#include "libanostat/core/regression_options.hpp"
#include "libanostat/core/inference_result.hpp"
#include "libanostat/utils/distributions.hpp"
#include "libanostat/solvers/ols_solver.hpp"
#include "libanostat/solvers/ridge_solver.hpp"
#include "libanostat/solvers/elastic_net_solver.hpp"
#include "libanostat/solvers/wls_solver.hpp"
#include "libanostat/solvers/rls_solver.hpp"

using namespace libanostat;

TEST_CASE("Library headers compile together", "[compilation]") {
	// This test just verifies that all headers can be included together
	// without compilation errors or symbol conflicts

	SECTION("Can create core structures") {
		core::RegressionResult result;
		core::RegressionOptions options;
		core::InferenceResult inference;

		REQUIRE(result.rank == 0);
		REQUIRE(options.intercept == true);
		REQUIRE(!inference.has_coefficient_inference);
	}

	SECTION("Can use distribution functions") {
		double gamma = utils::log_gamma(5.0);
		double beta = utils::log_beta(2.0, 3.0);
		double t_cdf = utils::student_t_cdf(1.96, 100);

		REQUIRE(std::isfinite(gamma));
		REQUIRE(std::isfinite(beta));
		REQUIRE(std::isfinite(t_cdf));
		REQUIRE(t_cdf > 0.5); // 1.96 is positive, so CDF > 0.5
	}

	SECTION("Can instantiate solver classes") {
		// Just verify that solver classes exist and have expected methods
		// (actual functionality tested in dedicated solver tests)

		Eigen::VectorXd y(10);
		Eigen::MatrixXd X(10, 3);
		y.setRandom();
		X.setRandom();

		auto opts_ols = core::RegressionOptions::OLS();
		auto opts_ridge = core::RegressionOptions::Ridge(0.1);
		auto opts_enet = core::RegressionOptions::ElasticNet(0.1, 0.5);

		REQUIRE(opts_ols.lambda == 0.0);
		REQUIRE(opts_ridge.lambda == 0.1);
		REQUIRE(opts_enet.lambda == 0.1);
		REQUIRE(opts_enet.alpha == 0.5);

		// Verify solver methods exist (we won't actually call them here
		// to keep this test fast - detailed testing in solver-specific tests)
		using OLS = solvers::OLSSolver;
		using Ridge = solvers::RidgeSolver;
		using ElasticNet = solvers::ElasticNetSolver;
		using WLS = solvers::WLSSolver;
		using RLS = solvers::RLSSolver;

		// These lines just verify the methods exist and have correct signatures
		(void)&OLS::Fit;
		(void)&OLS::FitWithStdErrors;
		(void)&OLS::DetectConstantColumns;
		(void)&OLS::IsFullRank;

		(void)&Ridge::Fit;
		(void)&Ridge::FitWithStdErrors;

		(void)&ElasticNet::Fit;
		(void)&ElasticNet::FitWithStdErrors;

		(void)&WLS::Fit;
		(void)&WLS::FitWithStdErrors;

		(void)&RLS::Fit;
		(void)&RLS::FitWithStdErrors;
	}
}

TEST_CASE("No symbol conflicts between headers", "[compilation]") {
	// Verify that including all headers doesn't create symbol conflicts

	SECTION("RegressionResult from both locations") {
		core::RegressionResult result1;
		libanostat::core::RegressionResult result2;

		// Both should be the same type
		REQUIRE(sizeof(result1) == sizeof(result2));
	}

	SECTION("Distribution functions accessible via namespace") {
		// Verify functions can be called via full namespace path
		double val1 = libanostat::utils::log_gamma(3.0);
		double val2 = utils::log_gamma(3.0);

		REQUIRE(val1 == val2);
	}
}
