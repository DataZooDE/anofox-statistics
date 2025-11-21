#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "libanostat/solvers/i_regressor.hpp"
#include "libanostat/solvers/ols_solver.hpp"
#include "libanostat/solvers/ridge_solver.hpp"
#include <memory>

using namespace libanostat;
using namespace libanostat::solvers;

TEST_CASE("IRegressor interface - OLS adapter", "[i_regressor][ols]") {
	// Create OLS solver via adapter
	auto ols = std::make_unique<RegressorAdapter<OLSSolver>>(
	    "OLS",
	    "ols",
	    "Ordinary Least Squares - QR decomposition with rank deficiency handling",
	    "Simple linear regression without regularization",
	    true,  // supports std errors
	    true   // supports rank deficiency
	);

	SECTION("Adapter provides correct metadata") {
		REQUIRE(ols->GetName() == "OLS");
		REQUIRE(ols->GetType() == "ols");
		REQUIRE(ols->GetDescription() == "Ordinary Least Squares - QR decomposition with rank deficiency handling");
		REQUIRE(ols->GetUseCase() == "Simple linear regression without regularization");
		REQUIRE(ols->SupportsStdErrors() == true);
		REQUIRE(ols->SupportsRankDeficiency() == true);
	}

	SECTION("Adapter can fit simple linear regression") {
		// Create simple data: y = 2x + 1
		Eigen::VectorXd y(4);
		y << 3, 5, 7, 9;

		Eigen::MatrixXd X(4, 1);
		X << 1, 2, 3, 4;

		core::RegressionOptions options = core::RegressionOptions::OLS();
		options.intercept = true;

		auto result = ols->Fit(y, X, options);

		// With auto-intercept, we get 2 coefficients: intercept + slope
		REQUIRE(result.coefficients.size() == 2);
		REQUIRE_THAT(result.coefficients(0), Catch::Matchers::WithinAbs(1.0, 1e-10));  // intercept
		REQUIRE_THAT(result.coefficients(1), Catch::Matchers::WithinAbs(2.0, 1e-10));  // slope
		REQUIRE_THAT(result.r_squared, Catch::Matchers::WithinAbs(1.0, 1e-10));
	}

	SECTION("Adapter can compute standard errors") {
		Eigen::VectorXd y(4);
		y << 3, 5, 7, 9;

		Eigen::MatrixXd X(4, 1);
		X << 1, 2, 3, 4;

		core::RegressionOptions options = core::RegressionOptions::OLS();
		options.intercept = true;

		auto result = ols->FitWithStdErrors(y, X, options);

		REQUIRE(result.has_std_errors == true);
		// With auto-intercept, we get 2 standard errors: intercept + slope
		REQUIRE(result.std_errors.size() == 2);
		REQUIRE(result.std_errors(0) >= 0);  // Intercept SE should be non-negative
		REQUIRE(result.std_errors(1) >= 0);  // Slope SE should be non-negative
	}
}

TEST_CASE("IRegressor interface - Ridge adapter", "[i_regressor][ridge]") {
	// Create Ridge solver via adapter
	auto ridge = std::make_unique<RegressorAdapter<RidgeSolver>>(
	    "Ridge",
	    "ridge",
	    "Ridge Regression - L2 regularization to prevent overfitting",
	    "Multicollinearity issues or when preventing overfitting with L2 penalty",
	    true,  // supports std errors (approximate)
	    true   // supports rank deficiency
	);

	SECTION("Adapter provides correct metadata") {
		REQUIRE(ridge->GetName() == "Ridge");
		REQUIRE(ridge->GetType() == "ridge");
		REQUIRE(ridge->SupportsStdErrors() == true);
		REQUIRE(ridge->SupportsRankDeficiency() == true);
	}

	SECTION("Adapter can fit ridge regression") {
		Eigen::VectorXd y(4);
		y << 3, 5, 7, 9;

		Eigen::MatrixXd X(4, 1);
		X << 1, 2, 3, 4;

		core::RegressionOptions options = core::RegressionOptions::Ridge(0.1);
		options.intercept = true;

		auto result = ridge->Fit(y, X, options);

		// With auto-intercept, we get 2 coefficients: intercept + slope
		REQUIRE(result.coefficients.size() == 2);
		// With L2 penalty, slope coefficient should be shrunk (< 2.0)
		REQUIRE(result.coefficients(1) < 2.0);
		REQUIRE(result.coefficients(1) > 0.0);
	}
}

TEST_CASE("IRegressor interface - Polymorphic use", "[i_regressor][polymorphism]") {
	// Create vector of different solvers
	std::vector<std::unique_ptr<IRegressor>> solvers;

	solvers.push_back(std::make_unique<RegressorAdapter<OLSSolver>>(
	    "OLS", "ols",
	    "Ordinary Least Squares",
	    "Simple linear regression",
	    true, true
	));

	solvers.push_back(std::make_unique<RegressorAdapter<RidgeSolver>>(
	    "Ridge", "ridge",
	    "Ridge Regression",
	    "Multicollinearity",
	    true, true
	));

	SECTION("Can iterate through different solvers polymorphically") {
		Eigen::VectorXd y(4);
		y << 3, 5, 7, 9;

		Eigen::MatrixXd X(4, 1);
		X << 1, 2, 3, 4;

		for (const auto &solver : solvers) {
			// Each solver should have a name
			REQUIRE(!solver->GetName().empty());
			REQUIRE(!solver->GetType().empty());

			// Use appropriate options based on solver type
			core::RegressionOptions options;
			if (solver->GetType() == "ols") {
				options = core::RegressionOptions::OLS();
			} else if (solver->GetType() == "ridge") {
				options = core::RegressionOptions::Ridge(0.1);
			}
			options.intercept = true;

			// All solvers should be able to fit the data
			auto result = solver->Fit(y, X, options);
			// With auto-intercept, we get 2 coefficients: intercept + slope
			REQUIRE(result.coefficients.size() == 2);
			REQUIRE(result.coefficients(1) > 0);  // Positive slope expected

			// All solvers should support standard errors in this example
			if (solver->SupportsStdErrors()) {
				auto result_with_se = solver->FitWithStdErrors(y, X, options);
				REQUIRE(result_with_se.has_std_errors == true);
			}
		}
	}
}

TEST_CASE("IRegressor interface - Solver factory pattern", "[i_regressor][factory]") {
	// Factory function to create solvers by type
	auto create_solver = [](const std::string &type) -> std::unique_ptr<IRegressor> {
		if (type == "ols") {
			return std::make_unique<RegressorAdapter<OLSSolver>>(
			    "OLS", "ols", "Ordinary Least Squares", "Simple linear regression",
			    true, true
			);
		} else if (type == "ridge") {
			return std::make_unique<RegressorAdapter<RidgeSolver>>(
			    "Ridge", "ridge", "Ridge Regression", "Multicollinearity",
			    true, true
			);
		}
		return nullptr;
	};

	SECTION("Factory creates OLS solver") {
		auto solver = create_solver("ols");
		REQUIRE(solver != nullptr);
		REQUIRE(solver->GetType() == "ols");
		REQUIRE(solver->GetName() == "OLS");
	}

	SECTION("Factory creates Ridge solver") {
		auto solver = create_solver("ridge");
		REQUIRE(solver != nullptr);
		REQUIRE(solver->GetType() == "ridge");
		REQUIRE(solver->GetName() == "Ridge");
	}

	SECTION("Factory returns nullptr for unknown type") {
		auto solver = create_solver("unknown");
		REQUIRE(solver == nullptr);
	}

	SECTION("Factory-created solvers can fit data") {
		Eigen::VectorXd y(4);
		y << 3, 5, 7, 9;

		Eigen::MatrixXd X(4, 1);
		X << 1, 2, 3, 4;

		auto ols_solver = create_solver("ols");
		auto options = core::RegressionOptions::OLS();
		options.intercept = true;

		auto result = ols_solver->Fit(y, X, options);
		// With auto-intercept, we get 2 coefficients: intercept + slope
		REQUIRE(result.coefficients.size() == 2);
		REQUIRE_THAT(result.coefficients(0), Catch::Matchers::WithinAbs(1.0, 1e-10));  // intercept
		REQUIRE_THAT(result.coefficients(1), Catch::Matchers::WithinAbs(2.0, 1e-10));  // slope
	}
}
