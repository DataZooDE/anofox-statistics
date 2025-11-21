#include <catch2/catch_all.hpp>
#include "libanostat/core/regression_result.hpp"
#include "libanostat/core/regression_options.hpp"
#include "libanostat/core/inference_result.hpp"

using namespace libanostat::core;

TEST_CASE("RegressionResult - Basic construction", "[core][result]") {
	SECTION("Default constructor") {
		RegressionResult result;
		REQUIRE(result.rank == 0);
		REQUIRE(result.n_params == 0);
		REQUIRE(result.n_obs == 0);
		REQUIRE(!result.has_std_errors);
	}

	SECTION("Constructor with dimensions") {
		RegressionResult result(100, 5, 4);
		REQUIRE(result.n_obs == 100);
		REQUIRE(result.n_params == 5);
		REQUIRE(result.rank == 4);
		REQUIRE(result.coefficients.size() == 5);
		REQUIRE(result.residuals.size() == 100);
		REQUIRE(result.is_aliased.size() == 5);
		REQUIRE(result.permutation_indices.size() == 5);
	}

	SECTION("Degrees of freedom calculations") {
		RegressionResult result(100, 5, 4);
		REQUIRE(result.df_model() == 4);
		REQUIRE(result.df_residual() == 96);
	}
}

TEST_CASE("RegressionOptions - Default values", "[core][options]") {
	RegressionOptions opts;

	SECTION("Defaults are sensible") {
		REQUIRE(opts.intercept == true);
		REQUIRE(opts.full_output == false);
		REQUIRE(opts.lambda == 0.0);
		REQUIRE(opts.alpha == 0.0);
		REQUIRE(opts.forgetting_factor == 1.0);
		REQUIRE(opts.confidence_level == 0.95);
		REQUIRE(opts.max_iterations == 1000);
		REQUIRE(opts.tolerance == 1e-6);
		REQUIRE(opts.solver == "qr");
	}

	SECTION("Validation passes for defaults") {
		REQUIRE_NOTHROW(opts.Validate());
	}
}

TEST_CASE("RegressionOptions - Convenience constructors", "[core][options]") {
	SECTION("OLS") {
		auto opts = RegressionOptions::OLS();
		REQUIRE(opts.intercept == true);
		REQUIRE(opts.lambda == 0.0);
		REQUIRE(opts.alpha == 0.0);
	}

	SECTION("Ridge") {
		auto opts = RegressionOptions::Ridge(0.1);
		REQUIRE(opts.lambda == 0.1);
		REQUIRE(opts.alpha == 0.0);
		REQUIRE_NOTHROW(opts.Validate());
	}

	SECTION("Lasso") {
		auto opts = RegressionOptions::Lasso(0.1);
		REQUIRE(opts.lambda == 0.1);
		REQUIRE(opts.alpha == 1.0);
		REQUIRE_NOTHROW(opts.Validate());
	}

	SECTION("Elastic Net") {
		auto opts = RegressionOptions::ElasticNet(0.1, 0.5);
		REQUIRE(opts.lambda == 0.1);
		REQUIRE(opts.alpha == 0.5);
		REQUIRE_NOTHROW(opts.Validate());
	}

	SECTION("RLS") {
		auto opts = RegressionOptions::RLS(0.9);
		REQUIRE(opts.forgetting_factor == 0.9);
		REQUIRE_NOTHROW(opts.Validate());
	}
}

TEST_CASE("RegressionOptions - Validation catches errors", "[core][options]") {
	SECTION("Negative lambda") {
		RegressionOptions opts;
		opts.lambda = -0.1;
		REQUIRE_THROWS_AS(opts.Validate(), std::invalid_argument);
	}

	SECTION("Alpha out of range") {
		RegressionOptions opts;
		opts.alpha = 1.5;
		REQUIRE_THROWS_AS(opts.Validate(), std::invalid_argument);
	}

	SECTION("Invalid forgetting factor") {
		RegressionOptions opts;
		opts.forgetting_factor = 1.5;
		REQUIRE_THROWS_AS(opts.Validate(), std::invalid_argument);
	}

	SECTION("Invalid confidence level") {
		RegressionOptions opts;
		opts.confidence_level = 1.5;
		REQUIRE_THROWS_AS(opts.Validate(), std::invalid_argument);
	}

	SECTION("Alpha without lambda") {
		RegressionOptions opts;
		opts.alpha = 0.5;
		opts.lambda = 0.0;
		REQUIRE_THROWS_AS(opts.Validate(), std::invalid_argument);
	}
}

TEST_CASE("InferenceResult - Basic construction", "[core][inference]") {
	SECTION("Default constructor") {
		InferenceResult result;
		REQUIRE(!result.has_coefficient_inference);
		REQUIRE(!result.has_prediction_intervals);
	}

	SECTION("With coefficient inference") {
		auto result = InferenceResult::WithCoefficientInference(5, 0.95);
		REQUIRE(result.has_coefficient_inference);
		REQUIRE(!result.has_prediction_intervals);
		REQUIRE(result.coefficient_inference.std_errors.size() == 5);
		REQUIRE(result.coefficient_inference.confidence_level == 0.95);
	}

	SECTION("With prediction intervals") {
		auto result = InferenceResult::WithPredictionIntervals(100, 0.99);
		REQUIRE(!result.has_coefficient_inference);
		REQUIRE(result.has_prediction_intervals);
		REQUIRE(result.prediction_intervals.predictions.size() == 100);
		REQUIRE(result.prediction_intervals.confidence_level == 0.99);
	}

	SECTION("With both") {
		auto result = InferenceResult::WithBoth(5, 100, 0.90);
		REQUIRE(result.has_coefficient_inference);
		REQUIRE(result.has_prediction_intervals);
		REQUIRE(result.coefficient_inference.confidence_level == 0.90);
		REQUIRE(result.prediction_intervals.confidence_level == 0.90);
	}
}
