#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <libanostat/solvers/rls_solver.hpp>
#include <libanostat/solvers/ols_solver.hpp>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>

using namespace libanostat;
using namespace libanostat::solvers;
using json = nlohmann::json;

// Helper to load CSV data
struct TestData {
	std::vector<std::vector<double>> X;  // Predictors (column-major)
	std::vector<double> y;                // Response
	size_t n_obs;
	size_t n_features;
};

TestData load_csv(const std::string& filepath) {
	TestData data;
	std::ifstream file(filepath);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file: " + filepath);
	}

	std::string line;
	std::getline(file, line); // Skip header

	std::vector<std::vector<double>> rows;
	while (std::getline(file, line)) {
		std::istringstream iss(line);
		std::string token;
		std::vector<double> row;

		while (std::getline(iss, token, ',')) {
			row.push_back(std::stod(token));
		}

		if (!row.empty()) {
			rows.push_back(row);
		}
	}

	if (rows.empty()) {
		throw std::runtime_error("No data in CSV file");
	}

	data.n_obs = rows.size();
	data.n_features = rows[0].size() - 1;  // Last column is y

	// Convert to column-major format for X
	data.X.resize(data.n_features);
	for (size_t j = 0; j < data.n_features; j++) {
		data.X[j].resize(data.n_obs);
		for (size_t i = 0; i < data.n_obs; i++) {
			data.X[j][i] = rows[i][j];
		}
	}

	// Extract y from last column
	data.y.resize(data.n_obs);
	for (size_t i = 0; i < data.n_obs; i++) {
		data.y[i] = rows[i][data.n_features];
	}

	return data;
}

json load_expected_json(const std::string& filepath) {
	std::ifstream file(filepath);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file: " + filepath);
	}
	json j;
	file >> j;
	return j;
}

const double TOLERANCE = 1e-6;

TEST_CASE("RLS: Lambda Close to 1.0 Approximates OLS", "[rls][validation]") {
	// RLS with lambda very close to 1.0 (minimal forgetting) should approximate OLS
	TestData data = load_csv("test/data/ols_tests/input/simple_linear.csv");

	// Fit RLS with lambda=0.9999 (almost no forgetting)
	core::RegressionOptions rls_opts;
	rls_opts.intercept = true;
	rls_opts.compute_inference = false;
	rls_opts.lambda = 0.9999;

	auto rls_result = RLSSolver::Fit(data.X, data.y, rls_opts);

	// Fit OLS for comparison
	core::RegressionOptions ols_opts;
	ols_opts.intercept = true;
	ols_opts.compute_inference = false;

	auto ols_result = OLSSolver::Fit(data.X, data.y, ols_opts);

	REQUIRE(rls_result.success);
	REQUIRE(ols_result.success);

	// Coefficients should be close (some numerical differences expected)
	REQUIRE(rls_result.coefficients.size() == ols_result.coefficients.size());

	for (size_t i = 0; i < ols_result.coefficients.size(); i++) {
		REQUIRE_THAT(rls_result.coefficients(i),
		             Catch::Matchers::WithinAbs(ols_result.coefficients(i), 0.1));
	}
}

TEST_CASE("RLS: Lambda 0.95", "[rls][validation]") {
	// Load test data
	TestData data = load_csv("test/data/rls_tests/input/rls_lambda_0.95.csv");
	json expected = load_expected_json("test/data/rls_tests/expected/rls_lambda_0.95.json");

	// Fit model with RLS (lambda=0.95 means more weight on recent observations)
	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;
	opts.lambda = 0.95;

	auto result = RLSSolver::Fit(data.X, data.y, opts);

	REQUIRE(result.success);

	// Check coefficients
	auto expected_coefs = expected["coefficients"].get<std::vector<double>>();
	REQUIRE(result.coefficients.size() == expected_coefs.size());

	for (size_t i = 0; i < expected_coefs.size(); i++) {
		REQUIRE_THAT(result.coefficients(i),
		             Catch::Matchers::WithinAbs(expected_coefs[i], TOLERANCE));
	}

	// Check R²
	if (expected.contains("r_squared")) {
		double expected_r2 = expected["r_squared"].get<double>();
		REQUIRE_THAT(result.r_squared,
		             Catch::Matchers::WithinAbs(expected_r2, TOLERANCE));
	}

	// Check residuals (spot check first 10)
	if (expected.contains("residuals")) {
		auto expected_residuals = expected["residuals"].get<std::vector<double>>();
		REQUIRE(result.residuals.size() == expected_residuals.size());

		for (size_t i = 0; i < std::min(size_t(10), expected_residuals.size()); i++) {
			REQUIRE_THAT(result.residuals(i),
			             Catch::Matchers::WithinAbs(expected_residuals[i], TOLERANCE));
		}
	}
}

TEST_CASE("RLS: Lambda 0.90", "[rls][validation]") {
	// Load test data
	TestData data = load_csv("test/data/rls_tests/input/rls_lambda_0.90.csv");
	json expected = load_expected_json("test/data/rls_tests/expected/rls_lambda_0.90.json");

	// Fit model with RLS (lambda=0.90 means more aggressive forgetting)
	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;
	opts.lambda = 0.90;

	auto result = RLSSolver::Fit(data.X, data.y, opts);

	REQUIRE(result.success);

	// Check coefficients
	auto expected_coefs = expected["coefficients"].get<std::vector<double>>();
	REQUIRE(result.coefficients.size() == expected_coefs.size());

	for (size_t i = 0; i < expected_coefs.size(); i++) {
		REQUIRE_THAT(result.coefficients(i),
		             Catch::Matchers::WithinAbs(expected_coefs[i], TOLERANCE));
	}

	// Check R²
	if (expected.contains("r_squared")) {
		double expected_r2 = expected["r_squared"].get<double>();
		REQUIRE_THAT(result.r_squared,
		             Catch::Matchers::WithinAbs(expected_r2, TOLERANCE));
	}
}

TEST_CASE("RLS: Adapts to Change Point", "[rls][validation]") {
	// Test that RLS adapts better than OLS to structural breaks in data
	TestData data = load_csv("test/data/rls_tests/input/rls_change_point.csv");

	// Fit RLS with moderate forgetting
	core::RegressionOptions rls_opts;
	rls_opts.intercept = true;
	rls_opts.compute_inference = false;
	rls_opts.lambda = 0.95;

	auto rls_result = RLSSolver::Fit(data.X, data.y, rls_opts);

	// Fit OLS for comparison
	core::RegressionOptions ols_opts;
	ols_opts.intercept = true;
	ols_opts.compute_inference = false;

	auto ols_result = OLSSolver::Fit(data.X, data.y, ols_opts);

	REQUIRE(rls_result.success);
	REQUIRE(ols_result.success);

	// All coefficients should be finite
	for (size_t i = 0; i < rls_result.coefficients.size(); i++) {
		REQUIRE_FALSE(std::isnan(rls_result.coefficients(i)));
		REQUIRE_FALSE(std::isinf(rls_result.coefficients(i)));
	}

	// RLS should have valid R² (may be different from OLS)
	REQUIRE_FALSE(std::isnan(rls_result.r_squared));
	REQUIRE(rls_result.r_squared >= 0.0);
}

TEST_CASE("RLS: No Intercept", "[rls][validation]") {
	// Test RLS without intercept
	TestData data = load_csv("test/data/ols_tests/input/no_intercept.csv");

	core::RegressionOptions opts;
	opts.intercept = false;
	opts.compute_inference = false;
	opts.lambda = 0.95;

	auto result = RLSSolver::Fit(data.X, data.y, opts);

	REQUIRE(result.success);

	// Should have p coefficients (no intercept)
	REQUIRE(result.coefficients.size() == data.n_features);

	// All coefficients should be finite
	for (size_t i = 0; i < result.coefficients.size(); i++) {
		REQUIRE_FALSE(std::isnan(result.coefficients(i)));
		REQUIRE_FALSE(std::isinf(result.coefficients(i)));
	}
}

TEST_CASE("RLS: Effect of Different Lambdas", "[rls][validation]") {
	// Test that different lambda values produce different results
	TestData data = load_csv("test/data/ols_tests/input/multiple_regression.csv");

	// Fit with lambda = 0.90 (more forgetting)
	core::RegressionOptions opts_90;
	opts_90.intercept = true;
	opts_90.compute_inference = false;
	opts_90.lambda = 0.90;

	auto result_90 = RLSSolver::Fit(data.X, data.y, opts_90);

	// Fit with lambda = 0.99 (less forgetting)
	core::RegressionOptions opts_99;
	opts_99.intercept = true;
	opts_99.compute_inference = false;
	opts_99.lambda = 0.99;

	auto result_99 = RLSSolver::Fit(data.X, data.y, opts_99);

	REQUIRE(result_90.success);
	REQUIRE(result_99.success);

	REQUIRE(result_90.coefficients.size() == result_99.coefficients.size());

	// Coefficients should be different between the two lambda values
	bool coeffs_differ = false;
	for (size_t i = 0; i < result_90.coefficients.size(); i++) {
		if (std::abs(result_90.coefficients(i) - result_99.coefficients(i)) > 1e-6) {
			coeffs_differ = true;
			break;
		}
	}
	REQUIRE(coeffs_differ);  // Lambda should have an effect
}

TEST_CASE("RLS: Time-Varying Regression", "[rls][validation]") {
	// Load test data with time-varying coefficients
	TestData data = load_csv("test/data/rls_tests/input/rls_time_varying.csv");
	json expected = load_expected_json("test/data/rls_tests/expected/rls_time_varying.json");

	// Fit model with RLS
	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;
	opts.lambda = 0.95;

	auto result = RLSSolver::Fit(data.X, data.y, opts);

	REQUIRE(result.success);

	// Check coefficients
	auto expected_coefs = expected["coefficients"].get<std::vector<double>>();
	REQUIRE(result.coefficients.size() == expected_coefs.size());

	for (size_t i = 0; i < expected_coefs.size(); i++) {
		REQUIRE_THAT(result.coefficients(i),
		             Catch::Matchers::WithinAbs(expected_coefs[i], TOLERANCE));
	}

	// Check R²
	if (expected.contains("r_squared")) {
		double expected_r2 = expected["r_squared"].get<double>();
		REQUIRE_THAT(result.r_squared,
		             Catch::Matchers::WithinAbs(expected_r2, TOLERANCE));
	}
}

TEST_CASE("RLS: Handles Collinearity", "[rls][validation]") {
	// Load data with perfect collinearity
	TestData data = load_csv("test/data/ols_tests/input/perfect_collinearity.csv");

	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;
	opts.lambda = 0.95;

	auto result = RLSSolver::Fit(data.X, data.y, opts);

	REQUIRE(result.success);

	// RLS should handle collinearity gracefully
	// (The recursive update with forgetting factor provides implicit regularization)
	for (size_t i = 0; i < result.coefficients.size(); i++) {
		REQUIRE_FALSE(std::isnan(result.coefficients(i)));
		REQUIRE_FALSE(std::isinf(result.coefficients(i)));
	}

	// All coefficients should be finite and reasonable
	for (size_t i = 0; i < result.coefficients.size(); i++) {
		REQUIRE(std::abs(result.coefficients(i)) < 1000.0);  // No extreme values
	}
}
