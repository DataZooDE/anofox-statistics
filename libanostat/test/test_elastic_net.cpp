#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <libanostat/solvers/elastic_net_solver.hpp>
#include <libanostat/solvers/ols_solver.hpp>
#include <libanostat/solvers/ridge_solver.hpp>
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

TEST_CASE("Elastic Net: Alpha 0.5, Lambda 0.1", "[elastic_net][validation]") {
	// Load test data
	TestData data = load_csv("test/data/elastic_net_tests/input/elastic_net_alpha_0.5_lambda_0.1.csv");
	json expected = load_expected_json("test/data/elastic_net_tests/expected/elastic_net_alpha_0.5_lambda_0.1.json");

	// Fit model with Elastic Net regularization (alpha=0.5 is halfway between Ridge and Lasso)
	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;
	opts.lambda = 0.1;
	opts.alpha = 0.5;

	auto result = ElasticNetSolver::Fit(data.X, data.y, opts);

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

TEST_CASE("Elastic Net: Alpha 0.0 equals Ridge", "[elastic_net][validation]") {
	// Elastic Net with alpha=0 should equal Ridge regression
	TestData data = load_csv("test/data/ols_tests/input/simple_linear.csv");

	// Fit Elastic Net with alpha=0 (pure Ridge)
	core::RegressionOptions enet_opts;
	enet_opts.intercept = true;
	enet_opts.compute_inference = false;
	enet_opts.lambda = 0.5;
	enet_opts.alpha = 0.0;

	auto enet_result = ElasticNetSolver::Fit(data.X, data.y, enet_opts);

	// Fit Ridge for comparison
	core::RegressionOptions ridge_opts;
	ridge_opts.intercept = true;
	ridge_opts.compute_inference = false;
	ridge_opts.lambda = 0.5;

	auto ridge_result = RidgeSolver::Fit(data.X, data.y, ridge_opts);

	REQUIRE(enet_result.success);
	REQUIRE(ridge_result.success);

	// Coefficients should be very close (small numerical differences allowed)
	REQUIRE(enet_result.coefficients.size() == ridge_result.coefficients.size());

	for (size_t i = 0; i < ridge_result.coefficients.size(); i++) {
		REQUIRE_THAT(enet_result.coefficients(i),
		             Catch::Matchers::WithinAbs(ridge_result.coefficients(i), 1e-4));
	}

	// R² should also be very close
	REQUIRE_THAT(enet_result.r_squared,
	             Catch::Matchers::WithinAbs(ridge_result.r_squared, 1e-4));
}

TEST_CASE("Elastic Net: Alpha 1.0, Lambda 0.1", "[elastic_net][validation]") {
	// Load test data
	TestData data = load_csv("test/data/elastic_net_tests/input/elastic_net_alpha_1.0_lambda_0.1.csv");
	json expected = load_expected_json("test/data/elastic_net_tests/expected/elastic_net_alpha_1.0_lambda_0.1.json");

	// Fit model with pure Lasso (alpha=1.0)
	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;
	opts.lambda = 0.1;
	opts.alpha = 1.0;

	auto result = ElasticNetSolver::Fit(data.X, data.y, opts);

	REQUIRE(result.success);

	// Check coefficients (Lasso should produce sparse solutions)
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

TEST_CASE("Elastic Net: Variable Selection (Sparsity)", "[elastic_net][validation]") {
	// Test that Elastic Net with high lambda and high alpha produces sparse solutions
	TestData data = load_csv("test/data/elastic_net_tests/input/elastic_net_sparse.csv");

	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;
	opts.lambda = 1.0;   // High lambda
	opts.alpha = 0.9;    // High alpha (close to Lasso)

	auto result = ElasticNetSolver::Fit(data.X, data.y, opts);

	REQUIRE(result.success);

	// Count how many coefficients are effectively zero
	size_t n_zero = 0;
	for (size_t i = 0; i < result.coefficients.size(); i++) {
		if (std::abs(result.coefficients(i)) < 1e-6) {
			n_zero++;
		}
	}

	// Should have at least some zero coefficients (variable selection)
	REQUIRE(n_zero > 0);

	// All coefficients should be finite
	for (size_t i = 0; i < result.coefficients.size(); i++) {
		REQUIRE_FALSE(std::isnan(result.coefficients(i)));
		REQUIRE_FALSE(std::isinf(result.coefficients(i)));
	}
}

TEST_CASE("Elastic Net: No Intercept", "[elastic_net][validation]") {
	// Test Elastic Net without intercept
	TestData data = load_csv("test/data/ols_tests/input/no_intercept.csv");

	core::RegressionOptions opts;
	opts.intercept = false;
	opts.compute_inference = false;
	opts.lambda = 0.5;
	opts.alpha = 0.5;

	auto result = ElasticNetSolver::Fit(data.X, data.y, opts);

	REQUIRE(result.success);

	// Should have p coefficients (no intercept)
	REQUIRE(result.coefficients.size() == data.n_features);

	// All coefficients should be finite
	for (size_t i = 0; i < result.coefficients.size(); i++) {
		REQUIRE_FALSE(std::isnan(result.coefficients(i)));
		REQUIRE_FALSE(std::isinf(result.coefficients(i)));
	}
}

TEST_CASE("Elastic Net: Handles Multicollinearity", "[elastic_net][validation]") {
	// Load data with perfect collinearity
	TestData data = load_csv("test/data/ols_tests/input/perfect_collinearity.csv");

	// Elastic Net should handle this gracefully
	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;
	opts.lambda = 0.5;
	opts.alpha = 0.5;

	auto result = ElasticNetSolver::Fit(data.X, data.y, opts);

	REQUIRE(result.success);

	// Elastic Net doesn't set coefficients to NaN even with collinearity
	for (size_t i = 0; i < result.coefficients.size(); i++) {
		REQUIRE_FALSE(std::isnan(result.coefficients(i)));
		REQUIRE_FALSE(std::isinf(result.coefficients(i)));
	}

	// All coefficients should be finite and reasonable
	for (size_t i = 0; i < result.coefficients.size(); i++) {
		REQUIRE(std::abs(result.coefficients(i)) < 1000.0);  // No extreme values
	}
}
