#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <libanostat/solvers/wls_solver.hpp>
#include <libanostat/solvers/ols_solver.hpp>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>

using namespace libanostat;
using namespace libanostat::solvers;
using json = nlohmann::json;

// Helper to load CSV data with weights
struct TestData {
	std::vector<std::vector<double>> X;  // Predictors (column-major)
	std::vector<double> y;                // Response
	std::vector<double> weights;          // Observation weights
	size_t n_obs;
	size_t n_features;
};

TestData load_csv_with_weights(const std::string& filepath) {
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
	// Last column is y, second-to-last is weights
	data.n_features = rows[0].size() - 2;

	// Convert to column-major format for X
	data.X.resize(data.n_features);
	for (size_t j = 0; j < data.n_features; j++) {
		data.X[j].resize(data.n_obs);
		for (size_t i = 0; i < data.n_obs; i++) {
			data.X[j][i] = rows[i][j];
		}
	}

	// Extract weights and y
	data.weights.resize(data.n_obs);
	data.y.resize(data.n_obs);
	for (size_t i = 0; i < data.n_obs; i++) {
		data.weights[i] = rows[i][data.n_features];      // Second-to-last column
		data.y[i] = rows[i][data.n_features + 1];         // Last column
	}

	return data;
}

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

TEST_CASE("WLS: Uniform Weights Equals OLS", "[wls][validation]") {
	// WLS with all weights = 1.0 should equal OLS
	TestData data = load_csv("../../../test/data/ols_tests/input/simple_linear.csv");

	// Create uniform weights
	std::vector<double> weights(data.n_obs, 1.0);

	// Fit WLS with uniform weights
	core::RegressionOptions wls_opts;
	wls_opts.intercept = true;
	wls_opts.compute_inference = false;

	auto wls_result = WLSSolver::Fit(data.X, data.y, weights, wls_opts);

	// Fit OLS for comparison
	core::RegressionOptions ols_opts;
	ols_opts.intercept = true;
	ols_opts.compute_inference = false;

	auto ols_result = OLSSolver::Fit(data.X, data.y, ols_opts);

	REQUIRE(wls_result.success);
	REQUIRE(ols_result.success);

	// Coefficients should be very close (small numerical differences allowed)
	REQUIRE(wls_result.coefficients.size() == ols_result.coefficients.size());

	for (size_t i = 0; i < ols_result.coefficients.size(); i++) {
		REQUIRE_THAT(wls_result.coefficients(i),
		             Catch::Matchers::WithinAbs(ols_result.coefficients(i), 1e-4));
	}

	// R² should also be very close
	REQUIRE_THAT(wls_result.r_squared,
	             Catch::Matchers::WithinAbs(ols_result.r_squared, 1e-4));
}

TEST_CASE("WLS: Heteroscedastic Weights", "[wls][validation]") {
	// Load test data with heteroscedastic variance structure
	TestData data = load_csv_with_weights("../../../test/data/wls_tests/input/wls_heteroscedastic.csv");
	json expected = load_expected_json("../../../test/data/wls_tests/expected/wls_heteroscedastic.json");

	// Fit model with WLS
	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;

	auto result = WLSSolver::Fit(data.X, data.y, data.weights, opts);

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

TEST_CASE("WLS: Downweight Outliers", "[wls][validation]") {
	// Load test data where some observations are outliers with low weights
	TestData data = load_csv_with_weights("../../../test/data/wls_tests/input/wls_outliers.csv");
	json expected = load_expected_json("../../../test/data/wls_tests/expected/wls_outliers.json");

	// Fit model with WLS
	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;

	auto result = WLSSolver::Fit(data.X, data.y, data.weights, opts);

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

TEST_CASE("WLS: No Intercept", "[wls][validation]") {
	// Test WLS without intercept
	TestData data = load_csv("../../../test/data/ols_tests/input/no_intercept.csv");

	// Create some varying weights
	std::vector<double> weights(data.n_obs);
	for (size_t i = 0; i < data.n_obs; i++) {
		weights[i] = 0.5 + 0.5 * (i % 3);  // Weights vary: 0.5, 1.0, 1.5
	}

	core::RegressionOptions opts;
	opts.intercept = false;
	opts.compute_inference = false;

	auto result = WLSSolver::Fit(data.X, data.y, weights, opts);

	REQUIRE(result.success);

	// Should have p coefficients (no intercept)
	REQUIRE(result.coefficients.size() == data.n_features);

	// All coefficients should be finite
	for (size_t i = 0; i < result.coefficients.size(); i++) {
		REQUIRE_FALSE(std::isnan(result.coefficients(i)));
		REQUIRE_FALSE(std::isinf(result.coefficients(i)));
	}
}

TEST_CASE("WLS: Zero Weights", "[wls][validation]") {
	// Test WLS with some zero weights (excluded observations)
	TestData data = load_csv("../../../test/data/ols_tests/input/simple_linear.csv");

	// Create weights where some observations are excluded
	std::vector<double> weights(data.n_obs, 1.0);
	weights[0] = 0.0;  // Exclude first observation
	weights[1] = 0.0;  // Exclude second observation

	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;

	auto result = WLSSolver::Fit(data.X, data.y, weights, opts);

	REQUIRE(result.success);

	// All coefficients should be finite
	for (size_t i = 0; i < result.coefficients.size(); i++) {
		REQUIRE_FALSE(std::isnan(result.coefficients(i)));
		REQUIRE_FALSE(std::isinf(result.coefficients(i)));
	}

	// Should still produce valid R²
	REQUIRE_FALSE(std::isnan(result.r_squared));
	REQUIRE(result.r_squared >= 0.0);
	REQUIRE(result.r_squared <= 1.0);
}

TEST_CASE("WLS: Variance Proportional to X", "[wls][validation]") {
	// Load test data where variance is proportional to predictor
	TestData data = load_csv_with_weights("../../../test/data/wls_tests/input/wls_variance_prop_x.csv");
	json expected = load_expected_json("../../../test/data/wls_tests/expected/wls_variance_prop_x.json");

	// Fit model with WLS
	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;

	auto result = WLSSolver::Fit(data.X, data.y, data.weights, opts);

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

TEST_CASE("WLS: Handles Collinearity", "[wls][validation]") {
	// Load data with perfect collinearity
	TestData data = load_csv("../../../test/data/ols_tests/input/perfect_collinearity.csv");

	// Create uniform weights
	std::vector<double> weights(data.n_obs, 1.0);

	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;

	auto result = WLSSolver::Fit(data.X, data.y, weights, opts);

	REQUIRE(result.success);

	// QR decomposition should detect collinearity
	REQUIRE(result.rank < data.n_features);

	// Verify residuals still make sense
	for (size_t i = 0; i < result.residuals.size(); i++) {
		REQUIRE_FALSE(std::isnan(result.residuals(i)));
		REQUIRE_FALSE(std::isinf(result.residuals(i)));
	}
}
