#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <libanostat/solvers/ridge_solver.hpp>
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

TEST_CASE("Ridge: Lambda 0.1", "[ridge][validation]") {
	// Load test data
	TestData data = load_csv("../../../test/data/ridge_tests/input/ridge_lambda_0.1.csv");
	json expected = load_expected_json("../../../test/data/ridge_tests/expected/ridge_lambda_0.1.json");

	// Fit model with Ridge regularization
	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;
	opts.lambda = 0.1;

	auto result = RidgeSolver::Fit(data.X, data.y, opts);

	REQUIRE(result.success);

	// Check coefficients (Ridge coefficients will be shrunk compared to OLS)
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

	// Check fitted values (spot check first 10)
	if (expected.contains("fitted_values")) {
		auto expected_fitted = expected["fitted_values"].get<std::vector<double>>();
		REQUIRE(result.fitted_values.size() == expected_fitted.size());

		for (size_t i = 0; i < std::min(size_t(10), expected_fitted.size()); i++) {
			REQUIRE_THAT(result.fitted_values(i),
			             Catch::Matchers::WithinAbs(expected_fitted[i], TOLERANCE));
		}
	}
}

TEST_CASE("Ridge: Lambda 1.0", "[ridge][validation]") {
	// Load test data
	TestData data = load_csv("../../../test/data/ridge_tests/input/ridge_lambda_1.0.csv");
	json expected = load_expected_json("../../../test/data/ridge_tests/expected/ridge_lambda_1.0.json");

	// Fit model with stronger Ridge regularization
	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;
	opts.lambda = 1.0;

	auto result = RidgeSolver::Fit(data.X, data.y, opts);

	REQUIRE(result.success);

	// Check coefficients (more shrinkage with larger lambda)
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

	// Verify that lambda=1.0 produces more shrinkage than lambda=0.1
	// (this is tested indirectly by matching R's glmnet results)
}

TEST_CASE("Ridge: Lambda 0 equals OLS", "[ridge][validation]") {
	// Ridge with lambda=0 should equal OLS
	TestData data = load_csv("../../../test/data/ols_tests/input/simple_linear.csv");

	// Fit Ridge with lambda=0
	core::RegressionOptions ridge_opts;
	ridge_opts.intercept = true;
	ridge_opts.compute_inference = false;
	ridge_opts.lambda = 0.0;

	auto ridge_result = RidgeSolver::Fit(data.X, data.y, ridge_opts);

	// Fit OLS for comparison
	core::RegressionOptions ols_opts;
	ols_opts.intercept = true;
	ols_opts.compute_inference = false;

	auto ols_result = OLSSolver::Fit(data.X, data.y, ols_opts);

	REQUIRE(ridge_result.success);
	REQUIRE(ols_result.success);

	// Coefficients should be very close (small numerical differences allowed)
	REQUIRE(ridge_result.coefficients.size() == ols_result.coefficients.size());

	for (size_t i = 0; i < ols_result.coefficients.size(); i++) {
		REQUIRE_THAT(ridge_result.coefficients(i),
		             Catch::Matchers::WithinAbs(ols_result.coefficients(i), 1e-4));
	}

	// R² should also be very close
	REQUIRE_THAT(ridge_result.r_squared,
	             Catch::Matchers::WithinAbs(ols_result.r_squared, 1e-4));
}

TEST_CASE("Ridge: Handles Multicollinearity Better Than OLS", "[ridge][validation]") {
	// Load data with perfect collinearity
	TestData data = load_csv("../../../test/data/ols_tests/input/perfect_collinearity.csv");

	// Ridge should handle this better than OLS
	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;
	opts.lambda = 0.5;

	auto result = RidgeSolver::Fit(data.X, data.y, opts);

	REQUIRE(result.success);

	// Ridge doesn't set coefficients to NaN even with collinearity
	for (size_t i = 0; i < result.coefficients.size(); i++) {
		REQUIRE_FALSE(std::isnan(result.coefficients(i)));
		REQUIRE_FALSE(std::isinf(result.coefficients(i)));
	}

	// All coefficients should be finite and reasonable
	for (size_t i = 0; i < result.coefficients.size(); i++) {
		REQUIRE(std::abs(result.coefficients(i)) < 1000.0);  // No extreme values
	}
}

TEST_CASE("Ridge: No Intercept", "[ridge][validation]") {
	// Test Ridge without intercept
	TestData data = load_csv("../../../test/data/ols_tests/input/no_intercept.csv");

	core::RegressionOptions opts;
	opts.intercept = false;
	opts.compute_inference = false;
	opts.lambda = 0.5;

	auto result = RidgeSolver::Fit(data.X, data.y, opts);

	REQUIRE(result.success);

	// Should have p coefficients (no intercept)
	REQUIRE(result.coefficients.size() == data.n_features);

	// All coefficients should be finite
	for (size_t i = 0; i < result.coefficients.size(); i++) {
		REQUIRE_FALSE(std::isnan(result.coefficients(i)));
		REQUIRE_FALSE(std::isinf(result.coefficients(i)));
	}
}

TEST_CASE("Ridge: Constant Feature", "[ridge][constant]") {
	// Test Ridge with constant feature (lambda > 0 should still handle it)
	std::vector<std::vector<double>> X(2);
	X[0] = {1.0, 2.0, 3.0, 4.0, 5.0};
	X[1] = {5.0, 5.0, 5.0, 5.0, 5.0};  // Constant feature
	std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0};

	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;
	opts.lambda = 0.1;

	auto result = RidgeSolver::Fit(X, y, opts);

	REQUIRE(result.success);
	
	// Ridge should handle constant feature (may mark as aliased or shrink to near-zero)
	// All coefficients should be finite (Ridge doesn't produce NaN)
	for (size_t i = 0; i < result.coefficients.size(); i++) {
		REQUIRE_FALSE(std::isnan(result.coefficients(i)));
		REQUIRE_FALSE(std::isinf(result.coefficients(i)));
	}
}

TEST_CASE("Ridge: Constant Feature with Lambda 0", "[ridge][constant]") {
	// Ridge with lambda=0 should behave like OLS
	std::vector<std::vector<double>> X(2);
	X[0] = {1.0, 2.0, 3.0, 4.0, 5.0};
	X[1] = {5.0, 5.0, 5.0, 5.0, 5.0};  // Constant feature
	std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0};

	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;
	opts.lambda = 0.0;

	auto result = RidgeSolver::Fit(X, y, opts);

	REQUIRE(result.success);
	
	// With lambda=0, should behave like OLS (may have aliased features)
	// But coefficients should still be finite
	for (size_t i = 0; i < result.coefficients.size(); i++) {
		REQUIRE_FALSE(std::isinf(result.coefficients(i)));
	}
}
