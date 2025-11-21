#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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

TEST_CASE("OLS: Simple Linear Regression", "[ols][validation]") {
	// Load test data
	TestData data = load_csv("../../../test/data/ols_tests/input/simple_linear.csv");
	json expected = load_expected_json("../../../test/data/ols_tests/expected/simple_linear.json");

	// Fit model with intercept
	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;

	auto result = OLSSolver::Fit(data.X, data.y, opts);

	REQUIRE(result.success);

	// Check coefficients
	auto expected_coefs = expected["coefficients"].get<std::vector<double>>();
	REQUIRE(result.coefficients.size() == expected_coefs.size());

	for (size_t i = 0; i < expected_coefs.size(); i++) {
		REQUIRE_THAT(result.coefficients(i),
		             Catch::Matchers::WithinAbs(expected_coefs[i], TOLERANCE));
	}

	// Check R²
	double expected_r2 = expected["r_squared"].get<double>();
	REQUIRE_THAT(result.r_squared,
	             Catch::Matchers::WithinAbs(expected_r2, TOLERANCE));

	// Check adjusted R²
	double expected_adj_r2 = expected["adj_r_squared"].get<double>();
	REQUIRE_THAT(result.adj_r_squared,
	             Catch::Matchers::WithinAbs(expected_adj_r2, TOLERANCE));

	// Check sigma (residual standard error)
	double expected_sigma = expected["sigma"].get<double>();
	double sigma = std::sqrt(result.mse);
	REQUIRE_THAT(sigma,
	             Catch::Matchers::WithinAbs(expected_sigma, TOLERANCE));

	// Check degrees of freedom
	int expected_df = expected["df_residual"].get<int>();
	size_t n_params = result.rank + (opts.intercept ? 1 : 0);
	size_t df_residual = data.n_obs - n_params;
	REQUIRE(df_residual == static_cast<size_t>(expected_df));

	// Check residuals
	auto expected_residuals = expected["residuals"].get<std::vector<double>>();
	REQUIRE(result.residuals.size() == expected_residuals.size());

	for (size_t i = 0; i < expected_residuals.size(); i++) {
		REQUIRE_THAT(result.residuals(i),
		             Catch::Matchers::WithinAbs(expected_residuals[i], TOLERANCE));
	}

	// Check fitted values
	auto expected_fitted = expected["fitted_values"].get<std::vector<double>>();
	REQUIRE(result.fitted_values.size() == expected_fitted.size());

	for (size_t i = 0; i < expected_fitted.size(); i++) {
		REQUIRE_THAT(result.fitted_values(i),
		             Catch::Matchers::WithinAbs(expected_fitted[i], TOLERANCE));
	}
}

TEST_CASE("OLS: Multiple Regression", "[ols][validation]") {
	// Load test data
	TestData data = load_csv("../../../test/data/ols_tests/input/multiple_regression.csv");
	json expected = load_expected_json("../../../test/data/ols_tests/expected/multiple_regression.json");

	// Fit model with intercept
	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;

	auto result = OLSSolver::Fit(data.X, data.y, opts);

	REQUIRE(result.success);

	// Check coefficients
	auto expected_coefs = expected["coefficients"].get<std::vector<double>>();
	REQUIRE(result.coefficients.size() == expected_coefs.size());

	for (size_t i = 0; i < expected_coefs.size(); i++) {
		REQUIRE_THAT(result.coefficients(i),
		             Catch::Matchers::WithinAbs(expected_coefs[i], TOLERANCE));
	}

	// Check R²
	double expected_r2 = expected["r_squared"].get<double>();
	REQUIRE_THAT(result.r_squared,
	             Catch::Matchers::WithinAbs(expected_r2, TOLERANCE));

	// Check adjusted R²
	double expected_adj_r2 = expected["adj_r_squared"].get<double>();
	REQUIRE_THAT(result.adj_r_squared,
	             Catch::Matchers::WithinAbs(expected_adj_r2, TOLERANCE));

	// Check residuals (spot check first 10)
	auto expected_residuals = expected["residuals"].get<std::vector<double>>();
	REQUIRE(result.residuals.size() == expected_residuals.size());

	for (size_t i = 0; i < std::min(size_t(10), expected_residuals.size()); i++) {
		REQUIRE_THAT(result.residuals(i),
		             Catch::Matchers::WithinAbs(expected_residuals[i], TOLERANCE));
	}
}

TEST_CASE("OLS: No Intercept", "[ols][validation]") {
	// Load test data
	TestData data = load_csv("../../../test/data/ols_tests/input/no_intercept.csv");
	json expected = load_expected_json("../../../test/data/ols_tests/expected/no_intercept.json");

	// Fit model WITHOUT intercept
	core::RegressionOptions opts;
	opts.intercept = false;
	opts.compute_inference = false;

	auto result = OLSSolver::Fit(data.X, data.y, opts);

	REQUIRE(result.success);

	// Check coefficients (should NOT have intercept)
	auto expected_coefs = expected["coefficients"].get<std::vector<double>>();
	REQUIRE(result.coefficients.size() == expected_coefs.size());

	for (size_t i = 0; i < expected_coefs.size(); i++) {
		REQUIRE_THAT(result.coefficients(i),
		             Catch::Matchers::WithinAbs(expected_coefs[i], TOLERANCE));
	}

	// Check R²
	double expected_r2 = expected["r_squared"].get<double>();
	REQUIRE_THAT(result.r_squared,
	             Catch::Matchers::WithinAbs(expected_r2, TOLERANCE));
}

TEST_CASE("OLS: Rank Deficient Matrix", "[ols][validation]") {
	// Load test data with perfect collinearity
	TestData data = load_csv("../../../test/data/ols_tests/input/rank_deficient.csv");
	json expected = load_expected_json("../../../test/data/ols_tests/expected/rank_deficient.json");

	// Fit model with intercept
	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;

	auto result = OLSSolver::Fit(data.X, data.y, opts);

	REQUIRE(result.success);

	// Check that rank is less than number of features
	REQUIRE(result.rank < data.n_features);

	// Check that some coefficients are marked as aliased
	bool has_aliased = false;
	for (bool aliased : result.is_aliased) {
		if (aliased) {
			has_aliased = true;
			break;
		}
	}
	REQUIRE(has_aliased);

	// Check R² (should still be high since collinearity doesn't affect fit)
	double expected_r2 = expected["r_squared"].get<double>();
	REQUIRE_THAT(result.r_squared,
	             Catch::Matchers::WithinAbs(expected_r2, TOLERANCE));

	// Check that non-aliased coefficients match expected
	auto expected_coefs = expected["coefficients"].get<std::vector<double>>();

	for (size_t i = 0; i < result.coefficients.size(); i++) {
		if (!result.is_aliased[i]) {
			// Non-aliased coefficients should match
			REQUIRE_THAT(result.coefficients(i),
			             Catch::Matchers::WithinAbs(expected_coefs[i], TOLERANCE));
		} else {
			// Aliased coefficients should be NaN or zero
			REQUIRE((std::isnan(result.coefficients(i)) || result.coefficients(i) == 0.0));
		}
	}
}

TEST_CASE("OLS: Perfect Collinearity", "[ols][validation]") {
	// Load test data with perfect collinearity
	TestData data = load_csv("../../../test/data/ols_tests/input/perfect_collinearity.csv");
	json expected = load_expected_json("../../../test/data/ols_tests/expected/perfect_collinearity.json");

	// Fit model with intercept
	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;

	auto result = OLSSolver::Fit(data.X, data.y, opts);

	REQUIRE(result.success);

	// QR decomposition should detect collinearity
	REQUIRE(result.rank < data.n_features);

	// Verify residuals still make sense
	auto expected_residuals = expected["residuals"].get<std::vector<double>>();

	// Spot check first few residuals
	for (size_t i = 0; i < std::min(size_t(5), expected_residuals.size()); i++) {
		REQUIRE_THAT(result.residuals(i),
		             Catch::Matchers::WithinAbs(expected_residuals[i], TOLERANCE));
	}
}
