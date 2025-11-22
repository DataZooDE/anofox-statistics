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

TEST_CASE("OLS: Constant Feature", "[ols][constant]") {
	// Test with single constant feature
	// y = [1, 2, 3, 4, 5], x1 = [1, 2, 3, 4, 5], x2 = [5, 5, 5, 5, 5] (constant)
	std::vector<std::vector<double>> X(2);
	X[0] = {1.0, 2.0, 3.0, 4.0, 5.0};
	X[1] = {5.0, 5.0, 5.0, 5.0, 5.0};  // Constant feature
	std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0};

	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;

	auto result = OLSSolver::Fit(X, y, opts);

	REQUIRE(result.success);
	
	// Constant feature should be marked as aliased
	bool has_aliased = false;
	for (size_t i = 0; i < result.is_aliased.size(); i++) {
		if (result.is_aliased[i]) {
			has_aliased = true;
			// Aliased coefficient should be NaN
			REQUIRE(std::isnan(result.coefficients(static_cast<Eigen::Index>(i))));
		}
	}
	REQUIRE(has_aliased);
	
	// Rank should be less than number of features
	REQUIRE(result.rank < 2);
}

TEST_CASE("OLS: Multiple Constant Features", "[ols][constant]") {
	// Test with multiple constant features
	std::vector<std::vector<double>> X(3);
	X[0] = {1.0, 2.0, 3.0, 4.0, 5.0};
	X[1] = {5.0, 5.0, 5.0, 5.0, 5.0};  // Constant
	X[2] = {7.0, 7.0, 7.0, 7.0, 7.0};  // Constant
	std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0};

	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;

	auto result = OLSSolver::Fit(X, y, opts);

	REQUIRE(result.success);
	
	// Should have at least one aliased feature
	int aliased_count = 0;
	for (size_t i = 0; i < result.is_aliased.size(); i++) {
		if (result.is_aliased[i]) {
			aliased_count++;
			REQUIRE(std::isnan(result.coefficients(static_cast<Eigen::Index>(i))));
		}
	}
	REQUIRE(aliased_count >= 1);
}

TEST_CASE("OLS: All Features Constant", "[ols][constant]") {
	// Test with all features constant (should fail gracefully or return rank=0)
	std::vector<std::vector<double>> X(2);
	X[0] = {5.0, 5.0, 5.0, 5.0, 5.0};  // Constant
	X[1] = {7.0, 7.0, 7.0, 7.0, 7.0};  // Constant
	std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0};

	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;

	auto result = OLSSolver::Fit(X, y, opts);

	// Should either fail or return rank=0
	if (result.success) {
		REQUIRE(result.rank == 0);
	}
}

TEST_CASE("OLS: Constant Feature with Perfect Collinearity", "[ols][constant]") {
	// Test constant feature combined with perfect collinearity
	// x1 = [1, 2, 3, 4, 5], x2 = [2, 4, 6, 8, 10] (2*x1), x3 = [5, 5, 5, 5, 5] (constant)
	std::vector<std::vector<double>> X(3);
	X[0] = {1.0, 2.0, 3.0, 4.0, 5.0};
	X[1] = {2.0, 4.0, 6.0, 8.0, 10.0};  // 2*x1 (perfect collinearity)
	X[2] = {5.0, 5.0, 5.0, 5.0, 5.0};   // Constant
	std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0};

	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;

	auto result = OLSSolver::Fit(X, y, opts);

	REQUIRE(result.success);
	
	// Should detect both issues
	REQUIRE(result.rank < 3);
	
	// Should have multiple aliased features
	int aliased_count = 0;
	for (size_t i = 0; i < result.is_aliased.size(); i++) {
		if (result.is_aliased[i]) {
			aliased_count++;
		}
	}
	REQUIRE(aliased_count >= 1);
}

TEST_CASE("OLS: R-squared Properties", "[ols][properties]") {
	// Test that R² is in valid range [0, 1]
	TestData data = load_csv("../../../test/data/ols_tests/input/simple_linear.csv");

	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;

	auto result = OLSSolver::Fit(data.X, data.y, opts);

	REQUIRE(result.success);
	REQUIRE(result.r_squared >= 0.0);
	REQUIRE(result.r_squared <= 1.0);
	REQUIRE(result.adj_r_squared >= 0.0);
	REQUIRE(result.adj_r_squared <= 1.0);
	REQUIRE(result.adj_r_squared <= result.r_squared);
}

TEST_CASE("OLS: Residuals Sum to Zero with Intercept", "[ols][properties]") {
	// Test that residuals sum to zero when intercept is included
	TestData data = load_csv("../../../test/data/ols_tests/input/simple_linear.csv");

	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;

	auto result = OLSSolver::Fit(data.X, data.y, opts);

	REQUIRE(result.success);
	
	// Sum of residuals should be approximately zero
	double residual_sum = result.residuals.sum();
	REQUIRE_THAT(std::abs(residual_sum), Catch::Matchers::WithinAbs(0.0, 1e-10));
}

TEST_CASE("OLS: Fitted Values Consistency", "[ols][properties]") {
	// Test that residuals = y - fitted_values
	TestData data = load_csv("../../../test/data/ols_tests/input/simple_linear.csv");

	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;

	auto result = OLSSolver::Fit(data.X, data.y, opts);

	REQUIRE(result.success);
	
	// Verify residuals = y - fitted_values
	for (size_t i = 0; i < data.n_obs; i++) {
		double residual_manual = data.y[i] - result.fitted_values(static_cast<Eigen::Index>(i));
		REQUIRE_THAT(result.residuals(static_cast<Eigen::Index>(i)),
		             Catch::Matchers::WithinAbs(residual_manual, TOLERANCE));
	}
}

TEST_CASE("OLS: Standard Errors Always Positive", "[ols][properties]") {
	// Test that standard errors are always positive
	Eigen::VectorXd y(10);
	Eigen::MatrixXd X(10, 1);

	X << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0;
	y << 2.1, 4.0, 6.1, 8.0, 10.1, 12.0, 14.1, 16.0, 18.1, 20.0;

	core::RegressionOptions opts;
	opts.intercept = true;

	auto result = OLSSolver::FitWithStdErrors(y, X, opts);

	REQUIRE(result.has_std_errors);
	
	// All standard errors should be positive (or NaN for aliased)
	for (size_t i = 0; i < result.is_aliased.size(); i++) {
		if (!result.is_aliased[i]) {
			REQUIRE(result.std_errors(static_cast<Eigen::Index>(i)) > 0.0);
		}
	}
}

TEST_CASE("OLS: Edge Case - Single Observation", "[ols][edge]") {
	// Test with minimum data (should fail or handle gracefully)
	std::vector<std::vector<double>> X(1);
	X[0] = {1.0};
	std::vector<double> y = {2.0};

	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;

	auto result = OLSSolver::Fit(X, y, opts);

	// Should either fail or return rank=0
	if (result.success) {
		REQUIRE(result.rank == 0);
	}
}

TEST_CASE("OLS: Edge Case - Two Observations", "[ols][edge]") {
	// Test with minimum for regression (n=2, p=1, intercept)
	std::vector<std::vector<double>> X(1);
	X[0] = {1.0, 2.0};
	std::vector<double> y = {2.0, 4.0};

	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;

	auto result = OLSSolver::Fit(X, y, opts);

	// Should succeed with perfect fit
	REQUIRE(result.success);
	REQUIRE(result.r_squared >= 0.0);
	REQUIRE(result.r_squared <= 1.0);
}
