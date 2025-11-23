#include <iostream>
#include <vector>
#include <cmath>
#include "include/libanostat/solvers/ridge_solver.hpp"

using namespace libanostat;
using namespace libanostat::solvers;

int main() {
	// This test matches the DuckDB fit-predict test case:
	// Training data: rows 1-6 with y = i*2 + 1, x1 = i (i=1..6)
	// Test data: rows 7-10 with x1 = i (i=7..10)
	// Lambda = 1.0, intercept = true

	std::cout << "Testing Ridge Fit-Predict (matching DuckDB test case)" << std::endl;
	std::cout << "=====================================================" << std::endl << std::endl;

	// Training data (rows 1-6)
	std::vector<std::vector<double>> X_train(1);  // Single feature x1
	X_train[0] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	std::vector<double> y_train = {3.0, 5.0, 7.0, 9.0, 11.0, 13.0};  // i*2 + 1

	std::cout << "Training data:" << std::endl;
	for (size_t i = 0; i < y_train.size(); i++) {
		std::cout << "  Row " << (i+1) << ": x1=" << X_train[0][i] << ", y=" << y_train[i] << std::endl;
	}
	std::cout << std::endl;

	// Fit Ridge model with lambda=1.0
	core::RegressionOptions opts;
	opts.intercept = true;
	opts.compute_inference = false;
	opts.lambda = 1.0;

	std::cout << "Fitting Ridge model with lambda=" << opts.lambda << ", intercept=" << opts.intercept << std::endl;

	auto result = RidgeSolver::Fit(X_train, y_train, opts);

	if (!result.success) {
		std::cerr << "ERROR: Ridge fit failed!" << std::endl;
		return 1;
	}

	std::cout << "Fit successful!" << std::endl << std::endl;

	// Check that we got valid coefficients
	if (result.coefficients.size() != 2) {
		std::cerr << "ERROR: Expected 2 coefficients (intercept + 1 feature), got " << result.coefficients.size() << std::endl;
		return 1;
	}

	if (std::isnan(result.coefficients(0)) || std::isnan(result.coefficients(1))) {
		std::cerr << "ERROR: Got NaN coefficients!" << std::endl;
		return 1;
	}

	// Print coefficients
	std::cout << "Ridge model coefficients:" << std::endl;
	std::cout << "  Intercept (coef[0]): " << result.coefficients(0) << std::endl;
	std::cout << "  Beta[x1] (coef[1]):  " << result.coefficients(1) << std::endl;
	std::cout << "  R²: " << result.r_squared << std::endl;
	std::cout << std::endl;

	// Make predictions for all data points (rows 1-10)
	std::vector<double> x_vals = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
	std::vector<double> predictions;

	std::cout << "Predictions:" << std::endl;
	for (double x : x_vals) {
		// Prediction: yhat = intercept + beta * x
		double yhat = result.coefficients(0) + result.coefficients(1) * x;
		predictions.push_back(yhat);
		std::cout << "  x=" << x << " -> yhat=" << yhat << std::endl;
	}
	std::cout << std::endl;

	// Expected predictions from DuckDB test:
	std::vector<double> expected = {2.85, 4.77, 6.68, 8.60, 10.51, 12.43, 14.34, 16.26, 18.17, 20.09};

	// Check predictions match expected values
	std::cout << "Comparing with expected values:" << std::endl;
	bool all_passed = true;
	double tolerance = 0.01;

	for (size_t i = 0; i < expected.size(); i++) {
		double diff = std::abs(predictions[i] - expected[i]);
		bool passed = diff < tolerance;
		all_passed = all_passed && passed;

		std::cout << "  Row " << (i+1) << ": predicted=" << predictions[i]
		          << ", expected=" << expected[i]
		          << ", diff=" << diff
		          << " [" << (passed ? "PASS" : "FAIL") << "]" << std::endl;
	}
	std::cout << std::endl;

	if (all_passed) {
		std::cout << "✓ All predictions match expected values!" << std::endl;
		return 0;
	} else {
		std::cout << "✗ Some predictions don't match expected values" << std::endl;
		return 1;
	}
}
