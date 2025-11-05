#pragma once

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * Options structure for regression functions
 *
 * Provides a unified interface for configuring regression algorithms.
 * All options have sensible defaults and can be overridden via MAP parameters.
 */
struct RegressionOptions {
	// Common options
	bool intercept = true;

	// Regularization (Ridge, Elastic Net)
	double lambda = 0.0;  // L2 penalty strength
	double alpha = 0.0;   // Elastic net mixing: 0=ridge, 1=lasso, (0,1)=elastic net

	// RLS specific
	double forgetting_factor = 1.0; // Weight recent observations more: (0,1]

	// Window functions
	idx_t window_size = 0;     // Rolling window size (0 = not set)
	idx_t min_periods = 0;     // Expanding minimum periods (0 = not set)

	// Statistical inference (future extensibility)
	double confidence_level = 0.95;
	bool robust_se = false;

	// Computational (future extensibility)
	idx_t max_iterations = 1000;
	double tolerance = 1e-6;
	string solver = "qr"; // "qr", "svd", "cholesky"

	// Grouped regression
	bool compute_per_group = false;

	/**
	 * Parse options from a DuckDB MAP parameter
	 *
	 * @param options_map MAP<VARCHAR, ANY> containing option key-value pairs
	 * @return RegressionOptions with parsed values or defaults
	 * @throws InvalidInputException if invalid option keys or values provided
	 */
	static RegressionOptions ParseFromMap(const Value &options_map);

	/**
	 * Validate option combinations
	 * Checks for incompatible options and invalid values
	 *
	 * @throws InvalidInputException if validation fails
	 */
	void Validate() const;

	/**
	 * Get default options
	 * @return RegressionOptions with all default values
	 */
	static RegressionOptions Defaults() {
		return RegressionOptions();
	}
};

/**
 * Helper function to extract a typed value from MAP with default fallback
 *
 * @param map_param The MAP parameter
 * @param key The key to look up
 * @param default_value Value to return if key not found
 * @return The value associated with key, or default_value if not found
 */
template <typename T>
T GetMapValueOrDefault(const Value &map_param, const string &key, T default_value);

} // namespace anofox_statistics
} // namespace duckdb
