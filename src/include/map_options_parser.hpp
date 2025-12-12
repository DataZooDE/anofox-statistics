#pragma once

#include "duckdb.hpp"
#include <optional>
#include <string>

namespace duckdb {

/**
 * Null policy for handling NULL values in y (response variable)
 */
enum class NullPolicy {
    DROP,          // Drop rows with NULL y from training, but include in output with predictions
    DROP_Y_ZERO_X  // Drop rows with NULL y OR zero x values from training
};

/**
 * Parsed regression options from a MAP parameter.
 * All fields are optional - only set if present in the MAP.
 */
struct RegressionMapOptions {
    // Common options
    std::optional<bool> fit_intercept;
    std::optional<bool> compute_inference;
    std::optional<double> confidence_level;

    // Ridge/ElasticNet regularization
    std::optional<double> alpha;   // Regularization strength (also accepts 'lambda')
    std::optional<double> lambda;  // Alias for alpha (Ridge)

    // ElasticNet specific
    std::optional<double> l1_ratio;          // Mix between L1 and L2 (0=Ridge, 1=Lasso)
    std::optional<uint32_t> max_iterations;  // Max iterations for coordinate descent
    std::optional<double> tolerance;         // Convergence tolerance

    // RLS specific
    std::optional<double> forgetting_factor;   // Forgetting factor (0-1)
    std::optional<double> initial_p_diagonal;  // Initial P matrix diagonal value

    // Null handling
    std::optional<NullPolicy> null_policy;  // How to handle NULL y values

    /**
     * Parse options from a DuckDB MAP Value.
     * Supports both integer (0/1) and boolean values for boolean options.
     * Keys are case-insensitive.
     */
    static RegressionMapOptions ParseFromValue(const Value &map_value);

    /**
     * Parse options from an Expression (evaluates constant expression first).
     */
    static RegressionMapOptions ParseFromExpression(ClientContext &context, Expression &expr);

    // Helper to get alpha/lambda (returns alpha if set, otherwise lambda)
    std::optional<double> GetRegularizationStrength() const {
        if (alpha.has_value()) {
            return alpha;
        }
        return lambda;
    }
};

} // namespace duckdb
