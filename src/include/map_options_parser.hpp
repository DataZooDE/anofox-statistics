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
 * Poisson link functions
 */
enum class PoissonLink {
    LOG,
    IDENTITY,
    SQRT
};

/**
 * Binomial link functions
 */
enum class BinomialLink {
    LOGIT,
    PROBIT,
    CLOGLOG
};

/**
 * ALM distribution families
 */
enum class AlmDistribution {
    NORMAL = 0,
    LAPLACE = 1,
    STUDENT_T = 2,
    LOGISTIC = 3,
    ASYMMETRIC_LAPLACE = 4,
    GENERALISED_NORMAL = 5,
    S = 6,
    LOG_NORMAL = 7,
    LOG_LAPLACE = 8,
    LOG_S = 9,
    LOG_GENERALISED_NORMAL = 10,
    FOLDED_NORMAL = 11,
    RECTIFIED_NORMAL = 12,
    BOX_COX_NORMAL = 13,
    GAMMA = 14,
    INVERSE_GAUSSIAN = 15,
    EXPONENTIAL = 16,
    BETA = 17,
    LOGIT_NORMAL = 18,
    POISSON = 19,
    NEGATIVE_BINOMIAL = 20,
    BINOMIAL = 21,
    GEOMETRIC = 22,
    CUMULATIVE_LOGISTIC = 23,
    CUMULATIVE_NORMAL = 24
};

/**
 * ALM loss functions
 */
enum class AlmLoss {
    LIKELIHOOD = 0,
    MSE = 1,
    MAE = 2,
    HAM = 3,
    ROLE = 4
};

/**
 * AID outlier detection methods
 */
enum class AidOutlierMethod {
    ZSCORE = 0,
    IQR = 1
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

    // GLM specific
    std::optional<PoissonLink> poisson_link;    // Link function for Poisson
    std::optional<BinomialLink> binomial_link;  // Link function for Binomial
    std::optional<double> tweedie_power;        // Power parameter for Tweedie (1 < p < 2)

    // ALM specific
    std::optional<AlmDistribution> distribution;  // Distribution family
    std::optional<AlmLoss> loss;                  // Loss function
    std::optional<double> quantile;               // Quantile for AsymmetricLaplace (0-1)
    std::optional<double> role_trim;              // ROLE trim fraction

    // BLS specific
    std::optional<double> lower_bound;   // Lower bound for all coefficients
    std::optional<double> upper_bound;   // Upper bound for all coefficients

    // AID specific
    std::optional<double> intermittent_threshold;    // Zero proportion threshold (default: 0.3)
    std::optional<AidOutlierMethod> outlier_method;  // Outlier detection method

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
