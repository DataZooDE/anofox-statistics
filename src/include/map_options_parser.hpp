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

// ============================================================================
// Statistical Hypothesis Test Options
// ============================================================================

/**
 * Alternative hypothesis for statistical tests
 */
enum class Alternative {
    TWO_SIDED = 0,
    LESS = 1,
    GREATER = 2
};

/**
 * Kendall tau variant
 */
enum class KendallType {
    TAU_A = 0,
    TAU_B = 1,
    TAU_C = 2
};

/**
 * T-test kind
 */
enum class TTestKind {
    STUDENT = 0,    // Equal variances assumed
    WELCH = 1       // Unequal variances (default)
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

    // PLS specific
    std::optional<size_t> n_components;  // Number of components for PLS

    // Quantile specific
    std::optional<double> tau;  // Quantile to estimate (0 < tau < 1)

    // Isotonic specific
    std::optional<bool> increasing;  // Whether function is increasing or decreasing

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

// ============================================================================
// Statistical Test Option Structs
// ============================================================================

/**
 * Options for t-test
 */
struct TTestMapOptions {
    std::optional<Alternative> alternative;
    std::optional<double> confidence_level;
    std::optional<TTestKind> kind;          // Student (var_equal=true) vs Welch (default)
    std::optional<bool> paired;
    std::optional<double> mu;               // Population mean for one-sample test

    static TTestMapOptions ParseFromValue(const Value &map_value);
};

/**
 * Options for Mann-Whitney U test
 */
struct MannWhitneyMapOptions {
    std::optional<Alternative> alternative;
    std::optional<double> confidence_level;
    std::optional<bool> continuity_correction;

    static MannWhitneyMapOptions ParseFromValue(const Value &map_value);
};

/**
 * Options for Wilcoxon signed-rank test
 */
struct WilcoxonMapOptions {
    std::optional<Alternative> alternative;
    std::optional<double> confidence_level;
    std::optional<bool> continuity_correction;

    static WilcoxonMapOptions ParseFromValue(const Value &map_value);
};

/**
 * Options for Brunner-Munzel test
 */
struct BrunnerMunzelMapOptions {
    std::optional<Alternative> alternative;
    std::optional<double> confidence_level;

    static BrunnerMunzelMapOptions ParseFromValue(const Value &map_value);
};

/**
 * Options for Pearson/Spearman correlation
 */
struct CorrelationMapOptions {
    std::optional<double> confidence_level;

    static CorrelationMapOptions ParseFromValue(const Value &map_value);
};

/**
 * Options for Kendall correlation
 */
struct KendallMapOptions {
    std::optional<double> confidence_level;
    std::optional<KendallType> variant;

    static KendallMapOptions ParseFromValue(const Value &map_value);
};

/**
 * Options for Chi-square test
 */
struct ChiSquareMapOptions {
    std::optional<bool> continuity_correction;

    static ChiSquareMapOptions ParseFromValue(const Value &map_value);
};

/**
 * Options for Fisher exact test
 */
struct FisherExactMapOptions {
    std::optional<Alternative> alternative;

    static FisherExactMapOptions ParseFromValue(const Value &map_value);
};

/**
 * Options for Energy distance test
 */
struct EnergyDistanceMapOptions {
    std::optional<uint32_t> n_permutations;

    static EnergyDistanceMapOptions ParseFromValue(const Value &map_value);
};

/**
 * Options for MMD test
 */
struct MmdMapOptions {
    std::optional<double> bandwidth;
    std::optional<uint32_t> n_permutations;

    static MmdMapOptions ParseFromValue(const Value &map_value);
};

/**
 * Options for TOST equivalence tests
 */
struct TostMapOptions {
    std::optional<Alternative> alternative;
    std::optional<double> confidence_level;
    std::optional<TTestKind> kind;
    std::optional<bool> paired;
    std::optional<double> mu;
    std::optional<double> delta;            // Equivalence bound (symmetric)
    std::optional<double> bound_lower;      // Asymmetric lower bound
    std::optional<double> bound_upper;      // Asymmetric upper bound

    static TostMapOptions ParseFromValue(const Value &map_value);
};

/**
 * Options for Yuen trimmed-mean test
 */
struct YuenMapOptions {
    std::optional<Alternative> alternative;
    std::optional<double> confidence_level;
    std::optional<double> trim;             // Trim proportion (default 0.2)

    static YuenMapOptions ParseFromValue(const Value &map_value);
};

/**
 * Options for permutation t-test
 */
struct PermutationMapOptions {
    std::optional<Alternative> alternative;
    std::optional<uint32_t> n_permutations;

    static PermutationMapOptions ParseFromValue(const Value &map_value);
};

} // namespace duckdb
