#pragma once

#include "duckdb.hpp"

#include <limits>

// Mirrors AnofoxAftDistribution in anofox_stats_ffi.h; kept as a plain int so this
// header does not have to include the C ABI header.
using AnofoxAftDistribution_t = int;
#include <optional>
#include <string>

namespace duckdb {

/**
 * Null policy for handling NULL values in y (response variable)
 */
enum class NullPolicy {
	DROP,         // Drop rows with NULL y from training, but include in output with predictions
	DROP_Y_ZERO_X // Drop rows with NULL y OR zero x values from training
};

/**
 * Poisson link functions
 */
enum class PoissonLink { LOG, IDENTITY, SQRT };

/**
 * Binomial link functions
 */
enum class BinomialLink { LOGIT, PROBIT, CLOGLOG };

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
enum class AlmLoss { LIKELIHOOD = 0, MSE = 1, MAE = 2, HAM = 3, ROLE = 4 };

/**
 * Solver (decomposition) type for linear regression
 */
enum class SolverType { QR, SVD, CHOLESKY };

/**
 * Heteroscedasticity-consistent standard error type
 */
enum class HcType { NONE, HC0, HC1, HC2, HC3 };

/**
 * Lambda scaling convention for regularized regression
 */
/// Prior family requested for a coefficient.
enum class PriorKindOpt { FLAT, NORMAL, LAPLACE };

/// One coefficient's prior as written in the options MAP.
struct PriorSpecOpt {
	PriorKindOpt kind = PriorKindOpt::FLAT;
	double loc = 0.0;
	double scale = std::numeric_limits<double>::infinity();
};

/// Coefficient covariance type for penalized / MAP fits.
enum class VcovTypeOpt {
	/// (X'WX + P)^-1 -- curvature of the log posterior at the mode. Default.
	LAPLACE,
	/// (X'WX + P)^-1 X'WX (X'WX + P)^-1.
	SANDWICH,
	/// (X'WX)^-1 -- ignores the penalty; reproduces pre-Laplace standard errors.
	NAIVE
};

enum class LambdaScaling { RAW, GLMNET };

/**
 * AID outlier detection methods
 */
enum class AidOutlierMethod { ZSCORE = 0, IQR = 1 };

// ============================================================================
// Statistical Hypothesis Test Options
// ============================================================================

/**
 * Alternative hypothesis for statistical tests
 */
enum class Alternative { TWO_SIDED = 0, LESS = 1, GREATER = 2 };

/**
 * Kendall tau variant
 */
enum class KendallType { TAU_A = 0, TAU_B = 1, TAU_C = 2 };

/**
 * T-test kind
 */
enum class TTestKind {
	STUDENT = 0, // Equal variances assumed
	WELCH = 1    // Unequal variances (default)
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
	std::optional<double> alpha;  // Regularization strength (also accepts 'lambda')
	std::optional<double> lambda; // Alias for alpha (Ridge)

	// ElasticNet specific
	std::optional<double> l1_ratio;         // Mix between L1 and L2 (0=Ridge, 1=Lasso)
	std::optional<uint32_t> max_iterations; // Max iterations for coordinate descent
	std::optional<double> tolerance;        // Convergence tolerance

	// Huber specific
	std::optional<double> epsilon; // Huber threshold parameter (must be > 1.0)

	// RANSAC specific
	std::optional<double> residual_threshold; // Inlier threshold; None → MAD(y)
	std::optional<uint32_t> max_trials;       // Max RANSAC trials (also reused by other iterative solvers)
	std::optional<double> stop_probability;   // Fischler-Bolles probability in [0,1]
	std::optional<uint32_t> stop_n_inliers;   // Stop as soon as this many inliers found
	std::optional<uint32_t> min_samples;      // Subsample size per trial; None → n_features + 1
	std::optional<uint64_t> random_state;     // Seed for the trial subsampler

	// Theil-Sen specific
	std::optional<uint32_t> max_subpopulation; // Cap on subsamples examined (sklearn default 10_000)
	std::optional<uint32_t> n_subsamples;      // Subsample size; None → n_features + 1

	// RLS specific
	std::optional<double> forgetting_factor;  // Forgetting factor (0-1)
	std::optional<double> initial_p_diagonal; // Initial P matrix diagonal value

	// Null handling
	std::optional<NullPolicy> null_policy; // How to handle NULL y values

	// GLM specific
	std::optional<PoissonLink> poisson_link;         // Link function for Poisson
	std::optional<BinomialLink> binomial_link;       // Link function for Binomial
	std::optional<double> tweedie_power;             // Power parameter for Tweedie (1 < p < 2)
	std::optional<AnofoxAftDistribution_t> aft_dist; // AFT error distribution
	// Mixed-effects GLMs (issue #107)
	std::optional<int> glmm_family;     // Mirrors AnofoxGlmmFamily
	std::optional<bool> reml;           // REML vs ML variance components
	std::optional<idx_t> offset_column; // 1-based index into x; 0/unset = none
	//! 1-based indices into x of columns that also carry a random slope (GLMM).
	std::optional<vector<idx_t>> random_slopes;
	//! 1-based indices into x of additional crossed grouping-factor columns (GLMM).
	std::optional<vector<idx_t>> group_columns;

	// Empirical-Bayes shrinkage (issue #107)
	std::optional<double> tau_squared; // Fixed between-group variance; unset = estimate
	std::optional<bool> tau_method;    // true = complete pooling, false = DerSimonian-Laird

	std::optional<double> nb_theta; // Negative Binomial dispersion; unset = estimate

	// ALM specific
	std::optional<AlmDistribution> distribution; // Distribution family
	std::optional<AlmLoss> loss;                 // Loss function
	std::optional<double> quantile;              // Quantile for AsymmetricLaplace (0-1)
	std::optional<double> role_trim;             // ROLE trim fraction

	// BLS specific
	std::optional<double> lower_bound; // Lower bound for all coefficients
	std::optional<double> upper_bound; // Upper bound for all coefficients

	// AID specific
	std::optional<double> intermittent_threshold;   // Zero proportion threshold (default: 0.3)
	std::optional<AidOutlierMethod> outlier_method; // Outlier detection method

	// PLS specific
	std::optional<size_t> n_components; // Number of components for PLS

	// Quantile specific
	std::optional<double> tau; // Quantile to estimate (0 < tau < 1)

	// Isotonic specific
	std::optional<bool> increasing; // Whether function is increasing or decreasing

	// Solver/inference options (OLS, Ridge, WLS)
	std::optional<SolverType> solver; // Decomposition method: qr, svd, cholesky
	std::optional<HcType> hc_type;    // HC standard errors: none, hc0, hc1, hc2, hc3

	// Lambda scaling (Ridge, ElasticNet)
	std::optional<LambdaScaling> lambda_scaling; // Lambda scaling: raw, glmnet

	// GLM regularization
	std::optional<double> glm_lambda; // L2 regularization for GLM (Poisson)

	// Classification (Logistic)
	std::optional<double> threshold; // Classification threshold on P(y=1) — Logistic

	// Explicit priors (issue #107)
	//
	// Feature names are needed because the aggregate signature only ever sees
	// `x LIST(DOUBLE)` — there is no other place a name could come from. They are
	// resolved to positions before the FFI boundary, which stays positional POD.
	std::optional<vector<string>> feature_names;
	/// The raw `prior` value, held until n_features is known. Options are parsed at
	/// bind time but the feature count only arrives with the first LIST in update,
	/// so name-to-position resolution has to be deferred.
	std::optional<Value> prior_value;
	std::optional<VcovTypeOpt> vcov;

	// LARS specific
	std::optional<bool> lars_lasso;           // 'method': 'lar' (false) | 'lasso' (true)
	std::optional<int64_t> n_nonzero_coefs;  // Cap on non-zero coefficients
	std::optional<bool> standardize;         // Standardize features before fitting

	/// Raw value of the context-dependent `link` key; resolved to poisson_link or
	/// binomial_link by ParseFromValue according to the function's supported keys.
	std::optional<Value> link_value;

	/// Resolve `prior_value` against `feature_names` into a positional vector of
	/// length `n_features` (+1 in front when an intercept is fitted).
	///
	/// Throws on an unknown feature name, a bad distribution name, a non-positive
	/// scale, or a name list whose length disagrees with `n_features`.
	vector<PriorSpecOpt> ResolvePriors(idx_t n_features, bool fit_intercept) const;

	/// True when a prior was supplied at all.
	bool HasPriors() const {
		return prior_value.has_value();
	}

	/**
	 * Parse options from a DuckDB MAP or STRUCT Value.
	 *
	 * `function_name` is used in error messages. `supported_keys` lists the
	 * canonical option keys (see the key table in map_options_parser.cpp) that the
	 * calling function actually reads; every other key -- a typo or a valid key
	 * that this function would silently ignore -- raises InvalidInputException
	 * naming the function and listing its supported keys.
	 *
	 * The supported set also resolves the context-dependent aliases:
	 *   - `link`            -> `poisson_link` or `binomial_link`, whichever is supported
	 *   - `alpha`, `lambda` -> `glm_lambda` when the function supports `glm_lambda`
	 *                          but not the key itself (GLMs)
	 *   - `lambda`          -> `forgetting_factor` (RLS, where lambda is the
	 *                          conventional symbol of the forgetting factor)
	 *   - `tau`             -> `quantile` (ALM); `quantile` -> `tau` (quantile regression)
	 *
	 * Boolean options accept BOOLEAN or numeric values. Keys are case-insensitive.
	 */
	static RegressionMapOptions ParseFromValue(const Value &map_value, const string &function_name,
	                                           const vector<string> &supported_keys);

	/**
	 * Parse options from an Expression. The expression must be foldable (a
	 * constant); otherwise "options must be a constant" is raised.
	 */
	static RegressionMapOptions ParseFromExpression(ClientContext &context, Expression &expr,
	                                                const string &function_name, const vector<string> &supported_keys);

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
//
// Every ParseFromValue below accepts both MAP {...} and STRUCT {'k': v} literals
// (DuckDB yields either), rejects unknown keys with an error naming
// `function_name` and listing the supported keys, rejects invalid enum values,
// and validates confidence_level to lie strictly inside (0, 1).

/**
 * Evaluate an options argument at bind time. Throws "options must be a constant"
 * when the expression is not foldable instead of silently ignoring it.
 */
Value EvaluateConstantOptions(ClientContext &context, Expression &expr, const string &function_name);

/**
 * Resolve the significance level of a TOST procedure from the user's `alpha`
 * and/or `confidence_level`.
 *
 * Mapping (matches tost_t_test_agg's long-standing semantics):
 *     alpha = 1 - confidence_level
 * i.e. `confidence_level` is the confidence of each one-sided test, NOT of the
 * reported interval. A TOST at level alpha corresponds to a (1 - 2*alpha)
 * two-sided confidence interval, so {'confidence_level': 0.95} and
 * {'alpha': 0.05} are the same request and both yield a 90% CI.
 * Supplying both with inconsistent values is an error.
 */
double ResolveTostAlpha(const string &function_name, const std::optional<double> &alpha,
                        const std::optional<double> &confidence_level, double default_alpha = 0.05);

/**
 * Options for t-test
 */
struct TTestMapOptions {
	std::optional<Alternative> alternative;
	std::optional<double> confidence_level;
	std::optional<TTestKind> kind; // Student (var_equal=true) vs Welch (default)
	std::optional<bool> paired;
	std::optional<double> mu; // Hypothesized mean difference

	static TTestMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

/**
 * Options for Mann-Whitney U test
 */
struct MannWhitneyMapOptions {
	std::optional<Alternative> alternative;
	std::optional<double> confidence_level;
	std::optional<bool> continuity_correction;
	std::optional<bool> exact; // Exact null distribution (default false: normal approximation)
	std::optional<double> mu;  // Hypothesized location shift

	static MannWhitneyMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

/**
 * Options for Wilcoxon signed-rank test
 */
struct WilcoxonMapOptions {
	std::optional<Alternative> alternative;
	std::optional<double> confidence_level;
	std::optional<bool> continuity_correction;

	static WilcoxonMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

/**
 * Options for Brunner-Munzel test
 */
struct BrunnerMunzelMapOptions {
	std::optional<Alternative> alternative;
	std::optional<double> confidence_level;

	static BrunnerMunzelMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

/**
 * Options for Pearson/Spearman correlation
 */
struct CorrelationMapOptions {
	std::optional<double> confidence_level;

	static CorrelationMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

/**
 * Options for Kendall correlation
 */
struct KendallMapOptions {
	std::optional<double> confidence_level;
	std::optional<KendallType> variant;

	static KendallMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

/**
 * Options for Chi-square test
 */
struct ChiSquareMapOptions {
	std::optional<bool> continuity_correction;

	static ChiSquareMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

/**
 * Options for Fisher exact test
 */
struct FisherExactMapOptions {
	std::optional<Alternative> alternative;
	std::optional<double> confidence_level;

	static FisherExactMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

/**
 * Options for Energy distance test
 */
struct EnergyDistanceMapOptions {
	std::optional<uint32_t> n_permutations;
	std::optional<uint64_t> seed; // also accepted as random_state

	static EnergyDistanceMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

/**
 * Options for MMD test
 */
struct MmdMapOptions {
	std::optional<double> bandwidth; // Not supported by the core (median heuristic); the key is rejected
	std::optional<uint32_t> n_permutations;
	std::optional<uint64_t> seed; // also accepted as random_state

	static MmdMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

/**
 * Options for the two-sample TOST t-test (tost_t_test_agg)
 */
struct TostMapOptions {
	std::optional<Alternative> alternative; // unused; the key is rejected
	std::optional<double> confidence_level;
	std::optional<double> alpha; // = 1 - confidence_level, see ResolveTostAlpha
	std::optional<TTestKind> kind;
	std::optional<bool> paired; // unused; the key is rejected
	std::optional<double> mu;   // unused; the key is rejected
	std::optional<double> delta;       // Equivalence bound (symmetric)
	std::optional<double> bound_lower; // Asymmetric lower bound
	std::optional<double> bound_upper; // Asymmetric upper bound

	static TostMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

/**
 * Options for the paired TOST t-test (tost_paired_agg)
 */
struct TostPairedMapOptions {
	std::optional<double> confidence_level;
	std::optional<double> alpha;
	std::optional<double> delta;
	std::optional<double> bound_lower;
	std::optional<double> bound_upper;

	static TostPairedMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

/**
 * Options for the TOST correlation test (tost_correlation_agg)
 */
struct TostCorrelationMapOptions {
	std::optional<double> confidence_level;
	std::optional<double> alpha;
	std::optional<double> delta;
	std::optional<double> bound_lower;
	std::optional<double> bound_upper;
	std::optional<double> rho_null;
	std::optional<bool> spearman; // 'method': 'pearson' (false) | 'spearman' (true)

	static TostCorrelationMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

/**
 * Options for Yuen trimmed-mean test
 */
struct YuenMapOptions {
	std::optional<Alternative> alternative;
	std::optional<double> confidence_level;
	std::optional<double> trim; // Trim proportion (default 0.2)

	static YuenMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

/**
 * Options for permutation t-test
 */
struct PermutationMapOptions {
	std::optional<Alternative> alternative;
	std::optional<uint32_t> n_permutations;
	std::optional<uint64_t> seed; // also accepted as random_state

	static PermutationMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

/**
 * Options for the distance-correlation test
 */
struct DistanceCorMapOptions {
	std::optional<uint32_t> n_permutations;
	std::optional<uint64_t> seed; // also accepted as random_state

	static DistanceCorMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

/**
 * Options for the Diebold-Mariano test
 */
struct DieboldMarianoMapOptions {
	std::optional<bool> absolute_loss; // 'loss': 'squared' (false) | 'absolute' (true)
	std::optional<bool> bartlett;      // 'var_estimator': 'acf' (false) | 'bartlett' (true)
	std::optional<uint32_t> horizon;
	std::optional<Alternative> alternative;

	static DieboldMarianoMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

/**
 * Options for the Clark-West test
 */
struct ClarkWestMapOptions {
	std::optional<uint32_t> horizon;

	static ClarkWestMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

/**
 * Options for one-sample proportion tests (binom_test_agg, prop_test_one_agg)
 */
struct ProportionMapOptions {
	std::optional<double> p0;
	std::optional<Alternative> alternative;

	static ProportionMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

/**
 * Options for the two-sample proportion test
 */
struct PropTestTwoMapOptions {
	std::optional<Alternative> alternative;
	std::optional<bool> correction;

	static PropTestTwoMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

/**
 * Options for the intraclass correlation coefficient
 */
struct IccMapOptions {
	std::optional<bool> average; // 'type': 'single' (false) | 'average' (true)

	static IccMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

/**
 * Options for McNemar's test
 */
struct McNemarMapOptions {
	std::optional<bool> correction;
	std::optional<bool> exact; // Exact binomial test on the discordant pairs

	static McNemarMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

/**
 * Options for Cohen's kappa
 */
struct CohenKappaMapOptions {
	std::optional<bool> weighted;

	static CohenKappaMapOptions ParseFromValue(const Value &map_value, const string &function_name);
};

} // namespace duckdb
