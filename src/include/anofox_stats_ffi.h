/* Warning: this file is manually maintained to match crates/anofox-stats-ffi/src/lib.rs */

#ifndef ANOFOX_STATS_FFI_H
#define ANOFOX_STATS_FFI_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Error codes for FFI boundary
 */
typedef enum {
	ANOFOX_ERROR_SUCCESS = 0,
	ANOFOX_ERROR_INVALID_INPUT = 1,
	ANOFOX_ERROR_SINGULAR_MATRIX = 2,
	ANOFOX_ERROR_CONVERGENCE_FAILURE = 3,
	ANOFOX_ERROR_INVALID_ALPHA = 4,
	ANOFOX_ERROR_INVALID_L1_RATIO = 5,
	ANOFOX_ERROR_INSUFFICIENT_DATA = 6,
	ANOFOX_ERROR_ALLOCATION_FAILURE = 7,
	ANOFOX_ERROR_SERIALIZATION_ERROR = 8,
	ANOFOX_ERROR_DIMENSION_MISMATCH = 9,
	ANOFOX_ERROR_NO_VALID_DATA = 10,
	ANOFOX_ERROR_INTERNAL = 99,
} AnofoxErrorCode;

/**
 * Error information for FFI
 */
typedef struct {
	AnofoxErrorCode code;
	char message[256];
} AnofoxError;

/**
 * Array of f64 values with validity mask for NULL handling
 */
typedef struct {
	/** Pointer to data values */
	const double *data;
	/** Validity bitmask: bit i is 1 if data[i] is valid, 0 if NULL. Can be NULL if all values are valid. */
	const uint8_t *validity;
	/** Number of elements */
	size_t len;
} AnofoxDataArray;

#define ANOFOX_MODEL_NAME_LEN 32

/**
 * What was fitted, its likelihood-based fit statistics and the intercept's
 * inference (#152). Names are NUL-terminated; an empty family means the
 * estimator has no likelihood (NULL in SQL). Statistics that do not apply or
 * were not computed are NaN.
 */
typedef struct {
	char model_type[ANOFOX_MODEL_NAME_LEN];
	char family[ANOFOX_MODEL_NAME_LEN];
	char link[ANOFOX_MODEL_NAME_LEN];
	double log_likelihood;
	double aic;
	double bic;
	double intercept_std_error;
	double intercept_statistic;
	double intercept_p_value;
	double intercept_conf_low;
	double intercept_conf_high;
	/* Leave-one-out fit statistics (#220); NaN except for OLS / WLS. */
	double press;
	double press_rmse;
	double loo_r_squared;
} AnofoxModelSummaryFFI;

#define ANOFOX_TEST_METHOD_LEN 96
#define ANOFOX_TEST_NAME_LEN 32

/**
 * Unified hypothesis-test fields (#159, the integration contract's `test` row),
 * copied from upstream anofox_statistics::TestResult. Must match TestSummaryFFI
 * in crates/anofox-stats-ffi/src/types.rs. A field that does not apply is NaN
 * (numbers), -1 (n) or an empty string (names): NULL in SQL. Every test result
 * struct ends with one.
 */
typedef struct {
	double statistic;
	double df1;
	double df2;
	double p_value;
	double estimate;
	double conf_low;
	double conf_high;
	double conf_level;
	double effect_size;
	int64_t n;
	char method[ANOFOX_TEST_METHOD_LEN];
	char alternative[ANOFOX_TEST_NAME_LEN];
	char effect_name[ANOFOX_TEST_NAME_LEN];
} AnofoxTestSummaryFFI;

/**
 * Core fit result (always returned)
 */
typedef struct {
	/** Pointer to coefficients array */
	double *coefficients;
	/** Number of coefficients */
	size_t coefficients_len;
	/** Intercept value (NaN if no intercept) */
	double intercept;
	/** R-squared */
	double r_squared;
	/** Adjusted R-squared */
	double adj_r_squared;
	/** Residual standard error */
	double residual_std_error;
	/** Number of observations */
	size_t n_observations;
	/** Number of features */
	size_t n_features;
	/** Model description, fit statistics and intercept inference */
	AnofoxModelSummaryFFI summary;
} AnofoxFitResultCore;

/**
 * Inference results (optional)
 */
typedef struct {
	/** Standard errors of coefficients */
	double *std_errors;
	/** t-values */
	double *t_values;
	/** p-values */
	double *p_values;
	/** Lower confidence interval bounds */
	double *ci_lower;
	/** Upper confidence interval bounds */
	double *ci_upper;
	/** Number of elements in each array */
	size_t len;
	/** Confidence level used */
	double confidence_level;
	/** F-statistic (NaN if not computed) */
	double f_statistic;
	/** F p-value (NaN if not computed) */
	double f_pvalue;
} AnofoxFitResultInference;

/**
 * Solver (decomposition) type for linear regression
 */
typedef enum {
	ANOFOX_SOLVER_QR = 0,
	ANOFOX_SOLVER_SVD = 1,
	ANOFOX_SOLVER_CHOLESKY = 2,
} AnofoxSolverType;

/**
 * Lambda scaling convention for regularized regression
 */
typedef enum {
	ANOFOX_LAMBDA_SCALING_RAW = 0,
	ANOFOX_LAMBDA_SCALING_GLMNET = 1,
} AnofoxLambdaScaling;

/**
 * Heteroscedasticity-consistent standard error type
 */
typedef enum {
	ANOFOX_HC_NONE = 0,
	ANOFOX_HC_HC0 = 1,
	ANOFOX_HC_HC1 = 2,
	ANOFOX_HC_HC2 = 3,
	ANOFOX_HC_HC3 = 4,
} AnofoxHcType;

/**
 * OLS options for FFI
 */
typedef struct {
	/** Whether to fit intercept */
	bool fit_intercept;
	/** Whether to compute inference statistics */
	bool compute_inference;
	/** Confidence level for CIs */
	double confidence_level;
	/** Solver type (QR, SVD, Cholesky) */
	AnofoxSolverType solver;
	/** HC standard errors type (None = classical) */
	AnofoxHcType hc_type;
	/** Tolerance of R lm's aliasing test (tol); <= 0 = default 1e-7. Zero-initialise the struct. */
	double alias_tolerance;
	/** Compute the leave-one-out statistics (press, press_rmse, loo_r_squared) in the summary (#220) */
	bool compute_loo;
} AnofoxOlsOptions;

/**
 * Fit an OLS regression model
 *
 * @param y Response variable array
 * @param x Pointer to array of feature arrays
 * @param x_count Number of feature arrays
 * @param options OLS fitting options
 * @param out_core Output: core fit results (required)
 * @param out_inference Output: inference results (can be NULL if not needed)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_ols_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxOlsOptions options,
                    AnofoxFitResultCore *out_core, AnofoxFitResultInference *out_inference, AnofoxError *out_error);

/**
 * Free memory allocated by anofox_ols_fit for core results
 */
void anofox_free_result_core(AnofoxFitResultCore *result);

/**
 * Free memory allocated by anofox_ols_fit for inference results
 */
void anofox_free_result_inference(AnofoxFitResultInference *result);

/**
 * Huber M-estimator robust regression options for FFI
 */
typedef struct {
	/** Huber threshold parameter (must be > 1.0). Default 1.35. */
	double epsilon;
	/** L2 regularization (must be >= 0). Default 0.0001. */
	double alpha;
	/** Whether to fit intercept */
	bool fit_intercept;
	/** Whether to compute inference statistics */
	bool compute_inference;
	/** Confidence level for CIs */
	double confidence_level;
	/** Maximum IRLS iterations */
	uint32_t max_iterations;
	/** Convergence tolerance */
	double tolerance;
	/** Joint coefficient / scale estimate (sklearn HuberRegressor, #305); false = MAD scale (MASS::rlm) */
	bool joint_scale;
} AnofoxHuberOptions;

/**
 * Huber-specific extras returned alongside core/inference results.
 *
 * Memory: `outliers` is allocated by anofox_huber_fit when out_extras != NULL
 * and must be freed via anofox_free_huber_extras. Each byte is 0 (inlier) or
 * 1 (outlier). Length equals the number of non-NaN observations used in the
 * fit (which can be less than the input length when NaNs were filtered out).
 */
typedef struct {
	/** MAD-based scale estimate (sigma) */
	double scale;
	/** Echoed epsilon used for the fit */
	double epsilon;
	/** Per-observation outlier mask: 1 = |r_i| > epsilon * scale */
	uint8_t *outliers;
	/** Length of the outliers array */
	size_t outliers_len;
	/** Number of observations flagged as outliers */
	size_t n_outliers;
} AnofoxHuberFitExtras;

/**
 * Fit a Huber M-estimator robust regression model.
 *
 * @param y Response variable array
 * @param x Pointer to array of feature arrays
 * @param x_count Number of feature arrays
 * @param options Huber fitting options
 * @param out_core Output: core fit results (required)
 * @param out_inference Output: inference results (NULL if not needed)
 * @param out_extras Output: Huber-specific scale + outlier mask (NULL if not needed)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_huber_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxHuberOptions options,
                      AnofoxFitResultCore *out_core, AnofoxFitResultInference *out_inference,
                      AnofoxHuberFitExtras *out_extras, AnofoxError *out_error);

/**
 * Free the outliers array inside a AnofoxHuberFitExtras previously filled by
 * anofox_huber_fit. The core / inference parts are freed via the standard
 * anofox_free_result_core / anofox_free_result_inference helpers.
 */
void anofox_free_huber_extras(AnofoxHuberFitExtras *extras);

/**
 * RANSAC robust regression options for FFI.
 *
 * The `*_set` / `*_value` pairs encode Option<T> from Rust: when `*_set`
 * is false the upstream solver picks the default (e.g. min_samples
 * defaults to n_features + 1 with intercept; residual_threshold defaults
 * to MAD(y); stop_n_inliers defaults to disabled).
 */
typedef struct {
	bool fit_intercept;
	bool compute_inference;
	double confidence_level;
	uint32_t max_trials;
	/** Fischler-Bolles stop probability in [0, 1]. */
	double stop_probability;
	/** Random seed for the trial subsampler. */
	uint64_t random_state;

	bool min_samples_set;
	size_t min_samples_value;

	bool residual_threshold_set;
	double residual_threshold_value;

	bool stop_n_inliers_set;
	size_t stop_n_inliers_value;
} AnofoxRansacOptions;

/**
 * RANSAC-specific extras returned alongside core / inference results.
 *
 * Memory: `inliers` is allocated by anofox_ransac_fit when out_extras !=
 * NULL and must be freed via anofox_free_ransac_extras. Each byte is 0
 * (outlier) or 1 (inlier). Length equals the number of non-NaN
 * observations used in the fit.
 */
typedef struct {
	/** Residual threshold actually used (user-supplied or MAD(y) default). */
	double residual_threshold;
	/** Per-observation inlier mask. */
	uint8_t *inliers;
	/** Length of the inliers array. */
	size_t inliers_len;
	/** Number of inliers in the final consensus set. */
	size_t n_inliers;
	/** Actual number of RANSAC trials run before early termination. */
	size_t n_trials;
} AnofoxRansacFitExtras;

/**
 * Fit a RANSAC robust regression model.
 *
 * @param y Response variable array
 * @param x Pointer to array of feature arrays
 * @param x_count Number of feature arrays
 * @param options RANSAC fitting options
 * @param out_core Output: core fit results (required)
 * @param out_inference Output: inference results (NULL if not needed)
 * @param out_extras Output: inlier mask / trial count (NULL if not needed)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_ransac_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxRansacOptions options,
                       AnofoxFitResultCore *out_core, AnofoxFitResultInference *out_inference,
                       AnofoxRansacFitExtras *out_extras, AnofoxError *out_error);

/**
 * Free the inliers array inside an AnofoxRansacFitExtras previously filled
 * by anofox_ransac_fit. The core / inference parts are freed via the
 * standard anofox_free_result_core / anofox_free_result_inference helpers.
 */
void anofox_free_ransac_extras(AnofoxRansacFitExtras *extras);

/**
 * Theil-Sen robust regression options for FFI.
 *
 * `n_subsamples_set` / `n_subsamples_value` encode Rust's Option<usize>:
 * when `n_subsamples_set` is false the upstream solver picks
 * `n_features + 1` (sklearn default).
 */
typedef struct {
	bool fit_intercept;
	bool compute_inference;
	double confidence_level;
	/** Cap on the number of subsamples examined (sklearn default 10_000). */
	uint32_t max_subpopulation;
	uint32_t max_iterations;
	double tolerance;
	/** Random seed for the trial subsampler. */
	uint64_t random_state;

	bool n_subsamples_set;
	size_t n_subsamples_value;

	/** Interval method with compute_inference: 0 = auto (exact for one feature,
	 *  bootstrap otherwise), 1 = exact (Sen), 2 = pairs bootstrap, 3 = exact where it
	 *  applies, otherwise no intervals ('exact_or_none'). */
	int32_t ci_method;
	/** Bootstrap resamples (seeded by random_state). */
	uint32_t n_bootstrap;
} AnofoxTheilSenOptions;

/**
 * Fit a Theil-Sen robust regression model.
 *
 * Unlike Huber and RANSAC, Theil-Sen does not surface any consensus-set
 * diagnostics — the FFI returns only the standard core / inference
 * results.
 *
 * @param y Response variable array
 * @param x Pointer to array of feature arrays
 * @param x_count Number of feature arrays
 * @param options Theil-Sen fitting options
 * @param out_core Output: core fit results (required)
 * @param out_inference Output: inference results (NULL if not needed)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_theilsen_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxTheilSenOptions options,
                         AnofoxFitResultCore *out_core, AnofoxFitResultInference *out_inference,
                         AnofoxError *out_error);

/**
 * Ridge regression options for FFI
 */
typedef struct {
	/** L2 regularization parameter (alpha/lambda) */
	double alpha;
	/** Whether to fit intercept */
	bool fit_intercept;
	/** Whether to compute inference statistics */
	bool compute_inference;
	/** Confidence level for CIs */
	double confidence_level;
	/** Solver type (QR, SVD, Cholesky) */
	AnofoxSolverType solver;
	/** Lambda scaling convention (Raw or Glmnet) */
	AnofoxLambdaScaling lambda_scaling;
	/** Compute the leave-one-out statistics (press, press_rmse, loo_r_squared) in the summary (#235). Zero-initialise the struct. */
	bool compute_loo;
	/** Conditioning-aware lambda (#305): lambda_small when kappa(X'X) <= cond_threshold, alpha otherwise */
	bool conditioning;
	/** Lambda on a well-conditioned design; NaN = upstream default */
	double lambda_small;
	/** kappa(X'X) threshold; NaN = upstream default */
	double cond_threshold;
} AnofoxRidgeOptions;

/**
 * Fit a Ridge regression model
 *
 * @param y Response variable array
 * @param x Pointer to array of feature arrays
 * @param x_count Number of feature arrays
 * @param options Ridge fitting options
 * @param out_core Output: core fit results (required)
 * @param out_inference Output: inference results (can be NULL if not needed)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_ridge_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxRidgeOptions options,
                      AnofoxFitResultCore *out_core, AnofoxFitResultInference *out_inference, AnofoxError *out_error);

/**
 * Elastic Net regression options for FFI
 */
typedef struct {
	/** Regularization strength (must be >= 0) */
	double alpha;
	/** L1 ratio: 0 = Ridge, 1 = Lasso (must be in [0, 1]) */
	double l1_ratio;
	/** Whether to fit intercept */
	bool fit_intercept;
	/** Maximum iterations for coordinate descent */
	uint32_t max_iterations;
	/** Convergence tolerance */
	double tolerance;
	/** Lambda scaling convention (Raw or Glmnet) */
	AnofoxLambdaScaling lambda_scaling;
} AnofoxElasticNetOptions;

/**
 * Fit an Elastic Net regression model
 *
 * @param y Response variable array
 * @param x Pointer to array of feature arrays
 * @param x_count Number of feature arrays
 * @param options Elastic Net fitting options
 * @param out_core Output: core fit results (required)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_elasticnet_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxElasticNetOptions options,
                           AnofoxFitResultCore *out_core, AnofoxError *out_error);

/**
 * Options for Least Angle Regression (LARS / LassoLars)
 */
typedef struct {
	/** false = plain LARS, true = LassoLars (exact Lasso path) */
	bool method_lasso;
	/** Whether to fit an intercept term */
	bool fit_intercept;
	/** LassoLars early-stop alpha (0.0 = full path) */
	double alpha;
	/** Cap on non-zero coefficients (<= 0 = unlimited) */
	int64_t n_nonzero_coefs;
	/** Standardize features before fitting */
	bool standardize;
} AnofoxLarsOptions;

/**
 * Fit a Least Angle Regression (LARS / LassoLars) model
 *
 * @param y Response variable array
 * @param x Pointer to array of feature arrays
 * @param x_count Number of feature arrays
 * @param options LARS fitting options
 * @param out_core Output: core fit results (required)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_lars_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxLarsOptions options,
                     AnofoxFitResultCore *out_core, AnofoxError *out_error);

/**
 * WLS (Weighted Least Squares) options for FFI
 */
typedef struct {
	/** Whether to fit intercept */
	bool fit_intercept;
	/** Whether to compute inference statistics */
	bool compute_inference;
	/** Confidence level for CIs */
	double confidence_level;
	/** Solver type (QR, SVD, Cholesky) */
	AnofoxSolverType solver;
	/** HC standard errors type (None = classical) */
	AnofoxHcType hc_type;
	/** Tolerance of R lm's aliasing test (tol); <= 0 = default 1e-7. Zero-initialise the struct. */
	double alias_tolerance;
	/** Compute the leave-one-out statistics (press, press_rmse, loo_r_squared) in the summary (#220) */
	bool compute_loo;
} AnofoxWlsOptions;

/**
 * Fit a Weighted Least Squares regression model
 *
 * @param y Response variable array
 * @param x Pointer to array of feature arrays
 * @param x_count Number of feature arrays
 * @param weights Observation weights array (same length as y)
 * @param options WLS fitting options
 * @param out_core Output: core fit results (required)
 * @param out_inference Output: inference results (can be NULL if not needed)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_wls_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxDataArray weights,
                    AnofoxWlsOptions options, AnofoxFitResultCore *out_core, AnofoxFitResultInference *out_inference,
                    AnofoxError *out_error);

/**
 * Make predictions using fitted model coefficients
 *
 * @param x Pointer to array of feature arrays
 * @param x_count Number of feature arrays
 * @param coefficients Array of fitted coefficients
 * @param coefficients_len Number of coefficients
 * @param intercept Intercept value (use NaN if no intercept)
 * @param out_predictions Output: pointer to predictions array (caller must free with anofox_free_predictions)
 * @param out_predictions_len Output: number of predictions
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_predict(const AnofoxDataArray *x, size_t x_count, const double *coefficients, size_t coefficients_len,
                    double intercept, double **out_predictions, size_t *out_predictions_len, AnofoxError *out_error);

/**
 * Free memory allocated by anofox_predict
 */
void anofox_free_predictions(double *predictions);

/**
 * Get library version string
 */
const char *anofox_version(void);

/* ============================================================================
 * Diagnostics Functions
 * ============================================================================ */

/**
 * Compute VIF (Variance Inflation Factor) for each feature
 *
 * @param x Pointer to array of feature arrays
 * @param x_count Number of feature arrays
 * @param out_vif Output: pointer to VIF array (caller must free with anofox_free_vif)
 * @param out_vif_len Output: number of VIF values
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_compute_vif(const AnofoxDataArray *x, size_t x_count, double **out_vif, size_t *out_vif_len,
                        AnofoxError *out_error);

/**
 * Free memory allocated by anofox_compute_vif
 */
void anofox_free_vif(double *vif);

/**
 * Residuals computation result
 */
typedef struct {
	double *raw;
	double *standardized;
	double *studentized;
	double *leverage;
	double *cooks_distance;
	size_t len;
	bool has_standardized;
	bool has_studentized;
	bool has_leverage;
	bool has_cooks_distance;
} AnofoxResidualsResult;

/**
 * Compute residuals from y and predictions
 *
 * @param y Actual y values
 * @param y_hat Predicted y values
 * @param x Feature arrays (can be NULL if not computing studentized)
 * @param x_count Number of feature arrays
 * @param residual_std_error Residual standard error (use NaN if not available)
 * @param include_studentized Whether to compute studentized residuals
 * @param out_result Output: residuals result (caller must free with anofox_free_residuals)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_compute_residuals(AnofoxDataArray y, AnofoxDataArray y_hat, const AnofoxDataArray *x, size_t x_count,
                              double residual_std_error, bool include_studentized, AnofoxResidualsResult *out_result,
                              AnofoxError *out_error);

/**
 * Free memory allocated by anofox_compute_residuals
 */
void anofox_free_residuals(AnofoxResidualsResult *result);

/**
 * Compute AIC (Akaike Information Criterion)
 *
 * @param rss Residual sum of squares
 * @param n Number of observations
 * @param k Number of parameters (including intercept)
 * @param out_aic Output: AIC value
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_compute_aic(double rss, size_t n, size_t k, double *out_aic, AnofoxError *out_error);

/**
 * Compute BIC (Bayesian Information Criterion)
 *
 * @param rss Residual sum of squares
 * @param n Number of observations
 * @param k Number of parameters (including intercept)
 * @param out_bic Output: BIC value
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_compute_bic(double rss, size_t n, size_t k, double *out_bic, AnofoxError *out_error);

/* ============================================================================
 * RLS (Recursive Least Squares) Functions
 * ============================================================================ */

/**
 * RLS options for FFI
 */
typedef struct {
	/** Forgetting factor (lambda), typically 0.95-1.0 */
	double forgetting_factor;
	/** Whether to fit intercept */
	bool fit_intercept;
	/** Initial value for diagonal of P matrix */
	double initial_p_diagonal;
} AnofoxRlsOptions;

/**
 * Fit an RLS (Recursive Least Squares) regression model
 *
 * @param y Response variable array
 * @param x Pointer to array of feature arrays
 * @param x_count Number of feature arrays
 * @param options RLS fitting options
 * @param out_core Output: core fit results (required)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_rls_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxRlsOptions options,
                    AnofoxFitResultCore *out_core, AnofoxError *out_error);

/* ============================================================================
 * Jarque-Bera Test Functions
 * ============================================================================ */

/**
 * Result of Jarque-Bera test
 */
typedef struct {
	/** JB test statistic */
	double statistic;
	/** p-value for the test */
	double p_value;
	/** Sample skewness */
	double skewness;
	/** Sample kurtosis (excess) */
	double kurtosis;
	/** Number of observations */
	size_t n;
	/** Unified test fields (#159); keep last */
	AnofoxTestSummaryFFI unified;
} AnofoxJarqueBeraResult;

/**
 * Compute the Jarque-Bera test for normality
 *
 * @param data Sample data (typically residuals)
 * @param out_result Output: Jarque-Bera test results
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_jarque_bera(AnofoxDataArray data, AnofoxJarqueBeraResult *out_result, AnofoxError *out_error);

/* ============================================================================
 * Prediction Interval Functions
 * ============================================================================ */

/**
 * Get the critical value from the t-distribution for a given confidence level and degrees of freedom
 *
 * @param confidence_level Confidence level (e.g., 0.95 for 95% CI)
 * @param df Degrees of freedom (n - p - 1 for regression)
 * @return The t-critical value, or NaN if invalid inputs
 */
double anofox_t_critical(double confidence_level, size_t df);

/**
 * Prediction result with confidence interval
 */
typedef struct {
	/** Predicted value */
	double yhat;
	/** Lower bound of prediction interval */
	double yhat_lower;
	/** Upper bound of prediction interval */
	double yhat_upper;
} AnofoxPredictionResult;

/**
 * Compute prediction with confidence interval for a single new observation
 *
 * For OLS, the prediction interval is: yhat ± t_critical * se_pred
 * Uses a simplified formula assuming average leverage.
 *
 * @param coefficients Fitted coefficients array
 * @param coefficients_len Number of coefficients
 * @param intercept Intercept value (NaN if no intercept)
 * @param x_new New observation feature values
 * @param x_len Number of features (must equal coefficients_len)
 * @param residual_std_error Residual standard error from fit
 * @param n_observations Number of training observations
 * @param confidence_level Confidence level for interval (e.g., 0.95)
 * @param out_result Output: prediction result with interval
 * @return true on success, false on error
 */
bool anofox_predict_with_interval(const double *coefficients, size_t coefficients_len, double intercept,
                                  const double *x_new, size_t x_len, double residual_std_error, size_t n_observations,
                                  double confidence_level, AnofoxPredictionResult *out_result);

/**
 * Compute the leverage matrix M for leverage-aware prediction intervals
 *
 * The predictive variance factor of a new row x0 (augmented with a leading 1 when
 * fit_intercept) is x0' M x0. OLS: M = (X'X)^-1; with weights: M = (X'WX)^-1; with
 * ridge_lambda > 0: M = A X'(W)X A, A = (X'(W)X + lambda*I)^-1, intercept unpenalised
 * (matches anofox_ridge_fit; pass the effective lambda, i.e. lambda * n under glmnet
 * scaling). Columns whose coefficient is NaN (aliased/constant) or exactly 0.0 (inactive
 * in sparse fits such as elastic net) are excluded; their rows/columns are 0. Rows with
 * a non-finite x value (or non-finite/negative weight) are skipped. Fails with
 * ANOFOX_ERROR_SINGULAR_MATRIX if the (penalised) Gram matrix is singular.
 *
 * @param x Feature arrays (x_count columns of equal length)
 * @param x_count Number of feature columns
 * @param coefficients Fitted coefficients (one per column)
 * @param coefficients_len Must equal x_count
 * @param fit_intercept Whether the model has an intercept (adds a leading row/column)
 * @param weights Observation weights, or NULL for unweighted
 * @param ridge_lambda Effective ridge penalty (0 for none)
 * @param out_matrix Output: malloc'd row-major dim x dim matrix; free with
 *                   anofox_free_interval_matrix
 * @param out_dim Output: dim = x_count + (fit_intercept ? 1 : 0)
 * @param out_error Output: error information (may be NULL)
 * @return true on success, false on error
 */
bool anofox_interval_matrix(const AnofoxDataArray *x, size_t x_count, const double *coefficients,
                            size_t coefficients_len, bool fit_intercept, const AnofoxDataArray *weights,
                            double ridge_lambda, double **out_matrix, size_t *out_dim, AnofoxError *out_error);

/**
 * Free a matrix returned by anofox_interval_matrix (NULL is a no-op)
 */
void anofox_free_interval_matrix(double *matrix);

/**
 * Point prediction with a leverage-aware interval
 *
 * yhat +/- t_{df} * s * sqrt(1 + x0' M x0) for interval_type 0 (prediction interval) or
 * sqrt(x0' M x0) for interval_type 1 (confidence interval for the mean), with
 * df = n_observations - n_params_effective (n_params_effective counts the intercept).
 * If df == 0 or residual_std_error is not positive/finite the bounds are NaN (returns true).
 *
 * @param coefficients Fitted coefficients (NaN coefficients contribute nothing to yhat)
 * @param coefficients_len Number of coefficients
 * @param intercept Intercept value (NaN if no intercept)
 * @param x_new New observation feature values
 * @param x_len Number of features (must equal coefficients_len)
 * @param matrix Matrix from anofox_interval_matrix (dim x dim, row-major)
 * @param dim Matrix dimension; must equal x_len + (intercept is not NaN ? 1 : 0)
 * @param n_observations Number of training observations
 * @param n_params_effective Number of estimated parameters including the intercept
 * @param residual_std_error Residual standard error from the fit
 * @param confidence_level Confidence level in (0, 1)
 * @param interval_type 0 = prediction interval, 1 = confidence interval
 * @param out_result Output: prediction with interval
 * @param out_error Output: error information (may be NULL)
 * @return true on success, false on error
 */
bool anofox_predict_with_interval_matrix(const double *coefficients, size_t coefficients_len, double intercept,
                                         const double *x_new, size_t x_len, const double *matrix, size_t dim,
                                         size_t n_observations, size_t n_params_effective,
                                         double residual_std_error, double confidence_level, int32_t interval_type,
                                         AnofoxPredictionResult *out_result, AnofoxError *out_error);

/**
 * anofox_predict_with_interval_matrix for n_rows rows at once.
 *
 * x_rows holds the rows one after another (n_rows * x_len doubles, row-major). Row i's
 * results go to out_yhat[i], out_lower[i] and out_upper[i]. They equal the single-row
 * call's, but the matrix, the t quantile and the allocations are shared by all rows.
 *
 * @param out_yhat, out_lower, out_upper Output arrays of n_rows doubles each
 * @return true on success, false on error (then no output is written)
 */
bool anofox_predict_with_interval_matrix_batch(const double *coefficients, size_t coefficients_len, double intercept,
                                               const double *x_rows, size_t n_rows, size_t x_len,
                                               const double *matrix, size_t dim, size_t n_observations,
                                               size_t n_params_effective, double residual_std_error,
                                               double confidence_level, int32_t interval_type, double *out_yhat,
                                               double *out_lower, double *out_upper, AnofoxError *out_error);

/* ============================================================================
 * Compact least-squares states (#266)
 * ============================================================================
 * A state is the versioned byte encoding of upstream's SufficientStats (counts,
 * weighted means, centred co-moments). Byte buffers are malloc'ed by the library
 * and freed with anofox_free_bytes. Rows are added in a canonical order and states
 * are merged in the order of their bytes, so the bytes are deterministic.
 */

/** State of the rows (x: x_count feature columns; weights.data NULL = unweighted). */
bool anofox_ls_state_from_rows(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxDataArray weights,
                               uint8_t **out_bytes, size_t *out_len, AnofoxError *out_error);

/** Merge n_states states (states[i] has lens[i] bytes). */
bool anofox_ls_state_merge(const uint8_t *const *states, const size_t *lens, size_t n_states, uint8_t **out_bytes,
                           size_t *out_len, AnofoxError *out_error);

/** Row count (incl. zero-weight rows), feature count and weighting of a state. */
bool anofox_ls_state_info(const uint8_t *bytes, size_t len, size_t *out_n_observations, size_t *out_n_features,
                          bool *out_weighted, AnofoxError *out_error);

/**
 * Fits from a state; outputs as anofox_ols_fit / anofox_wls_fit / anofox_ridge_fit. With inference,
 * out_vcov receives the dim x dim coefficient covariance (intercept first; free with
 * anofox_free_interval_matrix), else NULL and 0. out_inference, out_vcov and out_dim may be NULL.
 * hc_type and compute_loo need the rows and are rejected.
 */
bool anofox_ols_fit_from_state(const uint8_t *bytes, size_t len, AnofoxOlsOptions options,
                               AnofoxFitResultCore *out_core, AnofoxFitResultInference *out_inference,
                               double **out_vcov, size_t *out_dim, AnofoxError *out_error);
bool anofox_wls_fit_from_state(const uint8_t *bytes, size_t len, AnofoxWlsOptions options,
                               AnofoxFitResultCore *out_core, AnofoxFitResultInference *out_inference,
                               double **out_vcov, size_t *out_dim, AnofoxError *out_error);
bool anofox_ridge_fit_from_state(const uint8_t *bytes, size_t len, AnofoxRidgeOptions options,
                                 AnofoxFitResultCore *out_core, AnofoxFitResultInference *out_inference,
                                 double **out_vcov, size_t *out_dim, AnofoxError *out_error);

/** Free a buffer returned by the anofox_ls_state_* functions. */
void anofox_free_bytes(uint8_t *bytes);

/* ============================================================================
 * GLM (Generalized Linear Models) Functions
 * ============================================================================ */

/**
 * Poisson link function codes
 */
typedef enum {
	ANOFOX_POISSON_LINK_LOG = 0,
	ANOFOX_POISSON_LINK_IDENTITY = 1,
	ANOFOX_POISSON_LINK_SQRT = 2,
} AnofoxPoissonLink;

/**
 * Binomial link function codes
 */
typedef enum {
	ANOFOX_BINOMIAL_LINK_LOGIT = 0,
	ANOFOX_BINOMIAL_LINK_PROBIT = 1,
	ANOFOX_BINOMIAL_LINK_CLOGLOG = 2,
} AnofoxBinomialLink;

/**
 * GLM fit result (uses deviance instead of R-squared)
 */
typedef struct {
	/** Pointer to coefficients array */
	double *coefficients;
	/** Number of coefficients */
	size_t coefficients_len;
	/** Intercept value (NaN if no intercept) */
	double intercept;
	/** Model deviance (residual deviance) */
	double deviance;
	/** Null deviance */
	double null_deviance;
	/** Pseudo R-squared (1 - deviance/null_deviance) */
	double pseudo_r_squared;
	/** AIC */
	double aic;
	/** Dispersion parameter (if applicable) */
	double dispersion;
	/** Number of observations */
	size_t n_observations;
	/** Number of features */
	size_t n_features;
	/** Number of iterations to converge */
	uint32_t iterations;
	/** Whether the IRLS solver reached the convergence tolerance. Appended last
	    to preserve the ABI; every construction site must value-initialise. */
	bool converged;
	/** Model description, fit statistics and intercept inference (#152) */
	AnofoxModelSummaryFFI summary;
	/** Coefficient covariance (#158): row-major vcov_dim x vcov_dim, intercept
	    first, dispersion included; NULL without inference. Freed by
	    anofox_free_glm_result. Appended last to preserve the ABI. */
	double *vcov;
	size_t vcov_dim;
} AnofoxGlmFitResultCore;

/*
 * NOTE ON ADDING FIELDS TO THE OPTIONS STRUCTS BELOW
 *
 * These structs are hand-mirrored against crates/anofox-stats-ffi/src/types.rs and
 * are plain C PODs: declaring one without an initialiser leaves every field as
 * garbage. Callers therefore declare them as `AnofoxXxxOptions options {};` and
 * assign afterwards. When you add a field here you must add it in the same
 * position on the Rust side AND confirm every construction site still
 * value-initialises, otherwise Rust reads uninitialised memory. Adding `priors`
 * without doing so produced a multi-terabyte allocation attempt from a garbage
 * pointer/length pair.
 */

/**
 * Prior family code (mirrors PriorKindFFI).
 */
typedef enum {
	/** No prior on this coefficient. */
	ANOFOX_PRIOR_FLAT = 0,
	/** Gaussian prior; `scale` is the prior standard deviation. */
	ANOFOX_PRIOR_NORMAL = 1,
	/** Laplace prior; `scale` is the Laplace scale b. */
	ANOFOX_PRIOR_LAPLACE = 2,
} AnofoxPriorKind;

/**
 * One coefficient's prior.
 *
 * Positionally aligned with the design matrix (intercept first when one is
 * fitted). Feature names are resolved to positions on the C++ side, so they
 * never cross this boundary.
 */
typedef struct {
	AnofoxPriorKind kind;
	double loc;
	double scale;
} AnofoxPriorSpec;

/**
 * Coefficient covariance type (mirrors VcovTypeFFI).
 */
typedef enum {
	/** (X'WX + P)^-1 -- curvature of the log posterior at the mode. */
	ANOFOX_VCOV_LAPLACE = 0,
	/** (X'WX + P)^-1 X'WX (X'WX + P)^-1. */
	ANOFOX_VCOV_SANDWICH = 1,
	/** (X'WX)^-1 -- ignores the penalty. */
	ANOFOX_VCOV_NAIVE = 2,
} AnofoxVcovType;

/**
 * Poisson regression options
 */
typedef struct {
	/** Whether to fit intercept */
	bool fit_intercept;
	/** Link function */
	AnofoxPoissonLink link;
	/** Maximum iterations for IRLS */
	uint32_t max_iterations;
	/** Convergence tolerance */
	double tolerance;
	/** Whether to compute inference statistics */
	bool compute_inference;
	/** Confidence level for CIs */
	double confidence_level;
	/** L2 regularization parameter (0 = no regularization) */
	double lambda;
	/** Per-coefficient priors, or NULL for none. */
	const AnofoxPriorSpec *priors;
	size_t priors_len;
	/** How to compute the coefficient covariance. */
	AnofoxVcovType vcov;
	/** 1-based index into x of an offset column (0 = none). Added to the linear
	    predictor with coefficient fixed at 1 and dropped from the design. */
	size_t offset_column;
} AnofoxPoissonOptions;

/**
 * Binomial (logistic) regression options
 */
typedef struct {
	/** Whether to fit intercept */
	bool fit_intercept;
	/** Link function */
	AnofoxBinomialLink link;
	/** Maximum iterations for IRLS */
	uint32_t max_iterations;
	/** Convergence tolerance */
	double tolerance;
	/** Whether to compute inference statistics */
	bool compute_inference;
	/** Confidence level for CIs */
	double confidence_level;
	/** L2 regularization parameter (0 = no regularization) */
	double lambda;
	/** Per-coefficient priors, or NULL for none. */
	const AnofoxPriorSpec *priors;
	size_t priors_len;
	/** How to compute the coefficient covariance. */
	AnofoxVcovType vcov;
	/** 1-based index into x of an offset column (0 = none). Added to the linear
	    predictor with coefficient fixed at 1 and dropped from the design. */
	size_t offset_column;
} AnofoxBinomialOptions;

/**
 * Negative Binomial regression options
 */
typedef struct {
	/** Dispersion (theta). NaN means "estimate from the data". */
	double alpha;
	/** Whether to fit intercept */
	bool fit_intercept;
	/** Maximum iterations for IRLS */
	uint32_t max_iterations;
	/** Convergence tolerance */
	double tolerance;
	/** Whether to compute inference statistics */
	bool compute_inference;
	/** Confidence level for CIs */
	double confidence_level;
	/** L2 regularization parameter (0 = no regularization) */
	double lambda;
	/** Per-coefficient priors, or NULL for none. */
	const AnofoxPriorSpec *priors;
	size_t priors_len;
	/** How to compute the coefficient covariance. */
	AnofoxVcovType vcov;
	/** 1-based index into x of an offset column (0 = none). Added to the linear
	    predictor with coefficient fixed at 1 and dropped from the design. */
	size_t offset_column;
} AnofoxNegBinomialOptions;

/**
 * Tweedie regression options
 */
typedef struct {
	/** Whether to fit intercept */
	bool fit_intercept;
	/** Tweedie power parameter (1 < p < 2 for compound Poisson-Gamma) */
	double power;
	/** Maximum iterations for IRLS */
	uint32_t max_iterations;
	/** Convergence tolerance */
	double tolerance;
	/** Whether to compute inference statistics */
	bool compute_inference;
	/** Confidence level for CIs */
	double confidence_level;
	/** L2 regularization parameter (0 = no regularization) */
	double lambda;
	/** Per-coefficient priors, or NULL for none. */
	const AnofoxPriorSpec *priors;
	size_t priors_len;
	/** How to compute the coefficient covariance. */
	AnofoxVcovType vcov;
	/** 1-based index into x of an offset column (0 = none). Added to the linear
	    predictor with coefficient fixed at 1 and dropped from the design. */
	size_t offset_column;
} AnofoxTweedieOptions;

/**
 * Gamma GLM options (var_power = 2.0 fixed; log link).
 */
typedef struct {
	bool fit_intercept;
	uint32_t max_iterations;
	double tolerance;
	bool compute_inference;
	double confidence_level;
	double lambda;
	/** Per-coefficient priors, or NULL for none. */
	const AnofoxPriorSpec *priors;
	size_t priors_len;
	/** How to compute the coefficient covariance. */
	AnofoxVcovType vcov;
	/** 1-based index into x of an offset column (0 = none). Added to the linear
	    predictor with coefficient fixed at 1 and dropped from the design. */
	size_t offset_column;
} AnofoxGammaOptions;

/**
 * Fit a Gamma GLM (var_power = 2.0; log link).
 *
 * @param y Strictly positive response variable
 * @param x Pointer to array of feature arrays
 * @param x_count Number of feature arrays
 * @param options Gamma fitting options
 * @param out_result Output: core fit results (required)
 * @param out_inference Output: inference results (NULL if not needed)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_gamma_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxGammaOptions options,
                      AnofoxGlmFitResultCore *out_result, AnofoxFitResultInference *out_inference,
                      AnofoxError *out_error);

/**
 * Fit a Poisson regression model
 *
 * @param y Response variable array (counts)
 * @param x Pointer to array of feature arrays
 * @param x_count Number of feature arrays
 * @param options Poisson fitting options
 * @param out_core Output: core fit results (required)
 * @param out_inference Output: inference results (can be NULL if not needed)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_poisson_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxPoissonOptions options,
                        AnofoxGlmFitResultCore *out_core, AnofoxFitResultInference *out_inference,
                        AnofoxError *out_error);

/**
 * Fit a Binomial (logistic) regression model
 *
 * @param y Response variable array (0/1 binary outcomes)
 * @param x Pointer to array of feature arrays
 * @param x_count Number of feature arrays
 * @param options Binomial fitting options
 * @param out_core Output: core fit results (required)
 * @param out_inference Output: inference results (can be NULL if not needed)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_binomial_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxBinomialOptions options,
                         AnofoxGlmFitResultCore *out_core, AnofoxFitResultInference *out_inference,
                         AnofoxError *out_error);

/**
 * Fit a Negative Binomial regression model
 *
 * @param y Response variable array (overdispersed counts)
 * @param x Pointer to array of feature arrays
 * @param x_count Number of feature arrays
 * @param options Negative Binomial fitting options
 * @param out_core Output: core fit results (required)
 * @param out_inference Output: inference results (can be NULL if not needed)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_negbinomial_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count,
                            AnofoxNegBinomialOptions options, AnofoxGlmFitResultCore *out_core,
                            AnofoxFitResultInference *out_inference, AnofoxError *out_error);

/**
 * Fit a Tweedie regression model
 *
 * @param y Response variable array (non-negative with zeros)
 * @param x Pointer to array of feature arrays
 * @param x_count Number of feature arrays
 * @param options Tweedie fitting options
 * @param out_core Output: core fit results (required)
 * @param out_inference Output: inference results (can be NULL if not needed)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_tweedie_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxTweedieOptions options,
                        AnofoxGlmFitResultCore *out_core, AnofoxFitResultInference *out_inference,
                        AnofoxError *out_error);

/**
 * Binary Logistic regression options (logit link; classifier-oriented API).
 */
typedef struct {
	bool fit_intercept;
	bool compute_inference;
	double confidence_level;
	/** L2 (ridge) penalty strength. 0.0 = unpenalised. */
	double lambda;
	/** Classification threshold on P(y=1). */
	double threshold;
	uint32_t max_iterations;
	double tolerance;
	/** Per-coefficient priors, or NULL for none. */
	const AnofoxPriorSpec *priors;
	size_t priors_len;
	/** How to compute the coefficient covariance. */
	AnofoxVcovType vcov;
	/** 1-based index into x of an offset column (0 = none). Added to the linear
	    predictor with coefficient fixed at 1 and dropped from the design. */
	size_t offset_column;
} AnofoxLogisticOptions;

/**
 * Logistic-specific diagnostics. Both fields are scalars; no allocations
 * to free.
 */
typedef struct {
	/** Classification accuracy on the training data with the configured threshold. */
	double accuracy;
	/** Classification threshold actually used (echoed from options). */
	double threshold;
} AnofoxLogisticFitExtras;

/**
 * Fit a binary Logistic regression.
 *
 * @param y Binary response (0.0 or 1.0)
 * @param x Pointer to array of feature arrays
 * @param x_count Number of feature arrays
 * @param options Logistic fitting options
 * @param out_result Output: core fit results (required)
 * @param out_inference Output: inference results (NULL if not needed)
 * @param out_extras Output: accuracy + threshold echo (NULL if not needed)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_logistic_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxLogisticOptions options,
                         AnofoxGlmFitResultCore *out_result, AnofoxFitResultInference *out_inference,
                         AnofoxLogisticFitExtras *out_extras, AnofoxError *out_error);

/**
 * Free memory allocated for GLM core results
 */
void anofox_free_glm_result(AnofoxGlmFitResultCore *result);

/* ============================================================================
 * ALM (Augmented Linear Models) Functions
 * ============================================================================ */

/**
 * ALM distribution codes
 */
typedef enum {
	ANOFOX_ALM_DIST_NORMAL = 0,
	ANOFOX_ALM_DIST_LAPLACE = 1,
	ANOFOX_ALM_DIST_STUDENT_T = 2,
	ANOFOX_ALM_DIST_LOGISTIC = 3,
	ANOFOX_ALM_DIST_ASYMMETRIC_LAPLACE = 4,
	ANOFOX_ALM_DIST_GENERALISED_NORMAL = 5,
	ANOFOX_ALM_DIST_S = 6,
	ANOFOX_ALM_DIST_LOG_NORMAL = 7,
	ANOFOX_ALM_DIST_LOG_LAPLACE = 8,
	ANOFOX_ALM_DIST_LOG_S = 9,
	ANOFOX_ALM_DIST_LOG_GENERALISED_NORMAL = 10,
	ANOFOX_ALM_DIST_FOLDED_NORMAL = 11,
	ANOFOX_ALM_DIST_RECTIFIED_NORMAL = 12,
	ANOFOX_ALM_DIST_BOX_COX_NORMAL = 13,
	ANOFOX_ALM_DIST_GAMMA = 14,
	ANOFOX_ALM_DIST_INVERSE_GAUSSIAN = 15,
	ANOFOX_ALM_DIST_EXPONENTIAL = 16,
	ANOFOX_ALM_DIST_BETA = 17,
	ANOFOX_ALM_DIST_LOGIT_NORMAL = 18,
	ANOFOX_ALM_DIST_POISSON = 19,
	ANOFOX_ALM_DIST_NEGATIVE_BINOMIAL = 20,
	ANOFOX_ALM_DIST_BINOMIAL = 21,
	ANOFOX_ALM_DIST_GEOMETRIC = 22,
	ANOFOX_ALM_DIST_CUMULATIVE_LOGISTIC = 23,
	ANOFOX_ALM_DIST_CUMULATIVE_NORMAL = 24,
} AnofoxAlmDistribution;

/**
 * ALM loss function codes
 */
typedef enum {
	ANOFOX_ALM_LOSS_LIKELIHOOD = 0,
	ANOFOX_ALM_LOSS_MSE = 1,
	ANOFOX_ALM_LOSS_MAE = 2,
	ANOFOX_ALM_LOSS_HAM = 3,
	ANOFOX_ALM_LOSS_ROLE = 4,
} AnofoxAlmLoss;

/**
 * ALM options
 */
typedef struct {
	/** Whether to fit intercept */
	bool fit_intercept;
	/** Distribution family */
	AnofoxAlmDistribution distribution;
	/** Loss function */
	AnofoxAlmLoss loss;
	/** Maximum iterations */
	uint32_t max_iterations;
	/** Convergence tolerance */
	double tolerance;
	/** Quantile for AsymmetricLaplace (0-1) */
	double quantile;
	/** ROLE trim fraction */
	double role_trim;
	/** Whether to compute inference statistics */
	bool compute_inference;
	/** Confidence level for CIs */
	double confidence_level;
} AnofoxAlmOptions;

/**
 * ALM fit result
 */
typedef struct {
	/** Pointer to coefficients array */
	double *coefficients;
	/** Number of coefficients */
	size_t coefficients_len;
	/** Intercept value (NaN if no intercept) */
	double intercept;
	/** Log-likelihood */
	double log_likelihood;
	/** AIC */
	double aic;
	/** BIC */
	double bic;
	/** Scale parameter */
	double scale;
	/** Number of observations */
	size_t n_observations;
	/** Number of features */
	size_t n_features;
	/** Number of iterations to converge */
	uint32_t iterations;
	/** Model description, fit statistics and intercept inference (#152) */
	AnofoxModelSummaryFFI summary;
} AnofoxAlmFitResultCore;

/**
 * Fit an Augmented Linear Model (ALM)
 *
 * ALM supports 24 error distribution families and multiple loss functions.
 *
 * @param y Response variable array
 * @param x Pointer to array of feature arrays
 * @param x_count Number of feature arrays
 * @param options ALM fitting options
 * @param out_core Output: core fit results (required)
 * @param out_inference Output: inference results (can be NULL if not needed)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_alm_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxAlmOptions options,
                    AnofoxAlmFitResultCore *out_core, AnofoxFitResultInference *out_inference, AnofoxError *out_error);

/**
 * Free memory allocated for ALM core results
 */
void anofox_free_alm_result(AnofoxAlmFitResultCore *result);

/* ============================================================================
 * BLS (Bounded Least Squares) Functions
 * ============================================================================ */

/**
 * BLS options
 */
typedef struct {
	/** Whether to fit intercept */
	bool fit_intercept;
	/** Pointer to lower bounds (NULL = no lower bounds, single value = apply to all) */
	const double *lower_bounds;
	/** Number of lower bounds (0 = no bounds, 1 = single value for all) */
	size_t lower_bounds_len;
	/** Pointer to upper bounds (NULL = no upper bounds, single value = apply to all) */
	const double *upper_bounds;
	/** Number of upper bounds (0 = no bounds, 1 = single value for all) */
	size_t upper_bounds_len;
	/** Maximum iterations */
	uint32_t max_iterations;
	/** Convergence tolerance */
	double tolerance;
} AnofoxBlsOptions;

/**
 * BLS fit result
 */
typedef struct {
	/** Pointer to coefficients array */
	double *coefficients;
	/** Number of coefficients */
	size_t coefficients_len;
	/** Intercept value (NaN if no intercept) */
	double intercept;
	/** Sum of squared residuals */
	double ssr;
	/** R-squared */
	double r_squared;
	/** Number of observations */
	size_t n_observations;
	/** Number of features */
	size_t n_features;
	/** Number of active constraints */
	size_t n_active_constraints;
	/** Pointer to at_lower_bound flags */
	bool *at_lower_bound;
	/** Pointer to at_upper_bound flags */
	bool *at_upper_bound;
	/** Model description and fit statistics (#152) */
	AnofoxModelSummaryFFI summary;
} AnofoxBlsFitResultCore;

/**
 * Fit a Bounded Least Squares (BLS) model
 *
 * Solves: minimize ||Xβ - y||² subject to lower ≤ β ≤ upper
 *
 * @param y Response variable array
 * @param x Pointer to array of feature arrays
 * @param x_count Number of feature arrays
 * @param options BLS fitting options (use lower_bounds=0, lower_bounds_len=0, etc. for NNLS)
 * @param out_core Output: core fit results (required)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_bls_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxBlsOptions options,
                    AnofoxBlsFitResultCore *out_core, AnofoxError *out_error);

/**
 * Fit a Non-Negative Least Squares (NNLS) model
 *
 * Convenience function for BLS with lower bounds of 0 for all coefficients.
 *
 * @param y Response variable array
 * @param x Pointer to array of feature arrays
 * @param x_count Number of feature arrays
 * @param out_core Output: core fit results (required)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_nnls_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxBlsFitResultCore *out_core,
                     AnofoxError *out_error);

/**
 * Free memory allocated for BLS core results
 */
void anofox_free_bls_result(AnofoxBlsFitResultCore *result);

/* ============================================================================
 * PLS (Partial Least Squares) Functions
 * ============================================================================ */

/**
 * PLS options
 */
typedef struct {
	/** Number of components to extract */
	size_t n_components;
	/** Whether to fit intercept */
	bool fit_intercept;
} AnofoxPlsOptions;

/**
 * PLS fit result
 */
typedef struct {
	/** Pointer to coefficients array */
	double *coefficients;
	/** Number of coefficients */
	size_t coefficients_len;
	/** Intercept value (NaN if no intercept) */
	double intercept;
	/** R-squared */
	double r_squared;
	/** Number of components used */
	size_t n_components;
	/** Number of observations */
	size_t n_observations;
	/** Number of features */
	size_t n_features;
	/** Model description and fit statistics (#152) */
	AnofoxModelSummaryFFI summary;
} AnofoxPlsFitResultCore;

/**
 * Fit a PLS (Partial Least Squares) regression model
 *
 * @param y Response variable array
 * @param x Pointer to array of feature arrays
 * @param x_count Number of feature arrays
 * @param options PLS fitting options
 * @param out_core Output: core fit results (required)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_pls_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxPlsOptions options,
                    AnofoxPlsFitResultCore *out_core, AnofoxError *out_error);

/**
 * Free memory allocated for PLS core results
 */
void anofox_free_pls_result(AnofoxPlsFitResultCore *result);

/* ============================================================================
 * Isotonic Regression Functions
 * ============================================================================ */

/**
 * Isotonic regression options
 */
typedef struct {
	/** Whether the function should be increasing (true) or decreasing (false) */
	bool increasing;
} AnofoxIsotonicOptions;

/**
 * Isotonic fit result
 */
typedef struct {
	/** Pointer to fitted values array (same length as input) */
	double *fitted_values;
	/** Number of fitted values */
	size_t fitted_values_len;
	/** R-squared */
	double r_squared;
	/** Number of observations */
	size_t n_observations;
	/** Whether increasing constraint was used */
	bool increasing;
	/** Model description and fit statistics (#152) */
	AnofoxModelSummaryFFI summary;
} AnofoxIsotonicFitResultCore;

/**
 * Fit an Isotonic regression model
 *
 * @param x Input values array (1D)
 * @param y Response variable array
 * @param options Isotonic fitting options
 * @param out_core Output: core fit results (required)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_isotonic_fit(AnofoxDataArray x, AnofoxDataArray y, AnofoxIsotonicOptions options,
                         AnofoxIsotonicFitResultCore *out_core, AnofoxError *out_error);

/**
 * Free memory allocated for Isotonic core results
 */
void anofox_free_isotonic_result(AnofoxIsotonicFitResultCore *result);

/* ============================================================================
 * Quantile Regression Functions
 * ============================================================================ */

/**
 * Quantile regression options
 */
typedef struct {
	/** Quantile to estimate (0 < tau < 1, e.g., 0.5 for median) */
	double tau;
	/** Whether to fit intercept */
	bool fit_intercept;
	/** Maximum iterations */
	uint32_t max_iterations;
	/** Convergence tolerance */
	double tolerance;
	/** L1 penalty on the slopes (sklearn QuantileRegressor(alpha), #305); 0 = unpenalized */
	double alpha;
} AnofoxQuantileOptions;

/**
 * Quantile fit result
 */
typedef struct {
	/** Pointer to coefficients array */
	double *coefficients;
	/** Number of coefficients */
	size_t coefficients_len;
	/** Intercept value (NaN if no intercept) */
	double intercept;
	/** Quantile estimated */
	double tau;
	/** Number of observations */
	size_t n_observations;
	/** Number of features */
	size_t n_features;
	/** Model description and fit statistics (#152) */
	AnofoxModelSummaryFFI summary;
} AnofoxQuantileFitResultCore;

/**
 * Fit a Quantile regression model
 *
 * @param y Response variable array
 * @param x Pointer to array of feature arrays
 * @param x_count Number of feature arrays
 * @param options Quantile fitting options
 * @param out_core Output: core fit results (required)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_quantile_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxQuantileOptions options,
                         AnofoxQuantileFitResultCore *out_core, AnofoxError *out_error);

/**
 * Free memory allocated for Quantile core results
 */
void anofox_free_quantile_result(AnofoxQuantileFitResultCore *result);

/* ============================================================================
 * AID (Automatic Identification of Demand) Functions
 * ============================================================================ */

/**
 * Outlier detection method codes
 */
typedef enum {
	ANOFOX_AID_OUTLIER_ZSCORE = 0,
	ANOFOX_AID_OUTLIER_IQR = 1,
} AnofoxAidOutlierMethod;

/**
 * AID options
 */
typedef struct {
	/** Zero proportion threshold for intermittent classification (default: 0.3) */
	double intermittent_threshold;
	/** Outlier detection method */
	AnofoxAidOutlierMethod outlier_method;
} AnofoxAidOptions;

/**
 * AID classification result
 */
typedef struct {
	/** Demand type string ("regular" or "intermittent") - must be freed */
	char *demand_type;
	/** Whether demand is intermittent */
	bool is_intermittent;
	/** Best-fit distribution name - must be freed */
	char *distribution;
	/** Mean of values */
	double mean;
	/** Variance of values */
	double variance;
	/** Proportion of zero values */
	double zero_proportion;
	/** Number of observations */
	size_t n_observations;
	/** Whether stockouts were detected */
	bool has_stockouts;
	/** Whether new product pattern was detected */
	bool is_new_product;
	/** Whether obsolete product pattern was detected */
	bool is_obsolete_product;
	/** Number of stockout observations */
	size_t stockout_count;
	/** Number of new product observations */
	size_t new_product_count;
	/** Number of obsolete product observations */
	size_t obsolete_product_count;
	/** Number of high outlier observations */
	size_t high_outlier_count;
	/** Number of low outlier observations */
	size_t low_outlier_count;
} AnofoxAidResult;

/**
 * Per-observation anomaly flags for AID
 */
typedef struct {
	/** Unexpected zero in positive demand (stockout) */
	bool stockout;
	/** Leading zeros pattern (new product) */
	bool new_product;
	/** Trailing zeros pattern (obsolete product) */
	bool obsolete_product;
	/** Unusually high value */
	bool high_outlier;
	/** Unusually low value */
	bool low_outlier;
} AnofoxAidAnomalyFlags;

/**
 * AID anomaly result (array of per-observation flags)
 */
typedef struct {
	/** Pointer to array of anomaly flags - must be freed */
	AnofoxAidAnomalyFlags *flags;
	/** Number of observations */
	size_t len;
} AnofoxAidAnomalyResult;

/**
 * Compute AID (Automatic Identification of Demand) classification
 *
 * Classifies demand patterns as regular or intermittent, identifies best-fit
 * distribution, and counts various anomaly types.
 *
 * @param y Time series data (must preserve order)
 * @param options AID options
 * @param out_result Output: AID classification result (required)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_aid(AnofoxDataArray y, AnofoxAidOptions options, AnofoxAidResult *out_result, AnofoxError *out_error);

/**
 * Compute AID per-observation anomaly flags
 *
 * Returns anomaly flags for each observation in the input, maintaining
 * the same order as the input.
 *
 * @param y Time series data (must preserve order)
 * @param options AID options
 * @param out_result Output: anomaly flags array (required)
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_aid_anomaly(AnofoxDataArray y, AnofoxAidOptions options, AnofoxAidAnomalyResult *out_result,
                        AnofoxError *out_error);

/**
 * Free memory allocated for AID result
 */
void anofox_free_aid_result(AnofoxAidResult *result);

/**
 * Free memory allocated for AID anomaly result
 */
void anofox_free_aid_anomaly_result(AnofoxAidAnomalyResult *result);

/* ============================================================================
 * Statistical Hypothesis Testing Functions
 * ============================================================================ */

/**
 * Alternative hypothesis codes
 */
typedef enum {
	ANOFOX_ALTERNATIVE_TWO_SIDED = 0,
	ANOFOX_ALTERNATIVE_LESS = 1,
	ANOFOX_ALTERNATIVE_GREATER = 2,
} AnofoxAlternative;

/**
 * Generic test result structure
 */
typedef struct {
	/** Test statistic */
	double statistic;
	/** p-value */
	double p_value;
	/** Degrees of freedom (NaN if N/A) */
	double df;
	/** Effect size (NaN if N/A) */
	double effect_size;
	/** Confidence interval lower bound */
	double ci_lower;
	/** Confidence interval upper bound */
	double ci_upper;
	/** Confidence level */
	double confidence_level;
	/** Total sample size */
	size_t n;
	/** Group 1 sample size */
	size_t n1;
	/** Group 2 sample size */
	size_t n2;
	/** Alternative hypothesis */
	AnofoxAlternative alternative;
	/** Method name - must be freed */
	char *method;
	/** Unified test fields (#159); keep last */
	AnofoxTestSummaryFFI unified;
} AnofoxTestResult;

/**
 * ANOVA result structure
 */
typedef struct {
	/** F statistic */
	double f_statistic;
	/** p-value */
	double p_value;
	/** Between-groups df */
	size_t df_between;
	/** Within-groups df */
	size_t df_within;
	/** Between-groups SS */
	double ss_between;
	/** Within-groups SS */
	double ss_within;
	/** Number of groups */
	size_t n_groups;
	/** Total sample size */
	size_t n;
	/** Method name - must be freed */
	char *method;
	/** Unified test fields (#159); keep last */
	AnofoxTestSummaryFFI unified;
} AnofoxAnovaResult;

/**
 * Correlation result structure
 */
typedef struct {
	/** Correlation coefficient */
	double r;
	/** Test statistic */
	double statistic;
	/** p-value */
	double p_value;
	/** CI lower bound */
	double ci_lower;
	/** CI upper bound */
	double ci_upper;
	/** Confidence level */
	double confidence_level;
	/** Sample size */
	size_t n;
	/** Method name - must be freed */
	char *method;
	/** Unified test fields (#159); keep last */
	AnofoxTestSummaryFFI unified;
} AnofoxCorrelationResult;

/**
 * Chi-square result structure
 */
typedef struct {
	/** Chi-square statistic */
	double statistic;
	/** p-value */
	double p_value;
	/** Degrees of freedom */
	size_t df;
	/** Method name - must be freed */
	char *method;
	/** Unified test fields (#159); keep last */
	AnofoxTestSummaryFFI unified;
} AnofoxChiSquareResult;

/**
 * TOST (equivalence) result structure
 */
typedef struct {
	/** Lower bound test statistic */
	double t_lower;
	/** Upper bound test statistic */
	double t_upper;
	/** p-value for lower test */
	double p_lower;
	/** p-value for upper test */
	double p_upper;
	/** Overall p-value */
	double p_value;
	/** Degrees of freedom */
	double df;
	/** Point estimate */
	double estimate;
	/** CI lower bound */
	double ci_lower;
	/** CI upper bound */
	double ci_upper;
	/** Equivalence bound lower */
	double bound_lower;
	/** Equivalence bound upper */
	double bound_upper;
	/** Whether equivalence was established */
	bool equivalent;
	/** Sample size */
	size_t n;
	/** Method name - must be freed */
	char *method;
	/** Unified test fields (#159); keep last */
	AnofoxTestSummaryFFI unified;
} AnofoxTostResult;

/**
 * T-test options
 */
typedef struct {
	/** Alternative hypothesis */
	AnofoxAlternative alternative;
	/** Confidence level */
	double confidence_level;
	/** Use equal variance (Student's t) vs Welch */
	bool var_equal;
	/** Hypothesized mean difference */
	double mu;
} AnofoxTTestOptions;

/**
 * Mann-Whitney U test options
 */
typedef struct {
	/** Alternative hypothesis */
	AnofoxAlternative alternative;
	/** Use exact distribution */
	bool exact;
	/** Apply continuity correction */
	bool continuity_correction;
	/** Confidence level */
	double confidence_level;
	/** Hypothesized location shift */
	double mu;
} AnofoxMannWhitneyOptions;

/**
 * Correlation test options
 */
typedef struct {
	/** Alternative hypothesis */
	AnofoxAlternative alternative;
	/** Confidence level */
	double confidence_level;
} AnofoxCorrelationOptions;

/**
 * Kendall tau type codes
 */
typedef enum {
	ANOFOX_KENDALL_TAU_A = 0,
	ANOFOX_KENDALL_TAU_B = 1,
	ANOFOX_KENDALL_TAU_C = 2,
} AnofoxKendallType;

/**
 * Kendall correlation options
 */
typedef struct {
	/** Alternative hypothesis */
	AnofoxAlternative alternative;
	/** Tau type */
	AnofoxKendallType tau_type;
	/** Confidence level */
	double confidence_level;
} AnofoxKendallOptions;

/**
 * Chi-square test options
 */
typedef struct {
	/** Apply Yates correction */
	bool correction;
} AnofoxChiSquareOptions;

/**
 * Fisher's exact test options
 */
typedef struct {
	/** Alternative hypothesis */
	AnofoxAlternative alternative;
	/** Confidence level */
	double confidence_level;
} AnofoxFisherExactOptions;

/**
 * Energy distance test options
 */
typedef struct {
	/** Number of permutations */
	size_t n_permutations;
	/** Random seed (0 = random) */
	uint64_t seed;
	/** Whether seed is set */
	bool has_seed;
} AnofoxEnergyDistanceOptions;

/**
 * MMD test options
 */
typedef struct {
	/** Number of permutations */
	size_t n_permutations;
	/** Random seed (0 = random) */
	uint64_t seed;
	/** Whether seed is set */
	bool has_seed;
} AnofoxMmdOptions;

/**
 * TOST options
 */
typedef struct {
	/** Lower equivalence bound */
	double bound_lower;
	/** Upper equivalence bound */
	double bound_upper;
	/** Significance level */
	double alpha;
	/** Use pooled variance (for two-sample t-test) */
	bool pooled;
} AnofoxTostOptions;

/**
 * Brunner-Munzel test options
 */
typedef struct {
	/** Alternative hypothesis */
	AnofoxAlternative alternative;
	/** Confidence level */
	double confidence_level;
} AnofoxBrunnerMunzelOptions;

/* --- Test Functions --- */

/**
 * Two-sample t-test
 */
bool anofox_t_test(AnofoxDataArray group1, AnofoxDataArray group2, AnofoxTTestOptions options,
                   AnofoxTestResult *out_result, AnofoxError *out_error);

/**
 * One-sample t-test of H0: mean(x) = options.mu (#299); var_equal is ignored.
 * effect_size is Cohen's d = (mean - mu) / sd.
 */
bool anofox_t_test_one_sample(AnofoxDataArray x, AnofoxTTestOptions options, AnofoxTestResult *out_result,
                              AnofoxError *out_error);

/**
 * Paired t-test of H0: mean(x - y) = options.mu (#299); var_equal is ignored.
 * effect_size is Cohen's d_z = (mean(d) - mu) / sd(d).
 */
bool anofox_t_test_paired(AnofoxDataArray x, AnofoxDataArray y, AnofoxTTestOptions options,
                          AnofoxTestResult *out_result, AnofoxError *out_error);

/**
 * Shapiro-Wilk normality test
 */
bool anofox_shapiro_wilk(AnofoxDataArray data, AnofoxTestResult *out_result, AnofoxError *out_error);

/**
 * D'Agostino K-squared normality test
 */
bool anofox_dagostino_k2(AnofoxDataArray data, AnofoxTestResult *out_result, AnofoxError *out_error);

/**
 * Pearson correlation test
 */
bool anofox_pearson_cor(AnofoxDataArray x, AnofoxDataArray y, AnofoxCorrelationOptions options,
                        AnofoxCorrelationResult *out_result, AnofoxError *out_error);

/**
 * Spearman correlation test
 */
bool anofox_spearman_cor(AnofoxDataArray x, AnofoxDataArray y, AnofoxCorrelationOptions options,
                         AnofoxCorrelationResult *out_result, AnofoxError *out_error);

/**
 * Kendall correlation test
 */
bool anofox_kendall_cor(AnofoxDataArray x, AnofoxDataArray y, AnofoxKendallOptions options,
                        AnofoxCorrelationResult *out_result, AnofoxError *out_error);

/**
 * Mann-Whitney U test
 */
bool anofox_mann_whitney_u(AnofoxDataArray group1, AnofoxDataArray group2, AnofoxMannWhitneyOptions options,
                           AnofoxTestResult *out_result, AnofoxError *out_error);

/**
 * Brunner-Munzel test
 */
bool anofox_brunner_munzel(AnofoxDataArray group1, AnofoxDataArray group2, AnofoxBrunnerMunzelOptions options,
                           AnofoxTestResult *out_result, AnofoxError *out_error);

/**
 * One-way ANOVA
 */
bool anofox_one_way_anova(AnofoxDataArray values, AnofoxDataArray groups, AnofoxAnovaResult *out_result,
                          AnofoxError *out_error);

/**
 * Kruskal-Wallis H test
 */
bool anofox_kruskal_wallis(AnofoxDataArray values, AnofoxDataArray groups, AnofoxTestResult *out_result,
                           AnofoxError *out_error);

/**
 * Chi-square test for independence
 */
bool anofox_chisq_test(AnofoxDataArray row_var, AnofoxDataArray col_var, AnofoxChiSquareOptions options,
                       AnofoxChiSquareResult *out_result, AnofoxError *out_error);

/**
 * Fisher's exact test (2x2 tables)
 *
 * statistic/effect_size = sample odds ratio ad/bc; CI = Woolf (log-odds Wald) interval
 * at options.confidence_level, which must be in (0, 1) (InvalidInput otherwise).
 */
bool anofox_fisher_exact(size_t a, size_t b, size_t c, size_t d, AnofoxFisherExactOptions options,
                         AnofoxTestResult *out_result, AnofoxError *out_error);

/**
 * Fisher's exact test (2x2 tables) with R fisher.test semantics
 *
 * statistic/effect_size = conditional maximum-likelihood odds ratio; ci_lower/ci_upper =
 * exact conditional interval at options.confidence_level (one-sided for less/greater:
 * [0, U] / [L, Inf)). Valid for any table with n >= 1. confidence_level must be in (0, 1).
 */
bool anofox_fisher_exact_conditional(size_t a, size_t b, size_t c, size_t d, AnofoxFisherExactOptions options,
                                     AnofoxTestResult *out_result, AnofoxError *out_error);

/**
 * Energy distance test
 */
bool anofox_energy_distance(AnofoxDataArray group1, AnofoxDataArray group2, AnofoxEnergyDistanceOptions options,
                            AnofoxTestResult *out_result, AnofoxError *out_error);

/**
 * Maximum Mean Discrepancy (MMD) test
 */
bool anofox_mmd(AnofoxDataArray group1, AnofoxDataArray group2, AnofoxMmdOptions options, AnofoxTestResult *out_result,
                AnofoxError *out_error);

/**
 * TOST two-sample t-test for equivalence
 */
bool anofox_tost_t_test(AnofoxDataArray group1, AnofoxDataArray group2, AnofoxTostOptions options,
                        AnofoxTostResult *out_result, AnofoxError *out_error);

/**
 * TOST paired t-test for equivalence
 */
bool anofox_tost_t_test_paired(AnofoxDataArray x, AnofoxDataArray y, AnofoxTostOptions options,
                               AnofoxTostResult *out_result, AnofoxError *out_error);

/**
 * TOST correlation method codes
 */
typedef enum {
	ANOFOX_TOST_COR_PEARSON = 0,
	ANOFOX_TOST_COR_SPEARMAN = 1,
} AnofoxTostCorrelationMethod;

/**
 * TOST correlation test for equivalence to zero
 */
bool anofox_tost_correlation(AnofoxDataArray x, AnofoxDataArray y, double rho_null, double bound_lower,
                             double bound_upper, double alpha, AnofoxTostCorrelationMethod method,
                             AnofoxTostResult *out_result, AnofoxError *out_error);

/**
 * Options of the TOST variants (#293); each kind reads only its own fields.
 */
typedef struct {
	double bound_lower;
	double bound_upper;
	double alpha;
	/** One-sample t: hypothesised mean. One proportion: hypothesised p0. */
	double mu;
	/** Yuen: proportion trimmed from each tail */
	double trim;
	/** Bootstrap: number of resamples */
	size_t n_bootstrap;
	uint64_t seed;
	bool has_seed;
} AnofoxTostVariantOptions;

/** Kinds of anofox_tost_variant */
typedef enum {
	ANOFOX_TOST_ONE_SAMPLE = 0,
	ANOFOX_TOST_WILCOXON_PAIRED = 1,
	ANOFOX_TOST_WILCOXON_TWO_SAMPLE = 2,
	ANOFOX_TOST_YUEN = 3,
	ANOFOX_TOST_BOOTSTRAP = 4,
} AnofoxTostVariantKind;

/**
 * TOST variant on samples (NaN entries dropped). One-sample t tests x against
 * options.mu (y ignored); paired Wilcoxon pairs x and y; the others compare x
 * (group 1) with y (group 2).
 */
bool anofox_tost_variant(int32_t kind, AnofoxDataArray x, AnofoxDataArray y, AnofoxTostVariantOptions options,
                         AnofoxTostResult *out_result, AnofoxError *out_error);

/**
 * TOST for one proportion (two_sample false: successes1 of trials1 against
 * options.mu as p0) or for the difference of two proportions (group 1 - group 2).
 */
bool anofox_tost_prop(size_t successes1, size_t trials1, size_t successes2, size_t trials2, bool two_sample,
                      AnofoxTostVariantOptions options, AnofoxTostResult *out_result, AnofoxError *out_error);

/**
 * Wilcoxon signed-rank test options
 */
typedef struct {
	/** Alternative hypothesis */
	AnofoxAlternative alternative;
	/** Use exact distribution */
	bool exact;
	/** Apply continuity correction */
	bool continuity_correction;
	/** Confidence level for CI */
	double confidence_level;
	/** Hypothesized median */
	double mu;
} AnofoxWilcoxonOptions;

/**
 * Wilcoxon signed-rank test for paired samples
 */
bool anofox_wilcoxon_signed_rank(AnofoxDataArray x, AnofoxDataArray y, AnofoxWilcoxonOptions options,
                                 AnofoxTestResult *out_result, AnofoxError *out_error);

/* --- Categorical Tests --- */

/**
 * Proportion test result
 */
typedef struct {
	/** Test statistic (z) */
	double statistic;
	/** p-value */
	double p_value;
	/** Estimated proportion */
	double estimate;
	/** Confidence interval lower bound */
	double ci_lower;
	/** Confidence interval upper bound */
	double ci_upper;
	/** Sample size */
	size_t n;
	/** Alternative hypothesis */
	AnofoxAlternative alternative;
	/** Method name (must be freed) */
	char *method;
	/** Unified test fields (#159); keep last */
	AnofoxTestSummaryFFI unified;
} AnofoxPropTestResult;

/**
 * Cohen's kappa result
 */
typedef struct {
	/** Kappa coefficient */
	double kappa;
	/** Standard error */
	double se;
	/** Confidence interval lower bound */
	double ci_lower;
	/** Confidence interval upper bound */
	double ci_upper;
	/** z-statistic */
	double z;
	/** p-value */
	double p_value;
	/** Unified test fields (#159); keep last */
	AnofoxTestSummaryFFI unified;
} AnofoxKappaResult;

/**
 * Chi-square goodness-of-fit test
 */
bool anofox_chisq_goodness_of_fit(const size_t *observed, size_t observed_len, const double *expected,
                                  size_t expected_len, AnofoxChiSquareResult *out_result, AnofoxError *out_error);

/**
 * One-sample proportion z-test
 */
bool anofox_prop_test_one(size_t successes, size_t trials, double p0, AnofoxAlternative alternative,
                          AnofoxPropTestResult *out_result, AnofoxError *out_error);

/**
 * Two-sample proportion z-test
 */
bool anofox_prop_test_two(size_t successes1, size_t trials1, size_t successes2, size_t trials2,
                          AnofoxAlternative alternative, bool correction, AnofoxPropTestResult *out_result,
                          AnofoxError *out_error);

/**
 * Exact binomial test
 */
bool anofox_binom_test(size_t successes, size_t trials, double p0, AnofoxAlternative alternative,
                       AnofoxPropTestResult *out_result, AnofoxError *out_error);

/**
 * Proportion / binomial tests with the confidence level of the reported
 * interval (in (0, 1)); the functions above use 0.95.
 */
bool anofox_prop_test_one_with_conf_level(size_t successes, size_t trials, double p0, AnofoxAlternative alternative,
                                          double confidence_level, AnofoxPropTestResult *out_result,
                                          AnofoxError *out_error);
bool anofox_prop_test_two_with_conf_level(size_t successes1, size_t trials1, size_t successes2, size_t trials2,
                                          AnofoxAlternative alternative, bool correction, double confidence_level,
                                          AnofoxPropTestResult *out_result, AnofoxError *out_error);
bool anofox_binom_test_with_conf_level(size_t successes, size_t trials, double p0, AnofoxAlternative alternative,
                                       double confidence_level, AnofoxPropTestResult *out_result,
                                       AnofoxError *out_error);

/**
 * Cramer's V effect size for contingency tables
 */
bool anofox_cramers_v(const size_t *table, const size_t *row_lengths, size_t n_rows, double *out_result,
                      AnofoxError *out_error);

/**
 * Cohen's kappa for inter-rater agreement
 */
bool anofox_cohen_kappa(const size_t *table, const size_t *row_lengths, size_t n_rows, bool weighted,
                        AnofoxKappaResult *out_result, AnofoxError *out_error);

/**
 * G-test (log-likelihood ratio test) for contingency tables
 * Takes a flattened contingency table (row-major order)
 */
bool anofox_g_test(const size_t *table, const size_t *row_lengths, size_t n_rows, AnofoxChiSquareResult *out_result,
                   AnofoxError *out_error);

/**
 * McNemar's test for paired categorical data
 * Takes a 2x2 contingency table as 4 cell counts: a, b, c, d
 */
bool anofox_mcnemar_test(size_t a, size_t b, size_t c, size_t d, bool correction, bool exact,
                         AnofoxChiSquareResult *out_result, AnofoxError *out_error);

/**
 * Phi coefficient for 2x2 contingency tables
 * Takes a 2x2 contingency table as 4 cell counts: a, b, c, d
 */
bool anofox_phi_coefficient(size_t a, size_t b, size_t c, size_t d, double *out_result, AnofoxError *out_error);

/**
 * Contingency coefficient (Pearson's C)
 */
bool anofox_contingency_coef(const size_t *table, const size_t *row_lengths, size_t n_rows, double *out_result,
                             AnofoxError *out_error);

/**
 * Free memory allocated by proportion test result functions
 */
void anofox_free_prop_test_result(AnofoxPropTestResult *result);

/* --- Correlation Tests --- */

/**
 * Distance correlation result
 */
typedef struct {
	/** Distance correlation coefficient */
	double dcor;
	/** Distance covariance */
	double dcov;
	/** Distance variance of x */
	double dvar_x;
	/** Distance variance of y */
	double dvar_y;
	/** Sample size */
	size_t n;
} AnofoxDistanceCorResult;

/**
 * Distance correlation
 */
bool anofox_distance_cor(AnofoxDataArray x, AnofoxDataArray y, AnofoxDistanceCorResult *out_result,
                         AnofoxError *out_error);

/**
 * Distance correlation test with permutations
 */
bool anofox_distance_cor_test(AnofoxDataArray x, AnofoxDataArray y, size_t n_permutations, AnofoxTestResult *out_result,
                              AnofoxError *out_error);

/**
 * Distance correlation test with permutations and optional seed
 *
 * Same as anofox_distance_cor_test; when has_seed is true the permutation RNG is
 * seeded with seed so the p-value is reproducible.
 */
bool anofox_distance_cor_test_seeded(AnofoxDataArray x, AnofoxDataArray y, size_t n_permutations, uint64_t seed,
                                     bool has_seed, AnofoxTestResult *out_result, AnofoxError *out_error);

/**
 * ICC type codes (Shrout & Fleiss 1979; psych::ICC naming)
 */
typedef enum {
	ANOFOX_ICC_SINGLE = 0,  /* ICC1: one-way random, single rater */
	ANOFOX_ICC_AVERAGE = 1, /* ICC1k: one-way random, average of k raters */
	ANOFOX_ICC_ICC2 = 2,    /* two-way random, absolute agreement, single rater */
	ANOFOX_ICC_ICC3 = 3,    /* two-way mixed, consistency, single rater */
	ANOFOX_ICC_ICC2K = 4,   /* two-way random, absolute agreement, average of k raters */
	ANOFOX_ICC_ICC3K = 5,   /* two-way mixed, consistency, average of k raters */
} AnofoxIccType;

/**
 * ICC result
 */
typedef struct {
	/** ICC value */
	double icc;
	/** F-statistic */
	double f_statistic;
	/** Lower CI bound */
	double ci_lower;
	/** Upper CI bound */
	double ci_upper;
	/** Confidence level */
	double confidence_level;
	/** Number of subjects */
	size_t n_subjects;
	/** Number of raters */
	size_t n_raters;
	/** Method name (must be freed) */
	char *method;
	/** p-value of the F test of H0: ICC = 0 */
	double p_value;
	/** Unified test fields (#159); keep last */
	AnofoxTestSummaryFFI unified;
} AnofoxIccResult;

/**
 * Intraclass correlation coefficient (ICC)
 *
 * @param data Data matrix in row-major order (n_subjects x n_raters)
 * @param n_subjects Number of subjects
 * @param n_raters Number of raters/measurements per subject
 * @param icc_type ICC type (ICC1, ICC1k, ICC2, ICC3, ICC2k, ICC3k)
 * @param out_result Output: ICC result
 * @param out_error Output: error information
 * @return true on success, false on error
 */
bool anofox_icc(const double *data, size_t n_subjects, size_t n_raters, AnofoxIccType icc_type,
                AnofoxIccResult *out_result, AnofoxError *out_error);

/**
 * Free memory allocated by ICC result functions
 */
void anofox_free_icc_result(AnofoxIccResult *result);

/* --- Parametric Tests --- */

/**
 * Yuen's trimmed mean test
 */
bool anofox_yuen_test(AnofoxDataArray group1, AnofoxDataArray group2, double trim, AnofoxAlternative alternative,
                      double confidence_level, AnofoxTestResult *out_result, AnofoxError *out_error);

/**
 * Brown-Forsythe test for homogeneity of variances
 */
bool anofox_brown_forsythe(AnofoxDataArray values, AnofoxDataArray groups, AnofoxTestResult *out_result,
                           AnofoxError *out_error);

/* --- Forecast Tests --- */

/**
 * Loss function for forecast comparison
 */
typedef enum {
	ANOFOX_FORECAST_LOSS_SQUARED = 0,
	ANOFOX_FORECAST_LOSS_ABSOLUTE = 1,
} AnofoxForecastLoss;

/**
 * Variance estimator for forecast tests
 */
typedef enum {
	ANOFOX_FORECAST_VAR_ACF = 0,
	ANOFOX_FORECAST_VAR_BARTLETT = 1,
} AnofoxForecastVarEstimator;

/**
 * Diebold-Mariano test for equal predictive accuracy
 */
bool anofox_diebold_mariano(AnofoxDataArray actual, AnofoxDataArray forecast1, AnofoxDataArray forecast2,
                            AnofoxForecastLoss loss, AnofoxForecastVarEstimator var_estimator, size_t horizon,
                            AnofoxAlternative alternative, AnofoxTestResult *out_result, AnofoxError *out_error);

/**
 * Clark-West test for nested model comparison
 */
bool anofox_clark_west(AnofoxDataArray actual, AnofoxDataArray forecast_restricted,
                       AnofoxDataArray forecast_unrestricted, size_t horizon, AnofoxTestResult *out_result,
                       AnofoxError *out_error);

/* --- Resampling Tests --- */

/**
 * Permutation t-test (distribution-free alternative to t-test)
 */
bool anofox_permutation_t_test(AnofoxDataArray group1, AnofoxDataArray group2, AnofoxAlternative alternative,
                               size_t n_permutations, uint64_t seed, bool has_seed, AnofoxTestResult *out_result,
                               AnofoxError *out_error);

/* --- Free Functions --- */

/**
 * Free memory allocated by test functions
 */
void anofox_free_test_result(AnofoxTestResult *result);

/* ----------------------------------------------------------------------------
 * Pairwise post-hoc tests (#159)
 * ------------------------------------------------------------------------- */

/**
 * One comparison of a pairwise post-hoc test. Must match PairwiseRowFFI in
 * crates/anofox-stats-ffi/src/types.rs. group1 / group2 are the caller's group
 * codes; NaN marks a value that does not apply.
 */
typedef struct {
	double estimate;
	double statistic;
	double df;
	double p_value;
	double p_adj;
	double conf_low;
	double conf_high;
	uint64_t group1;
	uint64_t group2;
} AnofoxPairwiseRow;

/**
 * Result of a pairwise post-hoc test. Must match PairwiseResultFFI. `rows` is
 * freed by anofox_free_pairwise_result; conf_level is NaN and p_adjust_method
 * empty when they do not apply.
 */
typedef struct {
	AnofoxPairwiseRow *rows;
	size_t len;
	double conf_level;
	char method[ANOFOX_TEST_METHOD_LEN];
	char p_adjust_method[ANOFOX_TEST_NAME_LEN];
	char alternative[ANOFOX_TEST_NAME_LEN];
} AnofoxPairwiseResult;

/** Pairwise t-tests (R pairwise.t.test); p_adjust NULL means holm. groups has values.len codes. */
bool anofox_pairwise_t_test(AnofoxDataArray values, const uint64_t *groups, bool pooled_sd, const char *p_adjust,
                            AnofoxAlternative alternative, AnofoxPairwiseResult *out_result, AnofoxError *out_error);

/** Pairwise t-tests with the confidence level of the difference intervals (per comparison for p_adjust
 * none, Bonferroni otherwise); conf_level in (0, 1). */
bool anofox_pairwise_t_test_with_conf_level(AnofoxDataArray values, const uint64_t *groups, bool pooled_sd,
                                            const char *p_adjust, AnofoxAlternative alternative, double conf_level,
                                            AnofoxPairwiseResult *out_result, AnofoxError *out_error);

/** Tukey HSD (R TukeyHSD(aov(...))). */
bool anofox_tukey_hsd(AnofoxDataArray values, const uint64_t *groups, double conf_level,
                      AnofoxPairwiseResult *out_result, AnofoxError *out_error);

/** Dunn's test (FSA::dunnTest); p_adjust NULL means holm. */
bool anofox_dunn_test(AnofoxDataArray values, const uint64_t *groups, const char *p_adjust,
                      AnofoxPairwiseResult *out_result, AnofoxError *out_error);

/**
 * Adjust p-values for multiple testing (R p.adjust(p, method, n), #219). NULL/NaN
 * entries pass through as NaN and are not counted; n < 0 means the number of
 * non-NULL p-values, and a smaller n is an error. p_adjust NULL means holm.
 * out has p.len doubles; with p.len == 0 the call only validates p_adjust and n.
 */
bool anofox_p_adjust(AnofoxDataArray p, const char *p_adjust, int64_t n, double *out, AnofoxError *out_error);

/** Free the rows of a pairwise result. */
void anofox_free_pairwise_result(AnofoxPairwiseResult *result);

/**
 * Free memory allocated by ANOVA functions
 */
void anofox_free_anova_result(AnofoxAnovaResult *result);

/**
 * Free memory allocated by correlation functions
 */
void anofox_free_correlation_result(AnofoxCorrelationResult *result);

/**
 * Free memory allocated by chi-square functions
 */
void anofox_free_chisq_result(AnofoxChiSquareResult *result);

/**
 * Free memory allocated by TOST functions
 */
void anofox_free_tost_result(AnofoxTostResult *result);

/* ============================================================================
 * LmDynamic (Time-Varying Coefficient) Functions
 * ============================================================================ */

/**
 * Information criterion for model weighting
 */
typedef enum {
	ANOFOX_IC_AIC = 0,
	ANOFOX_IC_AICC = 1,
	ANOFOX_IC_BIC = 2,
} AnofoxInformationCriterion;

/**
 * LmDynamic options
 */
typedef struct {
	/** Whether to fit intercept */
	bool fit_intercept;
	/** Information criterion for model weighting */
	AnofoxInformationCriterion ic;
	/** Distribution family (uses AnofoxAlmDistribution enum) */
	AnofoxAlmDistribution distribution;
	/** LOWESS smoothing span (0.0 = no smoothing, >0 = span value) */
	double lowess_span;
	/** Maximum number of candidate models (0 = default) */
	uint32_t max_models;
	/** Confidence level for intervals */
	double confidence_level;
} AnofoxLmDynamicOptions;

/**
 * LmDynamic fit result
 */
typedef struct {
	/** Averaged coefficients (heap-allocated, caller must free) */
	double *coefficients;
	/** Number of coefficients */
	size_t coefficients_len;
	/** Intercept value (NaN if no intercept) */
	double intercept;
	/** R-squared */
	double r_squared;
	/** Adjusted R-squared */
	double adj_r_squared;
	/** RMSE */
	double rmse;
	/** Number of observations */
	size_t n_observations;
	/** Number of features */
	size_t n_features;
	/** Flattened dynamic coefficients (n_observations x n_coefs_per_obs, row-major) */
	double *dynamic_coefficients;
	/** Number of coefficient columns per observation */
	size_t n_coefs_per_obs;
} AnofoxLmDynamicFitResult;

/**
 * Fit a time-varying coefficient model using information-theoretic model averaging
 *
 * @param y Response values (y_len)
 * @param x_flat Features, column-major (n_features columns of y_len values)
 * @param options LmDynamic fitting options
 * @param out_result Output: fit result (required)
 * @param out_error Output: error information (required); code ANOFOX_ERROR_SUCCESS on success
 *
 * Legacy entry point; the extension uses anofox_lm_dynamic_fit (#63).
 */
void anofox_fit_lm_dynamic(const double *y, size_t y_len, const double *x_flat, size_t n_features,
                           const AnofoxLmDynamicOptions *options, AnofoxLmDynamicFitResult *out_result,
                           AnofoxError *out_error);

/**
 * Free memory allocated for LmDynamic fit result
 */
void anofox_free_lm_dynamic_result(AnofoxLmDynamicFitResult *result);

/* ============================================================================
 * AFT (accelerated failure time) survival regression -- issue #107
 * ============================================================================ */

/**
 * AFT error distribution code.
 */
typedef enum {
	ANOFOX_AFT_WEIBULL = 0,
	ANOFOX_AFT_LOGNORMAL = 1,
	ANOFOX_AFT_LOGLOGISTIC = 2,
	/** Weibull with the scale held at 1. */
	ANOFOX_AFT_EXPONENTIAL = 3,
} AnofoxAftDistribution;

/**
 * AFT fitting options.
 */
typedef struct {
	AnofoxAftDistribution dist;
	bool fit_intercept;
	uint32_t max_iterations;
	double tolerance;
	bool compute_inference;
	double confidence_level;
	/** Per-coefficient priors, or NULL for none. */
	const AnofoxPriorSpec *priors;
	size_t priors_len;
	AnofoxVcovType vcov;
} AnofoxAftOptions;

/**
 * Core results of an AFT fit. Coefficients are on the log-time scale.
 */
typedef struct {
	double *coefficients;
	size_t coefficients_len;
	/** NaN when no intercept was fitted. */
	double intercept;
	/** The scale sigma; exactly 1.0 for the exponential distribution. */
	double scale;
	double log_likelihood;
	double null_log_likelihood;
	double aic;
	double bic;
	size_t n_observations;
	size_t n_events;
	size_t n_censored;
	size_t n_features;
	uint32_t iterations;
	bool converged;
	/** Model description, fit statistics and intercept inference (#152) */
	AnofoxModelSummaryFFI summary;
} AnofoxAftFitResultCore;

/**
 * Curvature-based inference for an AFT fit.
 */
typedef struct {
	double *std_errors;
	double *z_values;
	double *p_values;
	double *ci_lower;
	double *ci_upper;
	size_t len;
	double confidence_level;
	/** NaN when no intercept was fitted. */
	double intercept_std_error;
	/** NaN when the scale is fixed. */
	double log_scale_std_error;
} AnofoxAftInference;

/**
 * Fit an AFT survival model with right censoring.
 *
 * @param time Event or censoring times (must be strictly positive)
 * @param x Feature arrays (column-major)
 * @param x_count Number of feature arrays
 * @param event 1.0 when the event was observed, 0.0 when right-censored
 * @param options Fitting options
 * @param out_core Output: fit result (required)
 * @param out_inference Output: inference, or NULL to skip
 * @param out_error Output: error information (required)
 * @return true on success, false on error
 */
bool anofox_aft_fit(AnofoxDataArray time, const AnofoxDataArray *x, size_t x_count, AnofoxDataArray event,
                    AnofoxAftOptions options, AnofoxAftFitResultCore *out_core, AnofoxAftInference *out_inference,
                    AnofoxError *out_error);

/** Free the arrays owned by an AFT fit result. */
void anofox_free_aft_result(AnofoxAftFitResultCore *result);

/** Free the arrays owned by AFT inference. */
void anofox_free_aft_inference(AnofoxAftInference *inf);

/** P(T <= t) for a fitted AFT model. */
double anofox_aft_cdf(double t, double eta, double scale, AnofoxAftDistribution dist);

/** The p-quantile of T for a fitted AFT model. */
double anofox_aft_quantile(double p, double eta, double scale, AnofoxAftDistribution dist);

/* ============================================================================
 * Empirical-Bayes shrinkage -- issue #107
 * ============================================================================ */

/** Between-group variance estimator. */
typedef enum {
	ANOFOX_TAU_DERSIMONIAN_LAIRD = 0,
	/** Complete pooling. */
	ANOFOX_TAU_NONE = 1,
} AnofoxTauMethod;

typedef struct {
	AnofoxTauMethod method;
	/** Fixed between-group variance; NaN means "estimate it". */
	double tau_squared;
} AnofoxEbShrinkOptions;

/**
 * Shrinkage result. The five per-group arrays share `len` and are in input order.
 */
typedef struct {
	double mu;
	double mu_se;
	double tau_squared;
	double i_squared;
	double q;
	size_t n_groups;
	double *estimate;
	double *se;
	double *shrunken;
	double *shrunken_se;
	double *weight;
	size_t len;
} AnofoxEbShrinkResult;

/**
 * Shrink per-group estimates toward their precision-weighted mean.
 */
bool anofox_eb_shrink(AnofoxDataArray estimate, AnofoxDataArray se, AnofoxEbShrinkOptions options,
                      AnofoxEbShrinkResult *out_result, AnofoxError *out_error);

/** Free the arrays owned by a shrinkage result. */
void anofox_free_eb_shrink_result(AnofoxEbShrinkResult *result);

/* ============================================================================
 * Mixed-effects GLMs -- issue #107
 * ============================================================================ */

typedef enum {
	ANOFOX_GLMM_GAUSSIAN = 0,
	ANOFOX_GLMM_POISSON = 1,
	ANOFOX_GLMM_BINOMIAL = 2,
	ANOFOX_GLMM_NEGBINOMIAL = 3,
	ANOFOX_GLMM_GAMMA = 4,
	ANOFOX_GLMM_TWEEDIE = 5,
} AnofoxGlmmFamily;

/** Degrees-of-freedom method of the fixed-effect tests. */
typedef enum {
	/** The family default: Satterthwaite for gaussian, Wald otherwise. */
	ANOFOX_GLMM_DDF_DEFAULT = 0,
	ANOFOX_GLMM_DDF_WALD = 1,
	ANOFOX_GLMM_DDF_SATTERTHWAITE = 2,
} AnofoxGlmmDdf;

typedef struct {
	AnofoxGlmmFamily family;
	bool fit_intercept;
	uint32_t max_iterations;
	double tolerance;
	bool compute_inference;
	double confidence_level;
	bool reml;
	/** Negative Binomial size theta; NaN = estimate it (lme4::glmer.nb). Ignored by
	    other families. */
	double theta;
	/** Tweedie variance power; ignored by other families. */
	double power;
	/** 1-based index into x of an offset column; 0 means none. */
	size_t offset_column;
	/** Pointer to random_slopes_len 0-based indices into x of columns that also
	    carry a random slope. Null/zero-length = random intercept only. */
	const size_t *random_slopes;
	size_t random_slopes_len;
	AnofoxGlmmDdf ddf;
	/** lme4 nAGQ (#305): 0 = Laplace at the PIRLS fixed effects (default), 1 = joint Laplace,
	    > 1 = adaptive Gauss-Hermite quadrature (scalar random intercept only) */
	uint32_t n_agq;
} AnofoxGlmmOptions;

typedef struct {
	double *coefficients;
	size_t coefficients_len;
	double intercept;
	double *std_errors;
	double *z_values;
	double *p_values;
	double *ci_lower;
	double *ci_upper;
	size_t inference_len;
	double intercept_std_error;
	double confidence_level;
	double var_group;
	double var_residual;
	double icc;
	double log_likelihood;
	double aic;
	double bic;
	double deviance;
	size_t n_observations;
	size_t n_groups;
	size_t n_features;
	uint32_t iterations;
	bool converged;
	/** Per-group random effects, all of length ranef_len. */
	int32_t *ranef_group;
	double *ranef_value;
	double *ranef_se;
	int64_t *ranef_n;
	size_t ranef_len;
	/** Random-effects covariance Sigma, flattened row-major random_dim x random_dim. */
	double *random_cov;
	/** q = 1 + number of random slopes. */
	size_t random_dim;
	/** Per-group random-effect vectors [intercept, slope_1, ...], flattened
	    row-major ranef_len x random_dim. */
	double *ranef_effects;
	/** Per-factor variance components for crossed/nested fits (length factor_len);
	    empty for the single-factor path. */
	double *factor_var;
	int64_t *factor_n_levels;
	size_t factor_len;
	/** Negative-binomial size theta (estimated or fixed); NaN for other families. */
	double nb_theta;
	/** Satterthwaite degrees of freedom of the coefficients (inference_len); NULL for
	    Wald inference or without inference. */
	double *df;
	/** Satterthwaite degrees of freedom of the intercept; NaN otherwise. */
	double intercept_df;
	/** Conditional standard errors of ranef_effects, same layout. */
	double *ranef_effects_se;
	/** Length of each group's vector in ranef_effects / ranef_effects_se (differs from
	    random_dim for crossed fits, whose random_cov is empty without random slopes). */
	size_t ranef_dim;
	/** Normal intervals of ranef_effects at confidence_level, same layout. */
	double *ranef_effects_conf_low;
	double *ranef_effects_conf_high;
	/** Residual degrees of freedom n - k (lme4 df.residual()). */
	double df_residual;
	/** Model description, fit statistics and intercept inference (#152) */
	AnofoxModelSummaryFFI summary;
} AnofoxGlmmResult;

/** Per-row fitted values and residuals of a GLMM, one per input row (input order). */
typedef struct {
	double *fitted;
	double *fitted_population;
	double *residual;
	double *pearson_residual;
	bool *is_training;
	size_t len;
} AnofoxGlmmAugmentResult;

/**
 * Fit a mixed-effects GLM with a random intercept over one grouping factor.
 *
 * @param group_ids Dense group indices in 0..n_groups, one per observation
 */
bool anofox_glmm_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, const int32_t *group_ids,
                     size_t n_obs, const int32_t *const *extra_group_ids, size_t n_extra_factors,
                     AnofoxGlmmOptions options, AnofoxGlmmResult *out_result, AnofoxError *out_error);

/** Free the arrays owned by a GLMM result. */
void anofox_free_glmm_result(AnofoxGlmmResult *result);

/**
 * Per-row fitted values (conditional and population-level, response scale) and
 * response / Pearson residuals of the same fit as anofox_glmm_fit. Rows the fit
 * dropped are NaN with is_training = false.
 */
bool anofox_glmm_augment(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, const int32_t *group_ids,
                         size_t n_obs, const int32_t *const *extra_group_ids, size_t n_extra_factors,
                         AnofoxGlmmOptions options, AnofoxGlmmAugmentResult *out_result, AnofoxError *out_error);

/** Free the arrays owned by a GLMM augment result. */
void anofox_free_glmm_augment_result(AnofoxGlmmAugmentResult *result);

/* ============================================================================
 * Per-observation diagnostics (broom augment) -- issue #151
 * ============================================================================ */

/**
 * Per-row diagnostics of an OLS/WLS/Ridge/GLM fit (layout of AugmentResultFFI in
 * crates/anofox-stats-ffi/src/types.rs). Every array has len entries, one per
 * input row in input order; NaN marks a value that does not apply (rows not
 * used in the fit only have fitted). Released by anofox_free_augment_result.
 */
typedef struct {
	double *fitted;
	double *residual;
	double *std_residual;
	double *stud_residual;
	double *leverage;
	double *cooks_d;
	double *dffits;
	double *loo_residual; /* leave-one-out (PRESS) residual, #220 */
	double *loo_fitted;
	double *approx_loo_fitted; /* one-step approximate leave-one-out (GLMs), #235 */
	double *approx_loo_residual;
	bool *used;
	size_t len;
	size_t n_params;
	char resid_type[ANOFOX_MODEL_NAME_LEN];
} AnofoxAugmentResult;

/**
 * Fit OLS on the usable rows (finite y and features; NaN y marks a row that is
 * only predicted) and return per-row diagnostics for every input row.
 */
bool anofox_ols_augment(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxOlsOptions options,
                        AnofoxAugmentResult *out_result, AnofoxError *out_error);

/**
 * Fit Ridge on the usable rows (finite y and features) and return fitted,
 * residual, the ridge leverage and the exact leave-one-out residual / fitted
 * value for every input row (#235); the other diagnostics are NaN.
 */
bool anofox_ridge_augment(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxRidgeOptions options,
                          AnofoxAugmentResult *out_result, AnofoxError *out_error);

/**
 * Fit WLS on the usable rows (finite y and features, positive weight) and
 * return per-row diagnostics for every input row.
 */
bool anofox_wls_augment(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxDataArray weights,
                        AnofoxWlsOptions options, AnofoxAugmentResult *out_result, AnofoxError *out_error);

/**
 * GLM augment: fit the family on the usable rows (finite y, features and
 * offset; NaN y marks a row that is only predicted) with the options of the
 * matching *_fit call and return per-row diagnostics for every input row, with
 * deviance residuals (resid_type "deviance") and fitted on the response scale.
 * Penalized fits (lambda > 0 or a non-flat prior) fail with an error.
 */
bool anofox_poisson_augment(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxPoissonOptions options,
                            AnofoxAugmentResult *out_result, AnofoxError *out_error);
bool anofox_binomial_augment(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count,
                             AnofoxBinomialOptions options, AnofoxAugmentResult *out_result, AnofoxError *out_error);
bool anofox_logistic_augment(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count,
                             AnofoxLogisticOptions options, AnofoxAugmentResult *out_result, AnofoxError *out_error);
bool anofox_negbinomial_augment(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count,
                                AnofoxNegBinomialOptions options, AnofoxAugmentResult *out_result,
                                AnofoxError *out_error);
bool anofox_gamma_augment(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxGammaOptions options,
                          AnofoxAugmentResult *out_result, AnofoxError *out_error);
bool anofox_tweedie_augment(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxTweedieOptions options,
                            AnofoxAugmentResult *out_result, AnofoxError *out_error);

/** Free the arrays owned by an augment result. */
void anofox_free_augment_result(AnofoxAugmentResult *result);

/*
 * ============================================================================
 * Regularisation paths and CV curves (#154)
 * ============================================================================
 */

/** Options of anofox_reg_path / anofox_cv_reg_path (glmnet conventions). */
typedef struct {
	/** Elastic-net mixing in [0, 1]: 0 = ridge, 1 = lasso (glmnet's alpha). */
	double l1_ratio;
	bool fit_intercept;
	/** glmnet's standardize. */
	bool standardize;
	/** Penalties on glmnet's scale; NULL / 0 for glmnet's default grid. */
	const double *lambdas;
	size_t lambdas_len;
	/** Size of the default grid. */
	size_t n_lambda;
	/** Smallest / largest lambda of the default grid; NaN for glmnet's default. */
	double lambda_min_ratio;
} AnofoxRegPathOptions;

/** A coefficient path: n_lambda entries, coefficients row-major (n_lambda x n_features). */
typedef struct {
	double *lambdas;
	/** NaN without an intercept. */
	double *intercepts;
	/** NaN for columns the fit dropped. */
	double *coefficients;
	/** Number of non-zero coefficients per lambda. */
	double *df;
	size_t n_lambda;
	size_t n_features;
	size_t n_observations;
} AnofoxRegPathResult;

/** A k-fold CV curve (cv.glmnet, type.measure = "mse"). */
typedef struct {
	double *lambdas;
	double *cv_mean;
	double *cv_se;
	double *df;
	size_t n_lambda;
	size_t index_min;
	size_t index_1se;
	size_t n_folds;
	size_t n_observations;
} AnofoxCvPathResult;

/** Ridge / elastic-net path (solvers::path::fit_path); rows with a non-finite value are dropped. */
bool anofox_reg_path(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxRegPathOptions options,
                     AnofoxRegPathResult *out, AnofoxError *out_error);
/** LARS / Lasso-LARS path (FittedLars::path); lambdas are the knots' alphas. */
bool anofox_lars_path(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxLarsOptions options,
                      AnofoxRegPathResult *out, AnofoxError *out_error);
void anofox_free_reg_path_result(AnofoxRegPathResult *result);
/** CV strategy (#204): how the splits of anofox_cv_reg_path / anofox_nested_cv_reg_path are built. */
typedef enum {
	ANOFOX_CV_KFOLD = 0,
	ANOFOX_CV_FOLD_IDS = 1,
	ANOFOX_CV_GROUP = 2,
	ANOFOX_CV_LOGO = 3,
	ANOFOX_CV_STRATIFIED = 4,
	ANOFOX_CV_TIME_SERIES = 5,
} AnofoxCvStrategy;

typedef struct {
	int32_t strategy;
	/** k of the k-fold strategies, n_splits of time_series (outer folds of nested CV). */
	size_t n_folds;
	uint64_t seed;
	/** kfold / stratified: repetitions with seeds seed, seed + 1, ... */
	size_t repeats;
	/** time_series: time points per test block (0: n / (n_splits + 1)), gap, sliding window (0: expanding). */
	size_t test_size;
	size_t gap;
	size_t max_train_size;
	/** stratified without a key: quantile bins of y. */
	size_t n_bins;
	/** Mean absolute error instead of the mean squared error. */
	bool mae;
	/** Nested CV: inner folds and the lambda rule (lambda_1se instead of lambda_min). */
	size_t inner_folds;
	bool one_se;
} AnofoxCvSpec;

/**
 * CV of the ridge / elastic-net path on the splits of a CV strategy
 * (solvers::cv::cv_path_splits). keys: one value per row (fold label, group,
 * stratum or time, by strategy) or NULL.
 */
bool anofox_cv_reg_path(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, const double *keys,
                        AnofoxCvSpec spec, AnofoxRegPathOptions options, AnofoxCvPathResult *out,
                        AnofoxError *out_error);
void anofox_free_cv_path_result(AnofoxCvPathResult *result);

/** Nested CV (#204): one entry per outer split (arrays of n_folds), plus the weighted mean and its se. */
typedef struct {
	double *lambda;
	double *test_error;
	double *inner_cv_mean;
	double *inner_cv_se;
	double *n_train;
	double *n_test;
	double *n_inner_folds;
	size_t n_folds;
	double mean;
	double se;
	size_t n_observations;
} AnofoxNestedCvResult;

/** Nested CV of the ridge / elastic-net path (solvers::cv::nested_cv_path). */
bool anofox_nested_cv_reg_path(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, const double *keys,
                               AnofoxCvSpec spec, AnofoxRegPathOptions options, AnofoxNestedCvResult *out,
                               AnofoxError *out_error);
void anofox_free_nested_cv_result(AnofoxNestedCvResult *result);

/*
 * ============================================================================
 * Quantile-regression inference (#157)
 * ============================================================================
 */

/** Standard-error type (quantreg::summary.rq se). */
typedef enum {
	ANOFOX_QUANTILE_SE_NONE = 0,
	ANOFOX_QUANTILE_SE_IID = 1,
	ANOFOX_QUANTILE_SE_NID = 2,
	ANOFOX_QUANTILE_SE_BOOT = 3,
} AnofoxQuantileSe;

/** Coefficient tables of n_tau quantiles, k entries each (intercept first), row-major per tau. */
typedef struct {
	double *taus;
	double *estimate;
	double *std_error;
	double *statistic;
	double *p_value;
	double *conf_low;
	double *conf_high;
	size_t n_tau;
	size_t k;
	size_t n_observations;
	double conf_level;
	double df_residual;
	bool has_intercept;
} AnofoxQuantileProcessResult;

/** Fit at each tau with inference (solvers::quantile_inference::quantile_process). */
bool anofox_quantile_process(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, const double *taus,
                             size_t n_taus, bool fit_intercept, int32_t se_type, uint64_t seed, size_t n_boot,
                             double confidence_level, AnofoxQuantileProcessResult *out, AnofoxError *out_error);
void anofox_free_quantile_process_result(AnofoxQuantileProcessResult *result);

/*
 * ============================================================================
 * Coefficient covariance, predictions with SE, partial residuals (#158)
 * ============================================================================
 */

/**
 * Coefficient covariance sigma^2 M of an OLS / WLS / Ridge fit from its training
 * design (M as anofox_interval_matrix), row-major dim x dim, intercept first.
 * Free with anofox_free_interval_matrix.
 */
bool anofox_coefficient_vcov(const AnofoxDataArray *x, size_t x_count, const double *coefficients,
                             size_t coefficients_len, bool fit_intercept, const AnofoxDataArray *weights,
                             double ridge_lambda, double residual_std_error, double **out_matrix, size_t *out_dim,
                             AnofoxError *out_error);

/** Interval of anofox_predict_se. */
typedef enum {
	ANOFOX_PREDICT_INTERVAL_NONE = 0,
	ANOFOX_PREDICT_INTERVAL_CONFIDENCE = 1,
	ANOFOX_PREDICT_INTERVAL_PREDICTION = 2,
} AnofoxPredictInterval;

typedef struct {
	double yhat;
	double se_fit;
	double yhat_lower;
	double yhat_upper;
} AnofoxPredictSeResult;

/**
 * One prediction with standard error and interval from coefficients and their
 * covariance (vcov: dim x dim, intercept first; intercept NaN = none). link NULL:
 * linear model (t intervals with df_residual; prediction intervals add
 * residual_std_error^2). Otherwise the GLM link name: intervals on the link
 * scale, mapped through the inverse link when response_scale.
 */
bool anofox_predict_se(const double *coefficients, size_t coefficients_len, double intercept, const double *x,
                       const double *vcov, size_t dim, const char *link, bool response_scale,
                       double residual_std_error, double df_residual, int32_t interval, double level,
                       AnofoxPredictSeResult *out, AnofoxError *out_error);

/**
 * Partial-residual (CCPR) and added-variable data, term-major (n_terms x len):
 * entry j * len + i is x column j, input row i. Rows not used for fitting,
 * dropped columns and a GLM offset column are NaN; x_residual / y_residual are
 * NaN for the GLMs (has_added_variable false).
 */
typedef struct {
	double *x;
	double *component;
	double *partial_residual;
	double *x_residual;
	double *y_residual;
	size_t n_terms;
	size_t len;
	bool has_added_variable;
} AnofoxPartialResidualsResult;

bool anofox_ols_partial_residuals(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count,
                                  AnofoxOlsOptions options, AnofoxPartialResidualsResult *out,
                                  AnofoxError *out_error);
bool anofox_wls_partial_residuals(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count,
                                  AnofoxDataArray weights, AnofoxWlsOptions options,
                                  AnofoxPartialResidualsResult *out, AnofoxError *out_error);
bool anofox_poisson_partial_residuals(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count,
                                      AnofoxPoissonOptions options, AnofoxPartialResidualsResult *out,
                                      AnofoxError *out_error);
bool anofox_binomial_partial_residuals(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count,
                                       AnofoxBinomialOptions options, AnofoxPartialResidualsResult *out,
                                       AnofoxError *out_error);
bool anofox_logistic_partial_residuals(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count,
                                       AnofoxLogisticOptions options, AnofoxPartialResidualsResult *out,
                                       AnofoxError *out_error);
bool anofox_negbinomial_partial_residuals(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count,
                                          AnofoxNegBinomialOptions options, AnofoxPartialResidualsResult *out,
                                          AnofoxError *out_error);
bool anofox_gamma_partial_residuals(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count,
                                    AnofoxGammaOptions options, AnofoxPartialResidualsResult *out,
                                    AnofoxError *out_error);
bool anofox_tweedie_partial_residuals(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count,
                                      AnofoxTweedieOptions options, AnofoxPartialResidualsResult *out,
                                      AnofoxError *out_error);
void anofox_free_partial_residuals_result(AnofoxPartialResidualsResult *result);

/* ============================================================================
 * Time-series residual diagnostics -- issue #155
 * ============================================================================ */

/**
 * Sample (partial) autocorrelations. Must match AcfResultFFI. values[i] is the
 * value at lag first_lag + i (0 for the ACF, 1 for the PACF); free with
 * anofox_free_acf_result.
 */
typedef struct {
	double conf_bound;
	double *values;
	size_t len;
	size_t first_lag;
	size_t n;
} AnofoxAcfResult;

/** ACF (R acf) or, with partial, PACF (R pacf) of x in time order; max_lag < 0 uses R's default. */
bool anofox_autocorrelation(AnofoxDataArray x, int64_t max_lag, bool partial, AnofoxAcfResult *out_result,
                            AnofoxError *out_error);
void anofox_free_acf_result(AnofoxAcfResult *result);

/** Ljung-Box or, with box_pierce, Box-Pierce test (R Box.test). */
bool anofox_portmanteau(AnofoxDataArray x, size_t lag, size_t fitdf, bool box_pierce,
                        AnofoxTestSummaryFFI *out_result, AnofoxError *out_error);

/** Durbin-Watson statistic (p_value is NaN). */
bool anofox_durbin_watson(AnofoxDataArray residuals, AnofoxTestSummaryFFI *out_result, AnofoxError *out_error);

/** Breusch-Pagan test (R lmtest::bptest; studentize is its default). */
bool anofox_breusch_pagan(AnofoxDataArray residuals, const AnofoxDataArray *x, size_t x_count, bool studentize,
                          AnofoxTestSummaryFFI *out_result, AnofoxError *out_error);

/** White's test: Breusch-Pagan on x, its squares and cross-products. */
bool anofox_white_test(AnofoxDataArray residuals, const AnofoxDataArray *x, size_t x_count,
                       AnofoxTestSummaryFFI *out_result, AnofoxError *out_error);

/** Durbin-Watson test with p-value (R lmtest::dwtest); intercept prepended when fit_intercept; exact -1 = default (n < 100), 0 = approximation, 1 = exact. */
bool anofox_durbin_watson_test(AnofoxDataArray residuals, const AnofoxDataArray *x, size_t x_count, bool fit_intercept,
                               AnofoxAlternative alternative, int32_t exact, AnofoxTestSummaryFFI *out_result,
                               AnofoxError *out_error);

/* ============================================================================
 * Survival curves -- issue #156
 * ============================================================================ */

/**
 * Kaplan-Meier curve, one entry per distinct observed time. Must match
 * KaplanMeierResultFFI. Every array has len elements; std_err, conf_low and
 * conf_high are NaN where R gives NA. Free with anofox_free_kaplan_meier_result.
 */
typedef struct {
	double conf_level;
	double *time;
	double *survival;
	double *std_err;
	double *conf_low;
	double *conf_high;
	uint64_t *n_risk;
	uint64_t *n_event;
	uint64_t *n_censor;
	size_t len;
	size_t n;
} AnofoxKaplanMeierResult;

/** Kaplan-Meier (R survfit); event non-zero = observed; conf_type 0 log, 1 log-log, 2 plain. */
bool anofox_kaplan_meier(AnofoxDataArray time, AnofoxDataArray event, int32_t conf_type, double conf_level,
                         AnofoxKaplanMeierResult *out_result, AnofoxError *out_error);
void anofox_free_kaplan_meier_result(AnofoxKaplanMeierResult *result);

/** Log-rank test (R survdiff); groups holds time.len group codes. */
bool anofox_log_rank(AnofoxDataArray time, AnofoxDataArray event, const uint64_t *groups,
                     AnofoxTestSummaryFFI *out_result, AnofoxError *out_error);

/**
 * S(t | x) of a fitted AFT model at n_times times into out_survival (R psurvreg,
 * lower.tail = FALSE). intercept NaN = no intercept; x has n_coefficients values.
 */
bool anofox_aft_survival(const double *coefficients, size_t n_coefficients, double intercept, double scale,
                         AnofoxAftDistribution dist, const double *x, const double *times, size_t n_times,
                         double *out_survival);

/* ============================================================================
 * Classifier evaluation curves -- issue #153
 * ============================================================================ */

/**
 * One curve of a binary classifier. Must match ClassifierCurveFFI. Non-null
 * arrays have `len` entries in curve order; free with
 * anofox_free_classifier_curve.
 *   ROC: x = false positive rate, y = true positive rate.
 *   PR: x = recall, y = precision.
 *   Calibration: x = mean predicted probability, y = fraction of positives,
 *   y_low/y_high = Wilson bounds, bin = zero-based bin, n = bin count.
 * y_low, y_high, bin and n are NULL for ROC and PR. threshold is NaN where a
 * point has none (final PR point, calibration bins) and +inf at the first ROC
 * point.
 */
typedef struct {
	double *x;
	double *y;
	double *threshold;
	double *y_low;
	double *y_high;
	int64_t *bin;
	int64_t *n;
	size_t len;
} AnofoxClassifierCurve;

/** Scalar metric computed by anofox_classifier_metric. */
typedef enum {
	ANOFOX_CLASSIFIER_METRIC_AUC = 0,
	ANOFOX_CLASSIFIER_METRIC_BRIER = 1,
	ANOFOX_CLASSIFIER_METRIC_LOG_LOSS = 2,
} AnofoxClassifierMetric;

/**
 * Labels are 0/1 (nonzero = positive). Scores may be +-inf but not NaN;
 * probabilities must be in [0, 1].
 */
/** ROC curve (sklearn roc_curve). drop_intermediate as sklearn's default (true). */
bool anofox_roc_curve(AnofoxDataArray scores, AnofoxDataArray labels, bool drop_intermediate,
                      AnofoxClassifierCurve *out_result, AnofoxError *out_error);
/** Precision-recall curve (sklearn precision_recall_curve). */
bool anofox_pr_curve(AnofoxDataArray scores, AnofoxDataArray labels, AnofoxClassifierCurve *out_result,
                     AnofoxError *out_error);
/** Calibration curve (sklearn calibration_curve, uniform bins) with Wilson intervals. */
bool anofox_calibration_curve(AnofoxDataArray probs, AnofoxDataArray labels, size_t n_bins, double conf_level,
                              AnofoxClassifierCurve *out_result, AnofoxError *out_error);
/** Free the arrays owned by a classifier curve. */
void anofox_free_classifier_curve(AnofoxClassifierCurve *result);
/** ROC AUC, Brier score or log loss. */
bool anofox_classifier_metric(AnofoxClassifierMetric metric, AnofoxDataArray scores, AnofoxDataArray labels,
                              double *out_value, AnofoxError *out_error);

/* ============================================================================
 * Bayesian Ridge and ARD regression -- issue #61
 * (crates/anofox-stats-ffi/src/models_ffi.rs)
 * ============================================================================ */

/** Options of scikit-learn's BayesianRidge / ARDRegression. */
typedef struct {
	bool fit_intercept;
	uint32_t max_iterations;
	double tolerance;
	/** Gamma-prior shape / rate of the noise precision. */
	double alpha_1;
	double alpha_2;
	/** Gamma-prior shape / rate of the weight precision(s). */
	double lambda_1;
	double lambda_2;
	/** ARD only: features whose precision reaches this value are pruned. */
	double threshold_lambda;
	/** Level of the coefficient credible intervals. */
	double confidence_level;
} AnofoxBayesianOptions;

/**
 * A fitted Bayesian Ridge / ARD model. coefficients, posterior_sd and the
 * credible bounds credible_lower / credible_upper (at confidence_level) have
 * coefficients_len entries; weight_precisions (ARD) has weight_precisions_len
 * (0 and NULL for Bayesian Ridge). Must match BayesianFitResultFFI.
 */
typedef struct {
	double intercept;
	double r_squared;
	/** Noise precision (sklearn alpha_). */
	double noise_precision;
	/** Weight precision (sklearn lambda_); NaN for ARD. */
	double weight_precision;
	/** Log marginal likelihood (evidence); NaN for ARD. */
	double log_marginal_likelihood;
	double confidence_level;
	double *coefficients;
	double *posterior_sd;
	double *credible_lower;
	double *credible_upper;
	double *weight_precisions;
	size_t coefficients_len;
	size_t weight_precisions_len;
	size_t n_observations;
	size_t n_features;
	size_t iterations;
	AnofoxModelSummaryFFI summary;
} AnofoxBayesianFitResult;

/** Fit Bayesian Ridge (ard = false) or ARD (ard = true) on the complete rows of (y, x). */
bool anofox_bayesian_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxBayesianOptions options,
                         bool ard, AnofoxBayesianFitResult *out, AnofoxError *out_error);

/**
 * Fit on the complete rows of (y, x) and predict the n_new rows of x_new
 * (row-major, x_count columns) with posterior predictive intervals at level;
 * rows with a non-finite value predict NaN. The outputs hold n_new doubles each.
 */
bool anofox_bayesian_fit_predict(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, const double *x_new,
                                 size_t n_new, AnofoxBayesianOptions options, bool ard, double level,
                                 double *out_yhat, double *out_lower, double *out_upper, AnofoxError *out_error);

/** Free the arrays of a Bayesian fit result. */
void anofox_free_bayesian_result(AnofoxBayesianFitResult *result);

/* ============================================================================
 * Passive-Aggressive regression, lmDynamic and LOWESS -- issue #63
 * (crates/anofox-stats-ffi/src/models_ffi.rs)
 * ============================================================================ */

/** Options of anofox_passive_aggressive_fit. */
typedef struct {
	bool fit_intercept;
	/** PA-II (squared epsilon-insensitive loss) instead of PA-I. */
	bool squared_loss;
	/** Passes over the rows; 1 = a single online pass. */
	uint32_t max_epochs;
	double c;
	double epsilon;
	double tolerance;
} AnofoxPassiveAggressiveOptions;

/** A fitted Passive-Aggressive model; coefficients has n_features entries. Must match PassiveAggressiveResultFFI. */
typedef struct {
	double intercept;
	double r_squared;
	double *coefficients;
	size_t n_observations;
	size_t n_features;
	size_t epochs;
	AnofoxModelSummaryFFI summary;
} AnofoxPassiveAggressiveResult;

/** Fit a Passive-Aggressive regressor on the complete rows of (y, x), in input order (no shuffling). */
bool anofox_passive_aggressive_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count,
                                   AnofoxPassiveAggressiveOptions options, AnofoxPassiveAggressiveResult *out,
                                   AnofoxError *out_error);

/** Free the coefficients of a Passive-Aggressive result. */
void anofox_free_passive_aggressive_result(AnofoxPassiveAggressiveResult *result);

/**
 * A fitted lmDynamic model. coefficients (time-averaged) has n_features entries;
 * dynamic_coefficients is n_rows x n_coefs_per_row, row-major ([intercept, coef_1..coef_p]
 * per input row), fitted_values has n_rows; rows dropped for non-finite values are NaN.
 * Must match LmDynamicFitFFI.
 */
typedef struct {
	double intercept;
	double r_squared;
	double adj_r_squared;
	double residual_std_error;
	double *coefficients;
	double *dynamic_coefficients;
	double *fitted_values;
	size_t n_rows;
	size_t n_coefs_per_row;
	size_t n_observations;
	size_t n_features;
	size_t n_models;
	AnofoxModelSummaryFFI summary;
} AnofoxLmDynamicFit;

/**
 * Fit lmDynamic (greybox lmDynamic(bruteforce = TRUE)) on (y, x) in input order.
 * options.lowess_span 0 disables the weight smoothing; options.max_models 0 averages
 * every regressor subset.
 */
bool anofox_lm_dynamic_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxLmDynamicOptions options,
                           AnofoxLmDynamicFit *out, AnofoxError *out_error);

/** Free the arrays of an lmDynamic result. */
void anofox_free_lm_dynamic_fit_result(AnofoxLmDynamicFit *result);

/** A LOWESS curve: len points, x increasing. Must match LowessResultFFI. */
typedef struct {
	double *x;
	double *fitted;
	size_t len;
} AnofoxLowessResult;

/**
 * R's stats::lowess(x, y, f = frac, iter = iterations, delta) on the complete rows of
 * (x, y); a NaN delta uses R's default, 1% of the range of x.
 */
bool anofox_lowess(AnofoxDataArray x, AnofoxDataArray y, double frac, uint32_t iterations, double delta,
                   AnofoxLowessResult *out, AnofoxError *out_error);

/** Free the arrays of a LOWESS result. */
void anofox_free_lowess_result(AnofoxLowessResult *result);

/* ============================================================================
 * Causal mediation analysis -- issue #223
 * ============================================================================ */

typedef enum {
	ANOFOX_MEDIATION_BOOTSTRAP = 0,
	ANOFOX_MEDIATION_SOBEL = 1,
	ANOFOX_MEDIATION_QUASI_BAYESIAN = 2,
} AnofoxMediationMethod;

typedef enum {
	ANOFOX_MEDIATION_CI_PERCENTILE = 0,
	ANOFOX_MEDIATION_CI_BCA = 1,
} AnofoxMediationCi;

typedef struct {
	AnofoxMediationMethod method;
	AnofoxMediationCi ci_type;
	uint32_t n_sims;
	uint64_t seed;
	double conf_level;
	double treat_value;
	double control_value;
} AnofoxMediationOptions;

/** One effect; non-finite values mean "not available". */
typedef struct {
	double estimate;
	double std_error;
	double statistic;
	double p_value;
	double conf_low;
	double conf_high;
} AnofoxMediationEffect;

/** Mediation result (owns no memory). */
typedef struct {
	AnofoxMediationEffect acme;
	AnofoxMediationEffect ade;
	AnofoxMediationEffect total;
	AnofoxMediationEffect prop_mediated;
	double a;
	double b;
	double c_prime;
	double r_squared_mediator;
	double r_squared_outcome;
	size_t n_obs;
	size_t n_sims;
} AnofoxMediationResult;

/**
 * Causal mediation analysis of x -> m -> y with covariate_count covariates
 * (mediator model m ~ x + C, outcome model y ~ x + m + C, both OLS).
 */
bool anofox_mediation(AnofoxDataArray y, AnofoxDataArray m, AnofoxDataArray x, const AnofoxDataArray *covariates,
                      size_t covariate_count, AnofoxMediationOptions options, AnofoxMediationResult *out_result,
                      AnofoxError *out_error);

/* ============================================================================
 * Nested-model F test of two OLS fits -- issue #221
 * ============================================================================ */

/**
 * Result of anofox_ols_anova. NaN marks a value that does not apply (the
 * statistic and p-value when the full model adds no rank or leaves no residual
 * degrees of freedom). Must match OlsAnovaResultFFI in the Rust FFI crate.
 */
typedef struct {
	double statistic;
	double df1;
	double df2;
	double p_value;
	double rss_reduced;
	double rss_full;
	double adj_r_squared_reduced;
	double adj_r_squared_full;
	double delta_adj_r_squared;
	size_t n_observations;
	/** Unified test fields (#159); keep last */
	AnofoxTestSummaryFFI unified;
} AnofoxOlsAnovaResult;

/**
 * Nested-model F test (R's anova(lm(reduced), lm(full))) of the OLS fit of y on
 * the x_reduced columns against the fit on the x_full columns, on the rows where
 * every value is finite. x_reduced may be empty (NULL, 0): the intercept-only
 * model. Its columns must lie in the span of x_full (INVALID_INPUT otherwise).
 */
bool anofox_ols_anova(AnofoxDataArray y, const AnofoxDataArray *x_reduced, size_t x_reduced_count,
                      const AnofoxDataArray *x_full, size_t x_full_count, bool fit_intercept,
                      AnofoxOlsAnovaResult *out_result, AnofoxError *out_error);

/* ============================================================================
 * Data-drift tests (#245-#255)
 * ============================================================================ */

typedef enum {
	ANOFOX_DRIFT_KS = 0,
	ANOFOX_DRIFT_CVM = 1,
	ANOFOX_DRIFT_EPPS_SINGLETON = 2,
	ANOFOX_DRIFT_WASSERSTEIN = 3,
	ANOFOX_DRIFT_PSI = 4,
	ANOFOX_DRIFT_KL = 5,
	ANOFOX_DRIFT_JENSEN_SHANNON = 6,
	ANOFOX_DRIFT_HELLINGER = 7,
	ANOFOX_DRIFT_TVD = 8,
	ANOFOX_DRIFT_Z = 9,
	ANOFOX_DRIFT_CHISQ = 10,
	ANOFOX_DRIFT_G = 11,
} AnofoxDriftKind;

/**
 * Options of anofox_drift_test; each kind reads only its own. Must match
 * DriftOptionsFFI in crates/anofox-stats-ffi/src/drift_ffi.rs.
 */
typedef struct {
	/** Epps-Singleton evaluation points; NULL / 0 for the default [0.4, 0.8] */
	const double *t;
	size_t t_len;
	/** Wasserstein order */
	double p;
	/** Logarithm base of KL / Jensen-Shannon; NaN for e */
	double base;
	/** Zero fill of PSI / KL; NaN for evidently's rule */
	double epsilon;
	/** Permutations of Wasserstein / TVD (0: none) */
	size_t n_permutations;
	uint64_t seed;
	/** Bins of the quantile binning */
	size_t n_bins;
	/** AnofoxAlternative */
	int32_t alternative;
	/** KS exact p-value: -1 auto, 0 no, 1 yes */
	int32_t exact;
	/** Cramer-von Mises: 0 auto, 1 exact, 2 asymptotic */
	int32_t cvm_method;
	/** 0 auto (evidently), 1 sturges, 2 categorical, 3 quantile */
	int32_t binning;
	bool has_seed;
} AnofoxDriftOptions;

/**
 * Result of a drift test. A field that does not apply is NaN (numbers) or 0
 * (n_bins). Must match DriftResultFFI in crates/anofox-stats-ffi/src/drift_ffi.rs.
 */
typedef struct {
	double statistic;
	/** Normed Wasserstein distance, Jensen-Shannon divergence, or raw AD criterion */
	double statistic2;
	double p_value;
	double df;
	/** Reference size (Anderson-Darling: total size) */
	size_t n1;
	/** Current size (Anderson-Darling: number of groups) */
	size_t n2;
	size_t n_bins;
	/** z test: shares of the non-smallest category (reference, current) */
	double estimate_reference;
	double estimate_current;
	bool exact;
	/** Unified test fields (#159); keep last */
	AnofoxTestSummaryFFI unified;
} AnofoxDriftResult;

/**
 * Two-sample drift test of current against reference (NaN entries are dropped).
 */
bool anofox_drift_test(int32_t kind, AnofoxDataArray reference, AnofoxDataArray current, AnofoxDriftOptions options,
                       AnofoxDriftResult *out_result, AnofoxError *out_error);

/**
 * k-sample Anderson-Darling test of values grouped by the numeric codes in groups.
 */
bool anofox_ad_test_ksamp(AnofoxDataArray values, AnofoxDataArray groups, bool midrank, AnofoxDriftResult *out_result,
                          AnofoxError *out_error);

/**
 * Options of anofox_column_drift (#300). Must match ColumnDriftOptionsFFI in
 * crates/anofox-stats-ffi/src/drift_ffi.rs.
 */
typedef struct {
	/** evidently's test name; NULL or "auto" for evidently's default choice */
	const char *test;
	/** Drift threshold; NaN for the test's default */
	double threshold;
	/** Zero fill of PSI / KL; NaN for evidently's rule */
	double epsilon;
	/** Permutations of TVD / empirical_mmd */
	size_t n_permutations;
	uint64_t seed;
	/** Bins of the quantile binning */
	size_t n_bins;
	/** 0 auto (evidently), 1 sturges, 2 categorical, 3 quantile */
	int32_t binning;
	bool has_seed;
} AnofoxColumnDriftOptions;

/** Result of anofox_column_drift. Must match ColumnDriftResultFFI. */
typedef struct {
	/** A p-value or a distance (is_p_value) */
	double score;
	double threshold;
	size_t n1;
	size_t n2;
	bool drifted;
	bool is_p_value;
	/** evidently's name of the test that was run */
	char test[ANOFOX_TEST_NAME_LEN];
} AnofoxColumnDriftResult;

/**
 * evidently's drift of one column (NaN entries are dropped). categorical: the
 * values are category codes in label order.
 */
bool anofox_column_drift(AnofoxDataArray reference, AnofoxDataArray current, bool categorical,
                         AnofoxColumnDriftOptions options, AnofoxColumnDriftResult *out_result,
                         AnofoxError *out_error);

/** Result of anofox_dataset_drift. Must match DatasetDriftResultFFI. */
typedef struct {
	double share_drifted;
	size_t n_columns;
	size_t n_drifted;
	bool dataset_drift;
} AnofoxDatasetDriftResult;

/**
 * evidently's dataset drift: drifted when the share of drifted columns is at
 * least drift_share.
 */
bool anofox_dataset_drift(const bool *drifted, size_t len, double drift_share, AnofoxDatasetDriftResult *out_result,
                          AnofoxError *out_error);

/**
 * Drift statistic between two samples of category labels (#301), given as dense
 * codes in label order (NaN entries are dropped): upstream's *_categorical
 * divergences and tvd_test. kind: PSI, KL, Jensen-Shannon, Hellinger or TVD;
 * options.binning is ignored.
 */
bool anofox_drift_test_categorical(int32_t kind, AnofoxDataArray reference, AnofoxDataArray current,
                                   AnofoxDriftOptions options, AnofoxDriftResult *out_result, AnofoxError *out_error);

/**
 * Proportions of the reference and the current sample per category (#301,
 * upstream category_proportions), given dense codes in label order (NaN entries
 * are dropped). Writes n_categories values to each of out_p (reference) and
 * out_q (current), in code order; fails unless the samples have exactly
 * n_categories distinct codes.
 */
bool anofox_category_proportions(AnofoxDataArray reference, AnofoxDataArray current, size_t n_categories,
                                 double *out_p, double *out_q, AnofoxError *out_error);

/* ============================================================================
 * Per-term ANOVA table of an OLS fit (issue #231)
 * ============================================================================ */

/**
 * One term (row) of anofox_ols_anova_table. NaN marks a value that does not
 * apply. Must match AnovaTermFFI in the Rust FFI crate.
 */
typedef struct {
	double df;
	double sum_sq;
	double mean_sq;
	double statistic;
	double p_value;
} AnofoxAnovaTerm;

/**
 * Residual row of anofox_ols_anova_table. Must match OlsAnovaTableResultFFI in
 * the Rust FFI crate.
 */
typedef struct {
	double residual_df;
	double residual_sum_sq;
	double residual_mean_sq;
	size_t n_observations;
} AnofoxOlsAnovaTableResult;

/**
 * Per-term ANOVA table (R's anova(lm(...)) for anova_type 1, car::Anova(fit,
 * type = 2) for anova_type 2) of the OLS fit of y on the x columns, on the rows
 * where every value is finite. term_of_column[j] is the 0-based term of column
 * j; terms are numbered 0..n_terms-1 in table order and out_terms receives
 * n_terms rows.
 */
bool anofox_ols_anova_table(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count,
                            const uint64_t *term_of_column, size_t n_terms, bool fit_intercept, int32_t anova_type,
                            AnofoxAnovaTerm *out_terms, AnofoxOlsAnovaTableResult *out_result,
                            AnofoxError *out_error);

/**
 * Result of anofox_bootstrap_mean_ci (#294). Must match BootstrapMeanResultFFI
 * in crates/anofox-stats-ffi/src/bootstrap_ffi.rs.
 */
typedef struct {
	double estimate;
	double std_error;
	double conf_low;
	double conf_high;
	double conf_level;
	double block_length; /* (expected) block length, 0 for the IID bootstrap */
	size_t n;
	size_t n_bootstrap;
} AnofoxBootstrapMeanResult;

/**
 * Percentile bootstrap confidence interval of the mean of data (NaN entries are
 * dropped; for the block bootstrap the order of data is the series order).
 * method: 0 IID, 1 circular block bootstrap (block_length >= 2), 2 stationary
 * bootstrap (expected block length block_length >= 1), 3 moving block bootstrap
 * (block_length >= 2). The fixed block lengths must be whole numbers.
 */
bool anofox_bootstrap_mean_ci(AnofoxDataArray data, int32_t method, size_t n_bootstrap, double confidence_level,
                              double block_length, uint64_t seed, bool has_seed,
                              AnofoxBootstrapMeanResult *out_result, AnofoxError *out_error);
/* ===========================================================================
 * Collinearity and separation diagnostics (#306). Must match
 * crates/anofox-stats-ffi/src/diagnostics_ffi.rs.
 * ======================================================================== */

/** Result of anofox_condition_diagnostic; free with anofox_free_condition_diagnostic. */
typedef struct {
	double condition_number;
	double condition_number_xtx;
	double scaled_condition_number;
	/** 0 well_conditioned, 1 moderate, 2 high, 3 severe */
	int32_t severity;
	double *singular_values;
	double *condition_indices;
	/** n_components * n_terms, component-major: [k * n_terms + j] */
	double *variance_proportions;
	size_t n_components;
	size_t n_terms;
	/** Warning text or NULL */
	char *warning;
} AnofoxConditionDiagnostic;

/** Result of anofox_separation_check; free with anofox_free_separation_check. */
typedef struct {
	bool has_separation;
	/** 0-based indices into x of the flagged predictors */
	uint64_t *separated_predictors;
	/** Per flagged predictor: 1 complete, 2 quasi, 3 monotonic_response */
	int32_t *separation_types;
	size_t n_separated;
	char *warning;
} AnofoxSeparationCheck;

/** Result of anofox_detect_separation; free with anofox_free_linear_separation. */
typedef struct {
	bool separated;
	/** n_terms values: +inf / -inf for diverging coefficients, 0 otherwise */
	double *infinite;
	double *direction;
	size_t n_terms;
} AnofoxLinearSeparation;

/**
 * Belsley-Kuh-Welsch collinearity diagnostics of [1, x] (fit_intercept) or x:
 * kappa(X, exact = TRUE), condition indices and variance-decomposition
 * proportions of the unit-column-length design (perturb::colldiag).
 */
bool anofox_condition_diagnostic(const AnofoxDataArray *x, size_t x_count, bool fit_intercept,
                                 AnofoxConditionDiagnostic *out_result, AnofoxError *out_error);
void anofox_free_condition_diagnostic(AnofoxConditionDiagnostic *result);

/** Per-predictor separation check; kind 0: binary y (0/1), 1: count y (sparsity). */
bool anofox_separation_check(int32_t kind, AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count,
                             AnofoxSeparationCheck *out_result, AnofoxError *out_error);
void anofox_free_separation_check(AnofoxSeparationCheck *result);

/** Exact separation check of a binary y by the linear program of Konis (2007). */
bool anofox_detect_separation(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, bool fit_intercept,
                              AnofoxLinearSeparation *out_result, AnofoxError *out_error);
void anofox_free_linear_separation(AnofoxLinearSeparation *result);
/* ============================================================================
 * Option pass-throughs -- issue #305
 * ============================================================================ */

/**
 * The lambda anofox_ridge_fit uses on (y, x) with these options: options.alpha,
 * or with options.conditioning upstream's conditioning-aware choice
 * (lambda_small when kappa(X'X) <= cond_threshold) for the rows and columns the
 * fit uses. Same scale as options.alpha.
 */
bool anofox_ridge_selected_lambda(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count,
                                  AnofoxRidgeOptions options, double *out_lambda, AnofoxError *out_error);

/**
 * Poisson / Tweedie / Gamma fits with prior (case) weights w, as R's
 * glm(weights = w) (anofox-regression's PoissonRegressor / TweedieRegressor
 * with weights). Same results as the unweighted anofox_*_fit; a row with a
 * non-finite y, x or w is dropped. glm_lambda and priors are rejected.
 */
bool anofox_poisson_fit_weighted(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxDataArray w,
                                 AnofoxPoissonOptions options, AnofoxGlmFitResultCore *out_result,
                                 AnofoxFitResultInference *out_inference, AnofoxError *out_error);
bool anofox_tweedie_fit_weighted(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxDataArray w,
                                 AnofoxTweedieOptions options, AnofoxGlmFitResultCore *out_result,
                                 AnofoxFitResultInference *out_inference, AnofoxError *out_error);
bool anofox_gamma_fit_weighted(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxDataArray w,
                               AnofoxGammaOptions options, AnofoxGlmFitResultCore *out_result,
                               AnofoxFitResultInference *out_inference, AnofoxError *out_error);
/* ============================================================================
 * Partial / semi-partial correlation -- issue #296
 * ============================================================================ */

/** Options of anofox_partial_cor. Must match PartialCorOptionsFFI. */
typedef struct {
	/** 0 partial (controls removed from x and y), 1 semi-partial (from y only) */
	int32_t kind;
	/** 0 pearson, 1 spearman */
	int32_t method;
} AnofoxPartialCorOptions;

/** Result of anofox_partial_cor. Must match PartialCorResultFFI. */
typedef struct {
	double r;
	double statistic;
	double p_value;
	double df;
	size_t n;
	size_t n_controls;
	/** Unified test fields (#159), method name included; keep last */
	AnofoxTestSummaryFFI unified;
} AnofoxPartialCorResult;

/**
 * Partial or semi-partial correlation of x and y controlling for
 * controls[0..n_controls] (upstream correlation::partial_cor /
 * semi_partial_cor, R ppcor::pcor.test / spcor.test). NaN entries drop their row.
 */
bool anofox_partial_cor(AnofoxDataArray x, AnofoxDataArray y, const AnofoxDataArray *controls, size_t n_controls,
                        AnofoxPartialCorOptions options, AnofoxPartialCorResult *out_result, AnofoxError *out_error);
/* ============================================================================
 * AUC null distribution and studentized range distribution (#304)
 * ============================================================================ */

/** Quantile of the ROC AUC of a random classifier: qwilcox(p, n_pos, n_neg) / (n_pos n_neg). */
bool anofox_auc_null_quantile(double p, int64_t n_pos, int64_t n_neg, double *out_value, AnofoxError *out_error);

/** One-sided p-value P(AUC_random >= auc) of an observed AUC. */
bool anofox_auc_null_p_value(double auc, int64_t n_pos, int64_t n_neg, double *out_value, AnofoxError *out_error);

/** Studentized range distribution function (R ptukey). */
bool anofox_ptukey(double q, double n_means, double df, double n_ranges, bool lower_tail, double *out_value,
                   AnofoxError *out_error);

/** Studentized range quantile function (R qtukey). */
bool anofox_qtukey(double p, double n_means, double df, double n_ranges, bool lower_tail, double *out_value,
                   AnofoxError *out_error);
/* ============================================================================
 * Multiple-model forecast comparison: SPA, MSPE-adjusted SPA, MCS (#295)
 * ============================================================================ */

/**
 * Options of anofox_spa_test. Must match SpaOptionsFFI in
 * crates/anofox-stats-ffi/src/forecast_ffi.rs.
 */
typedef struct {
	size_t n_bootstrap;
	/** Expected stationary-bootstrap block length; <= 0 for T^(1/3) */
	double block_length;
	uint64_t seed;
	bool has_seed;
} AnofoxSpaOptions;

/**
 * Result of anofox_spa_test. Must match SpaResultFFI.
 */
typedef struct {
	double statistic;
	double p_value_lower;
	double p_value_consistent;
	double p_value_upper;
	/** Block length actually used */
	double block_length;
	/** 0-based index of the best competing model, -1 for none */
	int64_t best_model;
	size_t n_bootstrap;
	size_t n;
	size_t n_models;
	/** Unified test fields (#159); keep last */
	AnofoxTestSummaryFFI unified;
} AnofoxSpaResult;

/**
 * SPA-type test of n_models competing series against a benchmark. kind 0: the
 * SPA test on losses; kind 1: the MSPE-adjusted SPA test on forecast errors.
 */
bool anofox_spa_test(int32_t kind, AnofoxDataArray benchmark, const AnofoxDataArray *models, size_t n_models,
                     AnofoxSpaOptions options, AnofoxSpaResult *out_result, AnofoxError *out_error);

/**
 * Options of anofox_model_confidence_set. Must match McsOptionsFFI.
 */
typedef struct {
	double alpha;
	size_t n_bootstrap;
	/** (Expected) block length; <= 0 for T^(1/3) (rounded for the fixed-block schemes) */
	double block_length;
	uint64_t seed;
	/** 0 range (T_R), 1 max (T_max) */
	int32_t statistic;
	/** 0 stationary, 1 moving block (R MCSprocedure(k)), 2 circular block, 3 IID */
	int32_t bootstrap;
	bool has_seed;
} AnofoxMcsOptions;

/**
 * One elimination step of the MCS. Must match McsStepFFI.
 */
typedef struct {
	/** 0-based model index */
	uint64_t model;
	double p_value;
	/** MCS p-value (running maximum of the step p-values) */
	double mcs_p_value;
	bool eliminated;
} AnofoxMcsStep;

/**
 * Result of anofox_model_confidence_set. Must match McsResultFFI.
 */
typedef struct {
	double mcs_p_value;
	/** Block length actually used */
	double block_length;
	size_t n_bootstrap;
	size_t n;
	size_t n_models;
	/** Entries written to out_included */
	size_t n_included;
	/** Entries written to out_steps */
	size_t n_steps;
} AnofoxMcsResult;

/**
 * Model Confidence Set of n_models loss series. out_included, out_steps,
 * out_order (full elimination order, best model last) and out_mcs_p_values
 * (indexed by model) must each hold n_models entries.
 */
bool anofox_model_confidence_set(const AnofoxDataArray *models, size_t n_models, AnofoxMcsOptions options,
                                 uint64_t *out_included, AnofoxMcsStep *out_steps, uint64_t *out_order,
                                 double *out_mcs_p_values, AnofoxMcsResult *out_result, AnofoxError *out_error);

/* ============================================================================
 * Two-way and repeated-measures ANOVA (#297)
 * ============================================================================ */

/**
 * One F test of anofox_two_way_anova. Must match AnovaTermFFI in
 * crates/anofox-stats-ffi/src/anova_ffi.rs.
 */
typedef struct {
	double sum_sq;
	double df;
	double mean_sq;
	double f_statistic;
	double p_value;
	/** Unified test fields (#159); keep last */
	AnofoxTestSummaryFFI unified;
} AnofoxTwoWayAnovaTerm;

/**
 * Result of anofox_two_way_anova. Must match TwoWayAnovaResultFFI in
 * crates/anofox-stats-ffi/src/anova_ffi.rs.
 */
typedef struct {
	/** Factor A, factor B, A x B interaction */
	AnofoxTwoWayAnovaTerm terms[3];
	double residual_sum_sq;
	double residual_df;
	double residual_mean_sq;
	double total_sum_sq;
	double total_df;
	size_t levels_a;
	size_t levels_b;
	size_t n;
	double grand_mean;
} AnofoxTwoWayAnovaResult;

/**
 * Two-way ANOVA with interaction, Type III sums of squares (car::Anova(type = 3)
 * with sum contrasts), of values by the dense 0-based level codes factor_a and
 * factor_b. Every cell must be observed.
 */
bool anofox_two_way_anova(AnofoxDataArray values, AnofoxDataArray factor_a, AnofoxDataArray factor_b,
                          AnofoxTwoWayAnovaResult *out_result, AnofoxError *out_error);

/**
 * Result of anofox_repeated_measures_anova. The sphericity fields are NaN when
 * fewer than three conditions were observed or sphericity was not requested.
 * Must match RmAnovaResultFFI in crates/anofox-stats-ffi/src/anova_ffi.rs.
 */
typedef struct {
	double ss_condition;
	double df_condition;
	double ms_condition;
	double f_statistic;
	double p_value;
	double ss_subjects;
	double df_subjects;
	double ss_error;
	double df_error;
	double ms_error;
	double ss_total;
	size_t n_subjects;
	size_t n_conditions;
	double mauchly_w;
	double mauchly_chi_square;
	double mauchly_df;
	double mauchly_p_value;
	double gg_epsilon;
	double gg_df1;
	double gg_df2;
	double gg_p_value;
	double hf_epsilon;
	double hf_df1;
	double hf_df2;
	double hf_p_value;
	/** Unified test fields of the uncorrected F test (#159); keep last */
	AnofoxTestSummaryFFI unified;
} AnofoxRmAnovaResult;

/**
 * One-way repeated-measures ANOVA of the complete row-major n_subjects x
 * n_conditions matrix data; sphericity adds Mauchly's test and the
 * Greenhouse-Geisser correction (three or more conditions).
 */
bool anofox_repeated_measures_anova(const double *data, size_t n_subjects, size_t n_conditions, bool sphericity,
                                    AnofoxRmAnovaResult *out_result, AnofoxError *out_error);

/* ============================================================================
 * Principal component analysis (#303)
 * ============================================================================ */

/**
 * A fitted PCA. Must match PcaResultFFI in crates/anofox-stats-ffi/src/pca_ffi.rs.
 * The arrays are allocated by anofox_pca_fit; release them with
 * anofox_free_pca_result.
 */
typedef struct {
	/** Column means, n_features values */
	double *center;
	/** Component standard deviations (divisor n - 1), n_components values, decreasing */
	double *sdev;
	/** Share of the total variance per component, n_components values */
	double *proportion_of_variance;
	/** Loadings, component-major: loadings[j * n_features + i] is variable i of component j */
	double *loadings;
	size_t n_features;
	size_t n_components;
	size_t n_observations;
} AnofoxPcaResult;

/**
 * PCA (R's prcomp(x, center = TRUE, scale. = FALSE)) of n_rows rows of
 * n_features values each, row-major in data, keeping the first n_components
 * components (0: all min(n_rows - 1, n_features)).
 */
bool anofox_pca_fit(const double *data, size_t n_rows, size_t n_features, size_t n_components,
                    AnofoxPcaResult *out_result, AnofoxError *out_error);

/** Free the arrays of a result filled by anofox_pca_fit. */
void anofox_free_pca_result(AnofoxPcaResult *result);

/**
 * Scores of one row x (x_len values) on the n_components components of a
 * fitted PCA (predict.prcomp). loadings is component-major as in
 * AnofoxPcaResult; out_scores receives n_components values.
 */
bool anofox_pca_transform(const double *center, const double *sdev, const double *loadings, size_t n_features,
                          size_t n_components, const double *x, size_t x_len, double *out_scores,
                          AnofoxError *out_error);

/**
 * GVIF of one term. Must match GvifTermFFI in
 * crates/anofox-stats-ffi/src/gvif_ffi.rs.
 */
typedef struct {
	uint64_t df;
	double gvif;
	double gvif_adj;
} AnofoxGvifTerm;

/**
 * Generalized VIF (Fox & Monette 1992, car::vif) per term, on the rows where
 * every x value is finite (#307). term_of_column[j] is the 0-based term of
 * column j; out_terms receives n_terms entries.
 */
bool anofox_compute_gvif(const AnofoxDataArray *x, size_t x_count, const uint64_t *term_of_column, size_t n_terms,
                         AnofoxGvifTerm *out_terms, size_t *out_n_observations, AnofoxError *out_error);

/* ============================================================================
 * Predictive accuracy indexes (issue #308). The structs must match
 * crates/anofox-stats-ffi/src/accuracy_ffi.rs. An empty weights array means
 * unweighted.
 * ============================================================================ */

typedef struct {
	double concordant;
	double discordant;
	double tied_x;
	double tied_y;
	double tied_xy;
	double c;
	double dxy;
	double gamma;
	double tau_a;
} AnofoxConcordanceResult;

typedef struct {
	double c;
	double dxy;
	/** Observations (sum of weights when weighted) */
	double n;
} AnofoxSomers2Result;

typedef struct {
	double c;
	double dxy;
	double sd;
	double relevant_pairs;
	double concordant;
	double uncertain;
	size_t n;
	size_t missing;
	size_t uncensored;
} AnofoxRcorrCensResult;

/** Hmisc::R2Measures; the ess fields are NaN without an effective sample size. */
typedef struct {
	double r2;
	double r2_adj;
	double r2_ess;
	double r2_adj_ess;
} AnofoxR2MeasuresResult;

typedef struct {
	double sum_weights;
	double model_lr;
	double df;
	double p_value;
	double c;
	double dxy;
	double gamma;
	double tau_a;
	double r2;
	AnofoxR2MeasuresResult r2_measures;
	/** Effective sample size 3 n1 n0 / n of the R2(m) measures */
	double ess;
	double brier;
	double g;
	double gr;
	double gp;
	size_t n;
} AnofoxLrmStatsResult;

typedef struct {
	double dxy;
	double c;
	double r2;
	double d;
	double d_chisq;
	double d_p;
	double u;
	double u_chisq;
	double u_p;
	double q;
	double brier;
	double intercept;
	double slope;
	double emax;
	double e90;
	double eavg;
	double s_z;
	double s_p;
} AnofoxValProbResult;

/** Concordance of x with the ordered response y (survival::concordance counts). */
bool anofox_concordance(AnofoxDataArray x, AnofoxDataArray y, AnofoxDataArray weights,
                        AnofoxConcordanceResult *out_result, AnofoxError *out_error);

/** Hmisc::somers2(p, y, weights). */
bool anofox_somers2(AnofoxDataArray p, AnofoxDataArray y, AnofoxDataArray weights, AnofoxSomers2Result *out_result,
                    AnofoxError *out_error);

/** Hmisc::rcorr.cens(x, Surv(time, event)); event non-zero = observed. */
bool anofox_rcorr_cens(AnofoxDataArray x, AnofoxDataArray time, AnofoxDataArray event,
                       AnofoxRcorrCensResult *out_result, AnofoxError *out_error);

/** Hmisc::GiniMd(x); NaN for fewer than 2 values. */
bool anofox_gini_md(AnofoxDataArray x, double *out_value, AnofoxError *out_error);

/** Hmisc::R2Measures(lr, p, n, ess); ess NaN for none. */
bool anofox_r2_measures(double lr, double p, double n, double ess, AnofoxR2MeasuresResult *out_result,
                        AnofoxError *out_error);

/** The stats vector of a binary rms::lrm fit from its response and linear predictor. */
bool anofox_lrm_stats(AnofoxDataArray y, AnofoxDataArray lp, AnofoxDataArray weights, double df,
                      AnofoxLrmStatsResult *out_result, AnofoxError *out_error);

/** rms::val.prob(p, y, pl = FALSE). */
bool anofox_val_prob(AnofoxDataArray p, AnofoxDataArray y, AnofoxValProbResult *out_result, AnofoxError *out_error);

/* ============================================================================
 * Embedding drift and multivariate two-sample tests (#302)
 * A sample is n_rows embeddings of dim doubles each, row-major.
 * ============================================================================ */

/**
 * Options of anofox_embedding_drift; each method reads only its own. Must match
 * EmbeddingDriftOptionsFFI in crates/anofox-stats-ffi/src/embedding_ffi.rs.
 */
typedef struct {
	/** evidently's drift test of each component (ratio); NULL for "wasserstein" */
	const char *component_test;
	/** Fixed threshold; NaN for evidently's default */
	double threshold;
	/** Bootstrap quantile; NaN for evidently's default */
	double quantile;
	/** Per-component threshold (ratio); NaN for the test's default */
	double component_threshold;
	size_t n_bootstrap;
	uint64_t seed;
	/** 0 distance, 1 ratio, 2 mmd */
	int32_t method;
	/** 0 euclidean, 1 cosine, 2 cityblock, 3 chebyshev */
	int32_t metric;
	bool bootstrap;
	bool has_seed;
} AnofoxEmbeddingDriftOptions;

/** Result of anofox_embedding_drift. Must match EmbeddingDriftResultFFI. */
typedef struct {
	double score;
	double threshold;
	size_t n1;
	size_t n2;
	size_t dim;
	bool drifted;
} AnofoxEmbeddingDriftResult;

/**
 * evidently's embedding drift of current against reference. The options are
 * parsed before the samples are looked at: a call with empty samples validates
 * them (INVALID_INPUT for a bad name, otherwise insufficient data).
 */
bool anofox_embedding_drift(const double *reference, size_t n_reference, const double *current, size_t n_current,
                            size_t dim, AnofoxEmbeddingDriftOptions options, AnofoxEmbeddingDriftResult *out_result,
                            AnofoxError *out_error);

/** Options of anofox_multivariate_two_sample. Must match MultivariateTestOptionsFFI. */
typedef struct {
	/** Bandwidth of the gaussian / laplacian kernel */
	double bandwidth;
	/** Polynomial kernel (scale x.y + offset)^degree */
	double scale;
	double offset;
	size_t n_permutations;
	uint64_t seed;
	uint32_t degree;
	/** MMD kernel: 0 gaussian, 1 laplacian, 2 linear, 3 polynomial */
	int32_t kernel;
	bool has_seed;
} AnofoxMultivariateTestOptions;

/** Result of anofox_multivariate_two_sample. Must match MultivariateTestResultFFI. */
typedef struct {
	double statistic;
	double p_value;
	size_t n1;
	size_t n2;
	size_t dim;
	size_t n_permutations;
	/** Unified test fields (#159); keep last */
	AnofoxTestSummaryFFI unified;
} AnofoxMultivariateTestResult;

/**
 * Multivariate two-sample permutation test of x against y. kind: 0 energy
 * distance, 1 MMD with options.kernel.
 */
bool anofox_multivariate_two_sample(int32_t kind, const double *x, size_t n_x, const double *y, size_t n_y,
                                    size_t dim, AnofoxMultivariateTestOptions options,
                                    AnofoxMultivariateTestResult *out_result, AnofoxError *out_error);

/* ============================================================================
 * Cluster-robust (sandwich) covariance of OLS / WLS (#309)
 * ============================================================================
 * sandwich::vcovCL(lm(...), cluster, type, cadjust) through upstream's
 * cluster_vcov. clusters holds n_dims arrays of y.len non-negative codes (two
 * or more dimensions: multiway clustering, Cameron-Gelbach-Miller); the type is
 * options.hc_type (ANOFOX_HC_NONE = HC1, vcovCL's default for linear models).
 * With options.compute_inference the inference and the F test are the
 * cluster-robust ones and out_vcov receives the dim x dim clustered covariance
 * (intercept first; free with anofox_free_interval_matrix), else NULL and 0.
 * out_n_clusters (NULL or n_dims entries) receives the number of clusters per
 * dimension among the fitted rows (0 without inference). out_inference,
 * out_vcov and out_dim may be NULL.
 */
bool anofox_ols_cluster_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count,
                            const int32_t *const *clusters, size_t n_dims, bool cadjust,
                            AnofoxOlsOptions options, AnofoxFitResultCore *out_core,
                            AnofoxFitResultInference *out_inference, double **out_vcov, size_t *out_dim,
                            int64_t *out_n_clusters, AnofoxError *out_error);
bool anofox_wls_cluster_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxDataArray weights,
                            const int32_t *const *clusters, size_t n_dims, bool cadjust,
                            AnofoxWlsOptions options, AnofoxFitResultCore *out_core,
                            AnofoxFitResultInference *out_inference, double **out_vcov, size_t *out_dim,
                            int64_t *out_n_clusters, AnofoxError *out_error);

/* ============================================================================
 * Likelihood-ratio and score tests, profile-likelihood CIs (#310)
 * ============================================================================ */

/**
 * Likelihood summary of a fitted model STRUCT. Unknown numbers are NaN, unknown
 * names empty, an unknown n_observations is -1. Must match ModelLikelihoodFFI
 * in crates/anofox-stats-ffi/src/likelihood_ffi.rs.
 */
typedef struct {
	char model_type[ANOFOX_MODEL_NAME_LEN];
	char family[ANOFOX_MODEL_NAME_LEN];
	double log_likelihood;
	double null_log_likelihood;
	double deviance;
	double null_deviance;
	double dispersion;
	uint64_t n_parameters; /* intercept included */
	int64_t n_observations;
} AnofoxModelLikelihood;

/** Result of anofox_lr_test / anofox_glm_score_test. Must match ChiSqTestResultFFI. */
typedef struct {
	double statistic;
	double df;
	double p_value;
	double dispersion;
	double theta;
	int64_t n_observations;
	/** Unified test fields (#159); keep last */
	AnofoxTestSummaryFFI unified;
} AnofoxChiSqTestResult;

/**
 * GLM family: kind 0 gaussian, 1 poisson, 2 binomial, 3 negative binomial,
 * 4 gamma, 5 tweedie (the GLMM family codes); link: the AnofoxPoissonLink /
 * AnofoxBinomialLink code; param: Negative Binomial theta (NaN: estimate) or
 * Tweedie power. Must match LikelihoodFamilyFFI.
 */
typedef struct {
	int32_t kind;
	int32_t link;
	double param;
} AnofoxLikelihoodFamily;

/**
 * Likelihood-ratio test of full against reduced (R's anova(reduced, full,
 * test = "LRT") / lmtest::lrtest), or of full against its null model when
 * reduced is NULL. Incompatible models are INVALID_INPUT.
 */
bool anofox_lr_test(AnofoxModelLikelihood full, const AnofoxModelLikelihood *reduced,
                    AnofoxChiSqTestResult *out_result, AnofoxError *out_error);

/**
 * Rao score test for adding the x_add columns to the GLM of y on x_reduced
 * (R's anova(reduced, full, test = "Rao")), on the rows where every value is
 * finite. offset_column: 1-based column of x_reduced (0: none). out_z receives
 * x_add_count per-column z statistics.
 */
bool anofox_glm_score_test(AnofoxLikelihoodFamily family, AnofoxDataArray y, const AnofoxDataArray *x_reduced,
                           size_t x_reduced_count, const AnofoxDataArray *x_add, size_t x_add_count,
                           bool fit_intercept, uint64_t offset_column, double *out_z,
                           AnofoxChiSqTestResult *out_result, AnofoxError *out_error);

/**
 * Profile-likelihood confidence intervals (R's confint(glm)) of the GLM of y on
 * x with fitted feature coefficients `coefficients` (NaN: dropped column, NaN
 * interval). offset_column: 1-based column of x (0: none). out_lower/out_upper
 * get coefficients_len bounds, out_intercept 2 (NaN without intercept). An
 * unreached bound is NaN.
 */
bool anofox_glm_profile_confint(AnofoxLikelihoodFamily family, AnofoxDataArray y, const AnofoxDataArray *x,
                                size_t x_count, const double *coefficients, size_t coefficients_len,
                                bool fit_intercept, uint64_t offset_column, double level, double *out_lower,
                                double *out_upper, double *out_intercept, AnofoxError *out_error);

/* ---- Multinomial logistic regression (#314) ---- */

/**
 * Options of anofox_multinomial_fit. Must match MultinomialOptionsFFI in
 * crates/anofox-stats-ffi/src/multinomial_ffi.rs.
 */
typedef struct {
	double lambda;   /* penalty strength 1 / C, >= 0 */
	double l1_ratio; /* elastic-net mixing in [0, 1] */
	double tolerance;
	uint32_t max_iterations;
	bool fit_intercept;
} AnofoxMultinomialOptions;

/**
 * A fitted multinomial logistic regression. Must match MultinomialFitResultFFI.
 * coefficients: row-major n_rows x n_features (one row for two classes,
 * n_classes rows otherwise); intercepts: n_rows values. Release with
 * anofox_free_multinomial_result.
 */
typedef struct {
	double *coefficients;
	double *intercepts;
	size_t n_rows;
	size_t n_features;
	size_t n_classes;
	size_t n_observations;
	double log_likelihood;
	double objective;
	double accuracy;
	uint32_t iterations;
	bool converged;
} AnofoxMultinomialFitResult;

/**
 * Multinomial (softmax) logistic regression with an elastic-net penalty
 * (upstream MultinomialLogisticRegressor, sklearn LogisticRegression) of the
 * class indices y (0..K-1, every class present) on the x columns, with
 * optional sample weights (NULL: unweighted). Fewer than two rows or classes
 * is INSUFFICIENT_DATA.
 */
bool anofox_multinomial_fit(const double *y, size_t n, const AnofoxDataArray *x, size_t n_features,
                            const double *weights, AnofoxMultinomialOptions options,
                            AnofoxMultinomialFitResult *out_result, AnofoxError *out_error);

/** Free the arrays of a result filled by anofox_multinomial_fit. */
void anofox_free_multinomial_result(AnofoxMultinomialFitResult *result);

/**
 * Class probabilities (n_classes values to out_proba) and the most probable
 * class index (out_class) of one row x under a fitted model's coefficients
 * (row-major n_rows x n_features) and intercepts (n_rows).
 */
bool anofox_multinomial_predict_proba(const double *coefficients, const double *intercepts, size_t n_rows,
                                      size_t n_features, size_t n_classes, const double *x, size_t x_len,
                                      double *out_proba, size_t *out_class, AnofoxError *out_error);

/* ============================================================================
 * rms-style Wald ANOVA and fast backward step-down (issue #311)
 * ============================================================================ */

/**
 * One row of anofox_wald_anova (rms::anova.rms). NaN marks a value that does
 * not apply. Must match WaldAnovaRowFFI in crates/anofox-stats-ffi/src/wald_ffi.rs.
 */
typedef struct {
	char *label;
	char *kind;
	int64_t term;      /* 0-based term, -1 for the totals */
	uint64_t *columns; /* 0-based slope columns tested */
	size_t n_columns;
	double chi_square;
	double df;
	double f_statistic;
	double df_residual;
	double p_value;
	double partial_ss;
} AnofoxWaldAnovaRow;

/**
 * Result of anofox_wald_anova; free with anofox_free_wald_anova. The error row
 * is NaN unless an F test with sigma2 was requested. Must match
 * WaldAnovaResultFFI.
 */
typedef struct {
	AnofoxWaldAnovaRow *rows;
	size_t n_rows;
	double error_df;
	double error_ss;
	double error_ms;
} AnofoxWaldAnovaResult;

/**
 * rms::anova.rms table of a fitted model: slopes `coefficients` (p, feature
 * order), row-major covariance vcov (dim = p or p + 1, intercept first).
 * term_of_column[j]: 0-based term of slope j (negative: none); term_names:
 * n_terms names; interaction_factors (NULL: none): two 0-based main-effect
 * terms per term, negative for a main effect; nonlinear (NULL: all linear): p
 * flags. f_test refers the rows to F(df, df_residual); a finite sigma2 adds
 * partial sums of squares.
 */
bool anofox_wald_anova(const double *coefficients, size_t p, const double *vcov, size_t dim,
                       const int64_t *term_of_column, const bool *nonlinear, const char *const *term_names,
                       size_t n_terms, const int64_t *interaction_factors, bool f_test, double df_residual,
                       double sigma2, AnofoxWaldAnovaResult *out_result, AnofoxError *out_error);

void anofox_free_wald_anova(AnofoxWaldAnovaResult *result);

/**
 * One deletion of anofox_fastbw (rms::fastbw). Must match FastbwStepFFI.
 */
typedef struct {
	uint64_t term;
	double chi_square;
	double df;
	double p_value;
	double residual_chi_square;
	double residual_df;
	double residual_p_value;
	double aic;
} AnofoxFastbwStep;

/**
 * Result of anofox_fastbw; free with anofox_free_fastbw. Must match
 * FastbwResultFFI.
 */
typedef struct {
	AnofoxFastbwStep *deleted;
	size_t n_deleted;
	uint64_t *kept_terms;
	size_t n_kept;
	double *coefficients; /* approximate reduced-model slopes, 0 when deleted */
	size_t n_coefficients;
	double intercept; /* NaN without one */
} AnofoxFastbwResult;

/**
 * rms::fastbw of a fitted model (inputs as anofox_wald_anova; slopes without a
 * term are never deleted). rule 0: AIC (chi2 - k_aic * df <= aics), 1: P
 * (p > sls); individual applies the rule to the term's own test instead of the
 * residual test.
 */
bool anofox_fastbw(const double *coefficients, size_t p, double intercept, const double *vcov, size_t dim,
                   const int64_t *term_of_column, size_t n_terms, int32_t rule, bool individual, double k_aic,
                   double aics, double sls, AnofoxFastbwResult *out_result, AnofoxError *out_error);

void anofox_free_fastbw(AnofoxFastbwResult *result);

/* ============================================================================
 * Cox proportional hazards regression -- issue #312 (crates/anofox-stats-ffi/src/cox_ffi.rs)
 * ============================================================================ */

/** Options of a Cox fit. ties: 0 Efron, 1 Breslow. zph_transform (anofox_cox_zph only):
 * 0 km, 1 rank, 2 identity, 3 log. */
typedef struct {
	double tolerance;
	double ridge;
	double confidence_level;
	uint32_t max_iterations;
	int32_t ties;
	int32_t zph_transform;
	bool timefix;
	bool robust;
} AnofoxCoxOptions;

/** Rows of a Cox fit; x column-major (n_rows * n_features), optional columns NULL when absent.
 * Rows with a non-finite value in any column are left out. */
typedef struct {
	const double *time;
	const double *event;
	const double *x;
	const double *weights;
	const double *entry;
	const double *offset;
	const double *strata;
	const double *cluster;
	size_t n_rows;
	size_t n_features;
} AnofoxCoxData;

/** A fitted Cox model (release with anofox_free_cox_fit_result). vcov is n_features^2, row-major;
 * the baseline is the cumulative hazard at covariates 0 per stratum (stratum NaN unstratified). */
typedef struct {
	double *coefficients;
	double *std_errors;
	double *naive_std_errors;
	double *z_values;
	double *p_values;
	double *conf_low;
	double *conf_high;
	double *hazard_ratios;
	double *hr_conf_low;
	double *hr_conf_high;
	double *vcov;
	size_t n_features;
	double *baseline_stratum;
	double *baseline_time;
	double *baseline_cumhaz;
	size_t n_baseline;
	double log_likelihood;
	double null_log_likelihood;
	double lr_statistic;
	double lr_p_value;
	double wald_statistic;
	double wald_p_value;
	double score_statistic;
	double score_p_value;
	double concordance;
	double r_squared;
	double aic;
	double bic;
	double confidence_level;
	size_t df;
	size_t n_observations;
	size_t n_events;
	size_t n_strata;
	size_t iterations;
	bool converged;
	bool robust;
	AnofoxModelSummaryFFI summary;
} AnofoxCoxFitResult;

/** Proportional-hazards tests, one per covariate and the global test last (len = n_features + 1). */
typedef struct {
	double *chisq;
	double *df;
	double *p_value;
	size_t len;
} AnofoxCoxZphResult;

/** Per-input-row results (NaN for rows left out); *_mat are n_rows * n_features, row-major. */
typedef struct {
	double *linear_predictor;
	double *risk;
	double *martingale;
	double *deviance;
	double *stratum;
	double *score_mat;
	double *dfbeta_mat;
	double *dfbetas_mat;
	double *schoenfeld_mat;
	double *scaled_schoenfeld_mat;
	bool *is_training;
	size_t n_rows;
	size_t n_features;
} AnofoxCoxAugmentResult;

bool anofox_cox_fit(const AnofoxCoxData *data, AnofoxCoxOptions options, AnofoxCoxFitResult *out_result,
                    AnofoxError *out_error);
void anofox_free_cox_fit_result(AnofoxCoxFitResult *result);
bool anofox_cox_zph(const AnofoxCoxData *data, AnofoxCoxOptions options, AnofoxCoxZphResult *out_result,
                    AnofoxError *out_error);
void anofox_free_cox_zph_result(AnofoxCoxZphResult *result);
bool anofox_cox_augment(const AnofoxCoxData *data, AnofoxCoxOptions options, AnofoxCoxAugmentResult *out_result,
                        AnofoxError *out_error);
void anofox_free_cox_augment_result(AnofoxCoxAugmentResult *result);
/** S(t | x) at n_times times from the stored baseline (exp(-H0(t) exp(x'beta))). */
bool anofox_cox_survival(const double *coefficients, size_t n_coefficients, const double *base_time,
                         const double *base_cumhaz, size_t n_baseline, const double *x, const double *times,
                         size_t n_times, double *out_survival);

/* ---- AFT residuals, extended survfit / survdiff (#322) ---- */

/** Per-input-row results of an AFT fit (NaN for rows left out); score_mat is n_rows * n_params, row-major
 *  (intercept, coefficients, log(scale)). */
typedef struct {
	double *linear_predictor;
	double *standardized_residual;
	double *cox_snell_residual;
	double *score_mat;
	bool *is_training;
	size_t n_rows;
	size_t n_params;
} AnofoxAftAugmentResult;

bool anofox_aft_augment(AnofoxDataArray time, const AnofoxDataArray *x, size_t x_count, AnofoxDataArray event,
                        AnofoxAftOptions options, AnofoxAftAugmentResult *out_result, AnofoxError *out_error);
void anofox_free_aft_augment_result(AnofoxAftAugmentResult *result);

/** conf_type: 0 log, 1 log-log, 2 plain; hazard: 0 Nelson-Aalen, 1 Fleming-Harrington;
 *  robust: -1 survfit's default, 0 Greenwood, 1 infinitesimal jackknife. */
typedef struct {
	double conf_level;
	int32_t conf_type;
	int32_t hazard;
	int32_t robust;
	bool timefix;
} AnofoxSurvfitOptions;

/** time / event (non-zero = event) required; entry, weights, cluster NULL when absent. */
typedef struct {
	const double *time;
	const double *event;
	const double *entry;
	const double *weights;
	const double *cluster;
	size_t n_rows;
} AnofoxSurvfitData;

/** One curve, len distinct times; counts are weighted sums; NaN where R gives NA. */
typedef struct {
	double *time;
	double *n_risk;
	double *n_event;
	double *n_censor;
	double *surv;
	double *std_err;
	double *conf_low;
	double *conf_high;
	double *cumhaz;
	double *std_chaz;
	size_t len;
	size_t n;
} AnofoxSurvfitResult;

bool anofox_survfit(const AnofoxSurvfitData *data, AnofoxSurvfitOptions options, AnofoxSurvfitResult *out_result,
                    AnofoxError *out_error);
void anofox_free_survfit_result(AnofoxSurvfitResult *result);

/** G-rho test (survdiff(rho)); strata NULL or time.len values. */
bool anofox_survdiff(AnofoxDataArray time, AnofoxDataArray event, const uint64_t *groups, const double *strata,
                     double rho, AnofoxTestSummaryFFI *out_result, AnofoxError *out_error);

#ifdef __cplusplus
}
#endif

#endif /* ANOFOX_STATS_FFI_H */
