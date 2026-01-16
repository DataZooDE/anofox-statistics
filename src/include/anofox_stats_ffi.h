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
 * OLS options for FFI
 */
typedef struct {
    /** Whether to fit intercept */
    bool fit_intercept;
    /** Whether to compute inference statistics */
    bool compute_inference;
    /** Confidence level for CIs */
    double confidence_level;
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
 * WLS (Weighted Least Squares) options for FFI
 */
typedef struct {
    /** Whether to fit intercept */
    bool fit_intercept;
    /** Whether to compute inference statistics */
    bool compute_inference;
    /** Confidence level for CIs */
    double confidence_level;
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
    size_t len;
    bool has_standardized;
    bool has_studentized;
    bool has_leverage;
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
} AnofoxGlmFitResultCore;

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
} AnofoxBinomialOptions;

/**
 * Negative Binomial regression options
 */
typedef struct {
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
} AnofoxTweedieOptions;

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
bool anofox_negbinomial_fit(AnofoxDataArray y, const AnofoxDataArray *x, size_t x_count, AnofoxNegBinomialOptions options,
                            AnofoxGlmFitResultCore *out_core, AnofoxFitResultInference *out_inference,
                            AnofoxError *out_error);

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
                    AnofoxAlmFitResultCore *out_core, AnofoxFitResultInference *out_inference,
                    AnofoxError *out_error);

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
 */
bool anofox_fisher_exact(size_t a, size_t b, size_t c, size_t d, AnofoxFisherExactOptions options,
                         AnofoxTestResult *out_result, AnofoxError *out_error);

/**
 * Energy distance test
 */
bool anofox_energy_distance(AnofoxDataArray group1, AnofoxDataArray group2, AnofoxEnergyDistanceOptions options,
                            AnofoxTestResult *out_result, AnofoxError *out_error);

/**
 * Maximum Mean Discrepancy (MMD) test
 */
bool anofox_mmd(AnofoxDataArray group1, AnofoxDataArray group2, AnofoxMmdOptions options,
                AnofoxTestResult *out_result, AnofoxError *out_error);

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
bool anofox_tost_correlation(AnofoxDataArray x, AnofoxDataArray y, double rho_null,
                              double bound_lower, double bound_upper, double alpha,
                              AnofoxTostCorrelationMethod method,
                              AnofoxTostResult *out_result, AnofoxError *out_error);

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
} AnofoxKappaResult;

/**
 * Chi-square goodness-of-fit test
 */
bool anofox_chisq_goodness_of_fit(const size_t *observed, size_t observed_len,
                                   const double *expected, size_t expected_len,
                                   AnofoxChiSquareResult *out_result, AnofoxError *out_error);

/**
 * One-sample proportion z-test
 */
bool anofox_prop_test_one(size_t successes, size_t trials, double p0,
                          AnofoxAlternative alternative,
                          AnofoxPropTestResult *out_result, AnofoxError *out_error);

/**
 * Two-sample proportion z-test
 */
bool anofox_prop_test_two(size_t successes1, size_t trials1,
                          size_t successes2, size_t trials2,
                          AnofoxAlternative alternative, bool correction,
                          AnofoxPropTestResult *out_result, AnofoxError *out_error);

/**
 * Exact binomial test
 */
bool anofox_binom_test(size_t successes, size_t trials, double p0,
                       AnofoxAlternative alternative,
                       AnofoxPropTestResult *out_result, AnofoxError *out_error);

/**
 * Cramer's V effect size for contingency tables
 */
bool anofox_cramers_v(const size_t *table, const size_t *row_lengths, size_t n_rows,
                      double *out_result, AnofoxError *out_error);

/**
 * Cohen's kappa for inter-rater agreement
 */
bool anofox_cohen_kappa(const size_t *table, const size_t *row_lengths, size_t n_rows,
                        bool weighted, AnofoxKappaResult *out_result, AnofoxError *out_error);

/**
 * G-test (log-likelihood ratio test) for contingency tables
 * Takes a flattened contingency table (row-major order)
 */
bool anofox_g_test(const size_t *table, const size_t *row_lengths, size_t n_rows,
                   AnofoxChiSquareResult *out_result, AnofoxError *out_error);

/**
 * McNemar's test for paired categorical data
 * Takes a 2x2 contingency table as 4 cell counts: a, b, c, d
 */
bool anofox_mcnemar_test(size_t a, size_t b, size_t c, size_t d,
                         bool correction, bool exact,
                         AnofoxChiSquareResult *out_result, AnofoxError *out_error);

/**
 * Phi coefficient for 2x2 contingency tables
 * Takes a 2x2 contingency table as 4 cell counts: a, b, c, d
 */
bool anofox_phi_coefficient(size_t a, size_t b, size_t c, size_t d,
                            double *out_result, AnofoxError *out_error);

/**
 * Contingency coefficient (Pearson's C)
 */
bool anofox_contingency_coef(const size_t *table, const size_t *row_lengths, size_t n_rows,
                             double *out_result, AnofoxError *out_error);

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
bool anofox_distance_cor(AnofoxDataArray x, AnofoxDataArray y,
                         AnofoxDistanceCorResult *out_result, AnofoxError *out_error);

/**
 * Distance correlation test with permutations
 */
bool anofox_distance_cor_test(AnofoxDataArray x, AnofoxDataArray y, size_t n_permutations,
                              AnofoxTestResult *out_result, AnofoxError *out_error);

/**
 * ICC type codes
 */
typedef enum {
    ANOFOX_ICC_SINGLE = 0,
    ANOFOX_ICC_AVERAGE = 1,
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
} AnofoxIccResult;

/**
 * Intraclass correlation coefficient (ICC)
 *
 * @param data Data matrix in row-major order (n_subjects x n_raters)
 * @param n_subjects Number of subjects
 * @param n_raters Number of raters/measurements per subject
 * @param icc_type ICC type (single or average)
 * @param out_result Output: ICC result
 * @param out_error Output: error information
 * @return true on success, false on error
 */
bool anofox_icc(const double *data, size_t n_subjects, size_t n_raters,
                AnofoxIccType icc_type, AnofoxIccResult *out_result, AnofoxError *out_error);

/**
 * Free memory allocated by ICC result functions
 */
void anofox_free_icc_result(AnofoxIccResult *result);

/* --- Parametric Tests --- */

/**
 * Yuen's trimmed mean test
 */
bool anofox_yuen_test(AnofoxDataArray group1, AnofoxDataArray group2, double trim,
                      AnofoxAlternative alternative, double confidence_level,
                      AnofoxTestResult *out_result, AnofoxError *out_error);

/**
 * Brown-Forsythe test for homogeneity of variances
 */
bool anofox_brown_forsythe(AnofoxDataArray values, AnofoxDataArray groups,
                           AnofoxTestResult *out_result, AnofoxError *out_error);

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
                            AnofoxForecastLoss loss, AnofoxForecastVarEstimator var_estimator,
                            size_t horizon, AnofoxAlternative alternative,
                            AnofoxTestResult *out_result, AnofoxError *out_error);

/**
 * Clark-West test for nested model comparison
 */
bool anofox_clark_west(AnofoxDataArray actual, AnofoxDataArray forecast_restricted,
                       AnofoxDataArray forecast_unrestricted, size_t horizon,
                       AnofoxTestResult *out_result, AnofoxError *out_error);

/* --- Resampling Tests --- */

/**
 * Permutation t-test (distribution-free alternative to t-test)
 */
bool anofox_permutation_t_test(AnofoxDataArray group1, AnofoxDataArray group2,
                                AnofoxAlternative alternative, size_t n_permutations,
                                uint64_t seed, bool has_seed,
                                AnofoxTestResult *out_result, AnofoxError *out_error);

/* --- Free Functions --- */

/**
 * Free memory allocated by test functions
 */
void anofox_free_test_result(AnofoxTestResult *result);

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

#ifdef __cplusplus
}
#endif

#endif /* ANOFOX_STATS_FFI_H */
