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

#ifdef __cplusplus
}
#endif

#endif /* ANOFOX_STATS_FFI_H */
