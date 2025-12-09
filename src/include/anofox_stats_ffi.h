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
bool anofox_ols_fit(AnofoxDataArray y,
                    const AnofoxDataArray *x,
                    size_t x_count,
                    AnofoxOlsOptions options,
                    AnofoxFitResultCore *out_core,
                    AnofoxFitResultInference *out_inference,
                    AnofoxError *out_error);

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
bool anofox_ridge_fit(AnofoxDataArray y,
                      const AnofoxDataArray *x,
                      size_t x_count,
                      AnofoxRidgeOptions options,
                      AnofoxFitResultCore *out_core,
                      AnofoxFitResultInference *out_inference,
                      AnofoxError *out_error);

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
bool anofox_elasticnet_fit(AnofoxDataArray y,
                           const AnofoxDataArray *x,
                           size_t x_count,
                           AnofoxElasticNetOptions options,
                           AnofoxFitResultCore *out_core,
                           AnofoxError *out_error);

/**
 * Get library version string
 */
const char *anofox_version(void);

#ifdef __cplusplus
}
#endif

#endif /* ANOFOX_STATS_FFI_H */
