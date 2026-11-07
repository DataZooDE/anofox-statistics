#ifndef ANOFOX_RECONSTRUCTION_FFI_H
#define ANOFOX_RECONSTRUCTION_FFI_H

/* ============================================================================
 * PCA reconstruction-error drift (#366) and reference-band alert thresholds
 * (#367): upstream anofox-statistics monitoring. The structs must match
 * crates/anofox-stats-ffi/src/reconstruction_ffi.rs.
 * ============================================================================ */

#include "anofox_stats_ffi.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * A fitted reconstruction model. Filled by anofox_reconstruction_fit (release
 * with anofox_free_reconstruction_model) or by the caller as input (only read).
 */
typedef struct {
	/** Reference column means, n_features values */
	double *center;
	/** Population column sds, n_features values; NULL when unscaled */
	double *scale;
	/** Standard deviations of the kept components, n_components values */
	double *sdev;
	/** Share of the total variance per kept component, n_components values */
	double *proportion_of_variance;
	/** Loadings, component-major: loadings[j * n_features + i] is variable i of component j */
	double *loadings;
	size_t n_features;
	size_t n_components;
	size_t n_observations;
	bool impute_nan;
} AnofoxReconstructionModel;

/** Reconstruction error of one chunk. */
typedef struct {
	size_t n;
	double value;
	double std;
	double sampling_error;
	double lower_confidence;
	double upper_confidence;
} AnofoxReconstructionChunk;

/** 0 standard deviation, 1 constant, 2 relative; a NaN side is disabled. */
typedef struct {
	int32_t method;
	double lower;
	double upper;
} AnofoxThresholdMethod;

/** Value limits of a metric; NaN is no limit. */
typedef struct {
	double lower;
	double upper;
	bool override_with_none;
} AnofoxValueLimits;

/** An alert band; NaN is no threshold on that side. */
typedef struct {
	double lower;
	double upper;
} AnofoxThresholdBand;

/**
 * NannyML's reconstruction model of n_rows reference rows (row-major, NaN is
 * missing): the components explaining more than variance_share of the
 * variance, or n_components when non-zero.
 */
bool anofox_reconstruction_fit(const double *data, size_t n_rows, size_t n_features, double variance_share,
                               size_t n_components, bool scale, bool impute_nan, AnofoxReconstructionModel *out_result,
                               AnofoxError *out_error);

/** Free the arrays of a model filled by anofox_reconstruction_fit. */
void anofox_free_reconstruction_model(AnofoxReconstructionModel *model);

/** Reconstruction error of each of n_rows rows; out_errors receives n_rows values. */
bool anofox_reconstruction_row_errors(const AnofoxReconstructionModel *model, const double *data, size_t n_rows,
                                      double *out_errors, AnofoxError *out_error);

/** Mean reconstruction error of a chunk with its sampling error and confidence band. */
bool anofox_reconstruction_chunk(const AnofoxReconstructionModel *model, const double *data, size_t n_rows,
                                 AnofoxReconstructionChunk *out_result, AnofoxError *out_error);

/** Alert band of a metric from its reference values (upstream threshold_band). */
bool anofox_threshold_band(const double *values, size_t n, AnofoxThresholdMethod method, AnofoxValueLimits limits,
                           AnofoxThresholdBand *out_result, AnofoxError *out_error);

/** Alert band of the reconstruction error from its reference chunk values (upstream reconstruction_thresholds). */
bool anofox_reconstruction_thresholds(const double *values, size_t n, AnofoxThresholdMethod method,
                                      AnofoxThresholdBand *out_result, AnofoxError *out_error);

#ifdef __cplusplus
}
#endif

#endif /* ANOFOX_RECONSTRUCTION_FFI_H */
