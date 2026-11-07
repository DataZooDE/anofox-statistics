#ifndef ANOFOX_CBPE_FFI_H
#define ANOFOX_CBPE_FFI_H

/*
 * Confidence-based performance estimation of a binary classifier (#365), as
 * NannyML's CBPE: upstream anofox-statistics monitoring::cbpe. Must match
 * crates/anofox-stats-ffi/src/cbpe_ffi.rs.
 *
 * Calibration codes: 0 isotonic, 1 platt, 2 none. predictions: NULL (every row
 * uses score >= threshold) or one code per row: 0 negative, 1 positive, -1
 * missing (falls back to the threshold). Metrics are in the order
 * ANOFOX_CBPE_METRIC_NAMES.
 */

#include "anofox_stats_ffi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ANOFOX_CBPE_SAMPLING_FIELDS 15
#define ANOFOX_CBPE_METRICS         10

/** A fitted model; knots (isotonic only) released with anofox_free_cbpe_model. platt_a / b NaN unless Platt. */
typedef struct {
	double *knots_x;
	double *knots_y;
	size_t n_knots;
	size_t n_reference;
	int32_t calibration;
	double threshold;
	double platt_a;
	double platt_b;
	double sampling[ANOFOX_CBPE_SAMPLING_FIELDS];
} AnofoxCbpeModel;

/** The parts of a stored model (rebuilt upstream with CbpeModel::from_parts). */
typedef struct {
	const double *knots_x;
	const double *knots_y;
	size_t n_knots;
	int32_t calibration;
	double threshold;
	double platt_a;
	double platt_b;
	double sampling[ANOFOX_CBPE_SAMPLING_FIELDS];
} AnofoxCbpeModelParts;

typedef struct {
	double value;
	double sampling_error;
	double lower_confidence;
	double upper_confidence;
} AnofoxCbpeMetric;

typedef struct {
	AnofoxCbpeMetric metrics[ANOFOX_CBPE_METRICS];
	size_t n;
} AnofoxCbpeEstimate;

/** NaN where a metric is undefined (NannyML's rules). */
typedef struct {
	double metrics[ANOFOX_CBPE_METRICS];
	size_t n;
} AnofoxCbpeRealized;

bool anofox_cbpe_fit(const double *scores, const bool *labels, const int8_t *predictions, size_t n, int32_t calibration,
                     double threshold, AnofoxCbpeModel *out_result, AnofoxError *out_error);
void anofox_free_cbpe_model(AnofoxCbpeModel *model);
bool anofox_cbpe_estimate(AnofoxCbpeModelParts model, const double *scores, const int8_t *predictions, size_t n,
                          AnofoxCbpeEstimate *out_result, AnofoxError *out_error);
bool anofox_cbpe_realized(AnofoxCbpeModelParts model, const double *scores, const bool *labels,
                          const int8_t *predictions, size_t n, AnofoxCbpeRealized *out_result, AnofoxError *out_error);

#ifdef __cplusplus
}
#endif

#endif /* ANOFOX_CBPE_FFI_H */
