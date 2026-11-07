//! C FFI for the metric pack (#368: classification and regression metrics)
//! and the ranking metrics (#369). The structs are mirrored in
//! `src/include/anofox_stats_ffi.h`.

use super::*;
use anofox_stats_core::tests::metrics::{
    classification_metrics, ranking_metrics, regression_metrics, Average, ClassificationMetrics,
    RankingMetrics, RankingOptions, RegressionMetrics,
};

/// Result of `anofox_classification_metrics` (the confusion counts are
/// written to a separate buffer).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ClassificationMetricsFFI {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub specificity: f64,
    pub balanced_accuracy: f64,
    pub mcc: f64,
    /// NaN when undefined.
    pub cohen_kappa: f64,
    pub n: usize,
}

impl From<&ClassificationMetrics<i64>> for ClassificationMetricsFFI {
    fn from(r: &ClassificationMetrics<i64>) -> Self {
        Self {
            accuracy: r.accuracy,
            precision: r.precision,
            recall: r.recall,
            f1: r.f1,
            specificity: r.specificity,
            balanced_accuracy: r.balanced_accuracy,
            mcc: r.mcc,
            cohen_kappa: r.cohen_kappa,
            n: r.n,
        }
    }
}

/// Result of `anofox_regression_metrics`. MAPE and sMAPE are fractions; r2
/// and explained_variance are NaN for a single row.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RegressionMetricsFFI {
    pub mae: f64,
    pub mse: f64,
    pub rmse: f64,
    pub mape: f64,
    pub smape: f64,
    pub median_ae: f64,
    pub r2: f64,
    pub explained_variance: f64,
    pub max_error: f64,
    pub n: usize,
}

impl From<&RegressionMetrics> for RegressionMetricsFFI {
    fn from(r: &RegressionMetrics) -> Self {
        Self {
            mae: r.mae,
            mse: r.mse,
            rmse: r.rmse,
            mape: r.mape,
            smape: r.smape,
            median_ae: r.median_ae,
            r2: r.r2,
            explained_variance: r.explained_variance,
            max_error: r.max_error,
            n: r.n,
        }
    }
}

/// Options of `anofox_ranking_metrics`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RankingOptionsFFI {
    /// Relevance threshold of a relevant item; NaN for upstream's default
    /// (relevant when `> 0`, graded NDCG gain).
    pub min_relevance: f64,
    /// Cut-off, at least 1.
    pub k: usize,
    pub include_no_feedback: bool,
}

/// Result of `anofox_ranking_metrics`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RankingMetricsFFI {
    pub ndcg: f64,
    pub map: f64,
    pub mrr: f64,
    pub precision: f64,
    pub recall: f64,
    pub hit_rate: f64,
    pub n_queries: usize,
}

impl From<&RankingMetrics> for RankingMetricsFFI {
    fn from(r: &RankingMetrics) -> Self {
        Self {
            ndcg: r.ndcg,
            map: r.map,
            mrr: r.mrr,
            precision: r.precision,
            recall: r.recall,
            hit_rate: r.hit_rate,
            n_queries: r.n_queries,
        }
    }
}

unsafe fn invalid_input(out_error: *mut AnofoxError, msg: &str) {
    if !out_error.is_null() {
        (*out_error).set(ErrorCode::InvalidInput, msg);
    }
}

/// Classification metrics of `len` predicted against true label codes. The
/// codes are `0..n_labels` in label order, each occurring in `y_true` or
/// `y_pred`. `average`: 0 binary (of the class coded `pos_label`; a code
/// outside the data means the positive class never occurs), 1 macro,
/// 2 micro, 3 weighted. Writes the `n_labels * n_labels` confusion counts to
/// `out_counts`, row-major with true labels as rows (scikit-learn's layout).
///
/// # Safety
/// - `y_true` and `y_pred` must point to `len` codes
/// - `out_counts` must point to `n_labels * n_labels` values
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_classification_metrics(
    y_true: *const i64,
    y_pred: *const i64,
    len: usize,
    average: i32,
    pos_label: i64,
    zero_division: f64,
    n_labels: usize,
    out_counts: *mut u64,
    out_result: *mut ClassificationMetricsFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        if out_counts.is_null() || (len > 0 && (y_true.is_null() || y_pred.is_null())) {
            invalid_input(out_error, "null pointer argument");
            return false;
        }
        let average = match average {
            0 => Average::Binary(pos_label),
            1 => Average::Macro,
            2 => Average::Micro,
            3 => Average::Weighted,
            _ => {
                invalid_input(out_error, "invalid average");
                return false;
            }
        };
        let (t, p): (&[i64], &[i64]) = if len == 0 {
            (&[], &[])
        } else {
            (
                slice::from_raw_parts(y_true, len),
                slice::from_raw_parts(y_pred, len),
            )
        };
        match classification_metrics(t, p, average, zero_division) {
            Ok(r) => {
                let labels_ok = r.confusion.labels.len() == n_labels
                    && r.confusion
                        .labels
                        .iter()
                        .enumerate()
                        .all(|(i, &l)| l == i as i64);
                if !labels_ok {
                    invalid_input(
                        out_error,
                        &format!(
                            "expected the label codes 0..{}, found {} labels",
                            n_labels,
                            r.confusion.labels.len()
                        ),
                    );
                    return false;
                }
                let out = slice::from_raw_parts_mut(out_counts, n_labels * n_labels);
                for (i, row) in r.confusion.counts.iter().enumerate() {
                    out[i * n_labels..(i + 1) * n_labels].copy_from_slice(row);
                }
                *out_result = ClassificationMetricsFFI::from(&r);
                true
            }
            Err(e) => {
                set_error(out_error, &e);
                false
            }
        }
    })
}

/// Regression metrics of `y_pred` against `y_true` (equal lengths; a NULL or
/// non-finite value is a degenerate input).
///
/// # Safety
/// - `y_true` and `y_pred` must be valid DataArrays
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_regression_metrics(
    y_true: DataArray,
    y_pred: DataArray,
    out_result: *mut RegressionMetricsFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        match regression_metrics(&y_true.to_vec(), &y_pred.to_vec()) {
            Ok(r) => {
                *out_result = RegressionMetricsFFI::from(&r);
                true
            }
            Err(e) => {
                set_error(out_error, &e);
                false
            }
        }
    })
}

/// Ranking metrics at `options.k` of items grouped by the query codes in
/// `query` (one per item, `relevance.len` items), ranked within a query by
/// decreasing `score`, tied scores in input order.
///
/// # Safety
/// - `query` must point to `relevance.len` codes
/// - `relevance` and `score` must be valid DataArrays of equal length
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_ranking_metrics(
    query: *const i64,
    relevance: DataArray,
    score: DataArray,
    options: RankingOptionsFFI,
    out_result: *mut RankingMetricsFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let len = relevance.len;
        if len > 0 && query.is_null() {
            invalid_input(out_error, "null pointer argument");
            return false;
        }
        let q: &[i64] = if len == 0 {
            &[]
        } else {
            slice::from_raw_parts(query, len)
        };
        let opts = RankingOptions {
            k: options.k,
            min_relevance: if options.min_relevance.is_nan() {
                None
            } else {
                Some(options.min_relevance)
            },
            include_no_feedback: options.include_no_feedback,
        };
        match ranking_metrics(q, &relevance.to_vec(), &score.to_vec(), &opts) {
            Ok(r) => {
                *out_result = RankingMetricsFFI::from(&r);
                true
            }
            Err(e) => {
                set_error(out_error, &e);
                false
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(v: &[f64]) -> DataArray {
        DataArray {
            data: v.as_ptr(),
            validity: std::ptr::null(),
            len: v.len(),
        }
    }

    #[test]
    fn classification_counts_and_errors() {
        let t = [1i64, 1, 0, 0, 1];
        let p = [1i64, 0, 0, 1, 1];
        let mut counts = [0u64; 4];
        let mut r = std::mem::MaybeUninit::<ClassificationMetricsFFI>::zeroed();
        let mut e = AnofoxError::success();
        unsafe {
            assert!(anofox_classification_metrics(
                t.as_ptr(),
                p.as_ptr(),
                5,
                0,
                1,
                0.0,
                2,
                counts.as_mut_ptr(),
                r.as_mut_ptr(),
                &mut e
            ));
            let r = r.assume_init();
            assert!((r.precision - 2.0 / 3.0).abs() < 1e-12);
            assert_eq!(counts, [1, 1, 1, 2]);
            // Wrong label count announced by the caller.
            let mut r = std::mem::MaybeUninit::<ClassificationMetricsFFI>::zeroed();
            let mut big = [0u64; 9];
            assert!(!anofox_classification_metrics(
                t.as_ptr(),
                p.as_ptr(),
                5,
                1,
                1,
                0.0,
                3,
                big.as_mut_ptr(),
                r.as_mut_ptr(),
                &mut e
            ));
            assert_eq!(e.code, ErrorCode::InvalidInput);
        }
    }

    #[test]
    fn regression_and_ranking() {
        let mut r = std::mem::MaybeUninit::<RegressionMetricsFFI>::zeroed();
        let mut e = AnofoxError::success();
        unsafe {
            assert!(anofox_regression_metrics(
                arr(&[3.0, -0.5, 2.0, 7.0]),
                arr(&[2.5, 0.0, 2.0, 8.0]),
                r.as_mut_ptr(),
                &mut e
            ));
            assert!((r.assume_init().mae - 0.5).abs() < 1e-12);
            let q = [1i64, 1, 1, 2, 2, 2];
            let rel = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
            let s = [0.9, 0.8, 0.1, 0.7, 0.5, 0.2];
            let opts = RankingOptionsFFI {
                min_relevance: f64::NAN,
                k: 2,
                include_no_feedback: false,
            };
            let mut m = std::mem::MaybeUninit::<RankingMetricsFFI>::zeroed();
            assert!(anofox_ranking_metrics(
                q.as_ptr(),
                arr(&rel),
                arr(&s),
                opts,
                m.as_mut_ptr(),
                &mut e
            ));
            let m = m.assume_init();
            assert_eq!(m.n_queries, 2);
            assert!((m.hit_rate - 0.5).abs() < 1e-12);
        }
    }
}
