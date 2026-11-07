//! Performance metrics of labelled predictions (#368, #369): upstream
//! anofox-statistics 0.10 `metrics`.
//!
//! - [`classification_metrics`]: accuracy, precision / recall / F1 /
//!   specificity (binary, macro, micro or weighted), balanced accuracy, MCC,
//!   Cohen's kappa and the confusion matrix (scikit-learn, R yardstick)
//! - [`regression_metrics`]: MAE, MSE, RMSE, MAPE, sMAPE, median AE, R²,
//!   explained variance, max error (scikit-learn)
//! - [`ranking_metrics`]: NDCG@k (scikit-learn `ndcg_score`), MAP@k, MRR@k,
//!   precision@k, recall@k and hit rate@k (evidently), averaged over queries
//!
//! Labels and query ids reach this module as dense integer codes whose order
//! is the label order (the SQL layer codes VARCHAR, BIGINT and BOOLEAN labels).

use super::convert_error;
use crate::{StatsError, StatsResult};
use anofox_tests::metrics::ClassificationOptions;
pub use anofox_tests::metrics::{
    Average, ClassificationMetrics, RankingMetrics, RankingOptions, RegressionMetrics,
};

/// Classification metrics of the predicted against the true label codes
/// (upstream `classification_metrics`). `zero_division` is the value of a
/// ratio with a zero denominator.
pub fn classification_metrics(
    y_true: &[i64],
    y_pred: &[i64],
    average: Average<i64>,
    zero_division: f64,
) -> StatsResult<ClassificationMetrics<i64>> {
    let options = ClassificationOptions {
        average,
        zero_division,
    };
    anofox_tests::metrics::classification_metrics(y_true, y_pred, &options).map_err(convert_error)
}

/// Regression metrics of `y_pred` against `y_true` (upstream
/// `regression_metrics`).
pub fn regression_metrics(y_true: &[f64], y_pred: &[f64]) -> StatsResult<RegressionMetrics> {
    anofox_tests::metrics::regression_metrics(y_true, y_pred).map_err(convert_error)
}

/// Ranking metrics over the queries coded in `query` (upstream
/// `ranking_metrics`). A table without any relevant item is degenerate data
/// (no query to average over), not an invalid argument.
pub fn ranking_metrics(
    query: &[i64],
    relevance: &[f64],
    score: &[f64],
    options: &RankingOptions,
) -> StatsResult<RankingMetrics> {
    anofox_tests::metrics::ranking_metrics(query, relevance, score, options).map_err(|e| match e {
        anofox_tests::StatError::InvalidParameter(ref msg)
            if msg.contains("no query has a relevant item") =>
        {
            StatsError::InsufficientDataMsg(msg.clone())
        }
        other => convert_error(other),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_binary_and_multiclass() {
        // sklearn: precision_score([1,1,0,0,1], [1,0,0,1,1]) = 2/3
        let m = classification_metrics(&[1, 1, 0, 0, 1], &[1, 0, 0, 1, 1], Average::Binary(1), 0.0)
            .unwrap();
        assert!((m.precision - 2.0 / 3.0).abs() < 1e-12);
        assert!((m.specificity - 0.5).abs() < 1e-12);
        assert_eq!(m.confusion.counts, vec![vec![1, 1], vec![1, 2]]);
        assert!(classification_metrics(&[0, 1, 2], &[0, 1, 2], Average::Binary(1), 0.0).is_err());
        let m = classification_metrics(&[0, 1, 2, 2], &[0, 2, 2, 1], Average::Micro, 0.0).unwrap();
        assert!((m.accuracy - 0.5).abs() < 1e-12);
        assert!((m.f1 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn regression_and_ranking() {
        let m = regression_metrics(&[3.0, -0.5, 2.0, 7.0], &[2.5, 0.0, 2.0, 8.0]).unwrap();
        assert!((m.r2 - 0.9486081370449679).abs() < 1e-12);
        let opts = RankingOptions {
            k: 2,
            ..Default::default()
        };
        let r = ranking_metrics(
            &[1, 1, 1, 2, 2, 2],
            &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            &[0.9, 0.8, 0.1, 0.7, 0.5, 0.2],
            &opts,
        )
        .unwrap();
        assert!((r.mrr - 0.5).abs() < 1e-12);
        // No relevant item anywhere: degenerate, not an input error.
        let e = ranking_metrics(&[1, 1], &[0.0, 0.0], &[0.2, 0.1], &opts).unwrap_err();
        assert!(matches!(e, StatsError::InsufficientDataMsg(_)));
    }
}
