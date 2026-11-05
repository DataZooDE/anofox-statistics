//! Null distributions exposed as SQL scalars (#304): the ROC AUC of a random
//! classifier (`anofox_tests::auc_null_quantile` / `auc_null_p_value`, exact
//! from the Mann-Whitney distribution) and the studentized range
//! distribution (`anofox_tests::ptukey` / `qtukey`, ports of R's).
//!
//! Thin delegation: the wrappers only map errors. The upstream Tukey
//! functions signal invalid parameters with NaN (as R does); a NaN result
//! from non-NaN arguments is reported here as an invalid-input error, while
//! a NaN argument propagates as NaN.

use super::convert_error;
use crate::{StatsError, StatsResult};

/// Smallest AUC `a` with `P(AUC_random <= a) >= prob`
/// (R `qwilcox(prob, n_pos, n_neg) / (n_pos * n_neg)`).
pub fn auc_null_quantile(prob: f64, n_pos: usize, n_neg: usize) -> StatsResult<f64> {
    anofox_tests::auc_null_quantile(n_pos, n_neg, prob).map_err(convert_error)
}

/// One-sided p-value `P(AUC_random >= auc)`
/// (R `pwilcox(ceiling(u) - 1, n_pos, n_neg, lower.tail = FALSE)`, `u = auc n_pos n_neg`).
pub fn auc_null_p_value(auc: f64, n_pos: usize, n_neg: usize) -> StatsResult<f64> {
    anofox_tests::auc_null_p_value(auc, n_pos, n_neg).map_err(convert_error)
}

fn tukey_result(value: f64, args: &[f64], what: &str) -> StatsResult<f64> {
    if value.is_nan() && !args.iter().any(|a| a.is_nan()) {
        return Err(StatsError::InvalidInput(format!(
            "{} requires n_means >= 2, df >= 2 and n_ranges >= 1",
            what
        )));
    }
    Ok(value)
}

/// Studentized range distribution function (R `ptukey(q, nmeans, df, nranges, lower.tail)`).
pub fn ptukey(q: f64, n_means: f64, df: f64, n_ranges: f64, lower_tail: bool) -> StatsResult<f64> {
    let v = anofox_tests::ptukey(q, n_means, df, n_ranges, lower_tail);
    tukey_result(v, &[q, n_means, df, n_ranges], "ptukey")
}

/// Studentized range quantile function (R `qtukey(p, nmeans, df, nranges, lower.tail)`).
pub fn qtukey(p: f64, n_means: f64, df: f64, n_ranges: f64, lower_tail: bool) -> StatsResult<f64> {
    if !p.is_nan() && !(0.0..=1.0).contains(&p) {
        return Err(StatsError::InvalidInput(format!(
            "qtukey: p must be in [0, 1], got {}",
            p
        )));
    }
    let v = anofox_tests::qtukey(p, n_means, df, n_ranges, lower_tail);
    tukey_result(v, &[p, n_means, df, n_ranges], "qtukey")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auc_null_delegates() {
        // qwilcox(0.95, 10, 10) = 72
        assert_eq!(auc_null_quantile(0.95, 10, 10).unwrap(), 0.72);
        assert!((auc_null_p_value(1.0, 3, 4).unwrap() - 1.0 / 35.0).abs() < 1e-15);
        assert!(matches!(
            auc_null_quantile(1.5, 3, 4),
            Err(StatsError::InvalidInput(_))
        ));
        assert!(matches!(
            auc_null_p_value(0.5, 0, 4),
            Err(StatsError::InvalidInput(_))
        ));
    }

    #[test]
    fn tukey_delegates() {
        // R: ptukey(3.5, 3, 12) = 0.930004514724551; qtukey(0.95, 3, 12) = 3.77292895940835
        let p = ptukey(3.5, 3.0, 12.0, 1.0, true).unwrap();
        assert!((p - 0.930004514724551).abs() < 1e-9);
        let q = qtukey(0.95, 3.0, 12.0, 1.0, true).unwrap();
        assert!((q - 3.77292895940835).abs() < 1e-7);
        assert!(ptukey(f64::NAN, 3.0, 12.0, 1.0, true).unwrap().is_nan());
        assert!(matches!(
            ptukey(3.5, 1.0, 12.0, 1.0, true),
            Err(StatsError::InvalidInput(_))
        ));
        assert!(matches!(
            qtukey(1.5, 3.0, 12.0, 1.0, true),
            Err(StatsError::InvalidInput(_))
        ));
    }
}
