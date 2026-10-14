//! Time-series residual diagnostics (#155)
//!
//! - Sample autocorrelation (ACF) and partial autocorrelation (PACF)
//! - Ljung-Box and Box-Pierce portmanteau tests
//! - Durbin-Watson statistic
//!
//! All delegate to `anofox_statistics::timeseries` (R `acf`, `pacf`,
//! `Box.test`, `lmtest::dwtest`). The input is the series in time order.

use super::{convert_error, UnifiedTestResult};
use crate::{StatsError, StatsResult};

pub use anofox_tests::AcfResult;

/// R's default `lag.max` for `acf` / `pacf` on a univariate series:
/// `floor(10 * log10(n))`, at most `n - 1`.
pub fn default_max_lag(n: usize) -> usize {
    if n < 2 {
        return 0;
    }
    ((10.0 * (n as f64).log10()).floor() as usize).min(n - 1)
}

fn check_lag(n: usize, lag: usize, what: &str) -> StatsResult<()> {
    if lag >= n {
        return Err(StatsError::InsufficientDataMsg(format!(
            "{} needs more than {} observations, got {}",
            what, lag, n
        )));
    }
    Ok(())
}

/// ACF (`partial = false`, lags `0..=max_lag`) or PACF (`partial = true`,
/// lags `1..=max_lag`). `max_lag = None` uses R's default. A series no longer
/// than `max_lag` is insufficient data.
pub fn autocorrelation(x: &[f64], max_lag: Option<usize>, partial: bool) -> StatsResult<AcfResult> {
    if x.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "autocorrelation requires at least 2 observations".into(),
        ));
    }
    let max_lag = max_lag.unwrap_or_else(|| default_max_lag(x.len()));
    check_lag(x.len(), max_lag, "autocorrelation")?;
    if partial {
        anofox_tests::pacf(x, max_lag).map_err(convert_error)
    } else {
        anofox_tests::acf(x, max_lag).map_err(convert_error)
    }
}

/// Ljung-Box (`box_pierce = false`) or Box-Pierce portmanteau test of the
/// first `lag` autocorrelations, `fitdf` degrees of freedom subtracted.
pub fn portmanteau(
    x: &[f64],
    lag: usize,
    fitdf: usize,
    box_pierce: bool,
) -> StatsResult<UnifiedTestResult> {
    if lag >= 1 {
        check_lag(x.len(), lag, "portmanteau test")?;
    }
    let result = if box_pierce {
        anofox_tests::box_pierce(x, lag, fitdf)
    } else {
        anofox_tests::ljung_box(x, lag, fitdf)
    }
    .map_err(convert_error)?;
    Ok(UnifiedTestResult::from(&result))
}

/// Durbin-Watson statistic of residuals in time order. Upstream returns the
/// statistic only (the exact `dwtest` p-value needs the design matrix), so
/// `p_value` is not set.
pub fn durbin_watson(residuals: &[f64]) -> StatsResult<UnifiedTestResult> {
    if residuals.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "Durbin-Watson requires at least 2 residuals".into(),
        ));
    }
    let dw = anofox_tests::durbin_watson(residuals).map_err(convert_error)?;
    let mut t = UnifiedTestResult::new("Durbin-Watson test");
    t.statistic = Some(dw);
    t.n = Some(residuals.len());
    Ok(t)
}

/// Durbin-Watson test with a p-value (R `lmtest::dwtest`): residuals in time
/// order of a regression on the columns `x`; `fit_intercept` prepends the
/// intercept column. `exact = None` is R's default (exact for n < 100).
/// Delegates to `anofox_statistics::durbin_watson_test`.
pub fn durbin_watson_test(
    residuals: &[f64],
    x: &[Vec<f64>],
    fit_intercept: bool,
    alternative: super::Alternative,
    exact: Option<bool>,
) -> StatsResult<UnifiedTestResult> {
    let ones = vec![1.0; residuals.len()];
    let mut cols: Vec<&[f64]> = Vec::with_capacity(x.len() + 1);
    if fit_intercept {
        cols.push(&ones);
    }
    cols.extend(x.iter().map(|c| c.as_slice()));
    if cols.is_empty() {
        return Err(StatsError::InvalidInput(
            "Durbin-Watson test needs at least one design column (x or the intercept)".into(),
        ));
    }
    anofox_tests::durbin_watson_test(residuals, &cols, alternative, exact).map_err(convert_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // R: x <- c(1.2, -0.4, 0.8, 2.1, -1.3, 0.5, 0.9, -0.2, 1.7, -0.8)
    const X: [f64; 10] = [1.2, -0.4, 0.8, 2.1, -1.3, 0.5, 0.9, -0.2, 1.7, -0.8];

    #[test]
    fn default_lag_matches_r() {
        assert_eq!(default_max_lag(10), 9);
        assert_eq!(default_max_lag(100), 20);
        assert_eq!(default_max_lag(1000), 30);
    }

    #[test]
    fn acf_starts_at_one_and_pacf_at_lag_one() {
        let a = autocorrelation(&X, Some(3), false).unwrap();
        assert_eq!(a.lags, vec![0, 1, 2, 3]);
        assert_relative_eq!(a.values[0], 1.0);
        let p = autocorrelation(&X, Some(3), true).unwrap();
        assert_eq!(p.lags, vec![1, 2, 3]);
        assert_relative_eq!(p.values[0], a.values[1], epsilon = 1e-12);
    }

    #[test]
    fn short_series_is_insufficient_data() {
        assert!(matches!(
            autocorrelation(&X, Some(10), false),
            Err(StatsError::InsufficientDataMsg(_))
        ));
        assert!(matches!(
            portmanteau(&X[..3], 5, 0, false),
            Err(StatsError::InsufficientDataMsg(_))
        ));
    }

    #[test]
    fn durbin_watson_test_matches_dwtest() {
        // Sanity check only; the R-validated values are in the SQL reference test.
        let t: Vec<f64> = (1..=10).map(f64::from).collect();
        let r =
            durbin_watson_test(&X, &[t], true, super::super::Alternative::Greater, None).unwrap();
        assert_eq!(r.method, "Durbin-Watson test");
        assert!(r.p_value.unwrap() > 0.0 && r.p_value.unwrap() < 1.0);
    }

    #[test]
    fn durbin_watson_has_no_p_value() {
        let t = durbin_watson(&X).unwrap();
        assert!(t.statistic.unwrap() > 0.0);
        assert!(t.p_value.is_none());
        assert_eq!(t.n, Some(10));
    }
}
