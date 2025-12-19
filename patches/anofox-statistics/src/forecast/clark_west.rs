use crate::error::{Result, StatError};
use statrs::distribution::{ContinuousCDF, Normal};

/// Result of Clark-West test for nested model comparison
#[derive(Debug, Clone)]
pub struct CWResult {
    /// The Clark-West adjusted test statistic
    pub statistic: f64,
    /// The p-value (one-sided, testing H0: restricted model is adequate)
    pub p_value: f64,
    /// The p-value (two-sided)
    pub p_value_two_sided: f64,
}

/// Perform the Clark-West test for comparing forecasts from nested models.
///
/// Tests the null hypothesis that the restricted (parsimonious) model has
/// equal predictive accuracy to the unrestricted model. This adjusts the
/// standard Diebold-Mariano test for the bias that occurs when comparing
/// nested models under the null hypothesis.
///
/// # Arguments
/// * `e1` - Forecast errors from the restricted (null) model
/// * `e2` - Forecast errors from the unrestricted (alternative) model
/// * `h` - Forecast horizon (for HAC variance adjustment, h=1 for 1-step ahead)
///
/// # Returns
/// * `CWResult` containing test statistic and p-values
///
/// # Note
/// The one-sided p-value tests H0: restricted model is adequate vs
/// H1: unrestricted model has superior predictive ability.
/// Under the null, the unrestricted model's extra parameters are not
/// identified, causing asymptotic bias in the standard DM test.
///
/// # References
/// * Clark, T.E. and West, K.D. (2006) "Using Out-of-Sample Mean Squared
///   Prediction Errors to Test the Martingale Difference Hypothesis"
/// * Clark, T.E. and West, K.D. (2007) "Approximately Normal Tests for
///   Equal Predictive Accuracy in Nested Models"
pub fn clark_west(e1: &[f64], e2: &[f64], h: usize) -> Result<CWResult> {
    let n = e1.len();

    if n == 0 {
        return Err(StatError::EmptyData);
    }

    if e1.len() != e2.len() {
        return Err(StatError::InvalidParameter(
            "Forecast error vectors must have equal length".to_string(),
        ));
    }

    if n < 3 {
        return Err(StatError::InsufficientData { needed: 3, got: n });
    }

    // Compute Clark-West adjusted loss differential:
    // d_t = e1_t^2 - e2_t^2 + (e1_t - e2_t)^2
    // The adjustment term (e1_t - e2_t)^2 corrects for the bias that occurs
    // when the nested (unrestricted) model estimates parameters that are
    // zero under the null hypothesis.
    let d: Vec<f64> = e1
        .iter()
        .zip(e2.iter())
        .map(|(e1_t, e2_t)| {
            let adjustment = (e1_t - e2_t).powi(2);
            e1_t.powi(2) - e2_t.powi(2) + adjustment
        })
        .collect();

    let n_f = n as f64;

    // Mean of adjusted loss differentials
    let d_bar: f64 = d.iter().sum::<f64>() / n_f;

    // Compute HAC variance (same approach as Diebold-Mariano)
    let var_d_bar = cw_variance(&d, h);

    // Compute test statistic
    let statistic = if var_d_bar > 1e-30 {
        d_bar / var_d_bar.sqrt()
    } else {
        0.0
    };

    // Compute p-values from standard normal
    let normal = Normal::new(0.0, 1.0).unwrap();

    // One-sided: P(Z > statistic) - tests if unrestricted model is better
    // Positive statistic means e1^2 > e2^2 adjusted, i.e., unrestricted is better
    let p_value = 1.0 - normal.cdf(statistic);

    // Two-sided
    let p_value_two_sided = 2.0 * (1.0 - normal.cdf(statistic.abs()));

    Ok(CWResult {
        statistic,
        p_value,
        p_value_two_sided,
    })
}

/// Compute autocovariance at a specific lag.
fn autocovariance(centered: &[f64], lag: usize) -> f64 {
    let n_f = centered.len() as f64;
    centered
        .iter()
        .skip(lag)
        .zip(centered.iter())
        .map(|(d_t, d_t_k)| d_t * d_t_k)
        .sum::<f64>()
        / n_f
}

/// Compute HAC variance of the mean for Clark-West test.
/// Uses the same approach as the Diebold-Mariano test.
fn cw_variance(d: &[f64], h: usize) -> f64 {
    let n = d.len();
    let n_f = n as f64;
    let d_bar: f64 = d.iter().sum::<f64>() / n_f;
    let d_centered: Vec<f64> = d.iter().map(|x| x - d_bar).collect();

    let max_lag = h.saturating_sub(1);
    let gamma_0 = autocovariance(&d_centered, 0);

    let lagged_sum: f64 = (1..=max_lag)
        .take_while(|&k| k < n)
        .map(|k| 2.0 * autocovariance(&d_centered, k))
        .sum();

    (gamma_0 + lagged_sum) / n_f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cw_identical_errors() {
        // When e1 == e2, the adjusted differential is:
        // d_t = e1^2 - e2^2 + (e1 - e2)^2 = 0 + 0 = 0
        let e1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let e2 = e1.clone();
        let result = clark_west(&e1, &e2, 1).unwrap();
        assert!(result.statistic.abs() < 1e-10);
    }

    #[test]
    fn test_cw_unrestricted_better() {
        // e2 has consistently lower squared errors with some variation
        let e1: Vec<f64> = (0..50)
            .map(|i| 2.0 + (i as f64 * 0.1).sin() * 0.1)
            .collect();
        let e2: Vec<f64> = (0..50)
            .map(|i| 1.0 + (i as f64 * 0.1).cos() * 0.1)
            .collect();
        let result = clark_west(&e1, &e2, 1).unwrap();
        // Positive statistic indicates unrestricted (e2) is better
        assert!(result.statistic > 0.0);
        // One-sided p-value should be small
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_cw_restricted_better() {
        // e1 (restricted) has lower squared errors with some variation
        let e1: Vec<f64> = (0..50)
            .map(|i| 1.0 + (i as f64 * 0.1).sin() * 0.1)
            .collect();
        let e2: Vec<f64> = (0..50)
            .map(|i| 2.0 + (i as f64 * 0.1).cos() * 0.1)
            .collect();
        let result = clark_west(&e1, &e2, 1).unwrap();
        // Negative statistic indicates restricted (e1) is better
        assert!(result.statistic < 0.0);
        // One-sided p-value should be large (not rejecting null)
        assert!(result.p_value > 0.5);
    }

    #[test]
    fn test_cw_empty_error() {
        let e1: Vec<f64> = vec![];
        let e2: Vec<f64> = vec![];
        assert!(clark_west(&e1, &e2, 1).is_err());
    }

    #[test]
    fn test_cw_unequal_length_error() {
        let e1 = vec![1.0, 2.0, 3.0];
        let e2 = vec![1.0, 2.0];
        assert!(clark_west(&e1, &e2, 1).is_err());
    }

    #[test]
    fn test_cw_insufficient_data_error() {
        let e1 = vec![1.0, 2.0];
        let e2 = vec![1.0, 2.0];
        assert!(clark_west(&e1, &e2, 1).is_err());
    }

    #[test]
    fn test_cw_horizon_adjustment() {
        // Test that different horizons produce different results
        let e1: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() + 1.0).collect();
        let e2: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).cos() + 0.5).collect();

        let result_h1 = clark_west(&e1, &e2, 1).unwrap();
        let result_h3 = clark_west(&e1, &e2, 3).unwrap();

        // Statistics should differ due to different variance estimates
        assert!((result_h1.statistic - result_h3.statistic).abs() > 1e-10);
    }
}
