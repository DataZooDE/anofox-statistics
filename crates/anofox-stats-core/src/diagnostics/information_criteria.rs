//! Information criteria for model selection (AIC, BIC)

use crate::errors::{StatsError, StatsResult};

/// Compute AIC (Akaike Information Criterion)
///
/// AIC = n * ln(RSS/n) + 2k
///
/// where:
/// - n = number of observations
/// - RSS = residual sum of squares
/// - k = number of parameters (including intercept)
///
/// Lower AIC indicates better model fit (accounting for complexity).
pub fn compute_aic(rss: f64, n: usize, k: usize) -> StatsResult<f64> {
    if n == 0 {
        return Err(StatsError::InvalidInput("n must be > 0".into()));
    }
    if rss < 0.0 {
        return Err(StatsError::InvalidInput("RSS must be non-negative".into()));
    }

    let n_f = n as f64;
    let k_f = k as f64;

    // Handle edge case where RSS = 0 (perfect fit)
    if rss == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }

    let aic = n_f * (rss / n_f).ln() + 2.0 * k_f;
    Ok(aic)
}

/// Compute AICc (corrected AIC for small samples)
///
/// AICc = AIC + (2k² + 2k) / (n - k - 1)
///
/// Use AICc when n/k < 40
#[allow(dead_code)]
pub fn compute_aicc(rss: f64, n: usize, k: usize) -> StatsResult<f64> {
    let aic = compute_aic(rss, n, k)?;

    if n <= k + 1 {
        return Err(StatsError::InsufficientDataMsg(
            "Need n > k + 1 for AICc".into(),
        ));
    }

    let n_f = n as f64;
    let k_f = k as f64;

    let correction = (2.0 * k_f * k_f + 2.0 * k_f) / (n_f - k_f - 1.0);
    Ok(aic + correction)
}

/// Compute BIC (Bayesian Information Criterion)
///
/// BIC = n * ln(RSS/n) + k * ln(n)
///
/// where:
/// - n = number of observations
/// - RSS = residual sum of squares
/// - k = number of parameters (including intercept)
///
/// BIC penalizes model complexity more heavily than AIC for larger samples.
pub fn compute_bic(rss: f64, n: usize, k: usize) -> StatsResult<f64> {
    if n == 0 {
        return Err(StatsError::InvalidInput("n must be > 0".into()));
    }
    if rss < 0.0 {
        return Err(StatsError::InvalidInput("RSS must be non-negative".into()));
    }

    let n_f = n as f64;
    let k_f = k as f64;

    // Handle edge case where RSS = 0 (perfect fit)
    if rss == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }

    let bic = n_f * (rss / n_f).ln() + k_f * n_f.ln();
    Ok(bic)
}

/// Compute both AIC and BIC
pub fn compute_aic_bic(rss: f64, n: usize, k: usize) -> StatsResult<(f64, f64)> {
    Ok((compute_aic(rss, n, k)?, compute_bic(rss, n, k)?))
}

/// Compute RSS from residual standard error and degrees of freedom
///
/// RSS = (residual_std_error)² * df
/// where df = n - k (degrees of freedom)
#[allow(dead_code)]
pub fn rss_from_std_error(residual_std_error: f64, n: usize, k: usize) -> f64 {
    if n <= k {
        return 0.0;
    }
    let df = (n - k) as f64;
    residual_std_error * residual_std_error * df
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aic_basic() {
        // Example: RSS=10, n=100, k=3
        let aic = compute_aic(10.0, 100, 3).unwrap();
        // AIC = 100 * ln(10/100) + 2*3 = 100 * ln(0.1) + 6 ≈ -224.26
        assert!((aic - (-224.2585)).abs() < 0.01);
    }

    #[test]
    fn test_bic_basic() {
        // Example: RSS=10, n=100, k=3
        let bic = compute_bic(10.0, 100, 3).unwrap();
        // BIC = 100 * ln(10/100) + 3 * ln(100) ≈ -216.44
        assert!((bic - (-216.4430)).abs() < 0.01);
    }

    #[test]
    fn test_bic_penalizes_more_for_large_n() {
        // For same RSS and k, BIC penalty grows with ln(n)
        let rss = 10.0;
        let k = 3;

        let _aic_100 = compute_aic(rss, 100, k).unwrap();
        let _bic_100 = compute_bic(rss, 100, k).unwrap();
        let _aic_1000 = compute_aic(rss, 1000, k).unwrap();
        let _bic_1000 = compute_bic(rss, 1000, k).unwrap();

        // AIC penalty doesn't change with n (only 2k)
        let aic_penalty_100 = 2.0 * k as f64;
        let aic_penalty_1000 = 2.0 * k as f64;
        assert_eq!(aic_penalty_100, aic_penalty_1000);

        // BIC penalty increases with ln(n)
        let bic_penalty_100 = k as f64 * (100.0_f64).ln();
        let bic_penalty_1000 = k as f64 * (1000.0_f64).ln();
        assert!(bic_penalty_1000 > bic_penalty_100);
    }

    #[test]
    fn test_aicc_small_sample_correction() {
        let rss = 10.0;
        let n = 20;
        let k = 5;

        let aic = compute_aic(rss, n, k).unwrap();
        let aicc = compute_aicc(rss, n, k).unwrap();

        // AICc should be larger (more conservative) for small samples
        assert!(aicc > aic);
    }

    #[test]
    fn test_rss_from_std_error() {
        // If residual_std_error = 2.0, n = 10, k = 2
        // df = 10 - 2 = 8
        // RSS = 2.0² * 8 = 32
        let rss = rss_from_std_error(2.0, 10, 2);
        assert!((rss - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_perfect_fit() {
        let aic = compute_aic(0.0, 100, 3).unwrap();
        let bic = compute_bic(0.0, 100, 3).unwrap();
        assert!(aic.is_infinite() && aic < 0.0);
        assert!(bic.is_infinite() && bic < 0.0);
    }

    #[test]
    fn test_invalid_inputs() {
        assert!(compute_aic(-1.0, 100, 3).is_err());
        assert!(compute_aic(10.0, 0, 3).is_err());
    }
}
