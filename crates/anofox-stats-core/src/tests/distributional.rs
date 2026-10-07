//! Distributional tests
//!
//! - Shapiro-Wilk test (normality)
//! - D'Agostino K-squared test (normality)

use super::{convert_error, filter_nan, TestResult};
use crate::{StatsError, StatsResult};
use anofox_tests::{dagostino_k_squared as lib_dagostino_k_squared, Alternative};
use statrs::distribution::{ContinuousCDF, Normal};

/// Shapiro-Wilk test for normality
///
/// Tests whether a sample comes from a normal distribution.
/// Valid for sample sizes between 3 and 5000.
pub fn shapiro_wilk(data: &[f64]) -> StatsResult<TestResult> {
    let filtered = filter_nan(data);

    if filtered.len() < 3 {
        return Err(StatsError::InsufficientDataMsg(
            "Shapiro-Wilk test requires at least 3 observations".into(),
        ));
    }
    if filtered.len() > 5000 {
        return Err(StatsError::InvalidInput(
            "Shapiro-Wilk test is limited to n <= 5000".into(),
        ));
    }

    // Computed here (a port of R's swilk.c) rather than via anofox-statistics
    // <= 0.4.2, whose small-sample (n <= 11) p-value approximation is wrong.
    let mut sorted = filtered.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    let (statistic, p_value) = if sorted[n - 1] - sorted[0] < 1e-10
        || !sorted[n - 1].is_finite()
        || !sorted[0].is_finite()
    {
        // Constant (or non-finite) data: R stops with an error; keep the
        // previous contract (W = 1, p = 1) for constant data, NaN otherwise.
        if sorted[0].is_finite() && sorted[n - 1].is_finite() {
            (1.0, 1.0)
        } else {
            (f64::NAN, f64::NAN)
        }
    } else {
        swilk(&sorted)
    };

    Ok(TestResult {
        statistic,
        p_value,
        df: f64::NAN,
        effect_size: f64::NAN,
        ci_lower: f64::NAN,
        ci_upper: f64::NAN,
        confidence_level: f64::NAN,
        n: filtered.len(),
        n1: 0,
        n2: 0,
        alternative: Alternative::TwoSided,
        method: "Shapiro-Wilk test".into(),
    })
}

/// Implementation of the SWILK algorithm (Royston 1995, AS R94), ported
/// line-for-line from R's `src/library/stats/src/swilk.c` so that both the
/// W statistic and the p-value (including the small-sample branch for
/// 4 <= n <= 11) agree with `shapiro.test()` / `scipy.stats.shapiro`.
///
/// `x` must be sorted ascending and have non-zero range.
fn swilk(x: &[f64]) -> (f64, f64) {
    let n = x.len();
    let an = n as f64;
    let nn2 = n / 2;
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Polynomial coefficients (ascending powers), as in swilk.c
    const G: [f64; 2] = [-2.273, 0.459];
    const C1: [f64; 6] = [0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056];
    const C2: [f64; 6] = [0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633];
    const C3: [f64; 4] = [0.544, -0.39978, 0.025054, -6.714e-4];
    const C4: [f64; 4] = [1.3822, -0.77857, 0.062767, -0.0020322];
    const C5: [f64; 4] = [-1.5861, -0.31082, -0.083751, 0.0038915];
    const C6: [f64; 3] = [-0.4803, -0.082676, 0.0030302];

    // a is 1-based like the C code: a[1..=nn2]
    let mut a = vec![0.0; nn2 + 1];
    if n == 3 {
        a[1] = std::f64::consts::FRAC_1_SQRT_2;
    } else {
        let an25 = an + 0.25;
        let mut summ2 = 0.0;
        for (i, ai) in a.iter_mut().enumerate().skip(1) {
            *ai = normal.inverse_cdf((i as f64 - 0.375) / an25);
            summ2 += *ai * *ai;
        }
        summ2 *= 2.0;
        let ssumm2 = summ2.sqrt();
        let rsn = 1.0 / an.sqrt();
        let a1 = poly(&C1, rsn) - a[1] / ssumm2;

        let (i1, fac) = if n > 5 {
            let a2 = -a[2] / ssumm2 + poly(&C2, rsn);
            let fac = ((summ2 - 2.0 * a[1] * a[1] - 2.0 * a[2] * a[2])
                / (1.0 - 2.0 * a1 * a1 - 2.0 * a2 * a2))
                .sqrt();
            a[2] = a2;
            (3, fac)
        } else {
            let fac = ((summ2 - 2.0 * a[1] * a[1]) / (1.0 - 2.0 * a1 * a1)).sqrt();
            (2, fac)
        };
        a[1] = a1;
        for ai in a.iter_mut().take(nn2 + 1).skip(i1) {
            *ai /= -fac;
        }
    }

    let range = x[n - 1] - x[0];

    // Full antisymmetric coefficient vector: coef(i) = sign(i - j) * a[1 + min(i, j)],
    // j = n - 1 - i (0 for the middle element of odd n).
    let coef = |i: usize| -> f64 {
        let j = n - 1 - i;
        match i.cmp(&j) {
            std::cmp::Ordering::Less => -a[1 + i],
            std::cmp::Ordering::Greater => a[1 + j],
            std::cmp::Ordering::Equal => 0.0,
        }
    };

    // W as squared correlation between data (range-scaled) and coefficients
    let sa: f64 = (0..n).map(coef).sum::<f64>() / an;
    let sx: f64 = x.iter().map(|xi| xi / range).sum::<f64>() / an;
    let (mut ssa, mut ssx, mut sax) = (0.0, 0.0, 0.0);
    for (i, xi) in x.iter().enumerate() {
        let asa = coef(i) - sa;
        let xsx = xi / range - sx;
        ssa += asa * asa;
        ssx += xsx * xsx;
        sax += asa * xsx;
    }

    // w1 = 1 - W, computed to avoid rounding error for W near 1
    let ssassx = (ssa * ssx).sqrt();
    let w1 = (ssassx - sax) * (ssassx + sax) / (ssa * ssx);
    let w = 1.0 - w1;

    // Significance level for W
    if n == 3 {
        // 6/pi and asin(sqrt(3/4)) = pi/3
        let pi6 = 6.0 / std::f64::consts::PI;
        let stqr = std::f64::consts::FRAC_PI_3;
        let pw = pi6 * (w.sqrt().asin() - stqr);
        return (w, pw.clamp(0.0, 1.0));
    }

    let mut y = w1.ln();
    let xx = an.ln();
    let (m, s) = if n <= 11 {
        let gamma = poly(&G, an);
        if y >= gamma {
            return (w, 1e-99);
        }
        y = -(gamma - y).ln();
        (poly(&C3, an), poly(&C4, an).exp())
    } else {
        (poly(&C5, xx), poly(&C6, xx).exp())
    };

    let z = (y - m) / s;
    (w, normal.sf(z).clamp(0.0, 1.0))
}

/// Evaluate a polynomial with ascending coefficients: c[0] + c[1]*x + c[2]*x^2 + ...
fn poly(c: &[f64], x: f64) -> f64 {
    c.iter().rev().fold(0.0, |acc, &ci| acc * x + ci)
}

/// D'Agostino-Pearson K-squared test for normality
///
/// Omnibus test combining skewness and kurtosis tests.
pub fn dagostino_k_squared(data: &[f64]) -> StatsResult<TestResult> {
    let filtered = filter_nan(data);

    if filtered.len() < 8 {
        return Err(StatsError::InsufficientDataMsg(
            "D'Agostino K-squared test requires at least 8 observations".into(),
        ));
    }

    let result = lib_dagostino_k_squared(&filtered).map_err(convert_error)?;

    Ok(TestResult {
        statistic: result.statistic,
        p_value: result.p_value,
        df: 2.0, // Chi-squared with 2 df
        effect_size: f64::NAN,
        ci_lower: f64::NAN,
        ci_upper: f64::NAN,
        confidence_level: f64::NAN,
        n: filtered.len(),
        n1: 0,
        n2: 0,
        alternative: Alternative::TwoSided,
        method: "D'Agostino K-squared test".into(),
    })
}

/// Extended result for D'Agostino test including skewness and kurtosis
#[derive(Debug, Clone)]
pub struct DAgostinoResult {
    /// K-squared statistic
    pub statistic: f64,
    /// p-value
    pub p_value: f64,
    /// Skewness z-score
    pub z_skewness: f64,
    /// Kurtosis z-score
    pub z_kurtosis: f64,
    /// Sample size
    pub n: usize,
}

/// D'Agostino K-squared test with detailed results
pub fn dagostino_k_squared_detailed(data: &[f64]) -> StatsResult<DAgostinoResult> {
    let filtered = filter_nan(data);

    if filtered.len() < 8 {
        return Err(StatsError::InsufficientDataMsg(
            "D'Agostino K-squared test requires at least 8 observations".into(),
        ));
    }

    let result = lib_dagostino_k_squared(&filtered).map_err(convert_error)?;

    Ok(DAgostinoResult {
        statistic: result.statistic,
        p_value: result.p_value,
        z_skewness: result.z_skewness,
        z_kurtosis: result.z_kurtosis,
        n: filtered.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shapiro_wilk_normal() {
        let data = vec![
            -0.5, 0.1, -0.3, 0.8, 0.2, -0.1, 0.4, -0.2, 0.3, 0.0, -0.4, 0.5, 0.1, -0.6, 0.2, -0.1,
            0.3, -0.3, 0.4, 0.0,
        ];
        let result = shapiro_wilk(&data).unwrap();

        assert!(result.statistic > 0.9);
        assert!(result.p_value > 0.05);
    }

    #[test]
    fn test_dagostino_k_squared() {
        let data = vec![
            -0.5, 0.1, -0.3, 0.8, 0.2, -0.1, 0.4, -0.2, 0.3, 0.0, -0.4, 0.5, 0.1, -0.6, 0.2, -0.1,
            0.3, -0.3, 0.4, 0.0,
        ];
        let result = dagostino_k_squared(&data).unwrap();

        assert!(result.statistic >= 0.0);
        assert!(result.p_value > 0.0 && result.p_value <= 1.0);
    }

    /// R shapiro.test(); small-sample branch (n <= 11) and n >= 12.
    #[test]
    #[allow(clippy::excessive_precision)]
    fn test_shapiro_wilk_matches_r() {
        let r = shapiro_wilk(&[4.1, 5.3, 3.8, 6.9, 5.0, 4.4, 9.2, 5.7, 4.9, 6.1]).unwrap();
        assert!((r.statistic - 0.88790406283629908).abs() < 1e-10);
        assert!((r.p_value - 0.16058545199963783).abs() < 1e-10);
        let r = shapiro_wilk(&[1.2, 3.4, 2.2, 5.9]).unwrap();
        assert!((r.p_value - 0.74255791273476768).abs() < 1e-10);
        let r = shapiro_wilk(&[1.0, 2.0, 4.0]).unwrap();
        assert!((r.p_value - 0.6368868450289632).abs() < 1e-10);
    }
}
