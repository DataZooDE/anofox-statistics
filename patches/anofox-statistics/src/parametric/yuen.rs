use crate::error::{Result, StatError};
use crate::Alternative;
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Confidence interval for Yuen's test
#[derive(Debug, Clone)]
pub struct YuenConfInt {
    /// Lower bound of the confidence interval
    pub lower: f64,
    /// Upper bound of the confidence interval
    pub upper: f64,
    /// Confidence level (e.g., 0.95 for 95%)
    pub conf_level: f64,
}

/// Result of Yuen's test
#[derive(Debug, Clone)]
pub struct YuenResult {
    /// The t-statistic
    pub statistic: f64,
    /// Degrees of freedom
    pub df: f64,
    /// The p-value (two-sided)
    pub p_value: f64,
    /// Difference between trimmed means
    pub diff: f64,
    /// Trimmed mean of first sample
    pub trimmed_mean_x: f64,
    /// Trimmed mean of second sample
    pub trimmed_mean_y: f64,
    /// Confidence interval for the difference in trimmed means
    pub conf_int: Option<YuenConfInt>,
}

/// Perform Yuen's test for comparing trimmed means of two independent samples.
///
/// This is a robust alternative to the t-test that uses trimmed means
/// and winsorized variances, making it less sensitive to outliers.
///
/// # Arguments
/// * `x` - First sample
/// * `y` - Second sample
/// * `trim` - Proportion to trim from each tail (default 0.2)
/// * `alternative` - Alternative hypothesis (TwoSided, Less, Greater)
/// * `conf_level` - If Some, compute confidence interval at this level (e.g., 0.95)
///
/// # Returns
/// * `YuenResult` containing statistic, df, p-value, trimmed means, and optionally CI
pub fn yuen_test(
    x: &[f64],
    y: &[f64],
    trim: f64,
    alternative: Alternative,
    conf_level: Option<f64>,
) -> Result<YuenResult> {
    // Validate trim parameter
    if !(0.0..0.5).contains(&trim) {
        return Err(StatError::InvalidParameter(format!(
            "trim must be in [0, 0.5), got {}",
            trim
        )));
    }

    if x.is_empty() {
        return Err(StatError::EmptyData);
    }
    if y.is_empty() {
        return Err(StatError::EmptyData);
    }

    let nx = x.len();
    let ny = y.len();

    // Number of observations to trim from each tail
    let gx = (trim * nx as f64).floor() as usize;
    let gy = (trim * ny as f64).floor() as usize;

    // Effective sample sizes after trimming
    let hx = nx - 2 * gx;
    let hy = ny - 2 * gy;

    if hx < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: hx });
    }
    if hy < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: hy });
    }

    // Compute trimmed means
    let trimmed_mean_x = trimmed_mean(x, gx);
    let trimmed_mean_y = trimmed_mean(y, gy);

    // Compute winsorized variances
    let winvar_x = winsorized_variance(x, gx);
    let winvar_y = winsorized_variance(y, gy);

    let hx_f = hx as f64;
    let hy_f = hy as f64;

    // Standard errors for trimmed means
    // SE = sqrt(winvar / (h * (h-1)))
    let dx = (nx - 1) as f64 * winvar_x / (hx_f * (hx_f - 1.0));
    let dy = (ny - 1) as f64 * winvar_y / (hy_f * (hy_f - 1.0));

    // Difference between trimmed means
    let diff = trimmed_mean_x - trimmed_mean_y;

    // t-statistic
    let se = (dx + dy).sqrt();
    let t_stat = diff / se;

    // Welch-Satterthwaite degrees of freedom
    let df = (dx + dy).powi(2) / (dx.powi(2) / (hx_f - 1.0) + dy.powi(2) / (hy_f - 1.0));

    // Compute p-value based on alternative hypothesis
    let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
    let p_value = match alternative {
        Alternative::TwoSided => 2.0 * (1.0 - t_dist.cdf(t_stat.abs())),
        Alternative::Less => t_dist.cdf(t_stat),
        Alternative::Greater => 1.0 - t_dist.cdf(t_stat),
    };

    // Compute confidence interval if requested
    let conf_int = if let Some(level) = conf_level {
        if !(0.0 < level && level < 1.0) {
            return Err(StatError::InvalidParameter(
                "conf_level must be between 0 and 1".to_string(),
            ));
        }
        let alpha = 1.0 - level;
        let t_crit = t_dist.inverse_cdf(1.0 - alpha / 2.0);
        let margin = t_crit * se;
        Some(YuenConfInt {
            lower: diff - margin,
            upper: diff + margin,
            conf_level: level,
        })
    } else {
        None
    };

    Ok(YuenResult {
        statistic: t_stat,
        df,
        p_value,
        diff,
        trimmed_mean_x,
        trimmed_mean_y,
        conf_int,
    })
}

/// Compute trimmed mean by removing g observations from each tail
fn trimmed_mean(data: &[f64], g: usize) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    let trimmed = &sorted[g..n - g];

    trimmed.iter().sum::<f64>() / trimmed.len() as f64
}

/// Compute winsorized variance
/// Replace the g smallest values with the (g+1)th smallest
/// Replace the g largest values with the (n-g)th largest
fn winsorized_variance(data: &[f64], g: usize) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();

    // Winsorize: replace extreme values with boundary values
    let mut winsorized = sorted.clone();
    let low_val = sorted[g];
    let high_val = sorted[n - g - 1];

    for item in winsorized.iter_mut().take(g) {
        *item = low_val;
    }
    for item in winsorized.iter_mut().skip(n - g) {
        *item = high_val;
    }

    // Compute variance with n-1 denominator
    let mean: f64 = winsorized.iter().sum::<f64>() / n as f64;
    let sum_sq: f64 = winsorized.iter().map(|x| (x - mean).powi(2)).sum();
    sum_sq / (n - 1) as f64
}
