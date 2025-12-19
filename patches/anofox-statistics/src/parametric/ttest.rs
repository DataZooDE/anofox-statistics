use crate::error::{Result, StatError};
use crate::utils::math::{mean, variance};
use statrs::distribution::{ContinuousCDF, StudentsT};

/// The kind of t-test to perform
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TTestKind {
    /// Independent samples, unequal variances (Welch's t-test)
    Welch,
    /// Independent samples, equal variances assumed (Student's t-test)
    Student,
    /// Paired samples t-test
    Paired,
}

/// Alternative hypothesis direction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Alternative {
    /// Two-sided test (x != y)
    TwoSided,
    /// One-sided test (x < y)
    Less,
    /// One-sided test (x > y)
    Greater,
}

/// Confidence interval for t-test
#[derive(Debug, Clone)]
pub struct TTestConfInt {
    /// Lower bound of the confidence interval
    pub lower: f64,
    /// Upper bound of the confidence interval
    pub upper: f64,
    /// Confidence level (e.g., 0.95 for 95%)
    pub conf_level: f64,
}

/// Result of a t-test
#[derive(Debug, Clone)]
pub struct TTestResult {
    /// The t-statistic
    pub statistic: f64,
    /// Degrees of freedom
    pub df: f64,
    /// The p-value
    pub p_value: f64,
    /// Mean of first sample (or mean difference for paired)
    pub mean_x: f64,
    /// Mean of second sample (None for one-sample or paired)
    pub mean_y: Option<f64>,
    /// Confidence interval for the mean difference (if conf_level was specified)
    pub conf_int: Option<TTestConfInt>,
    /// Null hypothesis value (mu parameter)
    pub null_value: f64,
}

/// Perform a t-test comparing two samples.
///
/// # Arguments
/// * `x` - First sample
/// * `y` - Second sample
/// * `kind` - Type of t-test (Welch, Student, or Paired)
/// * `alternative` - Alternative hypothesis direction
/// * `mu` - Null hypothesis value for the true difference in means (default: 0.0)
/// * `conf_level` - If Some, compute confidence interval at this level (e.g., 0.95)
///
/// # Returns
/// * `TTestResult` containing statistic, df, p-value, and optionally CI
pub fn t_test(
    x: &[f64],
    y: &[f64],
    kind: TTestKind,
    alternative: Alternative,
    mu: f64,
    conf_level: Option<f64>,
) -> Result<TTestResult> {
    match kind {
        TTestKind::Welch => welch_t_test(x, y, alternative, mu, conf_level),
        TTestKind::Student => student_t_test(x, y, alternative, mu, conf_level),
        TTestKind::Paired => paired_t_test(x, y, alternative, mu, conf_level),
    }
}

/// Welch's t-test for independent samples with unequal variances
fn welch_t_test(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
    mu: f64,
    conf_level: Option<f64>,
) -> Result<TTestResult> {
    let nx = x.len();
    let ny = y.len();

    if nx < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: nx });
    }
    if ny < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: ny });
    }

    let mean_x = mean(x)?;
    let mean_y = mean(y)?;
    let var_x = variance(x)?;
    let var_y = variance(y)?;

    let nx_f = nx as f64;
    let ny_f = ny as f64;

    // Standard error of the difference
    let se_x = var_x / nx_f;
    let se_y = var_y / ny_f;
    let se = (se_x + se_y).sqrt();

    // t-statistic (testing H0: mean_x - mean_y = mu)
    let t_stat = (mean_x - mean_y - mu) / se;

    // Welch-Satterthwaite degrees of freedom
    let num = (se_x + se_y).powi(2);
    let denom = (se_x.powi(2) / (nx_f - 1.0)) + (se_y.powi(2) / (ny_f - 1.0));
    let df = num / denom;

    let p_value = compute_p_value(t_stat, df, alternative);

    // Confidence interval for the difference in means
    let conf_int = compute_conf_int(mean_x - mean_y, se, df, conf_level)?;

    Ok(TTestResult {
        statistic: t_stat,
        df,
        p_value,
        mean_x,
        mean_y: Some(mean_y),
        conf_int,
        null_value: mu,
    })
}

/// Student's t-test for independent samples with equal variances assumed
fn student_t_test(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
    mu: f64,
    conf_level: Option<f64>,
) -> Result<TTestResult> {
    let nx = x.len();
    let ny = y.len();

    if nx < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: nx });
    }
    if ny < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: ny });
    }

    let mean_x = mean(x)?;
    let mean_y = mean(y)?;
    let var_x = variance(x)?;
    let var_y = variance(y)?;

    let nx_f = nx as f64;
    let ny_f = ny as f64;

    // Pooled variance
    let pooled_var = ((nx_f - 1.0) * var_x + (ny_f - 1.0) * var_y) / (nx_f + ny_f - 2.0);

    // Standard error of the difference
    let se = (pooled_var * (1.0 / nx_f + 1.0 / ny_f)).sqrt();

    // t-statistic (testing H0: mean_x - mean_y = mu)
    let t_stat = (mean_x - mean_y - mu) / se;

    // Degrees of freedom
    let df = nx_f + ny_f - 2.0;

    let p_value = compute_p_value(t_stat, df, alternative);

    // Confidence interval for the difference in means
    let conf_int = compute_conf_int(mean_x - mean_y, se, df, conf_level)?;

    Ok(TTestResult {
        statistic: t_stat,
        df,
        p_value,
        mean_x,
        mean_y: Some(mean_y),
        conf_int,
        null_value: mu,
    })
}

/// Paired t-test for dependent samples
fn paired_t_test(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
    mu: f64,
    conf_level: Option<f64>,
) -> Result<TTestResult> {
    let n = x.len();

    if n != y.len() {
        return Err(StatError::InvalidParameter(format!(
            "paired t-test requires equal length samples, got {} and {}",
            n,
            y.len()
        )));
    }

    if n < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: n });
    }

    // Compute differences
    let diffs: Vec<f64> = x.iter().zip(y.iter()).map(|(xi, yi)| xi - yi).collect();

    let mean_diff = mean(&diffs)?;
    let var_diff = variance(&diffs)?;

    let n_f = n as f64;

    // Standard error of the mean difference
    let se = (var_diff / n_f).sqrt();

    // t-statistic (testing H0: mean_diff = mu)
    let t_stat = (mean_diff - mu) / se;

    // Degrees of freedom
    let df = n_f - 1.0;

    let p_value = compute_p_value(t_stat, df, alternative);

    // Confidence interval for the mean difference
    let conf_int = compute_conf_int(mean_diff, se, df, conf_level)?;

    Ok(TTestResult {
        statistic: t_stat,
        df,
        p_value,
        mean_x: mean_diff,
        mean_y: None,
        conf_int,
        null_value: mu,
    })
}

/// Compute p-value from t-statistic and degrees of freedom
fn compute_p_value(t_stat: f64, df: f64, alternative: Alternative) -> f64 {
    let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();

    match alternative {
        Alternative::TwoSided => {
            // Two-tailed: P(|T| > |t|) = 2 * P(T > |t|)
            2.0 * (1.0 - t_dist.cdf(t_stat.abs()))
        }
        Alternative::Less => {
            // Left-tailed: P(T < t)
            t_dist.cdf(t_stat)
        }
        Alternative::Greater => {
            // Right-tailed: P(T > t) = 1 - P(T < t)
            1.0 - t_dist.cdf(t_stat)
        }
    }
}

/// Compute confidence interval for mean difference
fn compute_conf_int(
    estimate: f64,
    se: f64,
    df: f64,
    conf_level: Option<f64>,
) -> Result<Option<TTestConfInt>> {
    match conf_level {
        Some(level) => {
            if !(0.0 < level && level < 1.0) {
                return Err(StatError::InvalidParameter(
                    "conf_level must be between 0 and 1".to_string(),
                ));
            }
            let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
            let alpha = 1.0 - level;
            let t_crit = t_dist.inverse_cdf(1.0 - alpha / 2.0);
            let margin = t_crit * se;
            Ok(Some(TTestConfInt {
                lower: estimate - margin,
                upper: estimate + margin,
                conf_level: level,
            }))
        }
        None => Ok(None),
    }
}
