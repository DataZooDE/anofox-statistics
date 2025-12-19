//! Intraclass Correlation Coefficient (ICC).
//!
//! ICC measures the reliability of ratings or measurements.

use crate::error::{Result, StatError};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

/// Type of ICC to compute
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ICCType {
    /// ICC(1): One-way random effects, absolute agreement, single rater
    ICC1,
    /// ICC(2): Two-way random effects, absolute agreement, single rater
    #[default]
    ICC2,
    /// ICC(3): Two-way mixed effects, consistency, single rater
    ICC3,
    /// ICC(1,k): One-way random effects, absolute agreement, average of k raters
    ICC1k,
    /// ICC(2,k): Two-way random effects, absolute agreement, average of k raters
    ICC2k,
    /// ICC(3,k): Two-way mixed effects, consistency, average of k raters
    ICC3k,
}

impl std::fmt::Display for ICCType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ICCType::ICC1 => write!(f, "ICC(1)"),
            ICCType::ICC2 => write!(f, "ICC(2,1)"),
            ICCType::ICC3 => write!(f, "ICC(3,1)"),
            ICCType::ICC1k => write!(f, "ICC(1,k)"),
            ICCType::ICC2k => write!(f, "ICC(2,k)"),
            ICCType::ICC3k => write!(f, "ICC(3,k)"),
        }
    }
}

/// Result of an ICC calculation
#[derive(Debug, Clone)]
pub struct ICCResult {
    /// ICC value
    pub icc: f64,
    /// Type of ICC computed
    pub icc_type: ICCType,
    /// F-statistic
    pub f_value: f64,
    /// Degrees of freedom (numerator)
    pub df1: f64,
    /// Degrees of freedom (denominator)
    pub df2: f64,
    /// p-value
    pub p_value: f64,
    /// 95% confidence interval lower bound
    pub conf_int_lower: f64,
    /// 95% confidence interval upper bound
    pub conf_int_upper: f64,
    /// Number of subjects/targets
    pub n_subjects: usize,
    /// Number of raters/measurements
    pub n_raters: usize,
    /// Method name
    pub method: String,
}

/// Compute Intraclass Correlation Coefficient.
///
/// ICC measures the reliability of ratings or measurements made by
/// different raters on the same subjects.
///
/// # Arguments
/// * `data` - Matrix where rows are subjects and columns are raters/measurements
/// * `icc_type` - Type of ICC to compute
///
/// # Returns
/// * `ICCResult` containing ICC value, F-test, and confidence interval
///
/// # Examples
/// ```
/// use anofox_statistics::correlation::{icc, ICCType};
///
/// // 6 subjects rated by 4 raters
/// let data = vec![
///     vec![9.0, 2.0, 5.0, 8.0],
///     vec![6.0, 1.0, 3.0, 2.0],
///     vec![8.0, 4.0, 6.0, 8.0],
///     vec![7.0, 1.0, 2.0, 6.0],
///     vec![10.0, 5.0, 6.0, 9.0],
///     vec![6.0, 2.0, 4.0, 7.0],
/// ];
///
/// let result = icc(&data, ICCType::ICC2).unwrap();
/// println!("ICC(2,1) = {:.4}", result.icc);
/// println!("F = {:.4}, p = {:.4}", result.f_value, result.p_value);
/// ```
///
/// # R equivalent
/// `psych::ICC(data)` or `irr::icc(data, model = "twoway", type = "agreement")`
pub fn icc(data: &[Vec<f64>], icc_type: ICCType) -> Result<ICCResult> {
    validate_icc_input(data)?;

    let n = data.len(); // Number of subjects
    let k = data[0].len(); // Number of raters

    // Compute ANOVA components
    let (ms_r, ms_c, ms_e, ms_w) = compute_anova_components(data);

    // Compute ICC based on type
    let (icc_value, f_value, df1, df2) = match icc_type {
        ICCType::ICC1 => {
            // One-way random: ICC(1) = (MS_R - MS_W) / (MS_R + (k-1)*MS_W)
            let icc = (ms_r - ms_w) / (ms_r + (k as f64 - 1.0) * ms_w);
            let f = ms_r / ms_w;
            (icc, f, (n - 1) as f64, n as f64 * (k as f64 - 1.0))
        }
        ICCType::ICC2 => {
            // Two-way random, absolute agreement
            // ICC(2,1) = (MS_R - MS_E) / (MS_R + (k-1)*MS_E + k*(MS_C - MS_E)/n)
            let numer = ms_r - ms_e;
            let denom = ms_r + (k as f64 - 1.0) * ms_e + k as f64 * (ms_c - ms_e) / n as f64;
            let icc = if denom > 0.0 { numer / denom } else { 0.0 };
            let f = ms_r / ms_e;
            (icc, f, (n - 1) as f64, ((n - 1) * (k - 1)) as f64)
        }
        ICCType::ICC3 => {
            // Two-way mixed, consistency
            // ICC(3,1) = (MS_R - MS_E) / (MS_R + (k-1)*MS_E)
            let icc = (ms_r - ms_e) / (ms_r + (k as f64 - 1.0) * ms_e);
            let f = ms_r / ms_e;
            (icc, f, (n - 1) as f64, ((n - 1) * (k - 1)) as f64)
        }
        ICCType::ICC1k => {
            // One-way random, average of k raters
            // ICC(1,k) = (MS_R - MS_W) / MS_R
            let icc = (ms_r - ms_w) / ms_r;
            let f = ms_r / ms_w;
            (icc, f, (n - 1) as f64, n as f64 * (k as f64 - 1.0))
        }
        ICCType::ICC2k => {
            // Two-way random, absolute agreement, average of k raters
            // ICC(2,k) = (MS_R - MS_E) / (MS_R + (MS_C - MS_E)/n)
            let numer = ms_r - ms_e;
            let denom = ms_r + (ms_c - ms_e) / n as f64;
            let icc = if denom > 0.0 { numer / denom } else { 0.0 };
            let f = ms_r / ms_e;
            (icc, f, (n - 1) as f64, ((n - 1) * (k - 1)) as f64)
        }
        ICCType::ICC3k => {
            // Two-way mixed, consistency, average of k raters
            // ICC(3,k) = (MS_R - MS_E) / MS_R
            let icc = (ms_r - ms_e) / ms_r;
            let f = ms_r / ms_e;
            (icc, f, (n - 1) as f64, ((n - 1) * (k - 1)) as f64)
        }
    };

    // Clamp ICC to [-1, 1]
    let icc_value = icc_value.clamp(-1.0, 1.0);

    // Compute p-value
    let p_value = if f_value > 0.0 && df1 > 0.0 && df2 > 0.0 {
        let f_dist = FisherSnedecor::new(df1, df2).unwrap();
        1.0 - f_dist.cdf(f_value)
    } else {
        1.0
    };

    // Compute confidence interval using F-distribution
    let (conf_int_lower, conf_int_upper) =
        compute_icc_ci(icc_value, f_value, df1, df2, n, k, icc_type);

    Ok(ICCResult {
        icc: icc_value,
        icc_type,
        f_value,
        df1,
        df2,
        p_value,
        conf_int_lower,
        conf_int_upper,
        n_subjects: n,
        n_raters: k,
        method: format!("{} - Intraclass Correlation Coefficient", icc_type),
    })
}

/// Validate input data for ICC calculation.
fn validate_icc_input(data: &[Vec<f64>]) -> Result<()> {
    if data.is_empty() {
        return Err(StatError::EmptyData);
    }

    let n = data.len();
    if n < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: n });
    }

    let k = data[0].len();
    if k < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: k });
    }

    // Check all rows have same length and no non-finite values
    for (i, row) in data.iter().enumerate() {
        if row.len() != k {
            return Err(StatError::InvalidParameter(format!(
                "Row {} has {} columns, expected {}",
                i,
                row.len(),
                k
            )));
        }
        for (j, &val) in row.iter().enumerate() {
            if !val.is_finite() {
                return Err(StatError::InvalidParameter(format!(
                    "Non-finite value at row {}, column {}",
                    i, j
                )));
            }
        }
    }

    Ok(())
}

/// Compute ANOVA components for ICC.
/// Returns (MS_between_subjects, MS_between_raters, MS_error, MS_within)
fn compute_anova_components(data: &[Vec<f64>]) -> (f64, f64, f64, f64) {
    let n = data.len(); // Subjects
    let k = data[0].len(); // Raters
    let n_f = n as f64;
    let k_f = k as f64;
    let total_n = (n * k) as f64;

    // Grand mean
    let grand_mean: f64 = data.iter().flat_map(|row| row.iter()).sum::<f64>() / total_n;

    // Row means (subject means)
    let row_means: Vec<f64> = data
        .iter()
        .map(|row| row.iter().sum::<f64>() / k_f)
        .collect();

    // Column means (rater means)
    let col_means: Vec<f64> = (0..k)
        .map(|j| data.iter().map(|row| row[j]).sum::<f64>() / n_f)
        .collect();

    // Sum of squares
    // SS_total = sum((x_ij - grand_mean)^2)
    let ss_total: f64 = data
        .iter()
        .flat_map(|row| row.iter())
        .map(|&x| (x - grand_mean).powi(2))
        .sum();

    // SS_between_subjects (rows) = k * sum((row_mean - grand_mean)^2)
    let ss_r: f64 = k_f
        * row_means
            .iter()
            .map(|&m| (m - grand_mean).powi(2))
            .sum::<f64>();

    // SS_between_raters (columns) = n * sum((col_mean - grand_mean)^2)
    let ss_c: f64 = n_f
        * col_means
            .iter()
            .map(|&m| (m - grand_mean).powi(2))
            .sum::<f64>();

    // SS_error (residual) = SS_total - SS_r - SS_c
    let ss_e = (ss_total - ss_r - ss_c).max(0.0);

    // SS_within = SS_total - SS_r (for one-way model)
    let ss_w = (ss_total - ss_r).max(0.0);

    // Degrees of freedom
    let df_r = n_f - 1.0;
    let df_c = k_f - 1.0;
    let df_e = (n_f - 1.0) * (k_f - 1.0);
    let df_w = n_f * (k_f - 1.0);

    // Mean squares
    let ms_r = if df_r > 0.0 { ss_r / df_r } else { 0.0 };
    let ms_c = if df_c > 0.0 { ss_c / df_c } else { 0.0 };
    let ms_e = if df_e > 0.0 { ss_e / df_e } else { 0.0 };
    let ms_w = if df_w > 0.0 { ss_w / df_w } else { 0.0 };

    (ms_r, ms_c, ms_e, ms_w)
}

/// Compute confidence interval for ICC.
fn compute_icc_ci(
    _icc: f64,
    f_value: f64,
    df1: f64,
    df2: f64,
    n: usize,
    k: usize,
    icc_type: ICCType,
) -> (f64, f64) {
    // Use F-based confidence interval
    // For ICC, the CI is typically derived from the F-distribution

    if df1 <= 0.0 || df2 <= 0.0 || !f_value.is_finite() {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }

    let alpha = 0.05;
    let f_dist = FisherSnedecor::new(df1, df2).unwrap();
    let f_lower = f_dist.inverse_cdf(alpha / 2.0);
    let f_upper = f_dist.inverse_cdf(1.0 - alpha / 2.0);

    let _n = n; // Suppress unused warning - kept for potential future use
    let k_f = k as f64;

    // The CI formulas depend on the ICC type
    // Using simplified approximation based on F-ratio
    let (lower, upper) = match icc_type {
        ICCType::ICC1 | ICCType::ICC2 | ICCType::ICC3 => {
            // For single rater ICC
            let f_l = f_value / f_upper;
            let f_u = f_value / f_lower;

            let lower = (f_l - 1.0) / (f_l + k_f - 1.0);
            let upper = (f_u - 1.0) / (f_u + k_f - 1.0);

            (lower.max(-1.0), upper.min(1.0))
        }
        ICCType::ICC1k | ICCType::ICC2k | ICCType::ICC3k => {
            // For average of k raters
            let f_l = f_value / f_upper;
            let f_u = f_value / f_lower;

            let lower = 1.0 - 1.0 / f_l;
            let upper = 1.0 - 1.0 / f_u;

            (lower.max(-1.0), upper.min(1.0))
        }
    };

    (lower, upper)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icc_perfect_agreement() {
        // All raters give same scores
        let data = vec![
            vec![1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0],
            vec![3.0, 3.0, 3.0],
            vec![4.0, 4.0, 4.0],
            vec![5.0, 5.0, 5.0],
        ];

        let result = icc(&data, ICCType::ICC2).unwrap();

        // Perfect agreement should give ICC = 1.0
        assert!((result.icc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_icc_no_agreement() {
        // Raters have no consistency
        let data = vec![
            vec![1.0, 5.0, 3.0],
            vec![5.0, 1.0, 2.0],
            vec![3.0, 4.0, 1.0],
            vec![2.0, 3.0, 5.0],
            vec![4.0, 2.0, 4.0],
        ];

        let result = icc(&data, ICCType::ICC2).unwrap();

        // Low agreement should give low ICC
        assert!(result.icc < 0.5);
    }

    #[test]
    fn test_icc_types() {
        let data = vec![
            vec![9.0, 2.0, 5.0, 8.0],
            vec![6.0, 1.0, 3.0, 2.0],
            vec![8.0, 4.0, 6.0, 8.0],
            vec![7.0, 1.0, 2.0, 6.0],
            vec![10.0, 5.0, 6.0, 9.0],
            vec![6.0, 2.0, 4.0, 7.0],
        ];

        let icc1 = icc(&data, ICCType::ICC1).unwrap();
        let icc2 = icc(&data, ICCType::ICC2).unwrap();
        let icc3 = icc(&data, ICCType::ICC3).unwrap();

        // All should be between -1 and 1
        assert!(icc1.icc >= -1.0 && icc1.icc <= 1.0);
        assert!(icc2.icc >= -1.0 && icc2.icc <= 1.0);
        assert!(icc3.icc >= -1.0 && icc3.icc <= 1.0);

        // ICC3 (consistency) is typically >= ICC2 (absolute agreement)
        // when there are systematic rater differences
        assert!(icc3.icc >= icc2.icc - 0.01);
    }

    #[test]
    fn test_icc_average_raters() {
        let data = vec![
            vec![9.0, 2.0, 5.0, 8.0],
            vec![6.0, 1.0, 3.0, 2.0],
            vec![8.0, 4.0, 6.0, 8.0],
            vec![7.0, 1.0, 2.0, 6.0],
            vec![10.0, 5.0, 6.0, 9.0],
            vec![6.0, 2.0, 4.0, 7.0],
        ];

        let icc2_single = icc(&data, ICCType::ICC2).unwrap();
        let icc2_avg = icc(&data, ICCType::ICC2k).unwrap();

        // Average of k raters should have higher reliability than single rater
        assert!(icc2_avg.icc >= icc2_single.icc);
    }

    #[test]
    fn test_icc_ci() {
        let data = vec![
            vec![9.0, 2.0, 5.0, 8.0],
            vec![6.0, 1.0, 3.0, 2.0],
            vec![8.0, 4.0, 6.0, 8.0],
            vec![7.0, 1.0, 2.0, 6.0],
            vec![10.0, 5.0, 6.0, 9.0],
            vec![6.0, 2.0, 4.0, 7.0],
        ];

        let result = icc(&data, ICCType::ICC2).unwrap();

        // CI bounds should be finite and in valid range
        assert!(result.conf_int_lower >= -1.0);
        assert!(result.conf_int_upper <= 1.0);
        // Lower bound should be less than upper bound
        assert!(result.conf_int_lower < result.conf_int_upper);
    }
}
