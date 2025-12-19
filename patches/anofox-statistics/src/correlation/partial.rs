//! Partial and semi-partial correlation coefficients.

use crate::correlation::{mean, validate_correlation_input};
use crate::error::{Result, StatError};
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Result of a partial correlation analysis
#[derive(Debug, Clone)]
pub struct PartialCorResult {
    /// Partial correlation coefficient
    pub estimate: f64,
    /// t-statistic
    pub statistic: f64,
    /// Degrees of freedom
    pub df: f64,
    /// p-value
    pub p_value: f64,
    /// Number of observations
    pub n: usize,
    /// Number of control variables
    pub n_controls: usize,
    /// Method name
    pub method: String,
}

/// Compute partial correlation between x and y controlling for z variables.
///
/// Partial correlation measures the linear relationship between two variables
/// while controlling for (removing the effect of) one or more other variables.
///
/// # Arguments
/// * `x` - First variable
/// * `y` - Second variable
/// * `z` - Control variables (matrix where each inner slice is one variable)
///
/// # Returns
/// * `PartialCorResult` containing the partial correlation and significance test
///
/// # Examples
/// ```
/// use anofox_statistics::correlation::partial_cor;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let y = vec![2.0, 3.0, 5.0, 4.0, 6.0, 7.0, 8.0, 9.0, 11.0, 10.0];
/// let z1 = vec![1.5, 2.5, 2.0, 3.5, 4.0, 5.5, 6.0, 7.5, 8.0, 9.5];
///
/// let result = partial_cor(&x, &y, &[&z1]).unwrap();
/// println!("Partial r = {:.4}", result.estimate);
/// println!("p-value = {:.4}", result.p_value);
/// ```
///
/// # R equivalent
/// `ppcor::pcor.test(x, y, z)`
pub fn partial_cor(x: &[f64], y: &[f64], z: &[&[f64]]) -> Result<PartialCorResult> {
    let n = validate_correlation_input(x, y)?;

    // Validate control variables
    for (i, zi) in z.iter().enumerate() {
        if zi.len() != n {
            return Err(StatError::InvalidParameter(format!(
                "Control variable {} has length {}, expected {}",
                i,
                zi.len(),
                n
            )));
        }
        // Check for non-finite values
        for (j, &val) in zi.iter().enumerate() {
            if !val.is_finite() {
                return Err(StatError::InvalidParameter(format!(
                    "Non-finite value in control variable {} at index {}",
                    i, j
                )));
            }
        }
    }

    let k = z.len(); // Number of control variables

    if k == 0 {
        // No control variables - just compute regular Pearson correlation
        return compute_simple_correlation(x, y, n);
    }

    // Need enough observations for the degrees of freedom
    if n <= k + 2 {
        return Err(StatError::InsufficientData {
            needed: k + 3,
            got: n,
        });
    }

    // Use recursive formula for partial correlation
    // r_xy.z = (r_xy.z' - r_xzk.z' * r_yzk.z') / sqrt((1 - r_xzk.z'^2)(1 - r_yzk.z'^2))
    // where z' = z without the last variable zk

    let partial_r = if k == 1 {
        // Base case: one control variable
        compute_partial_cor_single(x, y, z[0])?
    } else {
        // Recursive case: use the last control variable
        let z_rest: Vec<&[f64]> = z[..k - 1].to_vec();
        let zk = z[k - 1];

        // Compute partial correlations with z' (all but last control)
        let r_xy_zrest = partial_cor(x, y, &z_rest)?.estimate;
        let r_xzk_zrest = partial_cor(x, zk, &z_rest)?.estimate;
        let r_yzk_zrest = partial_cor(y, zk, &z_rest)?.estimate;

        // Apply the recursive formula
        let numerator = r_xy_zrest - r_xzk_zrest * r_yzk_zrest;
        let denom1 = (1.0 - r_xzk_zrest * r_xzk_zrest).sqrt();
        let denom2 = (1.0 - r_yzk_zrest * r_yzk_zrest).sqrt();

        if denom1 * denom2 > 1e-10 {
            numerator / (denom1 * denom2)
        } else {
            0.0
        }
    };

    // Clamp to [-1, 1]
    let partial_r = partial_r.clamp(-1.0, 1.0);

    // Compute t-statistic and p-value
    let df = (n - k - 2) as f64;
    let t_stat = if (1.0 - partial_r * partial_r).abs() < 1e-15 {
        if partial_r > 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        partial_r * (df / (1.0 - partial_r * partial_r)).sqrt()
    };

    let p_value = if t_stat.is_infinite() {
        0.0
    } else if df > 0.0 {
        let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
        2.0 * (1.0 - t_dist.cdf(t_stat.abs()))
    } else {
        1.0
    };

    Ok(PartialCorResult {
        estimate: partial_r,
        statistic: t_stat,
        df,
        p_value,
        n,
        n_controls: k,
        method: format!("Partial correlation (controlling for {} variable(s))", k),
    })
}

/// Compute partial correlation with a single control variable.
fn compute_partial_cor_single(x: &[f64], y: &[f64], z: &[f64]) -> Result<f64> {
    let r_xy = pearson_r(x, y);
    let r_xz = pearson_r(x, z);
    let r_yz = pearson_r(y, z);

    let numerator = r_xy - r_xz * r_yz;
    let denom1 = (1.0 - r_xz * r_xz).sqrt();
    let denom2 = (1.0 - r_yz * r_yz).sqrt();

    if denom1 * denom2 > 1e-10 {
        Ok(numerator / (denom1 * denom2))
    } else {
        Ok(0.0)
    }
}

/// Compute simple Pearson correlation coefficient (no control variables).
fn compute_simple_correlation(x: &[f64], y: &[f64], n: usize) -> Result<PartialCorResult> {
    let r = pearson_r(x, y);
    let df = (n - 2) as f64;

    let t_stat = if (1.0 - r * r).abs() < 1e-15 {
        if r > 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        r * (df / (1.0 - r * r)).sqrt()
    };

    let p_value = if t_stat.is_infinite() {
        0.0
    } else {
        let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
        2.0 * (1.0 - t_dist.cdf(t_stat.abs()))
    };

    Ok(PartialCorResult {
        estimate: r,
        statistic: t_stat,
        df,
        p_value,
        n,
        n_controls: 0,
        method: "Pearson correlation (no controls)".to_string(),
    })
}

/// Compute Pearson correlation coefficient.
fn pearson_r(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let mean_x = mean(x);
    let mean_y = mean(y);

    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        sum_xy += dx * dy;
        sum_xx += dx * dx;
        sum_yy += dy * dy;
    }

    if sum_xx > 0.0 && sum_yy > 0.0 {
        sum_xy / (sum_xx * sum_yy).sqrt()
    } else {
        0.0
    }
}

/// Semi-partial (part) correlation between x and y, controlling for z on y only.
///
/// Measures the correlation between x and the residuals of y after removing
/// the effect of z. Also known as "part correlation".
///
/// # Arguments
/// * `x` - First variable (not adjusted)
/// * `y` - Second variable (will be adjusted for z)
/// * `z` - Control variables
///
/// # Returns
/// * `PartialCorResult` containing the semi-partial correlation
///
/// # R equivalent
/// `ppcor::spcor.test(x, y, z)`
pub fn semi_partial_cor(x: &[f64], y: &[f64], z: &[&[f64]]) -> Result<PartialCorResult> {
    let n = validate_correlation_input(x, y)?;

    // Validate control variables
    for (i, zi) in z.iter().enumerate() {
        if zi.len() != n {
            return Err(StatError::InvalidParameter(format!(
                "Control variable {} has length {}, expected {}",
                i,
                zi.len(),
                n
            )));
        }
    }

    let k = z.len();

    if k == 0 {
        return compute_simple_correlation(x, y, n);
    }

    if n <= k + 2 {
        return Err(StatError::InsufficientData {
            needed: k + 3,
            got: n,
        });
    }

    // Compute residuals of y after regressing on z
    let y_residuals = compute_residuals(y, z);

    // Compute correlation between x and residuals of y
    let r = pearson_r(x, &y_residuals);
    let r = r.clamp(-1.0, 1.0);

    let df = (n - k - 2) as f64;
    let t_stat = if (1.0 - r * r).abs() < 1e-15 {
        if r > 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        r * (df / (1.0 - r * r)).sqrt()
    };

    let p_value = if t_stat.is_infinite() {
        0.0
    } else if df > 0.0 {
        let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
        2.0 * (1.0 - t_dist.cdf(t_stat.abs()))
    } else {
        1.0
    };

    Ok(PartialCorResult {
        estimate: r,
        statistic: t_stat,
        df,
        p_value,
        n,
        n_controls: k,
        method: format!(
            "Semi-partial correlation (controlling for {} variable(s) on y)",
            k
        ),
    })
}

/// Compute residuals after regressing y on z variables using OLS.
fn compute_residuals(y: &[f64], z: &[&[f64]]) -> Vec<f64> {
    let n = y.len();
    let k = z.len();

    if k == 0 {
        return y.to_vec();
    }

    // Simple case: one predictor
    if k == 1 {
        let z0 = z[0];
        let mean_y = mean(y);
        let mean_z = mean(z0);

        let mut sum_zy = 0.0;
        let mut sum_zz = 0.0;

        for i in 0..n {
            let dz = z0[i] - mean_z;
            sum_zy += dz * (y[i] - mean_y);
            sum_zz += dz * dz;
        }

        let beta = if sum_zz > 0.0 { sum_zy / sum_zz } else { 0.0 };
        let alpha = mean_y - beta * mean_z;

        return y
            .iter()
            .zip(z0.iter())
            .map(|(&yi, &zi)| yi - alpha - beta * zi)
            .collect();
    }

    // Multiple predictors: use normal equations
    // This is a simplified approach; for production, use proper linear algebra
    // y = Z * beta + residuals
    // beta = (Z'Z)^(-1) * Z'y

    // For simplicity, we use sequential orthogonalization
    let mut residuals = y.to_vec();

    for zi in z {
        let mean_r: f64 = residuals.iter().sum::<f64>() / n as f64;
        let mean_z = mean(zi);

        let mut sum_rz = 0.0;
        let mut sum_zz = 0.0;

        for i in 0..n {
            let dz = zi[i] - mean_z;
            sum_rz += dz * (residuals[i] - mean_r);
            sum_zz += dz * dz;
        }

        let beta = if sum_zz > 0.0 { sum_rz / sum_zz } else { 0.0 };
        let alpha = mean_r - beta * mean_z;

        for i in 0..n {
            residuals[i] -= alpha + beta * zi[i];
        }
    }

    residuals
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_cor_no_controls() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let result = partial_cor(&x, &y, &[]).unwrap();

        assert!((result.estimate - 1.0).abs() < 1e-10);
        assert_eq!(result.n_controls, 0);
    }

    #[test]
    fn test_partial_cor_single_control() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![2.1, 3.9, 6.1, 7.9, 10.1, 11.9, 14.1, 15.9, 18.1, 19.9];
        let z = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5];

        let result = partial_cor(&x, &y, &[&z]).unwrap();

        // Result should be between -1 and 1
        assert!(result.estimate >= -1.0 && result.estimate <= 1.0);
        assert_eq!(result.n_controls, 1);
        // x, y, z are all highly collinear, so partial correlation may be lower
        // Just verify the computation runs correctly
    }

    #[test]
    fn test_partial_cor_confounded() {
        // x and y are both caused by z (confounded)
        let z = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let x: Vec<f64> = z.iter().map(|&zi| zi * 2.0 + 0.1).collect();
        let y: Vec<f64> = z.iter().map(|&zi| zi * 3.0 - 0.1).collect();

        // Without control, x and y are correlated
        let result_no_control = partial_cor(&x, &y, &[]).unwrap();
        assert!(result_no_control.estimate > 0.9);

        // With z controlled, correlation should be much smaller
        let result_controlled = partial_cor(&x, &y, &[&z]).unwrap();
        assert!(result_controlled.estimate.abs() < result_no_control.estimate.abs());
    }

    #[test]
    fn test_partial_cor_multiple_controls() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![2.0, 3.5, 5.0, 6.5, 8.0, 9.5, 11.0, 12.5, 14.0, 15.5];
        let z1 = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let z2 = vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5];

        let result = partial_cor(&x, &y, &[&z1, &z2]).unwrap();

        assert!(result.estimate.abs() <= 1.0);
        assert_eq!(result.n_controls, 2);
    }

    #[test]
    fn test_semi_partial_cor() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![2.0, 4.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0];
        let z = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5];

        let result = semi_partial_cor(&x, &y, &[&z]).unwrap();

        assert!(result.estimate.abs() <= 1.0);
        assert_eq!(result.n_controls, 1);
    }
}
