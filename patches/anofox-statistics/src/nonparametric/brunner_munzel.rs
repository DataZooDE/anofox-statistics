use crate::error::{Result, StatError};
use crate::nonparametric::ranks::rank;
use crate::parametric::Alternative;
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Confidence interval for Brunner-Munzel estimate
#[derive(Debug, Clone)]
pub struct BrunnerMunzelConfInt {
    /// Lower bound of the confidence interval
    pub lower: f64,
    /// Upper bound of the confidence interval
    pub upper: f64,
    /// Confidence level (e.g., 0.95 for 95%)
    pub conf_level: f64,
}

/// Result of the Brunner-Munzel test
#[derive(Debug, Clone)]
pub struct BrunnerMunzelResult {
    /// The test statistic
    pub statistic: f64,
    /// Degrees of freedom (Welch-Satterthwaite approximation)
    pub df: f64,
    /// The p-value
    pub p_value: f64,
    /// Estimated probability P(X < Y) + 0.5 * P(X = Y)
    pub estimate: f64,
    /// Confidence interval for the estimate (if alpha was specified)
    pub conf_int: Option<BrunnerMunzelConfInt>,
}

/// Validate Brunner-Munzel test inputs.
fn validate_bm_inputs(n1: usize, n2: usize) -> Result<()> {
    if n1 == 0 || n2 == 0 {
        return Err(StatError::EmptyData);
    }
    if n1 < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: n1 });
    }
    if n2 < 2 {
        return Err(StatError::InsufficientData { needed: 2, got: n2 });
    }
    Ok(())
}

/// Compute Brunner-Munzel variance estimate for a sample.
///
/// v = sum((combined_rank - within_rank - mean_rank + (n+1)/2)^2) / (n-1)
fn compute_bm_variance(
    combined_ranks: &[f64],
    within_ranks: &[f64],
    mean_rank: f64,
    n: f64,
) -> f64 {
    combined_ranks
        .iter()
        .zip(within_ranks.iter())
        .map(|(ri, wi)| {
            let diff = ri - wi - mean_rank + (n + 1.0) / 2.0;
            diff * diff
        })
        .sum::<f64>()
        / (n - 1.0)
}

/// Compute p-value from t-distribution based on alternative hypothesis.
fn compute_bm_pvalue(statistic: f64, df: f64, alternative: Alternative) -> f64 {
    let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();

    match alternative {
        Alternative::TwoSided => {
            2.0 * (1.0 - t_dist.cdf(statistic.abs())).min(t_dist.cdf(statistic.abs()))
        }
        Alternative::Greater => t_dist.cdf(statistic),
        Alternative::Less => 1.0 - t_dist.cdf(statistic),
    }
}

/// Perform the Brunner-Munzel test for stochastic equality.
///
/// This is a robust alternative to the Mann-Whitney U test that handles
/// unequal variances and works well with ties. It tests whether
/// P(X < Y) + 0.5 * P(X = Y) = 0.5.
///
/// # Arguments
/// * `x` - First sample
/// * `y` - Second sample
/// * `alternative` - Alternative hypothesis
/// * `alpha` - If Some, compute confidence interval at (1-alpha) level (e.g., 0.05 for 95% CI)
///
/// # Returns
/// * `BrunnerMunzelResult` containing statistic, df, p-value, probability estimate, and optional CI
///
/// # References
/// * Brunner, E. and Munzel, U. (2000). "The Nonparametric Behrens-Fisher Problem:
///   Asymptotic Theory and a Small Sample Approximation"
pub fn brunner_munzel(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
    alpha: Option<f64>,
) -> Result<BrunnerMunzelResult> {
    let n1 = x.len();
    let n2 = y.len();

    validate_bm_inputs(n1, n2)?;

    let n1_f = n1 as f64;
    let n2_f = n2 as f64;

    // Compute ranks within each sample
    let r1 = rank(x)?;
    let r2 = rank(y)?;

    // Combine samples and compute overall ranks
    let combined: Vec<f64> = x.iter().chain(y.iter()).cloned().collect();
    let r = rank(&combined)?;

    // Split overall ranks back into groups
    let r_x = &r[0..n1];
    let r_y = &r[n1..];

    // Mean ranks in combined sample
    let m1: f64 = r_x.iter().sum::<f64>() / n1_f;
    let m2: f64 = r_y.iter().sum::<f64>() / n2_f;

    // Probability estimate: P(X < Y) + 0.5 * P(X = Y)
    let pst = (m2 - (n2_f + 1.0) / 2.0) / n1_f;

    // Variance estimates
    let v1 = compute_bm_variance(r_x, &r1, m1, n1_f);
    let v2 = compute_bm_variance(r_y, &r2, m2, n2_f);

    // Test statistic
    let n_total = n1_f + n2_f;
    let var_sum = n1_f * v1 + n2_f * v2;

    if var_sum < 1e-14 {
        return Err(StatError::InvalidParameter(
            "Variance is effectively zero".to_string(),
        ));
    }

    let statistic = n1_f * n2_f * (m2 - m1) / n_total / var_sum.sqrt();

    // Degrees of freedom (Welch-Satterthwaite approximation)
    let df =
        var_sum.powi(2) / ((n1_f * v1).powi(2) / (n1_f - 1.0) + (n2_f * v2).powi(2) / (n2_f - 1.0));

    // P-value from t-distribution
    let p_value = compute_bm_pvalue(statistic, df, alternative);

    // Confidence interval if alpha is specified
    // SE = sqrt(v1 / (n1 * n2^2) + v2 / (n2 * n1^2))
    let conf_int = if let Some(a) = alpha {
        if !(0.0 < a && a < 1.0) {
            return Err(StatError::InvalidParameter(
                "alpha must be between 0 and 1".to_string(),
            ));
        }
        let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
        let t_crit = t_dist.inverse_cdf(1.0 - a / 2.0);
        let se = (v1 / (n1_f * n2_f * n2_f) + v2 / (n2_f * n1_f * n1_f)).sqrt();
        let lower = pst - t_crit * se;
        let upper = pst + t_crit * se;
        Some(BrunnerMunzelConfInt {
            lower,
            upper,
            conf_level: 1.0 - a,
        })
    } else {
        None
    };

    Ok(BrunnerMunzelResult {
        statistic,
        df,
        p_value,
        estimate: pst,
        conf_int,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_brunner_munzel_basic() {
        // Test data from R: lawstat::brunner.munzel.test
        let x = vec![
            1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 1.0, 1.0,
        ];
        let y = vec![3.0, 3.0, 4.0, 3.0, 1.0, 2.0, 3.0, 1.0, 1.0, 5.0, 4.0];

        let result = brunner_munzel(&x, &y, Alternative::TwoSided, None).unwrap();

        // R reference values:
        // statistic = 3.1375, df = 17.683, p-value = 0.005786, estimate = 0.788961
        assert_relative_eq!(result.statistic, 3.1375, epsilon = 0.001);
        assert_relative_eq!(result.df, 17.683, epsilon = 0.01);
        assert_relative_eq!(result.p_value, 0.005786, epsilon = 0.0001);
        assert_relative_eq!(result.estimate, 0.788961, epsilon = 0.0001);
    }

    #[test]
    fn test_brunner_munzel_equal_samples() {
        // Equal samples should give statistic close to 0
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = brunner_munzel(&x, &y, Alternative::TwoSided, None).unwrap();

        // Statistic should be close to 0, estimate close to 0.5
        assert!(result.statistic.abs() < 0.5);
        assert_relative_eq!(result.estimate, 0.5, epsilon = 0.1);
    }

    #[test]
    fn test_brunner_munzel_empty() {
        let x: Vec<f64> = vec![];
        let y = vec![1.0, 2.0, 3.0];

        assert!(brunner_munzel(&x, &y, Alternative::TwoSided, None).is_err());
    }

    #[test]
    fn test_brunner_munzel_insufficient() {
        let x = vec![1.0];
        let y = vec![1.0, 2.0, 3.0];

        assert!(brunner_munzel(&x, &y, Alternative::TwoSided, None).is_err());
    }
}
