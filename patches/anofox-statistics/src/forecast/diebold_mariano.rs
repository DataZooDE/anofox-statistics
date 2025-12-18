use crate::error::{Result, StatError};
use crate::parametric::Alternative;
use statrs::distribution::{ContinuousCDF, Normal};

/// Loss function for Diebold-Mariano test
#[derive(Debug, Clone, Copy)]
pub enum LossFunction {
    /// Squared error loss: (e)^2
    SquaredError,
    /// Absolute error loss: |e|
    AbsoluteError,
}

/// Variance estimator for Diebold-Mariano test
#[derive(Debug, Clone, Copy, Default)]
pub enum VarEstimator {
    /// Autocorrelation function (ACF) estimator (default)
    /// Uses unweighted autocovariances up to lag h-1
    #[default]
    Acf,
    /// Bartlett kernel estimator
    /// Uses Bartlett weights to ensure positive variance estimate
    Bartlett,
}

/// Result of Diebold-Mariano test
#[derive(Debug, Clone)]
pub struct DMResult {
    /// The DM test statistic
    pub statistic: f64,
    /// The p-value
    pub p_value: f64,
    /// Forecast horizon used
    pub horizon: usize,
    /// Loss function used
    pub loss_function: LossFunction,
    /// Variance estimator used
    pub varestimator: VarEstimator,
    /// Alternative hypothesis tested
    pub alternative: Alternative,
}

/// Perform the Diebold-Mariano test for comparing forecast accuracy.
///
/// Tests the null hypothesis that two forecasts have equal predictive accuracy.
/// Implementation matches R's forecast::dm.test function.
///
/// # Arguments
/// * `e1` - Forecast errors from model 1
/// * `e2` - Forecast errors from model 2
/// * `loss` - Loss function to use (SquaredError or AbsoluteError)
/// * `h` - Forecast horizon (for variance adjustment, h=1 for 1-step ahead)
/// * `alternative` - Alternative hypothesis (TwoSided, Less, Greater)
/// * `varestimator` - Variance estimator (Acf or Bartlett)
///
/// # Returns
/// * `DMResult` containing test statistic and p-value
///
/// # References
/// * Diebold, F.X. and Mariano, R.S. (1995) "Comparing Predictive Accuracy"
pub fn diebold_mariano(
    e1: &[f64],
    e2: &[f64],
    loss: LossFunction,
    h: usize,
    alternative: Alternative,
    varestimator: VarEstimator,
) -> Result<DMResult> {
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

    // Compute loss differentials: d_t = g(e1_t) - g(e2_t)
    let d: Vec<f64> = e1
        .iter()
        .zip(e2.iter())
        .map(|(e1_t, e2_t)| {
            let l1 = apply_loss(*e1_t, loss);
            let l2 = apply_loss(*e2_t, loss);
            l1 - l2
        })
        .collect();

    let n_f = n as f64;

    // Mean of loss differentials
    let d_bar: f64 = d.iter().sum::<f64>() / n_f;

    // Compute variance using R's approach:
    // d.cov <- acf(d, lag.max=h-1, type="covariance")$acf
    // d.var <- sum(c(d.cov[1], 2*d.cov[-1])) / n
    let var_d_bar = match varestimator {
        VarEstimator::Acf => dm_variance_acf(&d, h),
        VarEstimator::Bartlett => dm_variance_bartlett(&d, h),
    };

    // Raw DM statistic
    let dm_raw = if var_d_bar > 1e-30 {
        d_bar / var_d_bar.sqrt()
    } else {
        0.0
    };

    // Apply Harvey, Leybourne, and Newbold (1997) small-sample correction
    // This improves the finite-sample properties of the test
    let h_f = h as f64;
    let correction = ((n_f + 1.0 - 2.0 * h_f + h_f * (h_f - 1.0) / n_f) / n_f).sqrt();
    let dm_stat = dm_raw * correction;

    // Compute p-value based on alternative hypothesis
    let normal = Normal::new(0.0, 1.0).unwrap();
    let p_value = match alternative {
        Alternative::TwoSided => 2.0 * (1.0 - normal.cdf(dm_stat.abs())),
        Alternative::Less => normal.cdf(dm_stat),
        Alternative::Greater => 1.0 - normal.cdf(dm_stat),
    };

    Ok(DMResult {
        statistic: dm_stat,
        p_value,
        horizon: h,
        loss_function: loss,
        varestimator,
        alternative,
    })
}

/// Apply loss function to forecast error
fn apply_loss(e: f64, loss: LossFunction) -> f64 {
    match loss {
        LossFunction::SquaredError => e * e,
        LossFunction::AbsoluteError => e.abs(),
    }
}

/// Compute variance of the mean for DM test using ACF approach
/// This matches forecast::dm.test variance calculation with varestimator="acf"
fn dm_variance_acf(d: &[f64], h: usize) -> f64 {
    let n = d.len();
    let n_f = n as f64;

    // Mean of d
    let d_bar: f64 = d.iter().sum::<f64>() / n_f;

    // Centered series
    let d_centered: Vec<f64> = d.iter().map(|x| x - d_bar).collect();

    // Compute autocovariances (matching R's acf with type="covariance")
    // R's acf uses division by n (not n-k)
    let max_lag = if h > 0 { h - 1 } else { 0 };

    // Lag 0 autocovariance (variance * n / n = variance)
    let gamma_0: f64 = d_centered.iter().map(|x| x * x).sum::<f64>() / n_f;

    // Sum autocovariances: gamma_0 + 2 * sum(gamma_k for k=1..h-1)
    let mut acov_sum = gamma_0;

    for k in 1..=max_lag {
        if k >= n {
            break;
        }
        // Autocovariance at lag k (R's style: divide by n)
        let gamma_k: f64 = d_centered
            .iter()
            .skip(k)
            .zip(d_centered.iter())
            .map(|(d_t, d_t_k)| d_t * d_t_k)
            .sum::<f64>()
            / n_f;

        acov_sum += 2.0 * gamma_k;
    }

    // Variance of the mean = sum(autocovariances) / n
    acov_sum / n_f
}

/// Compute variance of the mean for DM test using Bartlett kernel
/// This matches forecast::dm.test variance calculation with varestimator="bartlett"
/// Uses Bartlett weights to ensure positive variance estimate
///
/// R's implementation: d.var = (d.cov[1] + 2 * sum((1 - k/h) * d.cov[k+1] for k in 1..h-1)) / n
/// where d.cov is computed with lag.max = h-1
fn dm_variance_bartlett(d: &[f64], h: usize) -> f64 {
    // For h=1, Bartlett is identical to ACF (R uses same code path)
    if h == 1 {
        return dm_variance_acf(d, h);
    }

    let n = d.len();
    let n_f = n as f64;
    let h_f = h as f64;

    // Mean of d
    let d_bar: f64 = d.iter().sum::<f64>() / n_f;

    // Centered series
    let d_centered: Vec<f64> = d.iter().map(|x| x - d_bar).collect();

    // Same max_lag as ACF: h-1
    let max_lag = h - 1;

    // Lag 0 autocovariance
    let gamma_0: f64 = d_centered.iter().map(|x| x * x).sum::<f64>() / n_f;

    // Sum autocovariances with Bartlett weights: gamma_0 + 2 * sum(w_k * gamma_k)
    // Bartlett weight: w_k = 1 - k/h (where k goes from 1 to h-1)
    let mut acov_sum = gamma_0;

    for k in 1..=max_lag {
        if k >= n {
            break;
        }
        // Bartlett weight: (1 - k/h)
        let w_k = 1.0 - (k as f64) / h_f;

        // Autocovariance at lag k (R's style: divide by n)
        let gamma_k: f64 = d_centered
            .iter()
            .skip(k)
            .zip(d_centered.iter())
            .map(|(d_t, d_t_k)| d_t * d_t_k)
            .sum::<f64>()
            / n_f;

        acov_sum += 2.0 * w_k * gamma_k;
    }

    // Variance of the mean = sum(weighted autocovariances) / n
    acov_sum / n_f
}
