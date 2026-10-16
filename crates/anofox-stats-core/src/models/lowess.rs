//! LOWESS (Locally Weighted Scatterplot Smoothing) wrapper (#63)
//!
//! Cleveland's robust locally weighted regression, upstream's port of R's
//! `stats::lowess` (`anofox_regression::solvers::lowess::lowess`).

use crate::errors::{StatsError, StatsResult};
use crate::types::{LowessOptions, LowessResult};
use anofox_regression::solvers::lowess::{lowess, LowessOptions as UpstreamOptions};

/// Smooth `y` on `x` as R's `lowess(x, y, f = frac, iter = iterations, delta)`.
///
/// Rows with a non-finite `x` or `y` are dropped. Returns the curve at the
/// sorted `x` values of the remaining rows.
pub fn fit_lowess(x: &[f64], y: &[f64], options: &LowessOptions) -> StatsResult<LowessResult> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatchMsg(format!(
            "x has {} values but y has {}",
            x.len(),
            y.len()
        )));
    }
    if !(options.frac.is_finite() && options.frac > 0.0) {
        return Err(StatsError::InvalidValue {
            field: "frac",
            message: format!("frac must be finite and > 0, got {}", options.frac),
        });
    }
    if let Some(delta) = options.delta {
        if !(delta.is_finite() && delta >= 0.0) {
            return Err(StatsError::InvalidValue {
                field: "delta",
                message: format!("delta must be finite and >= 0, got {delta}"),
            });
        }
    }
    let rows: Vec<usize> = (0..x.len())
        .filter(|&i| x[i].is_finite() && y[i].is_finite())
        .collect();
    if rows.is_empty() {
        return Err(StatsError::NoValidData);
    }
    let xs: Vec<f64> = rows.iter().map(|&i| x[i]).collect();
    let ys: Vec<f64> = rows.iter().map(|&i| y[i]).collect();
    let fit = lowess(
        &xs,
        &ys,
        &UpstreamOptions {
            frac: options.frac,
            iterations: options.iterations,
            delta: options.delta,
        },
    )
    .map_err(StatsError::from)?;
    Ok(LowessResult {
        x: fit.x,
        fitted: fit.fitted,
        n_observations: rows.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lowess_sorts_and_smooths() {
        let x = vec![5.0, 1.0, 3.0, 2.0, 4.0, 6.0, 7.0, 8.0];
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
        let r = fit_lowess(&x, &y, &LowessOptions::default()).unwrap();
        assert_eq!(r.x, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        // A straight line is reproduced exactly by local linear fits.
        for (xi, fi) in r.x.iter().zip(&r.fitted) {
            assert!((fi - (2.0 * xi + 1.0)).abs() < 1e-10);
        }
        assert_eq!(r.n_observations, 8);
    }

    #[test]
    fn test_lowess_drops_non_finite_rows() {
        let x = vec![1.0, 2.0, f64::NAN, 4.0];
        let y = vec![1.0, f64::INFINITY, 3.0, 4.0];
        let r = fit_lowess(&x, &y, &LowessOptions::default()).unwrap();
        assert_eq!(r.x, vec![1.0, 4.0]);
    }

    #[test]
    fn test_lowess_invalid_options() {
        let x = vec![1.0, 2.0, 3.0];
        let bad = LowessOptions {
            frac: 0.0,
            ..Default::default()
        };
        assert!(fit_lowess(&x, &x, &bad).is_err());
        let bad = LowessOptions {
            delta: Some(-1.0),
            ..Default::default()
        };
        assert!(fit_lowess(&x, &x, &bad).is_err());
        assert!(fit_lowess(&x, &x[..2], &LowessOptions::default()).is_err());
        assert!(fit_lowess(&[f64::NAN], &[1.0], &LowessOptions::default()).is_err());
    }
}
