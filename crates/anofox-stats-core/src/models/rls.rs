//! Recursive Least Squares (RLS) implementation
//!
//! RLS is an online learning algorithm that updates regression coefficients
//! incrementally as new data points arrive. It's useful for streaming data
//! and adaptive filtering applications.

use crate::errors::{StatsError, StatsResult};

/// RLS model state that can be updated incrementally
#[derive(Debug, Clone)]
pub struct RlsState {
    /// Current coefficient estimates (including intercept as first element if fit_intercept)
    pub coefficients: Vec<f64>,
    /// Inverse covariance matrix (P matrix)
    pub p_matrix: Vec<Vec<f64>>,
    /// Forgetting factor (λ), typically 0.95-1.0
    pub forgetting_factor: f64,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Number of features (excluding intercept)
    pub n_features: usize,
    /// Number of observations processed
    pub n_observations: usize,
}

/// Options for RLS initialization
#[derive(Debug, Clone)]
pub struct RlsOptions {
    /// Forgetting factor (λ), typically 0.95-1.0
    /// 1.0 = no forgetting (standard RLS)
    /// <1.0 = exponential forgetting of old data
    pub forgetting_factor: f64,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Initial value for diagonal of P matrix (controls initial uncertainty)
    pub initial_p_diagonal: f64,
}

impl Default for RlsOptions {
    fn default() -> Self {
        Self {
            forgetting_factor: 1.0,
            fit_intercept: true,
            initial_p_diagonal: 100.0, // Lower default for better numerical stability
        }
    }
}

impl RlsState {
    /// Initialize RLS state for a given number of features
    #[allow(clippy::needless_range_loop)]
    pub fn new(n_features: usize, options: &RlsOptions) -> StatsResult<Self> {
        if n_features == 0 {
            return Err(StatsError::InvalidInput("n_features must be > 0".into()));
        }

        if options.forgetting_factor <= 0.0 || options.forgetting_factor > 1.0 {
            return Err(StatsError::InvalidInput(
                "forgetting_factor must be in (0, 1]".into(),
            ));
        }

        if options.initial_p_diagonal <= 0.0 {
            return Err(StatsError::InvalidInput(
                "initial_p_diagonal must be > 0".into(),
            ));
        }

        // Dimension includes intercept if fitting
        let dim = if options.fit_intercept {
            n_features + 1
        } else {
            n_features
        };

        // Initialize coefficients to zero
        let coefficients = vec![0.0; dim];

        // Initialize P matrix as diagonal with initial_p_diagonal
        let mut p_matrix = vec![vec![0.0; dim]; dim];
        for i in 0..dim {
            p_matrix[i][i] = options.initial_p_diagonal;
        }

        Ok(Self {
            coefficients,
            p_matrix,
            forgetting_factor: options.forgetting_factor,
            fit_intercept: options.fit_intercept,
            n_features,
            n_observations: 0,
        })
    }

    /// Update the RLS state with a new observation
    ///
    /// # Arguments
    /// * `y` - Response value
    /// * `x` - Feature values (length must equal n_features)
    ///
    /// # Returns
    /// The prediction error (y - y_hat) before the update
    #[allow(clippy::needless_range_loop)]
    pub fn update(&mut self, y: f64, x: &[f64]) -> StatsResult<f64> {
        if x.len() != self.n_features {
            return Err(StatsError::DimensionMismatchMsg(format!(
                "Expected {} features, got {}",
                self.n_features,
                x.len()
            )));
        }

        // Skip NaN values
        if y.is_nan() || x.iter().any(|v| v.is_nan()) {
            return Ok(f64::NAN);
        }

        let dim = self.coefficients.len();
        let lambda = self.forgetting_factor;

        // Build input vector (with intercept if needed)
        let x_vec: Vec<f64> = if self.fit_intercept {
            std::iter::once(1.0).chain(x.iter().copied()).collect()
        } else {
            x.to_vec()
        };

        // Compute prediction before update
        let y_hat: f64 = x_vec
            .iter()
            .zip(&self.coefficients)
            .map(|(xi, bi)| xi * bi)
            .sum();
        let error = y - y_hat;

        // Compute P * x
        let mut p_x = vec![0.0; dim];
        for i in 0..dim {
            for j in 0..dim {
                p_x[i] += self.p_matrix[i][j] * x_vec[j];
            }
        }

        // Compute x' * P * x
        let x_p_x: f64 = x_vec.iter().zip(&p_x).map(|(xi, pxi)| xi * pxi).sum();

        // Compute gain vector k = P * x / (λ + x' * P * x)
        let denom = lambda + x_p_x;
        let k: Vec<f64> = p_x.iter().map(|pxi| pxi / denom).collect();

        // Update coefficients: β = β + k * error
        for i in 0..dim {
            self.coefficients[i] += k[i] * error;
        }

        // Update P matrix: P = (P - k * x' * P) / λ
        // First compute k * x'
        let mut k_x_t = vec![vec![0.0; dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                k_x_t[i][j] = k[i] * x_vec[j];
            }
        }

        // P_new = (P - k * x' * P) / λ
        for i in 0..dim {
            for j in 0..dim {
                let mut kxp_ij = 0.0;
                for l in 0..dim {
                    kxp_ij += k_x_t[i][l] * self.p_matrix[l][j];
                }
                self.p_matrix[i][j] = (self.p_matrix[i][j] - kxp_ij) / lambda;
            }
        }

        self.n_observations += 1;

        Ok(error)
    }

    /// Get current coefficients (excluding intercept)
    pub fn get_coefficients(&self) -> Vec<f64> {
        if self.fit_intercept {
            self.coefficients[1..].to_vec()
        } else {
            self.coefficients.clone()
        }
    }

    /// Get intercept (None if not fitting intercept)
    pub fn get_intercept(&self) -> Option<f64> {
        if self.fit_intercept {
            Some(self.coefficients[0])
        } else {
            None
        }
    }

    /// Make a prediction for new feature values
    pub fn predict(&self, x: &[f64]) -> StatsResult<f64> {
        if x.len() != self.n_features {
            return Err(StatsError::DimensionMismatchMsg(format!(
                "Expected {} features, got {}",
                self.n_features,
                x.len()
            )));
        }

        let x_vec: Vec<f64> = if self.fit_intercept {
            std::iter::once(1.0).chain(x.iter().copied()).collect()
        } else {
            x.to_vec()
        };

        let y_hat: f64 = x_vec
            .iter()
            .zip(&self.coefficients)
            .map(|(xi, bi)| xi * bi)
            .sum();

        Ok(y_hat)
    }
}

/// Fit RLS on a batch of data (processes each observation sequentially)
///
/// Returns the final RLS state after processing all observations
pub fn fit_rls(y: &[f64], x: &[Vec<f64>], options: &RlsOptions) -> StatsResult<RlsState> {
    let n_features = x.len();

    if n_features == 0 {
        return Err(StatsError::EmptyInput { field: "x" });
    }

    let n_obs = x[0].len();
    if n_obs == 0 {
        return Err(StatsError::EmptyInput { field: "x[0]" });
    }

    if y.len() != n_obs {
        return Err(StatsError::DimensionMismatch {
            y_len: y.len(),
            x_rows: n_obs,
        });
    }

    // Filter out rows with NaN values
    let valid_indices: Vec<usize> = (0..n_obs)
        .filter(|&i| {
            !y[i].is_nan()
                && !y[i].is_infinite()
                && x.iter()
                    .all(|col| !col[i].is_nan() && !col[i].is_infinite())
        })
        .collect();

    if valid_indices.is_empty() {
        return Err(StatsError::NoValidData);
    }

    // Detect zero-variance (constant) columns
    let is_constant_column: Vec<bool> = x
        .iter()
        .map(|col| {
            if valid_indices.is_empty() {
                return true;
            }
            let first_val = col[valid_indices[0]];
            valid_indices
                .iter()
                .all(|&i| (col[i] - first_val).abs() < 1e-10)
        })
        .collect();

    // Get non-constant column indices
    let non_constant_indices: Vec<usize> = is_constant_column
        .iter()
        .enumerate()
        .filter_map(|(i, &is_const)| if !is_const { Some(i) } else { None })
        .collect();

    let n_effective_features = non_constant_indices.len();

    // If ALL columns are constant, return intercept-only model
    if n_effective_features == 0 {
        if !options.fit_intercept {
            return Err(StatsError::InsufficientData {
                rows: valid_indices.len(),
                cols: n_features,
            });
        }
        // Intercept-only model: compute mean of y as intercept
        let y_mean = valid_indices.iter().map(|&i| y[i]).sum::<f64>() / valid_indices.len() as f64;

        // Create a state with NaN for all feature coefficients
        let mut state = RlsState {
            coefficients: vec![y_mean], // Just intercept
            p_matrix: vec![vec![0.0]],  // Minimal P matrix
            forgetting_factor: options.forgetting_factor,
            fit_intercept: true,
            n_features,
            n_observations: valid_indices.len(),
        };
        // Expand coefficients to include NaN for each feature
        for _ in 0..n_features {
            state.coefficients.push(f64::NAN);
        }
        return Ok(state);
    }

    // Initialize RLS state with only effective features
    let mut state = RlsState::new(n_effective_features, options)?;

    // Process each valid observation using only non-constant features
    for &i in valid_indices.iter() {
        let xi: Vec<f64> = non_constant_indices.iter().map(|&j| x[j][i]).collect();
        state.update(y[i], &xi)?;
    }

    // Reconstruct full coefficients with NaN for constant columns
    let reduced_coefs = state.get_coefficients();
    let mut full_coefficients = vec![f64::NAN; n_features];
    for (reduced_idx, &orig_idx) in non_constant_indices.iter().enumerate() {
        full_coefficients[orig_idx] = reduced_coefs[reduced_idx];
    }

    // Update state with full coefficient set
    if options.fit_intercept {
        let intercept = state.coefficients[0];
        state.coefficients = std::iter::once(intercept)
            .chain(full_coefficients)
            .collect();
    } else {
        state.coefficients = full_coefficients;
    }
    state.n_features = n_features;
    state.n_observations = valid_indices.len();

    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rls_initialization() {
        let options = RlsOptions::default();
        let state = RlsState::new(3, &options).unwrap();

        assert_eq!(state.coefficients.len(), 4); // 3 features + intercept
        assert_eq!(state.n_features, 3);
        assert_eq!(state.n_observations, 0);
    }

    #[test]
    fn test_rls_simple_update() {
        let options = RlsOptions {
            forgetting_factor: 1.0,
            fit_intercept: true,
            initial_p_diagonal: 100.0, // Smaller initial uncertainty for faster convergence
        };
        let mut state = RlsState::new(1, &options).unwrap();

        // Update with more observations from y = 2x + 1
        for i in 1..=20 {
            let x = i as f64;
            let y = 2.0 * x + 1.0;
            state.update(y, &[x]).unwrap();
        }

        assert_eq!(state.n_observations, 20);

        // Check prediction at x=25 should be approximately 51 (2*25 + 1)
        let pred = state.predict(&[25.0]).unwrap();
        assert!((pred - 51.0).abs() < 1.0);
    }

    #[test]
    fn test_rls_batch_fit() {
        // More data points for better convergence
        let y: Vec<f64> = (1..=20).map(|i| 2.0 * i as f64 + 1.0).collect();
        let x = vec![(1..=20).map(|i| i as f64).collect()];

        let options = RlsOptions {
            forgetting_factor: 1.0,
            fit_intercept: true,
            initial_p_diagonal: 100.0,
        };
        let state = fit_rls(&y, &x, &options).unwrap();

        assert_eq!(state.n_observations, 20);

        // Should fit y = 2x + 1 well
        let pred = state.predict(&[25.0]).unwrap();
        assert!((pred - 51.0).abs() < 1.0);
    }

    #[test]
    fn test_rls_forgetting_factor() {
        let options = RlsOptions {
            forgetting_factor: 0.95,
            fit_intercept: true,
            initial_p_diagonal: 1000.0,
        };
        let mut state = RlsState::new(1, &options).unwrap();

        // Update with observations
        for i in 0..10 {
            let x = i as f64;
            let y = 2.0 * x + 1.0;
            state.update(y, &[x]).unwrap();
        }

        assert_eq!(state.n_observations, 10);
    }

    #[test]
    fn test_rls_no_intercept() {
        let options = RlsOptions {
            forgetting_factor: 1.0,
            fit_intercept: false,
            initial_p_diagonal: 1000.0,
        };
        let mut state = RlsState::new(1, &options).unwrap();

        // y = 3x (no intercept)
        state.update(3.0, &[1.0]).unwrap();
        state.update(6.0, &[2.0]).unwrap();
        state.update(9.0, &[3.0]).unwrap();

        assert!(state.get_intercept().is_none());

        let coef = state.get_coefficients();
        assert!((coef[0] - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_rls_multivariate() {
        let options = RlsOptions {
            forgetting_factor: 1.0,
            fit_intercept: true,
            initial_p_diagonal: 100.0,
        };
        let mut state = RlsState::new(2, &options).unwrap();

        // y = x1 + 2*x2 + 0.5, more data points for convergence
        for i in 1..=20 {
            let x1 = i as f64;
            let x2 = (i * 2) as f64;
            let y = x1 + 2.0 * x2 + 0.5;
            state.update(y, &[x1, x2]).unwrap();
        }

        let pred = state.predict(&[5.0, 10.0]).unwrap();
        // Should be approximately 5 + 20 + 0.5 = 25.5
        assert!((pred - 25.5).abs() < 2.0); // Allow tolerance for multivariate
    }
}
