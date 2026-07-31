//! Design-matrix assembly and the surrounding data contract.
//!
//! This module owns everything the old per-family wrappers did *around* the fit,
//! not just the matrix build: input validation, dropping rows with non-finite
//! values, detecting and excluding constant columns, the minimum-observation rule,
//! offset extraction, and reconstructing the full-width coefficient vector with
//! `NaN` in the dropped slots.
//!
//! Keeping this in one place matters: a parity test that only compares the numeric
//! core would still pass while the SQL-visible behaviour regressed.

use crate::errors::{StatsError, StatsResult};
use crate::types::{PriorSpec, VcovType};
use faer::Mat;

use super::penalty::{Penalty, QuadraticPenalty};

/// How the engine treats columns that are constant over the retained rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstantColumnPolicy {
    /// Drop them from the fit and report `NaN` for their coefficient. This is what
    /// Poisson has always done.
    Drop,
    /// Keep them; the rank-deficiency handling in the solve zeroes them out. This is
    /// what the other five families have always done.
    Keep,
}

/// Everything the engine needs about the data, resolved once.
#[derive(Debug, Clone)]
pub struct Design {
    /// Design matrix over retained rows and retained columns, intercept column
    /// prepended when `fit_intercept` is set.
    pub matrix: Mat<f64>,
    /// Response over retained rows.
    pub y: Vec<f64>,
    /// Offset over retained rows, if one was requested.
    pub offset: Option<Vec<f64>>,
    /// Indices into the original rows that survived filtering.
    pub valid_rows: Vec<usize>,
    /// Indices into the original feature columns that survived the constant-column
    /// check, in order.
    pub retained_columns: Vec<usize>,
    /// Original number of feature columns, before dropping.
    pub n_features_original: usize,
    pub fit_intercept: bool,
}

impl Design {
    /// Number of parameters (retained columns plus intercept).
    pub fn n_params(&self) -> usize {
        self.matrix.ncols()
    }

    pub fn n_observations(&self) -> usize {
        self.matrix.nrows()
    }

    /// Expand a parameter-order vector back to the caller's feature order,
    /// inserting `NaN` for dropped columns. Returns `(features, intercept)`.
    pub fn expand(&self, params: &[f64]) -> (Vec<f64>, Option<f64>) {
        let offset = usize::from(self.fit_intercept);
        let intercept = if self.fit_intercept {
            params.first().copied()
        } else {
            None
        };

        let mut full = vec![f64::NAN; self.n_features_original];
        for (reduced, &orig) in self.retained_columns.iter().enumerate() {
            if let Some(&v) = params.get(reduced + offset) {
                full[orig] = v;
            }
        }
        (full, intercept)
    }

    /// Map per-parameter priors onto the design, given priors in the caller's
    /// feature order with an optional leading intercept entry.
    ///
    /// `priors` may be empty (no explicit priors). When non-empty it must have
    /// either `n_features` entries or `n_features + 1` when an intercept is fitted.
    pub fn build_penalty(
        &self,
        priors: &[PriorSpec],
        legacy_lambda: f64,
    ) -> StatsResult<Penalty> {
        let p = self.n_params();

        if priors.is_empty() {
            return Ok(if legacy_lambda > 0.0 {
                // `glm_lambda` has always been a uniform ridge that skips the intercept.
                Penalty::ridge(p, legacy_lambda, self.fit_intercept)
            } else {
                Penalty::none(p)
            });
        }

        let expected_with_intercept = self.n_features_original + usize::from(self.fit_intercept);
        if priors.len() != expected_with_intercept && priors.len() != self.n_features_original {
            return Err(StatsError::InvalidInput(format!(
                "expected {} priors (one per feature{}), got {}",
                expected_with_intercept,
                if self.fit_intercept {
                    " plus one for the intercept"
                } else {
                    ""
                },
                priors.len()
            )));
        }

        // Whether the caller supplied a leading intercept prior.
        let has_intercept_entry =
            self.fit_intercept && priors.len() == expected_with_intercept;

        let mut aligned = vec![PriorSpec::flat(); p];
        let param_offset = usize::from(self.fit_intercept);

        if has_intercept_entry {
            aligned[0] = priors[0];
        }
        let feature_priors = if has_intercept_entry {
            &priors[1..]
        } else {
            priors
        };

        for (reduced, &orig) in self.retained_columns.iter().enumerate() {
            aligned[reduced + param_offset] = feature_priors[orig];
        }

        let mut penalty = Penalty::from_priors(&aligned);

        // A legacy `glm_lambda` composes additively with explicit priors: both are
        // Gaussian precisions on the same coefficients.
        if legacy_lambda > 0.0 {
            let start = usize::from(self.fit_intercept);
            for j in start..p {
                let current = penalty.quadratic.precisions()[j];
                let loc = penalty.quadratic.locations()[j];
                penalty.quadratic.set(j, current + legacy_lambda, loc);
            }
        }

        Ok(penalty)
    }
}

/// Inputs to design assembly.
#[derive(Debug, Clone)]
pub struct DesignSpec<'a> {
    pub y: &'a [f64],
    pub x: &'a [Vec<f64>],
    pub fit_intercept: bool,
    /// 1-based index into `x` of a column to be used as an offset (added to the
    /// linear predictor with coefficient 1 and removed from the design). `None`
    /// means no offset. The value is used as-is; take logs upstream if the link
    /// requires it.
    pub offset_column: Option<usize>,
    pub constant_policy: ConstantColumnPolicy,
}

/// Validate inputs and assemble the design.
pub fn build(spec: &DesignSpec<'_>) -> StatsResult<Design> {
    let DesignSpec {
        y,
        x,
        fit_intercept,
        offset_column,
        constant_policy,
    } = *spec;

    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }
    let n_obs = y.len();
    for col in x.iter() {
        if col.len() != n_obs {
            return Err(StatsError::DimensionMismatch {
                y_len: n_obs,
                x_rows: col.len(),
            });
        }
    }

    // Resolve the offset column and split it out of the feature set.
    let offset_idx = match offset_column {
        Some(one_based) => {
            if one_based == 0 || one_based > x.len() {
                return Err(StatsError::InvalidValue {
                    field: "offset",
                    message: format!(
                        "offset must be a 1-based index into x (1..={}), got {one_based}",
                        x.len()
                    ),
                });
            }
            Some(one_based - 1)
        }
        None => None,
    };

    let feature_indices: Vec<usize> = (0..x.len()).filter(|i| Some(*i) != offset_idx).collect();
    let n_features_original = feature_indices.len();
    if n_features_original == 0 && !fit_intercept {
        return Err(StatsError::InsufficientData {
            rows: n_obs,
            cols: 0,
        });
    }

    // Drop rows with any non-finite value, in y, in a feature, or in the offset.
    let valid_rows: Vec<usize> = (0..n_obs)
        .filter(|&i| {
            y[i].is_finite() && x.iter().all(|col| col[i].is_finite())
        })
        .collect();
    if valid_rows.is_empty() {
        return Err(StatsError::NoValidData);
    }
    let n_valid = valid_rows.len();

    // Constant-column detection over the retained rows only.
    let is_constant: Vec<bool> = feature_indices
        .iter()
        .map(|&ci| {
            let first = x[ci][valid_rows[0]];
            valid_rows
                .iter()
                .all(|&i| (x[ci][i] - first).abs() < 1e-10)
        })
        .collect();

    let retained_columns: Vec<usize> = match constant_policy {
        ConstantColumnPolicy::Drop => (0..n_features_original)
            .filter(|&j| !is_constant[j])
            .collect(),
        ConstantColumnPolicy::Keep => (0..n_features_original).collect(),
    };

    let n_effective = retained_columns.len();
    let min_obs = n_effective + usize::from(fit_intercept);

    if n_effective == 0 && !fit_intercept {
        return Err(StatsError::InsufficientData {
            rows: n_valid,
            cols: n_features_original,
        });
    }
    if n_valid < min_obs.max(1) {
        return Err(StatsError::InsufficientData {
            rows: n_valid,
            cols: n_features_original,
        });
    }

    let n_params = n_effective + usize::from(fit_intercept);
    let int_off = usize::from(fit_intercept);
    let matrix = Mat::from_fn(n_valid, n_params, |i, j| {
        if fit_intercept && j == 0 {
            1.0
        } else {
            let ci = feature_indices[retained_columns[j - int_off]];
            x[ci][valid_rows[i]]
        }
    });

    let y_valid: Vec<f64> = valid_rows.iter().map(|&i| y[i]).collect();
    let offset = offset_idx.map(|oi| valid_rows.iter().map(|&i| x[oi][i]).collect());

    Ok(Design {
        matrix,
        y: y_valid,
        offset,
        valid_rows,
        retained_columns,
        n_features_original,
        fit_intercept,
    })
}

/// Resolve the covariance type, defaulting to `Laplace`.
pub fn resolve_vcov(requested: VcovType) -> VcovType {
    requested
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_with_non_finite_values_are_dropped() {
        let y = vec![1.0, f64::NAN, 3.0, 4.0];
        let x = vec![vec![1.0, 2.0, f64::INFINITY, 4.0]];
        let d = build(&DesignSpec {
            y: &y,
            x: &x,
            fit_intercept: true,
            offset_column: None,
            constant_policy: ConstantColumnPolicy::Keep,
        })
        .unwrap();

        assert_eq!(d.valid_rows, vec![0, 3]);
        assert_eq!(d.y, vec![1.0, 4.0]);
    }

    #[test]
    fn constant_columns_are_dropped_under_the_drop_policy() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = vec![
            vec![7.0; 5],                     // constant
            vec![1.0, 2.0, 3.0, 4.0, 5.0],    // informative
        ];
        let d = build(&DesignSpec {
            y: &y,
            x: &x,
            fit_intercept: true,
            offset_column: None,
            constant_policy: ConstantColumnPolicy::Drop,
        })
        .unwrap();

        assert_eq!(d.retained_columns, vec![1]);
        assert_eq!(d.n_params(), 2); // intercept + one retained column
    }

    #[test]
    fn expand_puts_nan_in_the_dropped_slots() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = vec![vec![7.0; 5], vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let d = build(&DesignSpec {
            y: &y,
            x: &x,
            fit_intercept: true,
            offset_column: None,
            constant_policy: ConstantColumnPolicy::Drop,
        })
        .unwrap();

        let (features, intercept) = d.expand(&[0.5, 1.25]);
        assert_eq!(intercept, Some(0.5));
        assert!(features[0].is_nan());
        assert_eq!(features[1], 1.25);
    }

    #[test]
    fn offset_column_leaves_the_feature_set() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![
            vec![1.0, 2.0, 3.0, 4.0],  // feature
            vec![10.0, 20.0, 30.0, 40.0], // offset
        ];
        let d = build(&DesignSpec {
            y: &y,
            x: &x,
            fit_intercept: true,
            offset_column: Some(2),
            constant_policy: ConstantColumnPolicy::Keep,
        })
        .unwrap();

        assert_eq!(d.n_features_original, 1);
        assert_eq!(d.n_params(), 2);
        assert_eq!(d.offset.unwrap(), vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn out_of_range_offset_is_rejected() {
        let y = vec![1.0, 2.0];
        let x = vec![vec![1.0, 2.0]];
        let err = build(&DesignSpec {
            y: &y,
            x: &x,
            fit_intercept: true,
            offset_column: Some(9),
            constant_policy: ConstantColumnPolicy::Keep,
        });
        assert!(matches!(err, Err(StatsError::InvalidValue { .. })));
    }

    #[test]
    fn all_non_finite_input_reports_no_valid_data() {
        let y = vec![f64::NAN, f64::NAN];
        let x = vec![vec![1.0, 2.0]];
        let err = build(&DesignSpec {
            y: &y,
            x: &x,
            fit_intercept: true,
            offset_column: None,
            constant_policy: ConstantColumnPolicy::Keep,
        });
        assert!(matches!(err, Err(StatsError::NoValidData)));
    }

    #[test]
    fn priors_align_across_dropped_columns() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = vec![vec![7.0; 5], vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let d = build(&DesignSpec {
            y: &y,
            x: &x,
            fit_intercept: true,
            offset_column: None,
            constant_policy: ConstantColumnPolicy::Drop,
        })
        .unwrap();

        // Feature-order priors: the dropped column's prior must not leak onto the
        // retained one.
        let priors = vec![PriorSpec::normal(0.0, 0.1), PriorSpec::normal(0.0, 2.0)];
        let pen = d.build_penalty(&priors, 0.0).unwrap();

        assert_eq!(pen.n_params(), 2);
        assert_eq!(pen.quadratic.precisions()[0], 0.0); // intercept, flat
        assert!((pen.quadratic.precisions()[1] - 0.25).abs() < 1e-12); // 1/2^2
    }

    #[test]
    fn legacy_lambda_composes_with_explicit_priors() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let d = build(&DesignSpec {
            y: &y,
            x: &x,
            fit_intercept: true,
            offset_column: None,
            constant_policy: ConstantColumnPolicy::Keep,
        })
        .unwrap();

        let pen = d
            .build_penalty(&[PriorSpec::normal(0.0, 1.0)], 3.0)
            .unwrap();
        // 1/1^2 from the prior + 3.0 from glm_lambda.
        assert!((pen.quadratic.precisions()[1] - 4.0).abs() < 1e-12);
        // Intercept stays unpenalized.
        assert_eq!(pen.quadratic.precisions()[0], 0.0);
    }

    #[test]
    fn wrong_prior_count_is_rejected() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let d = build(&DesignSpec {
            y: &y,
            x: &x,
            fit_intercept: true,
            offset_column: None,
            constant_policy: ConstantColumnPolicy::Keep,
        })
        .unwrap();

        let err = d.build_penalty(&vec![PriorSpec::flat(); 7], 0.0);
        assert!(matches!(err, Err(StatsError::InvalidInput(_))));
    }

    #[test]
    fn legacy_lambda_alone_is_a_ridge_that_skips_the_intercept() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let d = build(&DesignSpec {
            y: &y,
            x: &x,
            fit_intercept: true,
            offset_column: None,
            constant_policy: ConstantColumnPolicy::Keep,
        })
        .unwrap();

        let pen = d.build_penalty(&[], 2.5).unwrap();
        assert_eq!(pen.quadratic.precisions(), &[0.0, 2.5]);
    }
}
