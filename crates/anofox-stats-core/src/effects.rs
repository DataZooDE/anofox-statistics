//! Partial effects, contrasts, effect grids and nomograms (#319).
//!
//! Thin wrapper around upstream `anofox_regression::effects::ModelSpec`
//! (`rms::datadist`, `summary.rms`, `contrast.rms`, `Predict`, `nomogram`).
//! The wrapper translates the extension's term list (linear, rcs, lsp,
//! categorical, interactions) and the training data of the predictors into a
//! `ModelSpec`, validates a fitted model's coefficients / covariance against
//! the specification's design, and passes everything to upstream.
//!
//! The `datadist` limits (quartiles, medians, prediction ranges, modes) are
//! computed by upstream from the predictor data: `ModelSpec` has no public
//! constructor that takes stored limits, so every operation receives the data.

use crate::splines::{SplineBasis, SplineKind, SplineSpec};
use crate::{StatsError, StatsResult};
pub use anofox_regression::effects::ContrastEstimate;
use anofox_regression::effects::{Critical, ModelSpec};
use faer::{Col, Mat};

/// How a term enters the design.
#[derive(Debug, Clone, PartialEq)]
pub enum TermKind {
    /// One linear column.
    Linear,
    /// Restricted cubic spline with these knots and `rcspline.eval` norm.
    Rcs { knots: Vec<f64>, norm: i32 },
    /// Linear spline with these knots.
    Lsp { knots: Vec<f64> },
    /// Categorical with treatment contrasts. `levels`: the expected levels
    /// (sorted, distinct); `None` takes them from the data.
    Categorical { levels: Option<Vec<f64>> },
    /// Interaction of the predictors `a` and `b` (indices among the
    /// non-interaction terms); `restricted` is `rms` `%ia%`.
    Interaction {
        a: usize,
        b: usize,
        restricted: bool,
    },
}

impl TermKind {
    fn is_interaction(&self) -> bool {
        matches!(self, TermKind::Interaction { .. })
    }
}

/// A fitted model: coefficients and covariance in the design's column order.
#[derive(Debug, Clone)]
pub struct FittedModel {
    /// Coefficients, intercept first when `intercept`.
    pub beta: Vec<f64>,
    /// Covariance, row-major `beta.len() x beta.len()`; may be empty for the
    /// nomogram.
    pub vcov: Vec<f64>,
    pub intercept: bool,
    /// Residual degrees of freedom of a t reference (OLS); NaN or `<= 0`: the
    /// standard normal (GLMs).
    pub df: f64,
    /// Confidence level in (0, 1).
    pub level: f64,
}

/// One `summary.rms` row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EffectRow {
    /// Index of the predictor among the non-interaction terms.
    pub predictor: usize,
    pub low: f64,
    pub high: f64,
    pub effect: ContrastEstimate,
}

/// A predictor setting: (predictor index, value).
pub type Setting = Vec<(usize, f64)>;

/// The second pair `(a2, b2)` of a double contrast.
pub type SecondPair<'a> = (&'a [(usize, f64)], &'a [(usize, f64)]);

/// One nomogram axis.
#[derive(Debug, Clone, PartialEq)]
pub struct NomogramAxis {
    pub values: Vec<f64>,
    pub xbeta: Vec<f64>,
    pub points: Vec<f64>,
}

/// A nomogram.
#[derive(Debug, Clone, PartialEq)]
pub struct Nomogram {
    pub axes: Vec<NomogramAxis>,
    pub scale: f64,
    pub intercept: f64,
    pub total_points: Vec<f64>,
    pub lp_at: Vec<f64>,
    pub lp_points: Vec<f64>,
    /// Ticks of the function axis (empty without a function).
    pub fun_at: Vec<f64>,
    pub fun_points: Vec<f64>,
}

/// Function of a nomogram's extra axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NomogramFun {
    None,
    /// The logistic function (`fun = plogis`): a risk axis.
    Plogis,
    /// `exp` (odds, rate or hazard ratio scale).
    Exp,
}

fn invalid(msg: String) -> StatsError {
    StatsError::InvalidInput(msg)
}

/// Sorted distinct values.
fn distinct(x: &[f64]) -> Vec<f64> {
    let mut v = x.to_vec();
    v.sort_by(f64::total_cmp);
    v.dedup();
    v
}

/// A `ModelSpec` with its term list.
#[derive(Debug, Clone)]
pub struct Effects {
    spec: ModelSpec,
    /// The non-interaction terms in order.
    predictors: Vec<TermKind>,
    /// Levels of each predictor (categorical) as used by the spec.
    levels: Vec<Option<Vec<f64>>>,
    /// Number of design columns (no intercept).
    n_columns: usize,
}

impl Effects {
    /// Build the specification from the terms and the training data of the
    /// predictors (`columns[k]` for the `k`-th non-interaction term, every
    /// column of the same length, all values finite).
    ///
    /// With `levels_from_terms`, a categorical term takes its levels from the
    /// term (required) and the data only need to be finite: this is the design
    /// of new rows (`effects_design`). Otherwise the levels come from the data
    /// and must equal the term's levels when the term gives them.
    pub fn new(
        terms: &[TermKind],
        columns: &[Vec<f64>],
        levels_from_terms: bool,
    ) -> StatsResult<Self> {
        let predictors: Vec<TermKind> = terms
            .iter()
            .filter(|t| !t.is_interaction())
            .cloned()
            .collect();
        if predictors.is_empty() {
            return Err(invalid("the terms contain no predictor".into()));
        }
        if columns.len() != predictors.len() {
            return Err(invalid(format!(
                "{} predictor terms but {} data columns",
                predictors.len(),
                columns.len()
            )));
        }
        let n = columns[0].len();
        if columns.iter().any(|c| c.len() != n) {
            return Err(invalid(
                "the predictor columns have different lengths".into(),
            ));
        }
        if !levels_from_terms && n == 0 {
            return Err(StatsError::NoValidData);
        }
        if columns.iter().any(|c| c.iter().any(|v| !v.is_finite())) {
            return Err(invalid("the predictor data must be finite".into()));
        }
        let mut spec = ModelSpec::new();
        let mut levels = Vec::with_capacity(predictors.len());
        for (k, (term, x)) in predictors.iter().zip(columns).enumerate() {
            // Upstream names the predictor in its errors (e.g. a value that is
            // not a level): use the 1-based position the extension reports.
            let name = format!("predictor {}", k + 1);
            // Data for the datadist limits; the design of new rows needs none,
            // a placeholder value keeps upstream's limits defined.
            let placeholder = [0.0];
            let data: &[f64] = if x.is_empty() { &placeholder } else { x };
            match term {
                TermKind::Linear => {
                    spec.linear(&name, data);
                    levels.push(None);
                }
                TermKind::Rcs { knots, norm } => {
                    let basis = SplineBasis::new(&spline(SplineKind::Rcs, knots, *norm))?;
                    let SplineBasis::Rcs(r) = basis else {
                        unreachable!()
                    };
                    spec.rcs(&name, r, data);
                    levels.push(None);
                }
                TermKind::Lsp { knots } => {
                    let basis = SplineBasis::new(&spline(SplineKind::Lsp, knots, 2))?;
                    let SplineBasis::Lsp(l) = basis else {
                        unreachable!()
                    };
                    spec.lsp(&name, l, data);
                    levels.push(None);
                }
                TermKind::Categorical { levels: given } => {
                    let given = given.as_ref().map(|g| distinct(g));
                    if let Some(g) = &given {
                        if g.is_empty() || g.iter().any(|v| !v.is_finite()) {
                            return Err(invalid(format!(
                                "categorical term {} has empty or non-finite levels",
                                k + 1
                            )));
                        }
                    }
                    if levels_from_terms {
                        let Some(g) = given else {
                            return Err(invalid(format!(
                                "categorical term {} needs its levels to build a design row",
                                k + 1
                            )));
                        };
                        if let Some(v) = x
                            .iter()
                            .find(|v| g.binary_search_by(|l| l.total_cmp(v)).is_err())
                        {
                            return Err(invalid(format!(
                                "value {} is not a level of categorical term {}",
                                v,
                                k + 1
                            )));
                        }
                        spec.categorical(&name, &g);
                        levels.push(Some(g));
                    } else {
                        let found = distinct(x);
                        if let Some(g) = given {
                            if g != found {
                                return Err(invalid(format!(
                                    "the levels of categorical term {} in the data ({:?}) differ from the term's levels ({:?})",
                                    k + 1,
                                    found,
                                    g
                                )));
                            }
                        }
                        spec.categorical(&name, x);
                        levels.push(Some(found));
                    }
                }
                TermKind::Interaction { .. } => unreachable!(),
            }
        }
        for term in terms {
            if let TermKind::Interaction { a, b, restricted } = *term {
                if a >= predictors.len() || b >= predictors.len() {
                    return Err(invalid(
                        "an interaction refers to an unknown predictor".into(),
                    ));
                }
                if a == b {
                    return Err(invalid(
                        "an interaction needs two different predictors".into(),
                    ));
                }
                if restricted {
                    spec.restricted_interaction(a, b);
                } else {
                    spec.interaction(a, b);
                }
            }
        }
        // Validate the whole specification once (ModelSpec::row expects a
        // valid one) and count its columns.
        let probe: Vec<Vec<f64>> = (0..predictors.len())
            .map(|k| vec![spec.predictors()[k].limits.adjust_to])
            .collect();
        let refs: Vec<&[f64]> = probe.iter().map(Vec::as_slice).collect();
        let n_columns = spec
            .design(&refs)
            .map_err(|e| invalid(e.to_string()))?
            .matrix()
            .ncols();
        Ok(Self {
            spec,
            predictors,
            levels,
            n_columns,
        })
    }

    /// Number of design columns (without the intercept).
    pub fn n_columns(&self) -> usize {
        self.n_columns
    }

    /// Number of predictors (non-interaction terms).
    pub fn n_predictors(&self) -> usize {
        self.predictors.len()
    }

    /// Whether predictor `k` is categorical.
    pub fn is_categorical(&self, k: usize) -> bool {
        matches!(self.predictors.get(k), Some(TermKind::Categorical { .. }))
    }

    /// The `datadist` adjust-to value of predictor `k` (median or mode).
    pub fn adjust_to(&self, k: usize) -> f64 {
        self.spec.predictors()[k].limits.adjust_to
    }

    /// Design rows of `columns` (one per predictor, finite values), row-major
    /// `n x n_columns`, main effects in predictor order then the interactions.
    pub fn design(&self, columns: &[Vec<f64>]) -> StatsResult<Vec<f64>> {
        if columns.len() != self.predictors.len() {
            return Err(invalid("wrong number of predictor columns".into()));
        }
        for (k, x) in columns.iter().enumerate() {
            self.check_values(k, x)?;
        }
        let refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
        let m = self
            .spec
            .design(&refs)
            .map_err(|e| invalid(e.to_string()))?
            .matrix();
        let mut out = Vec::with_capacity(m.nrows() * m.ncols());
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                out.push(m[(i, j)]);
            }
        }
        Ok(out)
    }

    fn check_finite(&self, k: usize, values: &[f64]) -> StatsResult<()> {
        if let Some(v) = values.iter().find(|v| !v.is_finite()) {
            return Err(invalid(format!(
                "value {} of predictor {} is not finite",
                v,
                k + 1
            )));
        }
        Ok(())
    }

    /// Finite values that are levels of a categorical predictor. Only the
    /// design needs this check: upstream's `row` (contrasts, effect grids)
    /// rejects a value that is not a level itself.
    fn check_values(&self, k: usize, values: &[f64]) -> StatsResult<()> {
        self.check_finite(k, values)?;
        if let Some(levels) = &self.levels[k] {
            if let Some(v) = values
                .iter()
                .find(|v| levels.binary_search_by(|l| l.total_cmp(v)).is_err())
            {
                return Err(invalid(format!(
                    "value {} is not a level of categorical predictor {} (levels {:?})",
                    v,
                    k + 1,
                    levels
                )));
            }
        }
        Ok(())
    }

    fn check_setting(&self, s: &[(usize, f64)]) -> StatsResult<()> {
        for &(k, v) in s {
            if k >= self.predictors.len() {
                return Err(invalid(format!(
                    "setting refers to unknown predictor {}",
                    k + 1
                )));
            }
            self.check_finite(k, &[v])?;
        }
        Ok(())
    }

    fn model(
        &self,
        m: &FittedModel,
        need_vcov: bool,
    ) -> StatsResult<(Col<f64>, Mat<f64>, Critical)> {
        let p = self.n_columns + usize::from(m.intercept);
        if m.beta.len() != p {
            return Err(invalid(format!(
                "the model has {} coefficients{} but the terms give {} design columns{}",
                m.beta.len(),
                if m.intercept {
                    " (with the intercept)"
                } else {
                    ""
                },
                self.n_columns,
                if m.intercept {
                    " plus the intercept"
                } else {
                    ""
                }
            )));
        }
        if need_vcov && m.vcov.len() != p * p {
            return Err(invalid(format!(
                "the covariance has {} values, expected {} x {}",
                m.vcov.len(),
                p,
                p
            )));
        }
        if !(m.level > 0.0 && m.level < 1.0) {
            return Err(invalid(format!("level must be in (0, 1), got {}", m.level)));
        }
        let beta = Col::from_fn(p, |i| m.beta[i]);
        let vcov = if m.vcov.len() == p * p {
            Mat::from_fn(p, p, |i, j| m.vcov[i * p + j])
        } else {
            Mat::zeros(p, p)
        };
        let crit = if m.df.is_finite() && m.df > 0.0 {
            Critical::StudentT(m.df)
        } else {
            Critical::Normal
        };
        Ok((beta, vcov, crit))
    }

    /// `summary.rms`.
    pub fn summary(&self, m: &FittedModel) -> StatsResult<Vec<EffectRow>> {
        let (beta, vcov, crit) = self.model(m, true)?;
        Ok(self
            .spec
            .summary(&beta, &vcov, m.intercept, crit, m.level)
            .into_iter()
            .map(|r| EffectRow {
                predictor: r.predictor,
                low: r.low,
                high: r.high,
                effect: r.effect,
            })
            .collect())
    }

    /// `contrast.rms(fit, a, b)`, or the double difference with `a2`, `b2`.
    pub fn contrast(
        &self,
        m: &FittedModel,
        a: &[(usize, f64)],
        b: &[(usize, f64)],
        second: Option<SecondPair<'_>>,
    ) -> StatsResult<ContrastEstimate> {
        let (beta, vcov, crit) = self.model(m, true)?;
        self.check_setting(a)?;
        self.check_setting(b)?;
        match second {
            None => self
                .spec
                .contrast(&beta, &vcov, m.intercept, a, b, crit, m.level),
            Some((a2, b2)) => {
                self.check_setting(a2)?;
                self.check_setting(b2)?;
                self.spec
                    .double_contrast(&beta, &vcov, m.intercept, a, b, a2, b2, crit, m.level)
            }
        }
        .map_err(StatsError::from)
    }

    /// `Predict(fit, x_k)`: the linear predictor over `values` of predictor
    /// `k` (default: upstream's 200-point grid; the levels for a categorical
    /// predictor), others at their adjust-to values. With `simultaneous` the
    /// band is simultaneous over the grid and its critical value is returned;
    /// otherwise the critical value is NaN.
    pub fn predict_grid(
        &self,
        m: &FittedModel,
        k: usize,
        values: Option<Vec<f64>>,
        simultaneous: bool,
    ) -> StatsResult<(Vec<f64>, Vec<ContrastEstimate>, f64)> {
        let (beta, vcov, crit) = self.model(m, true)?;
        if k >= self.predictors.len() {
            return Err(invalid(format!("unknown predictor {}", k + 1)));
        }
        let values = match values {
            Some(v) => {
                if v.is_empty() {
                    return Err(invalid("values must not be empty".into()));
                }
                self.check_finite(k, &v)?;
                Some(v)
            }
            // Upstream's default: its 200-point grid, the levels of a
            // categorical predictor.
            None => None,
        };
        if simultaneous {
            self.spec
                .predict_grid_simultaneous(&beta, &vcov, m.intercept, k, values, crit, m.level)
                .map_err(StatsError::from)
        } else {
            let (v, e) =
                self.spec
                    .predict_grid(&beta, &vcov, m.intercept, k, values, crit, m.level)?;
            Ok((v, e, f64::NAN))
        }
    }

    /// `rms::nomogram` of a main-effects model; `linear_predictors` are the
    /// fit's linear predictors (intercept included).
    pub fn nomogram(
        &self,
        m: &FittedModel,
        linear_predictors: &[f64],
        maxscale: f64,
        nint: usize,
        fun: NomogramFun,
    ) -> StatsResult<Nomogram> {
        let (beta, _, _) = self.model(m, false)?;
        if linear_predictors.is_empty() || linear_predictors.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "the linear predictors must be finite and not empty".into(),
            ));
        }
        if !(maxscale.is_finite() && maxscale > 0.0) {
            return Err(invalid(format!(
                "maxscale must be positive, got {}",
                maxscale
            )));
        }
        if !(1..=1000).contains(&nint) {
            return Err(invalid(format!(
                "nint must be between 1 and 1000, got {}",
                nint
            )));
        }
        let none: Vec<Option<Vec<f64>>> = vec![None; self.predictors.len()];
        let nom = self
            .spec
            .nomogram(&beta, m.intercept, linear_predictors, &none, maxscale, nint)
            .map_err(|e| invalid(e.to_string()))?;
        let (fun_at, fun_points) = match fun {
            NomogramFun::None => (Vec::new(), Vec::new()),
            NomogramFun::Plogis => nom.function_axis(|v| 1.0 / (1.0 + (-v).exp()), nint),
            NomogramFun::Exp => nom.function_axis(f64::exp, nint),
        };
        Ok(Nomogram {
            axes: nom
                .axes
                .iter()
                .map(|a| NomogramAxis {
                    values: a.values.clone(),
                    xbeta: a.xbeta.clone(),
                    points: a.points.clone(),
                })
                .collect(),
            scale: nom.scale,
            intercept: nom.intercept,
            total_points: nom.total_points.clone(),
            lp_at: nom.lp_at.clone(),
            lp_points: nom.lp_points.clone(),
            fun_at,
            fun_points,
        })
    }
}

fn spline(kind: SplineKind, knots: &[f64], norm: i32) -> SplineSpec {
    SplineSpec {
        kind,
        knots: knots.to_vec(),
        norm,
        degree: 3,
        extrapolation: crate::splines::Extrapolation::Constant,
        include_bias: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn data() -> (Vec<TermKind>, Vec<Vec<f64>>) {
        let x: Vec<f64> = (0..40).map(|i| f64::from(i) * 0.5).collect();
        let g: Vec<f64> = (0..40).map(|i| f64::from(i % 3 + 1)).collect();
        (
            vec![TermKind::Linear, TermKind::Categorical { levels: None }],
            vec![x, g],
        )
    }

    fn model(beta: Vec<f64>) -> FittedModel {
        let p = beta.len();
        let mut vcov = vec![0.0; p * p];
        for i in 0..p {
            vcov[i * p + i] = 0.01;
        }
        FittedModel {
            beta,
            vcov,
            intercept: true,
            df: f64::NAN,
            level: 0.95,
        }
    }

    #[test]
    fn summary_of_a_linear_and_a_factor() {
        let (terms, cols) = data();
        let e = Effects::new(&terms, &cols, false).unwrap();
        assert_eq!(e.n_columns(), 3);
        let rows = e.summary(&model(vec![1.0, 2.0, 0.5, -0.5])).unwrap();
        assert_eq!(rows.len(), 3);
        // IQR of x: quantiles 0.25 / 0.75 of 0..19.5
        assert_relative_eq!(
            rows[0].effect.estimate,
            2.0 * (rows[0].high - rows[0].low),
            epsilon = 1e-12
        );
        assert_eq!(rows[1].predictor, 1);
    }

    #[test]
    fn design_needs_levels_and_checks_values() {
        let terms = vec![
            TermKind::Linear,
            TermKind::Categorical {
                levels: Some(vec![3.0, 1.0, 2.0]),
            },
            TermKind::Interaction {
                a: 0,
                b: 1,
                restricted: false,
            },
        ];
        let e = Effects::new(&terms, &[vec![2.0], vec![3.0]], true).unwrap();
        assert_eq!(e.n_columns(), 5);
        assert_eq!(
            e.design(&[vec![2.0], vec![3.0]]).unwrap(),
            vec![2.0, 0.0, 1.0, 0.0, 2.0]
        );
        assert!(Effects::new(&terms, &[vec![2.0], vec![4.0]], true).is_err());
        let no_levels = vec![TermKind::Linear, TermKind::Categorical { levels: None }];
        assert!(Effects::new(&no_levels, &[vec![1.0], vec![1.0]], true).is_err());
    }

    #[test]
    fn model_shape_and_settings_are_validated() {
        let (terms, cols) = data();
        let e = Effects::new(&terms, &cols, false).unwrap();
        assert!(e.summary(&model(vec![1.0, 2.0])).is_err());
        let m = model(vec![1.0, 2.0, 0.5, -0.5]);
        assert!(e.contrast(&m, &[(1, 5.0)], &[(1, 1.0)], None).is_err());
        let c = e.contrast(&m, &[(1, 3.0)], &[(1, 2.0)], None).unwrap();
        assert_relative_eq!(c.estimate, -1.0, epsilon = 1e-12);
        let (v, est, crit) = e.predict_grid(&m, 1, None, false).unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
        assert_eq!(est.len(), 3);
        assert!(crit.is_nan());
    }

    #[test]
    fn levels_must_match_the_data() {
        let terms = vec![TermKind::Categorical {
            levels: Some(vec![1.0, 2.0]),
        }];
        assert!(Effects::new(&terms, &[vec![1.0, 2.0, 3.0]], false).is_err());
        assert!(Effects::new(&terms, &[vec![1.0, 2.0, 2.0]], false).is_ok());
    }
}
