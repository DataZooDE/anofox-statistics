//! Data reduction and transforms (#320): upstream anofox-regression
//! `reduction` (`Hmisc::spearman2`, `hoeffd`, `varclus`, `redun`) and
//! `transform` (`MASS::boxcox`, the Box–Cox transform and its inverse, and
//! `Hmisc::areg`, additive regression with optimal transformations, whose
//! stored model is rebuilt with `Areg::from_parts` to transform and predict).
//!
//! The wrappers only convert between the extension's column vectors and
//! upstream's matrices, and translate upstream's "the data cannot support a
//! result" errors (a constant column, a single level) into the degenerate
//! error kind so SQL returns NULL for them, as the sibling aggregates do.

use crate::{StatsError, StatsResult};
use anofox_regression::reduction::{
    self, Linkage, Node, RedunOptions, RedunVariable, Similarity, SimilarityTransform,
};
use anofox_regression::solvers::RegressionError;
pub use anofox_regression::transform::AregType;
use anofox_regression::transform::{self, Areg, AregBasis, AregOptions, BoxCoxOptions};
use faer::{Col, Mat};

/// Upstream input errors that describe degenerate data rather than invalid
/// arguments: reported as [`StatsError::InsufficientDataMsg`] (SQL NULL).
const DEGENERATE_MESSAGES: [&str; 5] = [
    "y is constant",
    "two or more levels",
    "could not be computed",
    "dissimilarities must be finite",
    // areg: a spline basis of a variable with fewer than 6 rows (upstream
    // wraps rcspline.eval's TooFewObservations in an InvalidInput string)
    "non-missing observations",
];

fn convert(e: RegressionError) -> StatsError {
    match e {
        RegressionError::InvalidInput(msg)
            if DEGENERATE_MESSAGES.iter().any(|m| msg.contains(m)) =>
        {
            StatsError::InsufficientDataMsg(msg)
        }
        other => StatsError::from(other),
    }
}

/// `n x p` matrix from `p` columns of equal length.
fn matrix(columns: &[Vec<f64>]) -> Mat<f64> {
    let n = columns.first().map_or(0, Vec::len);
    Mat::from_fn(n, columns.len(), |i, j| columns[j][i])
}

fn matrix_rows(m: &Mat<f64>) -> Vec<Vec<f64>> {
    (0..m.nrows())
        .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect())
        .collect()
}

// ------------------------------------------------------------- spearman2 --

/// Generalized Spearman ρ² (`Hmisc::spearman2`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spearman2Fit {
    pub rho2: f64,
    pub adjusted_rho2: f64,
    pub f_statistic: f64,
    pub df1: f64,
    pub df2: f64,
    pub p_value: f64,
    pub n: usize,
}

impl From<reduction::Spearman2> for Spearman2Fit {
    fn from(s: reduction::Spearman2) -> Self {
        Self {
            rho2: s.rho2,
            adjusted_rho2: s.adjusted_rho2,
            f_statistic: s.f,
            df1: s.df1,
            df2: s.df2,
            p_value: s.p_value,
            n: s.n,
        }
    }
}

/// `spearman2(x, y, p)` of a numeric predictor `x` with the response `y`.
pub fn spearman2(y: &[f64], x: &[f64], p: usize) -> StatsResult<Spearman2Fit> {
    reduction::spearman2(x, y, p)
        .map(Spearman2Fit::from)
        .map_err(convert)
}

/// `spearman2(factor(x), y)`: `x` holds level codes.
pub fn spearman2_categorical(y: &[f64], x: &[usize]) -> StatsResult<Spearman2Fit> {
    reduction::spearman2_categorical(x, y)
        .map(Spearman2Fit::from)
        .map_err(convert)
}

// ---------------------------------------------------------- Hoeffding's D --

/// Hoeffding's D of one pair (`hoeffd(x, y)`): `30 D` and its P-value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoeffdingFit {
    pub d: f64,
    pub p_value: f64,
    pub n: usize,
}

pub fn hoeffding_d(x: &[f64], y: &[f64]) -> StatsResult<HoeffdingFit> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch {
            y_len: y.len(),
            x_rows: x.len(),
        });
    }
    let m = matrix(&[x.to_vec(), y.to_vec()]);
    let h = reduction::hoeffding_d(&m).map_err(convert)?;
    Ok(HoeffdingFit {
        d: h.d[(0, 1)],
        p_value: h.p_value[(0, 1)],
        n: x.len(),
    })
}

// ---------------------------------------------------------------- Box–Cox --

/// Box–Cox profile (`MASS::boxcox`).
#[derive(Debug, Clone, PartialEq)]
pub struct BoxCoxFit {
    pub lambda: f64,
    pub max_log_likelihood: f64,
    /// NaN when the profile stays above the cut to the grid's end.
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub lambdas: Vec<f64>,
    pub log_likelihood: Vec<f64>,
    pub n_observations: usize,
}

/// Settings of [`box_cox`]; `None` keeps upstream's (MASS's) default.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BoxCoxSettings {
    pub lambdas: Option<Vec<f64>>,
    pub eps: Option<f64>,
    pub with_intercept: Option<bool>,
    pub confidence_level: Option<f64>,
}

/// `boxcox(lm(y ~ x))` with `x` given as columns (none: intercept only).
pub fn box_cox(y: &[f64], x: &[Vec<f64>], settings: &BoxCoxSettings) -> StatsResult<BoxCoxFit> {
    let n = y.len();
    let xm = Mat::from_fn(n, x.len(), |i, j| x[j][i]);
    let yc = Col::from_fn(n, |i| y[i]);
    let mut opts = BoxCoxOptions::default();
    if let Some(l) = &settings.lambdas {
        opts = opts.lambdas(l.clone());
    }
    if let Some(e) = settings.eps {
        opts = opts.eps(e);
    }
    if let Some(w) = settings.with_intercept {
        opts = opts.with_intercept(w);
    }
    if let Some(c) = settings.confidence_level {
        opts = opts.confidence_level(c);
    }
    let p = transform::box_cox(&xm, &yc, &opts).map_err(convert)?;
    Ok(BoxCoxFit {
        lambda: p.lambda_hat,
        max_log_likelihood: p.max_log_likelihood,
        ci_lower: p.ci_lower.unwrap_or(f64::NAN),
        ci_upper: p.ci_upper.unwrap_or(f64::NAN),
        lambdas: p.lambdas,
        log_likelihood: p.log_likelihood,
        n_observations: n,
    })
}

/// `(y^λ − 1)/λ`, `log y` at `λ = 0`.
pub fn box_cox_transform(y: f64, lambda: f64) -> f64 {
    transform::box_cox_transform(y, lambda)
}

/// `(λz + 1)^{1/λ}`, `exp z` at `λ = 0`.
pub fn box_cox_inverse(z: f64, lambda: f64) -> f64 {
    transform::box_cox_inverse(z, lambda)
}

// ---------------------------------------------------------------- varclus --

/// Similarity of [`varclus`] by name: `spearman`, `pearson`, `hoeffding`.
pub fn parse_similarity(name: &str) -> Option<Similarity> {
    match name {
        "spearman" => Some(Similarity::Spearman),
        "pearson" => Some(Similarity::Pearson),
        "hoeffding" => Some(Similarity::Hoeffding),
        _ => None,
    }
}

/// Correlation transform of [`varclus`] by name: `square`, `abs`, `none`.
pub fn parse_similarity_transform(name: &str) -> Option<SimilarityTransform> {
    match name {
        "square" => Some(SimilarityTransform::Square),
        "abs" => Some(SimilarityTransform::Abs),
        "none" => Some(SimilarityTransform::None),
        _ => None,
    }
}

/// Linkage by R's `hclust(method =)` name.
pub fn parse_linkage(name: &str) -> Option<Linkage> {
    match name {
        "complete" => Some(Linkage::Complete),
        "single" => Some(Linkage::Single),
        "average" => Some(Linkage::Average),
        "mcquitty" => Some(Linkage::Mcquitty),
        "median" => Some(Linkage::Median),
        "centroid" => Some(Linkage::Centroid),
        "ward.d" => Some(Linkage::WardD),
        "ward.d2" => Some(Linkage::WardD2),
        _ => None,
    }
}

/// Variable clustering (`Hmisc::varclus`).
#[derive(Debug, Clone, PartialEq)]
pub struct VarclusFit {
    /// `p x p` similarity matrix, row-major rows.
    pub similarity: Vec<Vec<f64>>,
    /// The `p − 1` merges in R's `hclust` convention: `-(j + 1)` for
    /// variable `j`, `k + 1` for the cluster formed at step `k`.
    pub merge: Vec<[i64; 2]>,
    pub height: Vec<f64>,
    /// Variables in dendrogram order, 0-based.
    pub order: Vec<usize>,
    /// Cluster of each variable after cutting at the requested height
    /// (0-based, numbered by first variable), if a height was given.
    pub clusters: Option<Vec<usize>>,
    pub n_observations: usize,
}

pub fn varclus(
    columns: &[Vec<f64>],
    similarity: Similarity,
    transform: SimilarityTransform,
    linkage: Linkage,
    cut_height: Option<f64>,
) -> StatsResult<VarclusFit> {
    let x = matrix(columns);
    let vc = reduction::varclus(&x, similarity, transform, linkage).map_err(convert)?;
    let r_node = |nd: Node| match nd {
        Node::Leaf(i) => -(i as i64 + 1),
        Node::Cluster(k) => k as i64 + 1,
    };
    Ok(VarclusFit {
        similarity: matrix_rows(&vc.sim),
        merge: vc
            .hclust
            .merge
            .iter()
            .map(|&(a, b)| [r_node(a), r_node(b)])
            .collect(),
        height: vc.hclust.height.clone(),
        order: vc.hclust.order.clone(),
        clusters: cut_height.map(|h| vc.hclust.cut_height(h)),
        n_observations: x.nrows(),
    })
}

// ------------------------------------------------------------------- areg --

/// The kind of an `areg` variable (`xtype` / `ytype`): `s`, `l`, `c`.
pub fn parse_areg_type(name: &str) -> Option<AregType> {
    match name.to_ascii_lowercase().as_str() {
        "s" | "spline" => Some(AregType::Spline),
        "l" | "linear" => Some(AregType::Linear),
        "c" | "categorical" => Some(AregType::Categorical),
        _ => None,
    }
}

/// The stored basis of one `areg` variable: `kind` 0 linear, 1 spline (its
/// knots in `values`), 2 categorical (its levels in `values`).
#[derive(Debug, Clone, PartialEq)]
pub struct StoredAregBasis {
    pub kind: u8,
    pub values: Vec<f64>,
}

impl StoredAregBasis {
    fn from_upstream(b: &AregBasis) -> Self {
        match b {
            AregBasis::Spline { knots } => Self {
                kind: 1,
                values: knots.clone(),
            },
            AregBasis::Categorical { levels } => Self {
                kind: 2,
                values: levels.clone(),
            },
            // Linear (and any basis added upstream, which has no parameters
            // the extension could store) is one untransformed column.
            _ => Self {
                kind: 0,
                values: Vec::new(),
            },
        }
    }

    fn to_upstream(&self) -> StatsResult<AregBasis> {
        match self.kind {
            0 => Ok(AregBasis::Linear),
            1 => Ok(AregBasis::Spline {
                knots: self.values.clone(),
            }),
            2 => Ok(AregBasis::Categorical {
                levels: self.values.clone(),
            }),
            k => Err(StatsError::InvalidInput(format!(
                "unknown areg basis kind {k} (0 linear, 1 spline, 2 categorical)"
            ))),
        }
    }
}

/// A fitted `Hmisc::areg` model.
#[derive(Debug, Clone, PartialEq)]
pub struct AregFit {
    /// Intercept, then the coefficients of the predictor bases (`xcoefficients`).
    pub x_coefficients: Vec<f64>,
    /// Intercept, then the coefficients of the response basis (`ycoefficients`).
    pub y_coefficients: Vec<f64>,
    pub r_squared: f64,
    /// Basis columns per predictor (`xdf`) and of the response (`ydf`).
    pub x_df: Vec<usize>,
    pub y_df: usize,
    /// Mean / median `|y − ŷ|`; `None` when the response transformation is
    /// not monotone (or categorical).
    pub mad: Option<f64>,
    pub med: Option<f64>,
    pub x_bases: Vec<StoredAregBasis>,
    pub y_basis: StoredAregBasis,
    /// `(min, max)` of the response.
    pub y_range: (f64, f64),
    pub n_observations: usize,
}

/// `Hmisc::areg(x, y, xtype, ytype, nk)` on the columns `x` (one per
/// predictor) and `y`; `x_types` `None` makes every predictor a spline.
pub fn fit_areg(
    y: &[f64],
    x: &[Vec<f64>],
    nk: usize,
    x_types: Option<Vec<AregType>>,
    y_type: AregType,
) -> StatsResult<AregFit> {
    if x.is_empty() {
        return Err(StatsError::InvalidInput(
            "areg needs at least one predictor".into(),
        ));
    }
    let opts = AregOptions::default()
        .nk(nk)
        .x_types(x_types)
        .y_type(y_type);
    let f = transform::areg(&matrix(x), y, &opts).map_err(convert)?;
    Ok(AregFit {
        x_coefficients: f.x_coefficients.iter().copied().collect(),
        y_coefficients: f.y_coefficients.iter().copied().collect(),
        r_squared: f.r_squared,
        x_df: f.x_df.clone(),
        y_df: f.y_df,
        mad: f.mad,
        med: f.med,
        x_bases: f
            .x_bases()
            .iter()
            .map(StoredAregBasis::from_upstream)
            .collect(),
        y_basis: StoredAregBasis::from_upstream(&f.y_basis()),
        y_range: f.y_range(),
        n_observations: y.len(),
    })
}

/// A stored `areg` model (the parts of [`AregFit`]).
#[derive(Debug, Clone)]
pub struct StoredAreg {
    pub x_bases: Vec<StoredAregBasis>,
    pub x_coefficients: Vec<f64>,
    pub y_basis: StoredAregBasis,
    pub y_coefficients: Vec<f64>,
    pub y_range: (f64, f64),
}

impl StoredAreg {
    /// Upstream's model rebuilt from the parts (`Areg::from_parts`).
    pub fn rebuild(&self) -> StatsResult<Areg> {
        let x_bases = self
            .x_bases
            .iter()
            .map(StoredAregBasis::to_upstream)
            .collect::<StatsResult<Vec<_>>>()?;
        let xc = &self.x_coefficients;
        let yc = &self.y_coefficients;
        Areg::from_parts(
            x_bases,
            Col::from_fn(xc.len(), |i| xc[i]),
            self.y_basis.to_upstream()?,
            Col::from_fn(yc.len(), |i| yc[i]),
            self.y_range,
        )
        .map_err(StatsError::from)
    }
}

/// The linear predictor (transformed-`y` scale) of each row of `x` (one
/// vector per row, one value per predictor).
pub fn areg_predict(model: &Areg, x: &[Vec<f64>]) -> StatsResult<Vec<f64>> {
    let p = model.x_df.len();
    if let Some(r) = x.iter().find(|r| r.len() != p) {
        return Err(StatsError::DimensionMismatchMsg(format!(
            "the model has {p} predictors but x has {} values",
            r.len()
        )));
    }
    Ok(model.predict(&Mat::from_fn(x.len(), p, |i, j| x[i][j])))
}

/// The transformation of the response at `y` (`ytrans`).
pub fn areg_transform_y(model: &Areg, y: &[f64]) -> Vec<f64> {
    model.transform_y(y)
}

/// The back-transformation of `t` to the `y` scale (`yinv`); `None` when the
/// transformation is not monotone over the fitted range or the response is
/// categorical.
pub fn areg_inverse_y(model: &Areg, t: &[f64]) -> Option<Vec<f64>> {
    model.inverse_y(t)
}

// ------------------------------------------------------------------ redun --

/// How a variable enters [`redun`] by name: `spline` (alias `numeric`,
/// `s`), `linear` (`l`) or `categorical` (`c`).
pub fn parse_redun_kind(name: &str) -> Option<RedunVariable> {
    match name {
        "spline" | "numeric" | "s" => Some(RedunVariable::Numeric),
        "linear" | "l" => Some(RedunVariable::Linear),
        "categorical" | "c" => Some(RedunVariable::Categorical),
        _ => None,
    }
}

/// Settings of [`redun`]; `None` keeps upstream's (Hmisc's) default.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RedunSettings {
    pub r2: Option<f64>,
    pub adjusted: Option<bool>,
    pub nk: Option<usize>,
    pub tlinear: Option<bool>,
    pub rank: Option<bool>,
    pub qrank: Option<bool>,
    pub allcat: Option<bool>,
}

/// Redundancy analysis (`Hmisc::redun`); variable indices are 0-based
/// columns of the input.
#[derive(Debug, Clone, PartialEq)]
pub struct RedunFit {
    pub kept: Vec<usize>,
    pub removed: Vec<usize>,
    pub r_squared: Vec<f64>,
    pub r_squared_later: Vec<Vec<f64>>,
    /// Variables analysed (all but `too_few`), aligned with `r_squared_first`.
    pub analysed: Vec<usize>,
    pub r_squared_first: Vec<f64>,
    pub too_few: Vec<usize>,
    pub n_observations: usize,
}

pub fn redun(
    columns: &[Vec<f64>],
    kinds: &[RedunVariable],
    settings: &RedunSettings,
) -> StatsResult<RedunFit> {
    let x = matrix(columns);
    let mut o = RedunOptions::default();
    if let Some(v) = settings.r2 {
        o = o.r2(v);
    }
    if let Some(v) = settings.adjusted {
        o = o.adjusted(v);
    }
    if let Some(v) = settings.nk {
        o = o.nk(v);
    }
    if let Some(v) = settings.tlinear {
        o = o.tlinear(v);
    }
    if let Some(v) = settings.rank {
        o = o.rank(v);
    }
    if let Some(v) = settings.qrank {
        o = o.qrank(v);
    }
    if let Some(v) = settings.allcat {
        o = o.allcat(v);
    }
    let r = reduction::redun(&x, kinds, &o).map_err(convert)?;
    let analysed = (0..columns.len())
        .filter(|j| !r.too_few.contains(j))
        .collect();
    Ok(RedunFit {
        kept: r.kept,
        removed: r.removed,
        r_squared: r.r_squared,
        r_squared_later: r.r_squared_later,
        analysed,
        r_squared_first: r.r_squared_first,
        too_few: r.too_few,
        n_observations: x.nrows(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> Vec<Vec<f64>> {
        let n = 30;
        let a: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.7).sin() + i as f64 * 0.05)
            .collect();
        let b: Vec<f64> = a
            .iter()
            .enumerate()
            .map(|(i, v)| v * 2.0 + ((i * 7) % 5) as f64 * 0.01)
            .collect();
        let c: Vec<f64> = (0..n).map(|i| ((i * 13) % 11) as f64).collect();
        vec![a, b, c]
    }

    #[test]
    fn spearman2_matches_upstream() {
        let d = data();
        let s = spearman2(&d[0], &d[2], 2).unwrap();
        let u = reduction::spearman2(&d[2], &d[0], 2).unwrap();
        assert_eq!(s.rho2, u.rho2);
        assert_eq!(s.df1, 2.0);
        let constant = vec![1.0; 30];
        assert!(matches!(
            spearman2(&constant, &d[0], 1),
            Err(StatsError::InsufficientDataMsg(_))
        ));
        assert!(matches!(
            spearman2(&d[0], &d[2], 3),
            Err(StatsError::InvalidInput(_))
        ));
        let codes: Vec<usize> = (0..30).map(|i| i % 3).collect();
        assert_eq!(spearman2_categorical(&d[0], &codes).unwrap().df1, 2.0);
        assert!(matches!(
            spearman2_categorical(&d[0], &[0; 30]),
            Err(StatsError::InsufficientDataMsg(_))
        ));
    }

    #[test]
    fn hoeffding_pair() {
        let d = data();
        let h = hoeffding_d(&d[0], &d[1]).unwrap();
        assert!(h.d > 0.5 && h.p_value < 0.01);
        assert!(hoeffding_d(&d[0][..4], &d[1][..4]).is_err());
    }

    #[test]
    fn varclus_and_redun() {
        let d = data();
        let vc = varclus(
            &d,
            Similarity::Spearman,
            SimilarityTransform::Square,
            Linkage::Complete,
            Some(0.3),
        )
        .unwrap();
        assert_eq!(vc.merge.len(), 2);
        assert_eq!(vc.merge[0], [-1, -2]);
        assert_eq!(vc.clusters.unwrap(), vec![0, 0, 1]);
        let r = redun(&d, &[RedunVariable::Numeric; 3], &RedunSettings::default()).unwrap();
        assert_eq!(r.removed.len(), 1);
        assert_eq!(r.analysed, vec![0, 1, 2]);
        assert!(parse_linkage("ward.d2").is_some() && parse_linkage("x").is_none());
    }

    #[test]
    fn box_cox_profile() {
        let y: Vec<f64> = (1..=40).map(|i| (1.0 + i as f64 / 10.0).powi(2)).collect();
        let x = vec![(1..=40).map(|i| i as f64).collect::<Vec<_>>()];
        let f = box_cox(&y, &x, &BoxCoxSettings::default()).unwrap();
        assert!((f.lambda - 0.5).abs() < 0.05);
        assert_eq!(f.lambdas.len(), 41);
        assert!(box_cox(&[-1.0, 2.0, 3.0], &[], &BoxCoxSettings::default()).is_err());
        let z = box_cox_transform(4.0, 0.5);
        assert!((box_cox_inverse(z, 0.5) - 4.0).abs() < 1e-12);
    }
}
