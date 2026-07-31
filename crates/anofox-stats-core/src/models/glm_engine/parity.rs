//! Parity and correctness gate for the engine.
//!
//! # Why this is not simply "compare against upstream everywhere"
//!
//! Two upstream defects were found while building this gate, and both bound what
//! upstream can serve as a reference for:
//!
//! 1. **Back-permutation of the pivoted QR is inverted.** Upstream undoes the
//!    column pivoting with `perm.inverse().arrays().0` where the forward array is
//!    required. The two agree whenever the pivot order is an involution — which
//!    covers every 2-column design and orders like `[2,1,0]` — so single-feature
//!    fits are correct. For a genuine cycle such as `[1,2,0]` the coefficient
//!    vector comes back rotated, IRLS is fed a wrong step, and the fit diverges.
//!    A plain 3-column Poisson design reproduces it.
//! 2. **The convergence criterion monitors the deviance alone.** That is right for
//!    an unpenalized fit but wrong for a MAP estimate: under a strong prior the
//!    deviance legitimately rises as the mode is pulled toward the prior location,
//!    so the criterion never settles. This one is not reachable through upstream's
//!    own API (it has no prior support), but it bites the engine directly, which is
//!    why `irls::fit_irls` monitors `deviance + 2 * penalty` instead.
//!
//! The engine addresses both (see `normal_eq::solve_weighted_ls_qr` and
//! `irls::fit_irls`). Consequently:
//!
//! * **Upstream parity** is asserted on single-feature designs, where upstream is
//!   sound. This is the real "engine is a faithful superset" gate.
//! * **Multi-feature correctness** is asserted against independently computed
//!   reference values (a plain NumPy IRLS, cross-checked against the closed-form
//!   algebra in `laplace.rs`), because upstream cannot fit those at all.
//! * Defect 1 is pinned by a test so the finding does not quietly evaporate if the
//!   dependency is bumped.
//!
//! `aic` is deliberately excluded from upstream comparison: the engine computes a
//! real per-family log-likelihood where upstream uses `-deviance / 2`.

use super::*;
use crate::models::glm_engine::design::ConstantColumnPolicy;
use crate::types::PriorSpec;
use anofox_regression::prelude::*;
use faer::{Col, Mat};

const TOL: f64 = 1e-10;
/// Tolerance against the NumPy reference values. Those were iterated far past the
/// engine's `1e-8` stopping rule, so agreement is limited by where the engine stops,
/// not by the reference.
const REF_TOL: f64 = 1e-7;

fn to_faer(y: &[f64], x: &[Vec<f64>]) -> (Col<f64>, Mat<f64>) {
    let n = y.len();
    let p = x.len();
    (
        Col::from_fn(n, |i| y[i]),
        Mat::from_fn(n, p, |i, j| x[j][i]),
    )
}

fn assert_close(label: &str, got: f64, want: f64, tol: f64) {
    assert!(
        (got - want).abs() < tol,
        "{label}: got {got}, want {want} (diff {})",
        (got - want).abs()
    );
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Single-feature count data. `x` spans 0..3 in steps of 1/3, `y` is a rounded
/// log-linear mean with a deterministic wobble so the fit is not degenerate.
fn count_1f() -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = 50;
    let x: Vec<f64> = (0..n).map(|i| (i % 10) as f64 / 3.0).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| ((0.6 + 0.25 * x[i]).exp() + ((i * 13) % 4) as f64 * 0.3).round())
        .collect();
    (y, vec![x])
}

/// Single-feature strictly-positive continuous data, for Gamma and Tweedie.
fn positive_1f() -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = 50;
    let x: Vec<f64> = (0..n).map(|i| (i % 10) as f64 / 3.0).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| (0.5 + 0.2 * x[i]).exp() * (1.0 + (i % 5) as f64 * 0.05))
        .collect();
    (y, vec![x])
}

/// Single-feature binary data.
fn binary_1f() -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = 50;
    let x: Vec<f64> = (0..n).map(|i| (i % 12) as f64 / 4.0 - 1.0).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| f64::from(u8::from(0.8 * x[i] + ((i % 3) as f64 - 1.0) * 0.5 > 0.0)))
        .collect();
    (y, vec![x])
}

/// Three-column count fixture. Upstream cannot fit this (defect 1).
fn count_2f() -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = 60;
    let x1: Vec<f64> = (0..n).map(|i| (i % 10) as f64 / 3.0).collect();
    let x2: Vec<f64> = (0..n).map(|i| ((i * 7) % 5) as f64 - 2.0).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            ((0.6 + 0.25 * x1[i] - 0.15 * x2[i]).exp() + ((i * 13) % 4) as f64 * 0.3).round()
        })
        .collect();
    (y, vec![x1, x2])
}

fn binary_2f() -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = 80;
    let x1: Vec<f64> = (0..n).map(|i| (i % 12) as f64 / 4.0 - 1.0).collect();
    let x2: Vec<f64> = (0..n).map(|i| ((i * 5) % 7) as f64 / 3.0).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            f64::from(u8::from(
                0.4 * x1[i] + 0.3 * x2[i] + ((i % 3) as f64 - 1.0) * 0.5 > 0.0,
            ))
        })
        .collect();
    (y, vec![x1, x2])
}

fn poisson_opts(inference: bool) -> EngineOptions {
    EngineOptions {
        compute_inference: inference,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Upstream parity: single-feature designs, where upstream is sound.
// ---------------------------------------------------------------------------

#[test]
fn poisson_matches_upstream_on_a_sound_design() {
    let (y, x) = count_1f();
    let (y_col, x_mat) = to_faer(&y, &x);

    let upstream = PoissonRegressor::log()
        .with_intercept(true)
        .max_iterations(100)
        .tolerance(1e-10)
        .compute_inference(true)
        .confidence_level(0.95)
        .build()
        .fit(&x_mat, &y_col)
        .expect("upstream poisson fit");

    let fit = fit(
        &PoissonFamily::log(),
        &y,
        &x,
        &poisson_opts(true),
        DispersionRule::PearsonFlooredAtOne,
        |_| LogLikKind::Poisson,
    )
    .expect("engine poisson fit");

    let up = upstream.result();
    assert_close("intercept", fit.irls.beta[0], up.intercept.unwrap(), TOL);
    assert_close("slope", fit.irls.beta[1], up.coefficients[0], TOL);
    assert_close("deviance", fit.irls.deviance, upstream.deviance, TOL);
    assert_close(
        "null_deviance",
        fit.irls.null_deviance,
        upstream.null_deviance,
        TOL,
    );
    assert_close("dispersion", fit.dispersion, upstream.dispersion, TOL);

    let inf = fit.inference.as_ref().unwrap();
    assert_close(
        "intercept se",
        inf.std_errors[0],
        up.intercept_std_error.unwrap(),
        TOL,
    );
    assert_close(
        "slope se",
        inf.std_errors[1],
        up.std_errors.as_ref().unwrap()[0],
        TOL,
    );
}

#[test]
fn binomial_matches_upstream_on_a_sound_design() {
    let (y, x) = binary_1f();
    let (y_col, x_mat) = to_faer(&y, &x);

    let upstream = BinomialRegressor::builder()
        .link(BinomialLink::Logit)
        .with_intercept(true)
        .max_iterations(100)
        .tolerance(1e-10)
        .build()
        .fit(&x_mat, &y_col)
        .expect("upstream binomial fit");

    let fit = fit(
        &BinomialFamily::logistic(),
        &y,
        &x,
        &poisson_opts(false),
        DispersionRule::Fixed,
        |_| LogLikKind::Binomial,
    )
    .expect("engine binomial fit");

    let up = upstream.result();
    assert_close("intercept", fit.irls.beta[0], up.intercept.unwrap(), 1e-6);
    assert_close("slope", fit.irls.beta[1], up.coefficients[0], 1e-6);
    assert_close("deviance", fit.irls.deviance, upstream.deviance, 1e-6);
}

#[test]
fn tweedie_matches_upstream_on_a_sound_design() {
    let (y, x) = positive_1f();
    let (y_col, x_mat) = to_faer(&y, &x);

    let upstream = TweedieRegressor::builder()
        .var_power(1.5)
        .link_power(0.0)
        .with_intercept(true)
        .max_iterations(100)
        .tolerance(1e-10)
        .build()
        .fit(&x_mat, &y_col)
        .expect("upstream tweedie fit");

    let fit = fit(
        &TweedieFamily::new(1.5, 0.0),
        &y,
        &x,
        &poisson_opts(false),
        DispersionRule::Pearson,
        |phi| LogLikKind::Tweedie {
            power: 1.5,
            dispersion: phi,
        },
    )
    .expect("engine tweedie fit");

    let up = upstream.result();
    assert_close("intercept", fit.irls.beta[0], up.intercept.unwrap(), TOL);
    assert_close("slope", fit.irls.beta[1], up.coefficients[0], TOL);
    assert_close("deviance", fit.irls.deviance, upstream.deviance, TOL);
}

#[test]
fn gamma_matches_upstream_on_a_sound_design() {
    let (y, x) = positive_1f();
    let (y_col, x_mat) = to_faer(&y, &x);

    let upstream = GammaRegressor::builder()
        .with_intercept(true)
        .max_iterations(100)
        .tolerance(1e-10)
        .build()
        .fit(&x_mat, &y_col)
        .expect("upstream gamma fit");

    let fit = fit(
        &TweedieFamily::new(2.0, 0.0),
        &y,
        &x,
        &poisson_opts(false),
        DispersionRule::Pearson,
        |phi| LogLikKind::Gamma { dispersion: phi },
    )
    .expect("engine gamma fit");

    let inner = upstream.inner();
    let up = inner.result();
    assert_close("intercept", fit.irls.beta[0], up.intercept.unwrap(), TOL);
    assert_close("slope", fit.irls.beta[1], up.coefficients[0], TOL);
    assert_close("deviance", fit.irls.deviance, inner.deviance, TOL);
}

#[test]
fn negbinomial_matches_upstream_on_a_sound_design() {
    let (y, x) = count_1f();
    let (y_col, x_mat) = to_faer(&y, &x);

    let upstream = NegativeBinomialRegressor::builder()
        .with_intercept(true)
        .max_iterations(100)
        .tolerance(1e-10)
        .build()
        .fit(&x_mat, &y_col)
        .expect("upstream negbin fit");

    // Upstream estimates theta; feed its estimate back so the comparison is
    // like-for-like.
    let theta = upstream.dispersion;
    let fit = fit(
        &NegativeBinomialFamily::new(theta),
        &y,
        &x,
        &poisson_opts(false),
        DispersionRule::Given(theta),
        |_| LogLikKind::NegativeBinomial { theta },
    )
    .expect("engine negbin fit");

    // Loose by design. Upstream re-estimates theta *during* IRLS, so the theta it
    // finally reports is not the one its last iteration used; feeding that value to
    // the engine (which treats theta as fixed) cannot reproduce it exactly. The
    // precision claim for Negative Binomial lives in the reference test below.
    let up = upstream.result();
    assert_close("intercept", fit.irls.beta[0], up.intercept.unwrap(), 5e-3);
    assert_close("slope", fit.irls.beta[1], up.coefficients[0], 5e-3);
}

#[test]
fn no_intercept_fit_matches_upstream() {
    let (y, x) = count_1f();
    let (y_col, x_mat) = to_faer(&y, &x);

    let upstream = PoissonRegressor::log()
        .with_intercept(false)
        .max_iterations(100)
        .tolerance(1e-10)
        .build()
        .fit(&x_mat, &y_col)
        .expect("upstream fit");

    let opts = EngineOptions {
        fit_intercept: false,
        ..Default::default()
    };
    let fit = fit(
        &PoissonFamily::log(),
        &y,
        &x,
        &opts,
        DispersionRule::PearsonFlooredAtOne,
        |_| LogLikKind::Poisson,
    )
    .expect("engine fit");

    assert_close(
        "slope",
        fit.irls.beta[0],
        upstream.result().coefficients[0],
        TOL,
    );
}

// ---------------------------------------------------------------------------
// Independent reference values, for designs upstream cannot fit.
//
// Produced by a plain NumPy IRLS using the same family definitions, run to
// convergence with a direct solve (no pivoting), so it shares no code with either
// implementation under test.
// ---------------------------------------------------------------------------

#[test]
fn poisson_single_feature_matches_the_reference() {
    let (y, x) = count_1f();
    let fit = fit(
        &PoissonFamily::log(),
        &y,
        &x,
        &poisson_opts(true),
        DispersionRule::PearsonFlooredAtOne,
        |_| LogLikKind::Poisson,
    )
    .unwrap();

    assert_close("intercept", fit.irls.beta[0], 0.792_438_587_563_215_5, REF_TOL);
    assert_close("slope", fit.irls.beta[1], 0.238_180_118_325_312_0, REF_TOL);
    assert_close("deviance", fit.irls.deviance, 2.196_259_953_161_739_7, REF_TOL);
    assert_close("dispersion", fit.dispersion, 1.0, REF_TOL);
    assert_close(
        "log-likelihood",
        fit.log_likelihood,
        -76.957_489_542_309_9,
        1e-8,
    );

    let inf = fit.inference.as_ref().unwrap();
    assert_close("intercept se", inf.std_errors[0], 0.163_214_083_892_442_5, REF_TOL);
    assert_close("slope se", inf.std_errors[1], 0.083_365_561_343_722_2, REF_TOL);
}

#[test]
fn poisson_two_feature_matches_the_reference() {
    let (y, x) = count_2f();
    let fit = fit(
        &PoissonFamily::log(),
        &y,
        &x,
        &poisson_opts(true),
        DispersionRule::PearsonFlooredAtOne,
        |_| LogLikKind::Poisson,
    )
    .expect("engine fits a design upstream cannot");

    assert_close("intercept", fit.irls.beta[0], 0.783_761_952_889_341_5, REF_TOL);
    assert_close("x1", fit.irls.beta[1], 0.241_563_412_876_373_3, REF_TOL);
    assert_close("x2", fit.irls.beta[2], -0.128_771_260_171_794_0, REF_TOL);
    assert_close("deviance", fit.irls.deviance, 4.058_105_133_758_942, REF_TOL);
    assert_close(
        "null deviance",
        fit.irls.null_deviance,
        16.531_193_379_846_84,
        REF_TOL,
    );
    assert_close("aic", fit.aic, 191.759_615_110_957_4, 1e-7);

    let inf = fit.inference.as_ref().unwrap();
    assert_close("intercept se", inf.std_errors[0], 0.151_817_171_836_096_0, REF_TOL);
    assert_close("x1 se", inf.std_errors[1], 0.078_215_310_636_067_5, REF_TOL);
    assert_close("x2 se", inf.std_errors[2], 0.053_881_779_210_963_2, REF_TOL);
}

#[test]
fn binomial_two_feature_matches_the_reference() {
    let (y, x) = binary_2f();
    let fit = fit(
        &BinomialFamily::logistic(),
        &y,
        &x,
        &poisson_opts(true),
        DispersionRule::Fixed,
        |_| LogLikKind::Binomial,
    )
    .unwrap();

    assert_close("intercept", fit.irls.beta[0], -0.487_597_085_001_402_8, 1e-7);
    assert_close("x1", fit.irls.beta[1], 2.032_116_216_778_389_7, 1e-7);
    assert_close("x2", fit.irls.beta[2], 1.253_874_832_725_110_4, 1e-7);
    assert_close("deviance", fit.irls.deviance, 65.457_295_009_204_61, 1e-7);

    let inf = fit.inference.as_ref().unwrap();
    assert_close("intercept se", inf.std_errors[0], 0.540_822_900_347_485_5, 1e-7);
    assert_close("x1 se", inf.std_errors[1], 0.499_186_947_693_966_3, 1e-7);
    assert_close("x2 se", inf.std_errors[2], 0.504_527_654_702_257_9, 1e-7);
}

#[test]
fn gamma_single_feature_matches_the_reference() {
    let (y, x) = positive_1f();
    let fit = fit(
        &TweedieFamily::new(2.0, 0.0),
        &y,
        &x,
        &poisson_opts(false),
        DispersionRule::Pearson,
        |phi| LogLikKind::Gamma { dispersion: phi },
    )
    .unwrap();

    assert_close("intercept", fit.irls.beta[0], 0.545_169_938_453_627_3, REF_TOL);
    assert_close("slope", fit.irls.beta[1], 0.233_092_575_836_164_4, REF_TOL);
    assert_close("deviance", fit.irls.deviance, 0.157_203_357_930_006_4, REF_TOL);
    assert_close("dispersion", fit.dispersion, 0.003_268_561_584_628_4, 1e-12);
}

#[test]
fn tweedie_single_feature_matches_the_reference() {
    let (y, x) = positive_1f();
    let fit = fit(
        &TweedieFamily::new(1.5, 0.0),
        &y,
        &x,
        &poisson_opts(false),
        DispersionRule::Pearson,
        |phi| LogLikKind::Tweedie {
            power: 1.5,
            dispersion: phi,
        },
    )
    .unwrap();

    assert_close("intercept", fit.irls.beta[0], 0.545_111_433_395_903_3, REF_TOL);
    assert_close("slope", fit.irls.beta[1], 0.233_128_099_664_133_0, REF_TOL);
    assert_close("deviance", fit.irls.deviance, 0.245_508_931_952_702_6, REF_TOL);
}

#[test]
fn negbinomial_single_feature_matches_the_reference() {
    let (y, x) = count_1f();
    let theta = 3.0;
    let fit = fit(
        &NegativeBinomialFamily::new(theta),
        &y,
        &x,
        &poisson_opts(true),
        DispersionRule::Given(theta),
        |_| LogLikKind::NegativeBinomial { theta },
    )
    .unwrap();

    assert_close("intercept", fit.irls.beta[0], 0.787_144_955_189_279_5, REF_TOL);
    assert_close("slope", fit.irls.beta[1], 0.241_472_737_764_617_3, REF_TOL);
    assert_close("deviance", fit.irls.deviance, 1.094_755_272_069_461_3, REF_TOL);

    let inf = fit.inference.as_ref().unwrap();
    assert_close("intercept se", inf.std_errors[0], 0.223_896_259_692_016_9, REF_TOL);
    assert_close("slope se", inf.std_errors[1], 0.119_889_845_294_418_0, REF_TOL);
}

// ---------------------------------------------------------------------------
// Pins for the two upstream defects.
// ---------------------------------------------------------------------------

/// Upstream's inverted back-permutation. If a future dependency bump fixes this,
/// the test fails loudly and the workaround notes above can be removed.
#[test]
fn upstream_cannot_fit_a_three_column_design_but_the_engine_can() {
    let (y, x) = count_2f();
    let (y_col, x_mat) = to_faer(&y, &x);

    let upstream = PoissonRegressor::log()
        .with_intercept(true)
        .max_iterations(100)
        .tolerance(1e-8)
        .build()
        .fit(&x_mat, &y_col);
    assert!(
        upstream.is_err(),
        "upstream unexpectedly fitted a 3-column design — the pivot back-permutation \
         defect may have been fixed; re-check the workaround in normal_eq.rs"
    );

    let fit = fit(
        &PoissonFamily::log(),
        &y,
        &x,
        &poisson_opts(false),
        DispersionRule::PearsonFlooredAtOne,
        |_| LogLikKind::Poisson,
    );
    assert!(fit.is_ok(), "the engine must handle 3-column designs");
}

/// A perfectly-fitting model drives the deviance to ~1e-14, where a purely
/// relative "deviance increased" test degenerates. The engine's absolute floor plus
/// convergence-before-halving ordering must still terminate cleanly and land on the
/// exact coefficients.
#[test]
fn the_engine_converges_on_noise_free_data() {
    let n = 40;
    let xs: Vec<f64> = (0..n).map(|i| (i % 10) as f64).collect();
    let y: Vec<f64> = xs.iter().map(|&v| (0.5 + 0.3 * v).exp()).collect();
    let x = vec![xs];

    let fit = fit(
        &PoissonFamily::log(),
        &y,
        &x,
        &poisson_opts(false),
        DispersionRule::PearsonFlooredAtOne,
        |_| LogLikKind::Poisson,
    )
    .expect("the engine must converge on a perfectly-fitting model");

    assert_close("intercept", fit.irls.beta[0], 0.5, 1e-8);
    assert_close("slope", fit.irls.beta[1], 0.3, 1e-8);
    assert!(fit.irls.deviance.abs() < 1e-10);
}

/// A strong prior makes the deviance rise as the mode is pulled toward the prior
/// location. Monitoring the deviance alone would never settle here.
#[test]
fn a_strong_prior_still_converges() {
    let (y, x) = count_1f();
    let opts = EngineOptions {
        priors: vec![PriorSpec::normal(2.0, 0.01)],
        compute_inference: true,
        ..Default::default()
    };
    let fit = fit(
        &PoissonFamily::log(),
        &y,
        &x,
        &opts,
        DispersionRule::PearsonFlooredAtOne,
        |_| LogLikKind::Poisson,
    )
    .expect("a strongly-penalized fit must converge");

    assert!(fit.irls.converged);
    assert!(
        fit.irls.beta[1] > 1.5,
        "slope {} should be pulled toward the prior location 2.0",
        fit.irls.beta[1]
    );
}

// ---------------------------------------------------------------------------
// Data-contract parity: behaviour the old wrappers implemented around the fit.
// ---------------------------------------------------------------------------

#[test]
fn rows_with_non_finite_values_are_excluded_from_the_fit() {
    let (mut y, mut x) = count_2f();
    let clean_n = y.len();

    y.push(f64::NAN);
    x[0].push(1.0);
    x[1].push(1.0);
    y.push(5.0);
    x[0].push(f64::INFINITY);
    x[1].push(1.0);

    let fit = fit(
        &PoissonFamily::log(),
        &y,
        &x,
        &poisson_opts(false),
        DispersionRule::PearsonFlooredAtOne,
        |_| LogLikKind::Poisson,
    )
    .expect("engine fit");

    assert_eq!(fit.design.n_observations(), clean_n);
    // And the estimates are unchanged from the clean fixture.
    assert_close("intercept", fit.irls.beta[0], 0.783_761_952_889_341_5, REF_TOL);
}

#[test]
fn constant_columns_are_dropped_and_reported_as_nan() {
    let (y, mut x) = count_2f();
    let n = y.len();
    x.insert(0, vec![3.0; n]);

    let opts = EngineOptions {
        compute_inference: true,
        constant_policy: ConstantColumnPolicy::Drop,
        ..Default::default()
    };
    let fit = fit(
        &PoissonFamily::log(),
        &y,
        &x,
        &opts,
        DispersionRule::PearsonFlooredAtOne,
        |_| LogLikKind::Poisson,
    )
    .expect("engine fit");

    let res = fit.to_glm_fit_result();
    assert_eq!(res.coefficients.len(), 3);
    assert!(res.coefficients[0].is_nan(), "constant column must be NaN");
    assert_close("x1", res.coefficients[1], 0.241_563_412_876_373_3, REF_TOL);
    assert_close("x2", res.coefficients[2], -0.128_771_260_171_794_0, REF_TOL);

    let inf = fit.to_glm_inference().unwrap();
    assert!(inf.std_errors[0].is_nan());
    assert!(inf.std_errors[1].is_finite());
}

#[test]
fn dropping_a_constant_column_reproduces_the_clean_fit() {
    let (y, x) = count_2f();
    let n = y.len();
    let mut padded = x.clone();
    padded.insert(1, vec![-2.0; n]);

    let opts = EngineOptions {
        constant_policy: ConstantColumnPolicy::Drop,
        ..Default::default()
    };
    let padded_fit = fit(
        &PoissonFamily::log(),
        &y,
        &padded,
        &opts,
        DispersionRule::PearsonFlooredAtOne,
        |_| LogLikKind::Poisson,
    )
    .unwrap();
    let clean_fit = fit(
        &PoissonFamily::log(),
        &y,
        &x,
        &poisson_opts(false),
        DispersionRule::PearsonFlooredAtOne,
        |_| LogLikKind::Poisson,
    )
    .unwrap();

    let padded_res = padded_fit.to_glm_fit_result();
    let clean_res = clean_fit.to_glm_fit_result();
    assert_close(
        "intercept",
        padded_res.intercept.unwrap(),
        clean_res.intercept.unwrap(),
        TOL,
    );
    assert_close("x1", padded_res.coefficients[0], clean_res.coefficients[0], TOL);
    assert!(padded_res.coefficients[1].is_nan());
    assert_close("x2", padded_res.coefficients[2], clean_res.coefficients[1], TOL);
}

// ---------------------------------------------------------------------------
// The corrected penalized covariance.
// ---------------------------------------------------------------------------

#[test]
fn penalized_laplace_se_is_smaller_than_the_old_naive_se() {
    let (y, x) = count_2f();

    let mk = |vcov: VcovType| EngineOptions {
        compute_inference: true,
        lambda: 5.0,
        vcov,
        ..Default::default()
    };

    let laplace_fit = fit(
        &PoissonFamily::log(),
        &y,
        &x,
        &mk(VcovType::Laplace),
        DispersionRule::PearsonFlooredAtOne,
        |_| LogLikKind::Poisson,
    )
    .unwrap();
    let naive_fit = fit(
        &PoissonFamily::log(),
        &y,
        &x,
        &mk(VcovType::Naive),
        DispersionRule::PearsonFlooredAtOne,
        |_| LogLikKind::Poisson,
    )
    .unwrap();

    // Identical point estimates; only the covariance differs.
    for j in 0..laplace_fit.irls.beta.len() {
        assert_close(
            &format!("beta[{j}] unchanged by vcov choice"),
            laplace_fit.irls.beta[j],
            naive_fit.irls.beta[j],
            1e-12,
        );
    }

    let lap = laplace_fit.inference.unwrap();
    let nv = naive_fit.inference.unwrap();
    for j in 1..lap.std_errors.len() {
        assert!(
            lap.std_errors[j] < nv.std_errors[j],
            "penalized SE at {j} should shrink: laplace {} vs naive {}",
            lap.std_errors[j],
            nv.std_errors[j]
        );
    }
}

#[test]
fn sandwich_differs_from_both_other_variants() {
    let (y, x) = count_2f();
    let mk = |vcov: VcovType| EngineOptions {
        compute_inference: true,
        lambda: 5.0,
        vcov,
        ..Default::default()
    };
    let run = |vcov| {
        fit(
            &PoissonFamily::log(),
            &y,
            &x,
            &mk(vcov),
            DispersionRule::PearsonFlooredAtOne,
            |_| LogLikKind::Poisson,
        )
        .unwrap()
        .inference
        .unwrap()
    };

    let lap = run(VcovType::Laplace);
    let sw = run(VcovType::Sandwich);
    let nv = run(VcovType::Naive);

    // Sandwich is the smallest of the three for a ridge-type penalty.
    assert!(sw.std_errors[1] < lap.std_errors[1]);
    assert!(lap.std_errors[1] < nv.std_errors[1]);
}

#[test]
fn an_unpenalized_fit_is_identical_under_every_vcov_variant() {
    let (y, x) = count_2f();
    let run = |vcov| {
        fit(
            &PoissonFamily::log(),
            &y,
            &x,
            &EngineOptions {
                compute_inference: true,
                vcov,
                ..Default::default()
            },
            DispersionRule::PearsonFlooredAtOne,
            |_| LogLikKind::Poisson,
        )
        .unwrap()
        .inference
        .unwrap()
    };

    let lap = run(VcovType::Laplace);
    let sw = run(VcovType::Sandwich);
    let nv = run(VcovType::Naive);
    for j in 0..3 {
        assert_close("laplace vs naive", lap.std_errors[j], nv.std_errors[j], 1e-9);
        assert_close("laplace vs sandwich", lap.std_errors[j], sw.std_errors[j], 1e-9);
    }
}

// ---------------------------------------------------------------------------
// AIC: intentionally different from upstream.
// ---------------------------------------------------------------------------

#[test]
fn aic_uses_the_real_log_likelihood_not_minus_half_deviance() {
    let (y, x) = count_1f();
    let (y_col, x_mat) = to_faer(&y, &x);

    let upstream = PoissonRegressor::log()
        .with_intercept(true)
        .tolerance(1e-10)
        .build()
        .fit(&x_mat, &y_col)
        .unwrap();

    let fit = fit(
        &PoissonFamily::log(),
        &y,
        &x,
        &poisson_opts(false),
        DispersionRule::PearsonFlooredAtOne,
        |_| LogLikKind::Poisson,
    )
    .unwrap();

    let upstream_ll = -upstream.deviance / 2.0;
    assert!(
        (fit.log_likelihood - upstream_ll).abs() > 1.0,
        "engine log-likelihood {} should differ materially from -deviance/2 {}",
        fit.log_likelihood,
        upstream_ll
    );
    assert_close("aic", fit.aic, 2.0 * 2.0 - 2.0 * fit.log_likelihood, 1e-12);
}

#[test]
fn nuisance_parameters_are_counted_in_aic() {
    let (y, x) = positive_1f();
    let fit = fit(
        &TweedieFamily::new(2.0, 0.0),
        &y,
        &x,
        &poisson_opts(false),
        DispersionRule::Pearson,
        |phi| LogLikKind::Gamma { dispersion: phi },
    )
    .unwrap();

    // 2 coefficients + 1 dispersion.
    assert_close("aic", fit.aic, 2.0 * 3.0 - 2.0 * fit.log_likelihood, 1e-12);
}
