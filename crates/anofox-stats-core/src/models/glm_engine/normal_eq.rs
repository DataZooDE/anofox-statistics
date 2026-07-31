//! Weighted normal-equation assembly and solves.
//!
//! This is the seam that keeps the mixed-effects phase from needing a second IRLS
//! implementation. The GLM case accumulates a dense `X`; the random-effects case
//! will accumulate the same `X'WX` / `X'Wz` from a block-structured `Z` without
//! ever materializing it. Both then hit the identical solve and inverse code below.

use crate::errors::{StatsError, StatsResult};
use faer::{Col, Mat};

/// Accumulated weighted normal equations `X'WX β = X'Wz`.
#[derive(Debug, Clone)]
pub struct NormalEquations {
    pub xtwx: Mat<f64>,
    pub xtwz: Col<f64>,
    pub n_params: usize,
}

impl NormalEquations {
    pub fn zeros(n_params: usize) -> Self {
        Self {
            xtwx: Mat::zeros(n_params, n_params),
            xtwz: Col::zeros(n_params),
            n_params,
        }
    }

    /// Accumulate a dense design block. `weights` and `z` are per-observation.
    ///
    /// Only the upper triangle is computed and then mirrored — the matrix is
    /// symmetric by construction, which halves the inner loop relative to the
    /// upstream implementation.
    pub fn accumulate_dense(&mut self, x: &Mat<f64>, z: &[f64], weights: &[f64]) {
        let n = x.nrows();
        let p = self.n_params;
        debug_assert_eq!(x.ncols(), p);

        for i in 0..n {
            let w = weights[i];
            if w == 0.0 {
                continue;
            }
            let wz = w * z[i];
            for j in 0..p {
                let xij = x[(i, j)];
                if xij == 0.0 {
                    continue;
                }
                self.xtwz[j] += xij * wz;
                let wx = w * xij;
                for k in j..p {
                    self.xtwx[(j, k)] += wx * x[(i, k)];
                }
            }
        }

        for j in 0..p {
            for k in (j + 1)..p {
                self.xtwx[(k, j)] = self.xtwx[(j, k)];
            }
        }
    }

    /// Solve the system by QR with back-substitution.
    ///
    /// Parameters whose pivot falls below `rank_tolerance` are set to zero rather
    /// than producing an infinity, matching the upstream rank-deficiency policy.
    pub fn solve(&self, rank_tolerance: f64) -> StatsResult<Col<f64>> {
        solve_qr(&self.xtwx, &self.xtwz, rank_tolerance)
    }

    /// Invert the accumulated matrix. Used for the covariance of the estimates.
    pub fn inverse(&self) -> StatsResult<Mat<f64>> {
        invert_spd(&self.xtwx)
    }
}

/// Solve `A x = b` by QR with back-substitution.
pub fn solve_qr(a: &Mat<f64>, b: &Col<f64>, rank_tolerance: f64) -> StatsResult<Col<f64>> {
    let n = a.nrows();
    if n == 0 {
        return Ok(Col::zeros(0));
    }
    let qr = a.qr();
    let q = qr.compute_Q();
    let r: Mat<f64> = qr.R().to_owned();
    let qtb = q.transpose() * b;

    let mut x: Col<f64> = Col::zeros(n);
    for i in (0..n).rev() {
        let mut sum = qtb[i];
        for j in (i + 1)..n {
            sum -= r[(i, j)] * x[j];
        }
        x[i] = if r[(i, i)].abs() > rank_tolerance {
            sum / r[(i, i)]
        } else {
            0.0
        };
    }

    if x.iter().any(|v| !v.is_finite()) {
        return Err(StatsError::SingularMatrix);
    }
    Ok(x)
}

/// Invert a symmetric matrix column by column via QR back-substitution.
pub fn invert_spd(a: &Mat<f64>) -> StatsResult<Mat<f64>> {
    let n = a.nrows();
    if n == 0 {
        return Ok(Mat::zeros(0, 0));
    }
    let qr = a.qr();
    let q = qr.compute_Q();
    let r: Mat<f64> = qr.R().to_owned();
    let qt = q.transpose().to_owned();

    let mut inv: Mat<f64> = Mat::zeros(n, n);
    for col in 0..n {
        // Column `col` of Q' (i.e. Q' e_col).
        let mut sol: Col<f64> = Col::zeros(n);
        for i in (0..n).rev() {
            let mut sum = qt[(i, col)];
            for j in (i + 1)..n {
                sum -= r[(i, j)] * sol[j];
            }
            sol[i] = if r[(i, i)].abs() > 1e-14 {
                sum / r[(i, i)]
            } else {
                0.0
            };
        }
        for i in 0..n {
            inv[(i, col)] = sol[i];
        }
    }
    Ok(inv)
}

/// Solve a weighted least-squares problem via column-pivoted QR on the
/// `sqrt(W)`-scaled design.
///
/// This is the numerically preferable route (it avoids squaring the condition
/// number) and is what the unpenalized path uses, matching upstream exactly so the
/// parity gate can hold to `1e-10`.
pub fn solve_weighted_ls_qr(
    x: &Mat<f64>,
    z: &[f64],
    weights: &[f64],
    rank_tolerance: f64,
) -> StatsResult<Col<f64>> {
    let n = x.nrows();
    let p = x.ncols();

    let mut xw: Mat<f64> = Mat::zeros(n, p);
    let mut zw: Col<f64> = Col::zeros(n);
    for i in 0..n {
        let s = weights[i].max(0.0).sqrt();
        for j in 0..p {
            xw[(i, j)] = s * x[(i, j)];
        }
        zw[i] = s * z[i];
    }

    let qr = xw.col_piv_qr();
    let q = qr.compute_Q();
    let r = qr.R();
    let perm = qr.P();
    let qtz = q.transpose() * zw;

    let mut beta_perm: Col<f64> = Col::zeros(p);
    for i in (0..p).rev() {
        let mut sum = qtz[i];
        for j in (i + 1)..p {
            sum -= r[(i, j)] * beta_perm[j];
        }
        beta_perm[i] = if r[(i, i)].abs() > rank_tolerance {
            sum / r[(i, i)]
        } else {
            0.0
        };
    }

    // Undo the column pivoting. `X * P = Q * R`, so the i-th pivoted column is the
    // original column `fwd[i]` and therefore `beta[fwd[i]] = beta_perm[i]`.
    //
    // Note this uses the *forward* array. Upstream indexes with
    // `perm.inverse().arrays().0` instead, which happens to agree whenever the
    // pivot order is an involution (any 2-column design, and 3-column orders like
    // [2,1,0]) but silently rotates the coefficient vector for a genuine cycle such
    // as [1,2,0]. That is why upstream diverges on ordinary 3-column fixtures — see
    // `permutation_is_undone_for_a_three_cycle` below.
    let mut beta: Col<f64> = Col::zeros(p);
    let fwd = perm.arrays().0;
    for i in 0..p {
        beta[fwd[i]] = beta_perm[i];
    }

    if beta.iter().any(|v| !v.is_finite()) {
        return Err(StatsError::SingularMatrix);
    }
    Ok(beta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_accumulation_matches_the_naive_triple_loop() {
        let x = Mat::from_fn(4, 2, |i, j| (i * 2 + j) as f64 + 1.0);
        let z = [1.0, 2.0, 3.0, 4.0];
        let w = [0.5, 1.0, 1.5, 2.0];

        let mut ne = NormalEquations::zeros(2);
        ne.accumulate_dense(&x, &z, &w);

        let mut expect_xtwx: Mat<f64> = Mat::zeros(2, 2);
        let mut expect_xtwz: Col<f64> = Col::zeros(2);
        for i in 0..4 {
            for j in 0..2 {
                expect_xtwz[j] += w[i] * x[(i, j)] * z[i];
                for k in 0..2 {
                    expect_xtwx[(j, k)] += w[i] * x[(i, j)] * x[(i, k)];
                }
            }
        }

        for j in 0..2 {
            assert!((ne.xtwz[j] - expect_xtwz[j]).abs() < 1e-12);
            for k in 0..2 {
                assert!((ne.xtwx[(j, k)] - expect_xtwx[(j, k)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn inverse_round_trips_to_the_identity() {
        let a = Mat::from_fn(3, 3, |i, j| if i == j { 4.0 } else { 1.0 });
        let inv = invert_spd(&a).unwrap();
        let prod = &a * &inv;
        for i in 0..3 {
            for j in 0..3 {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((prod[(i, j)] - expect).abs() < 1e-10);
            }
        }
    }

    /// Regression test for the back-permutation direction.
    ///
    /// Column norms are chosen so the pivot order is the 3-cycle `[1, 2, 0]`, whose
    /// inverse `[2, 0, 1]` differs. Indexing with the inverse array (as upstream
    /// does) returns the coefficients rotated by one position, which is exactly the
    /// failure that makes upstream's IRLS diverge on ordinary 3-column designs.
    #[test]
    fn permutation_is_undone_for_a_three_cycle() {
        let n = 12;
        let c0: Vec<f64> = (0..n).map(|i| ((i % 4) as f64) * 0.001 + 0.001).collect();
        let c1: Vec<f64> = (0..n).map(|i| ((i * 3) % 5) as f64 * 1000.0 + 5.0).collect();
        let c2: Vec<f64> = (0..n).map(|i| ((i * 7) % 6) as f64 * 10.0 + 1.0).collect();
        let x = Mat::from_fn(n, 3, |i, j| match j {
            0 => c0[i],
            1 => c1[i],
            _ => c2[i],
        });

        let truth = [7.0f64, -2.0, 0.5];
        let z: Vec<f64> = (0..n)
            .map(|i| truth[0] * c0[i] + truth[1] * c1[i] + truth[2] * c2[i])
            .collect();
        let w = vec![1.0; n];

        // Guard the premise: if faer ever changes its pivot choice this test would
        // silently stop covering the cycle case.
        let fwd = x.col_piv_qr().P().arrays().0.to_vec();
        let inv = x.col_piv_qr().P().inverse().arrays().0.to_vec();
        assert_ne!(fwd, inv, "fixture must produce a non-involutive permutation");

        let beta = solve_weighted_ls_qr(&x, &z, &w, 1e-12).unwrap();
        for j in 0..3 {
            assert!(
                (beta[j] - truth[j]).abs() < 1e-8,
                "coefficient {j}: got {}, want {}",
                beta[j],
                truth[j]
            );
        }
    }

    #[test]
    fn weighted_ls_recovers_an_exact_linear_fit() {
        // y = 3 + 2x with an explicit intercept column.
        let x = Mat::from_fn(5, 2, |i, j| if j == 0 { 1.0 } else { i as f64 });
        let z: Vec<f64> = (0..5).map(|i| 3.0 + 2.0 * i as f64).collect();
        let w = vec![1.0; 5];

        let beta = solve_weighted_ls_qr(&x, &z, &w, 1e-12).unwrap();
        assert!((beta[0] - 3.0).abs() < 1e-10);
        assert!((beta[1] - 2.0).abs() < 1e-10);
    }
}
