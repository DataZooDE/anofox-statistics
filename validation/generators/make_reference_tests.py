#!/usr/bin/env python3
"""scipy reference values for test/sql/reference/*_reference.test (non-regression).

Usage: python3 validation/generators/make_reference_tests.py
Complements make_reference_tests.R for tests that R's base stats lacks
(D'Agostino K^2 = scipy.stats.normaltest, Brunner-Munzel, Kendall tau-c) and
cross-checks a few R values.  Datasets are identical to the R script.
"""
import numpy as np
from scipy import stats


def out(name, x):
    print(f"{name:<40} = {float(x)!r}")


g1 = [5.1, 4.9, 6.2, 5.8, 6.05, 5.5, 5.3, 6.1]
g2 = [6.5, 7.1, 6.8, 7.4, 6.0, 7.9, 6.6, 7.2, 6.9, 7.05]
nv = [2.31, 3.85, 1.97, 4.42, 3.10, 2.76, 5.94, 3.33, 2.05, 4.88,
      3.61, 2.49, 7.12, 3.02, 2.88, 4.15, 3.47, 1.64, 5.21, 2.95]
kx = [1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 8, 8]
ky = [2, 1, 3, 3, 5, 4, 6, 6, 5, 8, 7, 9]

print("## dagostino_k2_agg  scipy.stats.normaltest(nv)")
r = stats.normaltest(nv)
out("dk2.statistic", r.statistic); out("dk2.p_value", r.pvalue)

print("## jarque_bera  scipy.stats.jarque_bera(nv)  (cross-check of R by-hand value)")
r = stats.jarque_bera(nv)
out("jb.statistic", r.statistic); out("jb.p_value", r.pvalue)
out("jb.skewness", stats.skew(nv)); out("jb.excess_kurtosis", stats.kurtosis(nv))

print("## shapiro_wilk_agg  scipy.stats.shapiro(nv)  (same AS R94 code as R)")
r = stats.shapiro(nv)
out("sw.statistic", r.statistic); out("sw.p_value", r.pvalue)

print("## brunner_munzel_agg  scipy.stats.brunnermunzel(g1, g2)")
r = stats.brunnermunzel(g1, g2)
out("bm.statistic", r.statistic); out("bm.p_value", r.pvalue)
# df, p-hat and CI by hand (Brunner & Munzel 2000; lawstat::brunner.munzel.test):
# CI = p_hat -/+ qt(1 - alpha/2, df) * se,  se = (p_hat - 0.5) / statistic
x, y = np.array(g1), np.array(g2)
n1, n2 = len(x), len(y)
rk = stats.rankdata(np.concatenate([x, y]))
rx, ry = rk[:n1], rk[n1:]
sx = ((rx - stats.rankdata(x) - rx.mean() + (n1 + 1) / 2) ** 2).sum() / (n1 - 1)
sy = ((ry - stats.rankdata(y) - ry.mean() + (n2 + 1) / 2) ** 2).sum() / (n2 - 1)
df = (n1 * sx + n2 * sy) ** 2 / ((n1 * sx) ** 2 / (n1 - 1) + (n2 * sy) ** 2 / (n2 - 1))
p_hat = (ry.mean() - (n2 + 1) / 2) / n1
se = np.sqrt(sx / (n1 * n2 ** 2) + sy / (n2 * n1 ** 2))
tq = stats.t.ppf(0.975, df)
out("bm.df", df); out("bm.effect_size", p_hat)
out("bm.ci_lower", p_hat - tq * se); out("bm.ci_upper", p_hat + tq * se)

print("## kendall_agg with ties  scipy.stats.kendalltau(kx, ky, variant=..., method='asymptotic')")
for v in ("b", "c"):
    r = stats.kendalltau(kx, ky, variant=v, method="asymptotic")
    out(f"kendall_ties.tau_{v}", r.statistic); out(f"kendall_ties.p_value_{v}", r.pvalue)
# tau-a = (C - D) / (n (n - 1) / 2)
n = len(kx); s = 0
for i in range(n):
    for j in range(i + 1, n):
        s += np.sign(kx[i] - kx[j]) * np.sign(ky[i] - ky[j])
out("kendall_ties.tau_a", s / (n * (n - 1) / 2))

print("## spearman_agg CI: Fisher z with se = 1/sqrt(n-3) (no Bonett-Wright 1.06 factor)")
cx = [1.2, 2.4, 3.1, 4.8, 5.0, 6.3, 7.7, 8.1, 9.6, 10.2, 11.5, 12.9]
cy = [2.0, 2.9, 4.4, 4.1, 6.2, 5.8, 8.9, 7.5, 9.1, 12.0, 10.8, 13.3]
rs = stats.spearmanr(cx, cy).statistic
z = np.arctanh(rs); h = stats.norm.ppf(0.975) / np.sqrt(len(cx) - 3)
out("spearman.r", rs); out("spearman.ci_lower", np.tanh(z - h)); out("spearman.ci_upper", np.tanh(z + h))
