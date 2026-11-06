# Reference values for embedding drift and the multivariate two-sample tests
# (#302): embedding_drift_agg, energy_distance_test_agg, mmd_test_agg. Held by
# test/sql/reference/embedding_drift_reference.test.
#
#   Rscript validation/generators/make_reference_embedding_drift.R
#
# Base R only. energy::eqdist.etest could not be installed here (it needs
# GSL), so the energy statistic is computed from its documented formula; the
# formulas of evidently's embedding drift methods
# (legacy/metrics/data_drift/embedding_drift_methods.py) are:
#   distance: scipy.spatial.distance.<metric>(colMeans(ref), colMeans(cur))
#   ratio:    share of components drifted by the per-component stattest
#             (default wasserstein: W1 / max(population sd of ref, 0.001) >= 0.1)
#   mmd:      unbiased MMD^2 with k(a, b) = exp(-|a - b|^2 / s2), s2 the squared
#             median euclidean distance between reference and current rows
# energy_distance_test_agg's statistic is 2 E|X-Y| - E|X-X'| - E|Y-Y'| with the
# within-sample means over i != j; energy::edist / eqdist.etest report
# n1 n2 / (n1 + n2) * (2 M12 - M11 - M22) with the means over all pairs, printed
# below for comparison. The permutation p-values use upstream's ChaCha8 stream
# and are not reproduced (the test checks the bound 1 / (B + 1) of separated
# samples and the determinism instead).
#
# Shared data (group 0 = reference A, group 1 = current B):
#   CREATE TABLE emb AS
#   SELECT 0 AS grp, [sin(0.7*i), cos(1.3*i), 0.1*(i % 5)] AS e FROM range(30) r(i)
#   UNION ALL
#   SELECT 1, [sin(0.9*i) + 0.4, cos(1.1*i) - 0.2, 0.1*(i % 7) + 0.1] FROM range(25) r(i);
p <- function(label, value) cat(sprintf("%-40s %.15g\n", label, value))

i0 <- 0:29; i1 <- 0:24
A <- cbind(sin(0.7 * i0), cos(1.3 * i0), 0.1 * (i0 %% 5))
B <- cbind(sin(0.9 * i1) + 0.4, cos(1.1 * i1) - 0.2, 0.1 * (i1 %% 7) + 0.1)
m <- nrow(A); n <- nrow(B)

# --- distance between the mean embeddings ------------------------------------
a <- colMeans(A); b <- colMeans(B)
p("distance euclidean", sqrt(sum((a - b)^2)))
p("distance cosine", 1 - sum(a * b) / sqrt(sum(a^2) * sum(b^2)))
p("distance cityblock", sum(abs(a - b)))
p("distance chebyshev", max(abs(a - b)))

# --- ratio: per-component drift ----------------------------------------------
w1 <- function(x, y) {
  # 1-D Wasserstein-1 distance: integral of |F_x - F_y|
  z <- sort(unique(c(x, y)))
  Fx <- ecdf(x)(z[-length(z)]); Fy <- ecdf(y)(z[-length(z)])
  sum(abs(Fx - Fy) * diff(z))
}
psd <- function(x) sqrt(mean((x - mean(x))^2))
wn <- sapply(1:3, function(j) w1(A[, j], B[, j]) / max(psd(A[, j]), 0.001))
for (j in 1:3) p(sprintf("ratio wasserstein normed %d", j), wn[j])
p("ratio wasserstein share (>= 0.1)", mean(wn >= 0.1))
p("ratio wasserstein share (>= 0.5)", mean(wn >= 0.5))
ks <- sapply(1:3, function(j) suppressWarnings(ks.test(A[, j], B[, j]))$p.value)
for (j in 1:3) p(sprintf("ratio ks p %d", j), ks[j])
p("ratio ks share (p < 0.05)", mean(ks < 0.05))

# --- mmd (evidently's median bandwidth) --------------------------------------
D <- as.matrix(dist(rbind(A, B)))
s2 <- median(D[1:m, m + 1:n])^2
K <- exp(-D^2 / s2)
mmd2 <- function(K, m, n) {
  Kxx <- K[1:m, 1:m]; Kyy <- K[m + 1:n, m + 1:n]; Kxy <- K[1:m, m + 1:n]
  (sum(Kxx) - sum(diag(Kxx))) / (m * (m - 1)) + (sum(Kyy) - sum(diag(Kyy))) / (n * (n - 1)) -
    2 * sum(Kxy) / (m * n)
}
p("mmd s2", s2)
p("mmd embedding", max(mmd2(K, m, n), 0))

# --- energy distance (multivariate) ------------------------------------------
Mxy <- mean(D[1:m, m + 1:n])
Uxx <- sum(D[1:m, 1:m]) / (m * (m - 1)); Uyy <- sum(D[m + 1:n, m + 1:n]) / (n * (n - 1))
p("energy statistic", 2 * Mxy - Uxx - Uyy)
Vxx <- mean(D[1:m, 1:m]); Vyy <- mean(D[m + 1:n, m + 1:n])
p("energy eqdist.etest form (V means)", m * n / (m + n) * (2 * Mxy - Vxx - Vyy))

# --- mmd_test_agg kernels -----------------------------------------------------
X <- rbind(A, B)
p("mmd gaussian h=1", mmd2(exp(-D^2 / 2), m, n))
# Default bandwidth of the gaussian kernel: the median heuristic,
# max(median(dist(rbind(x, y))), 0.01) (upstream modern::median_heuristic_bandwidth,
# the bandwidth of mmd_agg for 1-D rows).
h_med <- max(median(dist(X)), 0.01)
p("mmd median bandwidth", h_med)
p("mmd gaussian h=median", mmd2(exp(-D^2 / (2 * h_med^2)), m, n))
# embedding_drift_agg('mmd') reports its kernel bandwidth sqrt(s2 / 2)
# (drift::embedding_mmd_bandwidth: k = exp(-d^2 / s2) = exp(-d^2 / (2 h^2))).
p("embedding mmd bandwidth", sqrt(s2 / 2))
p("mmd gaussian h=0.5", mmd2(exp(-D^2 / (2 * 0.25)), m, n))
L1 <- as.matrix(dist(X, method = "manhattan"))
p("mmd laplacian h=1", mmd2(exp(-L1), m, n))
G <- X %*% t(X)
p("mmd linear", mmd2(G, m, n))
p("mmd polynomial d=2 s=1 o=1", mmd2((G + 1)^2, m, n))
p("mmd polynomial d=3 s=0.5 o=2", mmd2((0.5 * G + 2)^3, m, n))
