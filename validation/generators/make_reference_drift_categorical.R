# Reference values for the drift aggregates on category labels (#301):
# VARCHAR values of psi_agg, kl_divergence_agg, jensen_shannon_agg,
# hellinger_agg, tvd_agg, and category_proportions_agg. Held by
# test/sql/reference/drift_categorical_reference.test.
#
#   Rscript validation/generators/make_reference_drift_categorical.R
#
# Base R only. The formulas are evidently's stattests on the category shares p
# (reference) and q (current) over the sorted labels of both samples (byte
# order, sort(method = "radix")):
#   psi       = sum((P - Q) * log(P / Q)), P, Q zero-filled (evidently: 1e-4,
#               or min non-zero / 1e6 when that is <= 1e-4)
#   kl_div    = sum(P * log(P / Q)), P, Q zero-filled and renormalized
#               (scipy.stats.entropy(P, Q))
#   jensen-   = sqrt((KL(p || m) + KL(q || m)) / 2), m = (p + q) / 2
#   shannon     (scipy.spatial.distance.jensenshannon(p, q))
#   hellinger = sqrt(1 - sum(sqrt(p * q)))
#   TVD       = 0.5 * sum(abs(p - q)), permutation p-value
#               (1 + #{TVD* >= TVD}) / (1 + B)
# Cross-checked against scipy (entropy, jensenshannon).
#
# Shared data (group 'ref' = reference, 'cur' = current); 'amber' is only in
# the reference, 'violet' only in the current sample:
#   CREATE TABLE cat AS
#   SELECT 'ref' AS grp, ['red','green','blue','amber'][((i*7) % 11) % 4 + 1] AS s
#   FROM range(60) r(i)
#   UNION ALL
#   SELECT 'cur', ['red','green','blue','violet'][((i*5) % 13) % 4 + 1]
#   FROM range(45) r(i);
p <- function(label, value) cat(sprintf("%-40s %.15g\n", label, value))

i0 <- 0:59; i1 <- 0:44
xs <- c("red", "green", "blue", "amber")[((i0 * 7) %% 11) %% 4 + 1]
ys <- c("red", "green", "blue", "violet")[((i1 * 5) %% 13) %% 4 + 1]

shares <- function(r, c) {
  lv <- sort(unique(c(r, c)), method = "radix")
  list(lv = lv, p = as.numeric(table(factor(r, lv))) / length(r),
       q = as.numeric(table(factor(c, lv))) / length(c))
}
fill <- function(v, eps = NULL) {
  if (is.null(eps)) {
    nz <- v[v > 0]
    eps <- if (min(nz) <= 1e-4) min(nz) / 1e6 else 1e-4
  }
  v[v == 0] <- eps
  v
}
psi <- function(p, q, eps = NULL) { P <- fill(p, eps); Q <- fill(q, eps); sum((P - Q) * log(P / Q)) }
kl <- function(p, q, eps = NULL, base = exp(1)) {
  P <- fill(p, eps); P <- P / sum(P); Q <- fill(q, eps); Q <- Q / sum(Q)
  sum(P * log(P / Q)) / log(base)
}
rel <- function(a, b) ifelse(a > 0, a * log(a / b), 0)
jsd <- function(p, q, base = exp(1)) { m <- (p + q) / 2; (sum(rel(p, m)) + sum(rel(q, m))) / 2 / log(base) }
hel <- function(p, q) sqrt(1 - min(1, sum(sqrt(p * q))))
tvd <- function(p, q) 0.5 * sum(abs(p - q))

s <- shares(xs, ys)
cat("categories:", s$lv, "\n")
cat("p_reference:", sprintf("%.15g", s$p), "\n")
cat("p_current:  ", sprintf("%.15g", s$q), "\n")
p("psi", psi(s$p, s$q))
p("psi eps 1e-3", psi(s$p, s$q, 1e-3))
p("kl", kl(s$p, s$q))
p("kl base 2", kl(s$p, s$q, base = 2))
p("kl eps 1e-3", kl(s$p, s$q, 1e-3))
p("js distance", sqrt(jsd(s$p, s$q)))
p("js divergence", jsd(s$p, s$q))
p("js distance base 2", sqrt(jsd(s$p, s$q, 2)))
p("hellinger", hel(s$p, s$q))
p("tvd", tvd(s$p, s$q))

# Permutation p-value of TVD (Monte Carlo, B = 100000).
set.seed(1)
pooled <- c(xs, ys); m <- length(xs); B <- 100000; t0 <- tvd(s$p, s$q)
tb <- replicate(B, { k <- sample(length(pooled), m); u <- shares(pooled[k], pooled[-k]); tvd(u$p, u$q) })
p("tvd perm p (B = 1e5)", (1 + sum(tb >= t0 - 1e-12)) / (1 + B))

# Reversed reference (group 'cur' as reference): KL is not symmetric.
p("kl reversed", kl(s$q, s$p))

# Upstream's doc example: c("a", "a", "b") vs c("a", "b", "b").
u <- shares(c("a", "a", "b"), c("a", "b", "b"))
p("small psi", psi(u$p, u$q))
p("small tvd", tvd(u$p, u$q))
