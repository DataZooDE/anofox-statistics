# Reference values for the evidently drift aggregates (#300): z_drift_agg,
# chisq_drift_agg, g_drift_agg, column_drift_agg, dataset_drift_agg /
# dataset_drift_by. Held by test/sql/reference/drift_evidently_reference.test.
#
#   Rscript validation/generators/make_reference_drift_evidently.R
#
# Base R only (stats). The formulas are evidently's stattests
# (legacy/calculations/stattests): z = proportions_ztest (two-sided, no
# continuity correction) = prop.test(correct = FALSE); chisquare =
# scipy.stats.chisquare(current counts, reference shares * n_current) =
# chisq.test(o, p = r / sum(r)); g_test = power_divergence(lambda_ =
# "log-likelihood") with the expected counts scaled to the current size;
# ks = ks.test; t_test = t.test(var.equal = TRUE); mannw =
# wilcox.test(alternative = "less"); ed = scipy.stats.energy_distance;
# wasserstein = W1 / max(population sd of the reference, 0.001).
#
# Shared data (group 0 = reference, group 1 = current):
#   CREATE TABLE ev AS
#   SELECT 0 AS grp, 2*sin(0.37*i) AS v, (i % 4)::INTEGER AS c,
#          (i % 3 = 0)::INTEGER AS b, ['a','b','c'][i % 3 + 1] AS s
#   FROM range(50) r(i)
#   UNION ALL
#   SELECT 1, 2*sin(0.53*i) + 0.5, ((i*3) % 5)::INTEGER,
#          (i % 2 = 0)::INTEGER, ['a','b','c'][((i*7) % 5) % 3 + 1]
#   FROM range(40) r(i);
p <- function(label, value) cat(sprintf("%-40s %.15g\n", label, value))

i0 <- 0:49; i1 <- 0:39
x  <- 2 * sin(0.37 * i0);  y  <- 2 * sin(0.53 * i1) + 0.5
xc <- i0 %% 4;             yc <- (i1 * 3) %% 5
xb <- as.integer(i0 %% 3 == 0); yb <- as.integer(i1 %% 2 == 0)
xs <- c("a", "b", "c")[i0 %% 3 + 1]; ys <- c("a", "b", "c")[((i1 * 7) %% 5) %% 3 + 1]

# --- z_drift_agg: share of the non-smallest category ---------------------------
z <- function(r, c, alternative = "two.sided") {
  lv <- sort(unique(c(r, c)))
  x1 <- sum(r != lv[1]); x2 <- sum(c != lv[1]); m <- length(r); n <- length(c)
  pt <- suppressWarnings(prop.test(c(x1, x2), c(m, n), correct = FALSE, alternative = alternative))
  list(z = sign(x1 / m - x2 / n) * sqrt(unname(pt$statistic)), p = pt$p.value, p1 = x1 / m, p2 = x2 / n)
}
r <- z(xb, yb); p("z binary z", r$z); p("z binary p", r$p); p("z binary p1", r$p1); p("z binary p2", r$p2)
r <- z(xb, yb, "less"); p("z binary less p", r$p)
r <- z(xb, yb, "greater"); p("z binary greater p", r$p)
r <- z(xs, ys); p("z strings z", r$z); p("z strings p", r$p)

# --- chisq_drift_agg / g_drift_agg -----------------------------------------------
gof <- function(r, c) {
  lv <- sort(unique(c(r, c)))
  rc <- table(factor(r, lv)); cc <- as.vector(table(factor(c, lv)))
  e <- sum(cc) * as.vector(rc) / sum(rc)
  X2 <- suppressWarnings(chisq.test(cc, p = as.vector(rc) / sum(rc))$statistic)
  G <- 2 * sum(cc[cc > 0] * log(cc[cc > 0] / e[cc > 0]))
  df <- length(lv) - 1
  list(X2 = unname(X2), pX2 = pchisq(X2, df, lower.tail = FALSE), G = G,
       pG = pchisq(G, df, lower.tail = FALSE), df = df)
}
r <- gof(xs, ys)
p("chisq strings X2", r$X2); p("chisq strings p", r$pX2); p("chisq strings df", r$df)
p("g strings G", r$G); p("g strings p", r$pG)
r <- gof(xb, yb)
p("chisq binary X2", r$X2); p("chisq binary p", r$pX2); p("g binary G", r$G); p("g binary p", r$pG)
# Numeric codes with group 1 as the reference ({'reference': 1}).
r <- gof(yc, xc)
p("chisq codes ref=1 X2", r$X2); p("chisq codes ref=1 p", r$pX2); p("g codes ref=1 G", r$G); p("g codes ref=1 p", r$pG)
# Group 0 as the reference: category 4 is absent from it, so X2 = G = Inf and p = 0
# (scipy's convention; chisq.test rejects a zero expected share).

# --- column_drift_agg ------------------------------------------------------------
# auto, numeric v (u > 5, reference <= 1000): ks
p("column ks p", ks.test(x, y)$p.value)
p("column t_test p", t.test(x, y, var.equal = TRUE)$p.value)
# n_reference = 50 is not < 50: normal approximation with continuity correction (R rule).
p("column mannw p", wilcox.test(x, y, alternative = "less", exact = FALSE)$p.value)
ed <- function(a, b) sqrt(2 * mean(abs(outer(a, b, "-"))) - mean(abs(outer(a, a, "-"))) -
                          mean(abs(outer(b, b, "-"))))
p("column ed", ed(x, y))
sdpop <- function(v) sqrt(mean((v - mean(v))^2))
w1 <- function(a, b) {  # 1-D Wasserstein-1 between the empirical distributions
  u <- sort(unique(c(a, b))); Fa <- ecdf(a)(u); Fb <- ecdf(b)(u)
  sum(abs(Fa - Fb)[-length(u)] * diff(u))
}
p("column wasserstein", w1(x, y) / max(sdpop(x), 0.001))
# auto, numeric c (u = 5): chisquare of the codes with the absent category 4 -> p = 0
# auto, binary b: z (above); auto, strings s (3 categories): chisquare (above).

# Large reference (> 1000 rows): wasserstein for numeric, jensenshannon for categorical.
xl <- 2 * sin(0.37 * (0:1199)); yl <- 2 * sin(0.53 * (0:399)) + 0.5
p("large wasserstein", w1(xl, yl) / max(sdpop(xl), 0.001))
sl <- c("a", "b", "c")[(0:1199) %% 3 + 1]; tl <- c("a", "b", "c")[(((0:399) * 7) %% 5) %% 3 + 1]
P <- as.vector(table(factor(sl, c("a", "b", "c")))) / 1200
Q <- as.vector(table(factor(tl, c("a", "b", "c")))) / 400
M <- (P + Q) / 2
kl <- function(a, b) sum(ifelse(a > 0, a * log(a / b), 0))
p("large jensenshannon", sqrt((kl(P, M) + kl(Q, M)) / 2))

# --- dataset_drift: share of drifted columns >= drift_share ------------------------
# Columns v (ks p = 0.0575 >= 0.05: no drift), c (chisquare p = 0: drift),
# b (z), s (chisquare): see the values above.
