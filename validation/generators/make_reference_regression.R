#!/usr/bin/env Rscript
# =============================================================================
# make_reference_regression.R
#
# Generates the regression / GLM / survival reference sqllogictests under
# test/sql/reference/ by fitting the same small, fixed datasets in R and
# writing the R values (17 significant digits) into the .test files together
# with a comment naming the R call each value comes from.
#
#   Rscript validation/generators/make_reference_regression.R        # from repo root
#
# Requires: MASS, survival, lme4, quantreg (all recommended / CRAN).
# glmnet is NOT required: ridge references use the closed form.
#
# The datasets are literal constants below (not seeded RNG draws), so the
# output does not depend on the R version's RNG.  They were drawn once with
# set.seed(20261006) / set.seed(7) / set.seed(11) and rounded.
#
# Where the extension intentionally (or knowingly) follows a different
# convention than R's default, the reference is computed by hand with the
# extension's convention and the generated test says so in a comment.
# =============================================================================
suppressPackageStartupMessages({
  library(MASS); library(survival); library(lme4); library(quantreg)
})

out_dir <- "test/sql/reference"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# ---------------------------------------------------------------- datasets ---
x1 <- c( 6, 3.7, 1.2, 4.6, 9.8, 1.6, 4.4, 2, 8.8, 6.2, 6.1, 9.4, 1.6, 2.1, 1.7, 8.7, 5.4, 2.4, 6.5, 8.1 )
x2 <- c( 4.3, 7, 5.7, 3.8, 4.9, 2.9, 9.5, 7.5, 4.1, 3, 7.7, 4.4, 6.8, 5.6, 2.3, 6.3, 5.4, 7.7, 1.6, 2.6 )
y <- c( 0.67, -1.05, -0.42, 3.19, 4.54, 2.12, 2.7, -0.74, 7.98, 6.84, 0.68, 8.8, 0.67, 0.26, 0.58, 1.05, 6.46, 0.16, 5.99, 9.62 )
w <- c( 0.277, 0.414, 0.718, 0.35, 0.164, 0.65, 0.363, 0.592, 0.186, 0.268, 0.273, 0.172, 0.65, 0.578, 0.635, 0.188, 0.305, 0.541, 0.256, 0.204 )
z1 <- c( 0.63, -0.79, -0.44, -0.34, 0.7, 0.52, 0.88, 0.08, -0.42, -0.57, -0.84, 0.66, 0.49, -0.66, -0.27, -0.19, 0.05, -0.92, 0.31, -0.52, 0.66, -0.99, 0.42, 0.91, 0.06, 0.55, 0.79, 0.2, -0.93, 0.13 )
z2 <- c( 1.68, 0.53, 0.07, 0.42, 1.87, 0.08, 1.07, 0.33, 1.41, 1.73, 0.6, 1.9, 0.58, 1.11, 0.81, 1.43, 0.08, 0.38, 1.36, 0.97, 1.17, 1.32, 0.7, 1.26, 0.45, 1.35, 0.97, 1.77, 0.16, 0.63 )
cnt <- c( 5, 3, 2, 2, 4, 4, 9, 1, 3, 1, 0, 9, 2, 2, 0, 1, 1, 0, 4, 1, 2, 2, 3, 4, 3, 3, 1, 4, 0, 1 )
nb <- c( 0, 2, 0, 2, 0, 1, 4, 5, 2, 7, 2, 4, 2, 5, 0, 8, 8, 2, 7, 2, 9, 0, 10, 6, 3, 3, 2, 7, 0, 10 )
bin <- c( 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1 )
gam <- c( 3.039, 1.658, 0.363, 1.284, 5.101, 0.264, 2.64, 1.224, 2.531, 1.058, 0.283, 2.11, 2.281, 1.261, 1.2, 0.938, 0.342, 0.232, 0.825, 2.642, 2.572, 2.041, 1.033, 1.075, 0.246, 1.931, 3.536, 1.53, 0.406, 2.637 )
a1 <- c( 1.98, 0.8, 0.23, 0.14, 0.49, 1.58, 0.68, 1.94, 0.33, 0.92, 0.34, 0.46, 1.55, 0.19, 0.91, 0.17, 1.12, 0.02, 1.97, 0.63, 1.28, 0.59, 1.99, 1.81, 1.98, 0.13, 1.25, 0.98, 1.94, 0.72, 1.36, 0.53, 0.37, 0.37, 0.76, 1.69, 1, 1.58, 1.68, 0.91 )
time <- c( 6.955, 7.106, 2.697, 4.436, 1.908, 3.026, 3.233, 6.467, 1.787, 1.437, 4.101, 2.109, 5.748, 3.427, 4.406, 4.837, 9.02, 5.871, 12.713, 4.42, 7.153, 5.282, 7.677, 7.032, 10.362, 3.128, 8.259, 2.555, 11.95, 4.649, 10.902, 4.394, 2.331, 3.863, 10.014, 9.295, 7.364, 10.508, 4.393, 2.043 )
ev <- c( 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1 )
g <- c( 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8 )
q1 <- c( 0.79, 0.62, 0.52, 0.61, 0.33, 0.03, -0.64, 0.13, -0.46, 0.44, -0.61, 0.85, -0.9, 0.93, -0.82, 0.81, -0.63, -0.59, -0.42, 0.6, -0.05, 0.54, 0.98, 0.62, 0.27, 0.69, 0, 0.33, 0.02, 0.05, -0.31, 0.62, -0.57, -0.15, -0.24, 0.57, -0.27, 0.46, -0.24, 0.43, -0.93, 0, 0.44, 0.96, 0.9, -0.41, -0.48, 0.85 )
ly <- c( 2.908, 2.598, 2.833, 2.186, 2.836, 1.297, 1.428, 2.111, 2.033, 2.697, 1.628, 2.103, 1.85, 2.175, 2.248, 3.405, 2.305, 1.973, 1.916, 3.072, 2.711, 3.107, 2.664, 3.629, 1.759, 3.226, 2.682, 2.574, 1.428, 3.011, 2.793, 3.011, 2.632, 3.328, 2.751, 3.082, 1.538, 3.293, 1.996, 3.168, 2.243, 2.645, 3.567, 2.985, 3.545, 2.863, 2.491, 3.791 )
pc <- c( 1, 4, 1, 2, 2, 0, 1, 1, 2, 1, 1, 3, 0, 3, 2, 5, 2, 0, 0, 4, 2, 4, 7, 3, 1, 1, 2, 1, 3, 2, 1, 7, 0, 4, 5, 6, 4, 3, 2, 5, 2, 6, 4, 4, 5, 5, 4, 6 )
bb <- c( 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1 )

# ----------------------------------------------------------------- helpers ---
num <- function(v) {
  ifelse(is.na(v), "NULL", sprintf("%.17g", v))
}
values_sql <- function(df) {
  rows <- apply(df, 1, function(r) paste0("(", paste(r, collapse = ", "), ")"))
  paste(rows, collapse = ",\n")
}
# A block of boolean checks: each (expr, ref) pair becomes one column that must
# be `true`.  The tolerance is mixed absolute/relative: |a-b| <= tol*(1+|b|).
# ref = NaN asserts isnan(expr).
checks <- function(comment, from, exprs, refs, tol) {
  stopifnot(length(exprs) == length(refs))
  out <- character()
  idx <- seq_along(exprs)
  for (chunk in split(idx, ceiling(idx / 6))) {
    cols <- vapply(chunk, function(i) {
      if (is.nan(refs[i])) sprintf("isnan(%s)", exprs[i])
      else sprintf("abs((%s) - (%s)) <= %g * (1 + abs(%s))", exprs[i], num(refs[i]), tol, num(refs[i]))
    }, "")
    out <- c(out,
      paste0("# ", comment),
      paste0("query ", strrep("I", length(chunk))),
      paste0("SELECT\n    ", paste(cols, collapse = ",\n    ")),
      paste0("FROM ", from, ";"),
      "----",
      paste(rep("true", length(chunk)), collapse = "\t"),
      "")
  }
  out
}
header <- function(name, desc) {
  c(sprintf("# name: test/sql/reference/%s", name),
    sprintf("# description: %s", desc),
    "# group: [reference]",
    "#",
    "# GENERATED by validation/generators/make_reference_regression.R -- do not edit",
    "# by hand; change the generator and re-run it instead.",
    sprintf("# Reference: %s", R.version.string),
    "",
    "require anofox_statistics",
    "")
}
write_test <- function(name, lines) {
  writeLines(lines, file.path(out_dir, name))
  cat("wrote", file.path(out_dir, name), "\n")
}
lst <- function(field, k) sprintf("r.%s[%d]", field, k)

d1_sql <- c("statement ok",
  "CREATE TABLE d1 (id INTEGER, y DOUBLE, x1 DOUBLE, x2 DOUBLE, w DOUBLE);", "",
  "statement ok",
  paste0("INSERT INTO d1 VALUES\n", values_sql(data.frame(seq_along(y), y, x1, x2, w)), ";"), "")
n <- length(y)

# =============================================================================
# 1. OLS
# =============================================================================
f  <- lm(y ~ x1 + x2); s <- summary(f); cf <- coef(s); ci <- confint(f)
X  <- model.matrix(f); e <- resid(f); p <- ncol(X); h <- hat(X)
B  <- solve(crossprod(X))
sandwich_se <- function(omega) sqrt(diag(B %*% crossprod(X * sqrt(omega)) %*% B))
hc <- list(hc0 = sandwich_se(e^2),
           hc1 = sandwich_se(e^2 * n / (n - p)),
           hc2 = sandwich_se(e^2 / (1 - h)),
           hc3 = sandwich_se(e^2 / (1 - h)^2))
tq <- qt(0.975, n - p)

L <- header("regression_ols.test", "ols_fit_agg / ols_fit_predict_agg against R lm(), summary.lm, confint, predict.lm and hand-coded HC sandwich estimators")
L <- c(L, d1_sql)
src <- "(SELECT ols_fit_agg(y, [x1, x2], {'compute_inference': true}) AS r FROM d1) s"
L <- c(L, checks("coefficients: coef(lm(y ~ x1 + x2))", src,
  c("r.intercept", lst("coefficients", 1:2)), cf[, 1], 1e-10))
L <- c(L, checks("fit statistics: summary(lm)$r.squared, $adj.r.squared, $sigma, $fstatistic, pf(F, 2, 17, lower.tail=FALSE)", src,
  c("r.r_squared", "r.adj_r_squared", "r.residual_std_error", "r.f_statistic", "r.f_pvalue"),
  c(s$r.squared, s$adj.r.squared, s$sigma, s$fstatistic[1],
    pf(s$fstatistic[1], s$fstatistic[2], s$fstatistic[3], lower.tail = FALSE)), 1e-10))
L <- c(L, "# std_errors/t/p/CI cover the slopes only (the extension reports no intercept SE in ols_fit_agg).",
  checks("slope inference: coef(summary(lm))[-1, 2:4]", src,
  c(lst("std_errors", 1:2), lst("t_values", 1:2), lst("p_values", 1:2)),
  c(cf[2:3, 2], cf[2:3, 3], cf[2:3, 4]), 1e-9))
L <- c(L, checks("95% CI: confint(lm)[-1, ]", src,
  c(lst("ci_lower", 1:2), lst("ci_upper", 1:2)), c(ci[2:3, 1], ci[2:3, 2]), 1e-9))
ci90 <- confint(f, level = 0.90)
L <- c(L, checks("90% CI: confint(lm, level = 0.90)[-1, ]",
  "(SELECT ols_fit_agg(y, [x1, x2], {'compute_inference': true, 'confidence_level': 0.90}) AS r FROM d1) s",
  c(lst("ci_lower", 1:2), lst("ci_upper", 1:2)), c(ci90[2:3, 1], ci90[2:3, 2]), 1e-9))
for (k in names(hc)) {
  se <- hc[[k]][2:3]; tv <- cf[2:3, 1] / se
  L <- c(L, checks(sprintf("%s: sqrt(diag((X'X)^-1 X' diag(omega) X (X'X)^-1)), omega = %s; t = b/se, p = 2*pt(-|t|, n-p), CI = b -/+ qt(.975, n-p)*se", toupper(k),
      switch(k, hc0 = "e^2", hc1 = "e^2*n/(n-p)", hc2 = "e^2/(1-h)", hc3 = "e^2/(1-h)^2")),
    sprintf("(SELECT ols_fit_agg(y, [x1, x2], {'compute_inference': true, 'hc_type': '%s'}) AS r FROM d1) s", k),
    c(lst("std_errors", 1:2), lst("t_values", 1:2), lst("p_values", 1:2), lst("ci_lower", 1:2), lst("ci_upper", 1:2)),
    c(se, tv, 2 * pt(-abs(tv), n - p), cf[2:3, 1] - tq * se, cf[2:3, 1] + tq * se), 1e-9))
}

# simple regression (one predictor) incl. QR / Cholesky solvers
f1 <- lm(y ~ x1); s1 <- summary(f1)
for (solver in c("svd", "qr", "cholesky")) {
  L <- c(L, checks(sprintf("simple regression, solver=%s: coef(summary(lm(y ~ x1))), $r.squared, $sigma", solver),
    sprintf("(SELECT ols_fit_agg(y, [x1], {'compute_inference': true, 'solver': '%s'}) AS r FROM d1) s", solver),
    c("r.intercept", "r.coefficients[1]", "r.std_errors[1]", "r.p_values[1]", "r.r_squared", "r.residual_std_error"),
    c(coef(f1), s1$coefficients[2, 2], s1$coefficients[2, 4], s1$r.squared, s1$sigma), 1e-9))
}

# no intercept
f0 <- lm(y ~ 0 + x1 + x2); s0 <- summary(f0); cf0 <- coef(s0); ci0 <- confint(f0)
L <- c(L, checks("no intercept: coef(summary(lm(y ~ 0 + x1 + x2))), confint, $sigma", 
  "(SELECT ols_fit_agg(y, [x1, x2], {'compute_inference': true, 'intercept': false}) AS r FROM d1) s",
  c(lst("coefficients", 1:2), lst("std_errors", 1:2), lst("p_values", 1:2), lst("ci_lower", 1:2), lst("ci_upper", 1:2), "r.residual_std_error"),
  c(cf0[, 1], cf0[, 2], cf0[, 4], ci0[, 1], ci0[, 2], s0$sigma), 1e-9))
L <- c(L,
  "# no intercept: R's summary.lm uses the UNCENTERED total sum of squares when the",
  "# model has no intercept: R^2 = 1 - RSS/sum(y^2), adj. R^2 with n/(n - p), F on p and n-p df.",
  checks("no intercept fit statistics: summary(lm(y ~ 0 + x1 + x2))$r.squared, $adj.r.squared, $fstatistic",
  "(SELECT ols_fit_agg(y, [x1, x2], {'compute_inference': true, 'intercept': false}) AS r FROM d1) s",
  c("r.r_squared", "r.adj_r_squared", "r.f_statistic"), c(s0$r.squared, s0$adj.r.squared, s0$fstatistic[1]), 1e-9))

# rank deficiency (converted from test/data/ols_tests rank_deficient / perfect_collinearity)
L <- c(L,
  "# Constant feature (rank deficient; was test/data/ols_tests/rank_deficient.json):",
  "# R: coef(lm(y ~ x1 + I(rep(1, n)) + x2)) gives NA for the constant column; the",
  "# extension reports NaN there and the R values for the identifiable coefficients.",
  checks("constant column aliased", "(SELECT ols_fit_agg(y, [x1, 1.0, x2], {'compute_inference': true}) AS r FROM d1) s",
    c("r.intercept", lst("coefficients", 1:3), lst("std_errors", c(1, 3)), "r.r_squared"),
    c(cf[1, 1], cf[2, 1], NaN, cf[3, 1], cf[2:3, 2], s$r.squared), 1e-9))
L <- c(L,
  "# Perfect collinearity x1, 2*x1 (was test/data/ols_tests/perfect_collinearity.json):",
  "# R drops the aliased column (NA). The default SVD solver instead returns the",
  "# minimum-norm split b1 = c/5, b2 = 2c/5 of the identifiable combination",
  "# c = b1 + 2*b2 (= coef(lm(y ~ x1 + x2))['x1']); fitted values, R^2 and sigma are",
  "# identical. We assert the identifiable quantities only.",
  checks("collinear pair: identifiable combination and fit statistics",
    "(SELECT ols_fit_agg(y, [x1, 2 * x1, x2]) AS r FROM d1) s",
    c("r.intercept", "r.coefficients[1] + 2 * r.coefficients[2]", "r.coefficients[3]", "r.r_squared", "r.residual_std_error"),
    c(cf[1, 1], cf[2, 1], cf[3, 1], s$r.squared, s$sigma), 1e-9))

# prediction intervals
nd <- data.frame(x1 = c(x1, 5, 12, 0), x2 = c(x2, 5, 0.5, 10))
pi <- predict(f, nd, interval = "prediction", level = 0.95)
pi90 <- predict(f, nd, interval = "prediction", level = 0.90)
L <- c(L,
  "# ---- ols_fit_predict_agg: prediction intervals ----",
  "# Rows with y IS NULL are prediction rows. Expected: R predict.lm(fit, newdata,",
  "# interval = 'prediction'), i.e. yhat -/+ qt(1-a/2, n-p) * sigma * sqrt(1 + x0'(X'X)^-1 x0)",
  "# (leverage-aware), for training rows and new rows alike.",
  "statement ok",
  "CREATE TABLE d1p AS SELECT id, y, x1, x2 FROM d1 UNION ALL SELECT * FROM (VALUES (21, NULL, 5.0, 5.0), (22, NULL, 12.0, 0.5), (23, NULL, 0.0, 10.0)) t(id, y, x1, x2);",
  "")
for (i in c(1, 7, 20, 21, 22, 23)) {
  L <- c(L, checks(sprintf("row %d: predict.lm(interval='prediction')[%d, ] (%s row)", i, i, ifelse(i > 20, "new", "training")),
    sprintf("(SELECT ols_fit_predict_agg(y, [x1, x2] ORDER BY id)[%d] AS r FROM d1p) s", i),
    c("r.yhat", "r.yhat_lower", "r.yhat_upper"), pi[i, ], 1e-9))
}
L <- c(L, checks("row 22 at 90%: predict.lm(interval='prediction', level=0.90)[22, ]",
  "(SELECT ols_fit_predict_agg(y, [x1, x2], {'confidence_level': 0.90} ORDER BY id)[22] AS r FROM d1p) s",
  c("r.yhat", "r.yhat_lower", "r.yhat_upper"), pi90[22, ], 1e-9))
L <- c(L,
  "# Note: interval = 'confidence' (mean response) is not exposed by ols_fit_predict_agg;",
  "# the coefficient confidence intervals above cover confint().")
write_test("regression_ols.test", L)

# =============================================================================
# 2. WLS (was test/data/wls_tests)
# =============================================================================
fw <- lm(y ~ x1 + x2, weights = w); sw <- summary(fw); cw <- coef(sw); ciw <- confint(fw)
L <- header("regression_wls.test", "wls_fit_agg against R lm(weights = w)")
L <- c(L, d1_sql)
src <- "(SELECT wls_fit_agg(y, [x1, x2], w, {'compute_inference': true}) AS r FROM d1) s"
L <- c(L, checks("coef(lm(y ~ x1 + x2, weights = w)), summary: r.squared, adj.r.squared, sigma, fstatistic", src,
  c("r.intercept", lst("coefficients", 1:2), "r.r_squared", "r.adj_r_squared", "r.residual_std_error", "r.f_statistic",
    "r.f_pvalue"),
  c(cw[, 1], sw$r.squared, sw$adj.r.squared, sw$sigma, sw$fstatistic[1],
    pf(sw$fstatistic[1], 2, n - 3, lower.tail = FALSE)), 1e-9))
L <- c(L, checks("slope inference: coef(summary(lm(..., weights = w)))[-1, ], confint", src,
  c(lst("std_errors", 1:2), lst("t_values", 1:2), lst("p_values", 1:2), lst("ci_lower", 1:2), lst("ci_upper", 1:2)),
  c(cw[2:3, 2], cw[2:3, 3], cw[2:3, 4], ciw[2:3, 1], ciw[2:3, 2]), 1e-9))
L <- c(L, checks("equal weights (2.5) reproduce OLS: coef(summary(lm(y ~ x1 + x2)))",
  "(SELECT wls_fit_agg(y, [x1, x2], 2.5, {'compute_inference': true}) AS r FROM d1) s",
  c("r.intercept", lst("coefficients", 1:2), lst("std_errors", 1:2), "r.r_squared", "r.residual_std_error"),
  c(cf[, 1], cf[2:3, 2], s$r.squared, s$sigma), 1e-9))
write_test("regression_wls.test", L)

# =============================================================================
# 3. Ridge (closed form; was test/data/ridge_tests which used glmnet)
# =============================================================================
ridge_cf <- function(lambda) {
  Xc <- scale(cbind(x1, x2), scale = FALSE); yc <- y - mean(y)
  b <- solve(crossprod(Xc) + lambda * diag(2), crossprod(Xc, yc))
  a <- mean(y) - sum(colMeans(cbind(x1, x2)) * b)
  rss <- sum((y - a - cbind(x1, x2) %*% b)^2)
  c(a, b, 1 - rss / sum((y - mean(y))^2))
}
L <- header("regression_ridge.test", "ridge_fit_agg against the closed-form ridge solution (intercept unpenalised)")
L <- c(L, d1_sql,
  "# Reference: b = solve(t(Xc) %*% Xc + lambda * I, t(Xc) %*% yc) on centred data,",
  "# a = mean(y) - colMeans(X) %*% b (intercept not penalised); R^2 = 1 - RSS/TSS.",
  "# lambda_scaling = 'glmnet' multiplies lambda by n before solving (glmnet's 1/(2n)",
  "# loss scaling; note glmnet itself additionally standardises y, so its lambda",
  "# path is not directly comparable). Ridge std_errors use the uncentred",
  "# sqrt(MSE * diag((X'X + lambda I)^-1)) approximation and have no R reference.",
  "")
for (lam in c(0.1, 1, 10)) {
  L <- c(L, checks(sprintf("alpha = %g (raw): closed form ridge", lam),
    sprintf("(SELECT ridge_fit_agg(y, [x1, x2], {'alpha': %g}) AS r FROM d1) s", lam),
    c("r.intercept", lst("coefficients", 1:2), "r.r_squared"), ridge_cf(lam), 1e-9))
}
L <- c(L, checks("alpha = 0.5, lambda_scaling = 'glmnet': closed form with lambda = 0.5 * n",
  "(SELECT ridge_fit_agg(y, [x1, x2], {'alpha': 0.5, 'lambda_scaling': 'glmnet'}) AS r FROM d1) s",
  c("r.intercept", lst("coefficients", 1:2), "r.r_squared"), ridge_cf(0.5 * n), 1e-9))
L <- c(L, checks("alpha = 0 reduces to OLS: coef(lm(y ~ x1 + x2))",
  "(SELECT ridge_fit_agg(y, [x1, x2], {'alpha': 0.0}) AS r FROM d1) s",
  c("r.intercept", lst("coefficients", 1:2)), cf[, 1], 1e-9))
write_test("regression_ridge.test", L)

# =============================================================================
# 4. Quantile regression
# =============================================================================
rho <- function(u, t) sum(u * (t - (u < 0)))
L <- header("regression_quantile.test", "quantile_fit_predict_agg against quantreg::rq (method = 'br')")
L <- c(L, d1_sql,
  "# quantreg::rq solves the linear programme exactly; the extension uses IRLS with",
  "# a smoothed check-loss weight (epsilon = 1e-6, max 100 iterations), so yhat is",
  "# compared at 1e-4. n = 20 and n*tau is non-integer for both taus, and rq reports",
  "# a unique (non-degenerate) solution. The check-loss of the extension's fit must",
  "# not exceed the LP optimum by more than 1e-4.",
  "")
for (tau in c(0.37, 0.62)) {
  fq <- rq(y ~ x1 + x2, tau = tau); fv <- fitted(fq)
  for (i in c(1, 2, 4, 9)) {
    L <- c(L, checks(sprintf("tau = %g, row %d: fitted(rq(y ~ x1 + x2, tau = %g))[%d]", tau, i, tau, i),
      sprintf("(SELECT quantile_fit_predict_agg(y, [x1, x2], {'tau': %g} ORDER BY id)[%d] AS r FROM d1) s", tau, i),
      "r.yhat", fv[i], 1e-4))
  }
  L <- c(L, sprintf("# tau = %g: check loss sum(rho_tau(y - fitted(rq))) = %.12f", tau, rho(y - fv, tau)),
    "query I",
    sprintf("SELECT sum((p.y - p.yhat) * (%g - CASE WHEN p.y - p.yhat < 0 THEN 1 ELSE 0 END)) <= %s + 1e-4", tau, num(rho(y - fv, tau))),
    sprintf("FROM (SELECT UNNEST(quantile_fit_predict_agg(y, [x1, x2], {'tau': %g})) AS p FROM d1) s;", tau),
    "----", "true", "")
}
write_test("regression_quantile.test", L)

# =============================================================================
# 5. GLMs
# =============================================================================
d2_sql <- c("statement ok",
  "CREATE TABLE d2 (id INTEGER, z1 DOUBLE, z2 DOUBLE, cnt DOUBLE, nb DOUBLE, bin DOUBLE, gam DOUBLE);", "",
  "statement ok",
  paste0("INSERT INTO d2 VALUES\n", values_sql(data.frame(seq_along(z1), z1, z2, cnt, nb, bin, gam)), ";"), "")
m <- length(z1)
glm_block <- function(label, fit, call_sql, se_scale = 1, zp = TRUE, tol = 1e-6, extra = NULL) {
  sm <- summary(fit, dispersion = if (is.null(extra$disp)) NULL else extra$disp)
  cc <- coef(sm); se <- cc[, 2]; z <- cc[, 1] / se
  pv <- if (zp) 2 * pnorm(-abs(z)) else cc[, 4]
  zq <- qnorm(0.975)
  src <- sprintf("(SELECT %s AS r FROM d2) s", call_sql)
  out <- checks(sprintf("%s: coefficients", label), src, c("r.intercept", lst("coefficients", 1:2)), cc[, 1], tol)
  out <- c(out, checks(sprintf("%s: slope SE, z, p, Wald CI (b -/+ qnorm(.975)*se)", label), src,
    c(lst("std_errors", 1:2), lst("z_values", 1:2), lst("p_values", 1:2), lst("ci_lower", 1:2), lst("ci_upper", 1:2)),
    c(se[2:3], z[2:3], pv[2:3], cc[2:3, 1] - zq * se[2:3], cc[2:3, 1] + zq * se[2:3]), tol))
  dev <- c(fit$deviance, fit$null.deviance, 1 - fit$deviance / fit$null.deviance)
  aic <- if (is.null(extra$aic)) fit$aic else extra$aic
  out <- c(out, checks(sprintf("%s: deviance, null deviance, pseudo R^2 = 1 - dev/null, AIC", label), src,
    c("r.deviance", "r.null_deviance", "r.pseudo_r_squared", "r.aic"), c(dev, aic), tol))
  out
}
ctl <- glm.control(epsilon = 1e-14, maxit = 100)
L <- header("glm_fit.test", "poisson / binomial / negbinom / gamma fit_agg against R glm() and MASS::glm.nb")
L <- c(L, d2_sql,
  "# R fits use glm.control(epsilon = 1e-14); the extension is run with",
  "# {'tolerance': 1e-12}. Coefficients, SEs and derived values compare at 1e-6.",
  "")
# Poisson
fp <- glm(cnt ~ z1 + z2, family = poisson, control = ctl)
phi_p <- sum(residuals(fp, "pearson")^2) / fp$df.residual
L <- c(L,
  "# ---- Poisson (log link) ----",
  "# CONVENTION: the extension scales the Poisson covariance by max(1, Pearson",
  sprintf("# chi^2/df) (here %.6f), a floored quasi-Poisson correction; R's", phi_p),
  "# summary.glm uses dispersion 1. Reference: summary(glm(cnt ~ z1 + z2, poisson),",
  "# dispersion = max(1, sum(residuals(fit, 'pearson')^2)/df.residual)).",
  glm_block("glm(cnt ~ z1 + z2, family = poisson)", fp,
    "poisson_fit_agg(cnt, [z1, z2], {'compute_inference': true, 'tolerance': 1e-12})",
    extra = list(disp = max(1, phi_p))),
  checks("Poisson dispersion field = max(1, Pearson chi^2 / df)",
    "(SELECT poisson_fit_agg(cnt, [z1, z2], {'tolerance': 1e-12}) AS r FROM d2) s", "r.dispersion", max(1, phi_p), 1e-6))
# Binomial
for (lk in c("logit", "probit", "cloglog")) {
  fb <- glm(bin ~ z1 + z2, family = binomial(link = lk), control = ctl)
  L <- c(L, sprintf("# ---- Binomial (%s link) ----", lk),
    glm_block(sprintf("glm(bin ~ z1 + z2, family = binomial(link = '%s'))", lk), fb,
      sprintf("binomial_fit_agg(bin, [z1, z2], {'compute_inference': true, 'tolerance': 1e-12, 'binomial_link': '%s'})", lk)))
}
fb <- glm(bin ~ z1 + z2, family = binomial, control = ctl)
L <- c(L, "# logistic_fit_agg is the logit-binomial fit plus classification accuracy",
  glm_block("glm(bin ~ z1 + z2, family = binomial) via logistic_fit_agg", fb,
    "logistic_fit_agg(bin, [z1, z2], {'compute_inference': true, 'tolerance': 1e-12})"),
  checks("accuracy = mean((fitted > 0.5) == bin)",
    "(SELECT logistic_fit_agg(bin, [z1, z2], {'tolerance': 1e-12}) AS r FROM d2) s", "r.accuracy",
    mean((fitted(fb) > 0.5) == bin), 1e-12))
# Negative binomial
fnb <- glm.nb(nb ~ z1 + z2, control = glm.control(epsilon = 1e-14, maxit = 100))
th <- fnb$theta
fnb_fixed <- glm(nb ~ z1 + z2, family = negative.binomial(th), control = ctl)
L <- c(L,
  "# ---- Negative binomial ----",
  sprintf("# MASS::glm.nb estimates theta by maximum likelihood (theta = %.12f).", th),
  "# Passing that theta reproduces glm.nb's coefficients, SEs (conditional on theta)",
  "# and AIC (= -2 logLik + 2 (p + 1), theta counted as a parameter).",
  glm_block("MASS::glm.nb(nb ~ z1 + z2) at its ML theta", fnb_fixed,
    sprintf("negbinom_fit_agg(nb, [z1, z2], {'compute_inference': true, 'tolerance': 1e-12, 'theta': %s})", num(th)),
    extra = list(aic = fnb$aic)),
  checks("negbinom dispersion field reports theta",
    sprintf("(SELECT negbinom_fit_agg(nb, [z1, z2], {'theta': %s}) AS r FROM d2) s", num(th)), "r.dispersion", th, 1e-12))
# moment-estimated theta (extension default)
mom_theta <- function() {
  theta <- 1
  ft <- glm(nb ~ z1 + z2, family = negative.binomial(theta), control = ctl)
  for (k in 1:25) {
    mu <- fitted(ft)
    num_ <- sum((nb - mu)^2 - mu); den <- sum(mu^2)
    nxt <- if (den <= 0 || num_ <= 0) 1e6 else min(max(1 / max(num_ / den, 1e-12), 1e-6), 1e6)
    if (abs(nxt - theta) / max(theta, 1e-8) < 1e-6) { theta <- nxt; break }
    theta <- nxt
    ft <- glm(nb ~ z1 + z2, family = negative.binomial(theta), control = ctl)
  }
  list(theta = theta, fit = glm(nb ~ z1 + z2, family = negative.binomial(theta), control = ctl))
}
mt <- mom_theta()
L <- c(L,
  "# CONVENTION: without 'theta' the extension estimates theta by the method of",
  "# moments, alternating theta_{k+1} = sum(mu^2) / sum((y - mu)^2 - mu) with IRLS",
  sprintf("# (start 1, <= 25 rounds, rel. change < 1e-6); here theta = %.10f, whereas", mt$theta),
  sprintf("# glm.nb's ML estimate is %.10f. Reference: that iteration run with", th),
  "# glm(family = negative.binomial(theta)) in R.",
  checks("negbinom default: moment theta and coefficients at that theta",
    "(SELECT negbinom_fit_agg(nb, [z1, z2], {'tolerance': 1e-12}) AS r FROM d2) s",
    c("r.dispersion", "r.intercept", lst("coefficients", 1:2)), c(mt$theta, coef(mt$fit)), 1e-5))
# Gamma
fg <- glm(gam ~ z1 + z2, family = Gamma(link = "log"), control = ctl)
phi_g <- sum(residuals(fg, "pearson")^2) / fg$df.residual
mu_g <- fitted(fg)
aic_g <- -2 * sum(dgamma(gam, shape = 1 / phi_g, rate = 1 / (phi_g * mu_g), log = TRUE)) + 2 * (3 + 1)
L <- c(L,
  "# ---- Gamma (log link) ----",
  "# SEs use the Pearson dispersion exactly like summary.glm. CONVENTIONS: (1) the",
  "# extension reports normal (z) p-values where summary.glm uses t(n-p); the",
  "# reference p = 2*pnorm(-|b/se|). (2) R's Gamma()$aic plugs in dispersion =",
  "# deviance/n; the extension plugs in the Pearson dispersion. Reference AIC =",
  "# -2*sum(dgamma(y, shape = 1/phi_P, rate = 1/(phi_P*mu), log = TRUE)) + 2*(p + 1).",
  glm_block("glm(gam ~ z1 + z2, family = Gamma(link = 'log'))", fg,
    "gamma_fit_agg(gam, [z1, z2], {'compute_inference': true, 'tolerance': 1e-12})",
    extra = list(aic = aic_g)),
  checks("Gamma dispersion = Pearson chi^2 / df",
    "(SELECT gamma_fit_agg(gam, [z1, z2], {'tolerance': 1e-12}) AS r FROM d2) s", "r.dispersion", phi_g, 1e-6))
write_test("glm_fit.test", L)

# =============================================================================
# 6. GLMM
# =============================================================================
d4_sql <- c("statement ok",
  "CREATE TABLE d4 (g VARCHAR, q1 DOUBLE, ly DOUBLE, pc DOUBLE, bb DOUBLE);", "",
  "statement ok",
  paste0("INSERT INTO d4 VALUES\n", values_sql(data.frame(sprintf("'g%d'", g), q1, ly, pc, bb)), ";"), "")
L <- header("glm_glmm.test", "glmm_fit_agg against lme4::lmer and lme4::glmer(nAGQ = 0)")
L <- c(L, d4_sql)
for (reml in c(TRUE, FALSE)) {
  fl <- lmer(ly ~ q1 + (1 | g), REML = reml, control = lmerControl(optimizer = "bobyqa",
             optCtrl = list(rhoend = 1e-12)))
  vc <- as.data.frame(VarCorr(fl))$vcov; cc <- coef(summary(fl))
  L <- c(L, checks(sprintf("gaussian, reml = %s: lmer(ly ~ q1 + (1|g), REML = %s): fixef, SEs, VarCorr, logLik, AIC, BIC", reml, reml),
    sprintf("(SELECT glmm_fit_agg(ly, [q1], g, {'compute_inference': true, 'reml': %s}) AS r FROM d4) s", tolower(reml)),
    c("r.intercept", "r.coefficients[1]", "r.intercept_std_error", "r.std_errors[1]", "r.var_group", "r.var_residual",
      "r.icc", "r.log_likelihood", "r.aic", "r.bic"),
    c(cc[, 1], cc[, 2], vc, vc[1] / sum(vc), as.numeric(logLik(fl)), AIC(fl), BIC(fl)), 1e-6))
}
L <- c(L,
  "# Non-gaussian families: the extension maximises the Laplace deviance with the",
  "# fixed effects inside the penalised conditional mode, i.e. lme4::glmer(nAGQ = 0).",
  "# (The default glmer nAGQ = 1 optimises beta in the outer loop and gives",
  "# different estimates.) lme4's derivative-free optimiser stops at ~1e-6 relative",
  "# precision, hence tolerance 1e-4 (poisson). For the binomial fixture the profiled",
  "# deviance is very flat in the variance component: lme4 (bobyqa) reaches",
  "# 59.37314049 while the extension's golden-section search stops at 59.37314327,",
  "# which moves var_group by ~2e-3 relative; the binomial block uses 5e-3.",
  "")
for (fam in c("poisson", "binomial")) {
  yv <- if (fam == "poisson") "pc" else "bb"
  fm <- glmer(as.formula(sprintf("%s ~ q1 + (1 | g)", yv)), family = fam, nAGQ = 0)
  cc <- coef(summary(fm)); vg <- as.data.frame(VarCorr(fm))$vcov
  src <- sprintf("(SELECT glmm_fit_agg(%s, [q1], g, {'family': '%s', 'compute_inference': true}) AS r FROM d4) s", yv, fam)
  L <- c(L, checks(sprintf("%s: glmer(%s ~ q1 + (1|g), family = %s, nAGQ = 0): fixef, SEs, var_group", fam, yv, fam), src,
    c("r.intercept", "r.coefficients[1]", "r.intercept_std_error", "r.std_errors[1]", "r.var_group"),
    c(cc[, 1], cc[, 2], vg), if (fam == "binomial") 5e-3 else 1e-4))
  L <- c(L,
    "# logLik(glmer) is the full Laplace log-likelihood including the saturated-model",
    "# term sum(log f(y | mu = y)) that the deviance omits; AIC = -2 logLik + 2 * 3.",
    "# (The extension currently reports -deviance/2, i.e. it omits that term, so",
    "# log_likelihood/aic/bic are not comparable with R or with its own GLM fields.)",
    checks(sprintf("%s: logLik(glmer(..., nAGQ = 0)), AIC, BIC", fam), src,
      c("r.log_likelihood", "r.aic", "r.bic"), c(as.numeric(logLik(fm)), AIC(fm), BIC(fm)), if (fam == "binomial") 5e-3 else 1e-4))
}
write_test("glm_glmm.test", L)

# =============================================================================
# 7. AFT
# =============================================================================
d3_sql <- c("statement ok",
  "CREATE TABLE d3 (time DOUBLE, a1 DOUBLE, ev DOUBLE);", "",
  "statement ok",
  paste0("INSERT INTO d3 VALUES\n", values_sql(data.frame(time, a1, ev)), ";"), "")
L <- header("survival_aft.test", "aft_fit_agg against survival::survreg")
L <- c(L, d3_sql)
for (dist in c("weibull", "lognormal", "loglogistic", "exponential")) {
  fa <- survreg(Surv(time, ev) ~ a1, dist = dist, control = survreg.control(rel.tolerance = 1e-12, maxiter = 100))
  tb <- summary(fa)$table
  src <- sprintf("(SELECT aft_fit_agg(time, [a1], ev, {'dist': '%s', 'compute_inference': true}) AS r FROM d3) s", dist)
  ex <- c("r.intercept", "r.coefficients[1]", "r.scale", "r.log_likelihood", "r.null_log_likelihood", "r.aic", "r.bic")
  rv <- c(coef(fa), fa$scale, fa$loglik[2], fa$loglik[1], AIC(fa), BIC(fa))
  L <- c(L, checks(sprintf("%s: survreg(Surv(time, ev) ~ a1, dist = '%s'): coef, scale, loglik, AIC, BIC", dist, dist), src, ex, rv, 1e-6))
  ex <- c("r.intercept_std_error", "r.std_errors[1]", "r.z_values[1]", "r.p_values[1]")
  rv <- c(tb[1:2, 2], tb[2, 3], tb[2, 4])
  if (dist != "exponential") { ex <- c(ex, "r.log_scale_std_error"); rv <- c(rv, tb[3, 2]) }
  L <- c(L, checks(sprintf("%s: summary(survreg)$table standard errors (incl. Log(scale))", dist), src, ex, rv, 1e-5))
}
write_test("survival_aft.test", L)
