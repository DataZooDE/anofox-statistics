#!/usr/bin/env Rscript
# =============================================================================
# make_reference_paths_inference.R
#
# Generates the reference sqllogictests for regularisation paths / CV (#154),
# quantile-regression inference (#157) and predictions with standard errors /
# partial residuals (#158) under test/sql/reference/, in the style of
# make_reference_regression.R (fixed literal datasets, 17 significant digits,
# mixed absolute/relative tolerance |a - b| <= tol * (1 + |b|)).
#
#   Rscript validation/generators/make_reference_paths_inference.R     # from repo root
#
# Requires: quantreg, grpreg, MASS. glmnet is NOT required (and was not installed
# on the generating machine): the path / CV references implement glmnet's
# documented Gaussian objective, default lambda grid and cv.glmnet aggregates
# exactly as upstream anofox-regression's tests/r_scripts/
# generate_reg_path_validation.R does, with grpreg as an independent lasso solver.
#
# The LARS knots are read from the built extension (build/release) and the
# lasso solution at each knot is recomputed here from the documented objective;
# the first knot and the end of the path are checked against closed forms.
# =============================================================================
suppressPackageStartupMessages({
  library(quantreg); library(grpreg); library(MASS)
})

out_dir <- "test/sql/reference"
duckdb_bin <- "build/release/duckdb"
extension <- "build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension"

# ---------------------------------------------------------------- datasets ---
# d1 / d2: the datasets of make_reference_regression.R.
x1 <- c( 6, 3.7, 1.2, 4.6, 9.8, 1.6, 4.4, 2, 8.8, 6.2, 6.1, 9.4, 1.6, 2.1, 1.7, 8.7, 5.4, 2.4, 6.5, 8.1 )
x2 <- c( 4.3, 7, 5.7, 3.8, 4.9, 2.9, 9.5, 7.5, 4.1, 3, 7.7, 4.4, 6.8, 5.6, 2.3, 6.3, 5.4, 7.7, 1.6, 2.6 )
y <- c( 0.67, -1.05, -0.42, 3.19, 4.54, 2.12, 2.7, -0.74, 7.98, 6.84, 0.68, 8.8, 0.67, 0.26, 0.58, 1.05, 6.46, 0.16, 5.99, 9.62 )
w <- c( 0.277, 0.414, 0.718, 0.35, 0.164, 0.65, 0.363, 0.592, 0.186, 0.268, 0.273, 0.172, 0.65, 0.578, 0.635, 0.188, 0.305, 0.541, 0.256, 0.204 )
z1 <- c( 0.63, -0.79, -0.44, -0.34, 0.7, 0.52, 0.88, 0.08, -0.42, -0.57, -0.84, 0.66, 0.49, -0.66, -0.27, -0.19, 0.05, -0.92, 0.31, -0.52, 0.66, -0.99, 0.42, 0.91, 0.06, 0.55, 0.79, 0.2, -0.93, 0.13 )
z2 <- c( 1.68, 0.53, 0.07, 0.42, 1.87, 0.08, 1.07, 0.33, 1.41, 1.73, 0.6, 1.9, 0.58, 1.11, 0.81, 1.43, 0.08, 0.38, 1.36, 0.97, 1.17, 1.32, 0.7, 1.26, 0.45, 1.35, 0.97, 1.77, 0.16, 0.63 )
cnt <- c( 5, 3, 2, 2, 4, 4, 9, 1, 3, 1, 0, 9, 2, 2, 0, 1, 1, 0, 4, 1, 2, 2, 3, 4, 3, 3, 1, 4, 0, 1 )
bin <- c( 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1 )
gam <- c( 3.039, 1.658, 0.363, 1.284, 5.101, 0.264, 2.64, 1.224, 2.531, 1.058, 0.283, 2.11, 2.281, 1.261, 1.2, 0.938, 0.342, 0.232, 0.825, 2.642, 2.572, 2.041, 1.033, 1.075, 0.246, 1.931, 3.536, 1.53, 0.406, 2.637 )
# d3 (paths, n = 40, p = 4, x2 / x3 correlated): drawn with set.seed(20261012)
# as x1 = 10 + 3 N, x2 = z/2 + 0.2 N, x3 = z + 0.4 N, x4 = 100 U,
# y = 3 + 0.8 x1 - 2 x2 + 1.5 x3 + 0.01 x4 + 2 N, fold = sample(rep(1:5, length = 40)).
# d4 (quantile, n = 63, heteroskedastic): set.seed(157), q1 = 10 U, q2 = N,
# qy = 1 + 0.7 q1 - 1.2 q2 + (0.5 + 0.15 q1) N.
p1 <- c( 11.19, 11.38, 8.88, 11.71, 10.33, 8.4, 7.64, 11.83, 7.39, 9.73, 7.71, 8.92, 8.86, 5.64, 10.69, 7.45, 5.58, 9.64, 12.33, 10.86, 7.16, 7.85, 11.24, 7.73, 9.26, 11.22, 8.9, 9.88, 8, 12.63, 11.91, 8.54, 11.71, 8.36, 9.66, 12.05, 6.38, 11.83, 10.86, 13.22 )
p2 <- c( 0.07, -0.17, 0, 0.71, -0.69, 0.32, 1.14, 0.33, -0.06, 0.27, -0.07, -0.76, -0.61, -0.38, 0.18, 0.03, 0.4, -0.29, 1.01, 1.18, -0.18, 0.12, -0.32, -0.37, 0.94, 0.15, -0.15, 0.4, 0.12, -0.95, -0.41, 0.5, -0.6, -0.49, 1.58, 0.43, 0, 0.4, -0.42, -0.27 )
p3 <- c( 0.13, 0.3, -0.15, 1.98, -1.05, 0.93, 1.43, 0.68, 0.92, 1.33, 0.19, -0.49, -1.97, -1.12, -0.41, 0.77, -0.24, -0.15, 2.46, 2.51, -0.41, 0.83, -1.37, -0.7, 0.53, 0.92, -0.53, 0.31, 0.81, -1.4, -1.49, -0.13, 0.2, -0.02, 3.39, -0.16, 0.02, 0.4, -0.02, 0.33 )
p4 <- c( 35.1, 2.7, 33.3, 64.7, 53, 66.1, 94.9, 12.6, 26.1, 100, 5.3, 28.6, 57.4, 54.6, 98.1, 56, 23.7, 80.7, 60, 36.3, 8.4, 15.9, 5.4, 14, 52.2, 15.9, 66, 79.8, 14.5, 56.6, 61.8, 36.1, 78.5, 25.9, 49.9, 67.5, 53.1, 0.8, 33.4, 95.3 )
py <- c( 12.269, 14.998, 6.309, 14.961, 10.362, 13.453, 8.038, 15.396, 10.626, 11.474, 8.94, 10.355, 10.145, 11.006, 14.75, 11.26, 6.684, 11.508, 14.296, 13.354, 10.717, 13.589, 12.245, 7.876, 10.101, 15.353, 12.431, 11.223, 11.648, 15.252, 13.726, 6.197, 14.211, 11.632, 10.323, 9.582, 12.895, 12.575, 14.558, 17.063 )
pfold <- c( 3, 5, 5, 4, 1, 5, 3, 2, 3, 5, 3, 4, 5, 2, 1, 1, 2, 2, 1, 4, 5, 4, 2, 4, 3, 2, 4, 3, 4, 1, 1, 5, 4, 3, 5, 1, 2, 3, 1, 2 )
q1 <- c( 5.05, 0.22, 6.97, 5.49, 6.93, 8.66, 9.75, 0.35, 9.55, 8.11, 3.95, 2.95, 3.14, 5.11, 1.44, 4.76, 8.82, 2.08, 8.17, 7.47, 1.75, 7.52, 10, 0.01, 7.82, 9.22, 8.21, 3.86, 1.54, 9.75, 6.58, 5.38, 4.86, 6.56, 5, 4.25, 5.18, 1.67, 9.72, 0.14, 6.33, 0.27, 5.19, 8.08, 8.7, 3.44, 8.59, 2.13, 9.58, 5.29, 3.24, 4.76, 2.45, 3.71, 9.8, 0.42, 4.76, 9.94, 0.48, 7.19, 5.93, 5.46, 3.81 )
q2 <- c( 0.21, -0.96, -1.77, 1.87, -0.42, -0.83, 0.85, 0.58, -0.72, 2.06, 1.61, 0.61, 0.55, -1.3, -1.44, -0.08, -1.88, 0.14, -0.53, 1.26, 0.14, 0.06, -0.34, -0.01, 0.24, -0.74, 0.07, 0.02, -1.63, -2.3, -0.16, 0.53, 0.66, 1.54, 0.55, 1.64, 1.71, 0.21, 1.33, 0, -0.24, -1.75, -0.37, -1.84, 0.63, 0.26, -1.29, -1.4, -0.01, 0.32, 0.66, -0.55, -0.74, -0.35, -0.75, -1.06, -0.56, -1.48, 0.16, -1.87, 0.24, -0.85, 0.39 )
qy <- c( 3.41, 2.761, 9.9, 3.953, 4.525, 10.286, 8.928, 0.927, 8.982, 4.483, 2.39, 0.987, 2.769, 6.655, 4.776, 4.03, 8.035, 3.821, 8.983, 5.024, 1.271, 5.578, 6.65, 1.082, 4.2, 5.95, 4.954, 3.669, 3.039, 8.272, 5.022, 3.397, 4.982, 5.583, 4.234, 2.668, 2.101, 1.503, 7.347, 0.698, 5.289, 2.664, 3.03, 8.02, 6.227, 2.29, 8.854, 4.555, 7.371, 3.182, 2.566, 3.788, 4.35, 3.553, 9.515, 2.849, 2.555, 8.422, 0.784, 10.029, 6.437, 5.871, 3.518 )
# ----------------------------------------------------------------- helpers ---
num <- function(v) ifelse(is.na(v), "NULL", sprintf("%.17g", v))
values_sql <- function(df) {
  rows <- apply(df, 1, function(r) paste0("(", paste(r, collapse = ", "), ")"))
  paste(rows, collapse = ",\n")
}
checks <- function(comment, from, exprs, refs, tol) {
  stopifnot(length(exprs) == length(refs))
  out <- character()
  idx <- seq_along(exprs)
  for (chunk in split(idx, ceiling(idx / 6))) {
    cols <- vapply(chunk, function(i) {
      sprintf("abs((%s) - (%s)) <= %g * (1 + abs(%s))", exprs[i], num(refs[i]), tol, num(refs[i]))
    }, "")
    out <- c(out, paste0("# ", comment), paste0("query ", strrep("I", length(chunk))),
      paste0("SELECT\n    ", paste(cols, collapse = ",\n    ")), paste0("FROM ", from, ";"), "----",
      paste(rep("true", length(chunk)), collapse = "\t"), "")
  }
  out
}
header <- function(name, desc) {
  c(sprintf("# name: test/sql/reference/%s", name), sprintf("# description: %s", desc), "# group: [reference]",
    "#", "# GENERATED by validation/generators/make_reference_paths_inference.R -- do not edit",
    "# by hand; change the generator and re-run it instead.", sprintf("# Reference: %s", R.version.string), "",
    "require anofox_statistics", "")
}
write_test <- function(name, lines) {
  writeLines(lines, file.path(out_dir, name))
  cat("wrote", file.path(out_dir, name), "\n")
}
table_sql <- function(name, cols, df) {
  c("statement ok", sprintf("CREATE TABLE %s (%s);", name, cols), "", "statement ok",
    paste0("INSERT INTO ", name, " VALUES\n", values_sql(df), ";"), "")
}
sql_list <- function(v) paste0("[", paste(num(v), collapse = ", "), "]")

# ---------------------------------------------------- glmnet-objective refs ---
# Copied from anofox-regression tests/r_scripts/generate_reg_path_validation.R.
glmnet_ref <- function(x, y, lambdas, alpha, standardize, intercept = TRUE, tol = 1e-26) {
  n <- nrow(x); p <- ncol(x)
  xm <- if (intercept) colMeans(x) else rep(0, p)
  ym <- if (intercept) mean(y) else 0
  xc <- sweep(x, 2, xm)
  xs <- if (standardize) sqrt(colMeans(xc^2)) else rep(1, p)
  X <- sweep(xc, 2, xs, "/")
  sy <- sqrt(mean((y - ym)^2))
  Y <- (y - ym) / sy
  v <- colMeans(X^2)
  B <- matrix(0, p, length(lambdas))
  if (alpha == 0) {
    for (l in seq_along(lambdas)) {
      B[, l] <- solve(crossprod(X) / n + diag(lambdas[l] / sy, p), crossprod(X, Y) / n)
    }
  } else {
    soft <- function(z, g) sign(z) * max(abs(z) - g, 0)
    b <- rep(0, p); r <- Y
    for (l in order(-lambdas)) {
      lam <- lambdas[l] / sy
      repeat {
        dmax <- 0
        for (j in 1:p) {
          g <- sum(X[, j] * r) / n + v[j] * b[j]
          new <- soft(g, lam * alpha) / (v[j] + lam * (1 - alpha))
          d <- new - b[j]
          if (d != 0) { r <- r - d * X[, j]; b[j] <- new; dmax <- max(dmax, v[j] * d^2) }
        }
        if (dmax < tol) break
      }
      B[, l] <- b
    }
  }
  beta <- B * sy / xs
  list(beta = beta, a0 = ym - colSums(beta * xm))
}
lambda_grid_ref <- function(x, y, alpha, standardize, nlambda = 100, ratio = 1e-4) {
  n <- nrow(x)
  xc <- sweep(x, 2, colMeans(x))
  if (standardize) xc <- sweep(xc, 2, sqrt(colMeans(xc^2)), "/")
  lmax <- max(abs(crossprod(xc, y - mean(y)))) / (n * max(alpha, 1e-3))
  exp(seq(log(lmax), log(lmax * ratio), length.out = nlambda))
}
cv_ref <- function(x, y, lambdas, foldid, alpha, standardize) {
  K <- max(foldid)
  cvraw <- matrix(NA, K, length(lambdas)); N <- numeric(K)
  for (k in 1:K) {
    tr <- foldid != k
    f <- glmnet_ref(x[tr, , drop = FALSE], y[tr], lambdas, alpha, standardize)
    pred <- sweep(x[!tr, , drop = FALSE] %*% f$beta, 2, f$a0, "+")
    cvraw[k, ] <- colMeans((y[!tr] - pred)^2)
    N[k] <- sum(!tr)
  }
  cvm <- apply(cvraw, 2, weighted.mean, w = N)
  cvsd <- sqrt(apply(scale(cvraw, cvm, FALSE)^2, 2, weighted.mean, w = N) / (K - 1))
  lambda.min <- max(lambdas[cvm <= min(cvm)])
  idmin <- match(lambda.min, lambdas)
  lambda.1se <- max(lambdas[cvm <= (cvm + cvsd)[idmin]])
  list(cvm = cvm, cvsd = cvsd, imin = idmin, i1se = match(lambda.1se, lambdas))
}

# =============================================================================
# 1. Regularisation paths and CV (#154)
# =============================================================================
X3 <- cbind(p1, p2, p3, p4); n3 <- nrow(X3); P <- ncol(X3)
L <- header("regularization_path.test",
  "ridge/elasticnet/lars_path_agg and ridge/elasticnet_cv_agg against glmnet's documented objective, grid and cv.glmnet aggregates, grpreg and lm")
L <- c(L, table_sql("d3", "id INTEGER, x1 DOUBLE, x2 DOUBLE, x3 DOUBLE, x4 DOUBLE, y DOUBLE, fold BIGINT",
                    data.frame(seq_along(py), p1, p2, p3, p4, py, pfold)))
# element (l, j) of a path: lambda l (1-based), term j (0 = intercept)
el <- function(l, j) sprintf("r[%d].estimate", (l - 1) * (P + 1) + j + 1)
path_block <- function(label, call, lam, fit, tol) {
  src <- sprintf("(SELECT %s AS r FROM d3) s", call)
  exprs <- unlist(lapply(seq_along(lam), function(l) c(el(l, 0), sapply(1:P, function(j) el(l, j)))))
  refs <- unlist(lapply(seq_along(lam), function(l) c(fit$a0[l], fit$beta[, l])))
  out <- checks(sprintf("%s: (intercept, x1..x4) per lambda", label), src, exprs, refs, tol)
  c(out, checks(sprintf("%s: index_value is lambda", label), src,
    sprintf("r[%d].index_value", (seq_along(lam) - 1) * (P + 1) + 1), lam, 1e-12))
}
L <- c(L, "# Path elements are (Intercept), x1..x4 for each lambda in the order given.", "")
lam_r <- c(5, 1, 0.2, 0.01)
for (std in c(FALSE, TRUE)) {
  L <- c(L, path_block(sprintf("ridge, standardize = %s: glmnet objective with alpha = 0 (closed form)", std),
    sprintf("ridge_path_agg(y, [x1, x2, x3, x4], {'lambdas': %s, 'standardize': %s})", sql_list(lam_r), tolower(std)),
    lam_r, glmnet_ref(X3, py, lam_r, 0, std), 1e-9))
}
g_en <- lambda_grid_ref(X3, py, 0.5, FALSE, nlambda = 10)
L <- c(L, path_block("elastic net l1_ratio = 0.5 on glmnet's default grid (n_lambda = 10): coordinate descent to 1e-26",
  "elasticnet_path_agg(y, [x1, x2, x3, x4], {'l1_ratio': 0.5, 'n_lambda': 10})", g_en,
  glmnet_ref(X3, py, g_en, 0.5, FALSE), 1e-6))
g_la <- lambda_grid_ref(X3, py, 1, TRUE, nlambda = 10)
L <- c(L, path_block("lasso (l1_ratio = 1), standardize = true, default grid (n_lambda = 10)",
  "elasticnet_path_agg(y, [x1, x2, x3, x4], {'l1_ratio': 1.0, 'standardize': true, 'n_lambda': 10})", g_la,
  glmnet_ref(X3, py, g_la, 1, TRUE), 1e-6))
# grpreg cross-check: with unit 1/n-variance y glmnet's internal y scaling is the identity.
ys <- (py - mean(py)) / sqrt(mean((py - mean(py))^2))
g_gr <- lambda_grid_ref(X3, ys, 1, TRUE, nlambda = 8, ratio = 1e-3)
gr <- coef(grpreg(X3, ys, group = 1:P, penalty = "grLasso", lambda = g_gr, eps = 1e-12, max.iter = 1e7))
L <- c(L, table_sql("d3s", "id INTEGER, ys DOUBLE", data.frame(seq_along(ys), ys)))
L <- c(L, path_block("lasso on unit-variance y vs grpreg(penalty = 'grLasso', singleton groups) (independent solver)",
  sprintf("elasticnet_path_agg(ys, [x1, x2, x3, x4], {'l1_ratio': 1.0, 'standardize': true, 'lambdas': %s} ORDER BY d3.id)", sql_list(g_gr)),
  g_gr, list(a0 = gr[1, ], beta = gr[-1, ]), 1e-5) |> gsub(pattern = "FROM d3\\) s", replacement = "FROM d3 JOIN d3s USING (id)) s"))

# CV with explicit folds
cv_block <- function(label, call, lam, cv) {
  src <- sprintf("(SELECT %s AS r FROM d3) s", call)
  k <- seq_along(lam)
  out <- checks(sprintf("%s: cv_mean (cvm), cv_se (cvsd)", label), src,
    c(sprintf("r[%d].cv_mean", k), sprintf("r[%d].cv_se", k)), c(cv$cvm, cv$cvsd), 1e-7)
  c(out, sprintf("# %s: lambda.min is lambda %d, lambda.1se is lambda %d (getOptcv.glmnet)", label, cv$imin, cv$i1se),
    "query II", sprintf("SELECT list_position([e.is_min FOR e IN r], true), list_position([e.is_1se FOR e IN r], true) FROM %s;", src),
    "----", sprintf("%d\t%d", cv$imin, cv$i1se), "")
}
L <- c(L, "# cv.glmnet aggregates with the folds given (fold column).", "")
L <- c(L, cv_block("elasticnet_cv_agg l1_ratio = 0.5",
  sprintf("elasticnet_cv_agg(y, [x1, x2, x3, x4], fold, {'l1_ratio': 0.5, 'lambdas': %s})", sql_list(g_en)),
  g_en, cv_ref(X3, py, g_en, pfold, 0.5, FALSE)))
g_ri <- lambda_grid_ref(X3, py, 0, TRUE, nlambda = 12)
L <- c(L, cv_block("ridge_cv_agg standardize = true, default grid (n_lambda = 12)",
  "ridge_cv_agg(y, [x1, x2, x3, x4], fold, {'standardize': true, 'n_lambda': 12})", g_ri,
  cv_ref(X3, py, g_ri, pfold, 0, TRUE)))

# LARS / Lasso-LARS: knots from the extension. index_value is upstream's alpha
# (FittedLars::alphas), max_j |x~_j'(y - yhat)| on the centred design (columns
# scaled to unit norm with standardize), i.e. n * lambda (glmnet, standardize =
# false) or sqrt(n) * lambda (glmnet, standardize = true) of the lasso objective.
lars_knots <- function(lars_sql) {
  q <- paste0("LOAD '", extension, "'; CREATE TABLE d3 AS SELECT * FROM (VALUES ",
    paste0("(", p1, ", ", p2, ", ", p3, ", ", p4, ", ", py, ")", collapse = ", "),
    ") t(x1, x2, x3, x4, y); COPY (SELECT DISTINCT e.index_value AS l FROM (SELECT unnest(", lars_sql,
    ") AS e FROM d3) ORDER BY l DESC) TO '/dev/stdout' (HEADER false);")
  k <- as.numeric(system2(duckdb_bin, c("-unsigned", "-c", shQuote(q)), stdout = TRUE))
  stopifnot(length(k) >= 3)
  k
}
xc <- sweep(X3, 2, colMeans(X3))
fl <- coef(lm(py ~ X3))
for (std in c(FALSE, TRUE)) {
  lars_sql <- sprintf("lars_path_agg(y, [x1, x2, x3, x4], {'standardize': %s})", tolower(std))
  knots <- lars_knots(lars_sql)
  scale <- if (std) sqrt(n3) else n3
  xn <- if (std) sweep(xc, 2, sqrt(colSums(xc^2)), "/") else xc
  amax <- max(abs(crossprod(xn, py - mean(py))))
  # The last knot is (numerically) 0: the least-squares fit.
  inner <- knots[knots > 1e-8 * knots[1]]
  lasso_at <- glmnet_ref(X3, py, inner / scale, 1, std)
  src <- sprintf("(SELECT %s AS r FROM d3) s", lars_sql)
  L <- c(L, sprintf("# ---- lars_path_agg (method 'lasso', standardize %s) ----", tolower(std)),
    sprintf("# %d knots read from the extension (%s).", length(knots), paste(signif(knots, 6), collapse = ", ")),
    sprintf("# The first knot is max|x~'(y - mean(y))| with all coefficients 0; the coefficients at every knot are the lasso solution at lambda = alpha / %s (glmnet objective, alpha = 1, standardize = %s, coordinate descent); the last knot is coef(lm(y ~ x)).", if (std) "sqrt(n)" else "n", std), "")
  L <- c(L, checks("first knot: alpha_max", src, "r[1].index_value", amax, 1e-10))
  L <- c(L, "query I", sprintf("SELECT DISTINCT e.index_name FROM (SELECT unnest(r) AS e FROM %s);", src), "----", "alpha", "")
  L <- c(L, checks("lasso solution at every knot", src,
    unlist(lapply(seq_along(inner), function(l) c(el(l, 0), sapply(1:P, function(j) el(l, j))))),
    unlist(lapply(seq_along(inner), function(l) c(lasso_at$a0[l], lasso_at$beta[, l]))), 1e-6))
  nk <- length(knots)
  L <- c(L, checks("last knot: coef(lm(y ~ x1 + x2 + x3 + x4))", src, c(el(nk, 0), sapply(1:P, function(j) el(nk, j))), fl, 1e-8))
}
write_test("regularization_path.test", L)

# =============================================================================
# 2. Quantile-regression inference (#157)
# =============================================================================
L <- header("quantile_inference.test",
  "quantile_fit_agg inference, quantile_process_agg and quantile_tidy_by against quantreg::rq / summary.rq")
L <- c(L, table_sql("d4", "id INTEGER, q1 DOUBLE, q2 DOUBLE, qy DOUBLE", data.frame(seq_along(qy), q1, q2, qy)))
L <- c(L, "# n = 63, so n * tau is not an integer for the taus used and rq's solution is unique. The taus",
  "# give an odd number (h + 1) of points to the median regression behind the 'iid' sparsity, so",
  "# that regression has a unique slope too (with an even count R and upstream may pick different",
  "# optimal slopes).",
  "# summary.rq: statistic = estimate / se, two-sided t p-value with n - 3 df, CI estimate -/+ qt * se.", "")
m4 <- length(qy); tq <- qt(0.975, m4 - 3)
for (tau in c(0.2, 0.5, 0.8)) {
  f <- rq(qy ~ q1 + q2, tau = tau)
  for (se in c("iid", "nid")) {
    s <- summary(f, se = se)$coefficients
    src <- sprintf("(SELECT quantile_fit_agg(qy, [q1, q2], {'tau': %g, 'se': '%s'}) AS r FROM d4) s", tau, se)
    L <- c(L, checks(sprintf("tau = %g: coef(rq) and summary.rq(se = '%s') intercept and slopes", tau, se), src,
      c("r.intercept", "r.coefficients[1]", "r.coefficients[2]", "r.intercept_std_error", "r.std_errors[1]", "r.std_errors[2]"),
      c(s[, 1], s[, 2]), 1e-6))
    L <- c(L, checks(sprintf("tau = %g, se = '%s': t values, p values, 95%% CI", tau, se), src,
      c("r.intercept_statistic", "r.t_values[1]", "r.t_values[2]", "r.intercept_p_value", "r.p_values[1]", "r.p_values[2]",
        "r.intercept_conf_low", "r.conf_low[1]", "r.conf_low[2]", "r.intercept_conf_high", "r.conf_high[1]", "r.conf_high[2]"),
      c(s[, 3], s[, 4], s[, 1] - tq * s[, 2], s[, 1] + tq * s[, 2]), 1e-6))
  }
}
ptaus <- 1:9 / 10
pc <- coef(rq(qy ~ q1 + q2, tau = ptaus))
pse <- sapply(ptaus, function(t) summary(rq(qy ~ q1 + q2, tau = t), se = "nid")$coefficients[, 2])
src <- "(SELECT quantile_process_agg(qy, [q1, q2]) AS r FROM d4) s"
k <- seq_len(27)
L <- c(L, checks("quantile_process_agg default taus 1:9/10: coef(rq(qy ~ q1 + q2, tau = 1:9/10))", src,
  sprintf("r[%d].estimate", k), as.vector(pc), 1e-6))
L <- c(L, checks("quantile_process_agg: summary.rq(se = 'nid') standard errors per tau", src,
  sprintf("r[%d].std_error", k), as.vector(pse), 1e-6))
L <- c(L, checks("quantile_process_agg: index_value is tau", src, sprintf("r[%d].index_value", 3 * (1:9) - 2), ptaus, 1e-12))
set.seed(1)
fb <- summary(rq(qy ~ q1 + q2, tau = 0.5), se = "boot", bsmethod = "xy", R = 4000)$coefficients[, 2]
L <- c(L, "# xy-pair bootstrap: the generator differs from R's, so only a statistical check:",
  sprintf("# 500 seeded resamples within 25%% of summary.rq(se = 'boot', bsmethod = 'xy', R = 4000) = %s.", paste(signif(fb, 6), collapse = ", ")),
  "query III",
  paste0("SELECT abs(r.intercept_std_error / ", num(fb[1]), " - 1) < 0.25, abs(r.std_errors[1] / ", num(fb[2]),
    " - 1) < 0.25, abs(r.std_errors[2] / ", num(fb[3]), " - 1) < 0.25"),
  "FROM (SELECT quantile_fit_agg(qy, [q1, q2], {'se': 'boot', 'n_boot': 500, 'seed': 7}) AS r FROM d4) s;",
  "----", "true\ttrue\ttrue", "")
s5 <- summary(rq(qy ~ q1 + q2, tau = 0.5), se = "nid")$coefficients
L <- c(L, checks("quantile_tidy_by computes 'nid' inference by default (summary.rq(se = 'nid') at tau = 0.5)",
  "(SELECT list(std_error ORDER BY term) AS se FROM quantile_tidy_by('d4', 1, qy, [q1, q2])) s",
  c("se[1]", "se[2]", "se[3]"), s5[, 2], 1e-6))
write_test("quantile_inference.test", L)

# =============================================================================
# 3. Predictions with standard errors (#158)
# =============================================================================
L <- header("predict_se.test",
  "predict(model, x, {'se_fit', 'interval', 'level'}) against R predict.lm / predict.glm(se.fit = TRUE)")
L <- c(L, table_sql("d1", "id INTEGER, y DOUBLE, x1 DOUBLE, x2 DOUBLE, w DOUBLE", data.frame(seq_along(y), y, x1, x2, w)))
L <- c(L, table_sql("d2", "id INTEGER, z1 DOUBLE, z2 DOUBLE, cnt DOUBLE, bin DOUBLE, gam DOUBLE",
                    data.frame(seq_along(z1), z1, z2, cnt, bin, gam)))
nd <- data.frame(x1 = c(3, 9, 12), x2 = c(4, 1, 8))
lin_block <- function(label, fit, model_sql) {
  out <- character()
  for (i in seq_len(nrow(nd))) {
    xs <- sprintf("[%s, %s]", num(nd$x1[i]), num(nd$x2[i]))
    pc <- predict(fit, nd[i, ], se.fit = TRUE, interval = "confidence", level = 0.95)
    pp <- predict(fit, nd[i, ], interval = "prediction", level = 0.90)
    src <- sprintf("(SELECT predict(m, %s, {'se_fit': true, 'interval': 'confidence'}) AS c, predict(m, %s, {'interval': 'prediction', 'level': 0.9}) AS p FROM (SELECT %s AS m FROM d1)) s", xs, xs, model_sql)
    out <- c(out, checks(sprintf("%s at x = %s: predict(se.fit = TRUE, interval = 'confidence'); interval = 'prediction', level = 0.9", label, xs), src,
      c("c.yhat", "c.se_fit", "c.yhat_lower", "c.yhat_upper", "p.yhat_lower", "p.yhat_upper"),
      c(pc$fit[1, "fit"], pc$se.fit, pc$fit[1, "lwr"], pc$fit[1, "upr"], pp[1, "lwr"], pp[1, "upr"]), 1e-9))
  }
  out
}
L <- c(L, lin_block("OLS lm(y ~ x1 + x2)", lm(y ~ x1 + x2), "ols_fit_agg(y, [x1, x2], {'compute_inference': true})"))
L <- c(L, "# WLS: predict.lm's default prediction variance is for a new observation of weight 1.", "")
L <- c(L, suppressWarnings(lin_block("WLS lm(y ~ x1 + x2, weights = w)", lm(y ~ x1 + x2, weights = w),
  "wls_fit_agg(y, [x1, x2], w, {'compute_inference': true})")))
ctl <- glm.control(epsilon = 1e-14, maxit = 100)
nz <- data.frame(z1 = c(0.5, -0.8), z2 = c(1.2, 0.3))
glm_block <- function(label, fit, model_sql, linkinv) {
  out <- character(); zq <- qnorm(0.975)
  for (i in seq_len(nrow(nz))) {
    zs <- sprintf("[%s, %s]", num(nz$z1[i]), num(nz$z2[i]))
    pl <- predict(fit, nz[i, ], type = "link", se.fit = TRUE)
    pr <- predict(fit, nz[i, ], type = "response", se.fit = TRUE)
    lo <- pl$fit - zq * pl$se.fit; hi <- pl$fit + zq * pl$se.fit
    src <- sprintf("(SELECT predict(m, %s, {'type': 'link', 'interval': 'confidence'}) AS l, predict(m, %s, {'se_fit': true, 'interval': 'confidence'}) AS r FROM (SELECT %s AS m FROM d2)) s", zs, zs, model_sql)
    out <- c(out, checks(sprintf("%s at z = %s: predict.glm(type = 'link' / 'response', se.fit = TRUE); interval eta -/+ qnorm(.975) se mapped by the inverse link", label, zs), src,
      c("l.yhat", "l.se_fit", "l.yhat_lower", "l.yhat_upper", "r.yhat", "r.se_fit", "r.yhat_lower", "r.yhat_upper"),
      c(pl$fit, pl$se.fit, lo, hi, pr$fit, pr$se.fit, linkinv(lo), linkinv(hi)), 1e-6))
  }
  out
}
L <- c(L, "# GLM fits use glm.control(epsilon = 1e-14); the extension {'tolerance': 1e-12}.", "")
L <- c(L, glm_block("Poisson glm(cnt ~ z1 + z2)", glm(cnt ~ z1 + z2, family = poisson, control = ctl),
  "poisson_fit_agg(cnt, [z1, z2], {'compute_inference': true, 'tolerance': 1e-12})", exp))
L <- c(L, glm_block("Binomial glm(bin ~ z1 + z2)", glm(bin ~ z1 + z2, family = binomial, control = ctl),
  "binomial_fit_agg(bin, [z1, z2], {'compute_inference': true, 'tolerance': 1e-12})", plogis))
L <- c(L, "# Gamma: the covariance carries the Pearson dispersion, as vcov(glm).", "")
L <- c(L, glm_block("Gamma glm(gam ~ z1 + z2, family = Gamma('log'))", glm(gam ~ z1 + z2, family = Gamma(link = "log"), control = ctl),
  "gamma_fit_agg(gam, [z1, z2], {'compute_inference': true, 'tolerance': 1e-12})", exp))
write_test("predict_se.test", L)

# =============================================================================
# 4. Partial residuals and added-variable coordinates (#158)
# =============================================================================
L <- header("partial_residuals.test",
  "<model>_partial_residuals_by against R residuals(type = 'partial') (car::crPlots) and lsfit residuals (car::avPlots)")
L <- c(L, table_sql("d1", "id INTEGER, y DOUBLE, x1 DOUBLE, x2 DOUBLE, w DOUBLE, g INTEGER", data.frame(seq_along(y), y, x1, x2, w, 1)))
L <- c(L, table_sql("d2", "id INTEGER, z1 DOUBLE, z2 DOUBLE, cnt DOUBLE, g INTEGER", data.frame(seq_along(z1), z1, z2, cnt, 1)))
L <- c(L, "# row_id is the source row number (the insertion order of these small tables).", "")
pr_block <- function(label, fit, call, X, wt = NULL, av = TRUE, rows = c(1, 2, 7, 13, 20)) {
  pres <- residuals(fit, type = "partial")
  out <- character()
  for (j in 1:2) {
    others <- X[, -j, drop = FALSE]
    ex <- lsfit(others, X[, j], wt = wt)$residuals
    ey <- lsfit(others, fit$y %||% model.response(model.frame(fit)), wt = wt)$residuals
    for (i in rows) {
      src <- sprintf("(SELECT * FROM %s WHERE term_index = %d AND row_id = %d) s", call, j, i)
      if (av) {
        out <- c(out, checks(sprintf("%s, term %d, row %d: residuals(type = 'partial')[%d, %d]; lsfit residuals of x_%d and y on the other column", label, j, i, i, j, j), src,
          c("partial_residual", "x_residual", "y_residual"), c(pres[i, j], ex[i], ey[i]), 1e-8))
      } else {
        out <- c(out, checks(sprintf("%s, term %d, row %d: residuals(type = 'partial')[%d, %d]", label, j, i, i, j), src,
          "partial_residual", pres[i, j], 1e-6))
      }
    }
  }
  out
}
`%||%` <- function(a, b) if (is.null(a)) b else a
X1 <- cbind(x1, x2)
L <- c(L, pr_block("OLS lm(y ~ x1 + x2)", lm(y ~ x1 + x2), "ols_partial_residuals_by('d1', g, y, [x1, x2])", X1))
L <- c(L, "# WLS: raw residuals plus the component (unweighted column means); added-variable", "# coordinates from weighted lsfit (car::avPlots).", "")
L <- c(L, pr_block("WLS lm(y ~ x1 + x2, weights = w)", lm(y ~ x1 + x2, weights = w),
  "wls_partial_residuals_by('d1', g, y, [x1, x2], w)", X1, wt = w))
fp <- glm(cnt ~ z1 + z2, family = poisson, control = ctl)
L <- c(L, "# Poisson: working residual plus the component (residuals.glm(type = 'partial')); no added-variable coordinates.", "")
L <- c(L, pr_block("Poisson glm(cnt ~ z1 + z2)", fp,
  "poisson_partial_residuals_by('d2', g, cnt, [z1, z2], options := {'tolerance': 1e-12})", cbind(z1, z2), av = FALSE, rows = c(1, 5, 11, 30)))
write_test("partial_residuals.test", L)
