#!/usr/bin/env Rscript
# =============================================================================
# make_reference_pentrace.R
#
# Generates test/sql/reference/pentrace_reference.test (#323): pentrace_agg and
# penalized_ols_fit_agg against
#
#   pentrace_agg           rms::pentrace (results.all: df, aic, bic, aic.c;
#                          the best penalty for which = "aic.c" / "aic" /
#                          "bic") on ols, binary lrm and ordinal lrm fits, and
#                          orm(family = "probit", penalty = ) fits for a
#                          non-logistic link (see below)
#   penalized_ols_fit_agg  rms::ols(penalty = , var.penalty = "simple" /
#                          "sandwich"): coefficients, var, Model L.R., d.f.,
#                          R2, Sigma, effective.df.diagonal
#
#   Rscript validation/generators/make_reference_pentrace.R   # from repo root
#
# Requires: rms. The rcs() / interaction / dummy design columns come from the
# rms fits' design matrices (x = TRUE) and are written into the test table with 17
# significant digits, so the extension fits exactly the same design.
#
# rms::pentrace refits an orm model with lrm.fit (logistic) whatever the
# family, so it is no reference for the probit link: the probit rows are built
# from orm(family = "probit", penalty = lambda) fits directly (Model L.R. and
# d.f. of each fit, aic = LR - 2 df, bic = LR - df log n, aic.c as rms).
# =============================================================================
suppressPackageStartupMessages(library(rms))

out_dir <- "test/sql/reference"

set.seed(323)
n <- 120
age <- round(runif(n, 20, 70), 1)
bp <- round(rnorm(n, 120, 15), 1)
sex <- rbinom(n, 1, 0.5)
race <- factor(sample(c("a", "b", "c"), n, TRUE))
y <- round(0.05 * age + 0.002 * (age - 45)^2 + 0.03 * bp + 0.8 * sex + 0.01 * bp * sex +
           0.3 * (race == "b") + rnorm(n), 3)
yb <- rbinom(n, 1, plogis(-6 + 0.08 * age + 0.02 * bp + 0.5 * sex))
yo <- cut(y, quantile(y, c(0, 0.3, 0.6, 1)), include.lowest = TRUE, labels = FALSE)

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
sql_list <- function(v) paste0("[", paste(v, collapse = ", "), "]")
sql_str <- function(v) sql_list(paste0("'", v, "'"))
sql_bool <- function(v) sql_list(ifelse(v, "true", "false"))

design <- function(f, prefix) {
  X <- unclass(f$x)
  colnames(X) <- paste0(prefix, seq_len(ncol(X)))
  X
}

# Labels, nonlinear flags, interactions and categorical terms of an rms fit.
term_spec <- function(f) {
  d <- f$Design
  labels <- character(); nonlin <- logical(); ia <- character(); cat <- character()
  for (k in seq_along(d$name)) {
    nm <- d$name[k]
    m <- length(d$nonlinear[[nm]])
    labels <- c(labels, rep(nm, m))
    nonlin <- c(nonlin, d$nonlinear[[nm]])
    if (d$assume.code[k] == 5) cat <- c(cat, nm)
    if (d$assume.code[k] == 9) {
      parts <- strsplit(nm, " * ", fixed = TRUE)[[1]]
      ia <- c(ia, sprintf("'%s': ['%s', '%s']", nm, parts[1], parts[2]))
    }
  }
  opts <- sprintf("'terms': %s, 'nonlinear': %s", sql_str(labels), sql_bool(nonlin))
  if (length(ia)) opts <- paste0(opts, ", 'interactions': MAP {", paste(ia, collapse = ", "), "}")
  if (length(cat)) opts <- paste0(opts, ", 'categorical': ", sql_str(cat))
  list(labels = labels, nonlinear = nonlin, opts = opts)
}

# Compare the trace rows `r` (a data frame with the penalty columns, df, aic,
# bic, aic.c) with the pentrace_agg list `t` of `from`; `best` is the 1-based
# index of the best row.
trace_checks <- function(comment, from, r, best, tol) {
  out <- c(paste0("# ", comment, ": row count and best row"), "query II",
           sprintf("SELECT len(t), list_position(list_transform(t, lambda r: r.is_best), true) FROM %s;", from),
           "----", sprintf("%d\t%d", nrow(r), best), "")
  pen <- intersect(c("penalty", "simple", "nonlinear", "interaction", "nonlinear.interaction"), names(r))
  for (i in seq_len(nrow(r))) {
    exprs <- sprintf(c("t[%d].df", "t[%d].aic", "t[%d].bic", "t[%d].aic_c", "t[%d].lr_chisq"), i)
    refs <- c(r$df[i], r$aic[i], r$bic[i], r$aic.c[i], r$aic[i] + 2 * r$df[i])
    for (p in pen) {
      field <- switch(p, penalty = "simple", nonlinear.interaction = "nonlinear_interaction", p)
      exprs <- c(exprs, sprintf("t[%d].%s", i, field)); refs <- c(refs, r[[p]][i])
    }
    out <- c(out, checks(sprintf("%s: row %d", comment, i), from, exprs, refs, tol))
  }
  out
}

best_index <- function(p) {
  r <- p$results.all
  pen <- p$penalty
  if (is.null(names(pen))) pen <- list(penalty = pen)
  hit <- rep(TRUE, nrow(r))
  for (nm in names(pen)) {
    col <- if (nm %in% names(r)) nm else "penalty"
    hit <- hit & abs(r[[col]] - pen[[nm]]) < 1e-12
  }
  which(hit)[1]
}

# ------------------------------------------------------------------- fits ---
f1 <- ols(y ~ rcs(age, 4) + bp * sex + race, x = TRUE, y = TRUE)
f2 <- ols(y ~ rcs(age, 4) * sex + bp, x = TRUE, y = TRUE)
f3 <- ols(y ~ age + bp + sex, x = TRUE, y = TRUE)
g1 <- lrm(yb ~ rcs(age, 4) + bp * sex + race, x = TRUE, y = TRUE)
o1 <- lrm(yo ~ rcs(age, 4) + bp * sex + race, x = TRUE, y = TRUE)

X1 <- design(f1, "a"); X2 <- design(f2, "b"); X3 <- design(f3, "c")
stopifnot(max(abs(g1$x - f1$x)) == 0, max(abs(o1$x - f1$x)) == 0)
data <- data.frame(id = seq_len(n), y = y, yb = yb, yo = yo, X1, X2, X3)
cols <- setdiff(names(data), "id")

L <- c("# name: test/sql/reference/pentrace_reference.test",
  "# description: pentrace_agg and penalized_ols_fit_agg against rms::pentrace and rms::ols(penalty = ) (#323)",
  "# group: [reference]",
  "#",
  "# GENERATED by validation/generators/make_reference_pentrace.R -- do not edit",
  "# by hand; change the generator and re-run it instead.",
  sprintf("# Reference: %s, rms %s", R.version.string, as.character(packageVersion("rms"))),
  "",
  "require anofox_statistics",
  "",
  "statement ok",
  sprintf("CREATE TABLE d (id INTEGER, %s);", paste(cols, "DOUBLE", collapse = ", ")),
  "",
  "statement ok",
  paste0("INSERT INTO d VALUES\n",
         values_sql(data.frame(data$id, lapply(data[cols], function(v) sprintf("%.17g", v)))), ";"),
  "")

trace_src <- function(yv, X, opts) {
  sprintf("(SELECT pentrace_agg(%s, %s, {%s}) AS t FROM d) s", yv, sql_list(colnames(X)), opts)
}

# 1. OLS with rcs(age, 4), a linear interaction and a categorical term:
#    simple x nonlinear grid (complex.more drops nonlinear < simple)
s1 <- term_spec(f1)
grid1 <- list(simple = c(0, 2, 5), nonlinear = c(0, 2, 5, 20))
p1 <- pentrace(f1, grid1)
gopt1 <- "'penalties': {'simple': [0, 2, 5], 'nonlinear': [0, 2, 5, 20]}"
L <- c(L, trace_checks("ols, pentrace(f, list(simple = c(0, 2, 5), nonlinear = c(0, 2, 5, 20)))",
  trace_src("y", X1, paste(s1$opts, gopt1, sep = ", ")), p1$results.all, best_index(p1), 1e-8))
for (w in c("aic", "bic")) {
  pw <- pentrace(f1, grid1, which = w)
  L <- c(L, paste0("# same grid, which = '", w, "': best row"), "query I",
    sprintf("SELECT list_position(list_transform(t, lambda r: r.is_best), true) FROM %s;",
            trace_src("y", X1, sprintf("%s, %s, 'criterion': '%s'", s1$opts, gopt1, w))),
    "----", best_index(pw), "")
}

# 2. OLS with a nonlinear interaction rcs(age, 4) * sex: all four penalty types
s2 <- term_spec(f2)
grid2 <- list(simple = c(0, 1), nonlinear = c(1, 4), interaction = c(4, 8), nonlinear.interaction = c(8, 20))
p2 <- pentrace(f2, grid2)
L <- c(L, trace_checks("ols y ~ rcs(age, 4) * sex + bp, all four penalty types",
  trace_src("y", X2, paste(s2$opts, "'penalties': {'simple': [0, 1], 'nonlinear': [1, 4], 'interaction': [4, 8], 'nonlinear_interaction': [8, 20]}", sep = ", ")),
  p2$results.all, best_index(p2), 1e-8))

# 3. OLS, uniform penalty, every column its own linear term (no term options)
p3 <- pentrace(f3, c(0, 1, 5, 25, 100))
L <- c(L, trace_checks("ols y ~ age + bp + sex, pentrace(f, c(0, 1, 5, 25, 100)), default terms",
  trace_src("y", X3, "'penalties': [0, 1, 5, 25, 100]"), p3$results.all, best_index(p3), 1e-8))

# 4. Binary logistic (lrm)
pg <- pentrace(g1, c(0, 1, 2, 4, 8))
L <- c(L, trace_checks("binary lrm, pentrace(f, c(0, 1, 2, 4, 8))",
  trace_src("yb", X1, paste(s1$opts, "'penalties': [0, 1, 2, 4, 8], 'model': 'logistic'", sep = ", ")),
  pg$results.all, best_index(pg), 1e-6))

# 5. Ordinal logistic (lrm on a three-level response)
po <- pentrace(o1, c(0, 1, 2, 4, 8))
L <- c(L, trace_checks("ordinal lrm, pentrace(f, c(0, 1, 2, 4, 8))",
  trace_src("yo", X1, paste(s1$opts, "'penalties': [0, 1, 2, 4, 8], 'model': 'ordinal'", sep = ", ")),
  po$results.all, best_index(po), 1e-6))

# 6. Ordinal probit: orm(family = "probit", penalty = lambda) fits
lams <- c(0, 1, 2, 4, 8)
raw <- data.frame(yo = yo, age = age, bp = bp, sex = sex, race = race)
pr <- do.call(rbind, lapply(lams, function(l) {
  f <- orm(yo ~ rcs(age, 4) + bp * sex + race, data = raw, family = "probit", penalty = l)
  lr <- f$stats[["Model L.R."]]; df <- f$stats[["d.f."]]
  data.frame(penalty = l, df = df, aic = lr - 2 * df, bic = lr - df * log(n),
             aic.c = lr - 2 * df * (1 + (df + 1) / (n - df - 1)))
}))
L <- c(L, trace_checks("ordinal probit, orm(family = 'probit', penalty = lambda)",
  trace_src("yo", X1, paste(s1$opts, "'penalties': [0, 1, 2, 4, 8], 'model': 'ordinal', 'link': 'probit'", sep = ", ")),
  pr, which.max(pr$aic.c), 1e-6))

# 7. penalized_ols_fit_agg vs ols(penalty = list(simple = 2, nonlinear = 5))
fit_src <- function(X, opts) {
  sprintf("(SELECT penalized_ols_fit_agg(y, %s, {%s}) AS m FROM d) s", sql_list(colnames(X)), opts)
}
pen_checks <- function(comment, from, fp, spec, sandwich, tol) {
  k <- length(coef(fp))
  out <- checks(paste0(comment, ": coefficients"), from,
    c("m.intercept", sprintf("m.coefficients[%d]", seq_len(k - 1))), unname(coef(fp)), tol)
  se <- sqrt(diag(vcov(fp)))
  out <- c(out, checks(paste0(comment, ": standard errors"), from,
    c("m.intercept_std_error", sprintf("m.std_errors[%d]", seq_len(k - 1))), unname(se), tol))
  V <- vcov(fp)
  out <- c(out, checks(paste0(comment, ": vcov (row-major, intercept first)"), from,
    sprintf("m.vcov[%d]", seq_len(k * k)), as.vector(t(V)), tol))
  st <- fp$stats
  out <- c(out, checks(paste0(comment, ": Model L.R., d.f., R2, Sigma"), from,
    c("m.lr_chisq", "m.effective_df", "m.r_squared", "m.sigma", "m.n_observations"),
    c(st[["Model L.R."]], st[["d.f."]], st[["R2"]], st[["Sigma"]], n), tol))
  out <- c(out, checks(paste0(comment, ": effective.df.diagonal"), from,
    sprintf("m.effective_df_diagonal[%d]", seq_len(k)), unname(fp$effective.df.diagonal), tol))
  dag <- fp$effective.df.diagonal[-1]
  terms <- unique(spec$labels)
  by_term <- vapply(terms, function(t) sum(dag[spec$labels == t]), 0)
  out <- c(out, "# effective d.f. by term: labels", "query I",
    sprintf("SELECT m.terms = %s FROM %s;", sql_str(terms), from), "----", "true", "")
  c(out, checks(paste0(comment, ": effective d.f. by term"), from,
    sprintf("m.effective_df_by_term[%d]", seq_along(terms)), unname(by_term), tol))
}
fp1 <- ols(y ~ rcs(age, 4) + bp * sex + race, penalty = list(simple = 2, nonlinear = 5))
L <- c(L, pen_checks("ols(penalty = list(simple = 2, nonlinear = 5))",
  fit_src(X1, paste(s1$opts, "'penalty': {'simple': 2, 'nonlinear': 5}", sep = ", ")), fp1, s1, FALSE, 1e-8))
fp2 <- ols(y ~ rcs(age, 4) + bp * sex + race, penalty = list(simple = 2, nonlinear = 5), var.penalty = "sandwich")
L <- c(L, pen_checks("same, var.penalty = 'sandwich'",
  fit_src(X1, paste(s1$opts, "'penalty': {'simple': 2, 'nonlinear': 5}, 'var_penalty': 'sandwich'", sep = ", ")),
  fp2, s1, TRUE, 1e-8))
fp3 <- ols(y ~ rcs(age, 4) * sex + bp, penalty = list(simple = 1, nonlinear = 3, interaction = 6, nonlinear.interaction = 12))
L <- c(L, pen_checks("ols(y ~ rcs(age, 4) * sex + bp, penalty = all four types)",
  fit_src(X2, paste(s2$opts, "'penalty': {'simple': 1, 'nonlinear': 3, 'interaction': 6, 'nonlinear_interaction': 12}", sep = ", ")),
  fp3, s2, FALSE, 1e-8))
fp4 <- ols(y ~ age + bp + sex, penalty = 10)
L <- c(L, pen_checks("ols(y ~ age + bp + sex, penalty = 10), default terms",
  fit_src(X3, "'penalty': 10"), fp4, list(labels = paste0("x", 1:3)), FALSE, 1e-8))

writeLines(L, file.path(out_dir, "pentrace_reference.test"))
cat("wrote", file.path(out_dir, "pentrace_reference.test"), "\n")
