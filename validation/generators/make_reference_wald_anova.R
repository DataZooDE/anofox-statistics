#!/usr/bin/env Rscript
# =============================================================================
# make_reference_wald_anova.R
#
# Generates test/sql/reference/wald_anova_reference.test (#311): wald_anova and
# fastbw against
#
#   wald_anova  rms::anova.rms (F and chi-square tests, partial SS, the
#               Nonlinear / interaction chunks and the TOTAL rows)
#   fastbw      rms::fastbw (rule = "aic" / "p", type = "residual" /
#               "individual"), deletion order, tests, kept factors and the
#               approximate coefficients
#
#   Rscript validation/generators/make_reference_wald_anova.R   # from repo root
#
# Requires: rms. The rcs() / %ia% / `*` design columns come from the rms fits'
# model.matrix and are written into the test table with 17 significant digits,
# so the extension fits exactly the same design (ols_fit_agg, logistic_fit_agg)
# and reads its coefficients and vcov.
# =============================================================================
suppressPackageStartupMessages(library(rms))

out_dir <- "test/sql/reference"

set.seed(311)
n <- 80
age <- round(runif(n, 20, 70), 1)
bp <- round(rnorm(n, 120, 15), 1)
sex <- rbinom(n, 1, 0.5)
w <- round(rnorm(n), 2)
y <- round(0.05 * age + 0.002 * (age - 45)^2 + 0.03 * bp + 0.8 * sex + 0.02 * age * sex + rnorm(n), 3)
yb <- rbinom(n, 1, plogis(-6 + 0.08 * age + 0.02 * bp + 0.5 * sex))

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
sql_str <- function(v) sql_list(ifelse(is.na(v), "NULL", paste0("'", v, "'")))
sql_bool <- function(v) sql_list(ifelse(v, "true", "false"))

# Design columns of an rms fit (no intercept), named prefix1..k.
design <- function(f, prefix) {
  X <- model.matrix(f)[, -1, drop = FALSE]
  colnames(X) <- paste0(prefix, seq_len(ncol(X)))
  X
}

# The labels and nonlinear flags of an rms fit's columns, from its Design.
term_spec <- function(f) {
  d <- f$Design
  labels <- character(); nonlin <- logical()
  for (nm in d$name) {
    k <- length(d$nonlinear[[nm]])
    labels <- c(labels, rep(nm, k))
    nonlin <- c(nonlin, d$nonlinear[[nm]])
  }
  list(labels = labels, nonlinear = nonlin)
}

# Compare every row of anova.rms with the wald_anova list `t` of `from`.
anova_checks <- function(comment, from, a, test, tol) {
  m <- unclass(a)
  # The ERROR row comes with the F test only (upstream reports it for an F test).
  if (test != "F" && rownames(m)[nrow(m)] == "ERROR") m <- m[-nrow(m), , drop = FALSE]
  rn <- rownames(m)
  out <- character()
  nr <- nrow(m)
  has_error <- rn[nr] == "ERROR"
  out <- c(out, paste0("# ", comment, ": row count and labels"),
    "query I",
    sprintf("SELECT len(t) FROM %s;", from), "----", nr, "")
  labels <- sub("^REGRESSION$", "TOTAL", rn)
  out <- c(out, "query I",
    sprintf("SELECT list_transform(t, lambda r: trim(r.label)) = %s FROM %s;",
            sql_str(trimws(labels)), from), "----", "true", "")
  for (i in seq_len(nr - has_error)) {
    exprs <- c(sprintf("t[%d].df1", i)); refs <- c(m[i, "d.f."])
    if (test == "F") {
      exprs <- c(exprs, sprintf("t[%d].statistic", i), sprintf("t[%d].df2", i))
      refs <- c(refs, m[i, "F"], m[nr, "d.f."])
    } else {
      exprs <- c(exprs, sprintf("t[%d].statistic", i), sprintf("t[%d].chi_square", i))
      refs <- c(refs, m[i, "Chi-Square"], m[i, "Chi-Square"])
    }
    exprs <- c(exprs, sprintf("t[%d].p_value", i)); refs <- c(refs, m[i, "P"])
    if ("Partial SS" %in% colnames(m)) {
      exprs <- c(exprs, sprintf("t[%d].partial_ss", i), sprintf("t[%d].mean_sq", i))
      refs <- c(refs, m[i, "Partial SS"], m[i, "MS"])
    }
    out <- c(out, checks(sprintf("%s: row %d '%s'", comment, i, trimws(rn[i])), from, exprs, refs, tol))
  }
  if (has_error) {
    out <- c(out, checks(sprintf("%s: ERROR row", comment), from,
      sprintf(c("t[%d].df1", "t[%d].partial_ss", "t[%d].mean_sq"), nr),
      c(m[nr, "d.f."], m[nr, "Partial SS"], m[nr, "MS"]), tol))
  }
  out
}

# Compare rms::fastbw with the fastbw STRUCT `b` of `from`.
fastbw_checks <- function(comment, from, fb, model_labels, tol) {
  out <- character()
  res <- fb$result
  nd <- if (is.null(res)) 0 else nrow(res)
  deleted <- if (nd == 0) character() else rownames(res)
  out <- c(out, paste0("# ", comment, ": deleted and kept factors"),
    "query II",
    sprintf("SELECT list_transform(b.deleted, lambda d: d.term) = %s, b.kept_terms = %s FROM %s;",
            if (nd == 0) "[]::VARCHAR[]" else sql_str(deleted),
            if (length(fb$names.kept) == 0) "[]::VARCHAR[]" else sql_str(fb$names.kept), from),
    "----", "true\ttrue", "")
  for (i in seq_len(nd)) {
    exprs <- sprintf(c("b.deleted[%d].chi_square", "b.deleted[%d].df", "b.deleted[%d].p_value",
                       "b.deleted[%d].residual_chi_square", "b.deleted[%d].residual_df",
                       "b.deleted[%d].residual_p_value", "b.deleted[%d].aic"), i)
    out <- c(out, checks(sprintf("%s: deletion %d (%s)", comment, i, deleted[i]), from, exprs,
                         as.numeric(res[i, 1:7]), tol))
  }
  # Approximate coefficients of the reduced model: kept ones from fastbw, 0 otherwise.
  cf <- fb$coefficients
  exprs <- character(); refs <- numeric()
  if ("Intercept" %in% names(cf)) {
    exprs <- c(exprs, "b.intercept"); refs <- c(refs, cf[["Intercept"]])
  }
  kept_cols <- which(model_labels %in% fb$names.kept)
  kept_names <- setdiff(names(cf), "Intercept")
  stopifnot(length(kept_cols) == length(kept_names))
  for (k in seq_along(kept_cols)) {
    exprs <- c(exprs, sprintf("b.coefficients[%d]", kept_cols[k])); refs <- c(refs, cf[[kept_names[k]]])
  }
  for (j in setdiff(seq_along(model_labels), kept_cols)) {
    exprs <- c(exprs, sprintf("b.coefficients[%d]", j)); refs <- c(refs, 0)
  }
  c(out, checks(paste0(comment, ": approximate coefficients after deletion"), from, exprs, refs, tol))
}

# ------------------------------------------------------------------- fits ---
f1 <- ols(y ~ rcs(age, 4) + bp + sex + rcs(age, 4) %ia% sex)
f2 <- ols(y ~ rcs(age, 4) + rcs(bp, 3) + w + rcs(age, 4) %ia% rcs(bp, 3))
f3 <- ols(y ~ rcs(age, 3) * sex + w)
g1 <- lrm(yb ~ rcs(age, 4) + bp + sex + w)

X1 <- design(f1, "a"); X2 <- design(f2, "b"); X3 <- design(f3, "c"); G1 <- design(g1, "g")
data <- data.frame(id = seq_len(n), y = y, yb = yb, X1, X2, X3, G1)
cols <- setdiff(names(data), "id")

L <- c("# name: test/sql/reference/wald_anova_reference.test",
  "# description: wald_anova and fastbw against rms::anova.rms and rms::fastbw (#311)",
  "# group: [reference]",
  "#",
  "# GENERATED by validation/generators/make_reference_wald_anova.R -- do not edit",
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

anova_src <- function(agg, yv, X, spec, extra = "", ia = NULL) {
  opts <- sprintf("'nonlinear': %s", sql_bool(spec$nonlinear))
  if (!is.null(ia)) opts <- paste0(opts, ", 'interactions': ", ia)
  if (extra != "") opts <- paste0(opts, ", ", extra)
  sprintf("(SELECT wald_anova(%s(%s, %s, {'compute_inference': true}), %s, {%s}) AS t FROM d) s",
          agg, yv, sql_list(colnames(X)), sql_str(spec$labels), opts)
}

# 1. OLS, rcs(age, 4) %ia% sex (linear sex: the restricted and full products agree)
s1 <- term_spec(f1)
ia1 <- "MAP {'age * sex': ['age', 'sex']}"
L <- c(L, anova_checks("ols y ~ rcs(age, 4) + bp + sex + rcs(age, 4) %ia% sex, anova(f) (F test)",
  anova_src("ols_fit_agg", "y", X1, s1, ia = ia1), anova(f1), "F", 1e-8))
L <- c(L, anova_checks("same model, anova(f, test = 'Chisq')",
  anova_src("ols_fit_agg", "y", X1, s1, "'test': 'chisq'", ia1), anova(f1, test = "Chisq"), "Chisq", 1e-8))

# 2. OLS, restricted interaction of two splines (no doubly nonlinear products). The
#    spline products make the design ill-conditioned (the fits agree to ~1e-8), hence 1e-6.
s2 <- term_spec(f2)
L <- c(L, anova_checks("ols y ~ rcs(age, 4) + rcs(bp, 3) + w + rcs(age, 4) %ia% rcs(bp, 3)",
  anova_src("ols_fit_agg", "y", X2, s2, ia = "MAP {'age * bp': ['age', 'bp']}"), anova(f2), "F", 1e-6))

# 3. OLS, full product rcs(age, 3) * sex
s3 <- term_spec(f3)
L <- c(L, anova_checks("ols y ~ rcs(age, 3) * sex + w",
  anova_src("ols_fit_agg", "y", X3, s3, ia = "MAP {'age * sex': ['age', 'sex']}"), anova(f3), "F", 1e-8))

# 4. Logistic (chi-square tests from the MLE covariance)
sg <- term_spec(g1)
lopt <- "{'compute_inference': true, 'tolerance': 1e-12, 'max_iterations': 100}"
lsrc <- sprintf("(SELECT wald_anova(logistic_fit_agg(yb, %s, %s), %s, {'nonlinear': %s}) AS t FROM d) s",
                sql_list(colnames(G1)), lopt, sql_str(sg$labels), sql_bool(sg$nonlinear))
L <- c(L, anova_checks("lrm yb ~ rcs(age, 4) + bp + sex + w, anova(f)", lsrc, anova(g1), "Chisq", 1e-6))

# 5. fastbw
fb_src <- function(agg, yv, X, labels, aopt, opts = NULL) {
  sprintf("(SELECT fastbw(%s(%s, %s, %s), %s%s) AS b FROM d) s", agg, yv, sql_list(colnames(X)), aopt,
          sql_str(labels), if (is.null(opts)) "" else paste0(", ", opts))
}
L <- c(L, fastbw_checks("fastbw(lrm) default (rule = 'aic', type = 'residual')",
  fb_src("logistic_fit_agg", "yb", G1, sg$labels, lopt), fastbw(g1), sg$labels, 1e-6))
L <- c(L, fastbw_checks("fastbw(lrm, rule = 'p', sls = 0.5)",
  fb_src("logistic_fit_agg", "yb", G1, sg$labels, lopt, "{'rule': 'p', 'sls': 0.5}"),
  fastbw(g1, rule = "p", sls = 0.5), sg$labels, 1e-6))
L <- c(L, fastbw_checks("fastbw(lrm, type = 'individual', aics = 1, k.aic = 3)",
  fb_src("logistic_fit_agg", "yb", G1, sg$labels, lopt, "{'type': 'individual', 'aics': 1, 'k_aic': 3}"),
  fastbw(g1, type = "individual", aics = 1, k.aic = 3), sg$labels, 1e-6))
# fastbw on an ols fit with an interaction: rms treats it as one more factor.
L <- c(L, fastbw_checks("fastbw(ols y ~ rcs(age, 4) + bp + sex + rcs(age, 4) %ia% sex)",
  fb_src("ols_fit_agg", "y", X1, s1$labels, "{'compute_inference': true}"), fastbw(f1), s1$labels, 1e-8))

writeLines(L, file.path(out_dir, "wald_anova_reference.test"))
cat("wrote", file.path(out_dir, "wald_anova_reference.test"), "\n")
