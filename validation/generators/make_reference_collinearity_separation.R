#!/usr/bin/env Rscript
# =============================================================================
# make_reference_collinearity_separation.R
#
# Generates test/sql/reference/collinearity_separation.test (#306):
#
#   condition_diagnostic_agg  <- base R: kappa(X, exact = TRUE) and the
#                                Belsley-Kuh-Welsch condition indices /
#                                variance-decomposition proportions from the SVD
#                                of the unit-column-length design, the
#                                definition of perturb::colldiag(scale = TRUE,
#                                center = FALSE) (perturb is archived on CRAN,
#                                so the formula is coded here)
#   detect_separation_agg     <- detectseparation::detect_separation
#                                (glm(..., method = "detect_separation"))
#
#   Rscript validation/generators/make_reference_collinearity_separation.R
#
# Requires: detectseparation (install into a user library if missing). Data:
# datasets::longley (the classic collinear design), detectseparation's
# endometrial data (Heinze & Schemper 2002) and small literal datasets.
# =============================================================================
suppressPackageStartupMessages(library(detectseparation))

out_dir <- "test/sql/reference"
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
exact <- function(comment, sql, expected) {
  c(paste0("# ", comment), paste0("query ", strrep("I", length(expected))), sql, "----",
    paste(expected, collapse = "\t"), "")
}

# --- BKW collinearity diagnostics (perturb::colldiag definition) -------------
bkw <- function(X) {
  Xs <- sweep(X, 2, sqrt(colSums(X^2)), "/")
  s <- svd(Xs)
  d <- s$d
  phi <- sweep(s$v^2, 2, d^2, "/")          # phi[j, k] = v_jk^2 / d_k^2
  pi <- t(sweep(phi, 1, rowSums(phi), "/")) # pi[k, j]: component k, coefficient j
  list(kappa = kappa(X, exact = TRUE), d = d, ci = d[1] / d, pi = pi)
}

ll <- datasets::longley
lx <- c("GNP.deflator", "GNP", "Unemployed", "Armed.Forces", "Population", "Year")
lcols <- c("deflator", "gnp", "unemployed", "armed", "population", "year")
L <- as.matrix(ll[, lx])

cond_block <- function(label, call, X) {
  b <- bkw(X)
  p <- ncol(X)
  from <- sprintf("(SELECT %s AS r FROM longley) s", call)
  out <- exact(paste(label, "- shape"),
               sprintf("SELECT len(r.singular_values), len(r.condition_indices), len(r.variance_proportions), len(r.variance_proportions[1]), r.n_observations FROM %s;", from),
               c(p, p, p, p, nrow(X)))
  out <- c(out, checks(paste(label, "- kappa(X, exact = TRUE), scaled kappa"), from,
                       c("r.condition_number", "r.condition_number_xtx", "r.scaled_condition_number"),
                       c(b$kappa, b$kappa^2, max(b$ci)), 1e-8))
  out <- c(out, checks(paste(label, "- singular values of the unit-length design"), from,
                       sprintf("r.singular_values[%d]", 1:p), b$d, 1e-9))
  out <- c(out, checks(paste(label, "- condition indices"), from,
                       sprintf("r.condition_indices[%d]", 1:p), b$ci, 1e-8))
  ex <- character(); ref <- numeric()
  for (k in 1:p) for (j in 1:p) {
    ex <- c(ex, sprintf("r.variance_proportions[%d][%d]", k, j)); ref <- c(ref, b$pi[k, j])
  }
  out <- c(out, checks(paste(label, "- variance-decomposition proportions pi[component, coefficient]"),
                       from, ex, ref, 1e-7))
  out
}

xs <- paste0("[", paste(lcols, collapse = ", "), "]")
cond <- c(
  cond_block("longley, intercept", sprintf("condition_diagnostic_agg(%s)", xs), cbind(1, L)),
  cond_block("longley, no intercept",
             sprintf("condition_diagnostic_agg(%s, {'fit_intercept': false})", xs), L),
  cond_block("longley [unemployed, armed], intercept",
             "condition_diagnostic_agg([unemployed, armed])", cbind(1, L[, 3:4])))
sev <- function(k) if (k < 30) "well_conditioned" else if (k < 100) "moderate" else if (k < 1000) "high" else "severe"
cond <- c(cond, exact("severity uses the BKW cut-offs 30 / 100 / 1000 of the scaled condition number",
  sprintf("SELECT condition_diagnostic_agg(%s).severity, condition_diagnostic_agg([unemployed, armed]).severity FROM longley;", xs),
  c(sev(max(bkw(cbind(1, L))$ci)), sev(max(bkw(cbind(1, L[, 3:4]))$ci)))))

# --- exact separation (detectseparation) ------------------------------------
sx1 <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2.5, 7.5)
sx2 <- c(3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8)
cx1 <- c(1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 1, 4)
cx2 <- c(1, 1, 1, 1, 4, 4, 4, 4, 2, 3, 3, 2)
qx1 <- c(1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 10)
sep <- rbind(
  data.frame(g = "'complete'", x1 = sx1, x2 = sx2, y = as.integer(sx1 > 5)),
  data.frame(g = "'combination'", x1 = cx1, x2 = cx2, y = as.integer(cx1 + cx2 > 5)),
  data.frame(g = "'quasi'", x1 = qx1, x2 = sx2, y = c(0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1)),
  data.frame(g = "'overlap'", x1 = sx1, x2 = sx2, y = c(0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0)))
inf_sql <- function(v) paste0("[", paste(ifelse(v == Inf, "'inf'::DOUBLE", ifelse(v == -Inf, "'-inf'::DOUBLE", "0.0")),
                                         collapse = ", "), "]")
ds <- function(formula, data) glm(formula, family = binomial, data = data, method = "detect_separation")
seps <- character()
for (g in unique(sep$g)) {
  d <- sep[sep$g == g, ]
  f <- ds(y ~ x1 + x2, d)
  f0 <- ds(y ~ x1 + x2 - 1, d)
  seps <- c(seps, exact(sprintf("%s: detect_separation(y ~ x1 + x2): outcome %s, coefficients %s",
                                gsub("'", "", g), f$outcome, paste(f$coefficients, collapse = " ")),
    sprintf(paste0("SELECT r.separated, r.infinite = %s, r0.separated, r0.infinite = %s, r.n_observations FROM ",
                   "(SELECT detect_separation_agg(y, [x1, x2]) AS r, detect_separation_agg(y, [x1, x2], {'fit_intercept': false}) AS r0 ",
                   "FROM sep WHERE g = %s) s;"), inf_sql(f$coefficients), inf_sql(f0$coefficients), g),
    c(tolower(f$outcome), "true", tolower(f0$outcome), "true", nrow(d))))
}
data("endometrial", package = "detectseparation")
fe <- ds(HG ~ NV + PI + EH, endometrial)
seps <- c(seps, exact(sprintf("endometrial (Heinze & Schemper 2002): outcome %s, coefficients %s",
                              fe$outcome, paste(fe$coefficients, collapse = " ")),
  sprintf("SELECT r.separated, r.infinite = %s, r.n_observations FROM (SELECT detect_separation_agg(hg, [nv, pi, eh]) AS r FROM endometrial) s;",
          inf_sql(fe$coefficients)),
  c(tolower(fe$outcome), "true", nrow(endometrial))))

ld <- ll[, lx]; names(ld) <- lcols
lines <- c(
  "# name: test/sql/reference/collinearity_separation.test",
  "# description: #306 condition_diagnostic_agg against the BKW definition of perturb::colldiag, detect_separation_agg against detectseparation",
  "# group: [reference]",
  "#",
  "# GENERATED by validation/generators/make_reference_collinearity_separation.R -- do not edit",
  "# by hand; change the generator and re-run it instead.",
  sprintf("# Reference: %s, detectseparation %s", R.version.string, packageVersion("detectseparation")),
  "",
  "require anofox_statistics",
  "",
  "statement ok",
  sprintf("CREATE TABLE longley (%s);", paste(lcols, "DOUBLE", collapse = ", ")),
  "",
  "statement ok",
  paste0("INSERT INTO longley VALUES\n", values_sql(format(ld, digits = 17, trim = TRUE)), ";"),
  "",
  "statement ok",
  "CREATE TABLE sep (g VARCHAR, x1 DOUBLE, x2 DOUBLE, y DOUBLE);",
  "",
  "statement ok",
  paste0("INSERT INTO sep VALUES\n", values_sql(sep), ";"),
  "",
  "statement ok",
  "CREATE TABLE endometrial (nv DOUBLE, pi DOUBLE, eh DOUBLE, hg DOUBLE);",
  "",
  "statement ok",
  paste0("INSERT INTO endometrial VALUES\n",
         values_sql(format(endometrial[, c("NV", "PI", "EH", "HG")], digits = 17, trim = TRUE)), ";"),
  "",
  cond,
  seps)
writeLines(lines, file.path(out_dir, "collinearity_separation.test"))
cat("wrote", file.path(out_dir, "collinearity_separation.test"), "\n")
