#!/usr/bin/env Rscript
# Reference values for test/sql/reference/partial_correlation_reference.test
# (partial_cor_agg / semi_partial_cor_agg, issue #296).
#
# Usage:  Rscript validation/generators/make_reference_partial_cor.R
# Prints "name = value" lines (17 significant digits); the values are pasted by
# hand into the .test file next to a comment naming the R call.
# Packages: ppcor (pcor.test / spcor.test). Install into a user library with
#   install.packages("ppcor", lib = "~/R/userlib")
# The base-R residualization below cross-checks ppcor on every case.

.libPaths(c("~/R/userlib", .libPaths()))
suppressPackageStartupMessages(library(ppcor))
options(digits = 17)
out <- function(name, x) cat(sprintf("%-40s = %.17g\n", name, as.numeric(x)))

## Dataset (keep in sync with the VALUES list in the .test file); n = 15.
## z1 and z2 are correlated; px / pz1 contain ties (Spearman average ranks).
x  <- c(3.1, 4.7, 2.2, 6.8, 5.5, 7.9, 4.1, 8.6, 6.0, 9.4, 5.2, 7.3, 3.8, 8.1, 6.6)
y  <- c(4.2, 3.9, 2.0, 4.9, 4.2, 6.4, 3.6, 6.9, 6.6, 10.2, 4.5, 8.0, 5.8, 8.6, 6.3)
z1 <- c(0.6, 2.6, -0.2, 1.3, 3.8, 2.8, 2.9, 4.3, 3.4, 3.4, 1.4, 2.7, 0.6, 4.2, 1.4)
z2 <- c(0.9, 2.8, 1.6, 2.0, 2.5, 3.6, 1.1, 3.0, 3.9, 4.2, 1.8, 2.4, 2.3, 4.8, 1.7)
tx  <- c(3, 4, 2, 6, 5, 7, 4, 8, 6, 9, 5, 7, 3, 8, 6)
tz1 <- c(1, 2, 1, 3, 2, 4, 2, 4, 3, 5, 2, 4, 1, 5, 3)
Z <- cbind(z1, z2)

# base-R checks: partial = cor of both residuals, semi-partial = cor(x, resid(y | z))
chk <- function(a, b) stopifnot(abs(a - b) < 1e-12)

emit <- function(tag, r) {
  out(paste0(tag, ".estimate"), r$estimate)
  out(paste0(tag, ".statistic"), r$statistic)
  out(paste0(tag, ".p_value"), r$p.value)
  out(paste0(tag, ".df"), r$n - 2 - r$gp)
  out(paste0(tag, ".n"), r$n)
}

cat("## partial_cor_agg  ppcor::pcor.test\n")
r <- pcor.test(x, y, z1); chk(r$estimate, cor(resid(lm(x ~ z1)), resid(lm(y ~ z1)))); emit("pcor_pearson_k1", r)
r <- pcor.test(x, y, Z);  chk(r$estimate, cor(resid(lm(x ~ Z)), resid(lm(y ~ Z))));   emit("pcor_pearson_k2", r)
r <- pcor.test(x, y, z1, method = "spearman"); emit("pcor_spearman_k1", r)
r <- pcor.test(x, y, Z,  method = "spearman"); emit("pcor_spearman_k2", r)
r <- pcor.test(tx, y, tz1, method = "spearman")
chk(r$estimate, cor(resid(lm(rank(tx) ~ rank(tz1))), resid(lm(rank(y) ~ rank(tz1)))))
emit("pcor_spearman_ties", r)

cat("## semi_partial_cor_agg  ppcor::spcor.test (z removed from y only)\n")
r <- spcor.test(x, y, z1); chk(r$estimate, cor(x, resid(lm(y ~ z1)))); emit("spcor_pearson_k1", r)
r <- spcor.test(x, y, z1, method = "spearman"); emit("spcor_spearman_k1", r)
r <- spcor.test(tx, y, tz1, method = "spearman")
chk(r$estimate, cor(rank(tx), resid(lm(rank(y) ~ rank(tz1))))); emit("spcor_spearman_ties", r)
# Not exposed (upstream anofox-statistics 0.7.0 residualizes y sequentially,
# which differs from OLS residuals for correlated controls):
r <- spcor.test(x, y, Z); chk(r$estimate, cor(x, resid(lm(y ~ Z)))); emit("spcor_pearson_k2", r)

cat("## no controls = Pearson cor.test\n")
r <- cor.test(x, y); out("pearson.estimate", r$estimate); out("pearson.p_value", r$p.value)
