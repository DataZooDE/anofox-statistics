#!/usr/bin/env Rscript
# Reference values for the two-way and repeated-measures ANOVA blocks of
# test/sql/reference/anova_reference.test (two_way_anova_agg,
# repeated_measures_anova_agg; issue #297).
#
# Usage:  Rscript validation/generators/make_reference_anova.R
# Prints "name = value" lines (17 significant digits); the values are pasted by
# hand into the .test file next to a comment naming the R call.
# Packages: stats and car (Type III sums of squares). afex is used for the
# repeated-measures cross-check when it is installed; the reference values
# themselves come from base R: aov(y ~ cond + Error(subject/cond)) for the F
# test, mauchly.test() on the wide matrix for Mauchly's W and the
# Greenhouse-Geisser / Huynh-Feldt epsilons from their textbook formulas
# (Greenhouse & Geisser 1959; Huynh & Feldt 1976, the one-group form also used
# by car::Anova and afex).

options(digits = 17)
if (dir.exists("~/R/userlib")) .libPaths(c(path.expand("~/R/userlib"), .libPaths()))
suppressPackageStartupMessages(library(car))
out <- function(name, x) cat(sprintf("%-36s = %.17g\n", name, as.numeric(x)))

## ---------------------------------------------------------------------------
## Two-way ANOVA (keep in sync with the VALUES lists in the .test file)
## ---------------------------------------------------------------------------
# balanced 3 x 2, 4 replicates per cell
bal <- data.frame(
  a = rep(c("ctrl", "high", "low"), each = 8),
  b = rep(rep(c(1, 2), each = 4), 3),
  y = c(4.2, 5.1, 3.8, 4.6,   5.0, 5.9, 6.3, 5.4,
        7.9, 8.4, 7.1, 8.8,   9.6, 10.9, 9.2, 10.1,
        6.1, 5.7, 6.6, 6.0,   6.4, 7.3, 6.8, 7.7))
# unbalanced 2 x 3 (cell sizes 3, 4, 2 / 5, 3, 4)
unb <- data.frame(
  a = c(rep("f", 9), rep("m", 12)),
  b = c(rep("x", 3), rep("y", 4), rep("z", 2), rep("x", 5), rep("y", 3), rep("z", 4)),
  y = c(12.1, 13.4, 11.8,   14.2, 15.0, 13.1, 14.7,   18.3, 17.2,
        11.0, 12.6, 10.9, 11.7, 12.2,   16.4, 17.8, 15.9,   16.1, 15.2, 17.0, 16.6))

two_way <- function(tag, d) {
  d$a <- factor(d$a); d$b <- factor(d$b)
  fit <- lm(y ~ a * b, data = d, contrasts = list(a = "contr.sum", b = "contr.sum"))
  t3 <- car::Anova(fit, type = 3)
  rows <- c(a = "a", b = "b", ab = "a:b")
  for (k in names(rows)) {
    r <- t3[rows[[k]], ]
    out(paste0(tag, ".", k, ".sum_sq"), r$`Sum Sq`)
    out(paste0(tag, ".", k, ".df"), r$Df)
    out(paste0(tag, ".", k, ".f"), r$`F value`)
    out(paste0(tag, ".", k, ".p_value"), r$`Pr(>F)`)
    out(paste0(tag, ".", k, ".partial_eta_sq"), r$`Sum Sq` / (r$`Sum Sq` + t3["Residuals", "Sum Sq"]))
  }
  out(paste0(tag, ".residual.sum_sq"), t3["Residuals", "Sum Sq"])
  out(paste0(tag, ".residual.df"), t3["Residuals", "Df"])
}
cat("## two_way_anova_agg  car::Anova(lm(y ~ a * b, contrasts = contr.sum), type = 3)\n")
two_way("bal", bal)
# balanced: type III == type I == summary(aov())
s <- summary(aov(y ~ factor(a) * factor(b), data = bal))[[1]]
out("bal.aov_check.a.sum_sq", s$`Sum Sq`[1])
two_way("unb", unb)

## ---------------------------------------------------------------------------
## Repeated-measures ANOVA: 8 subjects x 4 conditions (wide matrix, row = subject)
## ---------------------------------------------------------------------------
wide <- matrix(c(
  45, 50, 55, 70,
  42, 42, 45, 60,
  36, 41, 43, 62,
  39, 35, 40, 51,
  51, 55, 59, 77,
  44, 49, 56, 68,
  40, 41, 48, 55,
  48, 52, 56, 74), ncol = 4, byrow = TRUE)
# 6 subjects x 2 conditions (no sphericity test)
wide2 <- matrix(c(
  10.2, 11.9,
   9.8, 10.4,
  12.5, 13.1,
  11.1, 12.8,
   8.7, 10.0,
  10.9, 11.2), ncol = 2, byrow = TRUE)

rm_anova <- function(tag, w) {
  n <- nrow(w); k <- ncol(w)
  long <- data.frame(y = as.vector(t(w)),
                     subject = factor(rep(seq_len(n), each = k)),
                     cond = factor(rep(seq_len(k), n)))
  s <- summary(aov(y ~ cond + Error(subject / cond), data = long))
  within <- s[["Error: subject:cond"]][[1]]
  between <- s[["Error: subject"]][[1]]
  out(paste0(tag, ".ss_condition"), within$`Sum Sq`[1])
  out(paste0(tag, ".ss_error"), within$`Sum Sq`[2])
  out(paste0(tag, ".ss_subjects"), between$`Sum Sq`[1])
  out(paste0(tag, ".df1"), within$Df[1])
  out(paste0(tag, ".df2"), within$Df[2])
  out(paste0(tag, ".f"), within$`F value`[1])
  out(paste0(tag, ".p_value"), within$`Pr(>F)`[1])
  out(paste0(tag, ".partial_eta_sq"), within$`Sum Sq`[1] / sum(within$`Sum Sq`))
  if (k < 3) return(invisible())
  # Mauchly: sphericity of the within-subject contrasts
  mt <- mauchly.test(lm(w ~ 1), X = ~1)
  out(paste0(tag, ".mauchly_w"), mt$statistic)
  out(paste0(tag, ".mauchly_p_value_r"), mt$p.value)  # R adds a second-order term
  p <- k - 1
  chisq <- -(n - 1 - (2 * p^2 + p + 2) / (6 * p)) * log(mt$statistic)
  out(paste0(tag, ".mauchly_chisq"), chisq)
  out(paste0(tag, ".mauchly_df"), p * (p + 1) / 2 - 1)
  out(paste0(tag, ".mauchly_p_value"), pchisq(chisq, p * (p + 1) / 2 - 1, lower.tail = FALSE))
  # epsilons from the covariance of orthonormal contrasts
  C <- contr.poly(k)
  S <- t(C) %*% cov(w) %*% C
  ev <- eigen(S, symmetric = TRUE, only.values = TRUE)$values
  gg <- sum(ev)^2 / (p * sum(ev^2))
  hf <- (n * p * gg - 2) / (p * (n - 1 - p * gg))
  f <- within$`F value`[1]
  df1 <- within$Df[1]; df2 <- within$Df[2]
  out(paste0(tag, ".gg_epsilon"), gg)
  out(paste0(tag, ".gg_p_value"), pf(f, gg * df1, gg * df2, lower.tail = FALSE))
  out(paste0(tag, ".hf_epsilon"), hf)
  out(paste0(tag, ".hf_epsilon_capped"), min(hf, 1))
  out(paste0(tag, ".hf_p_value"), pf(f, min(hf, 1) * df1, min(hf, 1) * df2, lower.tail = FALSE))
  if (requireNamespace("afex", quietly = TRUE)) {
    a <- suppressMessages(afex::aov_ez("subject", "y", long, within = "cond",
                                      anova_table = list(correction = "none", es = "none")))
    sm <- summary(a)
    cat("# afex cross-check\n")
    out(paste0(tag, ".afex.mauchly_w"), sm$sphericity.tests[1, "Test statistic"])
    out(paste0(tag, ".afex.mauchly_p_value"), sm$sphericity.tests[1, "p-value"])
    out(paste0(tag, ".afex.gg_epsilon"), sm$pval.adjustments[1, "GG eps"])
    out(paste0(tag, ".afex.gg_p_value"), sm$pval.adjustments[1, "Pr(>F[GG])"])
    out(paste0(tag, ".afex.hf_epsilon"), sm$pval.adjustments[1, "HF eps"])
    out(paste0(tag, ".afex.hf_p_value"), sm$pval.adjustments[1, "Pr(>F[HF])"])
  }
}
cat("## repeated_measures_anova_agg  aov(y ~ cond + Error(subject/cond)), mauchly.test\n")
rm_anova("rm4", wide)
rm_anova("rm2", wide2)
