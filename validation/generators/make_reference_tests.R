#!/usr/bin/env Rscript
# Reference values for test/sql/reference/{hypothesis,nonparametric,normality,
# correlation,categorical,proportion}_reference.test
#
# Usage:  Rscript validation/generators/make_reference_tests.R
# Prints "name = value" lines (17 significant digits).  The values are pasted
# by hand into the .test files next to a comment naming the R call.
# scipy-only references (D'Agostino K^2, Brunner-Munzel, Kendall tau-c, ...)
# live in make_reference_tests.py.
# Packages: base R + stats only.

options(digits = 17)
out <- function(name, x) cat(sprintf("%-40s = %.17g\n", name, as.numeric(x)))

## ---------------------------------------------------------------------------
## Datasets (keep in sync with the VALUES lists in the .test files)
## ---------------------------------------------------------------------------
# two-sample, no ties, unequal n and unequal variance
g1 <- c(5.1, 4.9, 6.2, 5.8, 6.05, 5.5, 5.3, 6.1)
g2 <- c(6.5, 7.1, 6.8, 7.4, 6.0, 7.9, 6.6, 7.2, 6.9, 7.05)
# paired, no zero differences, no tied |d|
px <- c(12.1, 14.3, 11.8, 15.2, 13.9, 12.7, 14.8, 13.1, 12.4, 15.6)
py <- c(11.5, 13.2, 12.3, 13.9, 13.0, 11.1, 14.6, 11.4, 12.6, 13.85)
# three groups (k-sample), no ties
k1 <- c(23.1, 25.4, 21.8, 24.9, 22.7)
k2 <- c(27.3, 26.1, 29.8, 28.4, 25.2, 30.6)
k3 <- c(24.0, 31.5, 19.9, 26.6, 33.2, 21.3, 28.9)
# normality sample (n = 20)
nv <- c(2.31, 3.85, 1.97, 4.42, 3.10, 2.76, 5.94, 3.33, 2.05, 4.88,
        3.61, 2.49, 7.12, 3.02, 2.88, 4.15, 3.47, 1.64, 5.21, 2.95)
# correlation, no ties
cx <- c(1.2, 2.4, 3.1, 4.8, 5.0, 6.3, 7.7, 8.1, 9.6, 10.2, 11.5, 12.9)
cy <- c(2.0, 2.9, 4.4, 4.1, 6.2, 5.8, 8.9, 7.5, 9.1, 12.0, 10.8, 13.3)

## ---------------------------------------------------------------------------
cat("## t_test_agg  t.test(g1, g2, var.equal = FALSE|TRUE)\n")
for (ve in c(FALSE, TRUE)) {
  tag <- if (ve) "student" else "welch"
  r <- t.test(g1, g2, var.equal = ve)
  out(paste0(tag, ".statistic"), r$statistic)
  out(paste0(tag, ".df"), r$parameter)
  out(paste0(tag, ".p_value"), r$p.value)
  out(paste0(tag, ".ci_lower"), r$conf.int[1])
  out(paste0(tag, ".ci_upper"), r$conf.int[2])
}
r <- t.test(g1, g2, alternative = "less"); out("welch_less.p_value", r$p.value)
r <- t.test(g1, g2, conf.level = 0.99); out("welch_99.ci_lower", r$conf.int[1]); out("welch_99.ci_upper", r$conf.int[2])

cat("## mann_whitney_u_agg  wilcox.test(g1, g2)\n")
r <- wilcox.test(g1, g2, exact = TRUE)
out("mw_exact.statistic", r$statistic); out("mw_exact.p_value", r$p.value)
r <- wilcox.test(g1, g2, exact = FALSE, correct = TRUE)
out("mw_normal_cc.p_value", r$p.value)
r <- wilcox.test(g1, g2, exact = FALSE, correct = FALSE)
out("mw_normal_nocc.p_value", r$p.value)

cat("## wilcoxon_signed_rank_agg  wilcox.test(px, py, paired=TRUE)\n")
r <- wilcox.test(px, py, paired = TRUE, exact = TRUE)
out("wsr_exact.statistic", r$statistic); out("wsr_exact.p_value", r$p.value)
r <- wilcox.test(px, py, paired = TRUE, exact = FALSE, correct = TRUE)
out("wsr_normal_cc.p_value", r$p.value)
r <- wilcox.test(px, py, paired = TRUE, exact = FALSE, correct = FALSE)
out("wsr_normal_nocc.p_value", r$p.value)

kv <- c(k1, k2, k3); kg <- factor(rep(1:3, c(length(k1), length(k2), length(k3))))
cat("## kruskal_wallis_agg  kruskal.test(kv, kg)\n")
r <- kruskal.test(kv, kg); out("kw.statistic", r$statistic); out("kw.df", r$parameter); out("kw.p_value", r$p.value)

cat("## one_way_anova_agg  summary(aov(kv ~ kg))\n")
a <- summary(aov(kv ~ kg))[[1]]
out("anova.f_statistic", a$`F value`[1]); out("anova.p_value", a$`Pr(>F)`[1])
out("anova.ss_between", a$`Sum Sq`[1]); out("anova.ss_within", a$`Sum Sq`[2])
r <- oneway.test(kv ~ kg, var.equal = FALSE)
out("welch_anova.statistic", r$statistic); out("welch_anova.p_value", r$p.value)

cat("## brown_forsythe_agg  ANOVA on |x - median_group|  (== car::leveneTest(center=median))\n")
z <- abs(kv - ave(kv, kg, FUN = median))
a <- summary(aov(z ~ kg))[[1]]
out("bf.statistic", a$`F value`[1]); out("bf.p_value", a$`Pr(>F)`[1])

cat("## shapiro_wilk_agg  shapiro.test(nv)\n")
r <- shapiro.test(nv); out("sw.statistic", r$statistic); out("sw.p_value", r$p.value)

cat("## jarque_bera  JB = n/6 (S^2 + (K-3)^2/4), population moments (tseries::jarque.bera.test)\n")
n <- length(nv); m <- mean(nv); m2 <- mean((nv - m)^2)
S <- mean((nv - m)^3) / m2^1.5; K <- mean((nv - m)^4) / m2^2
JB <- n / 6 * (S^2 + (K - 3)^2 / 4)
out("jb.statistic", JB); out("jb.p_value", pchisq(JB, 2, lower.tail = FALSE))
out("jb.skewness", S); out("jb.kurtosis_raw", K); out("jb.excess_kurtosis", K - 3)

cat("## correlation  cor.test(cx, cy, method=...)\n")
r <- cor.test(cx, cy, method = "pearson")
out("pearson.r", r$estimate); out("pearson.statistic", r$statistic); out("pearson.p_value", r$p.value)
out("pearson.ci_lower", r$conf.int[1]); out("pearson.ci_upper", r$conf.int[2])
r <- cor.test(cx, cy, method = "spearman")
out("spearman.r", r$estimate); out("spearman.p_value_exact", r$p.value)
r <- cor.test(cx, cy, method = "spearman", exact = FALSE)
out("spearman.p_value_asym", r$p.value)
rs <- cor(cx, cy, method = "spearman"); nn <- length(cx)
tstat <- rs * sqrt((nn - 2) / (1 - rs^2)); out("spearman.t", tstat); out("spearman.p_value_t", 2 * pt(-abs(tstat), nn - 2))
r <- cor.test(cx, cy, method = "kendall")
out("kendall.tau", r$estimate); out("kendall.p_value_exact", r$p.value)
r <- cor.test(cx, cy, method = "kendall", exact = FALSE)
out("kendall.z", r$statistic); out("kendall.p_value_asym", r$p.value)

## ---------------------------------------------------------------------------
cat("## chisq_test_agg  chisq.test(tab)\n")
tab <- matrix(c(20, 15, 25,
                30, 10, 12), nrow = 2, byrow = TRUE)
r <- chisq.test(tab); out("chisq.statistic", r$statistic); out("chisq.df", r$parameter); out("chisq.p_value", r$p.value)
tab2 <- matrix(c(18, 7, 9, 16), nrow = 2, byrow = TRUE)
r <- chisq.test(tab2, correct = TRUE);  out("chisq2x2_yates.statistic", r$statistic); out("chisq2x2_yates.p_value", r$p.value)
r <- chisq.test(tab2, correct = FALSE); out("chisq2x2_noyates.statistic", r$statistic); out("chisq2x2_noyates.p_value", r$p.value)

cat("## g_test_agg  G = 2 sum O log(O/E)\n")
E <- outer(rowSums(tab), colSums(tab)) / sum(tab)
G <- 2 * sum(tab * log(tab / E)); out("g.statistic", G); out("g.p_value", pchisq(G, 2, lower.tail = FALSE))

cat("## fisher_exact_agg  fisher.test(tab2)\n")
r <- fisher.test(tab2)
out("fisher.p_value", r$p.value); out("fisher.odds_ratio_cmle", r$estimate)
out("fisher.ci_lower", r$conf.int[1]); out("fisher.ci_upper", r$conf.int[2])
out("fisher.odds_ratio_sample", (18 * 16) / (7 * 9))
out("fisher_greater.p_value", fisher.test(tab2, alternative = "greater")$p.value)
out("fisher_less.p_value", fisher.test(tab2, alternative = "less")$p.value)

cat("## mcnemar_agg  mcnemar.test(mtab)\n")
mtab <- matrix(c(30, 12, 4, 24), nrow = 2, byrow = TRUE)   # b = 12, c = 4
r <- mcnemar.test(mtab, correct = TRUE);  out("mcnemar_cc.statistic", r$statistic); out("mcnemar_cc.p_value", r$p.value)
r <- mcnemar.test(mtab, correct = FALSE); out("mcnemar_nocc.statistic", r$statistic); out("mcnemar_nocc.p_value", r$p.value)

cat("## cramers_v_agg  sqrt(X2 / (n (min(r,c)-1))), X2 uncorrected\n")
X2 <- chisq.test(tab, correct = FALSE)$statistic
out("cramers_v", sqrt(X2 / (sum(tab) * (min(dim(tab)) - 1))))

cat("## binom_test_agg  binom.test(13, 20, p)\n")
r <- binom.test(13, 20, p = 0.5)
out("binom.p_value", r$p.value); out("binom.ci_lower", r$conf.int[1]); out("binom.ci_upper", r$conf.int[2])
out("binom_p03.p_value", binom.test(13, 20, p = 0.3)$p.value)
out("binom_greater.p_value", binom.test(13, 20, p = 0.5, alternative = "greater")$p.value)

cat("## prop_test_one_agg  prop.test(13, 20, p)\n")
for (cc in c(TRUE, FALSE)) {
  r <- prop.test(13, 20, p = 0.5, correct = cc); tag <- if (cc) "prop1_cc" else "prop1_nocc"
  out(paste0(tag, ".statistic"), r$statistic); out(paste0(tag, ".p_value"), r$p.value)
  out(paste0(tag, ".ci_lower"), r$conf.int[1]); out(paste0(tag, ".ci_upper"), r$conf.int[2])
}
p0 <- 0.5; z <- (13/20 - p0) / sqrt(p0 * (1 - p0) / 20); out("prop1.z_nocc", z)

cat("## prop_test_two_agg  prop.test(c(18, 11), c(30, 28))\n")
for (cc in c(TRUE, FALSE)) {
  r <- prop.test(c(18, 11), c(30, 28), correct = cc); tag <- if (cc) "prop2_cc" else "prop2_nocc"
  out(paste0(tag, ".statistic"), r$statistic); out(paste0(tag, ".p_value"), r$p.value)
  out(paste0(tag, ".ci_lower"), r$conf.int[1]); out(paste0(tag, ".ci_upper"), r$conf.int[2])
}

cat("## yuen_agg  Yuen trimmed t (WRS2::yuen formula, trim = 0.2) computed by hand\n")
yuen <- function(x, y, tr = 0.2, alpha = 0.05) {
  h1 <- length(x) - 2 * floor(tr * length(x)); h2 <- length(y) - 2 * floor(tr * length(y))
  wv <- function(v) { v <- sort(v); g <- floor(tr * length(v)); n <- length(v)
    v[1:g] <- v[g + 1]; v[(n - g + 1):n] <- v[n - g]; var(v) }
  d1 <- (length(x) - 1) * wv(x) / (h1 * (h1 - 1)); d2 <- (length(y) - 1) * wv(y) / (h2 * (h2 - 1))
  df <- (d1 + d2)^2 / (d1^2 / (h1 - 1) + d2^2 / (h2 - 1))
  dif <- mean(x, trim = tr) - mean(y, trim = tr); se <- sqrt(d1 + d2)
  list(t = dif / se, df = df, p = 2 * (1 - pt(abs(dif / se), df)),
       ci = dif + c(-1, 1) * qt(1 - alpha / 2, df) * se)
}
r <- yuen(g1, g2); out("yuen.statistic", r$t); out("yuen.df", r$df); out("yuen.p_value", r$p)
out("yuen.ci_lower", r$ci[1]); out("yuen.ci_upper", r$ci[2])

cat("## extra cases\n")
# Fisher: sample odds ratio + Woolf (logit) CI, which is what fisher_exact_agg reports
# (R's fisher.test reports the conditional MLE and an exact conditional CI instead).
lor <- log((18 * 16) / (7 * 9)); se <- sqrt(1/18 + 1/7 + 1/9 + 1/16)
out("fisher.woolf_ci_lower", exp(lor - qnorm(0.975) * se)); out("fisher.woolf_ci_upper", exp(lor + qnorm(0.975) * se))
# prop.test variants
out("prop2_cc_greater.p_value", prop.test(c(18, 11), c(30, 28), alternative = "greater")$p.value)
out("prop1_nocc_greater.p_value", prop.test(13, 20, p = 0.5, alternative = "greater", correct = FALSE)$p.value)
r <- prop.test(13, 20, p = 0.3, correct = FALSE)
out("prop1_nocc_p03.z", sqrt(r$statistic)); out("prop1_nocc_p03.p_value", r$p.value)
# Kendall with ties (tau-b, tie-corrected normal approximation)
kx <- c(1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 8, 8); ky <- c(2, 1, 3, 3, 5, 4, 6, 6, 5, 8, 7, 9)
r <- cor.test(kx, ky, method = "kendall", exact = FALSE)
out("kendall_ties.tau_b", r$estimate); out("kendall_ties.z", r$statistic); out("kendall_ties.p_value", r$p.value)
# Shapiro-Wilk small sample (n = 10)
r <- shapiro.test(c(4.1, 5.3, 3.8, 6.9, 5.0, 4.4, 9.2, 5.7, 4.9, 6.1))
out("sw_n10.statistic", r$statistic); out("sw_n10.p_value", r$p.value)
