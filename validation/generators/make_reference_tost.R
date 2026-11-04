# Reference values for the TOST equivalence variants (#293), held by
# test/sql/reference/tost_reference.test.
#
#   Rscript validation/generators/make_reference_tost.R
#
# Base R only (TOSTER and WRS2 are not installed): each one-sided test is the
# matching base-R test with the null shifted to the bound, and the formulas that
# base R has no function for (Yuen's statistic, the Hodges-Lehmann CI order
# statistics, the TOST proportion z-tests) are coded from their definitions:
#
#   tost_one_sample_agg      t.test(x, mu = mu + bound, alternative = ...)
#   tost_prop_one_agg        z = (p_hat - p0 - bound) / sqrt(p_hat (1 - p_hat) / n)
#   tost_prop_two_agg        z = (p1 - p2 - bound) / sqrt(p1 q1 / n1 + p2 q2 / n2)
#   tost_wilcoxon_paired_agg wilcox.test(d, mu = bound, exact = FALSE, correct = TRUE)
#   tost_wilcoxon_agg        wilcox.test(x, y, mu = bound, exact = FALSE, correct = TRUE)
#   tost_yuen_agg            Yuen (1974): trimmed means, Winsorized variances,
#                            Welch-Satterthwaite df (as WRS2::yuen)
#   tost_bootstrap_agg       percentile bootstrap of mean(x) - mean(y); the
#                            resampling stream differs, so only the estimate is
#                            exact and the CI / p-values are compared with a
#                            Monte-Carlo tolerance.
#
# The (1 - 2 alpha) confidence interval of the Wilcoxon variants is the
# normal-approximation order-statistic interval of the Walsh averages / pairwise
# differences: indices floor(mu - z sd) and ceil(mu + z sd), 0-based.
#
# The data are the deterministic SQL expressions in the test (i = 0, 1, ...).
p <- function(label, value) cat(sprintf("%-36s %.15g\n", label, value))
alpha <- 0.05

# --- tost_one_sample_agg: x_i = 0.3 sin(0.7 i) + 0.05 cos(1.3 i), i < 30 ----
i <- 0:29
x <- 0.3 * sin(0.7 * i) + 0.05 * cos(1.3 * i)
mu <- 0.1; lo <- -0.3; hi <- 0.3
tl <- t.test(x, mu = mu + lo, alternative = "greater")
tu <- t.test(x, mu = mu + hi, alternative = "less")
ci <- t.test(x, conf.level = 1 - 2 * alpha)$conf.int - mu
p("one t_lower", tl$statistic); p("one t_upper", tu$statistic)
p("one p_lower", tl$p.value); p("one p_upper", tu$p.value)
p("one df", tl$parameter); p("one estimate", mean(x) - mu)
p("one ci_lower", ci[1]); p("one ci_upper", ci[2])

# --- tost_prop_one_agg: success = (i * 7) % 100 < 47, i < 100 ---------------
s <- sum(((0:99) * 7) %% 100 < 47); n <- 100; p0 <- 0.5; d <- 0.1
ph <- s / n; se <- sqrt(ph * (1 - ph) / n)
zl <- (ph - p0 + d) / se; zu <- (ph - p0 - d) / se
z <- qnorm(1 - alpha)
p("prop1 successes", s)
p("prop1 z_lower", zl); p("prop1 z_upper", zu)
p("prop1 p_lower", pnorm(zl, lower.tail = FALSE)); p("prop1 p_upper", pnorm(zu))
p("prop1 estimate", ph - p0)
p("prop1 ci_lower", ph - p0 - z * se); p("prop1 ci_upper", ph - p0 + z * se)

# --- tost_prop_two_agg: group 'a' i < 100 success (i*7)%100 < 45,
#     group 'b' i < 120 success (i*11)%120 < 62 ----------------------------
s1 <- sum(((0:99) * 7) %% 100 < 45); n1 <- 100
s2 <- sum(((0:119) * 11) %% 120 < 62); n2 <- 120
p1 <- s1 / n1; p2 <- s2 / n2; se <- sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
est <- p1 - p2; lo <- -0.15; hi <- 0.15
zl <- (est - lo) / se; zu <- (est - hi) / se
p("prop2 s1", s1); p("prop2 s2", s2)
p("prop2 z_lower", zl); p("prop2 z_upper", zu)
p("prop2 p_lower", pnorm(zl, lower.tail = FALSE)); p("prop2 p_upper", pnorm(zu))
p("prop2 estimate", est)
p("prop2 ci_lower", est - z * se); p("prop2 ci_upper", est + z * se)

# --- tost_wilcoxon_paired_agg: x_i = 10 + sin(i), y_i = x_i + 0.2 cos(1.7 i)
#     + 0.05, i < 25 ---------------------------------------------------------
i <- 0:24
xp <- 10 + sin(i); yp <- xp + 0.2 * cos(1.7 * i) + 0.05
dd <- xp - yp; lo <- -0.1; hi <- 0.1
wl <- wilcox.test(dd, mu = lo, alternative = "greater", exact = FALSE, correct = TRUE)
wu <- wilcox.test(dd, mu = hi, alternative = "less", exact = FALSE, correct = TRUE)
walsh <- sort(outer(dd, dd, "+")[upper.tri(diag(length(dd)), diag = TRUE)] / 2)
nn <- length(dd); m <- nn * (nn + 1) / 4; sg <- sqrt(nn * (nn + 1) * (2 * nn + 1) / 24)
kl <- min(floor(m - z * sg), length(walsh) - 1); ku <- min(ceiling(m + z * sg), length(walsh) - 1)
p("wp V_lower", wl$statistic); p("wp V_upper", wu$statistic)
p("wp p_lower", wl$p.value); p("wp p_upper", wu$p.value)
p("wp estimate", median(walsh))
p("wp ci_lower", walsh[kl + 1]); p("wp ci_upper", walsh[ku + 1])

# --- tost_wilcoxon_agg: group 1 v = 5 + sin(0.9 i), i < 20;
#     group 2 v = 5.1 + sin(1.3 i + 0.4), i < 24 ------------------------------
xw <- 5 + sin(0.9 * (0:19)); yw <- 5.1 + sin(1.3 * (0:23) + 0.4)
lo <- -0.6; hi <- 0.6
wl <- wilcox.test(xw, yw, mu = lo, alternative = "greater", exact = FALSE, correct = TRUE)
wu <- wilcox.test(xw, yw, mu = hi, alternative = "less", exact = FALSE, correct = TRUE)
pd <- sort(as.vector(outer(xw, yw, "-")))
nx <- length(xw); ny <- length(yw); m <- nx * ny / 2; sg <- sqrt(nx * ny * (nx + ny + 1) / 12)
kl <- min(floor(m - z * sg), length(pd) - 1); ku <- min(ceiling(m + z * sg), length(pd) - 1)
p("w2 U_lower", wl$statistic); p("w2 U_upper", wu$statistic)
p("w2 p_lower", wl$p.value); p("w2 p_upper", wu$p.value)
p("w2 estimate", median(pd))
p("w2 ci_lower", pd[kl + 1]); p("w2 ci_upper", pd[ku + 1])

# --- tost_yuen_agg: group 1 v = 12 + 2 sin(0.8 i) (+ 15 at i = 3), i < 21;
#     group 2 v = 12.3 + 2 cos(0.6 i), i < 18; trim 0.2 ----------------------
xy <- 12 + 2 * sin(0.8 * (0:20)); xy[4] <- xy[4] + 15
yy <- 12.3 + 2 * cos(0.6 * (0:17))
tr <- 0.2; lo <- -1; hi <- 1
winvar <- function(v, g) {
  s <- sort(v); n <- length(s)
  if (g > 0) { s[1:g] <- s[g + 1]; s[(n - g + 1):n] <- s[n - g] }
  var(s)
}
yuen <- function(x, y, tr) {
  gx <- floor(tr * length(x)); gy <- floor(tr * length(y))
  hx <- length(x) - 2 * gx; hy <- length(y) - 2 * gy
  dx <- (length(x) - 1) * winvar(x, gx) / (hx * (hx - 1))
  dy <- (length(y) - 1) * winvar(y, gy) / (hy * (hy - 1))
  list(est = mean(x, trim = tr) - mean(y, trim = tr), se = sqrt(dx + dy),
       df = (dx + dy)^2 / (dx^2 / (hx - 1) + dy^2 / (hy - 1)))
}
yr <- yuen(xy, yy, tr)
tl <- (yr$est - lo) / yr$se; tu <- (yr$est - hi) / yr$se
tc <- qt(1 - alpha, yr$df)
p("yuen t_lower", tl); p("yuen t_upper", tu)
p("yuen p_lower", pt(tl, yr$df, lower.tail = FALSE)); p("yuen p_upper", pt(tu, yr$df))
p("yuen df", yr$df); p("yuen estimate", yr$est)
p("yuen ci_lower", yr$est - tc * yr$se); p("yuen ci_upper", yr$est + tc * yr$se)
yr0 <- yuen(xy, yy, 0.1)
p("yuen10 estimate", yr0$est); p("yuen10 df", yr0$df)

# --- tost_bootstrap_agg: the tost_wilcoxon_agg groups, delta 0.6 -----------
set.seed(1)
B <- 20000
bd <- replicate(B, mean(sample(xw, replace = TRUE)) - mean(sample(yw, replace = TRUE)))
p("boot estimate", mean(xw) - mean(yw))
p("boot ci_lower (MC)", quantile(bd, alpha, type = 1))
p("boot ci_upper (MC)", quantile(bd, 1 - alpha, type = 1))
p("boot p_lower (MC)", mean(bd <= -0.6)); p("boot p_upper (MC)", mean(bd >= 0.6))
