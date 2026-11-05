# Reference values for the predictive accuracy indexes (#308), held by
# test/sql/reference/accuracy_reference.test.
#
#   Rscript validation/generators/make_reference_accuracy.R
#
# Packages: rms (lrm, val.prob), Hmisc (somers2, rcorr.cens, GiniMd,
# R2Measures), survival (concordance). Prints the data as a SQL VALUES list
# and every reference value with 15 significant digits.
suppressMessages({ library(rms); library(survival) })
p <- function(label, value) cat(sprintf("%-28s %.15g\n", label, value))
q <- function(v) sprintf("%.17g", v)

i  <- 0:39
x1 <- round(1.5 * sin(0.9 * i), 2)              # prediction with ties
x2 <- round(cos(0.4 * i), 2)
y  <- as.numeric(sin(0.9 * i) + 0.6 * cos(1.7 * i) + 0.3 > 0)
w  <- 1 + i %% 3                                # case weights
yo <- round(sin(0.9 * i) + 0.5 * cos(1.7 * i))  # ordered response
pr <- round(plogis(0.2 + 1.1 * sin(0.9 * i)), 2)
tm <- round(exp(1 + 0.5 * cos(0.8 * i) + 0.3 * sin(2.1 * i)), 1)  # tied times
ev <- as.numeric(i %% 4 != 0)
f  <- lrm(y ~ x1 + x2)
fw <- suppressWarnings(lrm(y ~ x1 + x2, weights = w))
lp <- unname(f$linear.predictors); lpw <- unname(fw$linear.predictors)

cat("-- (i, x1, x2, y, w, yo, p, tm, ev, lp, lpw)\n")
cat(paste0(" (", i, ", ", x1, ", ", x2, ", ", y, ", ", w, ", ", yo, ", ", pr, ", ", tm, ", ", ev,
           ", ", q(lp), ", ", q(lpw), ")", collapse = ",\n"), "\n")

# rms reports P as 1 - pchisq(), which loses digits below ~1e-12; the
# reference is the upper tail itself.
lrm_out <- function(tag, s) {
  nm <- names(s)
  for (k in seq_along(s)) if (nm[k] != "Max Deriv") p(paste(tag, nm[k]), s[k])
  p(paste(tag, "P upper tail"), pchisq(s["Model L.R."], s["d.f."], lower.tail = FALSE))
}
cat("## lrm(y ~ x1 + x2)$stats\n"); lrm_out("lrm", f$stats)
cat("## lrm(y ~ x1 + x2, weights = w)$stats\n"); lrm_out("lrmw", fw$stats)

cat("## somers2\n")
s <- somers2(pr, y); p("somers2 C", s["C"]); p("somers2 Dxy", s["Dxy"]); p("somers2 n", s["n"])
s <- somers2(pr, y, weights = w); p("somers2w C", s["C"]); p("somers2w Dxy", s["Dxy"]); p("somers2w n", s["n"])

cat("## survival::concordance(yo ~ x1) counts\n")
print(concordance(yo ~ x1)$count); print(concordance(yo ~ x1, weights = w)$count)

cat("## rcorr.cens(x1, Surv(tm, ev))\n")
r <- rcorr.cens(x1, Surv(tm, ev)); for (k in seq_along(r)) p(names(r)[k], r[k])

cat("## val.prob(p, y)\n")
v <- val.prob(pr, y, pl = FALSE); for (k in seq_along(v)) p(names(v)[k], v[k])
cat("## val.prob(p2, y): p2 = p with p2[1] = 1 (i = 0, y = 1) and p2[3] = 0 (i = 2, y = 1)\n")
p2 <- pr; p2[1] <- 1; p2[3] <- 0
v <- suppressWarnings(val.prob(p2, y, pl = FALSE)); for (k in seq_along(v)) p(names(v)[k], v[k])
cat("## val.prob(p3, y): p3[1] = 1 (i = 0, y = 1), p3[6] = 0 (i = 5, y = 0), p3[7] = 1 (i = 6, y = 0)\n")
p3 <- pr; p3[1] <- 1; p3[6] <- 0; p3[7] <- 1
v <- suppressWarnings(val.prob(p3, y, pl = FALSE)); for (k in seq_along(v)) p(names(v)[k], v[k])

cat("## GiniMd\n"); p("GiniMd(x1)", GiniMd(x1)); p("GiniMd(lp)", GiniMd(lp))
cat("## R2Measures(lr = 10, p = 2, n = 100, ess = 60)\n")
r <- R2Measures(10, 2, 100, 60); for (k in seq_along(r)) p(names(r)[k], r[k])
r <- R2Measures(10, 2, 100); for (k in seq_along(r)) p(names(r)[k], r[k])
