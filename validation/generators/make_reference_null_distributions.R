# Base-R reference values for test/sql/reference/null_distributions_reference.test (#304):
# auc_null_quantile / auc_null_p_value (qwilcox / pwilcox) and ptukey / qtukey.
#
# Usage: Rscript validation/generators/make_reference_null_distributions.R
# Prints the VALUES rows of the reference tables in the test.
options(digits = 17)
fmt <- function(x) sprintf("%.17g", x)
rows <- function(lines) cat(paste0(" ", lines, collapse = ",\n"), ";\n", sep = "")

sizes <- list(c(3, 4), c(10, 10), c(5, 12), c(20, 30), c(50, 40))

# The null AUC of a random classifier is U / (n_pos n_neg), U ~ Mann-Whitney.
cat("## auc_null_quantile: p n_pos n_neg expected = qwilcox(p, n_pos, n_neg) / (n_pos n_neg)\n")
out <- c()
# p = 0 and p = 1 are exact boundaries (0 and m n), as qwilcox; anofox-statistics
# < 0.8.1 returned 0.93 for p = 1 at 50 x 40 (#304).
for (s in sizes) for (p in c(0, 1e-12, 0.01, 0.05, 0.5, 0.95, 0.99, 1 - 1e-12, 1)) {
  m <- s[1]; n <- s[2]
  out <- c(out, sprintf("(%s, %d, %d, %s)", fmt(p), m, n, fmt(qwilcox(p, m, n) / (m * n))))
}
rows(out)

# p-value P(U >= u) at u = auc n_pos n_neg; an AUC from scores is a multiple of
# 1 / (2 n_pos n_neg), a half-integer u counts from ceiling(u).
cat("## auc_null_p_value: auc n_pos n_neg expected = pwilcox(ceiling(u) - 1, n_pos, n_neg, lower.tail = FALSE)\n")
out <- c()
for (s in sizes) for (f in c(0, 0.3, 0.5, 0.62, 0.75, 0.9, 1)) {
  m <- s[1]; n <- s[2]
  u <- round(f * m * n * 2) / 2   # a reachable AUC (possibly a half from ties)
  auc <- u / (m * n)
  out <- c(out, sprintf("(%s, %d, %d, %s)", fmt(auc), m, n,
                        fmt(pwilcox(ceiling(u) - 1, m, n, lower.tail = FALSE))))
}
rows(out)

# Beyond n_pos n_neg = 10000 upstream switches to the normal approximation
# (mean 0.5, var (n_pos + n_neg + 1) / (12 n_pos n_neg); the p-value with
# continuity correction as wilcox.test(exact = FALSE)). Base-R formulas:
cat("## large samples (normal approximation): kind x n_pos n_neg expected\n")
m <- 150; n <- 100; sd <- sqrt((m + n + 1) / (12 * m * n))
out <- c()
for (p in c(0.05, 0.5, 0.95)) out <- c(out, sprintf("('q', %s, %d, %d, %s)", fmt(p), m, n, fmt(min(max(0.5 + qnorm(p) * sd, 0), 1))))
for (auc in c(0.45, 0.55, 0.6)) {
  u <- auc * m * n
  out <- c(out, sprintf("('p', %s, %d, %d, %s)", fmt(auc), m, n,
                        fmt(pnorm((u - 0.5 - m * n / 2) / (sd * m * n), lower.tail = FALSE))))
}
rows(out)

cat("## ptukey: q n_means df n_ranges lower_tail expected\n")
out <- c()
for (q in c(0.5, 2, 3.5, 5)) for (k in c(2, 3, 5, 10)) for (df in c(5, 12, 30, 1000)) {
  out <- c(out, sprintf("(%s, %d, %d, 1, true, %s)", fmt(q), k, df, fmt(ptukey(q, k, df))))
}
for (q in c(2, 4)) for (r in c(2, 3)) {
  out <- c(out, sprintf("(%s, 4, 20, %d, true, %s)", fmt(q), r, fmt(ptukey(q, 4, 20, nranges = r))))
  out <- c(out, sprintf("(%s, 4, 20, %d, false, %s)", fmt(q), r, fmt(ptukey(q, 4, 20, nranges = r, lower.tail = FALSE))))
}
rows(out)

cat("## qtukey: p n_means df n_ranges lower_tail expected\n")
out <- c()
for (p in c(0.1, 0.5, 0.9, 0.95, 0.99)) for (k in c(2, 3, 5, 10)) for (df in c(5, 12, 30, 1000)) {
  out <- c(out, sprintf("(%s, %d, %d, 1, true, %s)", fmt(p), k, df, fmt(qtukey(p, k, df))))
}
for (p in c(0.05, 0.01)) for (r in c(1, 2)) {
  out <- c(out, sprintf("(%s, 4, 20, %d, false, %s)", fmt(p), r, fmt(qtukey(p, 4, 20, nranges = r, lower.tail = FALSE))))
}
rows(out)
