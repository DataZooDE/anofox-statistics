# pROC / base-R reference values for test/sql/reference/classification_reference.test (#153).
#
# Usage: Rscript validation/generators/make_reference_classification.R
# Same dataset as make_reference_classification.py (scikit-learn).
suppressMessages(library(pROC))
options(digits = 17)

y <- c(1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0)
s <- c(0.9, 0.1, 0.8, 0.65, 0.4, 0.3, 0.55, 0.55, 0.7, 0.2,
       0.35, 0.95, 0.6, 0.05, 0.45, 0.5, 0.25, 0.75, 0.55, 0.15)

# pROC: every distinct score is one point (= roc_agg(..., false)); pROC's
# thresholds are midpoints between scores, so compare fpr/tpr and the AUC only.
r <- roc(y, s, levels = c(0, 1), direction = "<", quiet = TRUE)
cat("## pROC::auc\n"); print(as.numeric(auc(r)))
co <- coords(r, "all", ret = c("specificity", "sensitivity"), transpose = FALSE)
cat("## pROC coords(all): fpr tpr (increasing fpr)\n")
print(unique(data.frame(fpr = 1 - co$specificity, tpr = co$sensitivity)))

# Calibration bins of width 0.2 (bin 0 = [0, 0.2], bin i = (0.2 i, 0.2 (i + 1)]),
# with the Wilson interval of prop.test(correct = FALSE).
bin <- pmax(ceiling(s / 0.2 - 1e-12) - 1, 0)
cat("## calibration n_bins = 5: bin n mean_pred frac_pos conf_low conf_high (95%, then 90%)\n")
for (b in sort(unique(bin))) {
  k <- bin == b
  x <- sum(y[k]); n <- sum(k)
  ci95 <- suppressWarnings(prop.test(x, n, correct = FALSE))$conf.int
  ci90 <- suppressWarnings(prop.test(x, n, conf.level = 0.9, correct = FALSE))$conf.int
  cat(b, n, mean(s[k]), x / n, ci95, ci90, "\n")
}

cat("## brier = mean((s - y)^2)\n"); print(mean((s - y)^2))
cat("## log loss = -mean(y log s + (1 - y) log(1 - s))\n"); print(-mean(y * log(s) + (1 - y) * log(1 - s)))
