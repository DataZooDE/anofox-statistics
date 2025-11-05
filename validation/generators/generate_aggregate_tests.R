#!/usr/bin/env Rscript
# Generate test data for aggregate regression validation
# This script should only be run when regenerating test data
# Output: test/data/aggregate_tests/

# Use local R library
local_lib <- file.path(dirname(dirname(getwd())), "validation", "R_libs")
if (dir.exists(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}

library(jsonlite)

cat("=== Generating Aggregate Test Data ===\n\n")

set.seed(42)  # For reproducibility

# Create output directories
test_dir <- "test/data/aggregate_tests"
dir.create(file.path(test_dir, "input"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(test_dir, "expected"), recursive = TRUE, showWarnings = FALSE)

# ==============================================================================
# Test 1: OLS Aggregate with GROUP BY (Two Groups)
# ==============================================================================

cat("Generating Test 1: OLS Aggregate with GROUP BY\n")

set.seed(42)
# Group A: Lower slope
group_a <- data.frame(
  category = "group_a",
  x1 = c(1, 2, 3, 4, 5),
  x2 = c(2, 4, 6, 8, 10),  # x2 = 2*x1 (collinear)
  y = c(3, 5, 7, 9, 11) + rnorm(5, 0, 0.5),
  weight = c(1.0, 1.0, 1.0, 1.0, 1.0)
)

# Group B: Higher slope
group_b <- data.frame(
  category = "group_b",
  x1 = c(2, 4, 6, 8, 10),
  x2 = c(4, 8, 12, 16, 20),  # x2 = 2*x1 (collinear)
  y = c(10, 20, 30, 40, 50) + rnorm(5, 0, 0.5),
  weight = c(1.0, 1.1, 1.2, 1.3, 1.4)
)

input1 <- rbind(group_a, group_b)
write.csv(input1, file.path(test_dir, "input/grouped_regression.csv"), row.names = FALSE)

# Compute expected results per group
expected1_groups <- by(input1, input1$category, function(df) {
  # OLS
  fit_ols <- lm(y ~ x1 + x2, data = df)
  summ_ols <- summary(fit_ols)

  # WLS
  fit_wls <- lm(y ~ x1 + x2, data = df, weights = weight)
  summ_wls <- summary(fit_wls)

  list(
    category = as.character(df$category[1]),
    ols = list(
      coefficients = as.numeric(coef(fit_ols)),
      intercept = coef(fit_ols)[1],
      r2 = summ_ols$r.squared,
      adj_r2 = summ_ols$adj.r.squared,
      sigma = summ_ols$sigma
    ),
    wls = list(
      coefficients = as.numeric(coef(fit_wls)),
      intercept = coef(fit_wls)[1],
      r2 = summ_wls$r.squared,
      adj_r2 = summ_wls$adj.r.squared,
      sigma = summ_wls$sigma
    )
  )
})

expected1 <- list(
  test_name = "grouped_regression",
  description = "OLS and WLS with GROUP BY, rank-deficient (x2 = 2*x1)",
  groups = lapply(expected1_groups, identity)
)

write_json(expected1, file.path(test_dir, "expected/grouped_regression.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = 15)

cat(sprintf("  ✓ Grouped regression: 2 groups, n=5 each\n"))

# ==============================================================================
# Test 2: Ridge Aggregate with GROUP BY
# ==============================================================================

cat("Generating Test 2: Ridge Aggregate with GROUP BY\n")

# Note: Ridge results require glmnet package
# For this generator, we'll just save the input data and note that
# Ridge validation requires glmnet

set.seed(43)
ridge_data <- data.frame(
  category = rep(c("cat_x", "cat_y"), each = 10),
  x1 = c(rnorm(10, 5, 1), rnorm(10, 8, 1)),
  x2 = c(rnorm(10, 10, 2), rnorm(10, 15, 2)),
  y = c(5 + 2*rnorm(10, 5, 1) + 3*rnorm(10, 10, 2) + rnorm(10, 0, 1),
        8 + 2*rnorm(10, 8, 1) + 3*rnorm(10, 15, 2) + rnorm(10, 0, 1))
)

write.csv(ridge_data, file.path(test_dir, "input/ridge_grouped.csv"), row.names = FALSE)

# Try to compute Ridge with glmnet if available
ridge_expected <- list(
  test_name = "ridge_grouped",
  description = "Ridge regression with lambda=1.0, grouped by category",
  note = "Ridge validation requires glmnet package"
)

if (requireNamespace("glmnet", quietly = TRUE)) {
  library(glmnet)

  ridge_expected$groups <- by(ridge_data, ridge_data$category, function(df) {
    X <- as.matrix(df[, c("x1", "x2")])
    y <- df$y
    n <- nrow(X)

    # Center for intercept
    y_mean <- mean(y)
    x_means <- colMeans(X)
    y_c <- y - y_mean
    X_c <- scale(X, center = TRUE, scale = FALSE)

    # glmnet uses lambda/n scaling
    lambda_param <- 1.0
    lambda_glmnet <- lambda_param / n

    fit <- glmnet(X_c, y_c, alpha = 0, lambda = lambda_glmnet, intercept = FALSE)
    beta <- as.numeric(coef(fit)[-1])
    intercept <- y_mean - sum(beta * x_means)

    # Compute R²
    y_pred <- intercept + X %*% beta
    ss_res <- sum((y - y_pred)^2)
    ss_tot <- sum((y - y_mean)^2)
    r2 <- 1 - ss_res / ss_tot

    list(
      category = as.character(df$category[1]),
      intercept = intercept,
      coefficients = beta,
      r2 = r2,
      lambda = lambda_param
    )
  })
  cat("  ✓ Ridge grouped: 2 categories, lambda=1.0, glmnet available\n")
} else {
  cat("  ⚠ Ridge grouped: Input data saved, glmnet not available for expected results\n")
}

write_json(ridge_expected, file.path(test_dir, "expected/ridge_grouped.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = 15)

# ==============================================================================
# Test 3: Intercept Handling (TRUE vs FALSE)
# ==============================================================================

cat("Generating Test 3: Intercept Handling Validation\n")

set.seed(44)
intercept_data <- data.frame(
  x1 = seq(1, 30, by = 1),
  x2 = seq(1, 30, by = 1) * 2.0,
  y = 10.0 + 3.0 * seq(1, 30, by = 1) + 1.5 * seq(1, 30, by = 1) * 2.0 + rnorm(30, 0, 2.0),
  weight = 1.0 + 0.05 * seq(1, 30, by = 1)
)

write.csv(intercept_data, file.path(test_dir, "input/intercept_test.csv"), row.names = FALSE)

# Compute with and without intercept
fit_with <- lm(y ~ x1 + x2, data = intercept_data)
fit_without <- lm(y ~ x1 + x2 - 1, data = intercept_data)
summ_with <- summary(fit_with)
summ_without <- summary(fit_without)

expected3 <- list(
  test_name = "intercept_handling",
  description = "Validates intercept=TRUE vs intercept=FALSE",
  with_intercept = list(
    intercept = coef(fit_with)[1],
    coefficients = as.numeric(coef(fit_with)[-1]),
    r2 = summ_with$r.squared,
    adj_r2 = summ_with$adj.r.squared,
    note = "SS_tot = sum((y - mean(y))^2)"
  ),
  without_intercept = list(
    intercept = 0.0,  # Must be exactly 0.0
    coefficients = as.numeric(coef(fit_without)),
    r2 = summ_without$r.squared,
    adj_r2 = summ_without$adj.r.squared,
    note = "SS_tot = sum(y^2), intercept forced to 0.0"
  ),
  validation_notes = list(
    r2_difference = "R² typically higher with intercept when data has non-zero mean",
    critical_test = "When intercept=FALSE, intercept field MUST be exactly 0.0",
    tolerance = "intercept=0.0 ± 1e-10"
  )
)

write_json(expected3, file.path(test_dir, "expected/intercept_test.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = 15)

cat(sprintf("  ✓ Intercept handling: with (R²=%.4f) vs without (R²=%.4f)\n",
            expected3$with_intercept$r2, expected3$without_intercept$r2))

# ==============================================================================
# Test 4: Known Intercept Recovery
# ==============================================================================

cat("Generating Test 4: Known Intercept Recovery\n")

set.seed(45)
known_intercept_data <- data.frame(
  x = seq(1, 20, by = 1),
  y = 50.0 + 2.0 * seq(1, 20, by = 1)  # True intercept = 50, slope = 2
)

write.csv(known_intercept_data, file.path(test_dir, "input/known_intercept.csv"), row.names = FALSE)

fit_known <- lm(y ~ x, data = known_intercept_data)
summ_known <- summary(fit_known)

expected4 <- list(
  test_name = "known_intercept",
  description = "Perfect linear data: y = 50 + 2*x (no noise)",
  true_intercept = 50.0,
  true_slope = 2.0,
  estimated_intercept = coef(fit_known)[1],
  estimated_slope = coef(fit_known)[2],
  r2 = summ_known$r.squared,
  note = "Should have R² = 1.0 (perfect fit)"
)

write_json(expected4, file.path(test_dir, "expected/known_intercept.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = 15)

cat(sprintf("  ✓ Known intercept: true=%.1f, estimated=%.10f, R²=%.10f\n",
            expected4$true_intercept, expected4$estimated_intercept, expected4$r2))

# ==============================================================================
# Test 5: RLS Test Data (Sequential Updates)
# ==============================================================================

cat("Generating Test 5: RLS Sequential Data\n")

set.seed(46)
rls_data <- data.frame(
  t = seq(1, 50, by = 1),
  x1 = cumsum(rnorm(50, 0.1, 0.5)),  # Random walk
  y = 10.0 + 2.0 * cumsum(rnorm(50, 0.1, 0.5)) + rnorm(50, 0, 1.0)
)

write.csv(rls_data, file.path(test_dir, "input/rls_sequential.csv"), row.names = FALSE)

# For RLS, we provide the final OLS estimate as a baseline
fit_rls_ols <- lm(y ~ x1, data = rls_data)
summ_rls <- summary(fit_rls_ols)

expected5 <- list(
  test_name = "rls_sequential",
  description = "Sequential data for RLS validation with forgetting_factor=1.0",
  ols_baseline = list(
    intercept = coef(fit_rls_ols)[1],
    slope = coef(fit_rls_ols)[2],
    r2 = summ_rls$r.squared,
    note = "OLS on full dataset (for comparison)"
  ),
  rls_notes = list(
    forgetting_factor = 1.0,
    algorithm = "Kalman-like sequential updates",
    initialization = "P = 1000*I, theta = 0",
    note = "With lambda=1.0, RLS converges to batch OLS"
  )
)

write_json(expected5, file.path(test_dir, "expected/rls_sequential.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = 15)

cat(sprintf("  ✓ RLS sequential: n=%d, OLS baseline R²=%.4f\n", nrow(rls_data), expected5$ols_baseline$r2))

# ==============================================================================
# Write metadata
# ==============================================================================

metadata <- list(
  generated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
  r_version = R.version.string,
  seed = 42,
  description = "Aggregate regression test data for GROUP BY validation",
  test_cases = list(
    grouped_regression = "OLS and WLS with GROUP BY (2 groups, rank-deficient)",
    ridge_grouped = "Ridge regression with GROUP BY (requires glmnet)",
    intercept_test = "Validates intercept=TRUE vs intercept=FALSE behavior",
    known_intercept = "Perfect fit with known true intercept (y = 50 + 2*x)",
    rls_sequential = "Sequential data for RLS validation (forgetting_factor=1.0)"
  ),
  validation_methodology = list(
    ols = "R lm(y ~ x, data=df) per group",
    wls = "R lm(y ~ x, weights=w, data=df) per group",
    ridge = "R glmnet(X, y, alpha=0, lambda=λ/n) per group",
    rls = "Custom RLS implementation with sequential Kalman-like updates"
  ),
  tolerance = list(
    strict = 1e-10,
    relaxed = 1e-8,
    note = "Use strict for coefficients/intercept, relaxed for R²"
  )
)

write_json(metadata, file.path(test_dir, "metadata.json"),
           auto_unbox = FALSE, pretty = TRUE)

cat("\n✅ Generated aggregate test data\n")
cat(sprintf("   Location: %s\n", test_dir))
cat("   Tests: 5 (grouped OLS/WLS, ridge, intercept handling, known intercept, RLS)\n")
