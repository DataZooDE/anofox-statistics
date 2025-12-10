#!/usr/bin/env Rscript
# Generate test data for OLS regression validation
# This script should only be run when regenerating test data
# Output: test/data/ols_tests/

# Use local R library
local_lib <- file.path(dirname(dirname(getwd())), "validation", "R_libs")
if (dir.exists(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}

library(jsonlite)

cat("=== Generating OLS Test Data ===\n\n")

set.seed(42)  # For reproducibility

# Create output directories
test_dir <- "test/data/ols_tests"
dir.create(file.path(test_dir, "input"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(test_dir, "expected"), recursive = TRUE, showWarnings = FALSE)

# ==============================================================================
# Test 1: Simple Linear Regression (y = 2x + 1 + noise)
# ==============================================================================

cat("Generating Test 1: Simple Linear Regression\n")

n1 <- 100
x1 <- seq(1, 10, length.out = n1)
y1 <- 2 * x1 + 1 + rnorm(n1, mean = 0, sd = 0.5)

# Save input data
input1 <- data.frame(x = x1, y = y1)
write.csv(input1, file.path(test_dir, "input/simple_linear.csv"), row.names = FALSE)

# Compute expected results with R
model1 <- lm(y ~ x, data = input1)
summary1 <- summary(model1)

expected1 <- list(
  test_name = "simple_linear",
  coefficients = as.numeric(coef(model1)),
  coefficient_names = names(coef(model1)),
  r_squared = summary1$r.squared,
  adj_r_squared = summary1$adj.r.squared,
  sigma = summary1$sigma,
  residuals = as.numeric(residuals(model1)),
  fitted_values = as.numeric(fitted(model1)),
  df_residual = summary1$df[2]
)

write_json(expected1, file.path(test_dir, "expected/simple_linear.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = 15)

cat(sprintf("  ✓ Simple linear: n=%d, R²=%.4f\n", n1, expected1$r_squared))

# ==============================================================================
# Test 2: Multiple Regression (3 predictors)
# ==============================================================================

cat("Generating Test 2: Multiple Regression\n")

n2 <- 200
x2_1 <- rnorm(n2, mean = 5, sd = 2)
x2_2 <- rnorm(n2, mean = 10, sd = 3)
x2_3 <- rnorm(n2, mean = 15, sd = 4)
y2 <- 10 + 2*x2_1 + 3*x2_2 - 1.5*x2_3 + rnorm(n2, mean = 0, sd = 2)

# Save input data
input2 <- data.frame(x1 = x2_1, x2 = x2_2, x3 = x2_3, y = y2)
write.csv(input2, file.path(test_dir, "input/multiple_regression.csv"), row.names = FALSE)

# Compute expected results
model2 <- lm(y ~ x1 + x2 + x3, data = input2)
summary2 <- summary(model2)

expected2 <- list(
  test_name = "multiple_regression",
  coefficients = as.numeric(coef(model2)),
  coefficient_names = names(coef(model2)),
  r_squared = summary2$r.squared,
  adj_r_squared = summary2$adj.r.squared,
  sigma = summary2$sigma,
  residuals = as.numeric(residuals(model2)),
  fitted_values = as.numeric(fitted(model2)),
  df_residual = summary2$df[2]
)

write_json(expected2, file.path(test_dir, "expected/multiple_regression.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = 15)

cat(sprintf("  ✓ Multiple regression: n=%d, p=3, R²=%.4f\n", n2, expected2$r_squared))

# ==============================================================================
# Test 3: No Intercept
# ==============================================================================

cat("Generating Test 3: No Intercept Regression\n")

n3 <- 150
x3 <- seq(0, 20, length.out = n3)
y3 <- 3 * x3 + rnorm(n3, mean = 0, sd = 1)

# Save input data
input3 <- data.frame(x = x3, y = y3)
write.csv(input3, file.path(test_dir, "input/no_intercept.csv"), row.names = FALSE)

# Compute expected results (no intercept)
model3 <- lm(y ~ x - 1, data = input3)
summary3 <- summary(model3)

expected3 <- list(
  test_name = "no_intercept",
  coefficients = as.numeric(coef(model3)),
  coefficient_names = names(coef(model3)),
  r_squared = summary3$r.squared,
  adj_r_squared = summary3$adj.r.squared,
  sigma = summary3$sigma,
  residuals = as.numeric(residuals(model3)),
  fitted_values = as.numeric(fitted(model3)),
  df_residual = summary3$df[2],
  note = "Model fitted without intercept"
)

write_json(expected3, file.path(test_dir, "expected/no_intercept.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = 15)

cat(sprintf("  ✓ No intercept: n=%d, R²=%.4f\n", n3, expected3$r_squared))

# ==============================================================================
# Test 4: Rank Deficient (constant feature)
# ==============================================================================

cat("Generating Test 4: Rank Deficient (Constant Feature)\n")

n4 <- 100
x4_1 <- rnorm(n4, mean = 5, sd = 2)
x4_2 <- rep(10, n4)  # Constant feature
y4 <- 5 + 2*x4_1 + rnorm(n4, mean = 0, sd = 1)

# Save input data
input4 <- data.frame(x1 = x4_1, x2 = x4_2, y = y4)
write.csv(input4, file.path(test_dir, "input/rank_deficient.csv"), row.names = FALSE)

# Compute expected results (R drops constant column)
model4 <- lm(y ~ x1 + x2, data = input4)
summary4 <- summary(model4)

expected4 <- list(
  test_name = "rank_deficient",
  coefficients = as.numeric(coef(model4)),
  coefficient_names = names(coef(model4)),
  r_squared = summary4$r.squared,
  adj_r_squared = summary4$adj.r.squared,
  sigma = summary4$sigma,
  residuals = as.numeric(residuals(model4)),
  fitted_values = as.numeric(fitted(model4)),
  df_residual = summary4$df[2],
  note = "x2 is constant (R drops it automatically)"
)

write_json(expected4, file.path(test_dir, "expected/rank_deficient.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = 15)

cat(sprintf("  ✓ Rank deficient: n=%d, constant feature dropped\n", n4))

# ==============================================================================
# Test 5: Perfect Collinearity
# ==============================================================================

cat("Generating Test 5: Perfect Collinearity\n")

n5 <- 80
x5_1 <- rnorm(n5, mean = 5, sd = 2)
x5_2 <- 2 * x5_1 + 3  # Perfectly collinear with x1
y5 <- 5 + 3*x5_1 + rnorm(n5, mean = 0, sd = 1)

# Save input data
input5 <- data.frame(x1 = x5_1, x2 = x5_2, y = y5)
write.csv(input5, file.path(test_dir, "input/perfect_collinearity.csv"), row.names = FALSE)

# Compute expected results (R drops collinear column)
model5 <- lm(y ~ x1 + x2, data = input5)
summary5 <- summary(model5)

expected5 <- list(
  test_name = "perfect_collinearity",
  coefficients = as.numeric(coef(model5)),
  coefficient_names = names(coef(model5)),
  r_squared = summary5$r.squared,
  adj_r_squared = summary5$adj.r.squared,
  sigma = summary5$sigma,
  residuals = as.numeric(residuals(model5)),
  fitted_values = as.numeric(fitted(model5)),
  df_residual = summary5$df[2],
  note = "x2 = 2*x1 + 3 (R drops it automatically)"
)

write_json(expected5, file.path(test_dir, "expected/perfect_collinearity.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = 15)

cat(sprintf("  ✓ Perfect collinearity: n=%d, collinear feature dropped\n", n5))

# ==============================================================================
# Write metadata
# ==============================================================================

metadata <- list(
  generated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
  r_version = R.version.string,
  seed = 42,
  description = "OLS regression test data with 5 test cases",
  test_cases = list(
    simple_linear = "Simple linear regression with 1 predictor",
    multiple_regression = "Multiple regression with 3 predictors",
    no_intercept = "Regression without intercept term",
    rank_deficient = "Rank-deficient matrix (constant feature)",
    perfect_collinearity = "Perfect collinearity between predictors"
  ),
  tolerance = list(
    strict = 1e-10,
    relaxed = 1e-8,
    note = "Use strict for coefficients/R², relaxed for p-values"
  )
)

write_json(metadata, file.path(test_dir, "metadata.json"),
           auto_unbox = FALSE, pretty = TRUE)

cat("\n✅ Generated OLS test data\n")
cat(sprintf("   Location: %s\n", test_dir))
cat("   Tests: 5 (simple, multiple, no-intercept, rank-deficient, collinear)\n")
