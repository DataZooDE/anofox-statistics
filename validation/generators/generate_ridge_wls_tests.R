#!/usr/bin/env Rscript
# Generate test data for Ridge and WLS regression validation
# This script should only be run when regenerating test data
# Output: test/data/ridge_tests/ and test/data/wls_tests/

# Use local R library
local_lib <- file.path(dirname(dirname(getwd())), "validation", "R_libs")
if (dir.exists(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}

library(jsonlite)
library(glmnet)

cat("=== Generating Ridge and WLS Test Data ===\n\n")

set.seed(42)  # For reproducibility

# ==============================================================================
# RIDGE REGRESSION TESTS
# ==============================================================================

cat("--- Ridge Regression Tests ---\n")

ridge_dir <- "test/data/ridge_tests"
dir.create(file.path(ridge_dir, "input"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(ridge_dir, "expected"), recursive = TRUE, showWarnings = FALSE)

# Test 1: Ridge with lambda = 0.1
cat("Generating Ridge Test 1: lambda = 0.1\n")

n_ridge <- 100
x_ridge <- matrix(rnorm(n_ridge * 3, mean = 5, sd = 2), ncol = 3)
y_ridge <- 10 + 2*x_ridge[,1] + 3*x_ridge[,2] - 1.5*x_ridge[,3] + rnorm(n_ridge, sd = 1.5)

# Save input data
input_ridge1 <- data.frame(
  x1 = x_ridge[,1],
  x2 = x_ridge[,2],
  x3 = x_ridge[,3],
  y = y_ridge
)
write.csv(input_ridge1, file.path(ridge_dir, "input/ridge_lambda_0.1.csv"), row.names = FALSE)

# Compute expected results with glmnet
lambda1 <- 0.1
ridge_model1 <- glmnet(x_ridge, y_ridge, alpha = 0, lambda = lambda1, standardize = FALSE, intercept = TRUE)

expected_ridge1 <- list(
  test_name = "ridge_lambda_0.1",
  lambda = lambda1,
  coefficients = as.numeric(coef(ridge_model1)),
  coefficient_names = c("intercept", "x1", "x2", "x3"),
  note = "Ridge regression with lambda=0.1, glmnet (alpha=0)"
)

write_json(expected_ridge1, file.path(ridge_dir, "expected/ridge_lambda_0.1.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = 15)

cat(sprintf("  ✓ Ridge lambda=0.1: n=%d, p=3\n", n_ridge))

# Test 2: Ridge with lambda = 1.0
cat("Generating Ridge Test 2: lambda = 1.0\n")

lambda2 <- 1.0
ridge_model2 <- glmnet(x_ridge, y_ridge, alpha = 0, lambda = lambda2, standardize = FALSE, intercept = TRUE)

# Save input data (same as test 1, but we'll save with different name for clarity)
write.csv(input_ridge1, file.path(ridge_dir, "input/ridge_lambda_1.0.csv"), row.names = FALSE)

expected_ridge2 <- list(
  test_name = "ridge_lambda_1.0",
  lambda = lambda2,
  coefficients = as.numeric(coef(ridge_model2)),
  coefficient_names = c("intercept", "x1", "x2", "x3"),
  note = "Ridge regression with lambda=1.0, glmnet (alpha=0)"
)

write_json(expected_ridge2, file.path(ridge_dir, "expected/ridge_lambda_1.0.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = 15)

cat(sprintf("  ✓ Ridge lambda=1.0: n=%d, p=3\n", n_ridge))

# Ridge metadata
ridge_metadata <- list(
  generated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
  r_version = R.version.string,
  glmnet_version = as.character(packageVersion("glmnet")),
  seed = 42,
  description = "Ridge regression test data with different lambda values",
  test_cases = list(
    ridge_lambda_0.1 = "Ridge with lambda=0.1 (light regularization)",
    ridge_lambda_1.0 = "Ridge with lambda=1.0 (stronger regularization)"
  ),
  note = "Uses glmnet with alpha=0, standardize=FALSE, intercept=TRUE"
)

write_json(ridge_metadata, file.path(ridge_dir, "metadata.json"),
           auto_unbox = FALSE, pretty = TRUE)

# ==============================================================================
# WEIGHTED LEAST SQUARES (WLS) TESTS
# ==============================================================================

cat("\n--- Weighted Least Squares Tests ---\n")

wls_dir <- "test/data/wls_tests"
dir.create(file.path(wls_dir, "input"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(wls_dir, "expected"), recursive = TRUE, showWarnings = FALSE)

# Test 1: WLS with equal weights (should match OLS)
cat("Generating WLS Test 1: Equal Weights\n")

n_wls <- 100
x_wls <- rnorm(n_wls, mean = 5, sd = 2)
y_wls <- 10 + 3*x_wls + rnorm(n_wls, sd = 1.5)
weights_equal <- rep(1.0, n_wls)

# Save input data
input_wls1 <- data.frame(x = x_wls, y = y_wls, weight = weights_equal)
write.csv(input_wls1, file.path(wls_dir, "input/wls_equal_weights.csv"), row.names = FALSE)

# Compute expected results with weighted lm
wls_model1 <- lm(y ~ x, data = input_wls1, weights = weight)
summary_wls1 <- summary(wls_model1)

expected_wls1 <- list(
  test_name = "wls_equal_weights",
  weights = "equal (all 1.0)",
  coefficients = as.numeric(coef(wls_model1)),
  coefficient_names = names(coef(wls_model1)),
  r_squared = summary_wls1$r.squared,
  adj_r_squared = summary_wls1$adj.r.squared,
  sigma = summary_wls1$sigma,
  residuals = as.numeric(residuals(wls_model1)),
  fitted_values = as.numeric(fitted(wls_model1)),
  note = "Equal weights - should match OLS"
)

write_json(expected_wls1, file.path(wls_dir, "expected/wls_equal_weights.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = 15)

cat(sprintf("  ✓ WLS equal weights: n=%d, R²=%.4f\n", n_wls, expected_wls1$r_squared))

# Test 2: WLS with inverse variance weights
cat("Generating WLS Test 2: Inverse Variance Weights\n")

n_wls2 <- 120
x_wls2 <- seq(1, 20, length.out = n_wls2)
# Heteroscedastic errors: variance increases with x
error_sd <- 0.5 + 0.3 * x_wls2
y_wls2 <- 5 + 2*x_wls2 + rnorm(n_wls2, sd = error_sd)
# Weights = 1 / variance
weights_inv_var <- 1 / (error_sd^2)

# Save input data
input_wls2 <- data.frame(x = x_wls2, y = y_wls2, weight = weights_inv_var)
write.csv(input_wls2, file.path(wls_dir, "input/wls_inverse_variance.csv"), row.names = FALSE)

# Compute expected results
wls_model2 <- lm(y ~ x, data = input_wls2, weights = weight)
summary_wls2 <- summary(wls_model2)

expected_wls2 <- list(
  test_name = "wls_inverse_variance",
  weights = "inverse_variance (1/sigma^2)",
  coefficients = as.numeric(coef(wls_model2)),
  coefficient_names = names(coef(wls_model2)),
  r_squared = summary_wls2$r.squared,
  adj_r_squared = summary_wls2$adj.r.squared,
  sigma = summary_wls2$sigma,
  residuals = as.numeric(residuals(wls_model2)),
  fitted_values = as.numeric(fitted(wls_model2)),
  note = "Inverse variance weights for heteroscedastic data"
)

write_json(expected_wls2, file.path(wls_dir, "expected/wls_inverse_variance.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = 15)

cat(sprintf("  ✓ WLS inverse variance: n=%d, R²=%.4f\n", n_wls2, expected_wls2$r_squared))

# WLS metadata
wls_metadata <- list(
  generated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
  r_version = R.version.string,
  seed = 42,
  description = "Weighted Least Squares test data with different weight schemes",
  test_cases = list(
    wls_equal_weights = "Equal weights (should match OLS)",
    wls_inverse_variance = "Inverse variance weights for heteroscedastic data"
  ),
  note = "Uses R lm() with weights parameter"
)

write_json(wls_metadata, file.path(wls_dir, "metadata.json"),
           auto_unbox = FALSE, pretty = TRUE)

cat("\n✅ Generated Ridge and WLS test data\n")
cat(sprintf("   Ridge tests: %s (2 tests)\n", ridge_dir))
cat(sprintf("   WLS tests: %s (2 tests)\n", wls_dir))
