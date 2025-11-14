#!/usr/bin/env Rscript
# Comprehensive Prediction-Based Validation for Ridge and Elastic Net
#
# This script validates Ridge and Elastic Net implementations by:
# 1. Comparing predictions (not coefficients) against glmnet
# 2. Testing regularization path behavior
# 3. Verifying mathematical properties
#
# Rationale: Coefficients can differ numerically while producing identical
# predictions. Prediction accuracy is what ultimately matters.

library(DBI)
library(duckdb)
library(glmnet)

cat("=== Ridge and Elastic Net: Prediction-Based Validation ===\n\n")

# Configuration
EXTENSION_PATH <- "../build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension"
PRED_TOL <- 1e-10  # Strict tolerance for predictions
PATH_CORR_MIN <- 0.99  # Minimum correlation for regularization paths

# Test results tracking
all_tests_passed <- TRUE
test_results <- list()

# Helper: Compare predictions
compare_predictions <- function(pred_duckdb, pred_glmnet, test_name, tolerance = PRED_TOL) {
  mse_diff <- mean((pred_duckdb - pred_glmnet)^2)
  max_diff <- max(abs(pred_duckdb - pred_glmnet))

  if (mse_diff < tolerance && max_diff < sqrt(tolerance)) {
    cat(sprintf("  ✓ %s: MSE diff = %.2e, Max diff = %.2e\n", test_name, mse_diff, max_diff))
    return(TRUE)
  } else {
    cat(sprintf("  ✗ %s: FAILED\n", test_name))
    cat(sprintf("    MSE diff = %.2e (tolerance: %.2e)\n", mse_diff, tolerance))
    cat(sprintf("    Max diff = %.2e\n", max_diff))
    all_tests_passed <<- FALSE
    return(FALSE)
  }
}

# Connect to DuckDB
cat("Connecting to DuckDB...\n")
con <- dbConnect(
  duckdb::duckdb(config = list(allow_unsigned_extensions = "true"))
)

tryCatch({
  dbExecute(con, sprintf("LOAD '%s';", EXTENSION_PATH))
  cat("✓ Extension loaded successfully\n\n")
}, error = function(e) {
  cat("✗ Failed to load extension:\n")
  cat("  ", e$message, "\n")
  dbDisconnect(con, shutdown = TRUE)
  stop("Cannot continue without extension")
})

# ==============================================================================
# PHASE 2A: PREDICTION-BASED VALIDATION
# ==============================================================================
cat("\n=== Phase 2A: Prediction-Based Validation ===\n\n")

# ------------------------------------------------------------------------------
# Test 1: Ridge Regression - Train/Test Split
# ------------------------------------------------------------------------------
cat("--- Test 1: Ridge Prediction Validation (Train/Test Split) ---\n")

set.seed(42)
n <- 100
p <- 5

# Generate data
X <- matrix(rnorm(n * p), n, p)
beta_true <- c(2.5, -1.5, 3.0, 0.5, -2.0)
y <- X %*% beta_true + rnorm(n, sd = 0.5)

# Train/test split (80/20)
train_idx <- sample(1:n, 0.8 * n)
X_train <- X[train_idx, ]
y_train <- y[train_idx]
X_test <- X[-train_idx, ]
y_test <- y[-train_idx]

# Test multiple lambda values
lambda_values <- c(0.1, 0.5, 1.0, 5.0, 10.0)

cat(sprintf("Testing %d lambda values with train/test split...\n", length(lambda_values)))

for (lambda_param in lambda_values) {
  # Create temp table with training data
  train_data <- data.frame(
    y = y_train,
    x1 = X_train[,1],
    x2 = X_train[,2],
    x3 = X_train[,3],
    x4 = X_train[,4],
    x5 = X_train[,5]
  )

  dbExecute(con, "DROP TABLE IF EXISTS train_data")
  dbWriteTable(con, "train_data", train_data, overwrite = TRUE)

  # Fit DuckDB Ridge
  query <- sprintf("
    SELECT
      (result).intercept as intercept,
      (result).coefficients as coefficients
    FROM (
      SELECT anofox_statistics_ridge_agg(
        y, [x1, x2, x3, x4, x5],
        {'intercept': true, 'lambda': %.10f}
      ) as result
      FROM train_data
    ) sub
  ", lambda_param)

  duckdb_result <- dbGetQuery(con, query)

  # Fit glmnet Ridge
  lambda_glmnet <- lambda_param / nrow(X_train)
  glmnet_fit <- glmnet(X_train, y_train, alpha = 0, lambda = lambda_glmnet,
                       intercept = TRUE, standardize = FALSE)

  # Make predictions on test set
  pred_duckdb <- duckdb_result$intercept[1] +
                 X_test %*% unlist(duckdb_result$coefficients[[1]])
  pred_glmnet <- predict(glmnet_fit, newx = X_test, s = lambda_glmnet)

  test_name <- sprintf("Ridge λ=%.1f predictions", lambda_param)
  compare_predictions(pred_duckdb, pred_glmnet, test_name)
}

# ------------------------------------------------------------------------------
# Test 2: Elastic Net - Train/Test Split
# ------------------------------------------------------------------------------
cat("\n--- Test 2: Elastic Net Prediction Validation ---\n")

# Test combinations of alpha and lambda
alpha_values <- c(0.0, 0.3, 0.5, 0.7, 1.0)
lambda_values <- c(0.1, 1.0, 5.0)

cat(sprintf("Testing %d alpha × %d lambda combinations...\n",
            length(alpha_values), length(lambda_values)))

for (alpha_param in alpha_values) {
  for (lambda_param in lambda_values) {
    # Fit DuckDB Elastic Net
    query <- sprintf("
      SELECT
        (result).intercept as intercept,
        (result).coefficients as coefficients
      FROM (
        SELECT anofox_statistics_elastic_net_agg(
          y, [x1, x2, x3, x4, x5],
          {'intercept': true, 'alpha': %.2f, 'lambda': %.2f}
        ) as result
        FROM train_data
      ) sub
    ", alpha_param, lambda_param)

    duckdb_result <- dbGetQuery(con, query)

    # Fit glmnet Elastic Net
    lambda_glmnet <- lambda_param / nrow(X_train)
    glmnet_fit <- glmnet(X_train, y_train, alpha = alpha_param,
                         lambda = lambda_glmnet,
                         intercept = TRUE, standardize = FALSE)

    # Make predictions on test set
    pred_duckdb <- duckdb_result$intercept[1] +
                   X_test %*% unlist(duckdb_result$coefficients[[1]])
    pred_glmnet <- predict(glmnet_fit, newx = X_test, s = lambda_glmnet)

    test_name <- sprintf("Elastic Net α=%.1f λ=%.1f", alpha_param, lambda_param)
    compare_predictions(pred_duckdb, pred_glmnet, test_name, tolerance = PRED_TOL * 10)
  }
}

# ==============================================================================
# PHASE 2B: REGULARIZATION PATH TESTS
# ==============================================================================
cat("\n=== Phase 2B: Regularization Path Behavior ===\n\n")

# ------------------------------------------------------------------------------
# Test 3: Ridge Path - Coefficient Shrinkage
# ------------------------------------------------------------------------------
cat("--- Test 3: Ridge Regularization Path ---\n")

lambda_sequence <- 10^seq(-3, 2, length = 20)

# Compute coefficient norms for both implementations
duckdb_norms <- numeric(length(lambda_sequence))
glmnet_norms <- numeric(length(lambda_sequence))

for (i in seq_along(lambda_sequence)) {
  lambda_param <- lambda_sequence[i]

  # DuckDB
  query <- sprintf("
    SELECT (result).coefficients as coefficients
    FROM (
      SELECT anofox_statistics_ridge_agg(
        y, [x1, x2, x3, x4, x5],
        {'intercept': true, 'lambda': %.10f}
      ) as result
      FROM train_data
    ) sub
  ", lambda_param)

  duckdb_result <- dbGetQuery(con, query)
  coefs <- unlist(duckdb_result$coefficients[[1]])
  duckdb_norms[i] <- sqrt(sum(coefs^2))

  # glmnet
  lambda_glmnet <- lambda_param / nrow(X_train)
  glmnet_fit <- glmnet(X_train, y_train, alpha = 0, lambda = lambda_glmnet,
                       intercept = TRUE, standardize = FALSE)
  coefs_glmnet <- as.numeric(coef(glmnet_fit, s = lambda_glmnet))[-1]
  glmnet_norms[i] <- sqrt(sum(coefs_glmnet^2))
}

# Check monotonic decrease
duckdb_decreasing <- all(diff(duckdb_norms) < 1e-6)
glmnet_decreasing <- all(diff(glmnet_norms) < 1e-6)

if (duckdb_decreasing) {
  cat("  ✓ DuckDB: Coefficient norms decrease monotonically with λ\n")
} else {
  cat("  ✗ DuckDB: Coefficient norms do NOT decrease monotonically\n")
  all_tests_passed <- FALSE
}

if (glmnet_decreasing) {
  cat("  ✓ glmnet: Coefficient norms decrease monotonically with λ\n")
} else {
  cat("  ✗ glmnet: Coefficient norms do NOT decrease monotonically\n")
}

# Check path correlation
path_correlation <- cor(duckdb_norms, glmnet_norms)
if (path_correlation > PATH_CORR_MIN) {
  cat(sprintf("  ✓ Regularization paths highly correlated: r = %.4f\n", path_correlation))
} else {
  cat(sprintf("  ✗ Paths correlation too low: r = %.4f (minimum: %.2f)\n",
              path_correlation, PATH_CORR_MIN))
  all_tests_passed <- FALSE
}

# ------------------------------------------------------------------------------
# Test 4: Elastic Net Path - Sparsity Increases with λ
# ------------------------------------------------------------------------------
cat("\n--- Test 4: Elastic Net Sparsity (α=0.8) ---\n")

alpha_param <- 0.8
sparsity_duckdb <- numeric(length(lambda_sequence))
sparsity_glmnet <- numeric(length(lambda_sequence))

for (i in seq_along(lambda_sequence)) {
  lambda_param <- lambda_sequence[i]

  # DuckDB
  query <- sprintf("
    SELECT
      (result).coefficients as coefficients,
      (result).n_nonzero as n_nonzero
    FROM (
      SELECT anofox_statistics_elastic_net_agg(
        y, [x1, x2, x3, x4, x5],
        {'intercept': true, 'alpha': %.2f, 'lambda': %.10f}
      ) as result
      FROM train_data
    ) sub
  ", alpha_param, lambda_param)

  duckdb_result <- dbGetQuery(con, query)
  sparsity_duckdb[i] <- p - duckdb_result$n_nonzero[1]  # Number of zeros

  # glmnet
  lambda_glmnet <- lambda_param / nrow(X_train)
  glmnet_fit <- glmnet(X_train, y_train, alpha = alpha_param,
                       lambda = lambda_glmnet,
                       intercept = TRUE, standardize = FALSE)
  coefs_glmnet <- as.numeric(coef(glmnet_fit, s = lambda_glmnet))[-1]
  sparsity_glmnet[i] <- sum(abs(coefs_glmnet) < 1e-10)
}

# Sparsity should increase (or stay same) with lambda
duckdb_sparsity_ok <- all(diff(sparsity_duckdb) >= -1e-6)
glmnet_sparsity_ok <- all(diff(sparsity_glmnet) >= -1e-6)

if (duckdb_sparsity_ok) {
  cat(sprintf("  ✓ DuckDB: Sparsity increases with λ (max zeros: %d/%d)\n",
              max(sparsity_duckdb), p))
} else {
  cat("  ✗ DuckDB: Sparsity does NOT increase monotonically\n")
  all_tests_passed <- FALSE
}

if (glmnet_sparsity_ok) {
  cat(sprintf("  ✓ glmnet: Sparsity increases with λ (max zeros: %d/%d)\n",
              max(sparsity_glmnet), p))
}

# ==============================================================================
# PHASE 2C: MATHEMATICAL PROPERTY TESTS
# ==============================================================================
cat("\n=== Phase 2C: Mathematical Property Verification ===\n\n")

# ------------------------------------------------------------------------------
# Test 5: λ=0 Should Match OLS
# ------------------------------------------------------------------------------
cat("--- Test 5: Ridge with λ=0 Should Match OLS ---\n")

# Fit OLS
query_ols <- "
  SELECT
    (result).intercept as intercept,
    (result).coefficients as coefficients
  FROM (
    SELECT anofox_statistics_ols_agg(
      y, [x1, x2, x3, x4, x5],
      {'intercept': true}
    ) as result
    FROM train_data
  ) sub
"
ols_result <- dbGetQuery(con, query_ols)

# Fit Ridge with λ=0
query_ridge_zero <- "
  SELECT
    (result).intercept as intercept,
    (result).coefficients as coefficients
  FROM (
    SELECT anofox_statistics_ridge_agg(
      y, [x1, x2, x3, x4, x5],
      {'intercept': true, 'lambda': 0.0}
    ) as result
    FROM train_data
  ) sub
"
ridge_zero_result <- dbGetQuery(con, query_ridge_zero)

# Compare
intercept_diff <- abs(ols_result$intercept[1] - ridge_zero_result$intercept[1])
coef_diff <- max(abs(unlist(ols_result$coefficients[[1]]) -
                     unlist(ridge_zero_result$coefficients[[1]])))

if (intercept_diff < 1e-10 && coef_diff < 1e-10) {
  cat(sprintf("  ✓ Ridge(λ=0) matches OLS: max diff = %.2e\n", max(intercept_diff, coef_diff)))
} else {
  cat(sprintf("  ✗ Ridge(λ=0) does NOT match OLS\n"))
  cat(sprintf("    Intercept diff: %.2e\n", intercept_diff))
  cat(sprintf("    Coefficient diff: %.2e\n", coef_diff))
  all_tests_passed <- FALSE
}

# ------------------------------------------------------------------------------
# Test 6: Very Large λ Should Shrink Coefficients to Near Zero
# ------------------------------------------------------------------------------
cat("\n--- Test 6: Ridge with Large λ → Small Coefficients ---\n")

query_ridge_large <- "
  SELECT
    (result).coefficients as coefficients
  FROM (
    SELECT anofox_statistics_ridge_agg(
      y, [x1, x2, x3, x4, x5],
      {'intercept': true, 'lambda': 1e6}
    ) as result
    FROM train_data
  ) sub
"
ridge_large_result <- dbGetQuery(con, query_ridge_large)
coefs_large <- unlist(ridge_large_result$coefficients[[1]])
max_coef <- max(abs(coefs_large))

if (max_coef < 1e-3) {
  cat(sprintf("  ✓ Large λ shrinks coefficients: max |β| = %.2e\n", max_coef))
} else {
  cat(sprintf("  ✗ Coefficients NOT sufficiently shrunk: max |β| = %.2e\n", max_coef))
  all_tests_passed <- FALSE
}

# ------------------------------------------------------------------------------
# Test 7: Elastic Net Sparsity (α=1.0, high λ)
# ------------------------------------------------------------------------------
cat("\n--- Test 7: Lasso (α=1.0) Produces Exact Zeros ---\n")

query_lasso <- "
  SELECT
    (result).coefficients as coefficients,
    (result).n_nonzero as n_nonzero
  FROM (
    SELECT anofox_statistics_elastic_net_agg(
      y, [x1, x2, x3, x4, x5],
      {'intercept': true, 'alpha': 1.0, 'lambda': 2.0}
    ) as result
    FROM train_data
  ) sub
"
lasso_result <- dbGetQuery(con, query_lasso)
coefs_lasso <- unlist(lasso_result$coefficients[[1]])
n_zeros <- sum(abs(coefs_lasso) < 1e-10)
n_nonzero_reported <- lasso_result$n_nonzero[1]

if (n_zeros > 0 && n_zeros == (p - n_nonzero_reported)) {
  cat(sprintf("  ✓ Lasso produces sparsity: %d/%d coefficients exactly zero\n", n_zeros, p))
  cat(sprintf("  ✓ n_nonzero correctly reported: %d\n", n_nonzero_reported))
} else {
  cat(sprintf("  ✗ Sparsity issue: %d zeros, %d nonzero (expected: %d)\n",
              n_zeros, n_nonzero_reported, p))
  all_tests_passed <- FALSE
}

# ------------------------------------------------------------------------------
# Test 8: Collinear Features (Ridge should handle gracefully)
# ------------------------------------------------------------------------------
cat("\n--- Test 8: Ridge Handles Collinearity ---\n")

# Create collinear data
X_collinear <- cbind(X_train[,1], X_train[,1], X_train[,2:5])
collinear_data <- data.frame(
  y = y_train,
  x1 = X_collinear[,1],
  x2 = X_collinear[,2],  # Duplicate of x1
  x3 = X_collinear[,3],
  x4 = X_collinear[,4],
  x5 = X_collinear[,5],
  x6 = X_collinear[,6]
)

dbExecute(con, "DROP TABLE IF EXISTS collinear_data")
dbWriteTable(con, "collinear_data", collinear_data, overwrite = TRUE)

query_collinear <- "
  SELECT
    (result).intercept as intercept,
    (result).coefficients as coefficients
  FROM (
    SELECT anofox_statistics_ridge_agg(
      y, [x1, x2, x3, x4, x5, x6],
      {'intercept': true, 'lambda': 1.0}
    ) as result
    FROM collinear_data
  ) sub
"

collinear_result <- tryCatch({
  dbGetQuery(con, query_collinear)
}, error = function(e) {
  NULL
})

if (!is.null(collinear_result)) {
  coefs_collinear <- unlist(collinear_result$coefficients[[1]])
  # Check if first two coefficients are similar (both handling same signal)
  coef_ratio <- abs(coefs_collinear[1] / coefs_collinear[2])

  cat("  ✓ Ridge handles collinearity without error\n")
  cat(sprintf("  ✓ Collinear coefficients: β₁=%.4f, β₂=%.4f (ratio: %.2f)\n",
              coefs_collinear[1], coefs_collinear[2], coef_ratio))

  # Make predictions to verify validity
  pred_collinear <- collinear_result$intercept[1] +
                    X_test[,1] %*% t(coefs_collinear[1:6])
  if (!any(is.na(pred_collinear))) {
    cat("  ✓ Predictions are valid (no NaN/Inf)\n")
  } else {
    cat("  ✗ Predictions contain NaN/Inf\n")
    all_tests_passed <- FALSE
  }
} else {
  cat("  ✗ Ridge FAILED with collinear features\n")
  all_tests_passed <- FALSE
}

# ==============================================================================
# SUMMARY
# ==============================================================================
cat("\n" , paste(rep("=", 70), collapse=""), "\n")
cat("VALIDATION SUMMARY\n")
cat(paste(rep("=", 70), collapse=""), "\n\n")

if (all_tests_passed) {
  cat("✓✓✓ ALL TESTS PASSED ✓✓✓\n\n")
  cat("Ridge and Elastic Net implementations are validated:\n")
  cat("  - Predictions match glmnet\n")
  cat("  - Regularization paths behave correctly\n")
  cat("  - Mathematical properties verified\n")
  cat("  - Edge cases handled properly\n")
} else {
  cat("✗✗✗ SOME TESTS FAILED ✗✗✗\n\n")
  cat("Review failed tests above for details.\n")
}

# Cleanup
dbDisconnect(con, shutdown = TRUE)

if (!all_tests_passed) {
  stop("Validation failed")
}

cat("\n✓ Validation complete!\n")
