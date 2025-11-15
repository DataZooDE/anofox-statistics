#!/usr/bin/env Rscript

# Elastic Net Validation Tests: DuckDB vs R (glmnet)
# Validates anofox_statistics_elastic_net against R's glmnet package

library(DBI)
library(duckdb)
library(glmnet)

cat("=== Elastic Net Validation: DuckDB Extension vs R (glmnet) ===\n\n")

# Configuration
EXTENSION_PATH <- "../build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension"
STRICT_TOL <- 1e-8  # Elastic Net may have slight differences due to convergence criteria
RELAXED_TOL <- 1e-6

# Test results tracking
test_results <- list()
test_counter <- 0

# Helper function to compare values
compare_values <- function(duckdb_val, r_val, tolerance = STRICT_TOL,
                          test_name = "", value_name = "") {
  test_counter <<- test_counter + 1

  # Handle NULL/NA
  if ((is.null(duckdb_val) || is.na(duckdb_val)) && (is.null(r_val) || is.na(r_val))) {
    result <- list(
      test = test_counter,
      name = test_name,
      value = value_name,
      status = "PASS",
      reason = "Both NULL/NA",
      duckdb = duckdb_val,
      r = r_val,
      diff = 0
    )
    cat(sprintf("  ✓ %s - %s: Both NULL/NA\n", test_name, value_name))
    test_results[[test_counter]] <<- result
    return(TRUE)
  }

  # Calculate difference
  diff <- abs(duckdb_val - r_val)

  if (diff < tolerance) {
    result <- list(
      test = test_counter,
      name = test_name,
      value = value_name,
      status = "PASS",
      reason = "Within tolerance",
      duckdb = duckdb_val,
      r = r_val,
      diff = diff
    )
    cat(sprintf("  ✓ %s - %s: %.2e (diff: %.2e)\n",
                test_name, value_name, duckdb_val, diff))
    test_results[[test_counter]] <<- result
    return(TRUE)
  } else {
    result <- list(
      test = test_counter,
      name = test_name,
      value = value_name,
      status = "FAIL",
      reason = "Exceeds tolerance",
      duckdb = duckdb_val,
      r = r_val,
      diff = diff
    )
    cat(sprintf("  ✗ %s - %s: MISMATCH\n", test_name, value_name))
    cat(sprintf("    DuckDB: %.15f\n", duckdb_val))
    cat(sprintf("    R:      %.15f\n", r_val))
    cat(sprintf("    Diff:   %.2e (tolerance: %.2e)\n", diff, tolerance))
    test_results[[test_counter]] <<- result
    return(FALSE)
  }
}

# Connect to DuckDB
cat("Connecting to DuckDB...\n")
con <- dbConnect(
  duckdb::duckdb(config = list(allow_unsigned_extensions = "true"))
)

# Load extension
cat("Loading extension:", EXTENSION_PATH, "\n")
tryCatch({
  dbExecute(con, sprintf("LOAD '%s';", EXTENSION_PATH))
  cat("✓ Extension loaded successfully\n\n")
}, error = function(e) {
  cat("✗ Failed to load extension:\n")
  cat("  ", e$message, "\n")
  cat("  Make sure the extension is built at:", EXTENSION_PATH, "\n")
  dbDisconnect(con, shutdown = TRUE)
  stop("Cannot continue without extension")
})

# ==============================================================================
# Test 1: Basic Elastic Net (alpha=0.5, lambda=0.1)
# ==============================================================================
cat("\n--- Test 1: Basic Elastic Net (alpha=0.5, lambda=0.1) ---\n")
cat("Testing: Balanced L1/L2 regularization\n")

set.seed(42)
x1 <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
x2 <- c(2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
x3 <- c(1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5)
y <- 2 + 3*x1 + 0.5*x2 + 2*x3 + rnorm(10, 0, 0.5)

alpha_param <- 0.5
lambda_param <- 0.1

# R computation using glmnet
X_matrix <- cbind(x1, x2, x3)
n <- length(y)

# glmnet uses lambda/n internally, we use lambda directly
# To match: glmnet_lambda = our_lambda / n
glmnet_lambda <- lambda_param / n

fit <- glmnet(X_matrix, y, alpha = alpha_param, lambda = glmnet_lambda,
              intercept = TRUE, standardize = FALSE, thresh = 1e-7)

r_intercept <- as.numeric(coef(fit)[1])
r_coefs <- as.numeric(coef(fit)[-1])

# Calculate R² manually
y_pred <- predict(fit, newx = X_matrix, s = glmnet_lambda)
ss_res <- sum((y - y_pred)^2)
ss_tot <- sum((y - mean(y))^2)
r_r2 <- 1 - ss_res / ss_tot

# Count non-zero coefficients
r_n_nonzero <- sum(abs(r_coefs) > 1e-10)

cat("\nR Results (glmnet):\n")
cat(sprintf("  Intercept:  %.10f\n", r_intercept))
cat(sprintf("  Coef 1:     %.10f\n", r_coefs[1]))
cat(sprintf("  Coef 2:     %.10f\n", r_coefs[2]))
cat(sprintf("  Coef 3:     %.10f\n", r_coefs[3]))
cat(sprintf("  R²:         %.10f\n", r_r2))
cat(sprintf("  n_nonzero:  %d\n", r_n_nonzero))
cat(sprintf("  Alpha:      %.2f\n", alpha_param))
cat(sprintf("  Lambda:     %.4f\n", lambda_param))

# DuckDB query
query <- sprintf("
SELECT * FROM anofox_statistics_elastic_net(
  [%s]::DOUBLE[],
  [[%s, %s, %s], [%s, %s, %s], [%s, %s, %s], [%s, %s, %s], [%s, %s, %s],
   [%s, %s, %s], [%s, %s, %s], [%s, %s, %s], [%s, %s, %s], [%s, %s, %s]]::DOUBLE[][],
  {'alpha': %f, 'lambda': %f, 'intercept': true}
);
",
  paste(y, collapse = ", "),
  x1[1], x2[1], x3[1], x1[2], x2[2], x3[2], x1[3], x2[3], x3[3],
  x1[4], x2[4], x3[4], x1[5], x2[5], x3[5], x1[6], x2[6], x3[6],
  x1[7], x2[7], x3[7], x1[8], x2[8], x3[8], x1[9], x2[9], x3[9],
  x1[10], x2[10], x3[10],
  alpha_param, lambda_param)

duckdb_result <- dbGetQuery(con, query)

cat("\nDuckDB Results:\n")
cat(sprintf("  Intercept:  %.10f\n", duckdb_result$intercept))
cat(sprintf("  Coef 1:     %.10f\n", duckdb_result$coefficients[[1]][1]))
cat(sprintf("  Coef 2:     %.10f\n", duckdb_result$coefficients[[1]][2]))
cat(sprintf("  Coef 3:     %.10f\n", duckdb_result$coefficients[[1]][3]))
cat(sprintf("  R²:         %.10f\n", duckdb_result$r_squared))
cat(sprintf("  n_nonzero:  %d\n", duckdb_result$n_nonzero))
cat(sprintf("  Alpha:      %.2f\n", duckdb_result$alpha))
cat(sprintf("  Lambda:     %.4f\n", duckdb_result$reg_lambda))
cat(sprintf("  Converged:  %s\n", duckdb_result$converged))

# Compare
cat("\nComparison:\n")
compare_values(duckdb_result$intercept, r_intercept, STRICT_TOL, "Test1", "intercept")
compare_values(duckdb_result$coefficients[[1]][1], r_coefs[1], STRICT_TOL, "Test1", "coef_1")
compare_values(duckdb_result$coefficients[[1]][2], r_coefs[2], STRICT_TOL, "Test1", "coef_2")
compare_values(duckdb_result$coefficients[[1]][3], r_coefs[3], STRICT_TOL, "Test1", "coef_3")
compare_values(duckdb_result$r_squared, r_r2, STRICT_TOL, "Test1", "r_squared")
compare_values(duckdb_result$n_nonzero, r_n_nonzero, 0, "Test1", "n_nonzero")

# ==============================================================================
# Test 2: Pure Ridge (alpha=0.0) - Should match Ridge results
# ==============================================================================
cat("\n--- Test 2: Pure Ridge (alpha=0.0) ---\n")
cat("Testing: Elastic Net with alpha=0 should match Ridge regression\n")

alpha_ridge <- 0.0
lambda_ridge <- 1.0

# R Ridge (glmnet with alpha=0)
glmnet_lambda_ridge <- lambda_ridge / n
fit_ridge <- glmnet(X_matrix, y, alpha = alpha_ridge, lambda = glmnet_lambda_ridge,
                    intercept = TRUE, standardize = FALSE, thresh = 1e-7)

r_intercept_ridge <- as.numeric(coef(fit_ridge)[1])
r_coefs_ridge <- as.numeric(coef(fit_ridge)[-1])

y_pred_ridge <- predict(fit_ridge, newx = X_matrix, s = glmnet_lambda_ridge)
ss_res_ridge <- sum((y - y_pred_ridge)^2)
r_r2_ridge <- 1 - ss_res_ridge / ss_tot

cat("\nR Results (Ridge via glmnet):\n")
cat(sprintf("  Intercept:  %.10f\n", r_intercept_ridge))
cat(sprintf("  Coef 1:     %.10f\n", r_coefs_ridge[1]))
cat(sprintf("  Coef 2:     %.10f\n", r_coefs_ridge[2]))
cat(sprintf("  Coef 3:     %.10f\n", r_coefs_ridge[3]))
cat(sprintf("  R²:         %.10f\n", r_r2_ridge))

# DuckDB Elastic Net with alpha=0
query_ridge <- sprintf("
SELECT * FROM anofox_statistics_elastic_net(
  [%s]::DOUBLE[],
  [[%s, %s, %s], [%s, %s, %s], [%s, %s, %s], [%s, %s, %s], [%s, %s, %s],
   [%s, %s, %s], [%s, %s, %s], [%s, %s, %s], [%s, %s, %s], [%s, %s, %s]]::DOUBLE[][],
  {'alpha': %f, 'lambda': %f, 'intercept': true}
);
",
  paste(y, collapse = ", "),
  x1[1], x2[1], x3[1], x1[2], x2[2], x3[2], x1[3], x2[3], x3[3],
  x1[4], x2[4], x3[4], x1[5], x2[5], x3[5], x1[6], x2[6], x3[6],
  x1[7], x2[7], x3[7], x1[8], x2[8], x3[8], x1[9], x2[9], x3[9],
  x1[10], x2[10], x3[10],
  alpha_ridge, lambda_ridge)

duckdb_result_ridge <- dbGetQuery(con, query_ridge)

cat("\nDuckDB Results (Elastic Net with alpha=0):\n")
cat(sprintf("  Intercept:  %.10f\n", duckdb_result_ridge$intercept))
cat(sprintf("  Coef 1:     %.10f\n", duckdb_result_ridge$coefficients[[1]][1]))
cat(sprintf("  Coef 2:     %.10f\n", duckdb_result_ridge$coefficients[[1]][2]))
cat(sprintf("  Coef 3:     %.10f\n", duckdb_result_ridge$coefficients[[1]][3]))
cat(sprintf("  R²:         %.10f\n", duckdb_result_ridge$r_squared))

# Compare
cat("\nComparison:\n")
compare_values(duckdb_result_ridge$intercept, r_intercept_ridge, STRICT_TOL, "Test2", "intercept")
compare_values(duckdb_result_ridge$coefficients[[1]][1], r_coefs_ridge[1], STRICT_TOL, "Test2", "coef_1")
compare_values(duckdb_result_ridge$coefficients[[1]][2], r_coefs_ridge[2], STRICT_TOL, "Test2", "coef_2")
compare_values(duckdb_result_ridge$coefficients[[1]][3], r_coefs_ridge[3], STRICT_TOL, "Test2", "coef_3")

# ==============================================================================
# Test 3: Pure Lasso (alpha=1.0) - Test Sparsity
# ==============================================================================
cat("\n--- Test 3: Pure Lasso (alpha=1.0) ---\n")
cat("Testing: L1 regularization creates sparse solutions\n")

# Create data with one irrelevant feature
set.seed(42)
x1_lasso <- rnorm(20, 0, 1)
x2_lasso <- rnorm(20, 0, 1)
x3_lasso <- rnorm(20, 0, 0.01)  # Nearly irrelevant feature
y_lasso <- 2 + 5*x1_lasso + 0.5*x2_lasso + rnorm(20, 0, 0.5)

alpha_lasso <- 1.0
lambda_lasso <- 0.5

X_lasso <- cbind(x1_lasso, x2_lasso, x3_lasso)
n_lasso <- length(y_lasso)
glmnet_lambda_lasso <- lambda_lasso / n_lasso

fit_lasso <- glmnet(X_lasso, y_lasso, alpha = alpha_lasso, lambda = glmnet_lambda_lasso,
                    intercept = TRUE, standardize = FALSE, thresh = 1e-7)

r_intercept_lasso <- as.numeric(coef(fit_lasso)[1])
r_coefs_lasso <- as.numeric(coef(fit_lasso)[-1])
r_n_nonzero_lasso <- sum(abs(r_coefs_lasso) > 1e-10)

cat("\nR Results (Lasso via glmnet):\n")
cat(sprintf("  Intercept:  %.10f\n", r_intercept_lasso))
cat(sprintf("  Coef 1:     %.10f\n", r_coefs_lasso[1]))
cat(sprintf("  Coef 2:     %.10f\n", r_coefs_lasso[2]))
cat(sprintf("  Coef 3:     %.10f (should be ~0)\n", r_coefs_lasso[3]))
cat(sprintf("  n_nonzero:  %d (sparsity)\n", r_n_nonzero_lasso))

# Build matrix rows for DuckDB
matrix_rows_lasso <- paste(sprintf("[%.10f, %.10f, %.10f]", x1_lasso, x2_lasso, x3_lasso), collapse = ", ")

query_lasso <- sprintf("
SELECT * FROM anofox_statistics_elastic_net(
  [%s]::DOUBLE[],
  [%s]::DOUBLE[][],
  {'alpha': %f, 'lambda': %f, 'intercept': true}
);
",
  paste(y_lasso, collapse = ", "),
  matrix_rows_lasso,
  alpha_lasso, lambda_lasso)

duckdb_result_lasso <- dbGetQuery(con, query_lasso)

cat("\nDuckDB Results (Lasso):\n")
cat(sprintf("  Intercept:  %.10f\n", duckdb_result_lasso$intercept))
cat(sprintf("  Coef 1:     %.10f\n", duckdb_result_lasso$coefficients[[1]][1]))
cat(sprintf("  Coef 2:     %.10f\n", duckdb_result_lasso$coefficients[[1]][2]))
cat(sprintf("  Coef 3:     %.10f\n", duckdb_result_lasso$coefficients[[1]][3]))
cat(sprintf("  n_nonzero:  %d\n", duckdb_result_lasso$n_nonzero))

# Compare
cat("\nComparison:\n")
compare_values(duckdb_result_lasso$intercept, r_intercept_lasso, STRICT_TOL, "Test3", "intercept")
compare_values(duckdb_result_lasso$coefficients[[1]][1], r_coefs_lasso[1], STRICT_TOL, "Test3", "coef_1")
compare_values(duckdb_result_lasso$coefficients[[1]][2], r_coefs_lasso[2], STRICT_TOL, "Test3", "coef_2")
compare_values(duckdb_result_lasso$coefficients[[1]][3], r_coefs_lasso[3], STRICT_TOL, "Test3", "coef_3")
compare_values(duckdb_result_lasso$n_nonzero, r_n_nonzero_lasso, 0, "Test3", "n_nonzero")

# ==============================================================================
# Test 4: No Intercept (intercept=false)
# ==============================================================================
cat("\n--- Test 4: Elastic Net without Intercept ---\n")
cat("Testing: intercept=false option\n")

alpha_no_int <- 0.5
lambda_no_int <- 0.1

# Use first test data
glmnet_lambda_no_int <- lambda_no_int / n
fit_no_int <- glmnet(X_matrix, y, alpha = alpha_no_int, lambda = glmnet_lambda_no_int,
                     intercept = FALSE, standardize = FALSE, thresh = 1e-7)

r_coefs_no_int <- as.numeric(coef(fit_no_int)[-1])  # Intercept will be 0

cat("\nR Results (no intercept):\n")
cat(sprintf("  Coef 1:     %.10f\n", r_coefs_no_int[1]))
cat(sprintf("  Coef 2:     %.10f\n", r_coefs_no_int[2]))
cat(sprintf("  Coef 3:     %.10f\n", r_coefs_no_int[3]))

query_no_int <- sprintf("
SELECT * FROM anofox_statistics_elastic_net(
  [%s]::DOUBLE[],
  [[%s, %s, %s], [%s, %s, %s], [%s, %s, %s], [%s, %s, %s], [%s, %s, %s],
   [%s, %s, %s], [%s, %s, %s], [%s, %s, %s], [%s, %s, %s], [%s, %s, %s]]::DOUBLE[][],
  {'alpha': %f, 'lambda': %f, 'intercept': false}
);
",
  paste(y, collapse = ", "),
  x1[1], x2[1], x3[1], x1[2], x2[2], x3[2], x1[3], x2[3], x3[3],
  x1[4], x2[4], x3[4], x1[5], x2[5], x3[5], x1[6], x2[6], x3[6],
  x1[7], x2[7], x3[7], x1[8], x2[8], x3[8], x1[9], x2[9], x3[9],
  x1[10], x2[10], x3[10],
  alpha_no_int, lambda_no_int)

duckdb_result_no_int <- dbGetQuery(con, query_no_int)

cat("\nDuckDB Results (no intercept):\n")
cat(sprintf("  Intercept:  %.10f (should be 0)\n", duckdb_result_no_int$intercept))
cat(sprintf("  Coef 1:     %.10f\n", duckdb_result_no_int$coefficients[[1]][1]))
cat(sprintf("  Coef 2:     %.10f\n", duckdb_result_no_int$coefficients[[1]][2]))
cat(sprintf("  Coef 3:     %.10f\n", duckdb_result_no_int$coefficients[[1]][3]))

# Compare
cat("\nComparison:\n")
compare_values(duckdb_result_no_int$coefficients[[1]][1], r_coefs_no_int[1], STRICT_TOL, "Test4", "coef_1")
compare_values(duckdb_result_no_int$coefficients[[1]][2], r_coefs_no_int[2], STRICT_TOL, "Test4", "coef_2")
compare_values(duckdb_result_no_int$coefficients[[1]][3], r_coefs_no_int[3], STRICT_TOL, "Test4", "coef_3")

# ==============================================================================
# Test 5: Aggregate Function with GROUP BY
# ==============================================================================
cat("\n--- Test 5: Elastic Net Aggregate with GROUP BY ---\n")
cat("Testing: Per-group elastic net regression\n")

# Create test data with groups
set.seed(42)
test_data <- data.frame(
  category = rep(c("group_a", "group_b"), each = 10),
  x1 = c(1:10, 2*(1:10)),
  y = c(3 + 2*(1:10) + rnorm(10, 0, 0.5),
        5 + 4*(2*(1:10)) + rnorm(10, 0, 1))
)

# Upload to DuckDB
dbWriteTable(con, "elastic_test_data", test_data, overwrite = TRUE)

# R computation per group
alpha_agg <- 0.5
lambda_agg <- 0.1

cat("\nComputing R baseline per group...\n")
r_agg_results <- by(test_data, test_data$category, function(df) {
  X_group <- as.matrix(df$x1)
  y_group <- df$y
  n_group <- length(y_group)
  glmnet_lambda_group <- lambda_agg / n_group

  fit_group <- glmnet(X_group, y_group, alpha = alpha_agg, lambda = glmnet_lambda_group,
                      intercept = TRUE, standardize = FALSE, thresh = 1e-7)

  list(
    category = df$category[1],
    intercept = as.numeric(coef(fit_group)[1]),
    coef_x1 = as.numeric(coef(fit_group)[2]),
    n_nonzero = sum(abs(coef(fit_group)[-1]) > 1e-10)
  )
})
r_agg_results <- do.call(rbind, lapply(r_agg_results, as.data.frame))
row.names(r_agg_results) <- NULL

cat("R results per group:\n")
print(r_agg_results, digits = 10)

# DuckDB aggregate
cat("\nComputing DuckDB aggregate...\n")
duckdb_agg_result <- dbGetQuery(con, sprintf("
  SELECT
    category,
    (result).intercept as intercept,
    (result).coefficients[1] as coef_x1,
    (result).n_nonzero as n_nonzero
  FROM (
    SELECT
      category,
      anofox_statistics_elastic_net_agg(y, [x1], {'alpha': %f, 'lambda': %f, 'intercept': true}) as result
    FROM elastic_test_data
    GROUP BY category
  ) sub
  ORDER BY category
", alpha_agg, lambda_agg))

cat("DuckDB aggregate results:\n")
print(duckdb_agg_result, digits = 10)

# Compare
for (i in 1:nrow(r_agg_results)) {
  group <- r_agg_results$category[i]
  cat(sprintf("\nValidating group: %s\n", group))

  r_row <- r_agg_results[i, ]
  d_row <- duckdb_agg_result[duckdb_agg_result$category == group, ]

  compare_values(d_row$intercept, r_row$intercept,
                 test_name = paste("ElasticNet_agg", group), value_name = "intercept")
  compare_values(d_row$coef_x1, r_row$coef_x1,
                 test_name = paste("ElasticNet_agg", group), value_name = "coef_x1")
  compare_values(d_row$n_nonzero, r_row$n_nonzero, 0,
                 test_name = paste("ElasticNet_agg", group), value_name = "n_nonzero")
}

# ==============================================================================
# Summary
# ==============================================================================
cat("\n\n=== VALIDATION SUMMARY ===\n")

# Convert results to dataframe
results_df <- do.call(rbind, lapply(test_results, as.data.frame))

total_tests <- nrow(results_df)
passed <- sum(results_df$status == "PASS")
failed <- sum(results_df$status == "FAIL")

cat("Total comparisons:", total_tests, "\n")
cat("Passed:", passed, "\n")
cat("Failed:", failed, "\n")
cat("Success rate:", sprintf("%.1f%%", 100 * passed / total_tests), "\n")

if (failed > 0) {
  cat("\nFailed tests:\n")
  failed_tests <- results_df[results_df$status == "FAIL", ]
  print(failed_tests[, c("name", "value", "duckdb", "r", "diff")])
}

# Disconnect
dbDisconnect(con, shutdown = TRUE)

cat("\n=== Elastic Net Validation Complete ===\n")

# Return results invisibly
invisible(results_df)
