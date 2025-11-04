# OLS Validation Tests: DuckDB vs R
# Validates anofox_statistics_ols_fit against R's lm()

library(DBI)
library(duckdb)

cat("=== OLS Validation: DuckDB Extension vs R ===\n\n")

# Configuration
EXTENSION_PATH <- "build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension"
STRICT_TOL <- 1e-10
RELAXED_TOL <- 1e-8

# Test results tracking
test_results <- list()
test_counter <- 0

# Helper function to compare values
compare_values <- function(duckdb_val, r_val, tolerance = STRICT_TOL,
                          test_name = "", value_name = "") {
  test_counter <<- test_counter + 1

  # Handle NULL/NA
  if ((is.null(duckdb_val) || is.na(duckdb_val)) && is.na(r_val)) {
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
con <- dbConnect(duckdb::duckdb())

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
# Test 1: Simple Linear Regression
# ==============================================================================
cat("\n--- Test 1: Simple Linear Regression ---\n")
cat("Testing: y = 2x + noise\n")

x <- c(1, 2, 3, 4, 5)
y <- c(2.1, 4.2, 5.9, 8.1, 10.0)

# R computation
r_model <- lm(y ~ x)
r_summary <- summary(r_model)

r_intercept <- coef(r_model)[1]
r_slope <- coef(r_model)[2]
r_r2 <- r_summary$r.squared
r_adj_r2 <- r_summary$adj.r.squared
r_rmse <- sqrt(mean(residuals(r_model)^2))
r_residuals <- residuals(r_model)

cat("\nR Results:\n")
cat("  Intercept:", r_intercept, "\n")
cat("  Slope:", r_slope, "\n")
cat("  R²:", r_r2, "\n")
cat("  Adjusted R²:", r_adj_r2, "\n")
cat("  RMSE:", r_rmse, "\n")

# DuckDB query
query <- "
SELECT * FROM anofox_statistics_ols_fit(
  y := [2.1, 4.2, 5.9, 8.1, 10.0]::DOUBLE[],
  x1 := [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
  add_intercept := true
);
"

duckdb_result <- dbGetQuery(con, query)

cat("\nDuckDB Results:\n")
cat("  Coefficients:", paste(duckdb_result$coefficients[[1]], collapse = ", "), "\n")
cat("  R²:", duckdb_result$r_squared, "\n")
cat("  Adjusted R²:", duckdb_result$adj_r_squared, "\n")
cat("  RMSE:", duckdb_result$rmse, "\n")

# Compare
cat("\nComparison:\n")
duckdb_coefs <- duckdb_result$coefficients[[1]]
compare_values(duckdb_coefs[1], r_intercept, STRICT_TOL, "Test1", "intercept")
compare_values(duckdb_coefs[2], r_slope, STRICT_TOL, "Test1", "slope")
compare_values(duckdb_result$r_squared, r_r2, STRICT_TOL, "Test1", "r_squared")
compare_values(duckdb_result$adj_r_squared, r_adj_r2, STRICT_TOL, "Test1", "adj_r_squared")
compare_values(duckdb_result$rmse, r_rmse, STRICT_TOL, "Test1", "rmse")

# ==============================================================================
# Test 2: Multiple Regression
# ==============================================================================
cat("\n--- Test 2: Multiple Regression ---\n")
cat("Testing: y = b0 + b1*x1 + b2*x2 + b3*x3\n")

x1 <- c(1, 2, 3, 4, 5)
x2 <- c(2, 4, 5, 4, 5)
x3 <- c(1, 1, 2, 2, 3)
y <- c(5, 10, 15, 18, 25)

# R computation
df <- data.frame(y, x1, x2, x3)
r_model <- lm(y ~ x1 + x2 + x3, data = df)
r_summary <- summary(r_model)

r_coefs <- coef(r_model)
r_r2 <- r_summary$r.squared

cat("\nR Coefficients:\n")
print(r_coefs)

# DuckDB query
query <- "
SELECT * FROM anofox_statistics_ols_fit(
  y := [5.0, 10.0, 15.0, 18.0, 25.0]::DOUBLE[],
  x1 := [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
  x2 := [2.0, 4.0, 5.0, 4.0, 5.0]::DOUBLE[],
  x3 := [1.0, 1.0, 2.0, 2.0, 3.0]::DOUBLE[],
  add_intercept := true
);
"

duckdb_result <- dbGetQuery(con, query)
duckdb_coefs <- duckdb_result$coefficients[[1]]

cat("\nDuckDB Coefficients:\n")
cat(paste(duckdb_coefs, collapse = ", "), "\n")

# Compare
cat("\nComparison:\n")
for (i in seq_along(r_coefs)) {
  coef_name <- paste0("coef_", i)
  compare_values(duckdb_coefs[i], r_coefs[i], STRICT_TOL, "Test2", coef_name)
}
compare_values(duckdb_result$r_squared, r_r2, STRICT_TOL, "Test2", "r_squared")

# ==============================================================================
# Test 3: OLS without Intercept
# ==============================================================================
cat("\n--- Test 3: OLS without Intercept ---\n")
cat("Testing: y = b*x (no intercept)\n")

x <- c(1, 2, 3, 4, 5)
y <- c(2, 4, 6, 8, 10)

# R computation
r_model <- lm(y ~ x - 1)  # -1 removes intercept
r_slope <- coef(r_model)[1]

cat("\nR slope:", r_slope, "\n")

# DuckDB query
query <- "
SELECT * FROM anofox_statistics_ols_fit(
  y := [2.0, 4.0, 6.0, 8.0, 10.0]::DOUBLE[],
  x1 := [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
  add_intercept := false
);
"

duckdb_result <- dbGetQuery(con, query)
duckdb_coef <- duckdb_result$coefficients[[1]][1]

cat("DuckDB slope:", duckdb_coef, "\n")

# Compare
cat("\nComparison:\n")
compare_values(duckdb_coef, r_slope, STRICT_TOL, "Test3", "slope_no_intercept")

# ==============================================================================
# Test 4: Rank-Deficient - Constant Feature
# ==============================================================================
cat("\n--- Test 4: Rank-Deficient (Constant Feature) ---\n")
cat("Testing: Constant feature should return NA/NULL\n")

x1 <- c(1, 2, 3, 4, 5)
x2 <- c(5, 5, 5, 5, 5)  # Constant feature
y <- c(2, 4, 6, 8, 10)

# R computation
df <- data.frame(y, x1, x2)
r_model <- lm(y ~ x1 + x2, data = df)
r_coefs <- coef(r_model)

cat("\nR Coefficients:\n")
print(r_coefs)
cat("Note: NA indicates aliased/rank-deficient\n")

# DuckDB query
query <- "
SELECT * FROM anofox_statistics_ols_fit(
  y := [2.0, 4.0, 6.0, 8.0, 10.0]::DOUBLE[],
  x1 := [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
  x2 := [5.0, 5.0, 5.0, 5.0, 5.0]::DOUBLE[],
  add_intercept := true
);
"

duckdb_result <- dbGetQuery(con, query)
duckdb_coefs <- duckdb_result$coefficients[[1]]

cat("\nDuckDB Coefficients:\n")
cat(paste(duckdb_coefs, collapse = ", "), "\n")
cat("Note: NULL indicates aliased/rank-deficient\n")

# Compare
cat("\nComparison:\n")
compare_values(duckdb_coefs[1], r_coefs[1], STRICT_TOL, "Test4", "intercept")
compare_values(duckdb_coefs[2], r_coefs[2], STRICT_TOL, "Test4", "x1_coef")
# x2 should be NA in R and NULL in DuckDB
compare_values(duckdb_coefs[3], r_coefs[3], STRICT_TOL, "Test4", "x2_coef_constant")

# ==============================================================================
# Test 5: Perfect Collinearity
# ==============================================================================
cat("\n--- Test 5: Perfect Collinearity ---\n")
cat("Testing: x2 = 2*x1 (perfect collinearity)\n")

x1 <- c(1, 2, 3, 4, 5)
x2 <- c(2, 4, 6, 8, 10)  # x2 = 2*x1
y <- c(3, 5, 7, 9, 11)

# R computation
df <- data.frame(y, x1, x2)
r_model <- lm(y ~ x1 + x2, data = df)
r_coefs <- coef(r_model)

cat("\nR Coefficients:\n")
print(r_coefs)

# DuckDB query
query <- "
SELECT * FROM anofox_statistics_ols_fit(
  y := [3.0, 5.0, 7.0, 9.0, 11.0]::DOUBLE[],
  x1 := [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
  x2 := [2.0, 4.0, 6.0, 8.0, 10.0]::DOUBLE[],
  add_intercept := true
);
"

duckdb_result <- dbGetQuery(con, query)
duckdb_coefs <- duckdb_result$coefficients[[1]]

cat("\nDuckDB Coefficients:\n")
cat(paste(duckdb_coefs, collapse = ", "), "\n")

# Compare
cat("\nComparison:\n")
for (i in seq_along(r_coefs)) {
  coef_name <- paste0("coef_", i)
  compare_values(duckdb_coefs[i], r_coefs[i], STRICT_TOL, "Test5", coef_name)
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

# Save results
results_file <- "validation/results/ols_validation_results.csv"
write.csv(results_df, results_file, row.names = FALSE)
cat("\nResults saved to:", results_file, "\n")

# Disconnect
dbDisconnect(con, shutdown = TRUE)

cat("\n=== OLS Validation Complete ===\n")

# Return results invisibly
invisible(results_df)
