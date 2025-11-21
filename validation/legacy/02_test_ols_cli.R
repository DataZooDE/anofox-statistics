# OLS Validation Tests: DuckDB Extension vs R
# Uses DuckDB CLI for queries instead of R duckdb package

cat("=== OLS Validation: DuckDB Extension vs R ===\n")
cat("Method: DuckDB CLI + R comparison\n\n")

# Configuration
DUCKDB_CLI <- "/tmp/duckdb"
EXTENSION_PATH <- "../../build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension"
STRICT_TOL <- 1e-10
RELAXED_TOL <- 1e-8

# Test results tracking
test_results <- list()
test_counter <- 0

# Helper: Run DuckDB query and parse result
run_duckdb_query <- function(query) {
  # Create temp file for query
  query_file <- tempfile(fileext = ".sql")
  writeLines(query, query_file)

  # Run DuckDB CLI
  cmd <- sprintf("%s -unsigned -csv < %s", DUCKDB_CLI, query_file)
  result <- system(cmd, intern = TRUE)

  # Clean up
  unlink(query_file)

  return(result)
}

# Helper: Parse CSV output from DuckDB
parse_duckdb_csv <- function(csv_lines) {
  if (length(csv_lines) < 2) {
    return(NULL)
  }

  # Use read.csv to properly handle quoted fields with commas
  con <- textConnection(paste(csv_lines, collapse = "\n"))
  df <- tryCatch({
    read.csv(con, stringsAsFactors = FALSE, na.strings = c("", "NULL"))
  }, error = function(e) {
    cat("Error parsing CSV:\n")
    cat(paste(csv_lines, collapse = "\n"), "\n")
    return(NULL)
  })
  close(con)

  if (is.null(df) || nrow(df) == 0) {
    return(NULL)
  }

  # Convert to named list and parse special types
  result <- list()
  for (col_name in names(df)) {
    value <- df[[col_name]][1]

    # Handle arrays (check for bracket syntax)
    if (is.character(value) && !is.na(value) && grepl("^\\[", value)) {
      # Remove brackets and parse as numeric vector
      value <- gsub("^\\[|\\]$", "", value)
      value_vec <- suppressWarnings(as.numeric(strsplit(value, ",\\s*")[[1]]))
      value <- value_vec
    } else if (is.character(value) && !is.na(value)) {
      # Try to convert to numeric
      value_num <- suppressWarnings(as.numeric(value))
      if (!is.na(value_num)) {
        value <- value_num
      }
    }
    # else keep as-is (including NA)

    result[[col_name]] <- value
  }

  return(result)
}

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
      duckdb = NA,
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
    cat(sprintf("  ✓ %s - %s: %.10e (diff: %.2e)\n",
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

# Check DuckDB CLI exists
if (!file.exists(DUCKDB_CLI)) {
  cat("✗ DuckDB CLI not found at:", DUCKDB_CLI, "\n")
  cat("  Please download DuckDB CLI or update DUCKDB_CLI path\n")
  stop("Cannot continue without DuckDB CLI")
}

# Check extension exists
# Resolve to absolute path
if (file.exists(EXTENSION_PATH)) {
  ext_full_path <- normalizePath(EXTENSION_PATH)
} else {
  # Try relative to parent directory
  ext_full_path <- file.path(dirname(dirname(getwd())), EXTENSION_PATH)
  if (!file.exists(ext_full_path)) {
    ext_full_path <- normalizePath(ext_full_path, mustWork = FALSE)
    cat("✗ Extension not found at:", ext_full_path, "\n")
    cat("  Please build extension: GEN=ninja make release\n")
    stop("Cannot continue without extension")
  }
  ext_full_path <- normalizePath(ext_full_path)
}

cat("✓ DuckDB CLI found\n")
cat("✓ Extension found\n\n")

# ==============================================================================
# Test 1: Simple Linear Regression
# ==============================================================================
cat("--- Test 1: Simple Linear Regression ---\n")
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
r_rmse <- summary(r_model)$sigma  # Residual standard error (uses df adjustment)

cat("\nR Results:\n")
cat(sprintf("  Intercept: %.10f\n", r_intercept))
cat(sprintf("  Slope:     %.10f\n", r_slope))
cat(sprintf("  R²:        %.10f\n", r_r2))
cat(sprintf("  Adj R²:    %.10f\n", r_adj_r2))
cat(sprintf("  RMSE:      %.10f\n", r_rmse))

# DuckDB query (matrix-based API with MAP options)
query <- sprintf("
LOAD '%s';
SELECT * FROM anofox_statistics_ols(
  [2.1, 4.2, 5.9, 8.1, 10.0]::DOUBLE[],
  [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],
  {'intercept': true}
);
", ext_full_path)

csv_result <- run_duckdb_query(query)
duckdb_result <- parse_duckdb_csv(csv_result)

if (is.null(duckdb_result)) {
  cat("✗ Failed to parse DuckDB result\n")
  cat("Output:\n")
  cat(paste(csv_result, collapse = "\n"), "\n")
  stop("Cannot parse DuckDB result")
}

cat("\nDuckDB Results:\n")
cat(sprintf("  Intercept:    %.10f\n", duckdb_result$intercept))
cat(sprintf("  Slope:        %.10f\n", duckdb_result$coefficients[1]))
cat(sprintf("  R²:           %.10f\n", duckdb_result$r_squared))
cat(sprintf("  Adj R²:       %.10f\n", duckdb_result$adj_r_squared))
cat(sprintf("  RMSE:         %.10f\n", duckdb_result$rmse))

# Compare
cat("\nComparison:\n")
compare_values(duckdb_result$intercept, r_intercept, STRICT_TOL, "Test1", "intercept")
compare_values(duckdb_result$coefficients[1], r_slope, STRICT_TOL, "Test1", "slope")
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
for (i in seq_along(r_coefs)) {
  cat(sprintf("  %s: %.10f\n", names(r_coefs)[i], r_coefs[i]))
}
cat(sprintf("R²: %.10f\n", r_r2))

# DuckDB query (matrix-based API with MAP options)
query <- sprintf("
LOAD '%s';
SELECT * FROM anofox_statistics_ols(
  [5.0, 10.0, 15.0, 18.0, 25.0]::DOUBLE[],
  [[1.0, 2.0, 1.0], [2.0, 4.0, 1.0], [3.0, 5.0, 2.0], [4.0, 4.0, 2.0], [5.0, 5.0, 3.0]]::DOUBLE[][],
  {'intercept': true}
);
", ext_full_path)

csv_result <- run_duckdb_query(query)
duckdb_result <- parse_duckdb_csv(csv_result)

cat("\nDuckDB Results:\n")
cat(sprintf("  Intercept: %.10f\n", duckdb_result$intercept))
for (i in seq_along(duckdb_result$coefficients)) {
  cat(sprintf("  Slope[%d]: %.10f\n", i, duckdb_result$coefficients[i]))
}
cat(sprintf("  R²: %.10f\n", duckdb_result$r_squared))

# Compare (R coefficients: [intercept, slope1, slope2, slope3])
cat("\nComparison:\n")
compare_values(duckdb_result$intercept, r_coefs[1], STRICT_TOL, "Test2", "intercept")
for (i in seq_along(duckdb_result$coefficients)) {
  coef_name <- sprintf("slope_%d", i)
  compare_values(duckdb_result$coefficients[i], r_coefs[i+1], STRICT_TOL, "Test2", coef_name)
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

cat(sprintf("\nR slope: %.10f\n", r_slope))

# DuckDB query (matrix-based API with intercept=false)
query <- sprintf("
LOAD '%s';
SELECT * FROM anofox_statistics_ols(
  [2.0, 4.0, 6.0, 8.0, 10.0]::DOUBLE[],
  [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],
  {'intercept': false}
);
", ext_full_path)

csv_result <- run_duckdb_query(query)
duckdb_result <- parse_duckdb_csv(csv_result)

cat(sprintf("DuckDB slope: %.10f\n", duckdb_result$coefficients[1]))

# Compare
cat("\nComparison:\n")
compare_values(duckdb_result$coefficients[1], r_slope, STRICT_TOL, "Test3", "slope_no_intercept")

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
for (i in seq_along(r_coefs)) {
  val <- if (is.na(r_coefs[i])) "NA" else sprintf("%.10f", r_coefs[i])
  cat(sprintf("  %s: %s\n", names(r_coefs)[i], val))
}
cat("Note: NA indicates aliased/rank-deficient\n")

# DuckDB query (matrix-based API)
query <- sprintf("
LOAD '%s';
SELECT * FROM anofox_statistics_ols(
  [2.0, 4.0, 6.0, 8.0, 10.0]::DOUBLE[],
  [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0], [5.0, 5.0]]::DOUBLE[][],
  {'intercept': true}
);
", ext_full_path)

csv_result <- run_duckdb_query(query)
duckdb_result <- parse_duckdb_csv(csv_result)

cat("\nDuckDB Results:\n")
cat(sprintf("  Intercept: %s\n", if (is.na(duckdb_result$intercept)) "NULL" else sprintf("%.10f", duckdb_result$intercept)))
for (i in seq_along(duckdb_result$coefficients)) {
  val <- if (is.na(duckdb_result$coefficients[i])) "NULL" else sprintf("%.10f", duckdb_result$coefficients[i])
  cat(sprintf("  Slope[%d]: %s\n", i, val))
}
cat("Note: NULL indicates aliased/rank-deficient\n")

# Compare (R: [intercept, x1_coef, x2_coef])
cat("\nComparison:\n")
compare_values(duckdb_result$intercept, r_coefs[1], STRICT_TOL, "Test4", "intercept")
compare_values(duckdb_result$coefficients[1], r_coefs[2], STRICT_TOL, "Test4", "x1_coef")
compare_values(duckdb_result$coefficients[2], r_coefs[3], STRICT_TOL, "Test4", "x2_coef_constant")

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
for (i in seq_along(r_coefs)) {
  val <- if (is.na(r_coefs[i])) "NA" else sprintf("%.10f", r_coefs[i])
  cat(sprintf("  %s: %s\n", names(r_coefs)[i], val))
}

# DuckDB query (matrix-based API)
query <- sprintf("
LOAD '%s';
SELECT * FROM anofox_statistics_ols(
  [3.0, 5.0, 7.0, 9.0, 11.0]::DOUBLE[],
  [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0]]::DOUBLE[][],
  {'intercept': true}
);
", ext_full_path)

csv_result <- run_duckdb_query(query)
duckdb_result <- parse_duckdb_csv(csv_result)

cat("\nDuckDB Results:\n")
cat(sprintf("  Intercept: %s\n", if (is.na(duckdb_result$intercept)) "NULL" else sprintf("%.10f", duckdb_result$intercept)))
for (i in seq_along(duckdb_result$coefficients)) {
  val <- if (is.na(duckdb_result$coefficients[i])) "NULL" else sprintf("%.10f", duckdb_result$coefficients[i])
  cat(sprintf("  Slope[%d]: %s\n", i, val))
}

# Compare - Special case for perfect collinearity
# R and DuckDB may choose different columns, both are valid
cat("\nComparison:\n")
compare_values(duckdb_result$intercept, r_coefs[1], STRICT_TOL, "Test5", "intercept")

# For perfect collinearity, check that:
# 1. Exactly one coefficient is non-NULL
# 2. The fit is perfect (checked via fitted values or R²)
n_non_null_duckdb <- sum(!is.na(duckdb_result$coefficients))
n_non_null_r <- sum(!is.na(r_coefs[-1]))  # Exclude intercept

cat(sprintf("  Note: R chose %d non-aliased column(s), DuckDB chose %d\n",
            n_non_null_r, n_non_null_duckdb))

if (n_non_null_duckdb == 1 && n_non_null_r == 1) {
  cat("  ✓ Test5 - Both systems correctly identified 1 non-aliased column\n")
  test_results[[length(test_results) + 1]] <- list(
    test = test_counter + 1,
    name = "Test5",
    value = "collinearity_detection",
    status = "PASS",
    reason = "Correct number of non-aliased columns",
    duckdb = n_non_null_duckdb,
    r = n_non_null_r,
    diff = 0
  )
  test_counter <<- test_counter + 1

  # Both solutions should give perfect fit - check R² if available
  if (!is.na(duckdb_result$r_squared)) {
    r_r2 <- summary(r_model)$r.squared
    compare_values(duckdb_result$r_squared, r_r2, STRICT_TOL, "Test5", "r_squared")
  }
} else {
  cat("  ✗ Test5 - Column selection mismatch\n")
  test_results[[length(test_results) + 1]] <- list(
    test = test_counter + 1,
    name = "Test5",
    value = "collinearity_detection",
    status = "FAIL",
    reason = "Wrong number of non-aliased columns",
    duckdb = n_non_null_duckdb,
    r = n_non_null_r,
    diff = abs(n_non_null_duckdb - n_non_null_r)
  )
  test_counter <<- test_counter + 1
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
results_dir <- "results"
if (!dir.exists(results_dir)) {
  dir.create(results_dir, recursive = TRUE)
}

results_file <- file.path(results_dir, "ols_validation_results.csv")
write.csv(results_df, results_file, row.names = FALSE)
cat("\nResults saved to:", results_file, "\n")

cat("\n=== OLS Validation Complete ===\n")

# Return success/failure
if (failed == 0) {
  cat("\n✓✓✓ ALL TESTS PASSED ✓✓✓\n")
  quit(status = 0)
} else {
  cat("\n✗✗✗ SOME TESTS FAILED ✗✗✗\n")
  quit(status = 1)
}
