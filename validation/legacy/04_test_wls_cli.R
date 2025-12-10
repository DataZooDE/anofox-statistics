#!/usr/bin/env Rscript

# Weighted Least Squares (WLS) Validation: DuckDB Extension vs R (lm with weights)
#
# Validates anofox_statistics_wls against R's lm() with weights parameter
# Uses DuckDB CLI for extension testing

# Configuration
STRICT_TOL <- 1e-10  # For coefficients, R²
RELAXED_TOL <- 1e-8  # For p-values

DUCKDB_CLI <- "/tmp/duckdb"
EXTENSION_PATH <- "../../build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension"

# Helper: Parse DuckDB CSV output
parse_duckdb_csv <- function(csv_lines) {
  con <- textConnection(paste(csv_lines, collapse = "\n"))
  df <- tryCatch({
    read.csv(con, stringsAsFactors = FALSE, na.strings = c("", "NULL", "nan"))
  }, error = function(e) {
    cat("Error parsing CSV:\n")
    cat(paste(csv_lines, collapse = "\n"))
    return(NULL)
  })
  close(con)

  if (is.null(df) || nrow(df) == 0) {
    return(NULL)
  }

  # Convert to named list and parse arrays
  result <- list()
  for (col_name in names(df)) {
    value <- df[[col_name]][1]

    # Handle arrays (format: [1.0, 2.0, 3.0])
    if (is.character(value) && !is.na(value) && grepl("^\\[", value)) {
      value <- gsub("^\\[|\\]$", "", value)
      value <- suppressWarnings(as.numeric(strsplit(value, ",\\s*")[[1]]))
    }

    result[[col_name]] <- value
  }

  return(result)
}

# Helper: Run DuckDB query
run_duckdb_query <- function(query) {
  temp_file <- tempfile(fileext = ".sql")
  writeLines(query, temp_file)

  # Separate stderr to avoid mixing log messages with CSV output
  stderr_file <- tempfile()
  output <- system2(
    DUCKDB_CLI,
    args = c("-unsigned", "-csv"),
    stdin = temp_file,
    stdout = TRUE,
    stderr = stderr_file
  )

  # Read stderr for error checking
  stderr_content <- readLines(stderr_file, warn = FALSE)

  unlink(temp_file)
  unlink(stderr_file)

  # Check for errors in stderr
  error_lines <- grep("Error|Fehler", stderr_content, value = TRUE)
  if (length(error_lines) > 0) {
    cat("DuckDB Error:\n")
    cat(paste(stderr_content, collapse = "\n"), "\n")
    return(NULL)
  }

  return(output)
}

# Helper: Compare values with tolerance
compare_values <- function(duckdb_val, r_val, tol, test_name, metric_name) {
  # Handle NULL/NA
  if ((is.null(duckdb_val) || is.na(duckdb_val)) && (is.null(r_val) || is.na(r_val))) {
    cat(sprintf("  ✓ %s - %s: Both NULL/NA\n", test_name, metric_name))
    return(TRUE)
  }

  if ((is.null(duckdb_val) || is.na(duckdb_val)) || (is.null(r_val) || is.na(r_val))) {
    cat(sprintf("  ✗ %s - %s: NULL/NA mismatch\n", test_name, metric_name))
    cat(sprintf("    DuckDB: %s\n", ifelse(is.null(duckdb_val) || is.na(duckdb_val), "NULL/NA", duckdb_val)))
    cat(sprintf("    R:      %s\n", ifelse(is.null(r_val) || is.na(r_val), "NULL/NA", r_val)))
    return(FALSE)
  }

  diff <- abs(duckdb_val - r_val)
  if (diff < tol) {
    cat(sprintf("  ✓ %s - %s: %.10e (diff: %.2e)\n", test_name, metric_name, duckdb_val, diff))
    return(TRUE)
  } else {
    cat(sprintf("  ✗ %s - %s: MISMATCH\n", test_name, metric_name))
    cat(sprintf("    DuckDB: %.15f\n", duckdb_val))
    cat(sprintf("    R:      %.15f\n", r_val))
    cat(sprintf("    Diff:   %.2e (tolerance: %.2e)\n", diff, tol))
    return(FALSE)
  }
}

cat("=== WLS Validation: DuckDB Extension vs R (lm with weights) ===\n")
cat("Method: DuckDB CLI + R lm() comparison\n\n")

# Check for DuckDB CLI
if (!file.exists(DUCKDB_CLI)) {
  stop("DuckDB CLI not found at: ", DUCKDB_CLI)
}
cat("✓ DuckDB CLI found\n")

# Check for extension
if (!file.exists(EXTENSION_PATH)) {
  stop("Extension not found at: ", EXTENSION_PATH)
}
cat("✓ Extension found\n\n")

# ============================================================================
# Test 1: Simple WLS with Equal Weights (should match OLS)
# ============================================================================
cat("--- Test 1: WLS with Equal Weights (should match OLS) ---\n")
cat("Testing: All weights = 1.0 should give same results as OLS\n\n")

# Data
x_test1 <- c(1, 2, 3, 4, 5)
y_test1 <- c(2.1, 4.2, 5.9, 8.1, 10.0)
weights_test1 <- c(1, 1, 1, 1, 1)  # Equal weights

# R computation
r_model <- lm(y_test1 ~ x_test1, weights = weights_test1)
r_intercept <- coef(r_model)[1]
r_slope <- coef(r_model)[2]
r_r2 <- summary(r_model)$r.squared

cat("R Results (lm with weights=1):\n")
cat(sprintf("  Intercept: %.10f\n", r_intercept))
cat(sprintf("  Slope:     %.10f\n", r_slope))
cat(sprintf("  R²:        %.10f\n", r_r2))

# DuckDB query
matrix_rows <- paste(sprintf("[%s]", x_test1), collapse = ", ")
query <- sprintf("
LOAD '%s';
SELECT * FROM anofox_statistics_wls(
    [%s]::DOUBLE[],
    [%s]::DOUBLE[][],
    [%s]::DOUBLE[],
    {'intercept': true}
);
", EXTENSION_PATH,
   paste(y_test1, collapse = ", "),
   matrix_rows,
   paste(weights_test1, collapse = ", "))

csv_result <- run_duckdb_query(query)
if (is.null(csv_result)) {
  cat("✗ DuckDB query failed\n\n")
} else {
  duckdb_result <- parse_duckdb_csv(csv_result)

  if (is.null(duckdb_result)) {
    cat("✗ Failed to parse DuckDB output\n\n")
  } else {
    cat("\nDuckDB Results:\n")
    cat(sprintf("  Intercept: %.10f\n", duckdb_result$intercept))
    cat(sprintf("  Slope:     %.10f\n", duckdb_result$coefficients[1]))
    cat(sprintf("  R²:        %.10f\n", duckdb_result$r_squared))

    cat("\nComparison:\n")
    compare_values(duckdb_result$intercept, r_intercept, STRICT_TOL, "Test1", "intercept")
    compare_values(duckdb_result$coefficients[1], r_slope, STRICT_TOL, "Test1", "slope")
    compare_values(duckdb_result$r_squared, r_r2, STRICT_TOL, "Test1", "r_squared")
  }
}

cat("\n")

# ============================================================================
# Test 2: WLS with Varying Weights (emphasize different observations)
# ============================================================================
cat("--- Test 2: WLS with Varying Weights ---\n")
cat("Testing: Higher weights on certain observations\n\n")

# Data with varying weights
x_test2 <- c(1, 2, 3, 4, 5)
y_test2 <- c(2.1, 4.2, 5.9, 8.1, 10.0)
weights_test2 <- c(1, 2, 3, 2, 1)  # More weight on middle observations

# R computation
r_model2 <- lm(y_test2 ~ x_test2, weights = weights_test2)
r_intercept2 <- coef(r_model2)[1]
r_slope2 <- coef(r_model2)[2]
r_r2_2 <- summary(r_model2)$r.squared

cat("R Results (lm with varying weights):\n")
cat(sprintf("  Intercept: %.10f\n", r_intercept2))
cat(sprintf("  Slope:     %.10f\n", r_slope2))
cat(sprintf("  R²:        %.10f\n", r_r2_2))
cat("  Weights:  ", paste(weights_test2, collapse = ", "), "\n")

# DuckDB query
matrix_rows2 <- paste(sprintf("[%s]", x_test2), collapse = ", ")
query2 <- sprintf("
LOAD '%s';
SELECT * FROM anofox_statistics_wls(
    [%s]::DOUBLE[],
    [%s]::DOUBLE[][],
    [%s]::DOUBLE[],
    {'intercept': true}
);
", EXTENSION_PATH,
   paste(y_test2, collapse = ", "),
   matrix_rows2,
   paste(weights_test2, collapse = ", "))

csv_result2 <- run_duckdb_query(query2)
if (is.null(csv_result2)) {
  cat("✗ DuckDB query failed\n\n")
} else {
  duckdb_result2 <- parse_duckdb_csv(csv_result2)

  if (is.null(duckdb_result2)) {
    cat("✗ Failed to parse DuckDB output\n\n")
  } else {
    cat("\nDuckDB Results:\n")
    cat(sprintf("  Intercept: %.10f\n", duckdb_result2$intercept))
    cat(sprintf("  Slope:     %.10f\n", duckdb_result2$coefficients[1]))
    cat(sprintf("  R²:        %.10f\n", duckdb_result2$r_squared))

    cat("\nComparison:\n")
    compare_values(duckdb_result2$intercept, r_intercept2, STRICT_TOL, "Test2", "intercept")
    compare_values(duckdb_result2$coefficients[1], r_slope2, STRICT_TOL, "Test2", "slope")
    compare_values(duckdb_result2$r_squared, r_r2_2, STRICT_TOL, "Test2", "r_squared")
  }
}

cat("\n")

# ============================================================================
# Test 3: WLS with Multiple Predictors
# ============================================================================
cat("--- Test 3: WLS with Multiple Predictors ---\n")
cat("Testing: Multiple regression with weights\n\n")

# Data
x1_test3 <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
x2_test3 <- c(2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
x3_test3 <- c(1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5)
y_test3 <- c(8.6, 17.4, 26.2, 35.0, 43.8, 52.6, 61.4, 70.2, 79.0, 87.8)
weights_test3 <- c(1, 1, 2, 2, 3, 3, 2, 2, 1, 1)  # Varying weights

# R computation
r_model3 <- lm(y_test3 ~ x1_test3 + x2_test3 + x3_test3, weights = weights_test3)
r_coefs3 <- coef(r_model3)
r_r2_3 <- summary(r_model3)$r.squared

cat("R Results (lm with 3 predictors + weights):\n")
cat(sprintf("  Intercept: %.10f\n", r_coefs3[1]))
cat(sprintf("  Slope 1:   %.10f\n", r_coefs3[2]))
cat(sprintf("  Slope 2:   %.10f\n", r_coefs3[3]))
cat(sprintf("  Slope 3:   %.10f\n", r_coefs3[4]))
cat(sprintf("  R²:        %.10f\n", r_r2_3))

# DuckDB query
matrix_rows3 <- paste(sprintf("[%s, %s, %s]", x1_test3, x2_test3, x3_test3), collapse = ", ")
query3 <- sprintf("
LOAD '%s';
SELECT * FROM anofox_statistics_wls(
    [%s]::DOUBLE[],
    [%s]::DOUBLE[][],
    [%s]::DOUBLE[],
    {'intercept': true}
);
", EXTENSION_PATH,
   paste(y_test3, collapse = ", "),
   matrix_rows3,
   paste(weights_test3, collapse = ", "))

csv_result3 <- run_duckdb_query(query3)
if (is.null(csv_result3)) {
  cat("✗ DuckDB query failed\n\n")
} else {
  duckdb_result3 <- parse_duckdb_csv(csv_result3)

  if (is.null(duckdb_result3)) {
    cat("✗ Failed to parse DuckDB output\n\n")
  } else {
    cat("\nDuckDB Results:\n")
    cat(sprintf("  Intercept: %.10f\n", duckdb_result3$intercept))
    cat(sprintf("  Slope 1:   %.10f\n", duckdb_result3$coefficients[1]))
    cat(sprintf("  Slope 2:   %.10f\n", duckdb_result3$coefficients[2]))
    cat(sprintf("  Slope 3:   %.10f\n", duckdb_result3$coefficients[3]))
    cat(sprintf("  R²:        %.10f\n", duckdb_result3$r_squared))

    cat("\nComparison:\n")
    compare_values(duckdb_result3$intercept, r_coefs3[1], STRICT_TOL, "Test3", "intercept")
    compare_values(duckdb_result3$coefficients[1], r_coefs3[2], STRICT_TOL, "Test3", "slope_1")
    compare_values(duckdb_result3$coefficients[2], r_coefs3[3], STRICT_TOL, "Test3", "slope_2")
    compare_values(duckdb_result3$coefficients[3], r_coefs3[4], STRICT_TOL, "Test3", "slope_3")
    compare_values(duckdb_result3$r_squared, r_r2_3, STRICT_TOL, "Test3", "r_squared")
  }
}

cat("\n")

# ============================================================================
# Test 4: WLS with Zero Weight (should exclude observation)
# ============================================================================
cat("--- Test 4: WLS with Zero Weight ---\n")
cat("Testing: Zero weight should exclude observation from fit\n\n")

# Data with one zero weight
x_test4 <- c(1, 2, 3, 4, 5)
y_test4 <- c(2.1, 4.2, 100.0, 8.1, 10.0)  # Outlier at position 3
weights_test4 <- c(1, 1, 0, 1, 1)  # Zero weight on outlier

# R computation
r_model4 <- lm(y_test4 ~ x_test4, weights = weights_test4)
r_intercept4 <- coef(r_model4)[1]
r_slope4 <- coef(r_model4)[2]
r_r2_4 <- summary(r_model4)$r.squared

cat("R Results (lm with zero weight on outlier):\n")
cat(sprintf("  Intercept: %.10f\n", r_intercept4))
cat(sprintf("  Slope:     %.10f\n", r_slope4))
cat(sprintf("  R²:        %.10f\n", r_r2_4))
cat("  Note: Observation 3 (y=100) has weight=0\n")

# DuckDB query
matrix_rows4 <- paste(sprintf("[%s]", x_test4), collapse = ", ")
query4 <- sprintf("
LOAD '%s';
SELECT * FROM anofox_statistics_wls(
    [%s]::DOUBLE[],
    [%s]::DOUBLE[][],
    [%s]::DOUBLE[],
    {'intercept': true}
);
", EXTENSION_PATH,
   paste(y_test4, collapse = ", "),
   matrix_rows4,
   paste(weights_test4, collapse = ", "))

csv_result4 <- run_duckdb_query(query4)
if (is.null(csv_result4)) {
  cat("✗ DuckDB query failed\n\n")
} else {
  duckdb_result4 <- parse_duckdb_csv(csv_result4)

  if (is.null(duckdb_result4)) {
    cat("✗ Failed to parse DuckDB output\n\n")
  } else {
    cat("\nDuckDB Results:\n")
    cat(sprintf("  Intercept: %.10f\n", duckdb_result4$intercept))
    cat(sprintf("  Slope:     %.10f\n", duckdb_result4$coefficients[1]))
    cat(sprintf("  R²:        %.10f\n", duckdb_result4$r_squared))

    cat("\nComparison:\n")
    compare_values(duckdb_result4$intercept, r_intercept4, STRICT_TOL, "Test4", "intercept")
    compare_values(duckdb_result4$coefficients[1], r_slope4, STRICT_TOL, "Test4", "slope")
    compare_values(duckdb_result4$r_squared, r_r2_4, STRICT_TOL, "Test4", "r_squared")
  }
}

cat("\n=== WLS Validation Complete ===\n")
