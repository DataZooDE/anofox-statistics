#!/usr/bin/env Rscript

# Ridge Regression Validation: DuckDB Extension vs R (glmnet)
#
# Validates anofox_statistics_ridge_fit against R's glmnet package
# Uses DuckDB CLI for extension testing

library(glmnet)

# Configuration
STRICT_TOL <- 1e-10  # For coefficients, R²
RELAXED_TOL <- 1e-8  # For p-values

DUCKDB_CLI <- "/tmp/duckdb"
EXTENSION_PATH <- "build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension"

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
    print(TRUE)
    return(TRUE)
  }

  if ((is.null(duckdb_val) || is.na(duckdb_val)) != (is.null(r_val) || is.na(r_val))) {
    cat(sprintf("  ✗ %s - %s: NULL/NA mismatch\n", test_name, metric_name))
    cat(sprintf("    DuckDB: %s\n", ifelse(is.null(duckdb_val) || is.na(duckdb_val), "NULL/NA", as.character(duckdb_val))))
    cat(sprintf("    R:      %s\n", ifelse(is.null(r_val) || is.na(r_val), "NULL/NA", as.character(r_val))))
    print(FALSE)
    return(FALSE)
  }

  # Numeric comparison
  diff <- abs(duckdb_val - r_val)
  if (diff < tol) {
    cat(sprintf("  ✓ %s - %s: %.10e (diff: %.2e)\n", test_name, metric_name, r_val, diff))
    print(TRUE)
    return(TRUE)
  } else {
    cat(sprintf("  ✗ %s - %s: MISMATCH\n", test_name, metric_name))
    cat(sprintf("    DuckDB: %.15f\n", duckdb_val))
    cat(sprintf("    R:      %.15f\n", r_val))
    cat(sprintf("    Diff:   %.2e (tolerance: %.2e)\n", diff, tol))
    print(FALSE)
    return(FALSE)
  }
}

cat("=== Ridge Regression Validation: DuckDB Extension vs R (glmnet) ===\n")
cat("Method: DuckDB CLI + glmnet comparison\n\n")

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

# Results tracking
all_results <- data.frame()

# ============================================================================
# Test 1: Simple Ridge Regression (λ=1.0)
# ============================================================================
cat("--- Test 1: Simple Ridge Regression (λ=1.0) ---\n")
cat("Testing: y ≈ 2*x1 + 1*x2 with ridge penalty\n\n")

# Data (2 INDEPENDENT predictors - glmnet requires at least 2)
# IMPORTANT: Previous data had perfect collinearity (x2 = x1 + 0.5)
# Using independent predictors for valid Ridge regression
x1_test1 <- c(1, 2, 3, 4, 5)
x2_test1 <- c(2, 1, 4, 3, 5)  # Independent of x1
y_test1 <- c(5, 6, 11, 10, 15)  # y ≈ 2*x1 + 1*x2
lambda_test1 <- 1.0

# R computation using glmnet
# Note: glmnet uses penalty = lambda * n, we use just lambda
# So we need to convert: glmnet_lambda = our_lambda / n
X_matrix <- cbind(x1_test1, x2_test1)
glmnet_lambda <- lambda_test1 / length(y_test1)

ridge_model <- glmnet(X_matrix, y_test1, alpha = 0, lambda = glmnet_lambda,
                      intercept = TRUE, standardize = FALSE, thresh = 1e-14)

r_intercept <- as.numeric(coef(ridge_model)[1])
r_slope1 <- as.numeric(coef(ridge_model)[2])
r_slope2 <- as.numeric(coef(ridge_model)[3])

# Compute R² manually
y_pred <- predict(ridge_model, newx = X_matrix, s = glmnet_lambda)
ss_res <- sum((y_test1 - y_pred)^2)
ss_tot <- sum((y_test1 - mean(y_test1))^2)
r_r2 <- 1 - ss_res / ss_tot

cat("R Results (glmnet):\n")
cat(sprintf("  Intercept: %.10f\n", r_intercept))
cat(sprintf("  Slope 1:   %.10f\n", r_slope1))
cat(sprintf("  Slope 2:   %.10f\n", r_slope2))
cat(sprintf("  R²:        %.10f\n", r_r2))
cat(sprintf("  Lambda:    %.10f (glmnet: %.10f)\n", lambda_test1, glmnet_lambda))

# DuckDB query
query <- sprintf("
LOAD '%s';
SELECT * FROM anofox_statistics_ridge_fit(
    [%s]::DOUBLE[],
    [%s]::DOUBLE[],
    [%s]::DOUBLE[],
    %f::DOUBLE,
    true
);
", EXTENSION_PATH,
   paste(y_test1, collapse = ", "),
   paste(x1_test1, collapse = ", "),
   paste(x2_test1, collapse = ", "),
   lambda_test1)

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
    cat(sprintf("  Slope 1:   %.10f\n", duckdb_result$coefficients[1]))
    cat(sprintf("  Slope 2:   %.10f\n", duckdb_result$coefficients[2]))
    cat(sprintf("  R²:        %.10f\n", duckdb_result$r_squared))

    cat("\nComparison:\n")
    compare_values(duckdb_result$intercept, r_intercept, STRICT_TOL, "Test1", "intercept")
    compare_values(duckdb_result$coefficients[1], r_slope1, STRICT_TOL, "Test1", "slope_1")
    compare_values(duckdb_result$coefficients[2], r_slope2, STRICT_TOL, "Test1", "slope_2")
    compare_values(duckdb_result$r_squared, r_r2, STRICT_TOL, "Test1", "r_squared")
  }
}

cat("\n")

# ============================================================================
# Test 2: Ridge with Multiple Predictors (λ=0.5)
# ============================================================================
cat("--- Test 2: Ridge with Multiple Predictors (λ=0.5) ---\n")
cat("Testing: y = -1.5 + 3.5*x1 + 0.5*x2 + 2.0*x3 + noise\n\n")

# Data
x1_test2 <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
x2_test2 <- c(2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
x3_test2 <- c(1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5)
y_test2 <- c(8.6, 17.4, 26.2, 35.0, 43.8, 52.6, 61.4, 70.2, 79.0, 87.8)
lambda_test2 <- 0.5

# R computation
X_matrix2 <- cbind(x1_test2, x2_test2, x3_test2)
glmnet_lambda2 <- lambda_test2 / length(y_test2)

ridge_model2 <- glmnet(X_matrix2, y_test2, alpha = 0, lambda = glmnet_lambda2,
                       intercept = TRUE, standardize = FALSE, thresh = 1e-14)

r_coefs2 <- as.numeric(coef(ridge_model2))
r_intercept2 <- r_coefs2[1]
r_slopes2 <- r_coefs2[-1]

y_pred2 <- predict(ridge_model2, newx = X_matrix2, s = glmnet_lambda2)
ss_res2 <- sum((y_test2 - y_pred2)^2)
ss_tot2 <- sum((y_test2 - mean(y_test2))^2)
r_r2_2 <- 1 - ss_res2 / ss_tot2

cat("R Results (glmnet):\n")
cat(sprintf("  Intercept: %.10f\n", r_intercept2))
cat(sprintf("  Slope 1:   %.10f\n", r_slopes2[1]))
cat(sprintf("  Slope 2:   %.10f\n", r_slopes2[2]))
cat(sprintf("  Slope 3:   %.10f\n", r_slopes2[3]))
cat(sprintf("  R²:        %.10f\n", r_r2_2))

# DuckDB query
query2 <- sprintf("
LOAD '%s';
SELECT * FROM anofox_statistics_ridge_fit(
    [%s]::DOUBLE[],
    [%s]::DOUBLE[],
    [%s]::DOUBLE[],
    [%s]::DOUBLE[],
    %f::DOUBLE,
    true
);
", EXTENSION_PATH,
   paste(y_test2, collapse = ", "),
   paste(x1_test2, collapse = ", "),
   paste(x2_test2, collapse = ", "),
   paste(x3_test2, collapse = ", "),
   lambda_test2)

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
    cat(sprintf("  Slope 1:   %.10f\n", duckdb_result2$coefficients[1]))
    cat(sprintf("  Slope 2:   %.10f\n", duckdb_result2$coefficients[2]))
    cat(sprintf("  Slope 3:   %.10f\n", duckdb_result2$coefficients[3]))
    cat(sprintf("  R²:        %.10f\n", duckdb_result2$r_squared))

    cat("\nComparison:\n")
    compare_values(duckdb_result2$intercept, r_intercept2, STRICT_TOL, "Test2", "intercept")
    compare_values(duckdb_result2$coefficients[1], r_slopes2[1], STRICT_TOL, "Test2", "slope_1")
    compare_values(duckdb_result2$coefficients[2], r_slopes2[2], STRICT_TOL, "Test2", "slope_2")
    compare_values(duckdb_result2$coefficients[3], r_slopes2[3], STRICT_TOL, "Test2", "slope_3")
    compare_values(duckdb_result2$r_squared, r_r2_2, STRICT_TOL, "Test2", "r_squared")
  }
}

cat("\n")

# ============================================================================
# Test 3: Ridge with λ=0 (should match OLS)
# ============================================================================
cat("--- Test 3: Ridge with λ=0 (OLS equivalence) ---\n")
cat("Testing: Ridge with λ=0 should match OLS results\n\n")

# Use Test 1 data with λ=0
lambda_test3 <- 0.0

# R OLS with 2 predictors
ols_model <- lm(y_test1 ~ x1_test1 + x2_test1)
r_intercept3 <- coef(ols_model)[1]
r_slope1_3 <- coef(ols_model)[2]
r_slope2_3 <- coef(ols_model)[3]
r_r2_3 <- summary(ols_model)$r.squared

cat("R Results (OLS via lm):\n")
cat(sprintf("  Intercept: %.10f\n", r_intercept3))
cat(sprintf("  Slope 1:   %.10f\n", r_slope1_3))
cat(sprintf("  Slope 2:   %.10f\n", r_slope2_3))
cat(sprintf("  R²:        %.10f\n", r_r2_3))

# DuckDB Ridge with λ=0
query3 <- sprintf("
LOAD '%s';
SELECT * FROM anofox_statistics_ridge_fit(
    [%s]::DOUBLE[],
    [%s]::DOUBLE[],
    [%s]::DOUBLE[],
    %f::DOUBLE,
    true
);
", EXTENSION_PATH,
   paste(y_test1, collapse = ", "),
   paste(x1_test1, collapse = ", "),
   paste(x2_test1, collapse = ", "),
   lambda_test3)

csv_result3 <- run_duckdb_query(query3)
if (is.null(csv_result3)) {
  cat("✗ DuckDB query failed\n\n")
} else {
  duckdb_result3 <- parse_duckdb_csv(csv_result3)

  if (is.null(duckdb_result3)) {
    cat("✗ Failed to parse DuckDB output\n\n")
  } else {
    cat("\nDuckDB Results (Ridge λ=0):\n")
    cat(sprintf("  Intercept: %.10f\n", duckdb_result3$intercept))
    cat(sprintf("  Slope 1:   %.10f\n", duckdb_result3$coefficients[1]))
    cat(sprintf("  Slope 2:   %.10f\n", duckdb_result3$coefficients[2]))
    cat(sprintf("  R²:        %.10f\n", duckdb_result3$r_squared))

    cat("\nComparison:\n")
    compare_values(duckdb_result3$intercept, r_intercept3, STRICT_TOL, "Test3", "intercept")
    compare_values(duckdb_result3$coefficients[1], r_slope1_3, STRICT_TOL, "Test3", "slope_1")
    compare_values(duckdb_result3$coefficients[2], r_slope2_3, STRICT_TOL, "Test3", "slope_2")
    compare_values(duckdb_result3$r_squared, r_r2_3, STRICT_TOL, "Test3", "r_squared")
  }
}

cat("\n")

# ============================================================================
# Test 4: High Regularization (λ=10.0, coefficients shrink)
# ============================================================================
cat("--- Test 4: High Regularization (λ=10.0) ---\n")
cat("Testing: High lambda should shrink coefficients towards zero\n\n")

lambda_test4 <- 10.0

# R computation (reuse Test 1 data with 2 predictors)
X_matrix4 <- cbind(x1_test1, x2_test1)
glmnet_lambda4 <- lambda_test4 / length(y_test1)
ridge_model4 <- glmnet(X_matrix4, y_test1, alpha = 0, lambda = glmnet_lambda4,
                       intercept = TRUE, standardize = FALSE, thresh = 1e-14)

r_intercept4 <- as.numeric(coef(ridge_model4)[1])
r_slope1_4 <- as.numeric(coef(ridge_model4)[2])
r_slope2_4 <- as.numeric(coef(ridge_model4)[3])

y_pred4 <- predict(ridge_model4, newx = X_matrix4, s = glmnet_lambda4)
ss_res4 <- sum((y_test1 - y_pred4)^2)
ss_tot4 <- sum((y_test1 - mean(y_test1))^2)
r_r2_4 <- 1 - ss_res4 / ss_tot4

cat("R Results (glmnet):\n")
cat(sprintf("  Intercept: %.10f\n", r_intercept4))
cat(sprintf("  Slope 1:   %.10f (shrinkage from OLS: %.1f%%)\n",
            r_slope1_4, 100 * (1 - r_slope1_4 / r_slope1_3)))
cat(sprintf("  Slope 2:   %.10f (shrinkage from OLS: %.1f%%)\n",
            r_slope2_4, 100 * (1 - r_slope2_4 / r_slope2_3)))
cat(sprintf("  R²:        %.10f\n", r_r2_4))

# DuckDB query
query4 <- sprintf("
LOAD '%s';
SELECT * FROM anofox_statistics_ridge_fit(
    [%s]::DOUBLE[],
    [%s]::DOUBLE[],
    [%s]::DOUBLE[],
    %f::DOUBLE,
    true
);
", EXTENSION_PATH,
   paste(y_test1, collapse = ", "),
   paste(x1_test1, collapse = ", "),
   paste(x2_test1, collapse = ", "),
   lambda_test4)

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
    cat(sprintf("  Slope 1:   %.10f\n", duckdb_result4$coefficients[1]))
    cat(sprintf("  Slope 2:   %.10f\n", duckdb_result4$coefficients[2]))
    cat(sprintf("  R²:        %.10f\n", duckdb_result4$r_squared))

    cat("\nComparison:\n")
    compare_values(duckdb_result4$intercept, r_intercept4, STRICT_TOL, "Test4", "intercept")
    compare_values(duckdb_result4$coefficients[1], r_slope1_4, STRICT_TOL, "Test4", "slope_1")
    compare_values(duckdb_result4$coefficients[2], r_slope2_4, STRICT_TOL, "Test4", "slope_2")
    compare_values(duckdb_result4$r_squared, r_r2_4, STRICT_TOL, "Test4", "r_squared")
  }
}

cat("\n=== Ridge Validation Complete ===\n")
