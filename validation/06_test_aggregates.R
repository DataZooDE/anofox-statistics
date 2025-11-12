library(DBI)
library(duckdb)
library(glmnet)  # For Ridge regression

cat("=== Aggregate Function Validation: DuckDB Extension vs R ===\n\n")

# Configuration
EXTENSION_PATH <- "extension/mingw/anofox_statistics.duckdb_extension"
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
con <- dbConnect(
  duckdb::duckdb(config = list(allow_unsigned_extensions = "true"))
)
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
# Test 1: OLS Aggregate with GROUP BY (intercept=TRUE)
# ==============================================================================
cat("\n--- Test 1: OLS Aggregate with GROUP BY (intercept=TRUE) ---\n")

# Create test data: Two groups with different slopes
set.seed(42)
test_data <- data.frame(
  category = rep(c("group_a", "group_b"), each = 5),
  x1 = c(1, 2, 3, 4, 5, 2, 4, 6, 8, 10),
  y = c(3, 5, 7, 9, 11, 10, 20, 30, 40, 50) + rnorm(10, 0, 0.5)
)

# Upload to DuckDB
dbWriteTable(con, "test_data", test_data, overwrite = TRUE)

# Compute R baseline: OLS per group
cat("Computing R baseline (lm per group)...\n")
r_results <- by(test_data, test_data$category, function(df) {
  fit <- lm(y ~ x1, data = df)
  summ <- summary(fit)
  list(
    category = df$category[1],
    intercept = coef(fit)[1],
    coef_x1 = coef(fit)[2],
    r2 = summ$r.squared,
    adj_r2 = summ$adj.r.squared,
    n_obs = nrow(df)
  )
})
r_results <- do.call(rbind, lapply(r_results, as.data.frame))
row.names(r_results) <- NULL

cat("R results:\n")
print(r_results, digits = 15)

# Compute DuckDB aggregate
cat("\nComputing DuckDB aggregate...\n")
duckdb_result <- dbGetQuery(con, "
  SELECT
    category,
    (result).intercept as intercept,
    (result).coefficients[1] as coef_x1,
    (result).r2 as r2,
    (result).adj_r2 as adj_r2,
    (result).n_obs as n_obs
  FROM (
    SELECT
      category,
      anofox_statistics_ols_agg(y, [x1], {'intercept': true}) as result
    FROM test_data
    GROUP BY category
  ) sub
  ORDER BY category
")

cat("DuckDB results:\n")
print(duckdb_result, digits = 15)

# Compare results for each group
for (i in 1:nrow(r_results)) {
  group <- r_results$category[i]
  cat(sprintf("\nValidating group: %s\n", group))

  r_row <- r_results[i, ]
  d_row <- duckdb_result[duckdb_result$category == group, ]

  compare_values(d_row$intercept, r_row$intercept,
                 test_name = paste("OLS_agg", group), value_name = "intercept")
  compare_values(d_row$coef_x1, r_row$coef_x1,
                 test_name = paste("OLS_agg", group), value_name = "coef_x1")
  compare_values(d_row$r2, r_row$r2,
                 test_name = paste("OLS_agg", group), value_name = "r2")
  compare_values(d_row$adj_r2, r_row$adj_r2,
                 test_name = paste("OLS_agg", group), value_name = "adj_r2")
  compare_values(d_row$n_obs, r_row$n_obs,
                 test_name = paste("OLS_agg", group), value_name = "n_obs")
}

# ==============================================================================
# Test 2: OLS Aggregate with GROUP BY (intercept=FALSE)
# ==============================================================================
cat("\n--- Test 2: OLS Aggregate with GROUP BY (intercept=FALSE) ---\n")

# Compute R baseline: OLS per group without intercept
cat("Computing R baseline (lm without intercept)...\n")
r_results_no_int <- by(test_data, test_data$category, function(df) {
  fit <- lm(y ~ x1 - 1, data = df)  # -1 removes intercept
  summ <- summary(fit)
  list(
    category = df$category[1],
    intercept = 0.0,  # Should be zero
    coef_x1 = coef(fit)[1],
    r2 = summ$r.squared,
    n_obs = nrow(df)
  )
})
r_results_no_int <- do.call(rbind, lapply(r_results_no_int, as.data.frame))
row.names(r_results_no_int) <- NULL

cat("R results (no intercept):\n")
print(r_results_no_int, digits = 15)

# Compute DuckDB aggregate without intercept
duckdb_result_no_int <- dbGetQuery(con, "
  SELECT
    category,
    (result).intercept as intercept,
    (result).coefficients[1] as coef_x1,
    (result).r2 as r2,
    (result).n_obs as n_obs
  FROM (
    SELECT
      category,
      anofox_statistics_ols_agg(y, [x1], {'intercept': false}) as result
    FROM test_data
    GROUP BY category
  ) sub
  ORDER BY category
")

cat("DuckDB results (no intercept):\n")
print(duckdb_result_no_int, digits = 15)

# Compare results
for (i in 1:nrow(r_results_no_int)) {
  group <- r_results_no_int$category[i]
  cat(sprintf("\nValidating group (no intercept): %s\n", group))

  r_row <- r_results_no_int[i, ]
  d_row <- duckdb_result_no_int[duckdb_result_no_int$category == group, ]

  compare_values(d_row$intercept, r_row$intercept,
                 test_name = paste("OLS_agg_no_int", group), value_name = "intercept")
  compare_values(d_row$coef_x1, r_row$coef_x1,
                 test_name = paste("OLS_agg_no_int", group), value_name = "coef_x1")
  compare_values(d_row$r2, r_row$r2,
                 test_name = paste("OLS_agg_no_int", group), value_name = "r2")
}


# ==============================================================================
# Test 3: WLS Aggregate with GROUP BY (intercept=TRUE)
# ==============================================================================
cat("\n--- Test 3: WLS Aggregate with GROUP BY (intercept=TRUE) ---\n")

# Add weights to test data
test_data$weights <- runif(nrow(test_data), 0.5, 2.0)
dbWriteTable(con, "test_data_weighted", test_data, overwrite = TRUE)

# Compute R baseline: WLS per group
cat("Computing R baseline (lm with weights)...\n")
r_wls_results <- by(test_data, test_data$category, function(df) {
  fit <- lm(y ~ x1, data = df, weights = df$weights)
  summ <- summary(fit)
  list(
    category = df$category[1],
    intercept = coef(fit)[1],
    coef_x1 = coef(fit)[2],
    r2 = summ$r.squared,
    adj_r2 = summ$adj.r.squared,
    n_obs = nrow(df)
  )
})
r_wls_results <- do.call(rbind, lapply(r_wls_results, as.data.frame))
row.names(r_wls_results) <- NULL

cat("R WLS results:\n")
print(r_wls_results, digits = 15)

# Compute DuckDB WLS aggregate
duckdb_wls_result <- dbGetQuery(con, "
  SELECT
    category,
    (result).intercept as intercept,
    (result).coefficients[1] as coef_x1,
    (result).r2 as r2,
    (result).adj_r2 as adj_r2,
    (result).n_obs as n_obs
  FROM (
    SELECT
      category,
      anofox_statistics_wls_agg(y, [x1], weights, {'intercept': true}) as result
    FROM test_data_weighted
    GROUP BY category
  ) sub
  ORDER BY category
")

cat("DuckDB WLS results:\n")
print(duckdb_wls_result, digits = 15)

# Compare results
for (i in 1:nrow(r_wls_results)) {
  group <- r_wls_results$category[i]
  cat(sprintf("\nValidating WLS group: %s\n", group))

  r_row <- r_wls_results[i, ]
  d_row <- duckdb_wls_result[duckdb_wls_result$category == group, ]

  compare_values(d_row$intercept, r_row$intercept,
                 test_name = paste("WLS_agg", group), value_name = "intercept")
  compare_values(d_row$coef_x1, r_row$coef_x1,
                 test_name = paste("WLS_agg", group), value_name = "coef_x1")
  compare_values(d_row$r2, r_row$r2,
                 test_name = paste("WLS_agg", group), value_name = "r2")
}

# ==============================================================================
# Test 4: Ridge Aggregate with GROUP BY (intercept=TRUE)
# ==============================================================================
cat("\n--- Test 4: Ridge Aggregate with GROUP BY (intercept=TRUE) ---\n")

# Compute R baseline: Ridge per group using glmnet
lambda_param <- 1.0
cat(sprintf("Computing R baseline (glmnet with lambda=%.1f)...\n", lambda_param))

r_ridge_results <- by(test_data, test_data$category, function(df) {
  # Center y and X for intercept estimation
  y_mean <- mean(df$y)
  x_mean <- mean(df$x1)
  y_centered <- df$y # - y_mean
  x_centered <- df$x1 # - x_mean

  # glmnet expects matrix input
  X_matrix <- as.matrix(x_centered)
  X_matrix = cbind(1, X_matrix)

  # glmnet uses lambda differently: lambda_glmnet = lambda_ours / n
  n <- nrow(df)
  lambda_glmnet <- lambda_param / n

  # Fit ridge (alpha=0 for L2 penalty only)
  fit <- glmnet(X_matrix, y_centered, alpha = 0, lambda = lambda_glmnet,
                intercept = T)  # Already centered

  # Extract coefficient
  coef_ridge <- as.numeric(coef(fit, s = lambda_glmnet))# [-1]  # Remove intercept row
  intercept_ridge <- coef_ridge[1] #  y_mean - coef_ridge * x_mean
  coef_ridge <- coef_ridge[3]

  # Compute R² on original scale
  y_pred <- intercept_ridge + coef_ridge * df$x1
  ss_res <- sum((df$y - y_pred)^2)
  ss_tot <- sum((df$y - mean(df$y))^2)
  r2_ridge <- 1 - ss_res / ss_tot

  list(
    category = df$category[1],
    intercept = intercept_ridge,
    coef_x1 = coef_ridge,
    r2 = r2_ridge,
    lambda = lambda_param,
    n_obs = n
  )
})

r_ridge_results <- do.call(rbind, lapply(r_ridge_results, as.data.frame))
row.names(r_ridge_results) <- NULL

cat("R Ridge results:\n")
print(r_ridge_results, digits = 15)

# Compute DuckDB Ridge aggregate
duckdb_ridge_result <- dbGetQuery(con, sprintf("
  SELECT
    category,
    (result).intercept as intercept,
    (result).coefficients[1] as coef_x1,
    (result).r2 as r2,
    (result).lambda as lambda,
    (result).n_obs as n_obs
  FROM (
    SELECT
      category,
      anofox_statistics_ridge_agg(y, [x1], {'intercept': true, 'lambda': %.1f}) as result
    FROM test_data
    GROUP BY category
  ) sub
  ORDER BY category
", lambda_param))

cat("DuckDB Ridge results:\n")
print(duckdb_ridge_result, digits = 15)

# Compare results
for (i in 1:nrow(r_ridge_results)) {
  group <- r_ridge_results$category[i]
  cat(sprintf("\nValidating Ridge group: %s\n", group))

  r_row <- r_ridge_results[i, ]
  d_row <- duckdb_ridge_result[duckdb_ridge_result$category == group, ]

  compare_values(d_row$intercept, r_row$intercept,
                 test_name = paste("Ridge_agg", group), value_name = "intercept")
  compare_values(d_row$coef_x1, r_row$coef_x1,
                 test_name = paste("Ridge_agg", group), value_name = "coef_x1")
  compare_values(d_row$r2, r_row$r2,
                 test_name = paste("Ridge_agg", group), value_name = "r2")
  compare_values(d_row$lambda, r_row$lambda,
                 test_name = paste("Ridge_agg", group), value_name = "lambda")
}


# ==============================================================================
# Test 5: RLS Aggregate with GROUP BY (intercept=TRUE)
# ==============================================================================
cat("\n--- Test 5: RLS Aggregate with GROUP BY (intercept=TRUE) ---\n")

# Custom RLS implementation for R
rls_fit <- function(y, x, lambda = 1.0, intercept = TRUE) {
  n <- length(y)

  if (intercept) {
    # Center data
    y_mean <- mean(y)
    x_mean <- mean(x)
    y_c <- y - y_mean
    x_c <- x - x_mean

    # Initialize
    P <- diag(2) * 1000  # Large initial covariance (2x2 for intercept + slope)
    theta <- c(0, 0)  # [intercept_offset, slope]

    # Sequential updates
    for (i in 1:n) {
      phi <- c(1, x_c[i])  # Feature vector [1, x]

      # RLS update equations
      K <- (P %*% phi) / (lambda + as.numeric(t(phi) %*% P %*% phi))
      theta <- theta + K * (y_c[i] - as.numeric(t(phi) %*% theta))
      P <- (P - K %*% t(phi) %*% P) / lambda
    }

    # Compute final intercept on original scale
    slope <- theta[2]
    intercept_val <- y_mean - slope * x_mean

    # Compute R²
    y_pred <- intercept_val + slope * x
    ss_res <- sum((y - y_pred)^2)
    ss_tot <- sum((y - mean(y))^2)
    r2 <- 1 - ss_res / ss_tot

    list(intercept = intercept_val, coef = slope, r2 = r2, n_obs = n)
  } else {
    # Without intercept (through origin)
    P <- 1000  # Scalar for single parameter
    theta <- 0

    for (i in 1:n) {
      phi <- x[i]
      K <- P * phi / (lambda + phi^2 * P)
      theta <- theta + K * (y[i] - phi * theta)
      P <- (P - K * phi * P) / lambda
    }

    # Compute R² from origin
    y_pred <- theta * x
    ss_res <- sum((y - y_pred)^2)
    ss_tot <- sum(y^2)
    r2 <- 1 - ss_res / ss_tot

    list(intercept = 0.0, coef = theta, r2 = r2, n_obs = n)
  }
}

# Compute R baseline: RLS per group
forgetting_factor <- 1.0
cat(sprintf("Computing R baseline (RLS with lambda=%.2f)...\n", forgetting_factor))

r_rls_results <- by(test_data, test_data$category, function(df) {
  fit <- rls_fit(df$y, df$x1, lambda = forgetting_factor, intercept = TRUE)
  data.frame(
    category = df$category[1],
    intercept = fit$intercept,
    coef_x1 = fit$coef,
    r2 = fit$r2,
    forgetting_factor = forgetting_factor,
    n_obs = fit$n_obs
  )
})
r_rls_results <- do.call(rbind, r_rls_results)
row.names(r_rls_results) <- NULL

cat("R RLS results:\n")
print(r_rls_results, digits = 15)

# Compute DuckDB RLS aggregate
duckdb_rls_result <- dbGetQuery(con, sprintf("
  SELECT
    category,
    (result).intercept as intercept,
    (result).coefficients[1] as coef_x1,
    (result).r2 as r2,
    (result).forgetting_factor as forgetting_factor,
    (result).n_obs as n_obs
  FROM (
    SELECT
      category,
      anofox_statistics_rls_agg(y, [x1], {'intercept': true, 'forgetting_factor': %.2f}) as result
    FROM test_data
    GROUP BY category
  ) sub
  ORDER BY category
", forgetting_factor))

cat("DuckDB RLS results:\n")
print(duckdb_rls_result, digits = 15)

# Compare results
for (i in 1:nrow(r_rls_results)) {
  group <- r_rls_results$category[i]
  cat(sprintf("\nValidating RLS group: %s\n", group))

  r_row <- r_rls_results[i, ]
  d_row <- duckdb_rls_result[duckdb_rls_result$category == group, ]

  compare_values(d_row$intercept, r_row$intercept,
                 test_name = paste("RLS_agg", group), value_name = "intercept")
  compare_values(d_row$coef_x1, r_row$coef_x1,
                 test_name = paste("RLS_agg", group), value_name = "coef_x1")
  compare_values(d_row$r2, r_row$r2,
                 test_name = paste("RLS_agg", group), value_name = "r2")
}

# ==============================================================================
# Final Summary
# ==============================================================================
cat("\n\n=== VALIDATION SUMMARY ===\n")

# Convert results to data frame
results_df <- do.call(rbind, lapply(test_results, as.data.frame))

# Count passes and fails
n_pass <- sum(results_df$status == "PASS")
n_fail <- sum(results_df$status == "FAIL")
n_total <- nrow(results_df)

cat(sprintf("Total tests: %d\n", n_total))
cat(sprintf("Passed: %d (%.1f%%)\n", n_pass, 100 * n_pass / n_total))
cat(sprintf("Failed: %d (%.1f%%)\n", n_fail, 100 * n_fail / n_total))

if (n_fail > 0) {
  cat("\nFailed tests:\n")
  failed <- results_df[results_df$status == "FAIL", ]
  print(failed[, c("test", "name", "value", "duckdb", "r", "diff")])
}

# Save results to CSV
output_dir <- "validation/results"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}
output_file <- file.path(output_dir, "aggregate_validation_results.csv")
write.csv(results_df, output_file, row.names = FALSE)
cat(sprintf("\n✓ Results saved to: %s\n", output_file))

# Cleanup
dbDisconnect(con, shutdown = TRUE)

# Exit with error code if tests failed
if (n_fail > 0) {
  stop(sprintf("Validation failed: %d/%d tests failed", n_fail, n_total))
}

cat("\n✓ All aggregate validation tests passed!\n")