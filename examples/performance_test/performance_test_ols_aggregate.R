#!/usr/bin/env Rscript
# ============================================================================
# Performance Test: OLS Aggregate Functions in R
# ============================================================================
# This script loads the same dataset used by SQL tests and performs
# equivalent OLS aggregate operations using R's lm() function with GROUP BY.
#
# Equivalent to: anofox_statistics_ols_fit_agg aggregate function
#
# Prerequisites:
# 1. Run generate_test_data.sql first to create the parquet file
# 2. Install required R packages: arrow, dplyr, broom
#
# Usage:
# Rscript examples/performance_test_ols_aggregate.R
# ============================================================================

library(arrow)
library(dplyr)
library(broom)

cat("============================================================================\n")
cat("OLS AGGREGATE PERFORMANCE TEST (R)\n")
cat("============================================================================\n\n")

# ============================================================================
# STEP 1: Load Performance Data from Parquet File
# ============================================================================

cat("Loading performance data from parquet file...\n")
performance_data <- read_parquet("examples/performance_test/data/performance_data_aggregate.parquet")

cat(sprintf("  - Total rows: %d\n", nrow(performance_data)))
cat(sprintf("  - Number of groups: %d\n", length(unique(performance_data$group_id))))
cat(sprintf("  - Observations per group: %d\n", nrow(performance_data) / length(unique(performance_data$group_id))))
cat("\nDataset loaded successfully!\n\n")

# ============================================================================
# STEP 2: Helper Functions for OLS Statistics
# ============================================================================

# Function to extract comprehensive OLS statistics
# Equivalent to: anofox_statistics_ols_fit_agg with full_output=true
extract_ols_stats <- function(data, full_output = FALSE) {
  model <- lm(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8, data = data)

  # Basic statistics
  model_summary <- summary(model)
  coefficients <- coef(model)[-1]  # Exclude intercept
  intercept <- coef(model)[1]

  n_obs <- nrow(data)
  df_model <- model$rank - 1  # Exclude intercept
  df_residual <- model$df.residual

  r2 <- model_summary$r.squared
  adj_r2 <- model_summary$adj.r.squared

  if (full_output) {
    # Full statistical output
    coef_summary <- coef(model_summary)

    result <- list(
      intercept = intercept,
      coefficients = list(coefficients),
      coefficient_std_errors = list(coef_summary[-1, "Std. Error"]),
      coefficient_t_statistics = list(coef_summary[-1, "t value"]),
      coefficient_p_values = list(coef_summary[-1, "Pr(>|t|)"]),
      r2 = r2,
      adj_r2 = adj_r2,
      f_statistic = model_summary$fstatistic[1],
      f_statistic_p_value = pf(
        model_summary$fstatistic[1],
        model_summary$fstatistic[2],
        model_summary$fstatistic[3],
        lower.tail = FALSE
      ),
      aic = AIC(model),
      bic = BIC(model),
      n_obs = n_obs,
      df_model = df_model,
      df_residual = df_residual
    )
  } else {
    # Basic output
    result <- list(
      intercept = intercept,
      coefficients = list(coefficients),
      r2 = r2,
      adj_r2 = adj_r2,
      n_obs = n_obs,
      df_model = df_model,
      df_residual = df_residual
    )
  }

  return(as.data.frame(result))
}

# ============================================================================
# STEP 3: Performance Test - GROUP BY with All Groups
# ============================================================================

cat("============================================================================\n")
cat("PERFORMANCE TEST 1: GROUP BY Aggregation (All Groups)\n")
cat("============================================================================\n")

start_time <- Sys.time()

group_models <- performance_data %>%
  group_by(group_id) %>%
  group_modify(~ extract_ols_stats(.x, full_output = FALSE)) %>%
  ungroup()

end_time <- Sys.time()
elapsed <- difftime(end_time, start_time, units = "secs")
cat(sprintf("\nElapsed time: %.3f seconds\n\n", elapsed))

# Display sample results
cat("Sample results (first 5 groups):\n")
print(group_models %>%
  filter(group_id <= 5) %>%
  select(group_id, intercept, r2, adj_r2, n_obs))
cat("\n")

# ============================================================================
# STEP 4: Performance Test - GROUP BY with Full Statistical Output
# ============================================================================

cat("============================================================================\n")
cat("PERFORMANCE TEST 2: GROUP BY with Full Statistical Output\n")
cat("============================================================================\n")

start_time <- Sys.time()

group_models_full <- performance_data %>%
  group_by(group_id) %>%
  group_modify(~ extract_ols_stats(.x, full_output = TRUE)) %>%
  ungroup()

end_time <- Sys.time()
elapsed <- difftime(end_time, start_time, units = "secs")
cat(sprintf("\nElapsed time: %.3f seconds\n\n", elapsed))

# Display comprehensive statistics for first group
cat("Sample comprehensive output (group 1):\n")
group_1_full <- group_models_full %>% filter(group_id == 1)
cat(sprintf("Intercept: %.4f\n", group_1_full$intercept))
cat(sprintf("Coefficients: %s\n", paste(sprintf("%.4f", unlist(group_1_full$coefficients)), collapse = ", ")))
cat(sprintf("R-squared: %.4f\n", group_1_full$r2))
cat(sprintf("Adjusted R-squared: %.4f\n", group_1_full$adj_r2))
cat(sprintf("F-statistic: %.4f\n", group_1_full$f_statistic))
cat(sprintf("AIC: %.4f\n", group_1_full$aic))
cat(sprintf("BIC: %.4f\n\n", group_1_full$bic))

# ============================================================================
# STEP 5: Performance Test - Subset of Groups
# ============================================================================

cat("============================================================================\n")
cat("PERFORMANCE TEST 3: GROUP BY on Subset (100 groups)\n")
cat("============================================================================\n")

start_time <- Sys.time()

subset_models <- performance_data %>%
  filter(group_id <= 100) %>%
  group_by(group_id) %>%
  group_modify(~ extract_ols_stats(.x, full_output = FALSE)) %>%
  ungroup()

end_time <- Sys.time()
elapsed <- difftime(end_time, start_time, units = "secs")
cat(sprintf("\nElapsed time: %.3f seconds\n", elapsed))
cat(sprintf("Fitted models for 100 groups\n\n"))

# ============================================================================
# STEP 6: Validation - Compare with True Coefficients
# ============================================================================

cat("============================================================================\n")
cat("VALIDATION: Estimated vs True Coefficients (Group 1)\n")
cat("============================================================================\n")

# Extract true coefficients for group 1
true_coefs <- performance_data %>%
  filter(group_id == 1) %>%
  select(beta_0, beta_1, beta_2, beta_3, beta_4, beta_5, beta_6, beta_7, beta_8) %>%
  slice(1)

# Extract estimated coefficients for group 1
est_coefs <- group_models %>% filter(group_id == 1)

# Create comparison table
comparison <- data.frame(
  parameter = c("Intercept", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"),
  true_value = c(
    true_coefs$beta_0,
    true_coefs$beta_1,
    true_coefs$beta_2,
    true_coefs$beta_3,
    true_coefs$beta_4,
    true_coefs$beta_5,
    true_coefs$beta_6,
    true_coefs$beta_7,
    true_coefs$beta_8
  ),
  estimated_value = c(
    est_coefs$intercept,
    unlist(est_coefs$coefficients)
  )
)
comparison$error <- comparison$estimated_value - comparison$true_value

print(comparison)

cat("\nNote: Small errors are expected due to random noise in the data\n\n")

# ============================================================================
# STEP 7: Save Results to Parquet Files
# ============================================================================

cat("============================================================================\n")
cat("Saving results to parquet files...\n")
cat("============================================================================\n")

# Create results directory if it doesn't exist
dir.create("examples/results", showWarnings = FALSE, recursive = TRUE)

# Prepare basic model results for saving
group_models_save <- group_models %>%
  mutate(
    # Convert list columns to string representation for parquet compatibility
    coefficients_str = sapply(coefficients, function(x) paste(x, collapse = ",")),
    # Keep individual coefficient columns for easier comparison
    coef_x1 = sapply(coefficients, function(x) x[1]),
    coef_x2 = sapply(coefficients, function(x) x[2]),
    coef_x3 = sapply(coefficients, function(x) x[3]),
    coef_x4 = sapply(coefficients, function(x) x[4]),
    coef_x5 = sapply(coefficients, function(x) x[5]),
    coef_x6 = sapply(coefficients, function(x) x[6]),
    coef_x7 = sapply(coefficients, function(x) x[7]),
    coef_x8 = sapply(coefficients, function(x) x[8])
  ) %>%
  select(-coefficients)  # Remove list column

# Save basic model results
write_parquet(
  group_models_save,
  "examples/performance_test/results/r_group_models.parquet"
)

# Prepare full model results for saving
group_models_full_save <- group_models_full %>%
  mutate(
    # Convert list columns for parquet
    coefficients_str = sapply(coefficients, function(x) paste(x, collapse = ",")),
    coef_x1 = sapply(coefficients, function(x) x[1]),
    coef_x2 = sapply(coefficients, function(x) x[2]),
    coef_x3 = sapply(coefficients, function(x) x[3]),
    coef_x4 = sapply(coefficients, function(x) x[4]),
    coef_x5 = sapply(coefficients, function(x) x[5]),
    coef_x6 = sapply(coefficients, function(x) x[6]),
    coef_x7 = sapply(coefficients, function(x) x[7]),
    coef_x8 = sapply(coefficients, function(x) x[8]),
    std_err_x1 = sapply(coefficient_std_errors, function(x) x[1]),
    std_err_x2 = sapply(coefficient_std_errors, function(x) x[2]),
    std_err_x3 = sapply(coefficient_std_errors, function(x) x[3]),
    std_err_x4 = sapply(coefficient_std_errors, function(x) x[4]),
    std_err_x5 = sapply(coefficient_std_errors, function(x) x[5]),
    std_err_x6 = sapply(coefficient_std_errors, function(x) x[6]),
    std_err_x7 = sapply(coefficient_std_errors, function(x) x[7]),
    std_err_x8 = sapply(coefficient_std_errors, function(x) x[8]),
    t_stat_x1 = sapply(coefficient_t_statistics, function(x) x[1]),
    t_stat_x2 = sapply(coefficient_t_statistics, function(x) x[2]),
    t_stat_x3 = sapply(coefficient_t_statistics, function(x) x[3]),
    t_stat_x4 = sapply(coefficient_t_statistics, function(x) x[4]),
    t_stat_x5 = sapply(coefficient_t_statistics, function(x) x[5]),
    t_stat_x6 = sapply(coefficient_t_statistics, function(x) x[6]),
    t_stat_x7 = sapply(coefficient_t_statistics, function(x) x[7]),
    t_stat_x8 = sapply(coefficient_t_statistics, function(x) x[8]),
    p_value_x1 = sapply(coefficient_p_values, function(x) x[1]),
    p_value_x2 = sapply(coefficient_p_values, function(x) x[2]),
    p_value_x3 = sapply(coefficient_p_values, function(x) x[3]),
    p_value_x4 = sapply(coefficient_p_values, function(x) x[4]),
    p_value_x5 = sapply(coefficient_p_values, function(x) x[5]),
    p_value_x6 = sapply(coefficient_p_values, function(x) x[6]),
    p_value_x7 = sapply(coefficient_p_values, function(x) x[7]),
    p_value_x8 = sapply(coefficient_p_values, function(x) x[8])
  ) %>%
  select(-coefficients, -coefficient_std_errors, -coefficient_t_statistics, -coefficient_p_values)

# Save full model results
write_parquet(
  group_models_full_save,
  "examples/performance_test/results/r_group_models_full.parquet"
)

cat("Results saved to examples/performance_test/results/\n\n")

# ============================================================================
# SUMMARY
# ============================================================================

cat("============================================================================\n")
cat("PERFORMANCE TEST SUMMARY (R)\n")
cat("============================================================================\n")
cat("Tests completed:\n")
cat("  1. GROUP BY aggregation on all groups (basic output)\n")
cat("  2. GROUP BY aggregation on all groups (full output)\n")
cat("  3. GROUP BY aggregation on subset (100 groups)\n")
cat("\n")
cat("Results saved to:\n")
cat("  - examples/performance_test/results/r_group_models.parquet\n")
cat("  - examples/performance_test/results/r_group_models_full.parquet\n")
cat("============================================================================\n")
