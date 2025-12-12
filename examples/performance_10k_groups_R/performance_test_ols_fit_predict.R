#!/usr/bin/env Rscript
# ============================================================================
# Performance Test: OLS Fit-Predict in R
# ============================================================================
# This script loads the same dataset used by SQL tests and performs
# equivalent OLS fit-predict operations using R's lm() function.
#
# Equivalent to: anofox_statistics_ols_fit_predict window functions
#
# Prerequisites:
# 1. Run generate_test_data.sql first to create the parquet file
# 2. Install required R packages: arrow, dplyr
#
# Usage:
# Rscript examples/performance_10k_groups_R_ols_fit_predict.R
# ============================================================================

library(arrow)
library(dplyr)

cat("============================================================================\n")
cat("OLS FIT-PREDICT PERFORMANCE TEST (R)\n")
cat("============================================================================\n\n")

# ============================================================================
# STEP 1: Load Performance Data from Parquet File
# ============================================================================

cat("Loading performance data from parquet file...\n")
performance_data <- read_parquet("examples/performance_10k_groups_R/data/performance_data_fit_predict.parquet")

cat(sprintf("  - Total rows: %d\n", nrow(performance_data)))
cat(sprintf("  - Number of groups: %d\n", length(unique(performance_data$group_id))))
cat(sprintf("  - Observations per group: %d\n", nrow(performance_data) / length(unique(performance_data$group_id))))
cat(sprintf("  - NULL values in y: %d (%.2f%%)\n",
            sum(is.na(performance_data$y)),
            100 * sum(is.na(performance_data$y)) / nrow(performance_data)))
cat("\nDataset loaded successfully!\n\n")

# ============================================================================
# STEP 2: Helper Function for OLS Fit-Predict
# ============================================================================

# Function to perform OLS fit-predict with expanding window
# Equivalent to: anofox_statistics_ols_fit_predict with 'expanding' mode
ols_fit_predict_expanding <- function(data, confidence_level = 0.95) {
  n <- nrow(data)
  results <- data.frame(
    group_id = data$group_id,
    obs_id = data$obs_id,
    y = data$y,
    yhat = rep(NA_real_, n),
    yhat_lower = rep(NA_real_, n),
    yhat_upper = rep(NA_real_, n)
  )

  for (i in 1:n) {
    # Use all data up to current row (expanding window)
    train_data <- data[1:i, ]
    train_data <- train_data[!is.na(train_data$y), ]  # Remove NULLs for training

    if (nrow(train_data) >= 10) {  # Need minimum observations
      # Fit model
      model <- lm(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8, data = train_data)

      # Predict current observation
      current_row <- data[i, ]
      pred <- predict(model, newdata = current_row, interval = "prediction", level = confidence_level)

      results$yhat[i] <- pred[1, "fit"]
      results$yhat_lower[i] <- pred[1, "lwr"]
      results$yhat_upper[i] <- pred[1, "upr"]
    }
  }

  return(results)
}

# Function to perform OLS fit-predict with fixed window
# Equivalent to: anofox_statistics_ols_fit_predict with 'fixed' mode
ols_fit_predict_fixed <- function(data, confidence_level = 0.95) {
  n <- nrow(data)
  results <- data.frame(
    group_id = data$group_id,
    obs_id = data$obs_id,
    y = data$y,
    yhat = rep(NA_real_, n),
    yhat_lower = rep(NA_real_, n),
    yhat_upper = rep(NA_real_, n)
  )

  # Fit once on all non-NULL training data
  train_data <- data[!is.na(data$y), ]

  if (nrow(train_data) >= 10) {
    model <- lm(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8, data = train_data)

    # Predict all observations
    pred <- predict(model, newdata = data, interval = "prediction", level = confidence_level)

    results$yhat <- pred[, "fit"]
    results$yhat_lower <- pred[, "lwr"]
    results$yhat_upper <- pred[, "upr"]
  }

  return(results)
}

# ============================================================================
# STEP 3: Performance Test - Expanding Window (Single Group)
# ============================================================================

cat("============================================================================\n")
cat("PERFORMANCE TEST 1: Expanding Window (Single Group)\n")
cat("============================================================================\n")
cat("Mode: expanding - model refits on each new observation\n")

start_time <- Sys.time()

group_1_data <- performance_data %>%
  filter(group_id == 1) %>%
  arrange(obs_id)

predictions_expanding_single <- ols_fit_predict_expanding(group_1_data)

end_time <- Sys.time()
elapsed <- difftime(end_time, start_time, units = "secs")
cat(sprintf("\nElapsed time: %.3f seconds\n\n", elapsed))

# Show sample predictions
cat("Sample predictions (first 10 rows):\n")
print(predictions_expanding_single %>%
  select(obs_id, y, yhat, yhat_lower, yhat_upper) %>%
  mutate(type = ifelse(is.na(y), "PREDICTED", "FITTED")) %>%
  head(10))
cat("\n")

# ============================================================================
# STEP 4: Performance Test - Fixed Window (Single Group)
# ============================================================================

cat("============================================================================\n")
cat("PERFORMANCE TEST 2: Fixed Window (Single Group)\n")
cat("============================================================================\n")
cat("Mode: fixed - fits once on training data, predicts all\n")

start_time <- Sys.time()

predictions_fixed_single <- ols_fit_predict_fixed(group_1_data)

end_time <- Sys.time()
elapsed <- difftime(end_time, start_time, units = "secs")
cat(sprintf("\nElapsed time: %.3f seconds\n\n", elapsed))

# Show sample predictions (NULL values only)
cat("Sample predictions (rows with NULL y):\n")
print(predictions_fixed_single %>%
  select(obs_id, y, yhat, yhat_lower, yhat_upper) %>%
  filter(is.na(y)) %>%
  head(10))
cat("\n")

# ============================================================================
# STEP 5: Performance Test - Expanding Window (Multiple Groups)
# ============================================================================

cat("============================================================================\n")
cat("PERFORMANCE TEST 3: Expanding Window (100 Groups)\n")
cat("============================================================================\n")

start_time <- Sys.time()

predictions_expanding_multi <- performance_data %>%
  filter(group_id <= 100) %>%
  arrange(group_id, obs_id) %>%
  group_by(group_id) %>%
  group_modify(~ ols_fit_predict_expanding(.x)) %>%
  ungroup()

end_time <- Sys.time()
elapsed <- difftime(end_time, start_time, units = "secs")
cat(sprintf("\nElapsed time: %.3f seconds\n\n", elapsed))

# Report results
cat(sprintf("Total predictions: %d\n", nrow(predictions_expanding_multi)))
cat(sprintf("Number of groups: %d\n", length(unique(predictions_expanding_multi$group_id))))
cat(sprintf("NULL predictions: %d\n\n", sum(is.na(predictions_expanding_multi$y))))

# ============================================================================
# STEP 6: Performance Test - Fixed Window (Multiple Groups)
# ============================================================================

cat("============================================================================\n")
cat("PERFORMANCE TEST 4: Fixed Window (100 Groups)\n")
cat("============================================================================\n")

start_time <- Sys.time()

predictions_fixed_multi <- performance_data %>%
  filter(group_id <= 100) %>%
  arrange(group_id, obs_id) %>%
  group_by(group_id) %>%
  group_modify(~ ols_fit_predict_fixed(.x)) %>%
  ungroup()

end_time <- Sys.time()
elapsed <- difftime(end_time, start_time, units = "secs")
cat(sprintf("\nElapsed time: %.3f seconds\n\n", elapsed))

# Report results
cat(sprintf("Total predictions: %d\n", nrow(predictions_fixed_multi)))
cat(sprintf("Number of groups: %d\n", length(unique(predictions_fixed_multi$group_id))))
cat(sprintf("NULL predictions: %d\n\n", sum(is.na(predictions_fixed_multi$y))))

# ============================================================================
# STEP 7: Validation - Prediction Accuracy for NULL Values
# ============================================================================

cat("============================================================================\n")
cat("VALIDATION: Prediction Accuracy (Group 1, NULL values only)\n")
cat("============================================================================\n")

validation_data <- predictions_expanding_single %>%
  inner_join(
    performance_data %>% filter(group_id == 1) %>% select(obs_id, y_true),
    by = "obs_id"
  ) %>%
  filter(is.na(y)) %>%
  mutate(
    abs_error = abs(y_true - yhat),
    in_interval = (y_true >= yhat_lower) & (y_true <= yhat_upper)
  )

if (nrow(validation_data) > 0) {
  cat(sprintf("Number of predictions: %d\n", nrow(validation_data)))
  cat(sprintf("Mean absolute error: %.4f\n", mean(validation_data$abs_error, na.rm = TRUE)))
  cat(sprintf("Std absolute error: %.4f\n", sd(validation_data$abs_error, na.rm = TRUE)))
  cat(sprintf("Min absolute error: %.4f\n", min(validation_data$abs_error, na.rm = TRUE)))
  cat(sprintf("Max absolute error: %.4f\n", max(validation_data$abs_error, na.rm = TRUE)))
  cat(sprintf("Coverage percent: %.2f%%\n", 100 * mean(validation_data$in_interval, na.rm = TRUE)))
  cat("\nNote: coverage_percent should be close to 95% for 95% prediction intervals\n\n")
}

# ============================================================================
# STEP 8: Validation - Compare Fixed vs Expanding Window
# ============================================================================

cat("============================================================================\n")
cat("VALIDATION: Fixed vs Expanding Window Predictions (Group 1)\n")
cat("============================================================================\n")

comparison <- predictions_fixed_single %>%
  inner_join(
    predictions_expanding_single %>% select(obs_id, yhat_expanding = yhat),
    by = "obs_id"
  ) %>%
  mutate(prediction_diff = abs(yhat - yhat_expanding)) %>%
  filter(obs_id <= 20) %>%
  select(obs_id, y, fixed_prediction = yhat, expanding_prediction = yhat_expanding, prediction_diff)

print(comparison)

cat("\nNote: Fixed mode predictions are often more stable (uses all training data)\n")
cat("      Expanding mode adapts as new data arrives (updates coefficients)\n\n")

# ============================================================================
# STEP 9: Save Results to Parquet Files
# ============================================================================

cat("============================================================================\n")
cat("Saving results to parquet files...\n")
cat("============================================================================\n")

# Create results directory if it doesn't exist
dir.create("examples/results", showWarnings = FALSE, recursive = TRUE)

# Save expanding window predictions (group 1)
write_parquet(
  predictions_expanding_single %>% select(group_id, obs_id, y, yhat, yhat_lower, yhat_upper),
  "examples/performance_10k_groups_R/results/r_predictions_expanding_single.parquet"
)

# Save fixed window predictions (group 1)
write_parquet(
  predictions_fixed_single %>% select(group_id, obs_id, y, yhat, yhat_lower, yhat_upper),
  "examples/performance_10k_groups_R/results/r_predictions_fixed_single.parquet"
)

# Save expanding window predictions (100 groups)
write_parquet(
  predictions_expanding_multi %>% select(group_id, obs_id, y, yhat, yhat_lower, yhat_upper),
  "examples/performance_10k_groups_R/results/r_predictions_expanding_multi.parquet"
)

# Save fixed window predictions (100 groups)
write_parquet(
  predictions_fixed_multi %>% select(group_id, obs_id, y, yhat, yhat_lower, yhat_upper),
  "examples/performance_10k_groups_R/results/r_predictions_fixed_multi.parquet"
)

cat("Results saved to examples/performance_10k_groups_R/results/\n\n")

# ============================================================================
# SUMMARY
# ============================================================================

cat("============================================================================\n")
cat("PERFORMANCE TEST SUMMARY (R)\n")
cat("============================================================================\n")
cat("Tests completed:\n")
cat("  1. Expanding window - single group\n")
cat("  2. Fixed window - single group\n")
cat("  3. Expanding window - 100 groups\n")
cat("  4. Fixed window - 100 groups\n")
cat("\n")
cat("Results saved to:\n")
cat("  - examples/performance_10k_groups_R/results/r_predictions_expanding_single.parquet\n")
cat("  - examples/performance_10k_groups_R/results/r_predictions_fixed_single.parquet\n")
cat("  - examples/performance_10k_groups_R/results/r_predictions_expanding_multi.parquet\n")
cat("  - examples/performance_10k_groups_R/results/r_predictions_fixed_multi.parquet\n")
cat("============================================================================\n")
