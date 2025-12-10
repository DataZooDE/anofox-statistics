#!/usr/bin/env Rscript
# Generate test data for statistical inference and prediction intervals
# This script should only be run when regenerating test data
# Output: test/data/inference_tests/

# Use local R library
local_lib <- file.path(dirname(dirname(getwd())), "validation", "R_libs")
if (dir.exists(local_lib)) {
  .libPaths(c(local_lib, .libPaths()))
}

library(jsonlite)

cat("=== Generating Inference and Prediction Test Data ===\n\n")

set.seed(42)  # For reproducibility

inference_dir <- "test/data/inference_tests"
dir.create(file.path(inference_dir, "input"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(inference_dir, "expected"), recursive = TRUE, showWarnings = FALSE)

# ==============================================================================
# Test 1: Simple Linear - Inference
# ==============================================================================

cat("Generating Test 1: Simple Linear Inference\n")

n1 <- 100
x1 <- seq(1, 10, length.out = n1)
y1 <- 5 + 2*x1 + rnorm(n1, sd = 1.0)

# Save input data
input1 <- data.frame(x = x1, y = y1)
write.csv(input1, file.path(inference_dir, "input/simple_inference.csv"), row.names = FALSE)

# Fit model and extract inference results
model1 <- lm(y ~ x, data = input1)
summary1 <- summary(model1)
coef_table <- summary1$coefficients

# Confidence intervals
conf_intervals <- confint(model1, level = 0.95)

expected1 <- list(
  test_name = "simple_inference",
  coefficients = list(
    estimates = as.numeric(coef_table[, "Estimate"]),
    std_errors = as.numeric(coef_table[, "Std. Error"]),
    t_values = as.numeric(coef_table[, "t value"]),
    p_values = as.numeric(coef_table[, "Pr(>|t|)"]),
    names = rownames(coef_table)
  ),
  confidence_intervals = list(
    lower_95 = as.numeric(conf_intervals[, 1]),
    upper_95 = as.numeric(conf_intervals[, 2])
  ),
  model_stats = list(
    r_squared = summary1$r.squared,
    adj_r_squared = summary1$adj.r.squared,
    sigma = summary1$sigma,
    df_residual = summary1$df[2],
    fstatistic = as.numeric(summary1$fstatistic)
  )
)

write_json(expected1, file.path(inference_dir, "expected/simple_inference.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = 15)

cat(sprintf("  ✓ Simple inference: n=%d, p-values computed\n", n1))

# ==============================================================================
# Test 2: Multiple Regression - Inference
# ==============================================================================

cat("Generating Test 2: Multiple Regression Inference\n")

n2 <- 150
x2_1 <- rnorm(n2, mean = 5, sd = 2)
x2_2 <- rnorm(n2, mean = 10, sd = 3)
x2_3 <- rnorm(n2, mean = 15, sd = 4)
y2 <- 20 + 1.5*x2_1 + 2.5*x2_2 - 1.0*x2_3 + rnorm(n2, sd = 2)

# Save input data
input2 <- data.frame(x1 = x2_1, x2 = x2_2, x3 = x2_3, y = y2)
write.csv(input2, file.path(inference_dir, "input/multiple_inference.csv"), row.names = FALSE)

# Fit model and extract inference
model2 <- lm(y ~ x1 + x2 + x3, data = input2)
summary2 <- summary(model2)
coef_table2 <- summary2$coefficients
conf_intervals2 <- confint(model2, level = 0.95)

expected2 <- list(
  test_name = "multiple_inference",
  coefficients = list(
    estimates = as.numeric(coef_table2[, "Estimate"]),
    std_errors = as.numeric(coef_table2[, "Std. Error"]),
    t_values = as.numeric(coef_table2[, "t value"]),
    p_values = as.numeric(coef_table2[, "Pr(>|t|)"]),
    names = rownames(coef_table2)
  ),
  confidence_intervals = list(
    lower_95 = as.numeric(conf_intervals2[, 1]),
    upper_95 = as.numeric(conf_intervals2[, 2])
  ),
  model_stats = list(
    r_squared = summary2$r.squared,
    adj_r_squared = summary2$adj.r.squared,
    sigma = summary2$sigma,
    df_residual = summary2$df[2],
    fstatistic = as.numeric(summary2$fstatistic)
  )
)

write_json(expected2, file.path(inference_dir, "expected/multiple_inference.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = 15)

cat(sprintf("  ✓ Multiple inference: n=%d, p=3\n", n2))

# ==============================================================================
# Test 3: Prediction Intervals
# ==============================================================================

cat("Generating Test 3: Prediction Intervals\n")

# Use simple model for predictions
n3 <- 80
x3_train <- seq(1, 15, length.out = n3)
y3_train <- 3 + 1.5*x3_train + rnorm(n3, sd = 0.8)

# New data for predictions
x3_new <- c(5, 10, 16, 20)

# Save training data
input3_train <- data.frame(x = x3_train, y = y3_train)
write.csv(input3_train, file.path(inference_dir, "input/prediction_train.csv"), row.names = FALSE)

# Save new data
input3_new <- data.frame(x = x3_new)
write.csv(input3_new, file.path(inference_dir, "input/prediction_new.csv"), row.names = FALSE)

# Fit model
model3 <- lm(y ~ x, data = input3_train)

# Prediction intervals
pred_intervals <- predict(model3, newdata = input3_new, interval = "prediction", level = 0.95)
conf_intervals_pred <- predict(model3, newdata = input3_new, interval = "confidence", level = 0.95)

expected3 <- list(
  test_name = "prediction_intervals",
  new_x_values = x3_new,
  predictions = list(
    fit = as.numeric(pred_intervals[, "fit"]),
    prediction_lower = as.numeric(pred_intervals[, "lwr"]),
    prediction_upper = as.numeric(pred_intervals[, "upr"]),
    confidence_lower = as.numeric(conf_intervals_pred[, "lwr"]),
    confidence_upper = as.numeric(conf_intervals_pred[, "upr"])
  ),
  model_coefficients = as.numeric(coef(model3)),
  model_sigma = summary(model3)$sigma,
  note = "Prediction and confidence intervals at 95% level"
)

write_json(expected3, file.path(inference_dir, "expected/prediction_intervals.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = 15)

cat(sprintf("  ✓ Prediction intervals: n_train=%d, n_new=%d\n", n3, length(x3_new)))

# ==============================================================================
# Write metadata
# ==============================================================================

metadata <- list(
  generated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
  r_version = R.version.string,
  seed = 42,
  description = "Statistical inference and prediction interval test data",
  test_cases = list(
    simple_inference = "Coefficient inference for simple linear regression",
    multiple_inference = "Coefficient inference for multiple regression",
    prediction_intervals = "Prediction and confidence intervals for new data"
  ),
  confidence_level = 0.95,
  tolerance = list(
    coefficients = 1e-10,
    p_values = 1e-8,
    intervals = 1e-8,
    note = "Use strict tolerance for coefficients, relaxed for p-values and intervals"
  )
)

write_json(metadata, file.path(inference_dir, "metadata.json"),
           auto_unbox = FALSE, pretty = TRUE)

cat("\n✅ Generated inference and prediction test data\n")
cat(sprintf("   Location: %s\n", inference_dir))
cat("   Tests: 3 (simple inference, multiple inference, prediction intervals)\n")
