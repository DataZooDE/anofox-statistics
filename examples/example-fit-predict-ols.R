# ============================================================================
# OLS Fit-Predict Examples in R using lm()
# ============================================================================
# This file demonstrates the same functionality as the SQL examples using
# R's built-in linear model (lm) function.

# ============================================================================
# Example 1: Simple Linear Regression with Train/Test Split
# ============================================================================
# Create a dataset where y = 2*x + 1, with first 10 rows for training
# and last 5 rows for prediction (y = NULL)

set.seed(42)  # For reproducibility

# Create data
simple_linear <- data.frame(
  id = 1:15,
  x = 1:15
)

# Add y values (only for first 10 rows, rest are NA)
simple_linear$y <- ifelse(
  simple_linear$id <= 10,
  2.0 * simple_linear$id + 1.0 + runif(15, -0.25, 0.25),
  NA
)

# View the data
print("Example 1: Simple Linear Regression Data")
print(simple_linear)

# Fit model on training data (rows where y is not NA)
train_data <- simple_linear[!is.na(simple_linear$y), ]
model <- lm(y ~ x, data = train_data)

# Display model summary
print("\nModel Summary:")
print(summary(model))

# Predict on all data (both train and test)
predictions <- predict(model, newdata = simple_linear, interval = "prediction", level = 0.95)

# Combine results
results <- data.frame(
  id = simple_linear$id,
  x = simple_linear$x,
  y = simple_linear$y,
  yhat = round(predictions[, "fit"], 2),
  lower = round(predictions[, "lwr"], 2),
  upper = round(predictions[, "upr"], 2)
)

print("\nPredictions (train and test):")
print(results)

# ============================================================================
# Example 2: Fixed Model - Train Once, Predict All (In-Sample Predictions)
# ============================================================================
# Fit ONE model using all training data, then use that SAME fixed model to
# predict on both training and test sets.

set.seed(123)

# Create data
train_test_split <- data.frame(
  id = 1:20,
  x = 1:20
)

# Add y values (only for first 12 rows)
train_test_split$y <- ifelse(
  train_test_split$id <= 12,
  2.5 * train_test_split$id + 3.0 + runif(20, -1.0, 1.0),
  NA
)

print("\nExample 2: Fixed Model - Train/Test Split")
print(sprintf("Total rows: %d", nrow(train_test_split)))
print(sprintf("Training rows: %d", sum(!is.na(train_test_split$y))))
print(sprintf("Test rows: %d", sum(is.na(train_test_split$y))))

# Fit model on ALL training data (rows 1-12)
train_data <- train_test_split[!is.na(train_test_split$y), ]
model_fixed <- lm(y ~ x, data = train_data)

# Predict on ALL rows (1-20) using the same fixed model
predictions_fixed <- predict(model_fixed, newdata = train_test_split,
                             interval = "prediction", level = 0.95)

# Combine results
results_fixed <- data.frame(
  id = train_test_split$id,
  x = train_test_split$x,
  y = train_test_split$y,
  yhat = round(predictions_fixed[, "fit"], 2),
  lower = round(predictions_fixed[, "lwr"], 2),
  upper = round(predictions_fixed[, "upr"], 2),
  abs_error = ifelse(
    !is.na(train_test_split$y),
    round(abs(train_test_split$y - predictions_fixed[, "fit"]), 2),
    NA
  ),
  dataset = ifelse(is.na(train_test_split$y), "Test", "Train")
)

print("\nFixed Model Predictions:")
print(results_fixed)

# Calculate in-sample metrics
train_predictions <- results_fixed[results_fixed$dataset == "Train", ]
rmse_train <- sqrt(mean(train_predictions$abs_error^2, na.rm = TRUE))
print(sprintf("\nIn-sample RMSE: %.4f", rmse_train))

# ============================================================================
# Example 10: Extracting Model Coefficients
# ============================================================================
# Extract and interpret model coefficients, standard errors, and statistics

set.seed(456)

# Create data with two features
coef_demo <- data.frame(
  id = 1:50,
  x1 = 1:50,
  x2 = (1:50) + runif(50, 0, 3)
)

coef_demo$y <- 2.0 * coef_demo$x1 +
               1.5 * coef_demo$x2 +
               10.0 +
               runif(50, -2.5, 2.5)

# Fit model
model_coef <- lm(y ~ x1 + x2, data = coef_demo)

cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("Example 10: Extracting Model Coefficients\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

# 1. Model Statistics
print("\n1. Model Statistics:")
summary_stats <- summary(model_coef)
print(summary_stats)

# 2. Extract coefficients with interpretation
print("\n2. Coefficient Interpretation:")
coefficients <- coef(model_coef)
print(data.frame(
  Parameter = names(coefficients),
  Estimate = round(coefficients, 4)
))

# Build equation string
equation <- sprintf("y = %.2f + %.2f*x1 + %.2f*x2",
                   coefficients["(Intercept)"],
                   coefficients["x1"],
                   coefficients["x2"])
print(sprintf("\nRegression Equation: %s", equation))

# 3. Coefficients with standard errors and t-statistics
print("\n3. Statistical Significance:")
coef_table <- summary_stats$coefficients
print(data.frame(
  Parameter = rownames(coef_table),
  Estimate = round(coef_table[, "Estimate"], 4),
  Std_Error = round(coef_table[, "Std. Error"], 4),
  t_statistic = round(coef_table[, "t value"], 4),
  p_value = format.pval(coef_table[, "Pr(>|t|)"], digits = 4)
))

# 4. Model quality metrics
print("\n4. Model Quality Metrics:")
print(sprintf("R-squared: %.4f", summary_stats$r.squared))
print(sprintf("Adjusted R-squared: %.4f", summary_stats$adj.r.squared))
print(sprintf("Residual Standard Error: %.4f", summary_stats$sigma))
print(sprintf("Degrees of Freedom: %d", summary_stats$df[2]))
print(sprintf("F-statistic: %.4f on %d and %d DF, p-value: %s",
             summary_stats$fstatistic[1],
             summary_stats$fstatistic[2],
             summary_stats$fstatistic[3],
             format.pval(pf(summary_stats$fstatistic[1],
                           summary_stats$fstatistic[2],
                           summary_stats$fstatistic[3],
                           lower.tail = FALSE), digits = 4)))

# 5. Compare coefficients across groups
print("\n5. Grouped Coefficients Comparison:")
set.seed(789)

grouped_coef <- data.frame(
  id = rep(1:30, 2),
  x = rep(1:30, 2),
  group_name = rep(c("Group_A", "Group_B"), each = 30)
)

# Generate y values with different slopes and intercepts per group
grouped_coef$y <- ifelse(
  grouped_coef$group_name == "Group_A",
  2.5 * grouped_coef$x + 10.0 + runif(30, -1.5, 1.5),
  -1.2 * grouped_coef$x + 25.0 + runif(30, -1.5, 1.5)
)

# Fit separate models for each group
group_results <- data.frame()
for (group in unique(grouped_coef$group_name)) {
  group_data <- grouped_coef[grouped_coef$group_name == group, ]
  group_model <- lm(y ~ x, data = group_data)
  group_summary <- summary(group_model)

  group_results <- rbind(group_results, data.frame(
    group_name = group,
    intercept = round(coef(group_model)["(Intercept)"], 2),
    slope = round(coef(group_model)["x"], 2),
    r_squared = round(group_summary$r.squared, 4),
    n = nrow(group_data)
  ))
}

print(group_results)

# ============================================================================
# Comparison Notes
# ============================================================================
cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("Comparison: R lm() vs DuckDB OLS Functions\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
print("\nKey Differences:")
print("1. Fit-Predict Modes:")
print("   - DuckDB 'fixed' mode: Fits ONCE on all training data, predicts all rows")
print("     (This is equivalent to what R lm() does - shown in Examples 1 and 2)")
print("   - DuckDB 'expanding' mode: Fits on growing window, adapts over time")
print("     (Not shown in R examples - would require a loop to replicate)")
print("")
print("2. Window Functions:")
print("   - DuckDB: Use OVER clause for expanding/rolling windows")
print("   - R: Need loops or apply functions to achieve similar behavior")
print("")
print("3. Predictions:")
print("   - DuckDB: fit_predict functions combine fitting and prediction")
print("   - R: Separate lm() for fitting, predict() for predictions")
print("")
print("4. Grouped Analysis:")
print("   - DuckDB: Use PARTITION BY in window functions")
print("   - R: Use split-apply-combine pattern or dplyr::group_by()")
print("")
print("5. In-Database Processing:")
print("   - DuckDB: All computation happens in the database")
print("   - R: Data must be loaded into memory")
