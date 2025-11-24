# Test case: Collinear features with intercept
# This matches the test data from aggregate_model_predict.test

# Category A data
category_a <- data.frame(
  sales = c(100, 120, 140, 160),
  price = c(10, 12, 14, 16),
  advertising = c(5, 6, 7, 8)
)

# Category B data (same structure)
category_b <- data.frame(
  sales = c(200, 220, 240, 260),
  price = c(20, 22, 24, 26),
  advertising = c(10, 11, 12, 13)
)

cat("=== Category A Analysis ===\n")
cat("Data:\n")
print(category_a)
cat("\nCheck collinearity:\n")
cat("advertising / price ratios:", category_a$advertising / category_a$price, "\n")
cat("Perfect collinearity:", length(unique(category_a$advertising / category_a$price)) == 1, "\n\n")

# Fit OLS model with intercept
cat("Fitting: sales ~ price + advertising (with intercept)\n")
model_a <- lm(sales ~ price + advertising, data = category_a)

cat("\nModel summary:\n")
print(summary(model_a))

cat("\n=== Key Statistics ===\n")
cat("Number of observations (n):", nobs(model_a), "\n")
cat("Residual degrees of freedom (df.residual):", df.residual(model_a), "\n")
cat("Model rank (from qr):", model_a$rank, "\n")
cat("Coefficients:\n")
print(coef(model_a))
cat("\nMSE (residual standard error^2):", summary(model_a)$sigma^2, "\n")
cat("Intercept SE:", summary(model_a)$coefficients["(Intercept)", "Std. Error"], "\n")

cat("\n=== Category B Analysis ===\n")
cat("Data:\n")
print(category_b)
model_b <- lm(sales ~ price + advertising, data = category_b)
cat("\nKey Statistics:\n")
cat("df.residual:", df.residual(model_b), "\n")
cat("Model rank:", model_b$rank, "\n")
cat("MSE:", summary(model_b)$sigma^2, "\n")

cat("\n=== Interpretation ===\n")
cat("In R, when features are perfectly collinear:\n")
cat("- One coefficient becomes NA (aliased)\n")
cat("- model$rank counts the number of NON-ALIASED parameters\n")
cat("- df.residual = n - rank\n")
cat("- rank INCLUDES the intercept when intercept=TRUE\n")
