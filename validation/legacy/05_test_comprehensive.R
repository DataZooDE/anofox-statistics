#!/usr/bin/env Rscript
#
# Comprehensive Phase 2 Validation
# Tests: ols_inference, ols_predict_interval, residual_diagnostics
# Uses: (1) Simple synthetic data, (2) Realistic dataset with noise
#

cat("\n=== PHASE 2 COMPREHENSIVE VALIDATION ===\n\n")

# Load DuckDB CLI path
duckdb_cli <- Sys.getenv("DUCKDB_CLI", "/tmp/duckdb")
if (!file.exists(duckdb_cli)) {
    stop("DuckDB CLI not found at: ", duckdb_cli)
}

extension_path <- "build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension"
if (!file.exists(extension_path)) {
    stop("Extension not found at: ", extension_path)
}

cat(sprintf("Using DuckDB CLI: %s\n", duckdb_cli))
cat(sprintf("Extension path: %s\n\n", extension_path))

# ==============================================================================
# DATASET 1: Simple Synthetic Data (y = 2x + 1, perfect fit)
# ==============================================================================

cat("========================================\n")
cat("DATASET 1: Simple Synthetic (y = 2x + 1)\n")
cat("========================================\n\n")

# Simple linear: y = 2x + 1
x_simple <- c(1, 2, 3, 4, 5)
y_simple <- 2 * x_simple + 1  # Perfect fit

cat("Data:\n")
cat("  x =", paste(x_simple, collapse=", "), "\n")
cat("  y =", paste(y_simple, collapse=", "), "\n\n")

# Fit in R
model_simple <- lm(y_simple ~ x_simple)
cat("R Results (lm):\n")
print(summary(model_simple))

# Predictions in R
x_new_simple <- c(6, 7)
pred_simple <- predict(model_simple, newdata=data.frame(x_simple=x_new_simple),
                      interval="prediction", level=0.95)
cat("\nR Prediction intervals for x_new = [6, 7]:\n")
print(pred_simple)

cat("\n")

# ==============================================================================
# DATASET 2: Realistic Multiple Regression with Noise
# ==============================================================================

cat("========================================\n")
cat("DATASET 2: Realistic Multiple Regression\n")
cat("========================================\n\n")

# Create realistic dataset: Housing prices
# y = price (in $1000s)
# x1 = size (in 100 sq ft)
# x2 = bedrooms
# x3 = age (years)

set.seed(42)
n <- 50

# Generate realistic housing data with correlations
size <- rnorm(n, mean=20, sd=5)  # 2000 sq ft average
bedrooms <- pmax(1, round(0.3 * size + rnorm(n, mean=0, sd=0.5)))  # Correlated with size
age <- runif(n, 0, 50)  # 0-50 years old

# True model: price = 50 + 15*size + 10*bedrooms - 0.5*age + noise
price <- 50 + 15*size + 10*bedrooms - 0.5*age + rnorm(n, mean=0, sd=10)

cat("Dataset summary:\n")
cat(sprintf("  n = %d observations\n", n))
cat(sprintf("  Price: mean=%.1f, sd=%.1f, range=[%.1f, %.1f]\n",
           mean(price), sd(price), min(price), max(price)))
cat(sprintf("  Size: mean=%.1f, sd=%.1f\n", mean(size), sd(size)))
cat(sprintf("  Bedrooms: mean=%.1f, sd=%.1f\n", mean(bedrooms), sd(bedrooms)))
cat(sprintf("  Age: mean=%.1f, sd=%.1f\n\n", mean(age), sd(age)))

# Fit in R
model_real <- lm(price ~ size + bedrooms + age)
cat("R Results (lm):\n")
print(summary(model_real))

# Inference in R
cat("\nR Inference (coefficients with p-values):\n")
print(summary(model_real)$coefficients)

# Predictions for new data in R
size_new <- c(25, 18, 22)
bedrooms_new <- c(4, 3, 3)
age_new <- c(10, 25, 5)
newdata <- data.frame(size=size_new, bedrooms=bedrooms_new, age=age_new)

pred_real <- predict(model_real, newdata=newdata, interval="prediction", level=0.95)
cat("\nR Prediction intervals for 3 new houses:\n")
cat("  House 1: size=25, beds=4, age=10\n")
cat("  House 2: size=18, beds=3, age=25\n")
cat("  House 3: size=22, beds=3, age=5\n")
print(pred_real)

# Residual diagnostics in R
cat("\nR Residual Diagnostics (first 10 observations):\n")
residuals_r <- residuals(model_real)
std_resid_r <- rstandard(model_real)
studentized_r <- rstudent(model_real)
leverage_r <- hatvalues(model_real)
cooks_r <- cooks.distance(model_real)

diag_df <- data.frame(
    obs = 1:10,
    residual = residuals_r[1:10],
    std_resid = std_resid_r[1:10],
    studentized = studentized_r[1:10],
    leverage = leverage_r[1:10],
    cooks_d = cooks_r[1:10]
)
print(round(diag_df, 4))

cat("\n")

# ==============================================================================
# Save datasets for DuckDB testing
# ==============================================================================

# Dataset 1: Simple
write.csv(data.frame(x=x_simple, y=y_simple),
         "validation/data/phase2_simple.csv", row.names=FALSE)
write.csv(data.frame(x=x_new_simple),
         "validation/data/phase2_simple_new.csv", row.names=FALSE)

# Dataset 2: Realistic
write.csv(data.frame(price=price, size=size, bedrooms=bedrooms, age=age),
         "validation/data/phase2_realistic.csv", row.names=FALSE)
write.csv(newdata,
         "validation/data/phase2_realistic_new.csv", row.names=FALSE)

cat("Datasets saved to validation/data/\n")
cat("  - phase2_simple.csv\n")
cat("  - phase2_simple_new.csv\n")
cat("  - phase2_realistic.csv\n")
cat("  - phase2_realistic_new.csv\n\n")

cat("=== R Validation Complete ===\n")
cat("\nExpected values documented above.\n")
cat("Now test with DuckDB to compare results.\n\n")
