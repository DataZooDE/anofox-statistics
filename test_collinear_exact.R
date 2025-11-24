# Test the exact collinear data from aggregate_model_predict.test
# Category A data
sales <- c(100, 120, 140, 160)
price <- c(10, 12, 14, 16)
advertising <- c(5, 6, 7, 8)

# Note: advertising = 0.5 * price (perfect collinearity)

model <- lm(sales ~ price + advertising)

cat("Model summary:\n")
print(summary(model))

cat("\nCoefficients:\n")
print(coef(model))

cat("\nIntercept:", coef(model)[1], "\n")
cat("Price coef:", coef(model)[2], "\n")
cat("Advertising coef:", coef(model)[3], "\n")

cat("\nRank:", model$rank, "\n")
cat("df.residual:", model$df.residual, "\n")
cat("MSE (sigma^2):", summary(model)$sigma^2, "\n")

cat("\nResiduals:\n")
print(residuals(model))

cat("\nPredicted values:\n")
print(fitted(model))
