# Test confidence intervals with correct df
y <- c(2.1, 4.2, 5.9, 8.1, 10.0)
x <- c(1.0, 2.0, 3.0, 4.0, 5.0)

model <- lm(y ~ x)

cat("Model summary:\n")
cat("Coefficients:\n")
print(coef(model))
cat("\ndf.residual:", model$df.residual, "\n")
cat("MSE (sigma^2):", summary(model)$sigma^2, "\n")
cat("RMSE (sigma):", summary(model)$sigma, "\n")

# Predict at x = 6, 7, 8 with confidence intervals
new_data <- data.frame(x = c(6, 7, 8))
pred_conf <- predict(model, newdata = new_data, interval = "confidence", level = 0.95)

cat("\nPredictions with 95% confidence intervals:\n")
print(pred_conf)

# Also get prediction intervals for x = 6
new_data_single <- data.frame(x = 6)
pred_pred <- predict(model, newdata = new_data_single, interval = "prediction", level = 0.95)
cat("\nPrediction with 95% prediction interval (x=6):\n")
print(pred_pred)

# CI widths for different confidence levels at x = 6
pred_90 <- predict(model, newdata = new_data_single, interval = "prediction", level = 0.90)
pred_99 <- predict(model, newdata = new_data_single, interval = "prediction", level = 0.99)
cat("\n90% prediction interval width:", pred_90[1, "upr"] - pred_90[1, "lwr"], "\n")
cat("99% prediction interval width:", pred_99[1, "upr"] - pred_99[1, "lwr"], "\n")
