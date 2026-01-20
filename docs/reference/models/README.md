# Regression Models Reference

Detailed documentation for each regression model supported by the Anofox Statistics Extension.

## Linear Models

- [OLS](ols.md) - Ordinary Least Squares
- [Ridge](ridge.md) - L2 Regularized Regression
- [Elastic Net](elasticnet.md) - Combined L1/L2 Regularization
- [WLS](wls.md) - Weighted Least Squares
- [RLS](rls.md) - Recursive Least Squares

## Generalized Linear Models

- [GLM](glm.md) - Generalized Linear Models (Poisson)
- [ALM](alm.md) - Augmented Linear Models (24 distributions)

## Constrained Models

- [BLS](bls.md) - Bounded Least Squares
- [NNLS](nnls.md) - Non-Negative Least Squares

## Specialized Models

- [PLS](pls.md) - Partial Least Squares
- [Isotonic](isotonic.md) - Monotonic Regression
- [Quantile](quantile.md) - Quantile Regression

## Model Selection Guide

| Scenario | Recommended Model |
|----------|-------------------|
| Standard regression | OLS |
| Multicollinearity | Ridge, PLS |
| Feature selection | Elastic Net (high l1_ratio) |
| Heteroscedasticity | WLS |
| Streaming/adaptive | RLS |
| Count data | Poisson (GLM) |
| Heavy-tailed errors | ALM (Student-t, Cauchy) |
| Constrained coefficients | BLS, NNLS |
| Monotonic relationships | Isotonic |
| Robust to outliers | Quantile |
