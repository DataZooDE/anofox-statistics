# Isotonic Regression

## Overview

Isotonic regression fits a non-decreasing (or non-increasing) step function to the data.

## Implementation

- **Algorithm:** Pool Adjacent Violators (PAVA)
- **Backend:** Rust
- **Complexity:** O(n log n)

## When to Use

- Monotonic relationships (e.g., dose-response)
- Calibrating probability predictions
- When domain knowledge requires monotonicity
- Non-parametric regression with shape constraints

## Available Functions

| Function | Type | Description |
|----------|------|-------------|
| `isotonic_fit` | Scalar | Fit on array data |
| `isotonic_fit_agg` | Aggregate | Streaming fit with GROUP BY support |
| `isotonic_fit_predict_agg` | Aggregate | Fit and return predictions array |
| `isotonic_fit_predict_by` | Table Macro | Fit per group, return predictions table |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| increasing | true | Enforce non-decreasing constraint |

## Example

```sql
-- Non-decreasing fit
SELECT isotonic_fit_agg(response, dose)
FROM dose_response_data;

-- Non-increasing fit
SELECT isotonic_fit_agg(y, x, {'increasing': false})
FROM data;

-- Probability calibration
SELECT isotonic_fit_agg(actual_outcome, predicted_probability)
FROM model_predictions;
```

## Output

Returns step function values at each unique x:
- Predictions are constant between consecutive x values
- New x values interpolated using nearest neighbors

## Related

- [Quantile](quantile.md) - Another non-parametric option
- [API Reference](../../api/07-regression-isotonic.md)
