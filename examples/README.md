# Examples

This directory contains practical examples demonstrating the anofox-statistics extension features.

## Model-Based Prediction

### Overview

The model-based prediction workflow allows you to:
1. **Fit a model once** with `full_output=true` to store all metadata
2. **Make predictions many times** without refitting the model
3. **Choose interval types**: confidence, prediction, or none

This is much more efficient than refitting the model for each prediction.

### Files

- **`model_prediction_demo.sql`**: SQL demonstration showing the complete workflow
- **`model_prediction_demo.py`**: Python demonstration (requires duckdb-python)

### Running the SQL Demo

```bash
# From the project root
duckdb -unsigned -init examples/model_prediction_demo.sql
```

### Running the Python Demo

```bash
# Install dependencies
pip install duckdb numpy

# Run the demo
python3 examples/model_prediction_demo.py
```

## Key Concepts

### Workflow Comparison

**Traditional approach (refitting each time):**
```sql
-- Inefficient: refits model for each prediction
SELECT * FROM anofox_statistics_ols_predict_interval(
    y_train, x_train, x_new, 0.95, 'prediction', true
);
```

**Model-based approach (fit once, predict many times):**
```sql
-- 1. Fit once with full_output
CREATE TABLE model AS
SELECT * FROM anofox_statistics_ols(
    y_train, x_train, MAP{'intercept': true, 'full_output': true}
);

-- 2. Predict many times (no refitting!)
SELECT p.* FROM model m,
LATERAL anofox_statistics_model_predict(
    m.intercept, m.coefficients, m.mse, m.x_train_means,
    m.coefficient_std_errors, m.intercept_std_error, m.df_residual,
    x_new, 0.95, 'prediction'
) p;
```

### Interval Types

1. **Confidence Intervals** (`'confidence'`):
   - Narrower intervals
   - Represent uncertainty about the mean prediction
   - Use when predicting average behavior

2. **Prediction Intervals** (`'prediction'`):
   - Wider intervals
   - Represent uncertainty about individual predictions
   - Include both model uncertainty and individual variation
   - Use when predicting specific future observations

3. **No Intervals** (`'none'`):
   - Fastest option
   - Only returns point predictions
   - Use for high-speed batch scoring

### Performance Benefits

- ✅ **10x-100x faster** for batch predictions
- ✅ **Scalable**: Score millions of observations efficiently
- ✅ **Memory efficient**: Model stored once, reused many times
- ✅ **Production-ready**: Perfect for prediction pipelines

## More Examples

For more comprehensive examples, see:
- [Quick Start Guide](../guides/01_quick_start.md)
- [Function Reference](../guides/function_reference.md)
- [Advanced Use Cases](../guides/05_advanced_use_cases.md)
