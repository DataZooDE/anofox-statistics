# Table Macros

Table macros that wrap `*_fit_predict_agg` functions for easy per-group regression with long-format output. These macros simplify common workflows by handling grouping, prediction, and column extraction automatically.

All source columns are **passed through** to the output, so you retain the original data alongside predictions.

## Overview

All regression table macros share a common interface:

```sql
<method>_fit_predict_by(
    source VARCHAR,           -- Table name (as string)
    group_col COLUMN,         -- Column to group by
    y_col COLUMN,             -- Response variable column
    x_cols LIST(COLUMN),      -- Feature columns as list
    [options STRUCT],         -- Optional configuration
    [split COLUMN]            -- Optional train/test split column
) -> TABLE
```

**Return Columns:**

All columns from the source table are preserved in the output (including the group column, y column, and all feature columns with their original names). The following prediction columns are appended:

| Column | Type | Description |
|--------|------|-------------|
| yhat | DOUBLE | Predicted value |
| yhat_lower | DOUBLE | Lower prediction interval bound |
| yhat_upper | DOUBLE | Upper prediction interval bound |
| is_training | BOOLEAN | True if row was used for training |

> **Note:** Column names in the output preserve the original names from the source table. For example, if you pass `region` as the group column and `revenue` as y_col, the output will have `region` and `revenue` columns, not generic names.
>
> PLS, isotonic, and quantile regression macros return only `yhat` and `is_training` (no prediction intervals).

**Split Parameter:**

The optional `split` parameter accepts a column containing `'train'`/`'test'` values. Rows where `split != 'train'` are treated as out-of-sample (y is NULLed before fitting). When not provided, the default behavior applies: rows where y is NULL are out-of-sample.

```sql
-- With split column
SELECT * FROM ols_fit_predict_by('data', group_id, target, [x1, x2],
    options := {'null_policy': 'drop'}, split := split_col);
```

**Common Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| fit_intercept | BOOLEAN | true | Include intercept term |
| confidence_level | DOUBLE | 0.95 | Prediction interval confidence |
| null_policy | VARCHAR | 'drop' | NULL handling: 'drop' or 'drop_y_zero_x' |

## ols_fit_predict_by

OLS regression per group with predictions in long format.

**Example:**
```sql
-- Per-group OLS regression
SELECT * FROM ols_fit_predict_by('sales_data', region, revenue, [advertising, price]);

-- With 99% prediction intervals
SELECT * FROM ols_fit_predict_by('sales_data', region, revenue, [advertising, price],
    {'confidence_level': 0.99});

-- Filter to out-of-sample predictions only
SELECT * FROM ols_fit_predict_by('forecast_data', store_id, sales, [inventory, promotions])
WHERE NOT is_training;
```

## ridge_fit_predict_by

Ridge regression per group with L2 regularization.

**Additional Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| alpha | DOUBLE | 1.0 | L2 regularization strength |

**Example:**
```sql
-- Ridge with default alpha
SELECT * FROM ridge_fit_predict_by('data', category, y, [x1, x2]);

-- Ridge with custom regularization
SELECT * FROM ridge_fit_predict_by('data', category, y, [x1, x2],
    {'alpha': 0.5});

-- Strong regularization
SELECT * FROM ridge_fit_predict_by('data', category, y, [x1, x2],
    {'alpha': 10.0, 'confidence_level': 0.99});
```

## elasticnet_fit_predict_by

Elastic Net regression per group with combined L1/L2 regularization.

**Additional Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| alpha | DOUBLE | 1.0 | Regularization strength |
| l1_ratio | DOUBLE | 0.5 | L1 ratio: 0=Ridge, 1=Lasso |
| max_iterations | INTEGER | 1000 | Max coordinate descent iterations |
| tolerance | DOUBLE | 1e-6 | Convergence tolerance |

**Example:**
```sql
-- ElasticNet with default settings
SELECT * FROM elasticnet_fit_predict_by('data', category, y, [x1, x2]);

-- More Lasso-like (70% L1)
SELECT * FROM elasticnet_fit_predict_by('data', category, y, [x1, x2],
    {'alpha': 0.1, 'l1_ratio': 0.7});
```

## wls_fit_predict_by

Weighted Least Squares per group. Requires a weight column.

**Signature:**
```sql
wls_fit_predict_by(
    source VARCHAR,
    group_col COLUMN,
    y_col COLUMN,
    x_cols LIST(COLUMN),
    weight_col COLUMN,        -- Weight column (required)
    [options STRUCT]
) -> TABLE
```

**Example:**
```sql
-- WLS with weight column
SELECT * FROM wls_fit_predict_by('weighted_data', segment, y, [x1, x2], weight);

-- WLS with custom confidence level
SELECT * FROM wls_fit_predict_by('weighted_data', segment, y, [x1, x2], weight,
    {'confidence_level': 0.99});
```

## rls_fit_predict_by

Recursive Least Squares per group for adaptive/online regression.

**Additional Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| forgetting_factor | DOUBLE | 1.0 | Exponential forgetting (0.95-1.0 typical) |
| initial_p_diagonal | DOUBLE | 100.0 | Initial covariance diagonal |

**Example:**
```sql
-- RLS with default settings
SELECT * FROM rls_fit_predict_by('streaming_data', sensor_id, reading, [temp, pressure]);

-- RLS with forgetting (adapts to recent data)
SELECT * FROM rls_fit_predict_by('streaming_data', sensor_id, reading, [temp, pressure],
    {'forgetting_factor': 0.95});
```

## bls_fit_predict_by

Bounded Least Squares per group with box constraints on coefficients.

**Additional Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| lower_bound | DOUBLE | 0.0 | Lower bound for coefficients |
| upper_bound | DOUBLE | +inf | Upper bound for coefficients |
| intercept | BOOLEAN | false | Include intercept term |
| max_iterations | INTEGER | 1000 | Maximum iterations |
| tolerance | DOUBLE | 1e-6 | Convergence tolerance |

**Example:**
```sql
-- BLS with default (non-negative coefficients)
SELECT * FROM bls_fit_predict_by('constrained_data', portfolio_id, returns, [factor1, factor2]);

-- Box constraints (coefficients between 0 and 1)
SELECT * FROM bls_fit_predict_by('portfolio_data', asset_class, returns, [factors],
    {'lower_bound': 0.0, 'upper_bound': 1.0});
```

## alm_fit_predict_by

Augmented Linear Model per group with flexible error distributions.

**Additional Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| distribution | VARCHAR | 'normal' | Error distribution |
| intercept | BOOLEAN | true | Include intercept term |
| max_iterations | INTEGER | 1000 | Maximum iterations |
| tolerance | DOUBLE | 1e-6 | Convergence tolerance |

**Distributions:** `normal`, `laplace`, `studentt`, `cauchy`, `huber`, `tukey`, `quantile`, `expectile`, `trimmed`, `winsorized`

**Example:**
```sql
-- ALM with default (normal distribution)
SELECT * FROM alm_fit_predict_by('robust_data', group_id, y, [x1, x2]);

-- Robust regression with Laplace (median regression)
SELECT * FROM alm_fit_predict_by('data_with_outliers', group_id, y, [x1, x2],
    {'distribution': 'laplace'});

-- Student-t for heavy tails
SELECT * FROM alm_fit_predict_by('heavy_tailed_data', group_id, y, [x1, x2],
    {'distribution': 'studentt'});
```

## poisson_fit_predict_by

Poisson GLM per group for count data.

**Additional Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| link | VARCHAR | 'log' | Link function: 'log', 'identity', 'sqrt' |
| intercept | BOOLEAN | true | Include intercept term |
| max_iterations | INTEGER | 100 | Maximum IRLS iterations |
| tolerance | DOUBLE | 1e-8 | Convergence tolerance |

**Example:**
```sql
-- Poisson with default log link
SELECT * FROM poisson_fit_predict_by('count_data', store_id, visitor_count, [marketing_spend]);

-- Poisson with identity link
SELECT * FROM poisson_fit_predict_by('count_data', store_id, visitor_count, [marketing_spend],
    {'link': 'identity'});
```

## aid_anomaly_by

Table macro for grouped anomaly detection using AID analysis.

**Signature:**
```sql
aid_anomaly_by(
    source VARCHAR,           -- Table name
    group_col COLUMN,         -- Column to group by (e.g., product_id)
    order_col COLUMN,         -- Column to order by within group (e.g., date)
    y_col COLUMN,             -- Numeric column to analyze for anomalies
    [options MAP]             -- Optional configuration
) -> TABLE
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| intermittent_threshold | DOUBLE | 0.3 | Zero proportion threshold |
| outlier_method | VARCHAR | 'zscore' | Outlier detection: 'zscore' or 'iqr' |

**Returns:**
| Column | Type | Description |
|--------|------|-------------|
| \<group_col\> | ANY | Group identifier (preserves original column name) |
| \<order_col\> | ANY | Order value (preserves original column name) |
| stockout | BOOLEAN | Unexpected zero in positive demand period |
| new_product | BOOLEAN | Part of leading zeros pattern |
| obsolete_product | BOOLEAN | Part of trailing zeros pattern |
| high_outlier | BOOLEAN | Unusually high value |
| low_outlier | BOOLEAN | Unusually low value |

**Example:**
```sql
-- Basic usage - returns anomaly flags with group and order columns
SELECT * FROM aid_anomaly_by('sales_data', product_id, sale_date, quantity, NULL);

-- With custom options
SELECT * FROM aid_anomaly_by('sales_data', product_id, sale_date, quantity,
    {'intermittent_threshold': 0.5, 'outlier_method': 'iqr'});

-- Filter to only stockout anomalies (column names are preserved)
SELECT sku, period
FROM aid_anomaly_by('inventory', sku, period, demand, NULL)
WHERE stockout;

-- Aggregate anomaly counts per product (column names are preserved)
SELECT product_id,
       SUM(stockout::INT) AS stockout_count,
       SUM(high_outlier::INT) AS high_outlier_count
FROM aid_anomaly_by('sales_data', product_id, sale_date, quantity, NULL)
GROUP BY product_id;
```

## null_policy Parameter

The `null_policy` option controls how NULL values and zero x values are handled:

| Value | Training Set | Predictions |
|-------|--------------|-------------|
| `'drop'` (default) | Rows where y IS NOT NULL | All rows get predictions |
| `'drop_y_zero_x'` | Rows where y IS NOT NULL AND all x != 0 | All rows get predictions |

## Summary Table

| Macro | Method | Key Options |
|-------|--------|-------------|
| ols_fit_predict_by | OLS | (common only) |
| ridge_fit_predict_by | Ridge | alpha |
| elasticnet_fit_predict_by | Elastic Net | alpha, l1_ratio |
| wls_fit_predict_by | WLS | weight_col |
| rls_fit_predict_by | RLS | forgetting_factor |
| bls_fit_predict_by | BLS | lower_bound, upper_bound |
| alm_fit_predict_by | ALM | distribution |
| poisson_fit_predict_by | Poisson | link |
| aid_anomaly_by | AID | intermittent_threshold, outlier_method |

## See Also

- [OLS](../regression/ols.md) - OLS functions
- [ALM](../glm/alm.md) - ALM distributions
- [AID](../aid/aid.md) - AID demand classification
