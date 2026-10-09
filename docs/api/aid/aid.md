# AID (Automatic Identification of Demand)

AID provides demand pattern classification and anomaly detection for time series data. Useful for inventory management, supply chain analysis, and demand forecasting.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `aid_by` | Table Macro | Grouped demand classification with wide-format output |
| `aid_anomaly_by` | Table Macro | Grouped anomaly detection with long-format output |
| `aid_agg` | Aggregate | Classify demand patterns and detect anomalies |
| `aid_anomaly_agg` | Aggregate | Per-observation anomaly flags |

The examples on this page use this table:

```sql
-- 52 weeks of demand for four SKUs with different patterns
CREATE OR REPLACE TABLE sales AS
SELECT sku, period,
       CASE sku
           WHEN 'WIDGET001' THEN 20 + (period * 7) % 9                     -- regular
           WHEN 'WIDGET002' THEN CASE WHEN period % 4 = 0 THEN 6 ELSE 0 END -- intermittent
           WHEN 'WIDGET003' THEN CASE WHEN period < 20 THEN 0 ELSE 15 + period % 5 END  -- new product
           ELSE CASE WHEN period IN (10, 11) THEN 0                        -- stockouts
                     WHEN period = 30 THEN 200 ELSE 30 + period % 6 END    -- and a spike
       END::DOUBLE AS demand
FROM (VALUES ('WIDGET001'), ('WIDGET002'), ('WIDGET003'), ('WIDGET004')) s(sku),
     range(1, 53) r(period);
```

## Table Macros (Recommended Entry Point)

Table macros are the easiest way to use AID functions. They handle the GROUP BY, column extraction, and result formatting automatically.

### aid_by

Classifies demand patterns for each group, returning one row per group with flat columns.

**Signature:**
```text
aid_by(
    source VARCHAR,           -- Table name (as string)
    group_col COLUMN,         -- Column to group by
    y_col COLUMN,             -- Demand/value column
    [options MAP]             -- Optional configuration (default: NULL)
) -> TABLE
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| intermittent_threshold | DOUBLE | 0.3 | Zero proportion cutoff for intermittent classification |
| outlier_method | VARCHAR | 'zscore' | Outlier detection: 'zscore' (mean±3σ) or 'iqr' (1.5×IQR) |

**Returns:**
| Column | Type | Description |
|--------|------|-------------|
| \<group_col\> | ANY | Group identifier (preserves original column name) |
| demand_type | VARCHAR | 'regular' or 'intermittent' |
| is_intermittent | BOOLEAN | True if zero_proportion >= threshold |
| distribution | VARCHAR | Best-fit distribution name |
| mean | DOUBLE | Mean of values |
| variance | DOUBLE | Variance of values |
| zero_proportion | DOUBLE | Proportion of zero values (0.0 to 1.0) |
| n_observations | BIGINT | Number of observations |
| has_stockouts | BOOLEAN | True if stockouts detected |
| is_new_product | BOOLEAN | True if new product pattern (leading zeros) |
| is_obsolete_product | BOOLEAN | True if obsolete pattern (trailing zeros) |
| stockout_count | BIGINT | Number of stockout observations |
| new_product_count | BIGINT | Number of leading zero observations |
| obsolete_product_count | BIGINT | Number of trailing zero observations |
| high_outlier_count | BIGINT | Number of unusually high values |
| low_outlier_count | BIGINT | Number of unusually low values |

**Example:**
```sql
-- Classify demand pattern for each SKU
SELECT * FROM aid_by('sales', sku, demand);

-- With custom intermittent threshold
SELECT * FROM aid_by('sales', sku, demand, {'intermittent_threshold': 0.4});

-- Find products with stockout issues
SELECT * FROM aid_by('sales', sku, demand)
WHERE has_stockouts
ORDER BY stockout_count DESC;
```

### aid_anomaly_by

Per-observation anomaly detection for each group, returning one row per observation.

**Signature:**
```text
aid_anomaly_by(
    source VARCHAR,           -- Table name
    group_col COLUMN,         -- Column to group by
    order_col COLUMN,         -- Column to order by within group
    y_col COLUMN,             -- Numeric column to analyze
    [options MAP]             -- Optional configuration
) -> TABLE
```

**Returns:**
| Column | Type | Description |
|--------|------|-------------|
| \<group_col\> | ANY | Group identifier (preserves original column name) |
| \<order_col\> | ANY | Order column value (preserves original column name) |
| stockout | BOOLEAN | Unexpected zero in positive demand |
| new_product | BOOLEAN | Leading zeros pattern |
| obsolete_product | BOOLEAN | Trailing zeros pattern |
| high_outlier | BOOLEAN | Unusually high value |
| low_outlier | BOOLEAN | Unusually low value |

**Example:**
```sql
-- Get anomaly flags per SKU and week
SELECT * FROM aid_anomaly_by('sales', sku, period, demand) LIMIT 5;

-- Filter to stockouts only (using actual column names)
SELECT sku, period
FROM aid_anomaly_by('sales', sku, period, demand)
WHERE stockout;
```

---

## Aggregate Functions

### aid_agg

Classifies demand patterns as regular or intermittent, identifies best-fit distribution, and detects various anomaly patterns.

**Signature:**
```text
aid_agg(y DOUBLE [, options MAP]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| intermittent_threshold | DOUBLE | 0.3 | Zero proportion cutoff for intermittent classification |
| outlier_method | VARCHAR | 'zscore' | Outlier detection: 'zscore' (mean±3σ) or 'iqr' (1.5×IQR) |

**Returns:**
```text
STRUCT(
    demand_type VARCHAR,           -- 'regular' or 'intermittent'
    is_intermittent BOOLEAN,       -- True if zero_proportion >= threshold
    distribution VARCHAR,          -- Best-fit distribution name
    mean DOUBLE,                   -- Mean of values
    variance DOUBLE,               -- Variance of values
    zero_proportion DOUBLE,        -- Proportion of zero values
    n_observations BIGINT,         -- Number of observations
    has_stockouts BOOLEAN,         -- True if stockouts detected
    is_new_product BOOLEAN,        -- True if new product pattern (leading zeros)
    is_obsolete_product BOOLEAN,   -- True if obsolete pattern (trailing zeros)
    stockout_count BIGINT,         -- Number of stockout observations
    new_product_count BIGINT,      -- Number of leading zero observations
    obsolete_product_count BIGINT, -- Number of trailing zero observations
    high_outlier_count BIGINT,     -- Number of unusually high values
    low_outlier_count BIGINT       -- Number of unusually low values
)
```

**Distribution Selection:**
- Count-like data: `poisson`, `negative_binomial`, `geometric`
- Continuous data: `normal`, `gamma`, `lognormal`, `rectified_normal`

**Example:**
```sql
-- Classify demand pattern for each SKU
SELECT sku, unnest(result)
FROM (SELECT sku, aid_agg(demand) AS result FROM sales GROUP BY sku)
ORDER BY sku;

-- With custom threshold
SELECT aid_agg(demand, {'intermittent_threshold': 0.4})
FROM sales
WHERE sku = 'WIDGET001';

-- Using IQR-based outlier detection
SELECT sku, (aid_agg(demand, {'outlier_method': 'iqr'})).high_outlier_count
FROM sales
GROUP BY sku
ORDER BY sku;
```

### aid_anomaly_agg

Returns per-observation anomaly flags for demand analysis. Maintains input order.

**Signature:**
```text
aid_anomaly_agg(y DOUBLE [, options MAP]) -> LIST(STRUCT)
```

The flags follow the order in which rows reach the aggregate, so pass an
`ORDER BY` inside the call (e.g. `aid_anomaly_agg(demand ORDER BY period)`)
when the order matters.

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| intermittent_threshold | DOUBLE | 0.3 | Zero proportion cutoff |
| outlier_method | VARCHAR | 'zscore' | Outlier detection: 'zscore' or 'iqr' |

**Returns:**
```text
LIST(STRUCT(
    stockout BOOLEAN,              -- Unexpected zero in positive demand
    new_product BOOLEAN,           -- Leading zeros pattern
    obsolete_product BOOLEAN,      -- Trailing zeros pattern
    high_outlier BOOLEAN,          -- Unusually high value
    low_outlier BOOLEAN            -- Unusually low value
))
```

**Anomaly Definitions:**
| Anomaly | Description |
|---------|-------------|
| **Stockout** | Zero value occurring between non-zero values |
| **New Product** | Leading sequence of zeros (before first non-zero) |
| **Obsolete Product** | Trailing sequence of zeros (after last non-zero) |
| **High Outlier** | Value > mean + 3*std (zscore) or > Q3 + 1.5*IQR (iqr) |
| **Low Outlier** | Non-zero value < mean - 3*std (zscore) or < Q1 - 1.5*IQR (iqr) |

**Example:**
```sql
-- Get anomaly flags for demand series
SELECT aid_anomaly_agg(demand ORDER BY t)
FROM (VALUES (1, 0), (2, 0), (3, 5), (4, 0), (5, 8), (6, 0), (7, 0)) AS v(t, demand);
-- Returns: [
--   {stockout: false, new_product: true, ...},   -- Leading zero
--   {stockout: false, new_product: true, ...},   -- Leading zero
--   {stockout: false, new_product: false, ...},  -- First non-zero
--   {stockout: true, new_product: false, ...},   -- Stockout (zero between)
--   {stockout: false, new_product: false, ...},  -- Normal
--   {stockout: false, obsolete_product: true,...}, -- Trailing zero
--   {stockout: false, obsolete_product: true,...}  -- Trailing zero
-- ]

-- Identify problematic SKUs with stockouts
WITH anomalies AS (
    SELECT sku, aid_agg(demand) as result
    FROM sales
    GROUP BY sku
)
SELECT sku, result.stockout_count
FROM anomalies
WHERE result.has_stockouts
ORDER BY result.stockout_count DESC;
```

---

## NULL Handling

A NULL `y` is kept as a missing value (NaN) so that positions are preserved:
`aid_anomaly_agg` returns one entry per input row, and `aid_anomaly_by` stays
aligned with the source rows. An empty group returns `NULL`.

## Use Cases

- **Inventory management**: Identify stockout patterns
- **Product lifecycle**: Detect new/obsolete products
- **Demand forecasting**: Choose appropriate models based on pattern type
- **Data quality**: Find outliers in demand data
- **Supply chain**: Monitor for demand anomalies

## See Also

- [Diagnostics](../diagnostics/diagnostics.md) - Model diagnostics
- [Table Macros](../macros/table_macros.md) - All table macros
