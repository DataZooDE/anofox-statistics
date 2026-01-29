# AID (Automatic Identification of Demand)

AID provides demand pattern classification and anomaly detection for time series data. Useful for inventory management, supply chain analysis, and demand forecasting.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `aid_agg` | Aggregate | Classify demand patterns and detect anomalies |
| `aid_anomaly_agg` | Aggregate | Per-observation anomaly flags |
| `aid_anomaly_by` | Table Macro | Grouped anomaly detection with long-format output |

## aid_agg / anofox_stats_aid_agg

Classifies demand patterns as regular or intermittent, identifies best-fit distribution, and detects various anomaly patterns.

**Signature:**
```sql
aid_agg(y DOUBLE, [options MAP]) -> STRUCT
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| intermittent_threshold | DOUBLE | 0.3 | Zero proportion cutoff for intermittent classification |
| outlier_method | VARCHAR | 'zscore' | Outlier detection: 'zscore' (mean±3σ) or 'iqr' (1.5×IQR) |

**Returns:**
```
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
SELECT
    sku,
    (aid_agg(demand)).*
FROM sales
GROUP BY sku;

-- With custom threshold
SELECT aid_agg(demand, {'intermittent_threshold': 0.4})
FROM sales
WHERE sku = 'WIDGET001';

-- Using IQR-based outlier detection
SELECT aid_agg(demand, {'outlier_method': 'iqr'})
FROM inventory_data;
```

## aid_anomaly_agg / anofox_stats_aid_anomaly_agg

Returns per-observation anomaly flags for demand analysis. Maintains input order.

**Signature:**
```sql
aid_anomaly_agg(y DOUBLE, [options MAP]) -> LIST(STRUCT)
```

**Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| intermittent_threshold | DOUBLE | 0.3 | Zero proportion cutoff |
| outlier_method | VARCHAR | 'zscore' | Outlier detection: 'zscore' or 'iqr' |

**Returns:**
```
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
SELECT aid_anomaly_agg(demand)
FROM (VALUES (0), (0), (5), (0), (8), (0), (0)) AS t(demand);
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

## aid_anomaly_by

Table macro for grouped anomaly detection. See [Table Macros](../macros/table_macros.md#aid_anomaly_by) for details.

**Signature:**
```sql
aid_anomaly_by(
    source VARCHAR,           -- Table name
    group_col COLUMN,         -- Column to group by
    order_col COLUMN,         -- Column to order by within group
    y_col COLUMN,             -- Numeric column to analyze
    [options MAP]             -- Optional configuration
) -> TABLE
```

**Example:**
```sql
-- Get anomaly flags per product with dates
SELECT * FROM aid_anomaly_by('sales_data', product_id, sale_date, quantity, NULL);

-- Filter to stockouts only
SELECT group_id, order_value AS sale_date
FROM aid_anomaly_by('inventory', sku, period, demand, NULL)
WHERE stockout;
```

## Use Cases

- **Inventory management**: Identify stockout patterns
- **Product lifecycle**: Detect new/obsolete products
- **Demand forecasting**: Choose appropriate models based on pattern type
- **Data quality**: Find outliers in demand data
- **Supply chain**: Monitor for demand anomalies

## See Also

- [Diagnostics](../diagnostics/diagnostics.md) - Model diagnostics
- [Table Macros](../macros/table_macros.md#aid_anomaly_by) - Grouped anomaly detection
