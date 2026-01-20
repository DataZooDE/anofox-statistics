# AID Functions

Automatic Identification of Demand patterns for inventory classification.

## aid_agg

Classify demand patterns using ADI (Average Demand Interval) and CV² (Squared Coefficient of Variation).

**Signature:**
```sql
aid_agg(demand DOUBLE, [options MAP]) -> STRUCT
```

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| classification | VARCHAR | 'smooth', 'erratic', 'intermittent', 'lumpy' |
| adi | DOUBLE | Average Demand Interval |
| cv2 | DOUBLE | Squared Coefficient of Variation |
| n | BIGINT | Number of observations |

**Example:**
```sql
-- Classify demand patterns per SKU
SELECT
    sku_id,
    (aid_agg(demand)).*
FROM sales
GROUP BY sku_id;
```

## aid_anomaly_agg

Detect anomalous demand patterns.

## Short Aliases

- `aid_agg` -> `anofox_stats_aid_agg`
- `aid_anomaly_agg` -> `anofox_stats_aid_anomaly_agg`
