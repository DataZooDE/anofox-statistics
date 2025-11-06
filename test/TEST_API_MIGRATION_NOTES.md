# Test API Migration Notes

## Residual Diagnostics API Change

The `anofox_statistics_residual_diagnostics` function API has been simplified:

### Old API (v1.3)
```sql
residual_diagnostics(
    y DOUBLE[],              -- Actual values
    x DOUBLE[][],            -- Feature matrix
    add_intercept BOOLEAN,   -- Whether to add intercept
    outlier_threshold DOUBLE,-- Threshold for outliers (std deviations)
    influence_threshold DOUBLE -- Threshold for high influence points
)
```

**Old API performed OLS fitting internally** and returned:
- `obs_id`, `residual`, `leverage`, `std_residual`, `cooks_d`, `dffits`, `is_outlier`, `is_influential`

### New API (v1.4.1)
```sql
anofox_statistics_residual_diagnostics(
    y_actual DOUBLE[],       -- Actual values
    y_predicted DOUBLE[],    -- Predicted values (from any model)
    outlier_threshold DOUBLE -- Threshold for outliers (std deviations)
)
```

**New API expects pre-computed predictions** and returns:
- `obs_id`, `residual`, `std_residual`, `is_outlier`

### Migration Strategy

For tests using the old API, you need to:

1. **Option A: Fit model first, then compute diagnostics**
   ```sql
   WITH model AS (
       SELECT * FROM anofox_statistics_ols([y_values], [x_matrix], MAP{})
   ),
   predictions AS (
       -- Manually compute predictions using coefficients
       SELECT ...
   )
   SELECT * FROM anofox_statistics_residual_diagnostics(
       y_actual, y_predicted, 2.5
   );
   ```

2. **Option B: Use the new aggregate function**
   ```sql
   WITH data AS (
       SELECT category, y_actual, y_predicted
       FROM ...
   )
   SELECT category,
          anofox_statistics_residual_diagnostics_agg(
              y_actual, y_predicted, MAP{'outlier_threshold': 2.5}
          ) as diagnostics
   FROM data
   GROUP BY category;
   ```

3. **Option C: Use OLS inference function for leverage/influence**
   ```sql
   -- For leverage, Cook's D, DFFITS, use:
   SELECT * FROM anofox_statistics_ols_inference(y, x, MAP{});
   ```

## Files Requiring API Updates

The following test files currently use the OLD API and will fail until updated:

1. `test/integration/test_all_guide_examples.sql` - 2 occurrences
2. `test/integration/test_statistics_guide.sql` - 1 occurrence
3. `test/sql/guide01_example_7_detect_outliers.sql` - 1 occurrence
4. `test/sql/guide03_leverage_and_influence.sql` - 1 occurrence
5. `test/sql/guide03_residual_analysis.sql` - 1 occurrence
6. `test/sql/guide05_1_always_validate_assumptions.sql` - 1 occurrence
7. `test/sql/readme_model_diagnostics.sql` - 1 occurrence

## Recommended Actions

1. Review each test file listed above
2. Determine the intent of the test (outlier detection, leverage, influence, etc.)
3. Update to use appropriate new function:
   - Simple residual/outlier checks → `anofox_statistics_residual_diagnostics`
   - Leverage/influence metrics → `anofox_statistics_ols_inference`
   - Group-wise diagnostics → `anofox_statistics_residual_diagnostics_agg`

## Files Already Updated

The following test files have been successfully updated:

- `test/rank_deficiency_simple_test.sql` - ✅ Updated to new API
- `test/rank_deficiency_comprehensive_test.sql` - ✅ Updated to new API
- `test/sql/elastic_net_and_diagnostics_test.sql` - ✅ New comprehensive test file created

## Function Renames Completed

All instances of the following functions have been renamed:

- `vif` → `anofox_statistics_vif` ✅
- `normality_test` → `anofox_statistics_normality_test` ✅
- `residual_diagnostics` → `anofox_statistics_residual_diagnostics` ✅
