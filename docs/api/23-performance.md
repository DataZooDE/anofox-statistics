# Performance Notes

Tips for optimal performance.

## Aggregate vs Scalar Functions

- **Aggregate functions** (`*_agg`): Best for streaming data, GROUP BY, window functions
- **Scalar functions**: Best for batch operations with pre-collected arrays

## Memory Considerations

- Aggregate functions accumulate data incrementally
- For very large groups, consider sampling or chunking
- Window functions with large frames may require significant memory

## Parallelization

- DuckDB automatically parallelizes across partitions
- Use `PARTITION BY` in window functions for parallel execution
- Table macros leverage DuckDB's parallel execution

## Best Practices

1. **Use appropriate precision:** DOUBLE is usually sufficient
2. **Filter early:** Apply WHERE clauses before aggregation
3. **Limit features:** More features = more computation
4. **Use regularization:** Ridge/Elastic Net for high-dimensional data
5. **Consider RLS:** For streaming/adaptive scenarios

## Benchmarks

Typical performance on modern hardware (10M rows, 5 features):

| Operation | Time |
|-----------|------|
| ols_fit_agg | ~200ms |
| ridge_fit_agg | ~250ms |
| ols_fit_predict_agg | ~300ms |
| GROUP BY (1000 groups) | ~500ms |
