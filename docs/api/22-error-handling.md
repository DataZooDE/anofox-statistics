# Error Handling

Common errors and how to handle them.

## Insufficient Data

```
Error: Insufficient data for regression (n < p+1)
```

**Cause:** More features than observations.
**Solution:** Reduce features or use regularization (Ridge, Elastic Net).

## Singular Matrix

```
Error: Matrix is singular or nearly singular
```

**Cause:** Perfect multicollinearity among features.
**Solution:** Remove redundant features or check VIF values.

## Convergence Failure

```
Error: Failed to converge within max_iterations
```

**Cause:** Optimization didn't converge.
**Solution:** Increase `max_iterations` or adjust `tolerance`.

## Invalid Parameters

```
Error: alpha must be >= 0
Error: l1_ratio must be in [0, 1]
Error: quantile must be in (0, 1)
```

**Cause:** Parameter out of valid range.
**Solution:** Check parameter constraints in function documentation.

## NULL Handling

- NULL values in `y` or `x` are skipped
- Functions require at least `p + 1` non-NULL observations
- Use `COALESCE` or `WHERE` to handle NULLs explicitly
