# Migrating from 0.9.x to 0.10.0

Version 0.10.0 renames the public SQL API and makes option validation strict.
Most queries need only a search-and-replace. This page lists every change that can
break an existing query, derived from the differences between the `v0.9.0` and
`v0.10.0` registration and option-parsing code.

## 1. The `anofox_stats_` prefix is gone

In 0.9.x every function was registered twice: once with the `anofox_stats_` prefix
and once (for most functions) as a short alias. In 0.10.0 only the short,
unprefixed name exists, and there are no deprecated aliases.

| 0.9.x | 0.10.0 |
|-------|--------|
| `anofox_stats_ols_fit_agg(y, x)` | `ols_fit_agg(y, x)` |
| `anofox_stats_ridge_fit_predict_agg(y, x, opts)` | `ridge_fit_predict_agg(y, x, opts)` |
| `anofox_stats_t_test_agg(v, g)` | `t_test_agg(v, g)` |
| `anofox_stats_aft_cdf(...)` | `aft_cdf(...)` |
| `anofox_stats_vif(x)` | `vif(x)` |

The rule is mechanical: strip the leading `anofox_stats_`. With `sed`:

```bash
sed -i 's/\banofox_stats_\([a-z_0-9]*\)(/\1(/g' your_queries.sql
```

Calling a prefixed name in 0.10.0 fails with a "function does not exist" catalog
error.

## 2. Renamed functions

Besides the prefix, a few names changed:

| 0.9.x | 0.10.0 | Notes |
|-------|--------|-------|
| `theilsen_fit`, `theilsen_fit_agg`, `theilsen_fit_predict`, `theilsen_fit_predict_agg`, `theilsen_fit_predict_by` | `theil_sen_fit`, `theil_sen_fit_agg`, `theil_sen_fit_predict`, `theil_sen_fit_predict_agg`, `theil_sen_fit_predict_by` | Underscore added |
| `ols_predict_agg` | `ols_fit_predict_agg` | Deprecated alias removed |
| `ridge_predict_agg` | `ridge_fit_predict_agg` | Deprecated alias removed |
| `wls_predict_agg` | `wls_fit_predict_agg` | Deprecated alias removed |
| `rls_predict_agg` | `rls_fit_predict_agg` | Deprecated alias removed |
| `elasticnet_predict_agg` | `elasticnet_fit_predict_agg` | Deprecated alias removed |

The table macros keep their names (`ols_fit_predict_by`, `aid_by`, and so on),
except `theilsen_fit_predict_by`, which becomes `theil_sen_fit_predict_by`.

## 3. Unsupported option keys now raise an error

In 0.9.x a key the parser did not recognise was silently ignored. A typo such as
`{'fit_intercpt': false}` therefore fitted an intercept without any warning. In
0.10.0 these keys raise an error at bind time, and the message lists the valid
keys:

```text
Invalid Input Error: unknown option 'fit_intercpt'; valid keys: fit_intercept (alias: intercept), ...
```

This applies to the regression/GLM option parser and to every hypothesis-test
option parser (`t_test_agg`, `mann_whitney_u_agg`, `kendall_agg`, and the others).

The option key names themselves did not change between 0.9.x and 0.10.0. Queries
that used keys the code never read now fail instead of being silently ignored.
Common ones found in older examples:

| Key that was silently ignored | Use instead |
|-------------------------------|-------------|
| `huber_epsilon` | `epsilon` |
| `lower_bounds` / `upper_bounds` (lists) | `lower_bound` / `upper_bound` (scalars applied to every coefficient) |
| `fit_intercpt`, `confidence_lvl`, ... (typos) | the correctly spelled key |

Coming next: per-function key validation, so that a key belonging to a different
function (for example `alpha` passed to `ols_fit_agg`) is also rejected.
The rule from then on: **option keys a function does not support raise an error**.

## 4. Result field names

No result-struct field was renamed between 0.9.0 and 0.10.0. Some older
documentation used field names the extension never emitted. If you copied those,
use the real names:

| Name in older docs | Real field |
|--------------------|------------|
| `.r2` | `.r_squared` |
| `n_obs` | `n_observations` |
| `n_iterations` (GLMs) | `iterations` |

GLM inference returns `z_values` rather than `t_values`, because GLM Wald tests use
the normal distribution. Linear models return `t_values`.

## 5. Behaviour changes

- **Degenerate windows return NULL.** When `ols_fit_agg` is used as a window function
  and a frame has `n_observations <= n_features + 1` rows (or `<= n_features`
  without an intercept), the result is now NULL. Before, it was a saturated fit
  with NaN statistics. The window `*_fit_predict` functions already behaved this way.
- **Clearer errors.** Numerical failures (singular matrix, convergence failure) and
  invalid input now raise errors that start with the SQL function name, for
  example `ols_fit_agg: All rows filtered due to NULL/NaN values`. `ols_fit_agg` and
  `bls_fit_agg` used to return NULL for some of these cases, and now raise instead.
- **NNLS** (`nnls_fit_agg`) is now implemented as BLS with a lower bound of 0
  and no upper bound. Results are unchanged.

## 6. Checklist

1. Strip `anofox_stats_` from every function call.
2. Replace `theilsen_*` with `theil_sen_*` and `*_predict_agg` with `*_fit_predict_agg`.
3. Run your queries once. Any `unknown option` error points at a key that 0.9.x
   was silently ignoring. Fix the key name, or remove it.
4. If you read `.r2`, `.n_obs` or GLM `.n_iterations`, switch to the real field names.
