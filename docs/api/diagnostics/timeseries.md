# Time-Series Residual Diagnostics

Checks for regressions fitted on ordered data (time-series, online and window
fits): residual autocorrelation and heteroskedasticity. Results match R's
`acf`, `pacf`, `Box.test` and `lmtest::dwtest` / `lmtest::bptest`.

## Functions

| Function | Type | Description | R reference |
|----------|------|-------------|-------------|
| `acf_agg` | Aggregate | Sample autocorrelation per lag, with the white-noise band | `acf(x, lag.max)` |
| `pacf_agg` | Aggregate | Sample partial autocorrelation per lag | `pacf(x, lag.max)` |
| `acf_by`, `pacf_by` | Table macro | One ACF / PACF curve per group (`curve` schema) | |
| `ljung_box_agg` | Aggregate | Ljung-Box portmanteau test | `Box.test(x, lag, "Ljung-Box", fitdf)` |
| `box_pierce_agg` | Aggregate | Box-Pierce portmanteau test | `Box.test(x, lag, "Box-Pierce", fitdf)` |
| `durbin_watson_agg` | Aggregate | Durbin-Watson test (statistic; with `x`, the p-value) | `lmtest::dwtest(fit)` |
| `breusch_pagan_agg` | Aggregate | Breusch-Pagan heteroskedasticity test | `lmtest::bptest(fit)` |
| `white_test_agg` | Aggregate | White's heteroskedasticity test | `bptest(fit, ~ x1 * x2 + I(x1^2) + I(x2^2))` |

**Row order.** `acf_agg`, `pacf_agg`, `ljung_box_agg`, `box_pierce_agg` and
`durbin_watson_agg` read the rows in the order of the aggregate's own
`ORDER BY`: always write `agg(resid ORDER BY t)`, with the constant arguments
before the `ORDER BY` (`acf_agg(resid, 12 ORDER BY t)`; in `acf_agg(resid ORDER BY t, 12)`
the 12 would be a second sort key). Without it DuckDB may feed the rows in any order.

**NULLs.** Rows with a NULL (or NaN) value are skipped, so lags are counted over
the remaining rows. Empty input, or a series no longer than the requested lag,
gives NULL. Constant or all-zero residuals give NULL values rather than an error.

**Constant arguments.** `max_lag`, `lag`, `fitdf`, `studentize`, `alternative` and
`fit_intercept` must be constants.

The tests return the unified test fields (`method, alternative, statistic, df1,
df2, p_value, estimate, conf_low, conf_high, conf_level, effect_size,
effect_name, n`; see [Hypothesis tests](../statistics/hypothesis.md)).

The examples on this page use this table of regression residuals:

```sql
CREATE OR REPLACE TABLE fit_resid AS
SELECT t, store, x1, x2,
       sin(t * 0.8) * (0.5 + 0.1 * x1) + 0.3 * cos(t * 0.3) AS resid
FROM (
    SELECT i // 2 AS t, 'store_' || (i % 2) AS store,
           ((i * 7) % 10)::DOUBLE AS x1, ((i * 3) % 5)::DOUBLE AS x2
    FROM range(120) r(i)
);
```

## acf_agg / pacf_agg

```sql skip
acf_agg(x DOUBLE [, max_lag BIGINT] ORDER BY t)
    -> STRUCT(lag BIGINT, acf DOUBLE, conf_low DOUBLE, conf_high DOUBLE)[]
pacf_agg(x DOUBLE [, max_lag BIGINT] ORDER BY t)
    -> STRUCT(lag BIGINT, pacf DOUBLE, conf_low DOUBLE, conf_high DOUBLE)[]
```

`acf_agg` returns lags `0..max_lag` (lag 0 is 1), `pacf_agg` lags `1..max_lag`.
`max_lag` defaults to R's `floor(10 * log10(n))` (also when NULL).
`conf_low` / `conf_high` are the 95% white-noise band `-/+ qnorm(0.975) / sqrt(n)`
that R's `plot.acf` draws; lags outside it suggest remaining autocorrelation.

```sql
SELECT store, unnest(r, recursive := true)
FROM (SELECT store, acf_agg(resid, 5 ORDER BY t) AS r FROM fit_resid GROUP BY store)
ORDER BY store;
```

## acf_by / pacf_by

```sql skip
acf_by(source, group_col, value_col, order_col [, max_lag := NULL])
pacf_by(source, group_col, value_col, order_col [, max_lag := NULL])
```

One curve per group in the integration contract's `curve` schema: the group
column, `model_id`, `series` (NULL), `curve_type` (`'acf'` / `'pacf'`), `x` (lag),
`y`, `y_low` / `y_high` (the white-noise band), `threshold`, `weight`,
`n_censor` (NULL).

```sql
SELECT * FROM pacf_by('fit_resid', store, resid, t, max_lag := 6);
```

## ljung_box_agg / box_pierce_agg

```sql skip
ljung_box_agg(x DOUBLE [, lag BIGINT [, fitdf BIGINT]] ORDER BY t)  -> STRUCT(lag, fitdf, <unified test fields>)
box_pierce_agg(x DOUBLE [, lag BIGINT [, fitdf BIGINT]] ORDER BY t) -> STRUCT(lag, fitdf, <unified test fields>)
```

Portmanteau tests of the first `lag` autocorrelations: Ljung-Box
`Q = n (n + 2) sum r_k^2 / (n - k)`, Box-Pierce `Q = n sum r_k^2`, chi-squared
with `df1 = lag - fitdf`. As in R, `lag` defaults to 1 and `fitdf` to 0; for the
residuals of an ARMA(p, q) model pass `fitdf = p + q`.

```sql
SELECT store, (ljung_box_agg(resid, 10 ORDER BY t)).p_value AS lb_p,
       (box_pierce_agg(resid, 10 ORDER BY t)).p_value AS bp_p
FROM fit_resid GROUP BY store ORDER BY store;
```

## durbin_watson_agg

```sql skip
durbin_watson_agg(resid DOUBLE ORDER BY t) -> STRUCT(<unified test fields>)
durbin_watson_agg(resid DOUBLE, x DOUBLE[] [, alternative VARCHAR [, fit_intercept BOOLEAN]] ORDER BY t)
    -> STRUCT(<unified test fields>)
```

`statistic = sum (e_t - e_{t-1})^2 / sum e_t^2`, between 0 and 4; values near 2
mean no first-order autocorrelation, values well below 2 positive autocorrelation.

The null distribution depends on the regression's design matrix, so the
p-value needs the regressors: pass the model's `x` (the same columns as the fit;
an intercept column is added unless `fit_intercept` is false, and `[]::DOUBLE[]`
is an intercept-only model). The p-value is that of R `lmtest::dwtest`: exact for
n < 100 (computed with Imhof's method), otherwise the normal approximation.
`alternative` is `'greater'` (default, positive autocorrelation), `'less'` or
`'two_sided'`. With the residuals alone, `p_value` is NULL.

R's exact method (a 15-term series) can fail for some designs, and `dwtest` then
falls back to the normal approximation; the extension returns the exact
p-value in those cases, so it can differ from R there. The exact p-value is
computed to an absolute accuracy of about 1e-11, so p-values smaller than that
are reliable only as "below 1e-11".

```sql
SELECT store, (durbin_watson_agg(resid ORDER BY t)).statistic AS dw,
       (durbin_watson_agg(resid, [x1, x2] ORDER BY t)).p_value AS p_value
FROM fit_resid GROUP BY store ORDER BY store;
```

## breusch_pagan_agg / white_test_agg

```sql skip
breusch_pagan_agg(resid DOUBLE, x DOUBLE[] [, studentize BOOLEAN]) -> STRUCT(<unified test fields>)
white_test_agg(resid DOUBLE, x DOUBLE[])                           -> STRUCT(<unified test fields>)
```

Score tests for heteroskedasticity: the squared residuals of a linear fit with
an intercept are regressed on `x` (Breusch-Pagan) or on `x`, its squares and
pairwise cross-products (White). `df1` is the rank of these auxiliary regressors.
`studentize` (default `true`) is Koenker's robust variant, `bptest`'s default;
`false` gives the original Breusch-Pagan (1979) statistic. Pass the residuals of
a model that has an intercept and the same regressors (row order does not matter).

```sql
SELECT (breusch_pagan_agg(resid, [x1, x2])).p_value AS bp_p,
       (breusch_pagan_agg(resid, [x1, x2], false)).statistic AS bp_1979,
       (white_test_agg(resid, [x1, x2])).p_value AS white_p
FROM fit_resid;
```

## See Also

- [Diagnostics](diagnostics.md) - VIF, residual diagnostics, Jarque-Bera
- [Hypothesis tests](../statistics/hypothesis.md) - the unified test fields
