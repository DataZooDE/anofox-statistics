# Explicit Priors and Laplace Intervals

Per-coefficient Gaussian or Laplace priors on any GLM, with standard errors taken
from the curvature of the log posterior at the mode.

This is regularized estimation with honest intervals — a MAP estimate plus its
observed information. There is no sampling and no change to the execution model.

## Functions

Priors are not a separate function. They ride inside the options MAP of the
existing GLM aggregates:

| Function | Type | Description |
|----------|------|-------------|
| `poisson_fit_agg` | Aggregate | Poisson GLM |
| `binomial_fit_agg` | Aggregate | Binomial GLM |
| `negbinom_fit_agg` | Aggregate | Negative Binomial GLM |
| `tweedie_fit_agg` | Aggregate | Tweedie GLM |
| `gamma_fit_agg` | Aggregate | Gamma GLM |
| `logistic_fit_agg` | Aggregate | Logistic regression |
| `aft_fit_agg` | Aggregate | AFT survival regression |

## Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| feature_names | VARCHAR[] | — | Names for the `x` columns, in order. Required to address a prior by name. |
| prior | MAP | — | Per-feature priors, keyed by name |
| vcov | VARCHAR | 'laplace' | Covariance type: `laplace`, `sandwich`, `naive` |
| glm_lambda | DOUBLE | 0.0 | Uniform ridge; equivalent to `normal(0, 1/sqrt(lambda))` on every non-intercept coefficient |

`feature_names` exists because the aggregate signature only ever sees
`x LIST(DOUBLE)` — there is nowhere else a name could come from. Names are
resolved to column positions before the fit; they never reach the numeric core.

## Prior syntax

The canonical entry is a `STRUCT(dist, loc, scale)`:

```sql
SELECT poisson_fit_agg(y, [x1, x2], {
    'feature_names': ['x1', 'x2'],
    'prior': MAP {
        'x1':       {'dist': 'normal',  'loc': 0.0, 'scale': 1.0},
        'x2':       {'dist': 'laplace', 'loc': 0.0, 'scale': 0.5},
        '_default': {'dist': 'normal',  'loc': 0.0, 'scale': 2.5}
    },
    'compute_inference': true
}) FROM t;
```

A DuckDB `MAP` requires a single value type, so the shorthand
`{'normal': [loc, scale]}` can only be used when every entry names the same
family:

```sql
'prior': MAP {'x1': {'normal': [0.0, 1.0]}, 'x2': {'normal': [0.0, 0.5]}}
```

Reserved keys:

| Key | Meaning |
|-----|---------|
| `_default` | Applies to every feature without its own entry |
| `(Intercept)` | The intercept, which is otherwise unpenalized |

Distribution names: `normal` (or `gaussian`), `laplace` (or `l1`, `lasso`),
`flat` (or `none`). `scale` is the prior standard deviation for a normal prior
and the scale parameter `b` for a Laplace prior; it must be strictly positive.

**Unknown prior keys raise an error** rather than being ignored. The rest of the
options MAP tolerates unknown keys for forward compatibility, but a silently
dropped prior would change the estimate with no signal at all.

## Covariance type

For an unpenalized fit all three coincide. They differ once a prior or
`glm_lambda` is in play:

| `vcov` | Formula | Meaning |
|--------|---------|---------|
| `laplace` | `(X'WX + P)^-1` | Curvature of the log posterior at the mode. The correct observed information for a MAP estimate, and the default. |
| `sandwich` | `(X'WX + P)^-1 X'WX (X'WX + P)^-1` | Frequentist sampling variance of the penalized estimator. |
| `naive` | `(X'WX)^-1` | Ignores the penalty. |

> **Behaviour change.** Before this feature, penalized fits (`glm_lambda > 0`)
> reported standard errors computed from the unpenalized `X'WX` — that is,
> `naive` — which is simply wrong for a MAP estimate. The default is now
> `laplace`. Pass `'vcov': 'naive'` to recover the old numbers.

## Laplace (L1) priors

A Laplace prior is an L1 penalty. Its mode is non-differentiable at the prior
location, so a coefficient shrunk exactly to that location has **no
curvature-based standard error** and comes back as `NaN`, alongside `NaN` for its
z-value, p-value and interval bounds. This matches how a dropped constant column
already surfaces.

The fit itself uses proximal coordinate descent, so exact zeros are reachable:

```sql
-- x2 is driven to exactly 0
SELECT (poisson_fit_agg(y, [x1, x2], {
    'feature_names': ['x1','x2'],
    'prior': MAP {'x2': {'dist':'laplace','loc':0.0,'scale':0.001}}
})).coefficients[2] FROM t;
```

## Interpreting a prior

A `normal(loc, scale)` prior pulls the coefficient toward `loc`, with strength
`1 / scale^2`. A wide scale is almost no constraint; a narrow one pins the
coefficient near `loc`.

Priors also compose additively with `glm_lambda`: both are Gaussian precisions on
the same coefficients.

## Use Cases

- **Ill-conditioned designs** — a weakly informative prior stabilises the fit
  without the arbitrariness of dropping columns.
- **Sparse groups** — a prior centred on a pooled estimate is a cheap way to
  borrow strength; see [Empirical-Bayes shrinkage](eb_shrink.md) and
  [Mixed-effects GLMs](glmm.md) for the principled versions.
- **Encoding domain knowledge** — pin a coefficient near a value the business
  already knows, rather than letting a small sample overrule it.
- **Feature selection** — a Laplace prior drives uninformative coefficients to
  exactly zero.

## See Also

- [Poisson GLM](poisson.md)
- [AFT survival regression](../survival/aft.md) — priors work there too
- [Mixed-effects GLMs](glmm.md) — a random effect is the same object as a Gaussian prior
