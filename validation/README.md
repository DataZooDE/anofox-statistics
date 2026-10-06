# Validation against R / scipy

The extension is validated against reference implementations (R `stats`, `MASS`,
`survival`, `lme4`, `quantreg`; Python `scipy`) through **sqllogictest files**
that run on every CI build. There are no separate validation runs to do by hand.

## Layout

| Path | Purpose |
|------|---------|
| `validation/generators/make_reference_tests.R` | Reference values for hypothesis tests, normality, correlation, categorical and proportion tests (base R). |
| `validation/generators/make_reference_tests.py` | Reference values only scipy provides (`normaltest`, `brunnermunzel`, `kendalltau`, ...). |
| `validation/generators/make_reference_regression.R` | Reference values for OLS/WLS (inference, HC1/HC3, intervals), GLMs, quantile, AFT and GLMM. |
| `test/sql/reference/*.test` | The sqllogictests that hold those values. Each value carries a comment naming the R/scipy call that produced it. |

Each generator builds small fixed datasets (the same ones inlined in the `.test`
files as `VALUES` or deterministic `range()` expressions) and prints the expected
numbers to full precision.

## Regenerating

```bash
Rscript validation/generators/make_reference_tests.R
python3 validation/generators/make_reference_tests.py
Rscript validation/generators/make_reference_regression.R
```

Paste changed values into the matching `test/sql/reference/*.test` file, then run:

```bash
build/release/test/unittest "test/sql/reference/*"
```

Required R packages: `MASS`, `survival`, `lme4`, `quantreg`, `jsonlite`
(`sandwich` is not required; HC estimators are computed by hand). Python: `scipy`, `numpy`.

## Conventions

* Tolerances are absolute and stated per assertion (normally 1e-9; looser only
  for iterative or approximate procedures, with the reason given in a comment).
* When the extension deliberately uses a different convention from R (for example
  centered R² for no-intercept models, or no Yates correction by default), the
  test says so in a comment and checks against a reference computed with the
  extension's convention.
