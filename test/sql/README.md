# Test Organization

Tests are organized by function category, mirroring the API documentation structure.

## Directory Structure

```
test/sql/
├── aid/                    # AID demand classification tests
├── categorical/            # Chi-square, Fisher, G-test, McNemar
├── correlation/            # Pearson, Spearman, Kendall, ICC, distance
├── diagnostics/            # VIF, residuals diagnostics
├── distribution/           # Energy distance, MMD
├── equivalence/            # TOST tests
├── fit_predict/            # Window function tests
├── fit_predict_agg/        # Aggregate fit-predict tests
├── forecast/               # Diebold-Mariano, Clark-West
├── hypothesis_tests/       # t-test, ANOVA, Mann-Whitney, etc.
├── macros/                 # Table macro tests (*_fit_predict_by)
├── normality/              # Shapiro-Wilk, Jarque-Bera, D'Agostino
├── predict_agg/            # Legacy predict_agg tests (deprecated)
├── proportion/             # Proportion tests
├── regression/             # Core fit_agg tests, GLM tests
└── scalar/                 # Scalar function tests
```

## Documentation Mapping

| Docs Category | Test Directories |
|---------------|------------------|
| [Regression](../../docs/api/regression/) | regression/, fit_predict/, fit_predict_agg/, scalar/ |
| [GLM](../../docs/api/glm/) | regression/test_glm_fit_agg.test, fit_predict_agg/ |
| [Statistics](../../docs/api/statistics/) | hypothesis_tests/, categorical/, correlation/, normality/, proportion/, equivalence/, distribution/, forecast/ |
| [AID](../../docs/api/aid/) | aid/ |
| [Diagnostics](../../docs/api/diagnostics/) | diagnostics/, scalar/test_diagnostics_scalar.test |
| [Macros](../../docs/api/macros/) | macros/ |

## Running Tests

```bash
# Run all tests
make test

# Run specific category
duckdb -c ".read test/sql/correlation/test_pearson_agg.test"

# Run with DuckDB test runner
python scripts/run_tests.py test/sql/
```

## Test File Naming

- `test_<function_name>.test` - Tests for specific function
- `test_<category>_tests.test` - Tests for category of related functions
