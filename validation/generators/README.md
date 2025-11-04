# Test Data Generators

This directory contains scripts that generate test data for validating the extension.

## Purpose

These scripts generate reference data that SQL tests use to validate extension functionality.
They should only be run when:
- Adding new test cases
- Updating test data
- Fixing bugs in data generation logic
- Updating statistical computations

## Dependencies

### R Scripts
- R (>= 4.0.0)
- Required packages:
  - `jsonlite` - JSON file I/O
  - `glmnet` - Ridge regression (for ridge tests)

Install R packages:
```r
install.packages(c("jsonlite", "glmnet"))
```

## Usage

### Generate all test data:
```bash
cd validation
./generate_all_data.sh
```

### Generate specific test data:
```bash
Rscript generators/generate_ols_tests.R
Rscript generators/generate_ridge_wls_tests.R
Rscript generators/generate_inference_tests.R
```

## Generator Scripts

### `generate_ols_tests.R`
Generates test data for OLS (Ordinary Least Squares) regression validation.

**Output:** `test/data/ols_tests/`

**Test cases:**
1. Simple linear regression (1 predictor)
2. Multiple regression (3 predictors)
3. No intercept regression
4. Rank-deficient matrix (constant feature)
5. Perfect collinearity

**Reference:** R `lm()` function

### `generate_ridge_wls_tests.R`
Generates test data for Ridge regression and Weighted Least Squares validation.

**Output:** `test/data/ridge_tests/` and `test/data/wls_tests/`

**Ridge test cases:**
1. Ridge with lambda = 0.1 (light regularization)
2. Ridge with lambda = 1.0 (stronger regularization)

**WLS test cases:**
1. Equal weights (should match OLS)
2. Inverse variance weights (for heteroscedastic data)

**Reference:** R `glmnet` (ridge) and `lm()` with weights (WLS)

### `generate_inference_tests.R`
Generates test data for statistical inference and prediction intervals.

**Output:** `test/data/inference_tests/`

**Test cases:**
1. Simple linear inference (coefficient tests, p-values, confidence intervals)
2. Multiple regression inference
3. Prediction intervals (confidence and prediction bands)

**Reference:** R `lm()` with `confint()` and `predict()`

## Adding New Tests

To add a new test generator:

1. Create `generators/generate_<test_name>.R`
2. Follow the template structure:
   ```r
   #!/usr/bin/env Rscript
   library(jsonlite)

   set.seed(42)  # For reproducibility

   # Create output directories
   test_dir <- "test/data/<test_name>"
   dir.create(file.path(test_dir, "input"), recursive = TRUE, showWarnings = FALSE)
   dir.create(file.path(test_dir, "expected"), recursive = TRUE, showWarnings = FALSE)

   # Generate input data
   # ... your data generation code ...
   write.csv(input_data, file.path(test_dir, "input/data.csv"), row.names = FALSE)

   # Compute expected results with R
   # ... your R computation ...
   write_json(expected_results, file.path(test_dir, "expected/results.json"),
              auto_unbox = TRUE, pretty = TRUE, digits = 15)

   # Write metadata
   metadata <- list(generated_at = Sys.time(), seed = 42, ...)
   write_json(metadata, file.path(test_dir, "metadata.json"), pretty = TRUE)
   ```
3. Test your generator: `Rscript generators/generate_<test_name>.R`
4. Create corresponding SQL validation test in `test/sql/validate_<test_name>.sql`
5. Create test documentation in `test/data/<test_name>/README.md`
6. Update this README with the new generator

## Generated Data Structure

Each test creates a structured output:

```
test/data/<test_name>/
├── input/              # Input data files (CSV format)
│   ├── test_case_1.csv
│   └── test_case_2.csv
├── expected/           # Expected output files (JSON format with high precision)
│   ├── test_case_1.json
│   └── test_case_2.json
├── metadata.json       # Generation metadata (timestamp, R version, seed, etc.)
└── README.md          # Test case documentation
```

## Important Notes

- **Deterministic generation:** Always use `set.seed(42)` for reproducibility
- **High precision:** Save expected results with `digits = 15` to avoid rounding errors
- **JSON format:** Use `auto_unbox = TRUE` for cleaner JSON output
- **Metadata:** Always include generation metadata (timestamp, R version, seed)
- **Documentation:** Document what each test case validates
- **Version control:** Generated data MUST be committed to git

## Troubleshooting

### "package 'glmnet' not found"
```bash
R
> install.packages("glmnet")
```

### "Error: cannot create directory"
Make sure you run scripts from the project root or use absolute paths.

### "Different results when regenerating"
Check that you're using `set.seed(42)` before any random number generation.

## Testing the Generated Data

After generating data, test it with:
```bash
# Run SQL validation tests
make test-validation

# Or test specific validation
scripts/test_sql_validation.sh
```

## References

- R documentation: `?lm`, `?glmnet`, `?predict`, `?confint`
- JSON format: Uses `jsonlite` package
- Reproducibility: Fixed seed (42) for all random generation
