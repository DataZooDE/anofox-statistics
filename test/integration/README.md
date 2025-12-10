# Integration Test Suites

These files contain comprehensive integration tests for the documentation guides:

- **test_all_guide_examples.sql** - Quick Start Guide test suite
- **test_technical_guide.sql** - Technical Guide test suite  
- **test_statistics_guide.sql** - Statistics Guide test suite
- **test_business_guide.sql** - Business Guide test suite
- **test_advanced_use_cases.sql** - Advanced Use Cases test suite

## Running Tests

```bash
# Run all tests
for test in test/integration/test_*.sql; do
  echo "Testing: $test"
  /tmp/duckdb -unsigned < "$test"
done

# Or run individually
/tmp/duckdb -unsigned < test/integration/test_business_guide.sql
```

## Purpose

These integration tests:
- Verify examples work together as complete workflows
- Include sample data setup with CREATE TABLE statements
- Test extension loading and configuration
- Provide smoke tests for entire guide sections

Note: The individual SQL examples are in `test/sql/` and are embedded into documentation.
