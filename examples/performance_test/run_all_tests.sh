#!/bin/bash
# ============================================================================
# Master Script: Run All Performance Tests
# ============================================================================
# This script runs the complete performance testing suite:
# 1. Generates test data
# 2. Runs SQL tests (DuckDB)
# 3. Runs R tests
# 4. Compares results
#
# Prerequisites:
# - DuckDB with anofox_statistics extension loaded
# - R with arrow, dplyr, broom packages installed
#
# Usage:
#   ./examples/performance_test/run_all_tests.sh
# ============================================================================

set -e  # Exit on error

# Configuration
DUCKDB_CLI=${DUCKDB_CLI:-duckdb}
PERF_TEST_DIR="examples/performance_test"
DATA_DIR="${PERF_TEST_DIR}/data"
RESULTS_DIR="${PERF_TEST_DIR}/results"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================================"
echo "ANOFOX STATISTICS PERFORMANCE TEST SUITE"
echo "============================================================================"
echo ""

# ============================================================================
# Step 1: Create directories
# ============================================================================

echo -e "${BLUE}Step 1: Creating directories...${NC}"
mkdir -p "${DATA_DIR}"
mkdir -p "${RESULTS_DIR}"
echo "  ✓ Created ${DATA_DIR}"
echo "  ✓ Created ${RESULTS_DIR}"
echo ""

# ============================================================================
# Step 2: Generate test data
# ============================================================================

echo -e "${BLUE}Step 2: Generating test data...${NC}"
echo "Running: ${DUCKDB_CLI} --unsigned  < ${PERF_TEST_DIR}/generate_test_data.sql"
${DUCKDB_CLI} < "${PERF_TEST_DIR}/generate_test_data.sql"
echo -e "${GREEN}  ✓ Test data generated${NC}"
echo ""

# ============================================================================
# Step 3: Run SQL fit-predict tests
# ============================================================================

echo -e "${BLUE}Step 3: Running SQL fit-predict tests...${NC}"
echo "Running: ${DUCKDB_CLI} < ${PERF_TEST_DIR}/performance_test_ols_fit_predict.sql"
${DUCKDB_CLI} --unsigned < "${PERF_TEST_DIR}/performance_test_ols_fit_predict.sql"
echo -e "${GREEN}  ✓ SQL fit-predict tests completed${NC}"
echo ""

# ============================================================================
# Step 4: Run SQL aggregate tests
# ============================================================================

echo -e "${BLUE}Step 4: Running SQL aggregate tests...${NC}"
echo "Running: ${DUCKDB_CLI} < ${PERF_TEST_DIR}/performance_test_ols_aggregate.sql"
${DUCKDB_CLI} < "${PERF_TEST_DIR}/performance_test_ols_aggregate.sql"
echo -e "${GREEN}  ✓ SQL aggregate tests completed${NC}"
echo ""

# ============================================================================
# Step 5: Run R fit-predict tests
# ============================================================================

echo -e "${BLUE}Step 5: Running R fit-predict tests...${NC}"
if command -v Rscript &> /dev/null; then
    echo "Running: Rscript ${PERF_TEST_DIR}/performance_test_ols_fit_predict.R"
    Rscript "${PERF_TEST_DIR}/performance_test_ols_fit_predict.R"
    echo -e "${GREEN}  ✓ R fit-predict tests completed${NC}"
else
    echo -e "${YELLOW}  ⚠ Rscript not found, skipping R tests${NC}"
fi
echo ""

# ============================================================================
# Step 6: Run R aggregate tests
# ============================================================================

echo -e "${BLUE}Step 6: Running R aggregate tests...${NC}"
if command -v Rscript &> /dev/null; then
    echo "Running: Rscript ${PERF_TEST_DIR}/performance_test_ols_aggregate.R"
    Rscript "${PERF_TEST_DIR}/performance_test_ols_aggregate.R"
    echo -e "${GREEN}  ✓ R aggregate tests completed${NC}"
else
    echo -e "${YELLOW}  ⚠ Rscript not found, skipping R tests${NC}"
fi
echo ""

# ============================================================================
# Summary
# ============================================================================

echo "============================================================================"
echo "TEST SUITE COMPLETE"
echo "============================================================================"
echo ""
echo "Generated data files:"
ls -lh "${DATA_DIR}"/*.parquet 2>/dev/null || echo "  (no files found)"
echo ""
echo "Generated result files:"
ls -lh "${RESULTS_DIR}"/*.parquet 2>/dev/null || echo "  (no files found)"
echo ""
echo "Next steps:"
echo "  1. Compare SQL vs R results using DuckDB"
echo "  2. See ${PERF_TEST_DIR}/README_performance_tests.md for analysis examples"
echo "============================================================================"
