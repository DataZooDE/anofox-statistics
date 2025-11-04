#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Configuration
EXTENSION_NAME="${EXTENSION_NAME:-anofox_statistics}"
BUILD_DIR="${BUILD_DIR:-./build/release}"
SQL_VALIDATION_DIR="${PROJECT_ROOT}/test/sql"

echo "🧪 Running SQL validation tests..."
echo ""

# Check if test data exists
if [ ! -d "${PROJECT_ROOT}/test/data" ]; then
    echo "❌ Test data not found in test/data/"
    echo "   Run 'make generate-test-data' or './validation/generate_all_data.sh' first"
    exit 1
fi

# Check if extension exists
EXTENSION_PATH="${BUILD_DIR}/extension/${EXTENSION_NAME}/${EXTENSION_NAME}.duckdb_extension"
if [ ! -f "$EXTENSION_PATH" ]; then
    echo "❌ Extension not found at: $EXTENSION_PATH"
    echo "   Build the extension first with: make release"
    exit 1
fi

echo "✅ Extension found: $EXTENSION_PATH"
echo "✅ Test data found: ${PROJECT_ROOT}/test/data/"
echo ""

# Test a SQL validation file
test_validation_file() {
    local sql_file=$1
    local filename=$(basename "$sql_file")

    # Skip non-validation files
    if [[ ! $filename =~ ^validate_ ]]; then
        return 0
    fi

    # Create temporary file with extension setup
    local test_file=$(mktemp)

    cat > "$test_file" <<EOF
-- Load extension
.bail on
LOAD '${EXTENSION_PATH}';

-- Run validation test
EOF

    cat "$sql_file" >> "$test_file"

    # Run test
    echo -n "  Testing: $filename ... "
    if duckdb -unsigned :memory: < "$test_file" > /dev/null 2>&1; then
        echo "✅ PASS"
        rm "$test_file"
        return 0
    else
        echo "❌ FAIL"
        echo "     Error output:"
        duckdb -unsigned :memory: < "$test_file" 2>&1 | tail -n 30
        rm "$test_file"
        return 1
    fi
}

# Main test execution
TOTAL=0
FAILED=0

# Find and test all validation SQL files
while IFS= read -r sql_file; do
    TOTAL=$((TOTAL + 1))
    if ! test_validation_file "$sql_file"; then
        FAILED=$((FAILED + 1))
    fi
done < <(find "$SQL_VALIDATION_DIR" -name "validate_*.sql" -type f 2>/dev/null | sort)

if [ $TOTAL -eq 0 ]; then
    echo "⚠️  No validation tests found in $SQL_VALIDATION_DIR"
    echo "   Validation tests should be named: validate_*.sql"
    exit 1
fi

echo ""
echo "================================"
echo "Validation Tests Summary"
echo "================================"
echo "Total: $TOTAL"
echo "Passed: $((TOTAL - FAILED))"
echo "Failed: $FAILED"
echo "================================"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "❌ Some tests failed"
    exit 1
fi

echo ""
echo "✅ All validation tests passed!"
