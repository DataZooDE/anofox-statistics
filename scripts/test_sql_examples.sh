#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration - adjust these to match your extension
EXTENSION_NAME="${EXTENSION_NAME:-anofox_statistics}"
BUILD_DIR="${BUILD_DIR:-./build/release}"
SQL_DIR="${PROJECT_ROOT}/test/sql"
DUCKDB_CLI="${DUCKDB_CLI:-/tmp/duckdb}"

# Test a SQL file
test_sql_file() {
    local sql_file=$1
    local filename=$(basename "$sql_file")

    # Create temporary file with extension setup
    local test_file=$(mktemp)

    # Check if SQL file already has a LOAD statement
    if grep -q "^LOAD.*${EXTENSION_NAME}" "$sql_file"; then
        # File has its own LOAD statement, just use it as-is
        cat > "$test_file" <<EOF
.bail on
EOF
        cat "$sql_file" >> "$test_file"
    else
        # File doesn't have LOAD statement, add one
        cat > "$test_file" <<EOF
-- Load extension if available
.bail on
LOAD '${BUILD_DIR}/extension/${EXTENSION_NAME}/${EXTENSION_NAME}.duckdb_extension';

-- Run actual SQL
EOF
        cat "$sql_file" >> "$test_file"
    fi

    # Run test
    if "$DUCKDB_CLI" -unsigned :memory: < "$test_file" > /dev/null 2>&1; then
        echo "  ✅ $filename"
        rm "$test_file"
        return 0
    else
        echo "  ❌ $filename FAILED"
        echo "     Error output:"
        "$DUCKDB_CLI" -unsigned :memory: < "$test_file" 2>&1 | head -n 20
        rm "$test_file"
        return 1
    fi
}

# Main function
main() {
    echo "🧪 Testing SQL example files..."
    echo ""

    # Check if DuckDB CLI exists
    if [ ! -f "$DUCKDB_CLI" ]; then
        echo "⚠️  DuckDB CLI not found at: $DUCKDB_CLI"
        echo "   Set DUCKDB_CLI environment variable or install DuckDB"
        exit 1
    fi

    # Check if extension exists
    if [ ! -f "${BUILD_DIR}/extension/${EXTENSION_NAME}/${EXTENSION_NAME}.duckdb_extension" ]; then
        echo "⚠️  Extension not found at: ${BUILD_DIR}/extension/${EXTENSION_NAME}/${EXTENSION_NAME}.duckdb_extension"
        echo "   Build the extension first with: make release"
        exit 1
    fi

    local total=0
    local failed=0

    # Test all SQL files
    while IFS= read -r sql_file; do
        ((total++))
        if ! test_sql_file "$sql_file"; then
            ((failed++))
        fi
    done < <(find "$SQL_DIR" -name "*.sql" -type f 2>/dev/null | sort)

    echo ""
    echo "================================"
    echo "Total: $total"
    echo "Passed: $((total - failed))"
    echo "Failed: $failed"
    echo "================================"

    if [ $failed -gt 0 ]; then
        exit 1
    fi
}

main "$@"
