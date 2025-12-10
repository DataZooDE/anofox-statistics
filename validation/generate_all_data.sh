#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "🔄 Regenerating all test data..."
echo ""

# Check for required tools
check_dependency() {
    if ! command -v "$1" &> /dev/null; then
        echo "❌ Error: $1 is not installed"
        return 1
    fi
    echo "✅ Found: $1"
}

echo "📋 Checking dependencies..."
MISSING_R=0
check_dependency Rscript || MISSING_R=1

if [ "${MISSING_R}" -eq 1 ]; then
    echo ""
    echo "❌ Missing R. Please install R first."
    echo "   On Ubuntu/Debian: sudo apt-get install r-base"
    echo "   On macOS: brew install r"
    exit 1
fi

# Set up R library path (use local validation R_libs)
export R_LIBS="$SCRIPT_DIR/R_libs:${R_LIBS:-}"

# Check for required R packages and install if missing
echo ""
echo "📦 Checking and installing R packages..."
Rscript -e "
# Use local library
local_lib <- file.path('$SCRIPT_DIR', 'R_libs')
if (!dir.exists(local_lib)) {
  dir.create(local_lib, recursive = TRUE)
}
.libPaths(c(local_lib, .libPaths()))

packages <- c('jsonlite', 'glmnet')
for (pkg in packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        cat('📥 Installing', pkg, '...\n')
        install.packages(pkg, repos='https://cloud.r-project.org', lib=local_lib, quiet=TRUE)
    } else {
        cat('✅', pkg, 'already installed\n')
    }
}
cat('\n✅ All required R packages ready\n')
"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Failed to install R packages"
    exit 1
fi

echo ""
echo "🔨 Running data generation scripts..."
echo ""

# Counter for statistics
TOTAL=0
FAILED=0

# Run all R generators
for script in "$SCRIPT_DIR"/generators/*.R; do
    if [ -f "$script" ]; then
        TOTAL=$((TOTAL + 1))
        echo "📊 Running: $(basename "$script")"
        if Rscript "$script"; then
            echo ""
        else
            echo "❌ Failed: $(basename "$script")"
            FAILED=$((FAILED + 1))
            echo ""
        fi
    fi
done

echo "================================"
echo "Data Generation Summary"
echo "================================"
echo "Total scripts: $TOTAL"
echo "Successful: $((TOTAL - FAILED))"
echo "Failed: $FAILED"
echo "================================"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "❌ Some data generation scripts failed"
    exit 1
fi

echo ""
echo "✅ All test data regenerated successfully!"
echo ""
echo "📝 Note: The generated data in test/data/ should be committed to git"
echo "   so that regular test runs don't require R dependencies."
