#!/bin/bash
# Run all fit_predict benchmarks with memory monitoring
# Usage: ./run_all_benchmarks.sh [path_to_duckdb]

set -e

DUCKDB=${1:-duckdb}
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

echo "=============================================="
echo "BENCHMARK: 1M Groups (100M Rows), 3 Features"
echo "=============================================="
echo ""
echo "DuckDB: $DUCKDB"
echo "Date: $(date -Iseconds)"
echo ""

run_benchmark() {
    local name=$1
    local sql_file=$2

    echo "=== $name ==="

    # Run benchmark in background for memory monitoring
    $DUCKDB < "$sql_file" &
    local pid=$!

    # Monitor peak RSS
    local peak_rss=0
    while [ -d /proc/$pid ] 2>/dev/null; do
        local rss=$(awk '/^VmRSS:/{print $2}' /proc/$pid/status 2>/dev/null || echo 0)
        if [ "$rss" -gt "$peak_rss" ] 2>/dev/null; then
            peak_rss=$rss
        fi
        sleep 0.5
    done

    wait $pid
    echo "Peak RSS: $((peak_rss / 1024)) MB"
    echo ""
}

run_benchmark "OLS" "$SCRIPT_DIR/benchmark_ols.sql"
run_benchmark "Ridge" "$SCRIPT_DIR/benchmark_ridge.sql"
run_benchmark "WLS" "$SCRIPT_DIR/benchmark_wls.sql"
run_benchmark "RLS" "$SCRIPT_DIR/benchmark_rls.sql"
run_benchmark "Elastic Net" "$SCRIPT_DIR/benchmark_elasticnet.sql"

echo "=============================================="
echo "Benchmark complete"
echo "=============================================="
