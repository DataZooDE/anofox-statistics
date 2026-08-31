#!/usr/bin/env bash
#
# Phase-4 benchmark harness (PERF-01 / PERF-02).
#
# One documented command runs three representative workloads against the LOCAL
# release build of the extension and captures timings to a diffable results
# file under bench/results/. This is the measurement foundation used to produce
# before/after numbers for the FFI refactor (Plan 02) and hotspot work (Plan 03).
#
# Usage:
#   bash scripts/bench.sh          # default: scaled 10K-group / small workloads
#   bash scripts/bench.sh --full   # additionally run the 1M-group official variant
#
# The extension is loaded by explicit local path (-unsigned + LOAD), never via
# the autoloaded community build, so numbers reflect this working tree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DUCKDB="$REPO_ROOT/build/release/duckdb"
EXT="$REPO_ROOT/build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension"

# --- Parse flags (accepted in any position) -------------------------------
RUN_FULL=0
for arg in "$@"; do
  case "$arg" in
    --full) RUN_FULL=1 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "bench.sh: unknown argument '$arg' (see --help)" >&2
      exit 2
      ;;
  esac
done

# --- Preconditions: the local release build must exist --------------------
if [ ! -x "$DUCKDB" ]; then
  echo "ERROR: local DuckDB CLI not found or not executable: $DUCKDB" >&2
  echo "       Build it first with: make release" >&2
  exit 1
fi
if [ ! -f "$EXT" ]; then
  echo "ERROR: built extension not found: $EXT" >&2
  echo "       Build it first with: make release" >&2
  exit 1
fi

WORKLOADS_DIR="$REPO_ROOT/bench/workloads"
RESULTS_DIR="$REPO_ROOT/bench/results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUTFILE="$RESULTS_DIR/bench-$TIMESTAMP.md"

FAILED_WORKLOADS=()

# --- Header -----------------------------------------------------------------
{
  echo "# anofox-statistics benchmark run"
  echo
  echo "- Date: $(date -Iseconds)"
  echo "- DuckDB: \`$DUCKDB\`"
  echo "- Extension: \`$EXT\`"
  echo "- Mode: $([ "$RUN_FULL" -eq 1 ] && echo 'full (includes 1M-group variant)' || echo 'default (scaled)')"
  echo
} | tee "$OUTFILE"

# run_workload <display name> <sql file>
# Loads the local extension by explicit path, runs the timed workload, and
# appends the full process stdout+stderr (which carries the .timer lines) to
# the results file. We capture the process stream directly -- NOT via DuckDB
# .output -- because .timer writes to stdout regardless of .output.
run_workload() {
  local name="$1"
  local sql_file="$2"

  if [ ! -f "$sql_file" ]; then
    echo "ERROR: workload SQL not found: $sql_file" >&2
    exit 1
  fi

  {
    echo "## $name"
    echo
    echo '```'
  } | tee -a "$OUTFILE"

  # Capture per-workload exit status without aborting the whole run: a failing
  # workload is recorded and the harness continues to the next one.
  set +e
  "$DUCKDB" -unsigned -cmd "LOAD '$EXT';" -f "$sql_file" 2>&1 | tee -a "$OUTFILE"
  local status=${PIPESTATUS[0]}
  set -e

  {
    echo '```'
    if [ "$status" -ne 0 ]; then
      echo
      echo "> ⚠ workload exited with status $status"
      FAILED_WORKLOADS+=("$name (exit $status)")
    fi
    echo
  } | tee -a "$OUTFILE"
}

# --- Default workloads (scaled for fast iteration) --------------------------
run_workload "W1 — aggregate dispatch (10K groups / 1M rows)" "$WORKLOADS_DIR/01-agg-dispatch.sql"
run_workload "W2 — scalar/window fit_predict (10K groups / 1M rows)" "$WORKLOADS_DIR/02-fit-predict.sql"
run_workload "W3 — FFI marshalling micro-bench (compute_inference)" "$WORKLOADS_DIR/03-ffi-micro.sql"

# --- Optional full-scale official variant -----------------------------------
if [ "$RUN_FULL" -eq 1 ]; then
  run_workload "W1-full — aggregate dispatch (1M groups / 100M rows)" "$WORKLOADS_DIR/01-agg-dispatch-1m.sql"
fi

echo "Results written to: $OUTFILE"

if [ "${#FAILED_WORKLOADS[@]}" -ne 0 ]; then
  echo "WARNING: ${#FAILED_WORKLOADS[@]} workload(s) failed:" >&2
  for w in "${FAILED_WORKLOADS[@]}"; do echo "  - $w" >&2; done
  exit 1
fi
