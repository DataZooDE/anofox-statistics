#!/usr/bin/env python3
"""Compare the benchmark workloads across DuckDB CLIs (e.g. v1.5.6 vs v2.0, #269).

Runs every workload in bench/workloads/ with each CLI, repeats it --runs times and
reports the median wall-clock time ("Run Time (s): real") of each timed statement
as a markdown table, with the ratio of each CLI to the first one.

    python3 scripts/bench_compare.py --cli v1.5.6=/path/duckdb156 --cli v2.0.0=/path/duckdb200 \\
        [--extension PATH | --install-from https://get.erpl.io] [--runs 3] [--workloads 04,05]

Extension source:
  --extension PATH       LOAD a local build (each CLI must match the build's DuckDB version,
                         so this is mostly for a single CLI).
  --install-from URL     FORCE INSTALL anofox_statistics FROM URL once per CLI, then LOAD it.
                         Each CLI gets the release built for its own DuckDB version.
  (neither)              LOAD the already installed extension.

Timed statements are labelled with `.print @@ <label>` lines in the workload files;
an unlabelled timed statement is named after its file. The Parquet inputs of the
workloads 04-06 are generated once into bench/data/ (setup-parquet.sql).
"""
import argparse
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKLOADS = REPO / "bench" / "workloads"
RUN_TIME = re.compile(r"Run Time \(s\): real ([0-9.]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


EXT_DIR = REPO / "bench" / "results" / "extensions"


def run_cli(cli, load_sql, sql_file):
    env = dict(os.environ)
    env.pop("AI_AGENT", None)  # DuckDB 2.0's CLI changes its output format for agents
    # Installed extensions go to a private directory, never the user's ~/.duckdb.
    setup = f"SET extension_directory = '{EXT_DIR}';"
    cmd = [cli, "-unsigned", "-list", "-noheader", "-cmd", setup]
    if load_sql:
        cmd += ["-cmd", load_sql]
    cmd += ["-f", str(sql_file)]
    out = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    if out.returncode != 0 or "Error" in out.stderr:
        sys.exit(f"{cli} failed on {sql_file.name}:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
    return ANSI.sub("", out.stdout)


def timings(stdout, default_label):
    """[(label, seconds)] for the timed statements, in order."""
    labelled = "\n@@ " in "\n" + stdout
    result, label, k = [], None, 0
    for line in stdout.splitlines():
        if line.startswith("@@ "):
            label = line[3:].strip()
            continue
        m = RUN_TIME.search(line)
        if not m:
            continue
        if label is not None:
            result.append((label, float(m.group(1))))
            label = None
        elif not labelled:
            k += 1
            result.append((f"{default_label} #{k}", float(m.group(1))))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli", action="append", required=True, help="LABEL=PATH of a DuckDB CLI")
    ap.add_argument("--extension", help="path of a local extension build to LOAD")
    ap.add_argument("--install-from", help="repository to FORCE INSTALL the extension from")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workloads", help="comma-separated workload prefixes (default: all except -1m)")
    args = ap.parse_args()

    clis = [c.split("=", 1) for c in args.cli]
    files = sorted(p for p in WORKLOADS.glob("[0-9]*.sql") if "-1m" not in p.name)
    if args.workloads:
        wanted = args.workloads.split(",")
        files = [p for p in files if any(p.name.startswith(w) for w in wanted)]

    load = f"LOAD '{args.extension}';" if args.extension else "LOAD anofox_statistics;"
    if args.install_from:
        for label, cli in clis:
            run_cli(cli, None, _tmp_sql(f"FORCE INSTALL anofox_statistics FROM '{args.install_from}';"))

    if not (REPO / "bench" / "data" / "panel.parquet").exists():
        print("generating bench/data ...", file=sys.stderr)
        (REPO / "bench" / "data").mkdir(parents=True, exist_ok=True)
        run_cli(clis[0][1], None, WORKLOADS / "setup-parquet.sql")

    versions = {label: run_cli(cli, load, _tmp_sql(
        "SELECT version() || ' / ' || extension_version FROM duckdb_extensions() "
        "WHERE extension_name = 'anofox_statistics';")).strip().splitlines() for label, cli in clis}

    rows = {}  # (file, label) -> {cli_label: [seconds]}
    order = []
    for f in files:
        for label, cli in clis:
            for _ in range(args.runs):
                for name, sec in timings(run_cli(cli, load, f), f.stem):
                    key = (f.stem, name)
                    if key not in rows:
                        rows[key] = {}
                        order.append(key)
                    rows[key].setdefault(label, []).append(sec)

    labels = [label for label, _ in clis]
    print(f"Median of {args.runs} runs, wall-clock seconds.\n")
    for label in labels:
        print(f"- {label}: DuckDB / anofox_statistics = {versions[label][-1]}")
    print()
    head = ["Workload", "Statement"] + labels + [f"{l} / {labels[0]}" for l in labels[1:]]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for key in order:
        med = [statistics.median(rows[key][l]) if l in rows[key] else float("nan") for l in labels]
        ratios = [f"{m / med[0]:.2f}" for m in med[1:]]
        print("| " + " | ".join([key[0], key[1]] + [f"{m:.3f}" for m in med] + ratios) + " |")


def _tmp_sql(sql):
    p = REPO / "bench" / "results" / "_stmt.sql"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(sql + "\n")
    return p


if __name__ == "__main__":
    main()
