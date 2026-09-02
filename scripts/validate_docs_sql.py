#!/usr/bin/env python3
"""Doc-SQL validation harness for the anofox-statistics DuckDB extension.

Extracts every fenced sql code block from the seven documentation files listed
in DOC_FILES, runs each file's blocks (concatenated in document order into a
single DuckDB session) against the locally-built extension, and reports pass or
fail per file.  Exits non-zero if any file's SQL fails.

Skip convention
---------------
A fenced block whose info-string carries the ``skip`` keyword is excluded from
validation::

    ```sql skip
    -- This block is intentional documentation of old API or migration guidance;
    -- it is not meant to be executed against the current extension.
    ```

Use the skip marker for illustrative examples that show deprecated names (e.g.
the Breaking Changes "Before" blocks in docs/API_CONVENTIONS.md), multi-step
examples that depend on external data not available in CI, or any block that
is correct prose but not intended as a runnable query.

Isolation note
--------------
The harness runs each file's blocks against a throwaway in-process DuckDB
(no ATTACH of persistent databases, no secrets in the environment, DuckDB has
no network access by default).  The ``--file`` flag rejects any path that
contains a parent-directory component (``..``) or that resolves outside the
repository root, so it cannot be pointed at arbitrary filesystem locations
(threat T-6-01 / T-6-01b mitigation).

Usage
-----
    python3 scripts/validate_docs_sql.py              # validate all 7 doc files
    python3 scripts/validate_docs_sql.py --file README.md   # single-file fast path
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# Paths — resolved from __file__ so the script works from any working directory
# ---------------------------------------------------------------------------

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUCKDB = os.path.join(REPO, "build", "release", "duckdb")
EXT = os.path.join(
    REPO,
    "build",
    "release",
    "extension",
    "anofox_statistics",
    "anofox_statistics.duckdb_extension",
)

# ---------------------------------------------------------------------------
# Documentation files in the extraction scope (DOCS-02)
# ---------------------------------------------------------------------------

DOC_FILES = [
    "README.md",
    "guides/01_quick_start.md",
    "guides/02_technical_guide.md",
    "guides/03_business_guide.md",
    "guides/04_advanced_use_cases.md",
    "docs/API_REFERENCE.md",
    "docs/API_CONVENTIONS.md",
]

# ---------------------------------------------------------------------------
# Regex for fenced sql blocks.
#
# Matches:
#   ```sql\n<body>\n```   → executable block  (group 1 is None / empty)
#   ```sql skip\n<body>\n``` → skip this block (group 1 == ' skip')
#
# Flags: MULTILINE so ^ matches line-start; DOTALL so . matches newlines in body.
# ---------------------------------------------------------------------------

BLOCK_RE = re.compile(
    r"^```sql( skip)?\n(.*?)^```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


def _check_preconditions() -> None:
    """Exit 1 with a diagnostic message if the local release build is missing.

    Mirrors the precondition checks in scripts/bench.sh (lines 44-53).
    """
    if not os.path.isfile(DUCKDB) or not os.access(DUCKDB, os.X_OK):
        print(
            f"ERROR: local DuckDB CLI not found or not executable: {DUCKDB}",
            file=sys.stderr,
        )
        print("       Build it first with: make release", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(EXT):
        print(
            f"ERROR: built extension not found: {EXT}",
            file=sys.stderr,
        )
        print("       Build it first with: make release", file=sys.stderr)
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate doc-file SQL blocks against the built anofox-statistics extension."
    )
    parser.add_argument(
        "--file",
        metavar="PATH",
        help=(
            "Restrict the run to a single doc file (relative to repo root or"
            " an absolute path inside the repo).  Used for fast single-file"
            " iteration during fix passes."
        ),
    )
    return parser.parse_args()


def _resolve_single_file(raw: str) -> str:
    """Validate and return the absolute path for ``--file`` argument.

    Rejects:
    - Paths containing ``..`` components (path-traversal guard, T-6-01b).
    - Paths that resolve outside REPO.

    Exits with code 2 on any rejected path.
    """
    # Reject parent-directory traversal components before any os.path resolution
    # so that crafted paths like ``../../etc/passwd`` are caught immediately.
    parts = raw.replace("\\", "/").split("/")
    if ".." in parts:
        print(
            f"ERROR: --file path contains a parent-directory component: {raw!r}",
            file=sys.stderr,
        )
        sys.exit(2)

    # Resolve to absolute; accept either absolute or repo-relative inputs
    if os.path.isabs(raw):
        candidate = os.path.realpath(raw)
    else:
        candidate = os.path.realpath(os.path.join(REPO, raw))

    repo_real = os.path.realpath(REPO)
    # Boundary safety: use prefix check with an explicit separator to prevent
    # false positives when REPO is a prefix of a sibling directory name.
    if candidate != repo_real and not candidate.startswith(repo_real + os.sep):
        print(
            f"ERROR: --file path resolves outside the repository root: {candidate!r}",
            file=sys.stderr,
        )
        print(f"       Repository root: {repo_real!r}", file=sys.stderr)
        sys.exit(2)

    return candidate


def _extract_blocks(text: str) -> list[tuple[bool, str, int]]:
    """Return a list of (skipped, sql_body, start_line) for every sql fence.

    ``skipped`` is True when the info-string carries the skip keyword.
    ``start_line`` is the 1-based line number of the opening fence.
    """
    results = []
    for m in BLOCK_RE.finditer(text):
        skipped = bool(m.group(1))  # group 1 is ' skip' or None
        body = m.group(2)
        start_line = text[: m.start()].count("\n") + 1
        results.append((skipped, body, start_line))
    return results


def _run_file(abs_path: str, rel_label: str) -> tuple[bool, int, str]:
    """Extract blocks from *abs_path*, concatenate, and run through the DuckDB CLI.

    Returns (passed, executed_block_count, stderr_on_failure).
    Files with zero executable blocks are skipped (reported as skipped, not failed).
    """
    try:
        text = open(abs_path, encoding="utf-8").read()
    except OSError as exc:
        return False, 0, f"Could not read file: {exc}"

    blocks = _extract_blocks(text)
    executable = [(body, lineno) for (skipped, body, lineno) in blocks if not skipped]

    if not executable:
        return True, 0, ""  # nothing to run; treat as pass

    # Build the temp SQL file:
    #   Line 1: .bail on  — makes the DuckDB CLI exit non-zero on the first
    #            failing statement, so a mid-file error aborts the remaining
    #            concatenated blocks rather than silently continuing.
    #   Lines 2+: all executable blocks joined in document order.
    sql_content = ".bail on\n" + "\n".join(body for body, _ in executable)

    tmp = tempfile.NamedTemporaryFile(
        suffix=".sql", mode="w", encoding="utf-8", delete=False
    )
    try:
        tmp.write(sql_content)
        tmp.flush()
        tmp.close()

        result = subprocess.run(
            [DUCKDB, "-unsigned", "-cmd", f"LOAD '{EXT}';", "-f", tmp.name],
            capture_output=True,
            text=True,
        )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    if result.returncode != 0:
        return False, len(executable), result.stderr
    return True, len(executable), ""


def main() -> int:
    _check_preconditions()
    args = _parse_args()

    if args.file is not None:
        # Single-file fast path consumed by Plans 02/03 verify commands
        abs_path = _resolve_single_file(args.file)
        rel_label = os.path.relpath(abs_path, REPO)
        files_to_run = [(abs_path, rel_label)]
    else:
        files_to_run = [
            (os.path.join(REPO, rel), rel) for rel in DOC_FILES
        ]

    total_executed = 0
    failed: list[str] = []

    for abs_path, rel_label in files_to_run:
        passed, n_blocks, stderr = _run_file(abs_path, rel_label)
        if n_blocks == 0:
            print(f"SKIP {rel_label} (no executable sql blocks)")
            continue
        total_executed += 1
        if passed:
            print(f"PASS {rel_label} ({n_blocks} block(s))")
        else:
            failed.append(rel_label)
            print(f"FAIL {rel_label}")
            if stderr.strip():
                # Indent the DuckDB stderr for readability
                for line in stderr.rstrip().splitlines():
                    print(f"     {line}")

    # Summary line — provides the baseline failure count used in the SUMMARY
    print()
    print(
        f"Executed: {total_executed} file(s)  "
        f"Passed: {total_executed - len(failed)}  "
        f"Failed: {len(failed)}"
    )

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
