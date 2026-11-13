#!/usr/bin/env python3
"""Plugin checks for the anofox-statistics Claude Code plugin.

scripts/validate_docs_sql.py runs every ```sql block. This script checks what
running the blocks cannot:

1. Names resolve. Every backticked call in the skills, README, guides and
   docs shaped like one of this extension's functions (``name(`` where name
   ends in ``_agg``, ``_by``, ``_fit``, ``_fit_predict``, ``_predict`` or
   ``_test``), and every backticked bare ``*_agg`` name, is registered in
   duckdb_functions(). sql blocks are executed by
   validate_docs_sql.py; a name mentioned only in prose never is, so a rename
   leaves it stale silently. Names cited from R or Python (``survfit()``,
   ``coxph()``) do not have those suffixes and are not checked.
2. Coverage. Every function the extension registers is mentioned in at least
   one skill, by name or alias, or by a suffix pattern a skill states
   (``<model>_tidy_by``, ``*_fit_predict_agg``) together with the model name
   (``poisson``) somewhere in the skills. An assistant cannot discover a
   function no skill mentions either way.
3. Version agreement. plugins/anofox-statistics/.claude-plugin/plugin.json,
   every SKILL.md frontmatter ``version:`` and every SKILL.md
   ``**Extension:** `anofox_statistics` vX`` header equal the newest released
   ``## [X.Y.Z]`` heading in CHANGELOG.md. Run scripts/bump_plugin_version.sh
   (make bump-plugin) in the release commit.

"The extension's functions" are the names duckdb_functions() tags with
``anofox.family = 'statistics'`` (scalars, aggregates and table macros), plus
any alias of one of them. The CLI links the extension statically, so a
loaded-minus-unloaded difference would be empty.

Usage:
    python3 scripts/check_plugin.py                 # all checks (needs make release)
    python3 scripts/check_plugin.py --version-only  # check 3 only, no build needed
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUCKDB = os.path.join(REPO, "build", "release", "duckdb")
EXT = os.path.join(REPO, "build", "release", "extension", "anofox_statistics",
                   "anofox_statistics.duckdb_extension")
PLUGIN = os.path.join(REPO, "plugins", "anofox-statistics")
SKILLS = sorted(glob.glob(os.path.join(PLUGIN, "skills", "*", "SKILL.md")))

# Files whose prose is name-checked (sql blocks are validate_docs_sql.py's job).
# MIGRATION.md, API_CONVENTIONS.md and CHANGELOG.md name removed functions on
# purpose (rename tables, "Before" examples), so they are left out.
NAME_CHECKED = ["README.md", "guides/0*.md", "docs/API_REFERENCE.md", "docs/METHODOLOGY.md",
                "docs/NULL_SEMANTICS.md", "docs/PERFORMANCE.md", "docs/api/**/*.md",
                "plugins/anofox-statistics/skills/*/SKILL.md"]

SQL_BLOCK_RE = re.compile(r"^```sql[^\n]*\n.*?^```[ \t]*$", re.M | re.S)
# A backticked call shaped like ours (`ols_fit_agg(y, x)`, `cox_zph_test(`), or a
# backticked bare aggregate name (`ols_fit_agg`). Bare names with the other
# suffixes are left alone: option keys and result fields such as `order_by`,
# `se_fit`, `n_test` and `po_test` share them.
OURS_RE = re.compile(r"`([a-z][a-z0-9_]*_(?:agg|by|fit|fit_predict|predict|test))\(|`([a-z][a-z0-9_]*_agg)\b")
# Shaped like ours but deliberately not registered: removed names that docs
# cite as history, and names owned by other extensions.
NOT_OURS = {"anofox_plot_by"}
WORD_RE = re.compile(r"\b[a-z][a-z0-9_]*\b")
RELEASE_RE = re.compile(r"^## \[(\d+\.\d+\.\d+)\]", re.M)
FRONTMATTER_VERSION_RE = re.compile(r"^version:\s*(\S+)\s*$", re.M)
HEADER_VERSION_RE = re.compile(r"^\*\*Extension:\*\* `anofox_statistics` v([0-9.]+)", re.M)


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def rel(path):
    return os.path.relpath(path, REPO)


def query(sql):
    cmd = [DUCKDB, "-unsigned", "-csv", "-noheader", "-cmd", f"LOAD '{EXT}';", "-c", sql]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [line.split(",") for line in out.splitlines() if line]


def catalog():
    """(all registered names, canonical -> set of its names incl. aliases)."""
    rows = query("SELECT DISTINCT function_name, coalesce(alias_of, ''), "
                 "coalesce(tags['anofox.family'], '') = 'statistics' FROM duckdb_functions()")
    all_names = {r[0] for r in rows}
    ours = {r[0] for r in rows if r[2] == "true"}
    ext = {name: {name} for name in ours}
    for name, alias_of, _ in rows:
        if alias_of in ours:
            ext[alias_of].add(name)
    return all_names, ext


def prose(text):
    return SQL_BLOCK_RE.sub("", text)


def check_names(all_names):
    failures = []
    paths = []
    for pattern in NAME_CHECKED:
        paths += sorted(glob.glob(os.path.join(REPO, pattern), recursive=True))
    for path in dict.fromkeys(paths):
        text = prose(read(path))
        names = {a or b for a, b in OURS_RE.findall(text)} - NOT_OURS
        for name in sorted(names - all_names):
            failures.append((rel(path), f"names unregistered function {name}"))
    return failures


# A suffix pattern a skill states for a whole set of models: `*_tidy_by`,
# `<model>_tidy_by`, `{model}_fit_predict_agg`.
PATTERN_RE = re.compile(r"(?:\*|<[a-z]+>|\{[a-z]+\})_([a-z][a-z0-9_]*)")


def covered_by_pattern(name, patterns, mentioned):
    """True when a skill states name's suffix as a pattern and names its model.

    `poisson_tidy_by` is covered when some skill says `<model>_tidy_by` and
    mentions `poisson`: that is how a reader composes the name.
    """
    for suffix in patterns:
        if name.endswith("_" + suffix):
            model = name[: -len(suffix) - 1]
            if model in mentioned:
                return True
    return False


def check_coverage(ext):
    mentioned, patterns = set(), set()
    for path in SKILLS:
        text = read(path)
        mentioned |= set(WORD_RE.findall(text))
        patterns |= set(PATTERN_RE.findall(text))
    failures = []
    for canonical in sorted(ext):
        if ext[canonical] & mentioned:
            continue
        if not covered_by_pattern(canonical, patterns, mentioned):
            names = " / ".join(sorted(ext[canonical]))
            failures.append(("plugins/anofox-statistics/skills", f"no skill covers {names}"))
    return failures


def check_versions():
    releases = RELEASE_RE.findall(read(os.path.join(REPO, "CHANGELOG.md")))
    want = releases[0] if releases else None
    manifest = os.path.join(PLUGIN, ".claude-plugin", "plugin.json")
    got = {rel(manifest): json.loads(read(manifest)).get("version")}
    for path in SKILLS:
        text = read(path)
        fm = FRONTMATTER_VERSION_RE.search(text.split("\n---", 1)[0])
        hd = HEADER_VERSION_RE.search(text)
        got[rel(path)] = fm.group(1) if fm else None
        got[rel(path) + " (header)"] = hd.group(1) if hd else None
    return [(where, f"version {v} != newest CHANGELOG release {want}")
            for where, v in got.items() if v != want]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--version-only", action="store_true",
                    help="only the version-agreement check (no build needed)")
    args = ap.parse_args()

    failures = check_versions()
    if not args.version_only:
        if not (os.access(DUCKDB, os.X_OK) and os.path.isfile(EXT)):
            print(f"ERROR: build the extension first (make release): {rel(DUCKDB)}, {rel(EXT)}",
                  file=sys.stderr)
            return 1
        all_names, ext = catalog()
        failures += check_names(all_names)
        failures += check_coverage(ext)
        print(f"catalog: {len(ext)} extension functions, {len(SKILLS)} skills")
    for where, detail in failures:
        print(f"FAIL {where}: {detail}")
    print(f"check_plugin: {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
