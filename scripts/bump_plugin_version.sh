#!/usr/bin/env bash
# Set the Claude Code plugin's version to the newest release in CHANGELOG.md.
#
# Part of every release, right after the "## [X.Y.Z] - YYYY-MM-DD" heading is
# written:
#   ./scripts/bump_plugin_version.sh        (or: make bump-plugin)
# It rewrites plugins/anofox-statistics/.claude-plugin/plugin.json "version",
# every SKILL.md "version:" line and every SKILL.md "**Extension:** ... vX"
# header. Prose that names a version ("New in 0.17.0") is left alone.
# scripts/check_plugin.py fails while any of them is behind.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
plugin="$root/plugins/anofox-statistics"

version="$(sed -nE 's/^## \[([0-9]+\.[0-9]+\.[0-9]+)\].*/\1/p' "$root/CHANGELOG.md" | head -1)"
if [[ -z "$version" ]]; then
  echo "no '## [X.Y.Z]' release heading in CHANGELOG.md" >&2
  exit 1
fi

sed -i -E 's/^(  "version": ")[^"]*(",?)$/\1'"$version"'\2/' "$plugin/.claude-plugin/plugin.json"
for f in "$plugin"/skills/*/SKILL.md; do
  sed -i -E \
    -e 's/^version: .*/version: '"$version"'/' \
    -e 's/^(\*\*Extension:\*\* `anofox_statistics` v)[0-9.]+/\1'"$version"'/' \
    "$f"
done

echo "plugin and skills set to $version"
grep -H '"version"' "$plugin/.claude-plugin/plugin.json"
grep -H -E '^version: ' "$plugin"/skills/*/SKILL.md
