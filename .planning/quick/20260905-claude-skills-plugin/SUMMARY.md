---
type: quick
slug: claude-skills-plugin
created: 2026-09-05
completed: 2026-09-05
status: complete
---

# Summary: Claude Code skills plugin (installable), README, batch skill

## What was delivered

Made `anofox-statistics` installable as a Claude Code plugin in a Claude session,
mirroring `DataZooDE/anofox-forecast`, and added an AI-ready batch skill.

### Files created
- `.claude-plugin/marketplace.json` — marketplace manifest (one plugin entry).
- `plugins/anofox-statistics/.claude-plugin/plugin.json` — plugin manifest, v0.10.0, BUSL-1.1.
- `plugins/anofox-statistics/skills/anofox-statistics-regression/SKILL.md`
- `plugins/anofox-statistics/skills/anofox-statistics-tests/SKILL.md`
- `plugins/anofox-statistics/skills/anofox-statistics-diagnostics/SKILL.md`
- `plugins/anofox-statistics/skills/anofox-statistics-batch/SKILL.md` (the "batch AI-ready" skill)

### Files modified
- `README.md` — added "🤖 Claude Code Skills (AI pair-programming)" section with the
  `/plugin marketplace add DataZooDE/anofox-statistics` + `/plugin install` block and a
  skill table; added a ToC entry.

## Install (end-user)
```
/plugin marketplace add DataZooDE/anofox-statistics
/plugin install anofox-statistics@anofox-statistics
```
In-repo dev: `claude --plugin-dir ./plugins/anofox-statistics`.

## Validation performed
- Both JSON manifests parse (`python3 -c json.load`).
- All four SKILL.md files have valid frontmatter (`name`, `version`).
- **Accuracy gate:** every function/macro named in the skills was cross-checked against
  `src/` registrations — all exist (only regex false-positives were `{model}_fit_agg` /
  `test_agg(...)` template placeholders).
- `*_fit_predict_by` signature verified against `src/macros/fit_predict_macros.cpp` and
  `test/`: `(source, group_col, y_col, x_cols[, options])`, output column `yhat`; noted
  WLS `weight_col` and isotonic single-`x_col` variants.

## Notes / follow-ups
- Found a doc drift: `docs/API_CONVENTIONS.md` §3 lists `n_obs`, but the actual registered
  return field is `n_observations` (28 occurrences in `src/`, zero for `n_obs`). Skills use
  the source-verified `n_observations`. Worth fixing the conventions doc separately.
- Skills are not yet exercised against a live built extension; examples are source-accurate
  but a runtime pass (build + run each non-`skip` block) would fully close the loop.
- Not pushed — committed to branch `gsd/v0.3.0-performance-polish` only.
