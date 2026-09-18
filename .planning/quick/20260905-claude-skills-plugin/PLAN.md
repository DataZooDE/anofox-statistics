---
type: quick
slug: claude-skills-plugin
created: 2026-09-05
status: in-progress
---

# Quick Task: Claude Code skills plugin (installable), README, batch skill

## Goal

Make `anofox-statistics` installable as a Claude Code plugin in a Claude session —
mirroring the structure of `DataZooDE/anofox-forecast` — and add an AI-ready **batch**
skill for large-scale per-group model fitting. Update the README with an installation
section for the skills.

## Context

- `.claude/` in this repo is currently empty; there are no skills yet.
- Reference: `DataZooDE/anofox-forecast` uses a Claude Code plugin marketplace:
  - `.claude-plugin/marketplace.json` (repo root)
  - `plugins/<name>/.claude-plugin/plugin.json`
  - `plugins/<name>/skills/<skill>/SKILL.md` (5 skills)
  - README "Claude Code Skills" install block: `/plugin marketplace add …` + `/plugin install …`
- Function inventory + return-struct fields verified against `src/` registrations and
  `docs/API_CONVENTIONS.md`. Note: actual return field is `n_observations` (NOT `n_obs`
  as the conventions doc §3 erroneously states) — skills use the source-verified name.

## Skill decomposition (confirmed with user)

4 skills under `plugins/anofox-statistics/skills/`:

1. `anofox-statistics-regression` — all `*_fit` / `*_fit_agg` models (OLS, Huber, RANSAC,
   Theil-Sen, Ridge, ElasticNet, WLS, RLS, BLS/NNLS, PLS, Isotonic, Quantile, GLMs,
   ALM, AFT, GLMM, EB), option-map keys, return-struct fields (incl. GLM/AFT `z_values`).
2. `anofox-statistics-tests` — hypothesis tests, correlation, effect sizes, equivalence (TOST),
   distribution comparison, forecast-evaluation tests.
3. `anofox-statistics-diagnostics` — VIF, AIC/BIC, residual diagnostics, AID demand classification,
   model-selection guidance.
4. `anofox-statistics-batch` — AI-ready batch/per-group fitting: `*_fit_predict_by` table macros
   + `*_fit_agg` with `GROUP BY` (fit thousands of models in one SQL query), scaling notes.

## Tasks

1. `.claude-plugin/marketplace.json` — marketplace with one plugin entry.
2. `plugins/anofox-statistics/.claude-plugin/plugin.json` — plugin manifest (v0.10.0).
3. Write the 4 `SKILL.md` files (accurate signatures/options/fields, `skip`-safe examples).
4. README.md — add "🤖 Claude Code Skills (AI pair-programming)" block under Installation.
5. Validate JSON + skill frontmatter; verify function names against source.
6. Atomic commit; update STATE.md quick-tasks table; write SUMMARY.md.

## Verification

- `python3 -c 'import json'` parses both JSON files.
- Every function named in the skills exists in `src/` registrations (grep check).
- README renders the install block; links resolve to `plugins/anofox-statistics/`.
- `git status` clean after commit.
