---
name: external-gitcode-ascend-megatron-migration-generator
description: Generate migration deliverables for bringing relevant Megatron changes
  into MindSpeed after branch alignment and impact mapping are complete. Use when
  Codex already has a confirmed MindSpeed-to-Megatron branch pairing and needs to
  produce a migration report, candidate patch, or guarded workspace edits instead
  of redoing upstream analysis from scratch.
original-name: megatron-migration-generator
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-04-29'
synced-commit: 8faee0275e457955c8f50989aef8972c0838db31
license: UNKNOWN
---

# Megatron Migration Generator

Generate implementation-oriented migration artifacts for MindSpeed only after upstream change analysis and impact mapping have narrowed the work to a high-confidence set of items. The goal is to port the upstream feature into MindSpeed with the upstream commit as the primary reference, not to emit a tiny patch hint.

## Default Output Mode

Default to `report + patch`, and generate the final markdown deliverable in Chinese unless the user explicitly requests another locale.

Do not default to direct code edits or automatic commits.
Default to an implementation-first package, not a minimal patch sketch.

## Supported Modes

1. `report`
   Produce a migration report only.
2. `patch`
   Produce a migration report plus a full implementation reference package. Default to per-feature folders containing `full.patch`, split `patches/`, upstream reference metadata, implementation checklists, and explicit uncovered scope.
3. `apply`
   Edit the local workspace without creating a commit. Use only when alignment and impact confidence are high.
4. `commit`
   Reserve for explicit user requests after patch review. Do not use as the first-pass default.

## Entry Conditions

Proceed only when all of the following are true:

- MindSpeed branch is known.
- Megatron base branch is known.
- Megatron target branch or change-set is known.
- Relevant impact items already exist.
- Each migration-worthy item includes upstream commit references and enough implementation scope to tell config/runtime/wrapper/test work apart.

If any of these are missing, stop and ask for or derive the missing prerequisite instead of guessing.

## Workflow

1. Read the impact report.
2. Separate high-confidence items from speculative items.
3. For each high-confidence item, describe:
   the upstream event,
   the upstream commit and changed files,
   the upstream implementation units,
   the local adaptation target,
   the intended code change,
   the uncertainty.
4. Generate:
   one integrated `migration_report.md`
   `impact_report.json` if needed for carry-forward
   `candidate_patch.md` as a compact summary index
   a `features/` directory with one subfolder per migration item, each containing:
   `full.patch` for the full local implementation draft
   `patches/` for split patch series organized by implementation unit
   `candidate.patch` as the compatibility entry patch
   `README.md`, `checklist.md`, `package_manifest.json`, and `upstream_reference.json`
5. If in `apply` mode, edit only the approved files and leave a clear explanation of what changed.

## Safety Rules

- Never generate migration code directly from raw commit history without an impact report.
- Avoid direct edits in exploration mode.
- When the target is `megatron main`, frame code output as a candidate adaptation draft, not a compatibility-guaranteed migration.
- Prefer patch artifacts over commits because they are easier to review and safer when branch alignment is still evolving.
- Do not treat one local file edit as a complete feature migration when the upstream commit spans config, runtime logic, wrappers, and tests.
- When a full migration package cannot be completed from the available inputs, explicitly record uncovered implementation units, omitted scope, and manual follow-ups instead of pretending the feature is complete.

## Static Validation

Keep validation lightweight and local:

- patch applies cleanly
- paths exist
- names and arguments match local code structure
- basic syntax or lint-level checks when available
- the package explains which upstream implementation units are already ported and which remain to be ported

Do not imply that training correctness, performance, or numerical accuracy has been verified.

## References

- Read [migration-modes.md](./references/migration-modes.md) before choosing the output form.
- Read [implementation-first-package.md](./references/implementation-first-package.md) before generating a migration package that claims to represent a feature implementation.
- Read [report-template.md](./references/report-template.md) for the expected integrated markdown deliverable structure. Default to Chinese report language.
- Run [synthesize_full_patch_series.py](./scripts/synthesize_full_patch_series.py) after impact mapping when you want known feature commits to expand into fuller multi-file patch series instead of staying as single-file placeholders.
- Run [render_migration_artifacts.py](./scripts/render_migration_artifacts.py) to turn an `impact_report.json` into reviewable report and patch-plan artifacts before any manual refinement.
