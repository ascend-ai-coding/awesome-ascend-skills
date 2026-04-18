---
name: external-gitcode-ascend-megatron-impact-mapper
description: Map migration-relevant Megatron changes onto the official MindSpeed repository
  by resolving branch alignment, locating affected subsystems, and identifying concrete
  adaptation points. Use when Codex has structured Megatron change events and needs
  to decide whether MindSpeed already covers them, which MindSpeed files are likely
  affected, and whether patch generation is safe.
original-name: megatron-impact-mapper
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-04-18'
synced-commit: 9f4c6c19a042f03239a07ac2f3196fb590d0a114
license: UNKNOWN
---

# Megatron Impact Mapper

Resolve the correct MindSpeed branch context and map each relevant Megatron event onto concrete MindSpeed implementation targets. This skill is the gatekeeper for implementation-oriented migration generation.

## Hard Gate

Do not generate migration patches until branch alignment is explicit.

Use one of these alignment sources:

1. user-supplied mapping
2. a known rule in [branch-alignment.md](./references/branch-alignment.md)
3. a repository-local document or commit that states the mapping clearly

If none of these is available, stop at an impact report.

## Core Rules

- Treat `MindSpeed branch <-> Megatron branch` alignment as mandatory for strict migration mode.
- Use only official repositories as the source of truth for first-pass analysis.
- Distinguish between `strict migration mode` and `exploration mode`.
- For `megatron main`, prefer exploration mode unless the user explicitly chooses a compatible MindSpeed baseline for comparison.
- Separate three outcomes:
  already adapted,
  likely adaptable but missing,
  not worth adapting now.
- Do not stop at "candidate paths" when a feature is clearly migration-worthy. Break the feature into implementation targets that correspond to the upstream implementation units.

## Workflow

1. Resolve the MindSpeed branch.
2. Resolve the mapped Megatron baseline branch.
3. Compare the baseline against the target Megatron branch or change-set.
4. For each relevant Megatron event, search for corresponding MindSpeed modules, wrappers, configs, launch paths, and tests.
5. Map upstream implementation units onto local implementation targets.
6. Produce an impact report with confidence, rationale, and implementation scope.
7. Allow handoff to migration generation only for high-confidence impact items.

## Required Output

Produce an `impact_report` with:

```json
{
  "mindspeed_branch": "master",
  "megatron_base_branch": "core_v0.12.1",
  "megatron_target_branch": "core_v0.15.3",
  "mode": "strict",
  "items": [
    {
      "event_title": "Add feature X",
      "status": "likely_missing",
      "confidence": 0.78,
      "candidate_paths": ["mindspeed/path_a.py", "mindspeed/path_b.py"],
      "reason": "Short factual rationale",
      "primary_commit": "sha1",
      "upstream_changed_files": ["megatron/path_a.py", "megatron/path_b.py"],
      "implementation_targets": [
        {
          "name": "Expose config on MindSpeed side",
          "source_unit": "Expose new config flag",
          "candidate_paths": ["mindspeed/arguments.py"],
          "required_change": "Add argument or YAML surface",
          "confidence": 0.82
        },
        {
          "name": "Port runtime behavior",
          "source_unit": "Add runtime behavior",
          "candidate_paths": ["mindspeed/training.py", "mindspeed/core/training.py"],
          "required_change": "Recreate the training shutdown logic locally",
          "confidence": 0.67
        }
      ],
      "covered_units": ["Expose config on MindSpeed side"],
      "missing_units": ["Port runtime behavior"]
    }
  ]
}
```

## Decision Rules

- If alignment is unresolved: report only.
- If alignment is resolved but implementation targets are weakly supported: report plus implementation plan, not direct edits.
- If alignment and implementation targets are both high confidence: allow migration patch generation.
- If an item has only one touched local file but the upstream event clearly spans multiple implementation units, mark the uncovered units explicitly instead of pretending the feature is fully mapped.

## References

- Read [branch-alignment.md](./references/branch-alignment.md) before doing any strict migration work.
- Read [mindspeed-focus-areas.md](./references/mindspeed-focus-areas.md) when searching for candidate adaptation paths.
- Run [resolve_branch_alignment.py](./scripts/resolve_branch_alignment.py) to deterministically classify a branch pair as `strict`, `exploration`, or `unresolved`.
- Run [scan_mindspeed_paths.py](./scripts/scan_mindspeed_paths.py) to search the official MindSpeed repository on a specific branch for likely adaptation points using feature names, file names, symbols, or config keys.
- Run [map_implementation_targets.py](./scripts/map_implementation_targets.py) to convert upstream implementation units into local MindSpeed implementation targets before patch generation.
- Hand off only approved items to [$megatron-migration-generator](/Users/wangjinyi/.codex/skills/megatron-migration-generator/SKILL.md).
