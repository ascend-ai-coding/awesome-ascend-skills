---
name: external-gitcode-ascend-megatron-change-analyzer
description: Analyze official Megatron-LM commits, PRs, and branch change sets to
  identify feature evolution, candidate breaking changes, and migration-relevant events.
  Use when Codex already has a normalized Megatron change set and needs to explain
  what changed, which new features matter, and which changes should flow into MindSpeed
  adaptation work.
original-name: megatron-change-analyzer
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-05'
synced-commit: 03b4bbd870d13fcd48044b285e84dd80e955214f
license: UNKNOWN
---

# Megatron Change Analyzer

Turn a raw Megatron change set into structured feature evolution findings that are ready for downstream implementation work. Focus on extracting the real upstream feature shape, not on writing MindSpeed code yet.

## Core Rules

- Assume input is a normalized Megatron `change-set`, not an arbitrary pile of commits.
- Group multiple low-level commits into higher-level change events when they serve one feature.
- Prioritize new features, API shape changes, config or CLI schema changes, and behavior-affecting refactors.
- Tag each event with migration relevance, but do not edit MindSpeed here.
- Treat the upstream commit itself as implementation evidence, not just as a summary source.
- When a feature is represented by a specific commit or tightly related commit cluster, preserve that grouping so downstream steps can port the implementation with the commit as the primary reference.
- Treat `megatron main` findings as exploratory unless the user has supplied a strict target mapping.

## Output Shape

Produce a structured report with one event per feature or migration-relevant change:

```json
{
  "branch": "core_v0.15.3",
  "events": [
    {
      "title": "Add feature X",
      "kind": "new_feature",
      "commits": ["sha1", "sha2"],
      "primary_commit": "sha2",
      "areas": ["training", "config"],
      "breaking_risk": "medium",
      "migration_relevance": "high",
      "notes": "Short factual summary",
      "evidence": ["file/path.py", "arguments.py"],
      "upstream_changed_files": ["megatron/path_a.py", "megatron/path_b.py"],
      "implementation_units": [
        {
          "name": "Expose new config flag",
          "kind": "config_surface",
          "upstream_files": ["megatron/training/config/training_config.py"],
          "summary": "What this unit changes in upstream code"
        },
        {
          "name": "Add runtime behavior",
          "kind": "runtime_logic",
          "upstream_files": ["megatron/training/global_vars.py"],
          "summary": "What runtime path must be ported downstream"
        }
      ],
      "porting_notes": [
        "Facts the downstream mapper should preserve when porting the feature"
      ]
    }
  ]
}
```

## Workflow

1. Read the normalized change-set.
2. Inspect commit titles, touched files, and diffs at a feature level.
3. Collapse related commits into one event when appropriate, but keep the primary implementation commit visible.
4. Label each event using the taxonomy in [change-taxonomy.md](./references/change-taxonomy.md).
5. Break each relevant event into implementation units:
   config and CLI exposure,
   runtime logic,
   wrappers or adaptors,
   tests and examples,
   lifecycle or cleanup behavior.
6. Separate events into:
   relevant for MindSpeed,
   probably already covered,
   not currently worth adaptation.
7. Hand off only the relevant subset to the impact mapper, including commit references, upstream changed files, and implementation units.

## What To Highlight

- Newly exposed features or workflows
- Public API additions or signature changes
- CLI/config additions, removals, or renames
- Checkpoint or state format changes
- Parallelism and distributed execution changes
- Data pipeline or runtime behavior changes

## What To Avoid

- Do not produce code patches here.
- Do not claim MindSpeed compatibility from Megatron evidence alone.
- Do not treat internal refactors as high-value migration work unless they change integration surfaces.
- Do not collapse a large feature into a one-line note if the upstream commit actually changes multiple integration surfaces. Preserve enough structure for downstream implementation.

## References

- Read [change-taxonomy.md](./references/change-taxonomy.md) before labeling events.
- Run [build_feature_events.py](./scripts/build_feature_events.py) when you need a deterministic first-pass event file that preserves primary commits, upstream changed files, and implementation units.
- Hand migration-relevant events to [$megatron-impact-mapper](/Users/wangjinyi/.codex/skills/megatron-impact-mapper/SKILL.md).
