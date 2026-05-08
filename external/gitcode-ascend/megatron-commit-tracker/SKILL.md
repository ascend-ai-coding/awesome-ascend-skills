---
name: external-gitcode-ascend-megatron-commit-tracker
description: Track and normalize change requests against the official Megatron-LM
  repository by branch, PR, commit, commit range, or time window. Use when Codex needs
  to collect the exact upstream change set before deeper analysis, especially for
  branch-aware Megatron and MindSpeed migration work, daily/periodic tracking, or
  preparing inputs for change analysis and migration generation.
original-name: megatron-commit-tracker
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-08'
synced-commit: 155ac37bd169ddb89479af528297cfb2237400aa
license: UNKNOWN
---

# Megatron Commit Tracker

Collect the smallest correct upstream change set from the official Megatron-LM repository and hand off a normalized artifact to downstream skills. Treat `branch` as a first-class input for every mode.

## Core Rules

- Use the official upstream repository as the source of truth: `https://github.com/NVIDIA/Megatron-LM`.
- Require a target `branch` unless the user gives a PR or commit that can be resolved unambiguously.
- Normalize every request into one `change-set` object before doing any semantic analysis.
- Keep this skill mechanical. Do not summarize feature evolution here.
- Prefer PR and range/time-window modes over single-commit mode for feature tracking.
- Treat `megatron main` as valid for exploration, but do not imply that it has a strict MindSpeed mapping.

## Supported Request Modes

Resolve the user request into exactly one of these modes:

1. `pr`
   Use when the user specifies a PR number or a merge request URL.
2. `commit`
   Use when the user specifies one commit SHA and wants to inspect that change in branch context.
3. `range`
   Use when the user specifies `base_sha..head_sha`.
4. `time_window`
   Use when the user specifies a branch plus a period such as `last 7 days`.
5. `scheduled`
   Use when an automation is checking a branch incrementally.

## Required Output

Produce a normalized `change-set` artifact in JSON or markdown code block form with these fields when available:

```json
{
  "repo": "NVIDIA/Megatron-LM",
  "branch": "core_v0.12.1",
  "source_type": "pr",
  "selector": {
    "pr": 1234,
    "commit": null,
    "base_sha": null,
    "head_sha": null,
    "since": null,
    "until": null
  },
  "resolved": {
    "commits": ["sha1", "sha2"],
    "head_sha": "sha2",
    "base_sha": "sha0"
  },
  "analysis_mode": "summary"
}
```

Also include a compact table of:

- commit SHA
- author
- authored date
- title
- touched files count

## Workflow

1. Parse the user request into one supported mode.
2. Resolve the request against the named Megatron branch.
3. Confirm that the branch context is explicit in the output.
4. Collect only the raw upstream artifacts needed downstream:
   commit list, merge metadata, changed files, base/head SHAs, and any linked PR metadata.
5. Stop after normalization unless the user explicitly asks for deeper analysis.

## Branch Handling

- Never silently substitute `main` for another branch.
- Verify the exact branch string against the official remote before fetch-heavy work when the requested name comes from local conventions or release notes.
- If the user gives a commit without branch context, try to infer the branch only when it is unambiguous. Otherwise state that branch confirmation is needed.
- For periodic tracking, store state by `repo + branch`, never by repository alone.
- For `main`, frame the result as exploratory upstream tracking, not strict migration-ready alignment.

## Handoff

Pass the normalized `change-set` to:

- [$megatron-change-analyzer](/Users/wangjinyi/.codex/skills/megatron-change-analyzer/SKILL.md) for feature evolution and change classification
- [$megatron-impact-mapper](/Users/wangjinyi/.codex/skills/megatron-impact-mapper/SKILL.md) only after change analysis has identified relevant events

## References

- Read [branch-tracking.md](./references/branch-tracking.md) for branch-aware request normalization and scheduling guidance.
- Run [normalize_change_request.py](./scripts/normalize_change_request.py) when a deterministic `change-set` structure is needed.
- Run [list_remote_branches.py](./scripts/list_remote_branches.py) to verify the exact branch names exposed by the official upstream remote before assuming a branch alias is fetchable.
- Run [fetch_upstream_changes.py](./scripts/fetch_upstream_changes.py) to collect commit metadata from the official upstream repository without manually rebuilding Git queries each time.
