# Branch Tracking

Use this reference when turning a Megatron tracking request into a normalized change set.

## Request Priorities

Prefer these selectors in order of semantic completeness:

1. PR
2. commit range
3. time window
4. single commit

Use a single commit only when the user explicitly asks for it or when no higher-level unit exists.

## Branch Rules

- Treat branch as a required dimension for all tracking modes.
- Keep branch names exactly as the upstream repository spells them.
- Distinguish between a logical version label and an exact remote branch name. Verify the exact remote branch string before fetching when there is any doubt.
- Do not collapse `core_v0.12.1`, `core_r0.15.3`, `dev`, and `main` into one family.
- For scheduled tracking, persist one state record per `repo + branch`.

## Suggested Scheduled State Shape

```json
{
  "repo": "NVIDIA/Megatron-LM",
  "branch": "dev",
  "last_seen_sha": "abc123",
  "last_checked_at": "2026-03-17T09:00:00Z"
}
```

## Output Contract

The tracker should hand downstream skills a `change-set` with:

- source type
- branch
- resolved commits
- base and head SHAs
- optional time window
- raw PR identifier when applicable

Leave interpretation, feature grouping, and migration judgments to downstream skills.
