# Branch Alignment

Use these rules before claiming that a Megatron change should map onto a specific MindSpeed branch.

## Confirmed High-Frequency Mappings

- `mindspeed/master -> megatron/core_v0.12.1`
- `mindspeed/2.3.0_core_r0.12.1 -> megatron/core_v0.12.1`
- `mindspeed/dev -> megatron/dev`
- `mindspeed/core_r0.15.3 -> megatron/core_v0.15.3`

## Important Constraints

- The mapping is deterministic for stable development work. If the wrong branch pair is used, training is expected to fail.
- The logical mapping still needs exact remote branch verification. A branch name quoted in docs or local shorthand may differ from the exact branch string exposed by the official remote.
- There is currently no confirmed MindSpeed branch that strictly maps to `megatron main`.
- `megatron main` may still be used for exploratory feature analysis against a MindSpeed baseline, but not for high-confidence migration claims.

## Alignment Modes

### Strict migration mode

Use when:

- the user supplies both sides explicitly, or
- the mapping is confirmed by the rules above

Allow patch generation only in this mode.

### Exploration mode

Use when:

- the target is `megatron main`, or
- the target branch is not in the known mapping table, or
- repository evidence is suggestive but not conclusive

In this mode, produce findings and candidate adaptation points, but avoid direct code edits by default.
