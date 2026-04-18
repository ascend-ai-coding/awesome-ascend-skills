# Implementation-First Migration Package

This migration generator should optimize for porting an upstream Megatron feature into MindSpeed with the upstream commit as the primary engineering reference.

## Principle

Do not stop at:

- a title
- a few candidate paths
- one tiny local diff

Instead, the package should explain:

- which upstream commit or commit cluster defines the feature
- which upstream files changed
- which implementation units exist in upstream
- which local MindSpeed files should absorb each implementation unit
- which units are already represented in the generated patch
- which units remain uncovered and must be completed manually

## Expected Item Fields

An implementation-oriented `impact_report` item should ideally include:

```json
{
  "event_title": "Training-level graceful shutdown",
  "primary_commit": "abc123",
  "commits": ["abc123"],
  "upstream_changed_files": [
    "megatron/training/config/training_config.py",
    "megatron/training/global_vars.py"
  ],
  "implementation_units": [
    {
      "name": "Expose config flag",
      "kind": "config_surface",
      "upstream_files": ["megatron/training/config/training_config.py"],
      "summary": "Adds a new config surface for training exit handling"
    },
    {
      "name": "Add runtime signal handling",
      "kind": "runtime_logic",
      "upstream_files": ["megatron/training/global_vars.py"],
      "summary": "Registers handler and performs orderly shutdown"
    }
  ],
  "implementation_targets": [
    {
      "name": "Expose MindSpeed argument",
      "source_unit": "Expose config flag",
      "candidate_paths": ["mindspeed/arguments.py"],
      "required_change": "Add CLI and YAML surface"
    },
    {
      "name": "Port shutdown runtime path",
      "source_unit": "Add runtime signal handling",
      "candidate_paths": ["mindspeed/training.py", "mindspeed/core/training.py"],
      "required_change": "Recreate orderly shutdown behavior"
    }
  ]
}
```

## Required Package Outputs

Each feature folder should contain:

- `full.patch`
- `patches/`
- `candidate.patch`
- `README.md`
- `checklist.md`
- `package_manifest.json`
- `upstream_reference.json`

## Truthfulness Rule

If only one implementation unit has been converted into local patch form, the package must say so explicitly. Do not imply that the whole feature has been ported.
