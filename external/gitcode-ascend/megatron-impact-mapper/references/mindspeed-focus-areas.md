# MindSpeed Focus Areas

Use this reference to keep impact mapping focused on the places most likely to carry Megatron adaptations.

## Common High-Value Search Areas

- training entrypoints and launch scripts
- argument and config handling
- distributed or parallelism wrappers
- checkpoint integration paths
- runtime hooks and plugin layers
- wrappers around Megatron public APIs

## Mapping Heuristics

- Start from Megatron files touched by the event and look for similarly named modules, wrappers, or argument plumbing in MindSpeed.
- Give more weight to files that bridge MindSpeed-specific hardware/runtime behavior with Megatron integration points.
- Distinguish code that mirrors Megatron interfaces from code that is purely internal to MindSpeed.

## Output Expectations

For each impact item, report:

- candidate MindSpeed paths
- why they are likely relevant
- whether the feature appears already adapted
- whether the gap is config-only, API-level, runtime-level, or unclear
