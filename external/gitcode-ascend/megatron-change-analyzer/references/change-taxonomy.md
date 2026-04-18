# Change Taxonomy

Use this taxonomy to classify Megatron changes before mapping them into MindSpeed work.

## Event Kinds

- `new_feature`: A new capability, runtime mode, or user-facing path.
- `api_change`: A public class, function, method, or module surface changed.
- `config_change`: A config key, schema field, argument, or CLI flag changed.
- `behavior_change`: Runtime semantics changed without a large API change.
- `internal_refactor`: Code moved or reorganized with low outward impact.
- `checkpoint_change`: Serialization, load/save, or state compatibility changed.

## Areas

- `training`
- `parallelism`
- `distributed`
- `checkpoint`
- `config`
- `cli`
- `data`
- `runtime`

## Breaking Risk

- `low`: additive and unlikely to break downstream users
- `medium`: behavior or interface changed in a way that may require adaptation
- `high`: downstream code is likely to fail or silently diverge without adaptation

## Migration Relevance

- `high`: likely worth mapping into MindSpeed now
- `medium`: useful but may depend on product priorities
- `low`: not worth patch generation in the first pass

## Screening Heuristics

Prefer higher relevance when:

- the change adds a feature MindSpeed users would expose
- arguments or config schema changed
- checkpoint compatibility changed
- distributed or NPU-adjacent execution flow changed

Prefer lower relevance when:

- the change is only test churn
- the change is documentation-only
- the change is internal cleanup without integration impact
