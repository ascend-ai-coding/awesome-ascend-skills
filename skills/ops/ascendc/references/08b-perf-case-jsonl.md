# Phase 8b — Performance case JSONL spec

Performance cases live **only** as JSONL (`<op>_perf_cases.jsonl`). Do not generate or
maintain a `.json` array file.

## File form

- JSONL: **one JSON object per line**, trailing newline; blank lines ignored; extension
  `.jsonl`.

## Per-case structure

Each case object MUST contain an `"inputs"` array. Element kinds:

| `type` | Meaning | Key fields |
|---|---|---|
| `tensor` | input tensor | `name`, `required`, `dtype`, `shape` |
| `tensor_list` | list of tensors | `name`, `required`, `dtype`, `shapes` |
| `attr` | scalar/list attribute | `name`, `required`, `dtype` (`int`/`float`/`bool`), `value` |
| `range` (int tensor) | int tensor by range | `name`, `dtype`, `shape`, `low`, `high` |

`name` is unique within a case.

## Example (Layer Norm)

```json
{
  "inputs": [
    { "name": "x", "type": "tensor", "required": true, "dtype": "float16", "shape": [8, 128] },
    { "name": "normalized_shape", "type": "attr", "required": true, "dtype": "int", "value": [128] },
    { "name": "use_affine", "type": "attr", "required": false, "dtype": "bool", "value": true },
    { "name": "eps", "type": "attr", "required": false, "dtype": "float", "value": 1e-05 }
  ]
}
```

## JSONL (two lines)

```json
{"inputs":[{"name":"x","type":"tensor","required":true,"dtype":"float16","shape":[2,128]},{"name":"normalized_shape","type":"attr","required":true,"dtype":"int","value":[128]},{"name":"use_affine","type":"attr","required":false,"dtype":"bool","value":true},{"name":"eps","type":"attr","required":false,"dtype":"float","value":1e-05}]}
{"inputs":[{"name":"x","type":"tensor","required":true,"dtype":"float16","shape":[4,256]},{"name":"normalized_shape","type":"attr","required":true,"dtype":"int","value":[256]},{"name":"use_affine","type":"attr","required":false,"dtype":"bool","value":false},{"name":"eps","type":"attr","required":false,"dtype":"float","value":1e-05}]}
```

## Rules

- Total cases **>= 8**; cover small/medium/large shapes and all supported dtypes.
- Attribute values must lie within `design.md` constraints; integer/index tensor values
  must be semantically valid (e.g. valid window offsets).
- A working reference lives in
  [`../examples/layer_norm_profiler_reference/`](../examples/layer_norm_profiler_reference/)
  (`layer_norm_perf_cases.jsonl`, `build_inputs`, benchmark script).
