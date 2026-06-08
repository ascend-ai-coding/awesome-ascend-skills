# [Operator Name] Test Cases

Unified test-case document, reused by precision (Phase 7) and performance (Phase 8).

## Test configuration

### SUPPORTED_DTYPES
```python
SUPPORTED_DTYPES = [torch.float16, torch.float32]   # add torch.bfloat16 if supported
```

### TEST_SHAPES (regular shapes)
```python
TEST_SHAPES = [
    ("1D",         "1024 elements",       (1024,)),
    ("1D",         "4096 elements",       (4096,)),
    ("2D",         "BERT-base 64x768",    (64, 768)),
    ("2D",         "BERT-large 128x1024", (128, 1024)),
    ("3D",         "8x16x64",            (8, 16, 64)),
    # pick only dimensions the operator supports; keep element count <= ~200K
]
```

### GENERAL_SHAPES (generalization)
```python
GENERAL_SHAPES = [
    ("Production", "ViT 8x197x768",       (8, 197, 768)),
    ("Odd",        "non-aligned 1x513",   (1, 513)),
]
```

### BOUNDARY_VALUES (small fixed shape, e.g. (1024,))
```python
BOUNDARY_VALUES = [
    ("domain lower bound", 1.0),
    ("near boundary",      1.001),
    ("moderate value",     10.0),
    ("large value",        1000.0),
    # driven by the operator's math domain
]
```

### Input domain
```python
INPUT_LOW  = 1.0      # lower bound of random inputs
INPUT_HIGH = 11.0     # upper bound (keep within fp16 range ~65504)
```

## Operator baseline

```python
# NPU call (cross-check with register.cpp m.def)
def NPU_CALL(x):
    return torch.ops.npu.[operator_name](x)

# CPU reference (fp16/bf16: compute in fp32 then cast back)
def CPU_REF(x, dtype):
    return torch.[reference](x.cpu().float()).to(dtype)
```

If no direct PyTorch equivalent exists, implement the baseline as a small-op composition
(tensor ops only, no Python scalar loops) so it can also run on NPU in Phase 8.

## Coverage check
- [ ] every supported dtype covered by every shape and every boundary value
- [ ] `(len(TEST_SHAPES) + len(GENERAL_SHAPES)) * len(SUPPORTED_DTYPES) >= 30`
- [ ] shapes respect the operator's supported dimensions and constraints
- [ ] boundary values respect the operator's input domain
