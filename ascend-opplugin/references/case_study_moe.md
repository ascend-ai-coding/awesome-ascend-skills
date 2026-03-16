# Case study: moe_init_routing_grouped_matmul_grad (Pattern C, xpu_kernel)

This document summarizes the end-to-end flow of integrating xpu_kernel's `moe_init_routing_grouped_matmul_grad` operator into op-plugin, as a reference for Pattern C (OpCommand) integration with **custom operator libraries**.

## End-to-end flow overview

```
1. Confirm operator source: xpu_kernel (op_def + infer_shape + tiling + kernel already implemented)
2. Choose Pattern C: graph op exists, only host wrapper needed
3. Read op_def for: graph op name, input/output names, dtype
4. Read infer_shape for: output shape derivation logic
5. Implement host: OpCommand + descName matching
6. Build: CMake host-only, no ascendc_library
7. Test: registration + shape validation
```

## 1. Operator source confirmation

- **Path**: `xpu_kernel/xpu_kernel/C_like/transformer/npu/moe_init_routing_grouped_matmul_grad/`
- **op_def**: `op_host/moe_init_routing_grouped_matmul_grad_def.cpp`
  - `OP_ADD(MoeInitRoutingGroupedMatmulGrad)` → graph op name: `MoeInitRoutingGroupedMatmulGrad`
  - Input: `inputGradY`, `inputWeight`, `expandedRowIdx`, `groupList`
  - Output: `outputGradX`, `outputGradWeight`
- **infer_shape**: `grad_x [batch, k]`, `grad_weight [group_num, k, n]`, where `batch = expanded_row_idx.shape[0] / topk`

## 2. OpCommand input/output naming

Use `descName` to match op_def Input/Output names one-to-one:

```cpp
cmd.Name("MoeInitRoutingGroupedMatmulGrad")
    .Input(input_grad_y, "inputGradY")
    .Input(input_weight, "inputWeight")
    .Input(expanded_row_idx, "expandedRowIdx")
    .Input(group_list, "groupList")
    .Output(output_grad_x, "outputGradX")
    .Output(output_grad_weight, "outputGradWeight")
    .Run();
```

## 3. Output shape computation

Pre-allocate in host according to infer logic:

```cpp
const int64_t batch = expanded_row_idx.numel() / topk;
const int64_t k = input_grad_y.size(1);
at::Tensor output_grad_x = at::empty({batch, k}, input_grad_y.options());
at::Tensor output_grad_weight = at::empty_like(input_weight);
```

## 4. Prerequisites

- xpu_kernel's `moe_init_routing_grouped_matmul_grad` must be built and installed
- CANN runtime must be able to load the graph op
- If not installed, `OpCommand::Run()` will fail with "op not found"

## 5. Tiling constraints (optional)

If the operator has tiling constraints (e.g. `totalLength % 128 == 0`), ensure test shapes satisfy them; otherwise execution may fail.

## 6. Reference implementations

- op-plugin: `examples/moe_init_routing_grouped_matmul_grad_extension/`
- xpu_kernel: `xpu_kernel/C_like/transformer/npu/moe_init_routing_grouped_matmul_grad/`

## 7. Common pitfalls

| Issue | Resolution |
|-------|------------|
| `os.path.F_OK` error | Use `os.F_OK`, `os.X_OK` |
| Graph op not found | Ensure xpu_kernel is built and installed |
| Output shape mismatch | Check infer_shape vs host allocation logic |
| Attr type ambiguity | Use `static_cast<int64_t>(...)` for integer Attrs |
