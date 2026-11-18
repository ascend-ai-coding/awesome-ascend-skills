# TileKernels RMSNorm in FSDPTurbo

Use this example when connecting the plain RMSNorm kernel to FSDPTurbo. The
paths below come from an FSDPTurbo TileKernels benchmark snapshot.
Recheck them and the installed TileKernels build in the target checkout.

## Find the insertion point

1. Read FSDPTurbo's `AGENTS.md` and its referenced development guide,
   `.agent/skills/fsdp-turbo-dev-guide/SKILL.md` in the benchmark snapshot.
   Use the repository's semantic `ops/` wrapper, device registration and
   CPU fallback. Keep third-party imports in the backend and follow the
   guide's constraints on modeling changes and its delivery self-check.
2. Start at `fsdp_turbo/ops/rms_norm.py::rms_norm`. It dispatches the
   `rms_norm` key and forces the registered Torch implementation when
   `use_eager=True`. Read `ops/registry.py`, `ops/__init__.py` and the
   backend package imports to verify registration order. In the snapshot,
   the registry has one slot per operator/device pair, so a second
   registration for `('rms_norm', 'npu')` replaces the CANN implementation.
3. Trace the requested consumer. `fsdp_turbo/modules/rms_norm.py::RMSNorm`
   calls the wrapper directly. DeepSeek V4.1's text and vision RMSNorms in
   `fsdp_turbo/models/deepseek/deepseek_v41/{layers,vision}.py` reach it
   only on their NPU branches in this snapshot. A CUDA backend registration
   alone therefore leaves those callers on their existing path. The separate
   adapter in `examples/deepseekv4/modeling_deepseek_v4.py` casts the weight
   passed to the wrapper to the activation dtype. Check the effective inputs
   for the caller being integrated.
4. Read `fsdp_turbo/ops/npu/training_benchmark.py::_tilekernels` and
   `tests/performance/tilekernels_950/P2/rms_norm/` for the existing
   benchmark adapter and fixtures. The benchmark calls forward and backward
   separately. Production training needs an autograd-connected wrapper.

## Wire the selected backend

- For an opt-in NPU integration with the snapshot registry, keep the existing
  `rms_norm` registration. One route is to select TileKernels inside the
  current NPU adapter and call the existing CANN function for cases outside
  the selected backend's support. Reuse an existing selector if the target
  checkout provides one. With the feature disabled, preserve NPU CANN,
  CUDA TransformerEngine behavior and Torch eager dispatch. Keep the optional
  TileKernels import local to the selected backend so CPU-only import still works.
- Inspect the matching `tile_kernels.quant.norm_forward_kernel` and
  `tile_kernels.quant.norm_backward_kernel` and their tests. The benchmark's
  plain RMSNorm branch calls `norm_forward(..., with_rstd=True)`, saves
  its input and reciprocal RMS, then supplies those and weight to
  `norm_backward`. Verify supported devices, ranks, row/hidden sizes,
  layouts, dtypes and gradient layouts for both passes. If the binding
  expects 2D contiguous rows, adapt the leading dimensions and restore
  the original shape. Check the upstream backward before accepting a shape.
- DeepSeek V4.1 can use BF16 activations with FP32 weight. The benchmark
  adapts that case by casting activations and upstream gradients to FP32
  and returning BF16 output and input gradient. Validate this approach
  against the production baseline and keep the weight and its gradient
  FP32. Use the existing path for mixed precision until an adaptation is
  verified. Keep residual-add and quantized RMSNorm out of plain RMSNorm scope.

## Verify through the wrapper and caller

- Use one small supported same-dtype case through `ops.rms_norm`.
  Check the selected kernel actually runs and compare output, input
  gradient and weight gradient against the existing path with a
  nonuniform upstream gradient.
- Add the caller's BF16/FP32 case and a representative leading-dimension
  or layout case if the caller can produce it. Check shape/device/dtype,
  weight identity and, when module changes are needed, state-dict keys.
  Reuse `tests/unit_tests/models/test_deepseek_v4_rms_norm.py` for its
  parameter and gradient assertions at the module boundary. That test
  extracts the separate example adapter's forward and selects the CPU path,
  so add an explicit mixed-input backend test for a caller that passes FP32
  weight directly.
- Test disabled behavior, `use_eager=True`, and any supported fallback when
  the optional kernel is missing or the input is outside its contract.
  Confirm the selected model or module reaches the intended wrapper branch.
  If the target accelerator is unavailable, record which native checks
  are blocked and which routing or reference checks actually ran.
