# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""impl_template.py — Starter for ``custom/<op>/modules/<op>_module<suffix_k>_impl.py``.

Per-Phase implementation; copy-and-fill in Layer G..K only.

This template covers the PyPTO side of the kernel: Layer G (cache/bridge),
Layer H (pypto_* sub-kernels), Layer I (kernel impl with pypto.loop),
Layer J (@pypto.frontend.jit entry), and Layer K (host wrapper).

The torch reference (golden) lives in ``golden-template.py``, the canonical golden
skeleton owned by skill pypto-golden-generate (used at scaffolding step A). The
test driver (Layer L) lives in ``test_template.py`` and is also owned by
pypto-op-verifier (scaffolding step C and cleanup E2E).

CRITICAL ANTI-PATTERNS — any of these will FAIL the per-Phase and cleanup
completion gates.

(1) Python ``for`` loop in Layer K (host wrapper) calling kernel per-chunk::

        BAD:
          def <op>_module1_wrapper(q, k, v):
              out = torch.empty_like(...)
              for chunk in range(NT):                    # <- Python loop
                  q_c = q[chunk*BT:(chunk+1)*BT]
                  kernel_npu(q_c, ..., out_c)             # <- per-chunk JIT call
              return out
        GOOD:
          def <op>_module1_wrapper(q, k, v):
              out = torch.empty_like(...)
              kernel_npu(q, k, v, out)                    # <- single call
              return out
          # ...the chunk iteration lives inside _kernel_impl as pypto.loop(NT)
          # plus pypto.view(..., offsets=[nt*BT, ...]).

(2) Wrapping ``pypto.loop(1)`` around an inner ``pypto.loop(N)``::

        BAD:
          def _kernel_impl(...):
              pypto.loop(1)                               # <- redundant
              pypto.loop(NT)
              ...
        GOOD (already has a real loop):
          def _kernel_impl(...):
              pypto.loop(NT)
              ...
        GOOD (no other loop, vector pipe simple op — wrapper required by layout
        check, see kernel-design-format.md §11b):
          def _kernel_impl(...):
              pypto.loop(1)                               # <- only loop in scope
              ...

(3) Single global ``set_*_tile_shapes`` call in Layer I while
    multiple ``pypto_*`` sub-kernels need different tiles::

        BAD:
          def _kernel_impl(q, k, v, out):
              pypto.set_cube_tile_shapes([128,128], [128,128], [128,128])
              pypto.loop(NT)
              pypto_stage_alpha(q, k)                     # wants [64,128] tiles
              pypto_stage_beta(attn, v)                   # wants [128,64] tiles
        GOOD (per-stage local tile shape, see kernel-design-format.md §11c):
          def pypto_stage_alpha(q, k):
              pypto.set_cube_tile_shapes([64,128], [64,128], [64,64])
              return pypto.matmul(q, k, ...)
          def pypto_stage_beta(attn, v):
              pypto.set_cube_tile_shapes([128,128], [128,64], [128,64])
              return pypto.matmul(attn, v, ...)
          def _kernel_impl(q, k, v, out):
              pypto.loop(NT)
              a = pypto_stage_alpha(q, k)
              b = pypto_stage_beta(a, v)

Per-file invariants:
  - @pypto.frontend.jit on every kernel function
  - write-back via [:] / move() / assemble(); never ``out = expr``
  - no ``return`` in JIT body
  - every tensor parameter has pypto.Tensor[...] annotation
  - no Python builtin min()/max() in kernel; use pypto.min/pypto.max
  - ``import pypto`` (and ``import torch_npu`` next to it)
  - wrapper name is ``<op>_module<suffix_k>_wrapper``
  - Layer K host wrapper must NOT drive the kernel with a Python
    ``for ... in range(...)`` (call the JIT entry exactly once)
  - inside the JIT graph (the @pypto.frontend.jit entry + every function it
    calls) the allowed loops are ``pypto.loop(...)``, ``pypto.loop_unroll(...)``
    and ``for ... in range(...)``; ``while`` and non-range Python ``for`` are
    forbidden (add ``submit_before_loop=True`` when iterations are dependent)
  - tensor args precede non-tensor args in JIT signatures

Layer K reshape guidance — collapse leading batch dims on the host so the kernel's
matmul stays 2D. When the math is a batched 2D contraction (leading batch axes +
one matmul, e.g. out[..., m, n] = sum_k a[..., m, k] * w[k, n]), reshape on the host
so pypto.matmul runs as plain [M,K]@[K,N], then reshape the output back. Do NOT
up-rank the other operand to match the input rank::

        GOOD:  *batch, M, K = a.shape; N = w.shape[1]
               a2d = a.contiguous().reshape(-1, M, K)   # [B0*B1, M, K], no-copy
               out = torch.empty(a2d.shape[0], M, N, dtype=a.dtype, device=a.device)
               kernel_npu(a2d, w, out)                  # one loop axis; matmul [M,K]@[K,N]
               return out.reshape(*batch, M, N)
        BAD:   w4d = w.unsqueeze(0).unsqueeze(0)        # [1,1,K,N] just to match a's rank
               kernel_npu(a, w4d, out)                  # degenerate [1,1,...] matmul + nested loops
"""

import pypto
import torch
import torch_npu  # noqa: F401  required for NPU device init


# =============================================================================
# Layer G — Cache / bridge
# Move Python lists, nested caches, or layouts into flat tensors that the
# JIT entry expects. Keep this layer torch-only; do NOT call PyPTO APIs here.
# =============================================================================

def prepare_cache_for_npu(...):
    """Flatten / pack / dtype-cast inputs for the JIT entry. Pure torch."""
    raise NotImplementedError


# =============================================================================
# Layer H — PyPTO sub-kernels (one logical step each)
# Each `pypto_*` helper:
#   1. Does ONE thing (build decay matrix / fuse local attn / ...).
#   2. Sets its OWN tile / pass options when the optimal shape differs from
#      the surrounding context (see anti-pattern (3) in the module docstring).
#   3. Returns every tensor the next stage needs (no hidden globals).
# =============================================================================

def pypto_stage_alpha(...):
    """Stage α (e.g. q@k^T → attn).

    Set this stage's local cube tile shape here when it differs from context.
    """
    raise NotImplementedError


def pypto_stage_beta(...):
    """Stage β (e.g. softmax(attn)·v → out). Set local tile shape if needed."""
    raise NotImplementedError


# =============================================================================
# Layer I — Kernel implementation (no @pypto.frontend.jit here)
# Reads as a high-level recipe: slice → stage1 → stage2 → write outputs.
# Owns all `pypto.loop` calls. Does NOT own `pypto.set_*_tile_shapes` when
# multiple sub-kernels would benefit from different tiles (see anti-pattern (3)).
#
# CRITICAL — `pypto.is_loop_begin` / `pypto.is_loop_end` constraint:
#   If this `_kernel_impl` helper body contains `pypto.is_loop_begin(idx)` or
#   `pypto.is_loop_end(idx)` (e.g. for accumulator first-iter init), you MUST
#   choose one of the following — otherwise the parser raises
#   `F00002, ValueError: Not concrete value` at compile time with no source line:
#     (1) RECOMMENDED — inline the entire body directly into the
#         `@pypto.frontend.jit` entry below (delete this helper).
#     (2) ALTERNATIVE — decorate THIS helper with `@pypto.frontend.function`
#         (tensor args only; non-tensor parameters are NOT currently supported).
#   The default helper-function split in this template is safe ONLY if the body
#   does not call `pypto.is_loop_begin` / `pypto.is_loop_end`.
# =============================================================================

def _your_op_kernel_impl(*tensor_args):
    """Top-level recipe. The pypto.loop count comes from a host-side static
    int (e.g. NT) passed via the JIT signature.

    Open a loop with pypto.loop(NT), read the axis with pypto.loop_axis(), then
    slice each chunk with pypto.view(q, [BT, D], offsets=[nt * BT, 0]) and run the
    per-stage recipe.
    """
    pypto.loop(...)                                                # NT or similar
    nt = pypto.loop_axis()  # noqa: F841  loop axis used to build per-chunk views
    raise NotImplementedError


# =============================================================================
# Layer J — JIT entry
# `@pypto.frontend.jit` boundary. Static signatures, runtime_options, and
# debug_options live here. Body is a thin call to Layer I.
# =============================================================================

@pypto.frontend.jit
def your_op_kernel_npu(
    q: pypto.Tensor[(...,), pypto.f16],
    # ... all tensor arguments first, then non-tensor int knobs ...
    output: pypto.Tensor[(...,), pypto.f16],
):
    """JIT entry. Tensor types declared, no body logic.

    Declare runtime_options / debug_options here as needed, e.g. selecting
    pypto.RunMode.NPU as the run mode.
    """
    _your_op_kernel_impl(q, ..., output)


# =============================================================================
# Layer K — Host wrapper (the public entry of this module file)
# RESPONSIBILITIES (only these three):
#   1. Move tensors to device, set layout (flatten / transpose / .contiguous()
#      / dtype) — pure torch.
#   2. Allocate output buffers via `torch.empty(...)` / `torch.empty_like(...)`.
#   3. Invoke the JIT entry ONCE.
#   4. Reshape outputs back to the user-facing layout.
#
# DO NOT do any of these here:
#   - Python `for ... in range(...)` calling the kernel per chunk
#   - pypto.loop(...) — that is Layer I's job
#   - Any pypto.* call — Layer K stays torch-only
#   - Any torch ARITHMETIC / operator math (torch.matmul / .exp / .sum / `@` /
#     softmax / ...) — Layer K's torch is for LAYOUT / ALLOC / CAST ONLY. The
#     operator's numeric computation MUST live inside the @pypto.frontend.jit
#     graph (Layer H/I, via pypto.* ops). Doing the math in host torch is a
#     dummy-JIT cheat and is not acceptable.
#   - pypto.from_torch(...) — not used in operator development. The
#     @pypto.frontend.jit entry takes RAW torch.Tensor; conversion happens INSIDE
#     the kernel (LaunchKernelTorch → TorchTensorConverter). Pass torch tensors
#     straight through — never wrap them in from_torch. Wrapping raises
#     "RuntimeError: Input tensor is not a valid torch tensor type" because
#     ParseTensorData expects a torch tensor and pypto.Tensor.data_ptr is an int,
#     not callable. Prepare inputs with raw torch.randn(...) etc.; to allocate new
#     tensors inside the kernel use pypto.zeros / pypto.ones / pypto.full.
#
# See the module docstring for the "collapse leading batch dims" reshape example.
# =============================================================================

def your_op_module<suffix_k>_wrapper(*user_inputs):
    """Public wrapper. Must be named exactly ``<op>_module<suffix_k>_wrapper`` so
    that pypto-op-verifier's ``test_<op>_module<suffix_k>.py`` can import it.

    Steps: (1) cache / layout adaptation, (2) allocate output(s) via
    ``torch.empty(...)``, (3) ONE call to the JIT entry
    ``your_op_kernel_npu(*tensor_inputs, output)``, (4) reshape back.
    """
    raise NotImplementedError
