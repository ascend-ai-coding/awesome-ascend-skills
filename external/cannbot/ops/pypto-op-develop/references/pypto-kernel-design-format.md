# PyPTO kernel source layout and design format

This document defines a **file organization pattern** and **documentation conventions** for PyPTO custom kernels. It applies to any operator (elementwise, reduction, attention-like recurrence, fusion, etc.): use the layers that match your complexity and omit the rest.

Goals:

- Human readers and LLM agents can **navigate by role** (reference vs PyPTO vs host glue vs test).
- **Debugging** stays tractable: each numbered stage matches a small function with a clear contract.
- **Portability**: swap math, tiling, or fusion without losing the overall structure.

**Canonical Python skeleton (complex-kernel workflow):** skill `pypto-op-develop`'s `templates/impl_template.py` — agents **must** use this file as the starting layout for **each** staged module file and the **full** kernel under `custom/<op>/` (see **skill `pypto-orchestration-manual`'s `references/rules.md`** rule **17** and **skill `pypto-orchestration-manual`'s `references/rules.md`** rule **17**). This document (layers A–L) and that template stay aligned.

---

## 1. Recommended vertical layers (top to bottom)

Order sections in the source file roughly as follows. Each layer only depends on layers above it.

| Layer | Purpose | Typical names (examples) |
| --- | --- | --- |
| **A. Utilities** | Reusable diagnostics (optional). | `tensor_compare_report`, logging helpers |
| **B. Small math building blocks** | Pure PyTorch (or numpy) fragments of the algorithm, reused by reference and sometimes mirrored on device. | `norm_fwd`, `softmax_chunk`, … |
| **C. Forward reference** | Ground-truth forward in PyTorch, written under **explicit constraints** (see §3). | `forward_ref` |
| **D. Host-side constants** | Matrices and masks that replace forbidden ops in the reference (e.g. prefix sum via `matmul`). | `make_chunk_constants` |
| **E. Backward reference (decomposed)** | Private helpers that mirror one loop-body or one pipeline stage; the full backward stitches them. | `_slice_chunk_inputs`, `_stage_attn`, … |
| **F. Golden backward** | Single entry that implements the full backward in PyTorch for numeric verification. | `torch_golden_*_backward_ref` |
| **G. Cache / bridge** | Convert Python lists, nested caches, or layouts into flat tensors for the NPU path. | `prepare_cache_for_npu`, … |
| **H. PyPTO sub-kernels** | Small, named PyPTO regions: views, matmuls, fused stages. One conceptual step per function. | `pypto_slice_inputs`, `pypto_fused_stage_ab`, … |
| **I. Kernel implementation** | The actual `pypto.loop` nest, calling (H); no `@jit` here if you split impl vs entry. | `_your_op_kernel_impl` |
| **J. JIT entry** | `@pypto.frontend.jit` function: tensor signatures, `runtime_options`, `debug_options`; delegates to (I). | `your_op_kernel_npu` |
| **K. Host wrapper** | Allocates outputs, packs torch tensors, calls (J), reshapes results to user layout. | `host_wrapper` |
| **L. Driver / test** | `main()` or pytest: config, forward ref, golden, reshape for PyPTO, compare. | `main` |

Not every kernel needs every layer. A minimal unary op might skip (C), (G), and most of (E)–(H); a training backward kernel typically needs (C)–(L).

---

## 2. Naming conventions (apply to any kernel)

| Pattern | Meaning |
| --- | --- |
| `forward_ref` / `*_forward_ref` | PyTorch reference for the forward pass. |
| `torch_golden_*` / `*_golden_*` | Full reference for backward or end-to-end numeric check. |
| Leading `_` | Private helper: one logical step, not intended as the public API. |
| `pypto_*` | Code that uses `pypto` APIs (views, matmul, loops, tile shapes). |
| `*_kernel_impl` | Implementation body containing `pypto.loop` and sub-kernel calls. |
| `*_kernel_npu` (or `*_jit`) | The `@pypto.frontend.jit` entry point with typed tensor signatures. |
| `host_wrapper` (or `launch_*`, `run_*`) | Torch-side launcher: layout conversion + `kernel_npu(...)` + reshape. |

Keep **one primary “golden” name** per direction (e.g. one backward golden) so tests and docs stay grep-friendly.

---

## 3. Reference implementation constraints (document in code)

The reference is not “any PyTorch”; it should follow rules you **state in a header comment**, for example:

- Allowed: `matmul`, elementwise ops, `sum` over last dim, explicit loops over batch/head/chunk.
- Disallowed (if they complicate NPU lowering or differ from your PyPTO path): `cumsum`, `masked_fill`, `tril`/`triu` factory ops, `flip`, or high-rank tensors beyond what the device supports.

Reproduce the same math using **explicit matrices** (e.g. `C_cum @ x` instead of `cumsum`) when the PyPTO side uses that pattern. This keeps **diff debugging** one-to-one between reference steps and `pypto_*` blocks.

---

## 4. Stage markers inside long functions

For long `forward_ref` or backward golden loops, use **labeled stages** so they map to PyPTO modules:

```text
# ===== (A) stage description =====
# ===== (B) stage description =====
# ===== (C) stage description =====
```

Use the **same letters or names** in the PyPTO fusion comments (e.g. “Fused Module A + B”) so a mismatch narrows to one pair of functions.

---

## 5. Contract tables (fill for your kernel)

Maintain a short **tensor contract** (in the module docstring or a dedicated comment block):

| Name | Shape (symbolic) | dtype | Producer | Consumer | Notes |
| --- | --- | --- | --- | --- | --- |
| … | … | … | forward_ref / cache | backward ref / pypto | e.g. “flattened BT×BT per chunk” |

For **forward → backward** dependencies, list **cache keys** and whether each tensor is required for backward:

| Cache key | Shape | Needed for bwd? |
| --- | --- | --- |
| … | … | yes / no |

---

## 6. Mapping reference steps to PyPTO

Use a **one-to-many table** (in documentation, not necessarily code):

| Ref helper / stage | PyPTO function(s) | Notes |
| --- | --- | --- |
| `_ref_stage_alpha` | `pypto_stage_alpha` | Same math, different layout or views. |
| … | … | … |

When precision diverges, compare **stage outputs** (saved tensors) instead of only the final output.

---

## 7. PyPTO sub-kernel responsibilities

Each `pypto_*` function should:

1. **Do one thing** (e.g. “build decay matrix”, “fuse local attn + recurrence”).
2. **Set tile / pass options locally** if needed (`set_vec_tile_shapes`, `set_pass_options`, `set_cube_tile_shapes`) and document why.
3. **Return** all tensors the next stage needs (avoid hidden globals).

The `*_kernel_impl` then reads as a **high-level recipe**: slice → stage1 → stage2 → write outputs.

---

## 8. JIT entry vs implementation split

The boundary between Layer I, J, and K is **strict**. Every layer owns specific responsibilities and must not encroach on the next:

| Layer | Owns | Forbidden |
|---|---|---|
| **I** `_kernel_impl` | All `pypto.loop(...)` calls, `pypto.view(...)` slicing, calls to `pypto_*` sub-kernels (Layer H), `pypto.loop_axis()` | `@jit` decorator, JIT signatures, `runtime_options` |
| **J** `*_kernel_npu` | `@pypto.frontend.jit`, static signatures (`pypto.DYNAMIC` / `pypto.STATIC`), `runtime_options`, `debug_options` | Logic — body is a one-line call to Layer I |
| **K** `host_wrapper` / `<op>_module<k>_wrapper` | torch tensor packing, output buffer allocation, single JIT call, output reshape | `pypto.loop`, Python `for ... in range(...)` calling the kernel, any `pypto.set_*_tile_shapes` |

This split makes it easier to **swap options** or **reuse the impl** from tests without recompiling different JIT shells.

---

## 9. Host wrapper (`host_wrapper`)

The Layer K host wrapper has exactly **four** responsibilities — no more, no less:

1. Move/cache tensors to the **device and layout** the JIT entry expects (flatten, transpose, `expand`, dtype).
2. **Allocate** output buffers with torch (`torch.empty(...)` / `torch.empty_like(...)`).
3. Invoke `*_kernel_npu(*inputs, *outputs)` **exactly once**.
4. **Reshape** outputs back to the user-facing layout (e.g. `[B, T, H, D]`).

Keep I/O reshaping **out** of the JIT function when possible.

### 9a. Anti-pattern: Python `for ... in range(...)` driving the kernel per chunk

A common mistake is to chunk the input on the host with a Python loop and call the JIT kernel once per chunk. This collapses the kernel into a single-chunk operation and only the first iteration actually runs through the PyPTO compilation path; subsequent calls re-launch identical compiled artifacts. **The chunk iteration belongs inside Layer I as `pypto.loop(NT)` + `pypto.view(..., offsets=[...])`.**

```python
# ❌ BAD — Layer K driving the chunk loop from Python
def attention_module1_wrapper(q, k, v):
    out = torch.empty_like(...)
    for chunk in range(NT):                                # Python loop in Layer K
        q_c = q[chunk*BT:(chunk+1)*BT]
        k_c = k[chunk*BT:(chunk+1)*BT]
        out_c = out[chunk*BT:(chunk+1)*BT]
        attention_kernel_npu(q_c, k_c, v, out_c)           # per-chunk JIT call
    return out

# ✅ GOOD — single call; iteration lives in Layer I
def attention_module1_wrapper(q, k, v):
    out = torch.empty_like(...)
    attention_kernel_npu(q, k, v, out)                     # ONE call
    return out

def _attention_kernel_impl(q, k, v, out):
    pypto.loop(NT)
    nt = pypto.loop_axis()
    q_chunk = pypto.view(q, [BT, D], offsets=[nt*BT, 0])   # [BT, D]
    # ... per-chunk work via pypto.view ...
```

This pattern blocks the per-Phase / cleanup / verification gates.

### 9a-2. Inside the JIT graph, the only loop is `pypto.loop`

§9a governs Layer K (the host wrapper). The complementary constraint governs the **JIT graph** — the `@pypto.frontend.jit` entry plus every function it calls (Layer I `_kernel_impl`, Layer H `pypto_*` sub-kernels, and any helper). Inside that JIT-reachable code the only permitted loop is `pypto.loop(...)` / `pypto.loop_unroll(...)` (`for ... in range(...)` is also allowed). Any other Python loop — `for ... in <list>`, a `while`, or a comprehension that issues `pypto.*` ops — is forbidden.

This holds even for a **static, small unroll**. Iterating a tensor axis (e.g. `for g in range(num_groups): pypto.view(...); pypto.assemble(...)`) must be `pypto.loop(num_groups)`; and a structural block decomposition (e.g. a block-wise matrix inverse) must also use `pypto.loop` (add `submit_before_loop=True` when iterations are dependent), not a Python `for`. A Python `for`/`while` in JIT code unrolls the iteration into the graph at construction time, defeats the framework's tiling / parallelism, and bloats compile time.

### 9b. Reshape inputs so the kernel's core compute stays low-rank (host-side `reshape`)

The host wrapper may use **torch** `reshape` / `squeeze` / `unsqueeze` freely — it runs before the JIT call, on raw torch tensors. Use this to fold leading batch dimensions so the kernel's core op (typically `pypto.matmul`) runs in its natural 2D form, then reshape the output back to the user layout.

When the math is a batched 2D contraction — leading batch axes plus one matmul, e.g. `out[..., m, n] = sum_k a[..., m, k] * w[k, n]` — collapse the batch axes on the host instead of up-ranking the other operand to match:

```python
# ✅ GOOD — collapse leading batch dims on the host so the matmul is 2D
def op_wrapper(a, w):                       # a: [B0, B1, M, K]; w: [K, N]
    *batch, M, K = a.shape
    N = w.shape[1]
    a2d = a.contiguous().reshape(-1, M, K)  # [B0*B1, M, K] — no-copy for row-major
    out = torch.empty(a2d.shape[0], M, N, dtype=a.dtype, device=a.device)
    op_kernel_npu(a2d, w, out)              # one collapsed loop axis; matmul is [M,K]@[K,N]
    return out.reshape(*batch, M, N)        # restore the user-facing rank
```

```python
# ❌ BAD — up-ranking an operand to match the input rank, forcing a high-rank matmul
def op_wrapper(a, w):                       # a: [B0, B1, M, K]; w: [K, N]
    w4d = w.unsqueeze(0).unsqueeze(0)       # [1, 1, K, N] only to "match" a's rank
    op_kernel_npu(a, w4d, out)              # nested dynamic loops + degenerate [1,1,...] matmul
```

Why the GOOD form is preferred: `pypto.matmul` is naturally 2D, so degenerate `[1, 1, ...]` leading dims add tiling complexity for no benefit, and folding the batch axes turns several nested dynamic loops into one. Reshaping the leading dims of a row-major-contiguous tensor is a no-copy view and the output reshape is its exact inverse, so correctness is preserved.

### 9c. Blocked scan / cumulative (carry) pattern — for UB-exceeding scan/reduction only

**Scope:** use this ONLY when a *native* scan/cumulative op (`pypto.cumsum`, `pypto.cumprod`, …) — or a native reduction — runs along an axis large enough that the op's single-op UB footprint (data + internal workspace) exceeds the UB budget. Example: `pypto.cumsum` over a 4000-long axis needs ~256 KB internal workspace > 192 KB UB. For ops that already fit UB, do not introduce this machinery.

Split the large axis into blocks of length `T`, chosen so the op's per-block UB fits the budget (cumsum: `T=1000` → ~64 KB < 192 KB). Loop over the *few* blocks, view a real `[.., T]` block, apply the native op on the block, and propagate a carry/accumulator across blocks:

```python
# ✅ GOOD — block the scan axis; native op per block; carry across blocks
def _op_kernel_impl(x, out):                 # x, out: [B, D] (D large, e.g. 4000)
    B = x.shape[0]                           # SymbolicScalar (dynamic batch)
    D = 4000; T = 1000; NB = D // T          # block length T fits per-block UB; NB small (4)
    for b in pypto.loop(B, name="batch"):
        carry = pypto.full([1, 1], 0.0, pypto.DT_FP32)            # running accumulator
        # block loop carries a cross-block dependency -> submit_before_loop=True (only NB iters)
        for t in pypto.loop(NB, name="block", unroll_list=[1], submit_before_loop=True):
            blk = pypto.view(x, [1, T], [b, t * T])               # real [1, T] block (NOT [1,1])
            scan = pypto.cumsum(blk, dim=1)                       # native op on the block
            scan[:] = scan + carry                                # fold in running carry ([1,1]->[1,T])
            pypto.assemble(scan, [b, t * T], out)
            carry[:] = pypto.view(scan, [1, 1], [0, T - 1])       # block tail = new running total
```

```python
# ❌ BAD #1 — native op on the whole axis at once  → UB OOM (F40005: 256000 > 196608)
scan = pypto.cumsum(x, dim=1)               # 4000-long workspace blows the UB budget

# ❌ BAD #2 — element-by-element loop over the large axis  → task explosion / host crash / timeout
for j in pypto.loop(D, submit_before_loop=True):     # D=4000 iters × B  = ~512K tasks
    x_col = pypto.view(x, [1, 1], [b, j])            # one scalar per iter (SLAB_ADD_CACHE_FAILED + segfault)
```

This is a **loop/view structure** rule, separate from vec-tile sizing. The block length `T` is the *view* shape, NOT the vec tile: the kernel above passes with a small vec tile (e.g. `set_vec_tile_shapes(16, 16)`) unchanged — correctness comes from the blocking structure, not the tile value. Do not couple the tile size to `T`, and do not change the normal vec-tile rules ([16, 64] per axis, rank match, single-op UB fit).

---

## 10. Test driver (`main` or pytest)

Suggested structure:

1. **Config**: shapes, dtypes, chunk size `BT`, seeds, device id, `run_mode`.
2. **Inputs**: random tensors with `requires_grad` if testing autograd-related paths.
3. **Constants**: `make_chunk_constants` or equivalent.
4. **Forward reference**: run `forward_ref`, get outputs + cache.
5. **Upstream grads**: random `do`, `dht`, etc.
6. **Golden backward**: `torch.no_grad()` + golden function.
7. **PyPTO path**: adapt cache tensors to PyPTO layout; call `host_wrapper`.
8. **Compare**: a detailed per-tensor report helper (or per-tensor `torch.allclose`) for each output.

---

## 11. Shape Annotation Convention

Every tensor assignment and tiling configuration line **must** carry an inline shape comment.

### 11b. `pypto.loop(1)` is a layout-check escape hatch — not a default wrapper

Some agents wrap their kernels in `pypto.loop(1)` "to be safe" — even when an inner `pypto.loop(N)` already exists. This is wrong. The only legitimate use of `pypto.loop(1)` is the case where the kernel has **no other** `pypto.loop` call and the layout check would otherwise complain about a vector-pipe op having no loop.

```python
# ❌ BAD — redundant pypto.loop(1) around an inner pypto.loop(N)
def _kernel_impl(...):
    pypto.loop(1)                                  # adds nothing, redundant wrapper
    pypto.loop(NT)
    ...

# ✅ GOOD — no wrapper when there is already a real loop
def _kernel_impl(...):
    pypto.loop(NT)
    ...

# ✅ GOOD — vector-pipe simple op with no other loop, wrapper required by the layout check
def _kernel_impl(...):
    pypto.loop(1)                                  # only loop call, satisfies the layout check
    ...
```

Rule of thumb:

| Pipe class | `pypto.loop(1)` policy |
|---|---|
| **Vector pipe**, simple op (no inner loop) | **REQUIRED** as the sole `pypto.loop` call |
| **Vector pipe**, op with `pypto.loop(N)` already present | **FORBIDDEN** |
| **Cube pipe** | Same as vector pipe: required iff no other `pypto.loop` exists |
| **Mixed cube + vec body** | Same as vector pipe: required iff no other `pypto.loop` exists |

### 11c. Tile shape scope — per-stage, not per-kernel

`set_vec_tile_shapes` and `set_cube_tile_shapes` change the active tile layout for every PyPTO call that follows them in the same JIT body. When a kernel has multiple stages with different tile-optimal shapes (e.g. one matmul wants `[64, 128]` cube tiles and another wants `[128, 64]`), setting tiles **once at the top of `_kernel_impl`** forces every stage to share the same tiles and loses optimization headroom. The example values below illustrate scope only; concrete tile values come from DESIGN.md §3.2.5.

The recommended pattern is to push the tile-shape call **into each `pypto_*` sub-kernel** so each stage sets its own optimal layout:

```python
# ❌ BAD-ish — one global tile shape, multiple stages
def _kernel_impl(q, k, v, out):
    pypto.set_cube_tile_shapes([128,128], [128,128], [128,128])
    pypto.loop(NT)
    a = pypto_stage_alpha(q, k)   # would prefer [64,128] tiles
    b = pypto_stage_beta(a, v)    # would prefer [128,64] tiles

# ✅ GOOD — per-stage local tiles
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
```

Avoid a `_kernel_impl` body that holds a global `set_*_tile_shapes` call AND dispatches to two or more `pypto_*` sub-kernels — configure the tile shapes inside each stage instead. If the kernel has only one stage that needs tile config, the global call is fine.

### Original — tile shape comment convention

**1. Every tensor assignment gets a shape comment**

```python
q = pypto.view(q_in, [B, N, Sq, D])          # [B, N, Sq, D]
k = pypto.view(k_in, [B, N, Skv, D])         # [B, N, Skv, D]
scores = pypto.matmul(q, k, pypto.DT_BF16)   # [B, N, Sq, Skv]
```

**2. Matmul uses the contraction form**

```python
# [M, K] @ [K, N] -> [M, N]
out = pypto.matmul(a, b, pypto.DT_BF16)      # [M, N]
```

> **Prefer the 2D contraction form.** When inputs carry leading batch axes,
> collapse them on the host (see §9b) instead of feeding
> `[..., M, K] @ [..., K, N]` with degenerate `[1, 1]` leading dims. Keep
> matmul operands 2D unless a genuine batched matmul over non-degenerate
> batch axes is actually required.

**3. Tile config lines show tile shape**

```python
pypto.set_vec_tile_shapes(1, 1, 8, 8)                         # tile dimensions per DESIGN.md §3.2.5
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128]) # each list is [L0, L1]
```

> Tile shape 具体值、参数格式规则和约束详见 skill `pypto-op-design` SKILL.md "第 2 轮：Tiling 推导" + quick_ref.md。

**4. Loop body tensors show the slice shape, not the full shape**

```python
for i in pypto.loop(range(Sq // tile_s), idx_name="i"):
    q_tile = pypto.view(q, [tile_s, D], ...)  # [tile_s, D]  (slice of [Sq, D])
    s_tile = pypto.matmul(q_tile, k_t, ...)   # [tile_s, Skv]
```

**5. Dynamic axes use the symbolic name, not `?`**

```python
x = pypto.view(x_in, [B, S, H])              # [B, S, H]  S=dynamic
```

**6. Reductions annotate both input and output shape**

```python
row_max = pypto.amax(scores, dim=-1)          # [B, N, Sq, Skv] -> [B, N, Sq, 1]
```

**Do not annotate:** import lines, `print`/logging, or plain Python scalars (`tile_m = 16`).

---

## 12. Checklist for new kernels (generic)

- [ ] Math spec and **symbolic shapes** written down.
- [ ] `forward_ref` (if applicable) respects **documented constraints**.
- [ ] Backward golden matches forward cache contract.
- [ ] Each non-trivial loop body chunk extracted as `_helper` or `pypto_*` with a **one-line role comment**.
- [ ] Ref stage labels align with PyPTO module comments.
- [ ] JIT signatures match actual buffer ranks (dynamic vs static dimensions).
- [ ] Host wrapper documents **layout** assumptions (row-major flatten order, head grouping, etc.).
- [ ] Test compares **all** outputs relevant to the API.

---

## 12. Minimal template (skeleton only)

```text
# --- utilities (optional) ---

# --- small torch helpers ---

# --- forward_ref (constraints in header comment) ---

# --- constants ---

# --- backward ref helpers (_*) ---

# --- torch_golden_* backward ---

# --- prepare_* cache bridge ---

# --- pypto_* sub-kernels ---

def _my_kernel_impl(...):
    for ... in pypto.loop(...):
        ...
        # call pypto_* helpers

@pypto.frontend.jit(...)
def my_kernel_npu(... typed tensors ...):
    _my_kernel_impl(...)

def host_wrapper(... torch tensors ...):
    # allocate, pack, call my_kernel_npu, reshape
    ...

def main():
    # config → forward_ref → golden → host_wrapper → compare
```

Adapt depth: a forward-only inference kernel omits backward/golden sections; a fused elementwise kernel may inline “sub-kernels” into a single `pypto_*` or the impl.
