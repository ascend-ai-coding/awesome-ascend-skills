# Phase 2 — Design (`design.md`)

The design document is the **single source of truth** for code generation. Code-gen reads
`design.md` and nothing else for the operator's intent. Fill
[`../templates/design-template.md`](../templates/design-template.md) and write it to
`csrc/ops/<op>/design.md`.

## Step 1 — Choose an implementation path

| Path | Use when | Notes |
|---|---|---|
| **AscendC Kernel** (default) | vector / elementwise / row / reduction / index / sort / pool | This skill's main path |
| CATLASS template lib | matmul / cube-heavy | only for GEMM-style operators |
| aclnn wrap | a CANN built-in already implements it | not custom development |

State the reason for the choice in `design.md`.

## Step 2 — Map the math to an AscendC API sequence

Translate the formula into a sequence of AscendC vector APIs (see
[`04a-kernel-api.md`](04a-kernel-api.md) for the API catalog). Example mapping:

| Math | AscendC API |
|---|---|
| `y = a + b` | `AscendC::Add` |
| `y = a * b` | `AscendC::Mul` |
| `y = exp(x)` | `AscendC::Exp` |
| `y = max(x, 0)` | `AscendC::Maxs(y, x, 0)` |
| `s = sum(x)` | `AscendC::ReduceSum` (backup source first) |
| `m = max(x)` | `AscendC::ReduceMax` (backup source first) |

Write the per-tile compute as pseudocode in `design.md`.

## Step 3 — Two-level tiling

AscendC operators use a **two-level tiling** strategy to exploit hardware parallelism.

```
GM (totalLength)
   │  Block-level tiling (inter-core): split across AI cores
   ▼
Core 0 .. Core N-1   (former cores: formerLength; tail core: tailLength)
   │  UB-level tiling (intra-core): loop tiles of tileLength through UB
   ▼
UB tile (tileLength)  →  CopyIn → Compute → CopyOut
```

### Block-level tiling (inter-core)

Balance work across cores with a former/tail split; align each core's chunk to a cache
line (512 bytes).

| Param | Formula |
|---|---|
| `totalLengthCore` | `(totalLength + CORE_NUM - 1) / CORE_NUM` |
| `totalLengthCoreAlign` | `(totalLengthCore + 511) / 512 * 512` |
| `usedCoreNum` | `(totalLength + totalLengthCoreAlign - 1) / totalLengthCoreAlign` |
| `formerNum` | `usedCoreNum - 1` |
| `formerLength` | `totalLengthCoreAlign` |
| `tailNum` | `1` |
| `tailLength` | `totalLength - formerNum * formerLength` |

Validate: `formerNum * formerLength + tailNum * tailLength == totalLength`.

### UB-level tiling (intra-core)

| Param | Formula |
|---|---|
| `bufferCoefficient` | from UB allocation table (sum of all buffer "total size" / `tileLength`) |
| `maxTileElements` | `UB_SIZE_LIMIT / bufferCoefficient` (UB size from platform API) |
| `alignElements` | `32 / dtypeSize` |
| `tileLength` | `(maxTileElements / alignElements) * alignElements` |

Tiling struct (typical):

```cpp
struct <Op>TilingData {
    int64_t totalLength;
    int64_t formerNum;
    int64_t formerLength;
    int64_t tailNum;
    int64_t tailLength;
    int64_t tileLength;     // UB single-pass length
};
```

## Step 4 — UB allocation table + FP16/BF16 up-cast

**The NPU vector unit cannot compute on float16/bfloat16 directly** — up-cast to float32,
compute, then down-cast. This needs extra float32 buffers, which raise `bufferCoefficient`.

| Buffer | Size (bytes) | Count | Total |
|---|---|---|---|
| inQueueX | `tileLength * dtypeSize` | `BUFFER_NUM` (=2) | … |
| outQueueZ | `tileLength * dtypeSize` | `BUFFER_NUM` | … |
| tmp fp32 (fp16/bf16 only) | `tileLength * 4` | 1+ | … |
| **Total** | — | — | **derive `bufferCoefficient`** |

| Input dtype | Handling | Compute precision | UB impact |
|---|---|---|---|
| float16 | up-cast to float32 | float32 | extra fp32 buffer |
| bfloat16 | up-cast to float32 | float32 | extra fp32 buffer |
| float32 | compute directly | float32 | none |

When the coefficient differs by dtype, branch in op_host:
`if (dtypeSize == 2) { ... } else { ... }`.

## Step 5 — Workspace

| Operator class | Workspace size |
|---|---|
| elementwise | `SYSTEM_WORKSPACE_SIZE` (typically 16 MB) |
| others | `sizeof(<Op>TilingData)` (or via `GetLibApiWorkSpaceSize()` when API tmp is used) |

## Tiling by operator type

The block-level split above is the elementwise default. Other operator families adjust
the **split axis** and **per-tile layout**:

- **Elementwise** (ReLU, GELU, Add): flat split over total elements; tile = contiguous
  span; both inputs/outputs share the same layout.
- **Reduction / row** (LayerNorm, Softmax, RMSNorm): split by **rows**; each core owns
  whole rows; the reduction axis must fit in UB (or use a two-pass reduce). Pad the
  reduction axis; back up the source before `ReduceSum/ReduceMax`.
- **Index** (IndexSelect / Gather / Scatter): split by output rows; `index` tensor is
  int32/int64; per-row (shared 1D index) vs per-elem (row-wise index) variants pick
  different templates.
- **Sort / TopK**: split by rows; sort works on aligned blocks; mind `repeatTime <= 255`
  for high-dim split APIs.
- **Pooling** (Max/Avg Pool, NDHWC): split by output spatial tiles; window/stride drive
  the input halo region read into UB.

## Hardware constraints to respect

- UB is on-chip and small (example limit ~192 KB; **always query via platform API**).
- 32-byte UB alignment; 512-byte cache-line alignment for inter-core chunks.
- `BUFFER_NUM = 2` enables double buffering (pipeline overlap of copy and compute).
- High-dim split APIs take `repeatTime` as `uint8_t` → values > 255 silently truncate.

## Gate (must pass before Phase 3)

- [ ] Function signature + supported dtypes recorded.
- [ ] Implementation path chosen with reason.
- [ ] Per-tile compute pseudocode (API sequence) written.
- [ ] Two-level tiling params + tiling struct defined.
- [ ] UB allocation table filled; `bufferCoefficient` derived per dtype.
- [ ] FP16/BF16 up-cast path described.
- [ ] Workspace sizing decided.

## Anti-patterns (NEVER)

- NEVER design without an explicit UB allocation table — code-gen needs it.
- NEVER plan FP16/BF16 to compute directly through complex math.
- NEVER hardcode core count or UB size in the design — they come from the platform API.
