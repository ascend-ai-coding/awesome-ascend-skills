# Phase 4a — AscendC kernel API essentials

Core APIs and constraints for op_kernel code. Use this alongside
[`04-code-gen.md`](04-code-gen.md).

## Kernel-side forbidden C++ features

### Standard-library math (banned in kernel)

`std::` math may compile but produces wrong runtime results. Use AscendC APIs:

| Forbidden | Replacement |
|---|---|
| `std::min(a,b)` | `a<b?a:b` (scalar) or `AscendC::Min(dst,s0,s1,n)` |
| `std::max(a,b)` | `a>b?a:b` (scalar) or `AscendC::Max(dst,s0,s1,n)` |
| `std::abs(x)` | `AscendC::Abs(dst,src,n)` |
| `std::sqrt(x)` | `AscendC::Sqrt(dst,src,n)` |
| `std::exp(x)` | `AscendC::Exp(dst,src,n)` |
| `std::log(x)` | `AscendC::Ln(dst,src,n)` |
| `#include <cmath>` | not needed in kernel |

### Dynamic allocation (banned in kernel)

| Forbidden | Replacement |
|---|---|
| `std::vector<T>` | `LocalTensor<T>` + `pipe.InitBuffer` |
| `new/delete`, `malloc/free` | `pipe.InitBuffer` |

### Host/kernel header isolation

| File | May include | Must NOT include |
|---|---|---|
| op_host (*.cpp) | `<cmath>`, `<algorithm>`, tiling headers | `kernel_operator.h` |
| op_kernel (*.cpp) | `kernel_operator.h` | `<cmath>`, `<algorithm>`, tiling headers |

## Data copy

**GM↔UB must use `DataCopyPad`.** `DataCopy` is only for UB↔UB.

| API | Use | Verdict |
|---|---|---|
| `DataCopyPad` | GM↔UB (all cases) | required |
| `DataCopy` | UB↔UB | allowed |
| `DataCopy` GM↔UB | only if `count*sizeof(T)` strictly 32B-aligned | debug/prototype only |
| `GlobalTensor::Set/GetValue` | per-element GM access | banned (extremely slow) |

`DataCopyExtParams{blockCount, blockLen, srcStride, dstStride, 0}`:

| Field | Meaning | Unit |
|---|---|---|
| blockCount | number of blocks (usually rows) | — [1,4095] |
| blockLen | per-block length | **bytes** |
| srcStride | gap between source blocks | **GM=bytes, UB=32B block** |
| dstStride | gap between dest blocks | **GM=bytes, UB=32B block** |

> The most common copy bug: stride unit differs (GM uses bytes, UB uses 32B DataBlocks).

`DataCopyPadExtParams<T>{isPad, leftPad, rightPad, padValue}` (GM→UB only).

Elementwise contiguous copy:

```cpp
AscendC::DataCopyExtParams p{1, (uint32_t)(tileLength * sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(srcLocal, srcGlobal[offset], p);   // GM→UB
inQueue.EnQue(srcLocal);
```

CopyIn / CopyOut must be **consistent**: if CopyIn uses `DataCopyPad`, CopyOut must too,
or rows misalign. For row operators, `srcStride`/`dstStride` carry the padding gap
(`(alignedCols - cols)`), not the full row length.

## Vector compute

Element-wise: `Add, Sub, Mul, Div, Adds, Muls, Maxs, Mins, Abs, Sqrt, Exp, Ln,
Reciprocal, Cast, Duplicate, Compare, Select`.
Reduction: `ReduceSum, ReduceMax, ReduceMin` (template `<T, isReuseSource>`).

- **FP16/BF16 must be Cast to FP32** before complex math, then Cast back.
- Back up the source before reduction (it may modify the source); never alias `dst` with
  `tmpBuffer`:

```cpp
AscendC::Adds(backup, src, 0.0f, len);
AscendC::ReduceSum<float, true>(result, backup, sharedTmp, dimLen);
```

## `repeatTime` overflow (uint8 → max 255)

High-dim split overloads (`Add/Sub/Mul/Div/Cast/Duplicate/Compare/Select/Exp/Ln/...`)
take `repeatTime` as `uint8_t`. Passing 256 silently truncates to 0 (no compute, no
error). Clamp on host (`tileRows = min(tileRows, 255)`) and batch in the kernel:

```cpp
int64_t remaining = rowCount, offset = 0;
while (remaining > 0) {
    uint8_t batch = (uint8_t)(remaining < 255 ? remaining : 255);
    AscendC::Sub(dst[offset], s0[offset], s1, mask, batch, params);
    offset += batch * alignedCols; remaining -= batch;
}
```

## Compare 256-byte alignment

`Compare` needs the region to be a multiple of 256 bytes; pad the remainder (ArgMax →
`-FLT_MAX`, ArgMin → `+FLT_MAX`):

```cpp
uint32_t a = 256 / sizeof(T);
uint32_t aligned = ((count + a - 1) / a) * a;
if (aligned > count) AscendC::Duplicate(src[count], padValue, aligned - count);
```

## Sync / resource management

- `TPipe pipe;` owns UB; `pipe.InitBuffer(queue, BUFFER_NUM, bytes)` allocates.
- `BUFFER_NUM = 2` → double buffering (copy/compute overlap).
- Pair `AllocTensor`/`FreeTensor` and `EnQue`/`DeQue`; a CopyIn `EnQue` must have a
  matching Compute `DeQue`.
- Keep total `InitBuffer` count <= 64; merge buffers if exceeding.

## Alignment quick reference

| Location | Alignment |
|---|---|
| UB (VECIN/VECOUT), L1 | 32 bytes |
| GM | by dtype size |

```cpp
uint32_t elemsPerBlock = 32 / sizeof(T);      // half:16, float:8
uint32_t alignedCols = ((cols + elemsPerBlock - 1) / elemsPerBlock) * elemsPerBlock;
```

## Diagnostic checklist (on kernel compile/run error)

1. Any `std::` math? → replace with AscendC API.
2. GM↔UB using `DataCopyPad`? → required.
3. `repeatTime > 255`? → batch.
4. `Compare` region 256B-aligned? → pad.
5. `ReduceMax/Sum` `dst` distinct from `tmpBuffer`? → separate buffers.
6. Total `InitBuffer` <= 64? → merge.
7. `EnQue`/`DeQue` paired? → synchronize after copy.

## Debug-only

```cpp
AscendC::printf("debug: xGm[0]=%f\n", xGm.GetValue(0));   // debug only, remove in prod
```
