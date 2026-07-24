# AscendC Common Pitfalls

Top 15 most frequent coding mistakes. Check this list before submitting any kernel code.

## 1. 32-Byte Alignment Violation
```
WRONG:  DataCopy(dst, src, 17);  // 17 × 2 bytes (FP16) = 34 bytes, not 32-aligned count
RIGHT:  DataCopy(dst, src, 16);  // 16 × 2 = 32 bytes, aligned

Rule: Element count for DataCopy must make total bytes a multiple of 32.
      For FP16: count must be multiple of 16
      For FP32: count must be multiple of 8
      For INT8: count must be multiple of 32
```

## 2. Forgetting FreeTensor After Processing
```cpp
// WRONG: Memory leak in UB
LocalTensor<half> x = inQueue.DeQue<half>();
Compute(x);
outQueue.EnQue(x);
// Missing: inQueue.FreeTensor(x) if x came from inQueue

// RIGHT:
LocalTensor<half> x = inQueue.DeQue<half>();
LocalTensor<half> y = outQueue.AllocTensor<half>();
Compute(y, x);
outQueue.EnQue(y);
inQueue.FreeTensor(x);  // Release input buffer
```

## 3. TBuf Lifetime Overlap
```cpp
// WRONG: Two tensors from same TBuf active simultaneously
TBuf<QuePosition::VECCALC> tmpBuf;
auto a = tmpBuf.Get<half>();
auto b = tmpBuf.Get<half>();  // Overlaps with a's memory!
Add(a, a, b, len);           // Undefined behavior

// RIGHT: Use separate buffers or sequential lifecycle
auto a = tmpBuf.Get<half>();
Process(a);  // Done with a
// Now safe to reuse tmpBuf
auto b = tmpBuf.Get<half>();
Process(b);
```

## 4. Wrong Cast Direction
```cpp
// WRONG: Direct Cast from FP16 to FP32 may lose data
Cast(dst_fp32, src_fp16, RoundMode::CAST_NONE, len);  // CAST_NONE is lossy

// RIGHT: Use appropriate rounding mode
Cast(dst_fp32, src_fp16, RoundMode::CAST_NONE, len);  // FP16→FP32: CAST_NONE is OK (widening)
Cast(dst_fp16, src_fp32, RoundMode::CAST_ROUND, len); // FP32→FP16: Use CAST_ROUND (narrowing)
Cast(dst_bf16, src_fp32, RoundMode::CAST_RINT, len);  // FP32→BF16: Use CAST_RINT
```

## 5. CompareScalar Type Mismatch
```cpp
// WRONG: Scalar type must match tensor type
LocalTensor<half> x;
float threshold = 0.5f;
CompareScalar(mask, x, threshold, CMPMODE::GT, len);  // float vs half!

// RIGHT: Use same type
half threshold = half(0.5f);
CompareScalar(mask, x, threshold, CMPMODE::GT, len);
```

## 6. Select API Parameter Order
```cpp
// WRONG: Common parameter order confusion
Select(dst, mask, true_val, false_val, SELMODE::VSEL_TENSOR_SCALAR_MODE, len);

// RIGHT: Check mode carefully
// VSEL_TENSOR_TENSOR_MODE: Select(dst, mask, src0_true, src1_false, mode, len)
// VSEL_TENSOR_SCALAR_MODE: Select(dst, mask, src_tensor, scalar, mode, len)
```

## 7. Repeat/Stride Calculation Errors
```cpp
// For DataCopy with repeat:
DataCopyExtParams params;
params.blockCount = repeatTimes;
params.blockLen = dataLen * sizeof(T) / 32;  // In 32-byte blocks
params.srcStride = srcGap * sizeof(T) / 32;  // Gap between repeats (32-byte units)
params.dstStride = dstGap * sizeof(T) / 32;

// Common mistake: Using element count instead of 32-byte block count for stride
```

## 8. Missing PipeBarrier When Needed
```cpp
// WRONG: No sync between write and read of same buffer
Compute(tmpBuf);     // VECTOR writes to tmpBuf
DataCopy(gm, tmpBuf, len);  // MTE3 reads tmpBuf — race condition!

// RIGHT: Sync between pipes
Compute(tmpBuf);
PipeBarrier<PIPE_V>();  // Wait for VECTOR to finish
DataCopy(gm, tmpBuf, len);
```

## 9. Incorrect Block Dimension
```cpp
// WRONG: Using fixed core count
SetBlockDim(20);  // Hardcoded — won't work on different chips

// RIGHT: Let host decide based on workload
// Host side:
uint32_t coreNum = min(availableCores, totalTiles);
context.SetBlockDim(coreNum);
```

## 10. Tail Block Handling
```cpp
// WRONG: Processing full tile size for tail block
for (int t = 0; t < tileNum; t++) {
    DataCopy(ub, gm + t * tileLen, tileLen);  // Last tile may read beyond buffer!
}

// RIGHT: Handle tail separately
for (int t = 0; t < tileNum; t++) {
    uint32_t curLen = (t < tileNum - 1) ? tileLen : tailLen;
    DataCopy(ub, gm + t * tileLen, curLen);
}
```

## 11. EnQue/DeQue Mismatch
```
Rule: Every EnQue must have a matching DeQue, and vice versa.
      Number of EnQues must equal number of DeQues over the lifetime of the kernel.

Mistake: CopyIn does EnQue but Compute doesn't always DeQue (conditional skip)
→ Queue fills up, deadlock
```

## 12. Incorrect GetBlockIdx/GetBlockNum Usage
```cpp
// GetBlockIdx(): Current core index (0-based)
// GetBlockNum(): Total number of active cores

// WRONG: Using GetBlockNum as loop bound for tiles
for (int t = 0; t < GetBlockNum(); t++) { ... }

// RIGHT: Each core processes its own tiles
int myStart = GetBlockIdx() * tilesPerCore;
int myEnd = min(myStart + tilesPerCore, totalTiles);
```

## 13. Workspace Buffer Overflow
```
When using GM workspace for intermediate results:
- Workspace size must be declared in host tiling
- Must account for ALL cores writing simultaneously
- workspace_size = per_core_size × blockDim

Mistake: Calculating workspace for 1 core but blockDim cores write concurrently
```

## 14. Mixing Up Input Layouts
```
BNSD: [Batch, NumHeads, SeqLen, HeadDim] — default for many attention ops
BSND: [Batch, SeqLen, NumHeads, HeadDim] — used by some frameworks
BSH:  [Batch, SeqLen, Hidden] — used for linear projections

Mistake: Assuming BNSD layout when input is BSND → incorrect stride calculations
→ Silently produces wrong results (not a crash)
```

## 15. Overflow in Address Computation
```cpp
// WRONG: 32-bit overflow for large tensors
int offset = batch * seqLen * numHeads * headDim;  // Can overflow int32!

// RIGHT: Use 64-bit for address computation
int64_t offset = (int64_t)batch * seqLen * numHeads * headDim;
```

## 16. Scalar For-Loop Instead of Vector API (Performance Killer)
```
This is the #1 performance pitfall in generated code.
TileLang→AscendC translation often produces raw pointer + scalar for-loops
that completely bypass the NPU Vector unit. The code compiles and runs
correctly, but is 10-1000x slower than it should be.

CHECKLIST (run on every generated kernel):
  1. grep -nE "for\s*\(.*\+\+.*sum\s*\+=|for\s*\(.*\+\+.*max\s*=" *.cpp
     → Replace with ReduceSum<T,true>() / WholeReduceSum() / ReduceMax()
  
  2. grep -nE "for\s*\(.*y\[.*\]=.*x\[" *.cpp
     → Replace with Adds<T>/Muls<T>/Subs<T>/Mul<T>/Add<T>/Div<T> vector chain
  
  3. grep -nE "for.*for.*swap|bubble|selection.sort" *.cpp
     → Replace with AscendC::Sort<T,true>() or Sort32 + MrgSort
  
  4. grep -nE "GetPhyAddr|reinterpret_cast" *.cpp
     → Replace with LocalTensor<T> + AscendC Vector API calls

  5. grep -nE "for.*GetValue.*Muls|for.*GetValue.*Adds" *.cpp
     → Replace with BroadCast + single Mul/Add (模式 6)

  6. grep -nE "for.*SetValue.*GetValue" *.cpp
     → Replace with GatherMask (模式 7)

DETAILED GUIDE: optimization_patterns/scalar_to_vector.md
SOURCE: NMS (30_NMS) 11.25x, GroupNorm (11_GroupNorm_evo) 1.84x, TopK (TopK_evo_0629) 4.08x
```
