# Elementwise Fusion Insights

## Operator Types
- **Unary**: GELU, SiLU, ReLU, Sigmoid, Tanh, Exp, Log, Rsqrt
- **Binary**: Add, Mul, Sub, Div, Max, Min
- **Compound**: GELU+Add, SiLU×x (SwiGLU), Add+LayerNorm, Mul+Sum

## Key Principle

> Elementwise operators are always memory-bound on Ascend 910B.
> The compute (GELU, Add, etc.) is trivially fast compared to data transfer.
> Optimization = minimize data movement, not reduce compute.

## Optimization Strategies

### 1. Operator Fusion
```
Separate operators:
  temp = GELU(x)     → Read x from GM, write temp to GM
  y = Add(temp, bias) → Read temp from GM, read bias, write y to GM
  Total: 3 GM reads + 2 GM writes

Fused operator:
  y = Add(GELU(x), bias)  → Read x from GM, read bias, write y to GM
  Total: 2 GM reads + 1 GM write
  Speedup: ~40% (from eliminating intermediate GM access)
```

### 2. In-Place Computation
```cpp
// If input tensor is not needed after computation:
LocalTensor<half> x = inQueue.DeQue<half>();
// Compute in-place (no separate output buffer needed)
Adds(x, x, bias, tileLen);   // x = x + bias, overwrites input
Gelu(x, x, tileLen);         // x = GELU(x), overwrites again
outQueue.EnQue(x);            // Use same buffer for output

// Saves: 1 AllocTensor + 1 FreeTensor + UB memory for separate output
```

### 3. Broadcast Optimization
```
When one operand is much smaller (e.g., bias with shape [D] vs input [B,S,D]):
  - Load bias ONCE into UB, keep across all tiles
  - Don't reload bias for every tile

  TBuf<QuePosition::VECCALC> biasBuf;
  // Load once:
  DataCopy(biasBuf, gmBias, D);
  // Reuse across tiles:
  for (int t = 0; t < tileNum; t++) {
      CopyIn(t);
      Adds(output, input, biasBuf, tileLen);
      CopyOut(t);
  }
```

### 4. Common Fusion Patterns on Ascend

| Pattern | Implementation | Note |
|---------|---------------|------|
| Add + ReLU | `Adds` then `Relu` (or fused `ReluV2`) | Check if fused API exists |
| Mul + Add (bias) | `Axpy` (y = a×x + y) | Single API call |
| x × sigmoid(x) (SiLU) | `Sigmoid` then `Mul` | Two steps, but in UB |
| GELU(x) | `Gelu(dst, src, len)` | Built-in API |
| x × GELU(x) (GeGLU) | `Gelu` then `Mul` | Keep x in UB |

## Performance Characteristics

```
Elementwise ops on 910B:
  Arithmetic intensity: 0.5-2 FLOPs/byte
  Peak: ~1.2 TB/s memory bandwidth

  Throughput ceiling: ~1.2 TB/s / sizeof(dtype)
    FP16: 600 G elements/s
    FP32: 300 G elements/s

  Key metric: How close to peak bandwidth?
    > 80%: Well optimized
    50-80%: Room for improvement (likely DMA granularity or alignment)
    < 50%: Significant optimization potential (likely missing double buffer or small tiles)
```
