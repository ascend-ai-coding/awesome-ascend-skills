# Ascend 910B Architecture Knowledge

## Core Architecture

### AI Core Types
- **AIC (AI Cube Core)**: 20 cores, specialized for matrix multiplication (Cube unit)
- **AIV (AI Vector Core)**: 40 cores, specialized for vector/scalar computation
- Each AIV core contains: SCALAR unit, VECTOR unit, MTE2 (GM→UB), MTE3 (UB→GM)

### Memory Hierarchy
```
Global Memory (HBM)
  ↕ MTE2 (load) / MTE3 (store)
Unified Buffer (UB): 256KB per core (192KB usable after system reservation)
  ↕ Direct access
VECTOR/SCALAR units
```

### Key Parameters (910B3)
| Parameter | Value | Note |
|-----------|-------|------|
| UB size | 256KB (192KB usable) | 32-byte alignment required |
| Vector width | 256 bytes | 128 FP16 / 64 FP32 elements per cycle |
| AI Core count | 20 AIC + 40 AIV | SetBlockDim determines active cores |
| HBM bandwidth | ~1.2 TB/s theoretical | Shared across all cores |
| L2 cache | 96MB shared | Helps with repeated access patterns |

## Pipeline Model

### Five-Stage Pipeline (per AIV core)
```
SCALAR → SCALARLDST → VECTOR → MTE2 → MTE3
  (addr     (param       (compute)  (GM→UB    (UB→GM
   calc)     load)                   load)     store)
```

### Pipeline Concurrency
- MTE2, VECTOR, MTE3 can execute **concurrently** (key to double buffering)
- SCALAR/SCALARLDST execute **sequentially** before issuing to other pipes
- Pipeline barriers (PipeBarrier) force all pipes to synchronize

### Typical Bottleneck Patterns

#### Pattern 1: Memory-Bound (most elementwise ops)
```
Symptom: MTE2 active > 60%, VECTOR idle > 40%
Root cause: Data transfer dominates, compute is cheap
Solution: Double buffering (P1) + larger tiles (P2) + vectorized copy (P10)
```

#### Pattern 2: Compute-Bound (matmul, complex math)
```
Symptom: VECTOR active > 80%, MTE2 idle > 30%
Root cause: Computation dominates, data arrives faster than consumed
Solution: Reduce compute (algorithmic), mixed precision (D1)
```

#### Pattern 3: Balanced (well-optimized kernels)
```
Symptom: MTE2 active ≈ VECTOR active (within 10%)
Root cause: Pipeline stages well overlapped
Ceiling: Can only improve by reducing BOTH sides simultaneously
→ This is the "balance wall" — instruction-level optimization has diminishing returns
→ Must explore algorithm-level reduction (skip computation, approximate)
```

#### Pattern 4: Scalar-Bound (complex control flow)
```
Symptom: SCALARLDST active > 30%, VECTOR frequently stalled
Root cause: Too many tiling parameters or complex address calculations
Solution: Simplify tiling, precompute addresses, reduce conditional branches
```

## Roofline Model

### Arithmetic Intensity Thresholds (910B3)
```
Peak compute: ~320 TFLOPS (FP16)
Peak BW: ~1.2 TB/s (HBM)

Inflection point ≈ 320/1.2 ≈ 267 FLOPs/byte

If op_intensity < 267: memory-bound → optimize data movement
If op_intensity > 267: compute-bound → optimize computation
```

### Quick Estimation
```
Elementwise (add, mul, gelu): ~0.5-2 FLOPs/byte → deeply memory-bound
Reduction (sum, mean, norm): ~2-10 FLOPs/byte → memory-bound
Softmax: ~5-15 FLOPs/byte → memory-bound
MatMul (large): ~100-1000 FLOPs/byte → compute-bound
Flash Attention: ~50-200 FLOPs/byte → mixed (depends on seq_len)
```

## UB Memory Planning

### Maximum Tile Size Formula
```
max_tile_elements = (UB_USABLE / (BUFFER_NUM × PIPE_COUNT × sizeof(dtype))) / 32 × 32

Examples (192KB usable):
  FP16, 2-buf, 2-pipe: (192*1024) / (2*2*2) / 32 * 32 = 24576 elements
  FP32, 2-buf, 2-pipe: (192*1024) / (2*2*4) / 32 * 32 = 12288 elements
  FP16, 1-buf, 1-pipe: (192*1024) / (1*1*2) / 32 * 32 = 98304 elements
```

### Common Allocation Patterns
```cpp
// Pattern 1: Input + Output double-buffered
TQue<QuePosition::VECIN, 2> inQueue;   // 2 buffers for overlap
TQue<QuePosition::VECOUT, 2> outQueue;

// Pattern 2: Input + Temp + Output (3 tenors, single buffer)
TBuf<QuePosition::VECCALC> tmpBuf;     // Shared temp buffer

// Pattern 3: Multiple inputs with broadcast
// Allocate largest input first, reuse buffer space for smaller inputs
```

## DMA Efficiency

### Optimal Transfer Granularity
```
Minimum efficient transfer: 256 bytes (below this, DMA setup overhead dominates)
Optimal transfer: 4KB-64KB (saturates bus bandwidth)
Maximum single transfer: limited by UB buffer size

Rule of thumb: tile_size × sizeof(dtype) should be ≥ 1KB for good DMA efficiency
```

### Alignment Requirements
- Base address: 32-byte aligned (MANDATORY for all DMA operations)
- Stride: 32-byte aligned (recommended, not mandatory for some APIs)
- DataCopyPad: Use when source data is not 32-byte aligned
