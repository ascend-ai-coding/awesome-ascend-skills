# Attention Family Optimization Insights

## Operator Variants
- **MHA** (Multi-Head Attention): num_q_heads == num_kv_heads
- **MQA** (Multi-Query Attention): num_kv_heads == 1
- **GQA** (Grouped Query Attention): num_q_heads = G × num_kv_heads

## Key Algorithmic Optimizations

### 1. Flash Attention (Tiled Online Softmax)
```
Standard attention: O = softmax(QK^T / √d) × V
  Problem: QK^T is S×S matrix — O(S²) memory, doesn't fit in SRAM

Flash Attention: Process Q×K^T in tiles, maintain running max and sum
  - Tile K,V along S2 dimension into blocks of size B
  - For each Q block, iterate over K,V blocks:
    score_ij = Q_i × K_j^T / √d
    m_new = max(m_old, rowmax(score_ij))
    P_ij = exp(score_ij - m_new)
    l_new = l_old × exp(m_old - m_new) + rowsum(P_ij)
    O_i = O_i × (l_old × exp(m_old - m_new) / l_new) + P_ij × V_j / l_new

  Benefit: O(S) memory, O(S²) compute (same), but fits in SRAM
```

### 2. Causal Masking Optimization
```
Causal mask: entry (i,j) is -inf if j > i

Block-level optimization:
  For S2 block [s2_start, s2_end] and S1 block [s1_start, s1_end]:
    if s2_start > s1_end: SKIP entire block (all masked)
    if s2_end ≤ s1_start: FULL block (no masking needed, all valid)
    else: PARTIAL block (needs element-level masking)

  Theoretical skip rate for causal: (n-1)/(2n) → 50% for large n
  Practical: 45% at S=32K, 37% at S=2K, 22% at S=512
```

### 3. GQA KV Sharing
```
When num_q_heads > num_kv_heads:
  KV head k serves Q heads [k×G, (k+1)×G - 1] where G = num_q_heads/num_kv_heads

  Implementation approaches:
  a) Expand KV: repeat_interleave K,V to match Q head count (wastes memory)
  b) Shared indexing: Map Q head to KV head via kv_idx = q_head_idx / G
     (saves memory, same compute)

  CRITICAL: When computing causal token for GQA:
    token = s1_row_within_block  (correct: based on row position)
    NOT: token = global_q_head_offset + s1_row  (WRONG for grouped layout)
```

### 4. KV-Cache (Inference Optimization)
```
During inference, past K,V are cached and only new tokens are computed.

Prefill phase: Process full sequence (like training)
Decode phase: Q has length 1, K,V have length seq_len_kv

Optimization for decode:
  - No tiling needed for Q dimension (single row)
  - Tile only along KV sequence dimension
  - Memory-bound (tiny compute per token)
```

### 5. Sliding Window Attention
```
Instead of full causal mask, only attend to last W tokens:
  valid if: i - W < j ≤ i

Block-level skip: skip blocks where s2_end < s1_start - W
Additional skip vs. standard causal: blocks far behind the window
```

## Industry Reference Implementations

### FlashAttention-2 (Tri Dao)
- Key innovation: Parallelize over sequence length, not batch/head
- Reduces non-matmul FLOPs by optimizing the online softmax rescaling
- 2× speedup over FlashAttention-1 on A100

### FlashAttention-3 (Tri Dao, 2024)
- Asynchronous execution with warp specialization
- FP8 quantization support
- Intra-warp pipelining for better hardware utilization

### PagedAttention (vLLM)
- Block-level KV cache management for inference
- Reduces memory fragmentation in serving scenarios

## Ascend-Specific Considerations

1. **Cube-Vector split**: On Ascend 910B, QK^T runs on Cube core, softmax+P×V runs on Vector core. Optimization must balance both.

2. **Workspace for online softmax**: Need GM workspace to store intermediate m, l, O across S2 tiles. MTE3 writes to workspace are significant overhead.

3. **Block size selection**: Must align with Cube unit's matmul granularity (typically 16×16 or 32×32 tiles).

4. **Pre/next token parameters**: Ascend attention APIs use `preTokens` and `nextTokens` to define the attention window, equivalent to a band mask.
