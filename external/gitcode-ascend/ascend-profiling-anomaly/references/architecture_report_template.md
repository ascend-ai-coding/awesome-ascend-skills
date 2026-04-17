# Model Architecture Report Template

## Table of Contents
1. [Report Purpose](#1-report-purpose)
2. [Report Filename Convention](#2-report-filename-convention)
3. [Required Sections](#3-required-sections)
4. [Section Details & Methodology](#4-section-details--methodology)
5. [Formatting Rules](#5-formatting-rules)
6. [Analysis Techniques](#6-analysis-techniques)

---

## 1. Report Purpose

The Model Architecture Report is a **standalone Markdown file** produced alongside the main anomaly discovery output. Its purpose is to reverse-engineer and document the model's architecture purely from profiling data — layer count, layer types, per-layer kernel composition, parallelism strategy, communication patterns, and execution timeline — so that a reader unfamiliar with the model can understand exactly what was profiled and how each component contributes to execution time.

This report is separate from the anomaly report. The anomaly report answers "what looks unnatural"; the architecture report answers "what is this model and how does it execute."

---

## 2. Report Filename Convention

Save the report as:

```
model_architecture_report_<profiling_dir_name>.md
```

Example: `model_architecture_report_0313_dp2_tp8_ep_100k_1k_bs10.md`

Place it in the profiling directory or working output directory.

---

## 3. Required Sections

Every architecture report MUST include all of the following sections, in order:

1. **Configuration Context** — parallelism config, sequence lengths, batch size, capture metadata
2. **Model Architecture Determination** — evidence chain proving layer count and pass structure
3. **Forward Pass Boundaries** — timestamp ranges, wall times, kernel counts per pass
4. **Layer Classification** — table of layer types with counts and characteristics
5. **Cross-Verification Table** — per-pass op counts that confirm the classification
6. **Per-Layer Sub-Structure** — detailed kernel sequence for EACH distinct layer type
7. **Decode Phase Analysis** — decode layer structure, dominant costs, comparison to prefill
8. **Communication Pipeline Structure** — stream roles, overlap strategy, pipeline diagram
9. **Layer-to-Layer Variation** — comparative table across layer types (wall time, FIA share, DFC cost, kernel count)
10. **Model Architecture Summary** — ASCII diagram of full model + per-pass execution timeline

---

## 4. Section Details & Methodology

### 4.1 Configuration Context

Extract from directory name and profiling metadata:

```markdown
- **Parallelism**: DP{X} × TP{Y} [with EP if expert parallelism detected]
- **Sequence lengths**: {prefill_len} context, {decode_len} generation, batch size {N}
- **Profiling scope**: [step markers if present, else "single pseudo-step"]
- **Capture span**: {duration}s, {kernel_count} device kernels across {stream_count} streams
- **Model**: {model_type} with ~{layer_count} layers, captured across {pass_count} forward passes
  ({prefill_FIA_count} prefill FIA + {decode_FIA_count} decode FIA)
```

### 4.2 Model Architecture Determination

Build an **evidence chain table** proving the model structure. Use FusedInferAttentionScore (FIA) invocations as the primary structural marker:

1. Count total FIA invocations
2. Separate prefill FIA (duration > 10ms) from decode FIA (duration < 1ms)
3. Divide to determine layers per pass and passes per capture
4. Identify phase transition points (prefill → decode boundaries) by timestamp gaps
5. Present as evidence table with columns: Evidence | Value | Interpretation

### 4.3 Forward Pass Boundaries

Table with columns:
- Pass number
- FIA index range
- Time span (absolute seconds from capture start)
- Wall time (ms)
- Average prefill FIA duration
- Total kernel count

Note any cross-pass variation (e.g., FIA duration increasing across passes due to KV cache growth).

### 4.4 Layer Classification

Table identifying EACH distinct layer type. Common patterns:

| Layer Type | Layers | Count | Characteristics |
|---|---|---|---|
| Dense | 0–N | K | No MoE routing; attention → projection → norm → FFN (direct MatMul) |
| MoE+DFC | N+1–M | J | Full MoE layers with DispatchFFNCombine for fused expert routing+compute |
| MoE+GMM | M+1 (last) | 1 | MoE with GroupedMatmul instead of DFC; includes output head, sampling, decode prep |

The classification is determined by examining which kernels appear in each layer's kernel sequence (presence/absence of DispatchFFNCombine, GroupedMatmul, MoeGatingTopK, alltoallv, etc.).

### 4.5 Cross-Verification Table

Count key ops per pass and verify they match the classification:

| Op | Per Pass Count | Interpretation |
|---|---|---|
| FIA (prefill) | 93 | 1 per layer |
| DispatchFFNCombine | 89 | Layers 3–91 only (MoE layers) |
| GroupedMatmul | 4 | 2 in transition layer + 2 in decode |
| MoeGatingTopK | 91 | 89 prefill MoE + 2 decode |
| split_qkv_rmsnorm_rope | N | Present in layers with next-layer prep |
| ReshapeAndCache | N+1 | KV cache updates |

### 4.6 Per-Layer Sub-Structure

For EACH distinct layer type, provide a **kernel execution sequence tree** showing the exact order of operations. Use indented tree notation:

```
FIA ({duration}ms)
├─ Attention projection: TensorMove×2 → MatMulV3 ({dur}ms)
├─ TP communication: hcom_reduceScatter ({dur}ms) → allgather_AICPU
├─ MoE routing: AddRmsNormBias → Cast → MatMulV2 → MoeGatingTopK
├─ Expert dispatch: DispatchFFNCombine ({dur}ms)
├─ Expert FFN:
│   DynamicQuant → QuantBatchMatmulV3 → SwiGlu → QuantBatchMatmulV3
├─ Post-MoE: AddRmsNormQuant → hcom_allGather
├─ Next-layer prep:
│   reduce_scatter_AICPU ({dur}ms, stream {N}, overlapped)
│   QuantBatchMatmulV3 → split_qkv_rmsnorm_rope → ReshapeAndCache
```

For each layer type, also provide:
- **Kernel count** per layer
- **Wall time** (typical)
- **Dominant op** and its share
- **Timing breakdown table** with Component | Wall time | Share of layer
- **Anomalies** specific to that layer type (e.g., warm-up overhead in layer 0)
- **Stream annotations** (which stream each kernel group runs on)

The transition/last layer deserves extra detail because it typically combines:
- Last transformer layer computation
- Output head (logit projection)
- Sampling logic (ArgMax, rejection sampling for speculative decode)
- Next-iteration input preparation (embedding, position encoding, KV cache setup)
- Any AICPU ops that should be flagged

### 4.7 Decode Phase Analysis

Decode layers have fundamentally different characteristics from prefill:

1. Show the kernel sequence tree (same format as prefill layers)
2. Provide a **dominant costs table** with Component | Duration (ms) | Share
3. Highlight the key contrast: FIA drops dramatically (e.g., 28ms → 0.2ms)
4. Identify what dominates decode cost instead (typically all-to-all communication for EP models)
5. Explain WHY the cost profile differs (latency-bound vs bandwidth-bound)

### 4.8 Communication Pipeline Structure

Table with columns: Stream | Purpose | Overlaps with

Then provide an ASCII pipeline diagram showing how communication overlaps with compute:

```
Layer N: [FIA 28.5ms][post-FIA 12ms]
Layer N comm:          [reduce_scatter 29.9ms on stream 82]
                       ↕ overlaps with ↕
Layer N+1: ............[FIA 28.5ms][post-FIA 12ms]
```

Explain why kernel_sum >> wall_ms (multi-stream overlap) and quantify how much communication is hidden.

### 4.9 Layer-to-Layer Variation

Comparative table across ALL layer types:

| Metric | Dense (0–2) | MoE+DFC (3–91) | Transition (92) | Decode |
|---|---|---|---|---|
| Wall time | X ms | Y ms (avg) | Z ms | W ms |
| FIA share | A% | B% | C% | D% |
| DFC cost | 0 | V ms | 0 | 0 |
| Post-FIA compute | ... | ... | ... | ... |
| Kernels per layer | ... | ... | ... | ... |

Note any variance within a type (e.g., DFC ranging from 5.4ms to 14.8ms across MoE layers) and explain likely causes.

### 4.10 Model Architecture Summary

Provide TWO visualizations:

**1. ASCII model diagram:**

```
┌─────────────────────────────────────────────────┐
│  MoE Transformer Model — N Layers               │
│  DP=X, TP=Y, EP                                 │
│  Sequence: {prefill}K prefill + {decode} decode  │
├─────────────────────────────────────────────────┤
│  Layer 0:  Dense Attention + Dense FFN           │
│  ...                                             │
│  Layer K:  Attention + MoE FFN (DFC)      ──┐   │
│  ...                                     M MoE  │
│  Layer K+M: Attention + MoE FFN + Sample ───┘   │
│  Layer last: MoE (GMM+alltoallv) + Output Head  │
├─────────────────────────────────────────────────┤
│  Decode Phase (L layers per pass):               │
│  Decode 0: Attention + MoE (GMM+alltoallv)      │
│  ...                                             │
└─────────────────────────────────────────────────┘
```

**2. Per-pass execution timeline:**

```
0ms        1000ms     2000ms     3000ms     3800ms
|──────────|──────────|──────────|──────────|
[Layer0][Layer1]...[Layer91][L92=87ms][D0=15ms][D1=14ms]
 33ms   33ms        45ms
  └── 93 prefill layers ──────────────┘  └ 2 decode ┘
            ~3785ms total prefill          ~29ms decode
```

---

## 5. Formatting Rules

- Use Markdown tables (pipe-delimited) for all tabular data
- Use indented tree notation (├─, │, └─) for kernel sequences
- Use ASCII box-drawing characters for summary diagrams
- Include units on ALL numeric values (ms, us, %, MB, GB/s)
- Bold key findings and anomalies within narrative text
- Use `code formatting` for kernel names, stream IDs, and op names
- Every table MUST have a heading row and alignment
- Narrative explanations between tables should be concise but complete

---

## 6. Analysis Techniques

### 6.1 Determining layer count from FIA

FusedInferAttentionScore (FIA) is the most reliable structural marker because:
- It appears exactly once per transformer layer per pass
- Prefill FIA (long sequence) has dramatically different duration than decode FIA
- The temporal spacing between FIAs reveals layer boundaries

```python
# Pseudocode for pass/layer detection
prefill_fia = [f for f in all_fia if f.duration > 10_000]  # > 10ms
decode_fia = [f for f in all_fia if f.duration < 1_000]     # < 1ms
layers_per_pass = len(prefill_fia) // num_passes
decode_per_pass = len(decode_fia) // num_passes
```

### 6.2 Classifying layer types

For each layer (delimited by consecutive FIA invocations):
1. Extract all kernels between FIA[i] start and FIA[i+1] start
2. Check for presence of: DispatchFFNCombine → MoE+DFC layer
3. Check for presence of: GroupedMatmul + alltoallv → MoE+GMM layer
4. Check for absence of MoE markers → Dense layer
5. Check for sampling ops (ArgMax, rejection_*) → includes decode/sampling logic

### 6.3 Measuring communication overlap

```python
# Compute overlap ratio between comm stream and compute stream
comm_intervals = [kernels on stream 82]
compute_intervals = [kernels on stream 47]
overlap = intersect(merge(comm_intervals), merge(compute_intervals))
overlap_ratio = sum(overlap durations) / sum(comm durations)
# overlap_ratio > 0.9 means communication is well-hidden
```

### 6.4 Computing per-layer timing breakdowns

For each layer, group kernels by functional role:
- **Attention**: FIA kernel duration
- **Attention projection**: MatMul/TensorMove between FIA and TP comm
- **TP communication**: hcom_reduceScatter, hcom_allGather
- **MoE routing**: MoeGatingTopK, Cast, HistogramV2
- **Expert dispatch**: DispatchFFNCombine or GroupedMatmul
- **Expert FFN**: QuantBatchMatmulV3, SwiGlu within MoE section
- **Post-MoE norm**: AddRmsNormQuant, AddRmsNormBias
- **Next-layer prep**: split_qkv_rmsnorm_rope, ReshapeAndCache
- **EP communication**: alltoallv, AivKernel
- **Sampling**: ArgMax, rejection_*, IndexFill, LogicalAnd
- **Small ops overhead**: remaining kernels

Sum wall time for each group to build the timing breakdown table.
