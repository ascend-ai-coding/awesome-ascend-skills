# Rulebook & Decision Tables

## Table of Contents
1. [General Principles](#1-general-principles)
2. [Bubble Rules](#2-bubble-rules)
3. [Host-Originated Risk Rules](#3-host-originated-risk-rules)
4. [Soft Attribution Rules](#4-soft-attribution-rules)
5. [Wait-Anchor Rules](#5-wait-anchor-rules)
6. [AICPU Rules](#6-aicpu-rules)
7. [Multi-Stream Rules](#7-multi-stream-rules)
8. [Communication / Wait Pollution Rules](#8-communication--wait-pollution-rules)
9. [Partial Capture Rules](#9-partial-capture-rules)
10. [Decision Tables](#10-decision-tables)
11. [Recommendation Rules](#11-recommendation-rules)
12. [Example Good vs Bad Outputs](#12-example-good-vs-bad-outputs)

---

## 1. General Principles

Two analysis pipelines run in parallel:
- **Structure breakdown** answers: where is it slow.
- **Anomaly discovery** answers: where is something unnatural, possibly hiding a real issue.

If anomaly facts are clear but root cause is not, the anomaly MUST still be output. Suppressing findings due to attribution uncertainty is never acceptable.

---

## 2. Bubble Rules

### 2.1 Basic computation

For step `i`:
- `underfeed_i = service_i - busy_union_i`
- `underfeed_ratio_i = underfeed_i / service_i`
- `internal_bubble_total_i = sum of all inter-segment gaps`

### 2.2 High-severity bubble thresholds

Tag `DEVICE_IDLE_GAP_HEAVY` if ANY of:
- `underfeed_ratio >= 0.30`
- `largest_internal_bubble_ms >= max(1.0, 0.10 * service_ms)`
- `prelaunch_gap_ms >= max(1.0, 0.10 * service_ms)`
- `tail_gap_ms >= max(1.0, 0.10 * service_ms)`

Tag `PRELAUNCH_GAP_HEAVY` if:
- `prelaunch_gap_ms >= max(1.0, 0.10 * service_ms)`

Tag `TAIL_GAP_HEAVY` if:
- `tail_gap_ms >= max(1.0, 0.10 * service_ms)`

Tag `INTERNAL_BUBBLE_HEAVY` if:
- `largest_internal_bubble_ms >= max(1.0, 0.10 * service_ms)`

### 2.3 Recurring bubble pattern

If within a step group >=60% of steps have `bubble_count > 0` with stable positions, tag `RECURRING_BUBBLE_PATTERN`.

### 2.4 Dominant idle pattern

Determined by largest contributor among: `prelaunch_gap_ms_avg`, `tail_gap_ms_avg`, `internal_bubble_total_ms_avg`. If multiple are close: `mixed`. If all near zero: `none`.

---

## 3. Host-Originated Risk Rules

### 3.1 High risk
If underfeed_ratio high + busy_union << service + recurring bubbles + host/ACL/sync events near bubbles: tag `HOST_ORIGINATED_RISK`

### 3.2 Risk despite sparse evidence
If underfeed_ratio high + recurring bubbles + low `host_visible_coverage_ratio`: tag `UNTRACED_HOST_BLOCKING_RISK` + `requires_host_followup=true`

---

## 4. Soft Attribution Rules

### 4.1 possible_sync_or_h2d
Bubble overlaps >=20% with: `aten::to`, `aten::_to_copy`, `aten::copy_`, `HostToDevice`, `torch_to_npu`, `aclrtMemcpy*`, `aclrtSynchronize*`

### 4.2 possible_comm_wait
Bubble overlaps >=20% with: `c10d`, `Hccl`, `hcom`, `StreamWaitEvent`, `Notify_Wait`, comm stream markers

### 4.3 possible_host_launch_lag
High prelaunch_gap + host events visible but not focused on sync/copy/comm + single-threaded CPU pattern

### 4.4 possible_python_serialization_or_lock
Large recurring bubbles + no heavy device-internal cause + sparse CPU/Python host fragments + low thread parallelism + no stronger H2D/comm evidence

### 4.5 insufficient_evidence
Fallback when no family has enough supporting evidence. This label MUST be output rather than omitting the anomaly.

---

## 5. Wait-Anchor Rules

### 5.1 Definition
`wait_ratio = wait / (duration + wait)`

If `wait_ratio > 0.95` AND `duration_us` is very small AND the op ranks in top total-cost: tag `WAIT_ANCHOR_FALSE_HOTSPOT`

### 5.2 Treatment
Wait-anchor ops still count toward total latency, but MUST be demoted in root-cause ranking. The real problem is upstream — something caused the device to be idle before this kernel started.

---

## 6. AICPU Rules

### 6.1 Classification by masked_ratio
- `masked_ratio >= 0.9`: `AICPU_MASKED_BUT_UNDESIRABLE` — hidden behind AI_CORE overlap, but still undesirable
- `0.2 <= masked_ratio < 0.9`: `AICPU_PARTIALLY_EXPOSED` — partially visible in timeline
- `masked_ratio < 0.2`: `AICPU_EXPOSED_NOT_ALLOWED` — fully exposed, directly causing device idle

### 6.2 Coupling with bubbles
If exposed AICPU ops overlap with step bubbles, increase anomaly severity. The AICPU execution itself may be the cause of the bubble.

---

## 7. Multi-Stream Rules

Always maintain ALL four timing perspectives at block/side level:
- `wall_ms` — elapsed wall clock
- `busy_union_ms` — merged device-busy across all streams
- `kernel_sum_ms` — arithmetic sum ignoring overlap
- `total_cost_ms` — sum of duration + wait for all kernels

Conclusions based on only one metric are incomplete. Never rely solely on `wall` or solely on `kernel_sum`.

---

## 8. Communication / Wait Pollution Rules

### 8.1 Authoritative source
`communication.json` is the truth source for total communication volume. If it exists, it overrides `kernel_details.csv` HCCL sums.

### 8.2 Wait pollution detection
If an op's `wait` overlaps with a communication window: tag `WAIT_POLLUTION_RISK` and mark `wait_not_reliable_for_rootcause=true`. The wait may be caused by the device waiting for communication to complete, not a host issue.

---

## 9. Partial Capture Rules

If the largest gap is near the capture start/end boundary, or a step is truncated at the boundary: tag `PARTIAL_CAPTURE_BOUNDARY`. Do not attribute boundary gaps to host issues.

---

## 10. Decision Tables

### 10.1 Hidden Issue Main Judgement

| Condition | Output |
|---|---|
| underfeed_ratio low, largest_internal_bubble small | `hidden_issue_present=false` |
| underfeed_ratio high, weak bubble periodicity | `hidden_issue_present=true`, `severity=medium` |
| underfeed_ratio high, strong bubble periodicity | `hidden_issue_present=true`, `severity=high` |
| Largest gap at capture boundary | Append `PARTIAL_CAPTURE_BOUNDARY` |

### 10.2 Dominant Idle Pattern

| Condition | Pattern |
|---|---|
| prelaunch_gap largest | `prelaunch` |
| tail_gap largest | `tail` |
| internal_bubble_total largest | `internal_bubble` |
| Inter-step gap >> intra-step gaps | `inter_step` |
| Multiple patterns similar magnitude | `mixed` |

### 10.3 Host-Originated Risk Level

| Condition | Risk |
|---|---|
| High underfeed + clear host/sync markers | `high` |
| High underfeed + moderate host markers | `medium` |
| High underfeed + sparse host markers | `medium` + `UNTRACED_HOST_BLOCKING_RISK` |
| Low underfeed | `low` |

### 10.4 Soft Root-Cause Family

| Condition | Label |
|---|---|
| High copy/sync/H2D marker overlap | `possible_sync_or_h2d` |
| High comm marker overlap | `possible_comm_wait` |
| High prelaunch + non-comm/copy host events | `possible_host_launch_lag` |
| Large bubbles + low host parallelism + no stronger evidence | `possible_python_serialization_or_lock` |
| Large bubbles + sparse host evidence | `possible_untraced_host_blocking` |
| None of the above | `insufficient_evidence` |

### 10.5 Wait-Anchor

| Condition | Output |
|---|---|
| `wait_ratio > 0.95` + tiny duration + top total-cost rank | `WAIT_ANCHOR_FALSE_HOTSPOT` |
| Otherwise | No tag |

### 10.6 AICPU

| Condition | Classification |
|---|---|
| `masked_ratio >= 0.9` | `AICPU_MASKED_BUT_UNDESIRABLE` |
| `0.2 <= masked_ratio < 0.9` | `AICPU_PARTIALLY_EXPOSED` |
| `masked_ratio < 0.2` | `AICPU_EXPOSED_NOT_ALLOWED` |

### 10.7 Follow-up Required

| Condition | Output |
|---|---|
| High hidden issue + low root-cause specificity | `followup_required=true` |
| Missing stacks, shapes, or high wait pollution | `followup_required=true` |
| Weak anomalies + sufficient evidence | `followup_required=false` |

---

## 11. Recommendation Rules

Recommendations should prioritize:
1. Anomaly discovery conclusions with evidence chains
2. Minimum additional data needed to narrow root cause
3. Uncertainty boundaries — what the analysis can and cannot say

Typical follow-up suggestions:
- Bubble high + stack missing → suggest `with_stack=true`
- Shape missing + unstable step grouping → suggest `record_shapes=true`
- Host risk high + evidence insufficient → suggest host-side sampling / thread view
- Wait pollution high → suggest checking communication and sync paths
- Inter-structure bubble high → suggest checking host dispatch latency between layers

---

## 12. Example Good vs Bad Outputs

### Good output
"Dominant step group shows significant device idle bubbles with underfeed_ratio 0.73, recurring in >80% of same-type steps. The largest bubble (2.3ms) occurs between structure block_11 and block_12, surrounded by 3 kernels: the last kernel before the gap is `MatMul` (AI_CORE, 45μs, stream 0) and the first kernel after is `Add` (AI_CORE, 12μs, stream 0). The profile appears host-originated rather than device-kernel-saturated. Specific root cause cannot be uniquely determined; candidates are possible_sync_or_h2d and possible_untraced_host_blocking."

### Bad outputs (avoid these patterns)
- "Looks like H2D." — too definitive, no evidence chain
- "Probably GIL." — asserting unique root cause without sufficient evidence
- "Device busy ratio isn't high, so it's host bound." — skipping bubble analysis
- "This small op with high wait time is the hotspot." — falling for wait-anchor trap
- "MatMul takes 45% of total_cost, so it's the bottleneck." — ignoring that most of the cost may be wait time, not compute
