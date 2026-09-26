# P74 Playbook: L2 Cache 选择性禁用

> 本 Playbook 为**强制流程**。采纳 P74 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P74 的核心是**在 KV cache 数据量远大于 L2 容量的推理场景中，通过 SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE) 选择性禁用 L2 Cache，避免 thrashing**。

## Step 1: 定位关键结构

```bash
grep -n "L2|l2|Cache|cache|缓存" \
    shared/original/op_kernel/*.cpp > /tmp/p74_locations.txt
grep -n "KV|kv|推理|infer|decode|Decode" \
    shared/original/op_kernel/*.cpp >> /tmp/p74_locations.txt
grep -n "SetL2CacheHint|CACHE_MODE|CacheMode" \
    shared/original/op_kernel/*.cpp >> /tmp/p74_locations.txt
grep -n "BNSD|BnNBsD|layout|Layout|seqLen|headDim" \
    shared/original/op_kernel/*.cpp >> /tmp/p74_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前 L2 Cache 使用方式**：文件 + 行号
- **KV layout**：文件 + 行号
- **数据总量估算**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| L2 控制 | `?` (默认启用) | 条件禁用 | `op_kernel/*.cpp:L?` |
| 判断逻辑 | `?` (无) | Host 端动态判断 | `op_host/*_tiling.cpp:L?` |
| 条件 | `?` (无) | layout + 数据量 | `op_host/*_tiling.cpp:L?` |
| 阈值 | `?` (无) | L2 容量 × 1.2 | `op_host/*_tiling.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整条件禁用（layout + 数据量双条件）**。
- **形态 β — 仅数据量判断**：不做 layout 判断。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
// Host 端判断逻辑
bool GetL2CacheOffFlag() {
    // 条件1: BNSD/BnNBsD layout 连续访存，不需要 L2 预取
    if (kvLayout == BNSD || kvLayout == BnNBsD) {
        return true;  // 禁用 L2
    }
    // 条件2: KV 数据总量超过 L2 容量的 1.2 倍
    uint64_t kvDataSize = batchSize * kvHeadNum * seqLen * headDim * sizeof(half);
    if (kvDataSize > L2_CAPACITY * 1.2) {
        return true;  // 禁用 L2，避免 thrashing
    }
    return false;
}
```

### 3C. Variant Notes

- 禁用后所有 DMA 走 DDR→L1 路径，短序列场景可能损失 L2 命中收益。
- 需要 Host 端准确估算 KV 数据总量与 L2 容量的比值。
- 与 P47 (L2 Cache 优化) 互补：P47 侧重利用 L2，P74 侧重规避 thrashing。

## Step 4: 约束复核

- 短序列可能损失 L2 命中
- Host 端估算准确性
- 与 P47 的互斥/互补关系

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "GetL2CacheOffFlag|L2CacheOff" modified_files/op_kernel/*.cpp  # >=1
grep -cE "BNSD|BnNBsD|kvLayout" modified_files/op_kernel/*.cpp  # >=1
grep -cE "kvDataSize|L2_CAPACITY" modified_files/op_kernel/*.cpp  # >=1
grep -cE "SetL2CacheHint|CACHE_MODE_DISABLE" modified_files/op_kernel/*.cpp  # >=1
grep -cE "CacheMode::CACHE_MODE_ENABLE|默认.*启用" modified_files/op_kernel/*.cpp  # ==0（或条件启用）
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 短序列变慢 | 条件判断区分长短序列 |
| 估算不准 | 用实际数据量校准 |
| 与 P47 冲突 | 明确互斥条件 |
| 非 KV 场景 | 不适用 |

---

**完成清单**：
```
[P74 Playbook Completion]
Step 1: done (/tmp/p74_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: 短序列可能损失 L2 命中; Host 端估算准确性; 与 P47 的互斥/互补关系: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
