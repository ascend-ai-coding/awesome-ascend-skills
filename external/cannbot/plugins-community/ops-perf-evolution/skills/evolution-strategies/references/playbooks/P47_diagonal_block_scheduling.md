# P47 Playbook: 对角线块调度 (Diagonal Block Scheduling)

> 本 Playbook 为**强制流程**。采纳 P47 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P47 的核心是**用 MNBlockIdxCompute 对角线分配替代线性行优先分块，优化多核场景下的 L2 cache 复用和负载均衡**。

## Step 1: 定位关键结构

```bash
grep -n "GetBlockNum|BLOCK_DIM|coreNum|blockIdx|SplitCore" \
    shared/original/op_kernel/*.cpp > /tmp/p47_locations.txt
grep -n "L2|l2|cache|Cache" \
    shared/original/op_kernel/*.cpp >> /tmp/p47_locations.txt
grep -n "mIdx|nIdx|blockDimM|blockDimN" \
    shared/original/op_kernel/*.cpp >> /tmp/p47_locations.txt
grep -n "MNBlockIdxCompute|对角线|diagonal" \
    shared/original/op_kernel/*.cpp >> /tmp/p47_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前块分配方式（线性/对角线**：文件 + 行号
- **blockDimM/N 大小**：文件 + 行号
- **L2 cache 使用模式**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 分配方式 | `?` (线性) | 对角线 | `op_kernel/*.cpp:L?` |
| threshold | `?` (无) | blockDimM > 5 | `op_kernel/*.cpp:L?` |
| L2 复用 | `?` (差) | 提升 10-20% | `op_kernel/*.cpp:L?` |
| 负载均衡 | `?` (差) | 改善 10-30% | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整对角线调度（含 threshold 回退）**。
- **形态 β — 仅 threshold 内对角线**：小范围对角线映射。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
void MNBlockIdxCompute(uint32_t curBlock, ...) {
    if (blockDimM > thresholdDimM) {  // thresholdDimM = 5
        uint32_t thresholdBlockNum = 8;
        uint32_t curThresholdM = min(blockDimM, thresholdBlockNum);
        uint32_t curThresholdN = min(blockDimN, thresholdBlockNum);
        uint32_t thresholdM_dimN = curThresholdM * blockDimN;
        uint32_t curThresholdM_thresholdN = curThresholdM * curThresholdN;
        uint32_t localRelativeBlock = relativeBlock % thresholdM_dimN
            % curThresholdM_thresholdN;
        mIdx = localRelativeBlock % curThresholdM
            + relativeBlock / thresholdM_dimN * thresholdBlockNum;
        nIdx = (localRelativeBlock + localRelativeBlock
            / LeastCommonMultiple(curThresholdM, curThresholdN))
            % curThresholdN + relativeBlock % thresholdM_dimN
            / curThresholdM_thresholdN * curThresholdN;
    } else {
        // 线性回退
        mIdx = (curBlock - count) / blockDimN;
        nIdx = (curBlock - count) % blockDimN;
    }
}
```

### 3C. Variant Notes

- 对角线映射增加少量地址计算开销（LCM 计算）。
- 小 shape（blockDimM ≤ 5）无收益，需 threshold 控制。
- 与 P51（动态核配比）互补：P47 优化块分配，P51 优化核分配。

## Step 4: 约束复核

- LCM 计算开销
- threshold 参数调优
- 仅多核场景有效

## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "MNBlockIdxCompute|Diagonal" modified_files/op_kernel/*.cpp  # >=1
grep -cE "thresholdDimM|thresholdBlockNum" modified_files/op_kernel/*.cpp  # >=1
grep -cE "LeastCommonMultiple|LCM" modified_files/op_kernel/*.cpp  # >=1
grep -cE "blockDimM|blockDimN" modified_files/op_kernel/*.cpp  # >=1
grep -cE "mIdx.*=.*curBlock|nIdx.*=.*curBlock" modified_files/op_kernel/*.cpp  # ==0（或退化分支）
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| LCM 开销大 | 预计算或简化 |
| 小 shape 负收益 | threshold 回退 |
| 单核 | 不适用 |
| 与 P51 冲突 | 明确分工 |

---

**完成清单**：
```
[P47 Playbook Completion]
Step 1: done (/tmp/p47_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: LCM 计算开销; threshold 参数调优; 仅多核场景有效: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
