# P76 Playbook: V_TEMPLATE KV 预合并 Workspace

> 本 Playbook 为**强制流程**。采纳 P76 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P76 的核心是**在 Sparse Attention 的 V_TEMPLATE 模式下，由 Vector 核预先将分散的稀疏 KV 数据合并到连续 GM workspace，Cube 核直接顺序读取**。

## Step 1: 定位关键结构

```bash
grep -n "sparse|Sparse|topk|TopK|attention|Attention" \
    shared/original/op_kernel/*.cpp > /tmp/p76_locations.txt
grep -n "PageAttention|PA|block.*table|blockTable" \
    shared/original/op_kernel/*.cpp >> /tmp/p76_locations.txt
grep -n "Cube|Vector|CV|MIX|LoadData|DataCopy" \
    shared/original/op_kernel/*.cpp >> /tmp/p76_locations.txt
grep -n "workspace|Workspace|merge|Merge" \
    shared/original/op_kernel/*.cpp >> /tmp/p76_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前 sparse KV 读取方式**：文件 + 行号
- **PA block table 结构**：文件 + 行号
- **V_TEMPLATE/C_TEMPLATE 路径**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 读取方式 | `?` (C_TEMPLATE 随机) | V_TEMPLATE 连续 | `op_kernel/*.cpp:L?` |
| 预合并 | `?` (无) | Vector0 MergeKv | `op_kernel/*.cpp:L?` |
| workspace | `?` (无) | 2.25MB | `op_kernel/*.cpp:L?` |
| 阈值 | `?` (无) | sparseBlockSize <= 4 | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整 V_TEMPLATE（MergeKv + ND→NZ + 连续 workspace）**。
- **形态 β — 仅路径选择**：自动切换 V_TEMPLATE/C_TEMPLATE。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
// V_TEMPLATE: Vector0 阶段 MergeKv
void MergeKv(const ExtraInfo& info) {
    for (int i = 0; i < sparseBlockCount; i++) {
        int realS2Idx = GetRealS2Idx(topKIndices, i);
        DataCopyPad(ubBuf, kvGm[realS2Idx * headDim], copyParams);
        TransDataTo5HD(nzBuf, ubBuf, transParams);
        DataCopy(kvMergeGm[mergeOffset], nzBuf, outParams);
        mergeOffset += blockSize * headDim;
    }
}
// Cube 阶段直接从 kvMergeGm 读取连续数据
LoadData(l1Tensor, kvMergeGm[offset], loadParams);
```

### 3C. Variant Notes

- 需要额外 4*512*576*sizeof(half) ≈ 2.25MB workspace 空间。
- Vector0 阶段增加额外搬运开销和 V0-C1 同步。
- sparseBlockSize 较大时 C_TEMPLATE 更优（无需预合并）。

## Step 4: 约束复核

- workspace HBM 占用
- Vector0 额外开销
- sparseBlockSize 阈值选择

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "MergeKv|V_TEMPLATE|C_TEMPLATE" modified_files/op_kernel/*.cpp  # >=1
grep -cE "kvMergeGm|mergeOffset" modified_files/op_kernel/*.cpp  # >=1
grep -cE "TransDataTo5HD|ND.*NZ" modified_files/op_kernel/*.cpp  # >=1
grep -cE "sparseBlockSize|<=.*4" modified_files/op_kernel/*.cpp  # >=1
grep -cE "LoadData.*kvGm\[" modified_files/op_kernel/*.cpp  # ==0（Cube 不再直接读 PA）
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| workspace 不足 | 评估 HBM 或回退 C_TEMPLATE |
| V0 开销大 | 确认与 Cube 重叠 |
| sparseBlock 大 | 切换 C_TEMPLATE |
| 非 PA 场景 | 不适用 |

---

**完成清单**：
```
[P76 Playbook Completion]
Step 1: done (/tmp/p76_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: workspace HBM 占用; Vector0 额外开销; sparseBlockSize 阈值选择: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
