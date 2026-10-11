# P64 Playbook: GM 地址冲突规避

> 本 Playbook 为**强制流程**。采纳 P64 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P64 的核心是**通过错位访问或切分策略调整，消除多核同时访问同一 512B GM 区域导致的串行等待**。

## Step 1: 定位关键结构

```bash
grep -n "GetBlockIdx|GetBlockNum|BLOCK_DIM|coreNum" \
    shared/original/op_kernel/*.cpp > /tmp/p64_locations.txt
grep -n "DataCopy|CopyIn|CopyOut|Fixpipe" \
    shared/original/op_kernel/*.cpp >> /tmp/p64_locations.txt
grep -n "SyncAll|SetFlag|WaitFlag|PipeBarrier" \
    shared/original/op_kernel/*.cpp >> /tmp/p64_locations.txt
grep -n "blockSize|loopOneCore|progress" \
    shared/original/op_kernel/*.cpp >> /tmp/p64_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **多核并行确认**：文件 + 行号
- **当前访问模式（连续/错位**：文件 + 行号
- **同步方式**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 访问模式 | `?` (连续) | 错位访问 | `op_kernel/*.cpp:L?` |
| 偏移计算 | `?` (无) | GetBlockIdx 参与 | `op_kernel/*.cpp:L?` |
| 同步 | `?` (可能无) | SyncAll 配合 | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 循环内错位访问（newProgress = (i + blockIdx) % loopOneCore）**。
- **形态 β — 切分策略调整**：Host 端改变分块方式。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
// 反例：所有核同一轮访问相同地址范围
for (int i = 0; i < loopOneCore; i++) {
    DataCopy(dst, src[i * blockSize], blockSize);
}

// 正例：错位访问
for (int i = 0; i < loopOneCore; i++) {
    int newProgress = (i + GetBlockIdx()) % loopOneCore;
    DataCopy(dst, src[newProgress * blockSize], blockSize);
}
```

### 3C. Variant Notes

- 512B 对齐粒度是冲突边界。
- 可能需要 SyncAll 全核同步配合。
- 切分策略变更可能影响其他优化。

## Step 4: 约束复核

- 错位增加地址计算复杂度
- 切分策略变更的副作用
- SyncAll 开销

## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "GetBlockIdx|blockIdx" modified_files/op_kernel/*.cpp  # >=1
grep -cE "%.*loopOneCore|newProgress" modified_files/op_kernel/*.cpp  # >=1
grep -cE "DataCopy.*src\[.*\*.*blockSize" modified_files/op_kernel/*.cpp  # >=1
grep -cE "SyncAll" modified_files/op_kernel/*.cpp  # >=1（或注释说明）
grep -cE "i \* blockSize.*DataCopy" modified_files/op_kernel/*.cpp  # ==0（消除连续）
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 地址计算复杂 | 预计算 offset 表 |
| 切分副作用 | 验证其他策略兼容性 |
| SyncAll 开销 | 评估是否必要 |
| 单核算子 | 不适用 |

---

**完成清单**：
```
[P64 Playbook Completion]
Step 1: done (/tmp/p64_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: 错位增加地址计算复杂度; 切分策略变更的副作用; SyncAll 开销: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
