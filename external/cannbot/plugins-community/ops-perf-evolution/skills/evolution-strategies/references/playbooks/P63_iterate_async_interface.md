# P63 Playbook: Iterate 异步接口

> 本 Playbook 为**强制流程**。采纳 P63 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P63 的核心是**在 MIX 场景中用 Iterate<false> 异步接口替代 Iterate<true> 同步接口，仅第一次发送 AIV→AIC 消息，减少核间通信开销**。

## Step 1: 定位关键结构

```bash
grep -n "Iterate|IterateAll|matmulObj" \
    shared/original/op_kernel/*.cpp > /tmp/p63_locations.txt
grep -n "SyncAll|SetFlag|WaitFlag|核间.*同步" \
    shared/original/op_kernel/*.cpp >> /tmp/p63_locations.txt
grep -n "SetWorkspace|workspace|Workspace" \
    shared/original/op_kernel/*.cpp >> /tmp/p63_locations.txt
grep -n "MIX|mix|AIC|AIV" \
    shared/original/op_kernel/*.cpp >> /tmp/p63_locations.txt
grep -n "true|false|boolean" \
    shared/original/op_kernel/*.cpp >> /tmp/p63_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **Iterate 调用位置**：文件 + 行号
- **同步开销**：文件 + 行号
- **workspace 分配**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| Iterate 模式 | `?` (true/同步) | false/异步 | `?_kernel.cpp:L?` |
| 消息次数 | `?` (每次) | 仅首次 | `?_kernel.cpp:L?` |
| Workspace | `?` (无) | 有 | `?_kernel.cpp:L?` |
| 场景 | `?` (MIX) | MIX | `?_kernel.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 全异步（最常见）**：IterateAll<false>。
- **形态 β — 部分异步**：首次同步，后续异步。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
// 改造前（同步，每次发送消息）
matmulObj.Iterate<true>(cGlobal);

// 改造后（异步，仅首次发送）
matmulObj.SetWorkspace(workspace);
matmulObj.Iterate<false>(cGlobal);
```

### 3C. Variant Notes

- 与 P46 协同：MatmulImpl 配合异步 Iterate 效果最佳。
- 与 P73 协同：多 workspace 配合异步减少气泡。

## Step 4: 约束复核

- 需额外 workspace（singleCoreM * singleCoreN * sizeof(float)）
- 仅 MIX 场景
- Iterate<false> 后不能依赖同步语义

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "Iterate.*false|IterateAll.*false" modified_files/op_kernel/*.cpp  # >=1
grep -cE "SetWorkspace|workspace" modified_files/op_kernel/*.cpp  # >=1
grep -cE "Iterate.*true|IterateAll.*true" modified_files/op_kernel/*.cpp  # ==0
grep -cE "SyncAll|SetFlag|WaitFlag" modified_files/op_kernel/*.cpp  # ==0（或 note）
grep -cE "MIX|mix" modified_files/op_kernel/*.cpp  # >=1
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| workspace 不足 | 按公式分配 |
| 非 MIX 场景 | 不适用 |
| 后续依赖同步 | 手动 SyncAll |

---

**完成清单**：
```
[P63 Playbook Completion]
Step 1: done (/tmp/p63_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: 需额外 workspace（singleCoreM×singleCoreK×sizeof(T)）; 仅 MIX 场景; Iterate<false> 后不能依赖同步语义: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
