# P50 Playbook: SwiGLU 融合流水线

> 本 Playbook 为**强制流程**。采纳 P50 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P50 的核心是**用高阶 SwiGLU API + Ping-Pong 双缓冲替代 5 步手动 sigmoid+gate，压缩为单条 API 调用**。

## Step 1: 定位关键结构

```bash
grep -n "SwiGLU|swiglu|sigmoid|gate|Mul.*Exp|FFN" \
    shared/original/op_kernel/*.cpp > /tmp/p50_locations.txt
grep -n "Muls|Exp|Adds|Duplicate|Div|Mul" \
    shared/original/op_kernel/*.cpp >> /tmp/p50_locations.txt
grep -n "PipeBarrier|SyncAll|SetFlag|WaitFlag" \
    shared/original/op_kernel/*.cpp >> /tmp/p50_locations.txt
grep -n "BUFFER_NUM|InitBuffer|TQue|Ping|Pong" \
    shared/original/op_kernel/*.cpp >> /tmp/p50_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前手动 sigmoid+gate 代码段**：文件 + 行号
- **PipeBarrier 数量**：文件 + 行号
- **buffer 分配**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| gate 计算 | `?` (5 步手动) | SwiGLU API 单条 | `op_kernel/*.cpp:L?` |
| PipeBarrier | `?` (4次) | 1次 | `op_kernel/*.cpp:L?` |
| buffer | `?` (onesLocal) | 省去 | `op_kernel/*.cpp:L?` |
| Ping-Pong | `?` (无) | 双缓冲 | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整 SwiGLU API + Ping-Pong（MoE FFN 融合）**。
- **形态 β — 仅 API 替换**：保留单缓冲，不做 Ping-Pong。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
// 改造前：5 步手动
Muls(tmpLocal, aLocal, -1.0f, tileSize_);
PipeBarrier<PIPE_V>();
Exp<float, 0>(bLocal, tmpLocal, tileSize_);
PipeBarrier<PIPE_V>();
Adds(bLocal, bLocal, 1.0f, tileSize_);
PipeBarrier<PIPE_V>();
Duplicate(onesLocal, 1.0f, tileSize_);
Div(tmpLocal, onesLocal, bLocal, tileSize_);
PipeBarrier<PIPE_V>();
Mul(aLocal, aLocal, tmpLocal, tileSize_);

// 改造后：单条 API + Ping-Pong
SwiGLU<float, false>(workspace, src0, src1, beta, halfTokenLen);
PipeBarrier<PIPE_V>();
```

### 3C. Variant Notes

- SwiGLU API 可能不支持所有数据类型，需检查 CANN 版本。
- Ping-Pong 双缓冲增加 UB 占用，需确认容量。
- 仅适用于 MoE FFN 融合算子。

## Step 4: 约束复核

- API 数据类型兼容性
- Ping-Pong UB 容量
- 适用范围窄（MoE FFN）

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "SwiGLU|swiglu" modified_files/op_kernel/*.cpp  # >=1
grep -cE "PipeBarrier" modified_files/op_kernel/*.cpp  # <=1（对比基线）
grep -cE "Muls.*Exp.*Adds.*Div.*Mul" modified_files/op_kernel/*.cpp  # ==0
grep -cE "BUFFER_NUM|Ping|Pong" modified_files/op_kernel/*.cpp  # >=1
grep -cE "onesLocal|Duplicate.*1\.0f" modified_files/op_kernel/*.cpp  # ==0
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| API 不支持 dtype | 检查 CANN 版本，回退手动实现 |
| UB 不足 | 减 tileSize 或取消 Ping-Pong |
| 非 MoE FFN | 不适用 |
| 精度差异 | 对比基准验证 |

---

**完成清单**：
```
[P50 Playbook Completion]
Step 1: done (/tmp/p50_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: API 数据类型兼容性; Ping-Pong UB 容量; 适用范围窄（MoE FFN）: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
