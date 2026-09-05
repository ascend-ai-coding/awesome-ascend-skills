# P87 Playbook: Manual Mmad Pipeline Timing Control

> 本 Playbook 为**强制流程**。采纳 P87 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P87 的核心是**绕过 Matmul 库，手动控制 L0A/L0B/L0C 双缓冲、HardEvent 同步、unitFlag 驱动的 Mmad/Fixpipe 融合，实现精确的 Cube 时序控制**。

## Step 1: 定位关键结构

```bash
grep -n "Matmul|matmul|Cube|Mmad|mmad" \
    shared/original/op_kernel/*.cpp > /tmp/p87_locations.txt
grep -n "L0A|L0B|L0C|l0a|l0b|l0c" \
    shared/original/op_kernel/*.cpp >> /tmp/p87_locations.txt
grep -n "HardEvent|unitFlag|PipeBarrier|SetFlag|WaitFlag" \
    shared/original/op_kernel/*.cpp >> /tmp/p87_locations.txt
grep -n "LoadData|Fixpipe|MmadParams" \
    shared/original/op_kernel/*.cpp >> /tmp/p87_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前 Matmul 使用方式（库/手动**：文件 + 行号
- **L0 buffer 分配**：文件 + 行号
- **同步方式**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| Matmul | `?` (库 API) | 手动 Mmad | `op_kernel/*.cpp:L?` |
| L0 buffer | `?` (库管理) | 手动分配 ping/pong | `op_kernel/*.cpp:L?` |
| 同步 | `?` (库隐式) | HardEvent 显式 | `op_kernel/*.cpp:L?` |
| unitFlag | `?` (无) | 条件设置 | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整手动流水线（L0 双缓冲 + HardEvent + unitFlag + 条件 PipeBarrier）**。
- **形态 β — 仅 L0 双缓冲**：不控制 unitFlag 和 PipeBarrier。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
// 手动 Mmad 流水线
TBuf<TPosition::A1> l0aBuf;  // 32K * 2 (ping/pong)
TBuf<TPosition::B1> l0bBuf;  // 32K * 2
TBuf<TPosition::C1> l0cBuf;  // 64K * 2

for (int k = 0; k < kLoops; k++) {
    int pp = k % 2;
    WaitFlag(HardEvent::M_MTE1, L0AB_EVENT + pp);

    LoadData(l0a[pp], l1Key[kvIdx], load3DParams);
    LoadData(l0b[pp], l1Query[qpIdx], load2DParams);

    SetFlag(HardEvent::MTE1_M, L0AB_EVENT + pp);
    WaitFlag(HardEvent::MTE1_M, L0AB_EVENT + pp);

    MmadParams mmadParams;
    mmadParams.unitFlag = (k == kLoops - 1) ? 0b11 : 0b10;
    Mmad(l0c[pp], l0a[pp], l0b[pp], mmadParams);

    SetFlag(HardEvent::M_MTE1, L0AB_EVENT + pp);
}
```

### 3C. Variant Notes

- 代码复杂度和维护成本极高。
- 错误的 unitFlag 或 PipeBarrier 条件会导致数据错误、死锁或竞争。
- 不同 headDim、K 切分层级和硬件环境下最佳参数需 profiling 验证。

## Step 4: 约束复核

- 极高代码复杂度
- 同步错误风险
- 参数需 profiling

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "Mmad|mmad" modified_files/op_kernel/*.cpp  # >=1
grep -cE "HardEvent|unitFlag" modified_files/op_kernel/*.cpp  # >=1
grep -cE "l0a|l0b|l0c|L0A|L0B|L0C" modified_files/op_kernel/*.cpp  # >=1
grep -cE "WaitFlag|SetFlag.*MTE1" modified_files/op_kernel/*.cpp  # >=1
grep -cE "Matmul<|mm\.SetTensor|mm\.IterateAll" modified_files/op_kernel/*.cpp  # ==0（或退化分支）
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 死锁 | 检查 HardEvent 配对 |
| 数据错误 | 验证 unitFlag 条件 |
| 参数不准 | profiling 调优 |
| 维护困难 | 充分注释 |

---

**完成清单**：
```
[P87 Playbook Completion]
Step 1: done (/tmp/p87_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: 极高代码复杂度; 同步错误风险; 参数需 profiling: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
