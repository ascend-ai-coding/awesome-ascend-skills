# P70 Playbook: FixPipe Buffer 随路量化

> 本 Playbook 为**强制流程**。采纳 P70 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P70 的核心是**将量化参数预搬到 C2PIPE2GM（Fixpipe Buffer），调用 Fixpipe 同时完成矩阵乘结果搬出和量化计算，省去 CO1→workspace→UB 的往返**。

## Step 1: 定位关键结构

```bash
grep -n "Fixpipe|CO1|workspace|C2PIPE2GM" \
    shared/original/op_kernel/*.cpp > /tmp/p70_locations.txt
grep -n "DataCopy|CopyOut|CopyIn|CopyIn1" \
    shared/original/op_kernel/*.cpp >> /tmp/p70_locations.txt
grep -n "Cast.*量化|quant|Quant|Mul.*deq" \
    shared/original/op_kernel/*.cpp >> /tmp/p70_locations.txt
grep -n "QuantMode|VQF322B8|nz2ndEn|SetFixpipeNz2ndFlag" \
    shared/original/op_kernel/*.cpp >> /tmp/p70_locations.txt
grep -n "TQue.*C1|TQue.*C2PIPE2GM" \
    shared/original/op_kernel/*.cpp >> /tmp/p70_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **CO1 输出路径**：文件 + 行号
- **workspace 使用**：文件 + 行号
- **量化计算位置**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 当前路径 | `?` (CO1→workspace→UB→GM) | CO1→GM（含量化） | `?_kernel.cpp:L?` |
| Fixpipe Buffer | `?` (无) | C2PIPE2GM | `?_kernel.cpp:L?` |
| 量化模式 | `?` (Vector) | Fixpipe 硬件 | `?_kernel.cpp:L?` |
| 搬运次数 | `?` (3+) | 1 | `?_kernel.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — Fixpipe Buffer 量化（最常见）**：C2PIPE2GM + QuantMode。
- **形态 β — 部分量化**：仅 scale 预搬，其他 Vector 后处理。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
// 改造前（CO1→workspace→UB→GM）
Fixpipe(xGm, c1Local, fixpipeParams);     // CO1→workspace
DataCopy(src0Local, xGm, cSize);           // workspace→UB
Cast(tmpLocal, src0Local, ...);            // UB 量化
Mul(tmpHalfBuffer, tmpLocal, deqLocal, cSize);
Cast(dstLocal, tmpHalfBuffer, ...);

// 改造后（CO1→GM，含量化）
TQue<QuePosition::C2PIPE2GM, 1> inQueueDeq;  // L1→FB
DataCopy(deqLocal, deq1Local, {...});        // L1→FB
SetFixpipeNz2ndFlag(1, 0, 0);
dataCopyParams.quantPre = QuantMode_t::VQF322B8_PRE;
dataCopyParams.nz2ndEn = true;
DataCopy(cGM, c1Local, dataCopyParams);      // CO1→GM 含量化
```

### 3C. Variant Notes

- 与 P48 协同：P48 分发量化路径，P70 优化量化通路。
- 与 P49 冲突：P49 是反量化，P70 是量化。不冲突，方向相反。

## Step 4: 约束复核

- 仅 Atlas A2 支持
- 需预搬量化参数到 C2PIPE2GM
- 格式受硬件限制

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "C2PIPE2GM|Fixpipe.*quant|QuantMode" modified_files/op_kernel/*.cpp  # >=1
grep -cE "CO1.*workspace|workspace.*UB" modified_files/op_kernel/*.cpp  # ==0
grep -cE "SetFixpipeNz2ndFlag|nz2ndEn" modified_files/op_kernel/*.cpp  # >=1
grep -cE "CopyIn1|CopyOut.*CopyIn" modified_files/op_kernel/*.cpp  # ==0
grep -cE "TQue.*C1|TQue.*C2" modified_files/op_kernel/*.cpp  # >=1
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 硬件不支持 | 回退 Vector 量化 |
| 格式不支持 | 检查 QuantMode 列表 |
| 参数未预搬 | 确保 SplitDeq 在 CopyOut 前完成 |

---

**完成清单**：
```
[P70 Playbook Completion]
Step 1: done (/tmp/p70_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: 仅 Atlas A2 支持; 需预搬量化参数到 C2PIPE2GM; 格式受硬件限制: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
