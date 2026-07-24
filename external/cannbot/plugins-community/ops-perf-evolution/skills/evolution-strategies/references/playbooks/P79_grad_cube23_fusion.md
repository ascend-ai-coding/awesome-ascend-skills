# P79 Playbook: Grad Cube2+Cube3 融合计算

> 本 Playbook 为**强制流程**。采纳 P79 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P79 的核心是**在 Flash Attention 反向传播中，当 headDim=64 且 sparseMode=0 时，将 Cube2(dS*K→dQ) 和 Cube3(dS^T*Q→dK) 融合，利用 L0A/B ping/pong 减少一次 L1→L0 搬运**。

## Step 1: 定位关键结构

```bash
grep -n "FlashAttention|flash.*attention|grad|backward|dQ|dK|dV|dS" \
    shared/original/op_kernel/*.cpp > /tmp/p79_locations.txt
grep -n "headDim|head.*dim|d=64|64|sparseMode" \
    shared/original/op_kernel/*.cpp >> /tmp/p79_locations.txt
grep -n "Cube2|Cube3|Cube23|cube2|cube3" \
    shared/original/op_kernel/*.cpp >> /tmp/p79_locations.txt
grep -n "L0A|L0B|L0C|ping|pong|LoadData" \
    shared/original/op_kernel/*.cpp >> /tmp/p79_locations.txt
grep -n "AtomicAdd|SetAtomicType|Fixpipe" \
    shared/original/op_kernel/*.cpp >> /tmp/p79_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **反向传播确认**：文件 + 行号
- **headDim**：文件 + 行号
- **sparseMode**：文件 + 行号
- **Cube2/3 分离代码**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| headDim | `?` | 64 | `?_kernel.cpp:L?` |
| sparseMode | `?` | 0 | `?_kernel.cpp:L?` |
| Cube2/3 | `?` (分离) | 融合 | `?_kernel.cpp:L?` |
| L0 buffer | `?` (2) | 4+ | `?_kernel.cpp:L?` |
| 搬运次数 | `?` (2次 L1→L0) | 1次 | `?_kernel.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — d=64 特化融合（唯一形态）**。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
bool cube23_mix_flag = (sparseMode == 0 && headDim == 64);

if (cube23_mix_flag) {
    // L0A ping: dS   L0A pong: dS^T
    LoadData(l0a_ping, l1_dS, loadParams);
    LoadData(l0a_pong, l1_dS, loadParamsTranspose);
    
    // L0B ping: K    L0B pong: Q
    LoadData(l0b_ping, l1_K, loadParams);
    LoadData(l0b_pong, l1_Q, loadParams);
    
    // L0C 4-buffer
    Mmad(l0c_c1, l0a_ping, l0b_ping, mmadParams);  // dQ += dS * K
    Mmad(l0c_c2, l0a_pong, l0b_pong, mmadParams);  // dK += dS^T * Q
    
    SetAtomicType<float>();
    Fixpipe(dqGm, l0c_c1, fixpipeParams);
}
```

### 3C. Variant Notes

- 仅 d=64 适用。
- L0C 需 4 buffer（128K）。

## Step 4: 约束复核

- 仅 d=64
- L0C 容量压力大
- 代码复杂度显著增加

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "Cube23|cube23|cube23_mix_flag" modified_files/op_kernel/*.cpp  # >=1
grep -cE "headDim.*==.*64|64.*headDim" modified_files/op_kernel/*.cpp  # >=1
grep -cE "sparseMode.*==.*0" modified_files/op_kernel/*.cpp  # >=1
grep -cE "l0a_ping|l0a_pong|l0b_ping|l0b_pong" modified_files/op_kernel/*.cpp  # >=1
grep -cE "SetAtomicType|AtomicAdd" modified_files/op_kernel/*.cpp  # >=1
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| d≠64 | 分离执行 |
| L0C 不足 | 检查容量，回退分离 |
| sparseMode≠0 | 不触发 |
| 结果累加错误 | 确认 AtomicAdd |

---

**完成清单**：
```
[P79 Playbook Completion]
Step 1: done (/tmp/p79_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: 仅 d=64; L0C 容量压力大; 代码复杂度显著增加: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
