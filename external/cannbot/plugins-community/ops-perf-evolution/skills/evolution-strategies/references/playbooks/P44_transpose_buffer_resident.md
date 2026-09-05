# P44 Playbook: Transpose Buffer 常驻复用

> 本 Playbook 为**强制流程**。采纳 P44 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P44 的核心是**gradTransposeBuf_ 作为转置后的梯度数据常驻 UB，在整个 numIters 迭代循环中被反复原地读写，只在首尾各做一次转置和 GM 搬运**。

## Step 1: 定位关键结构

```bash
grep -n "Transpose|transpose|转置|Permute" \
    shared/original/op_kernel/*.cpp > /tmp/p44_locations.txt
grep -n "TBuf|VECCALC|AllocTensor|FreeTensor" \
    shared/original/op_kernel/*.cpp >> /tmp/p44_locations.txt
grep -n "for.*iter|numIters|迭代|loop|colNormGrad|rowNormGrad" \
    shared/original/op_kernel/*.cpp >> /tmp/p44_locations.txt
grep -n "CopyOut|DataCopy|TransposeXIn|TransposeXOut" \
    shared/original/op_kernel/*.cpp >> /tmp/p44_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前转置频率**：文件 + 行号
- **迭代次数**：文件 + 行号
- **buffer 管理方式**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 转置频率 | `?` (每轮) | 首尾各一次 | `op_kernel/*.cpp:L?` |
| buffer | `?` (GM/临时) | TBuf 常驻 UB | `op_kernel/*.cpp:L?` |
| 迭代内搬运 | `?` (有) | 零 DMA | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整常驻（TBuf + TransposeXIn/Out + 迭代内零搬运）**。
- **形态 β — 仅减少转置次数**：不改为 TBuf。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
TBuf<TPosition::VECCALC> gradTransposeBuf_;
void Process() {
    TransposeXIn();
    for (int j = numIters_ - 1; j > 0; --j) {
        colNormGrad();
        rowNormGrad();
    }
    TransposeXOut();
    CopyOut(offset);
}
```

### 3C. Variant Notes

- 占用 tAlign_×n_×n_×4 字节，n 较大时成为 tiling 瓶颈。
- TransposeXIn/Out 本身有开销，仅在迭代次数较多（≥3）时收益明显。

## Step 4: 约束复核

- UB 空间 = tAlign × n × n × 4
- 迭代次数 ≥3 才有收益
- 转置本身开销

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "gradTransposeBuf_|TransposeBuf" modified_files/op_kernel/*.cpp  # >=1
grep -cE "TransposeXIn|TransposeXOut" modified_files/op_kernel/*.cpp  # >=1
grep -cE "colNormGrad|rowNormGrad" modified_files/op_kernel/*.cpp  # >=1
grep -cE "numIters_|for.*iter" modified_files/op_kernel/*.cpp  # >=1
grep -cE "loop.*Transpose|每.*轮.*Transpose" modified_files/op_kernel/*.cpp  # ==0
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| UB 不足 | 减 n 或 tile |
| 迭代 <3 | 收益不明显，回退 |
| 转置开销大 | 评估总体收益 |
| 无迭代 | 不适用 |

---

**完成清单**：
```
[P44 Playbook Completion]
Step 1: done (/tmp/p44_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: UB 空间 = tAlign × n × n × 4; 迭代次数 ≥3 才有收益; 转置本身开销: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
