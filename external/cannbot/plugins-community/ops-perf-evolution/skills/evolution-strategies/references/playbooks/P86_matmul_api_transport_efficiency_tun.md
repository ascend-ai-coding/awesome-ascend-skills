# P86 Playbook: Matmul 高阶 API 内部数据通路效率调优

> 本 Playbook 为**强制流程**。采纳 P86 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P86 的核心是**统一调优 Matmul 库内部 GM→L1→L0→Cube 执行路径：选择 MDL/NBuffer33/Preload 模板、放大 base block、使能 UnitFlag、开启 K 轴错峰访问**。

## Step 1: 定位关键结构

```bash
grep -n "Matmul|matmul|CFG_NORM|CFG_MDL|MatmulConfig" \
    shared/original/op_kernel/*.cpp > /tmp/p86_locations.txt
grep -n "N_BUFFER_33|Preload|preload|UnitFlag|base.*block" \
    shared/original/op_kernel/*.cpp >> /tmp/p86_locations.txt
grep -n "ScheduleType|MatmulConfigParams|SetMatmulConfigParams" \
    shared/original/op_kernel/*.cpp >> /tmp/p86_locations.txt
grep -n "MTE2|mte2|搬运|搬运.*空泡|overlap" \
    shared/original/op_kernel/*.cpp >> /tmp/p86_locations.txt
grep -n "L1|L0|Cube|cube" \
    shared/original/op_kernel/*.cpp >> /tmp/p86_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前模板**：文件 + 行号
- **base block 大小**：文件 + 行号
- **MTE2 瓶颈**：文件 + 行号
- **同步策略**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 模板 | `?` (Norm) | MDL/NBuffer33/Preload | `?_kernel.cpp:L?` |
| base block | `?` (小) | 放大 | `?_tiling.cpp:L?` |
| UnitFlag | `?` (无) | 使能 | `?_kernel.cpp:L?` |
| K 错峰 | `?` (无) | 使能 | `?_kernel.cpp:L?` |
| 预算 | `?` | L1/L0 足够 | `?_kernel.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — MDL（一次搬多个基本块）**。
- **形态 β — NBuffer33（3x3 错开搬运）**。
- **形态 γ — Preload（间隙预取）**。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
// MDL
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> matmulMdl;

// NBuffer33
matmul_tiling::MatmulConfigParams matmulConfigParams(
    1, false, matmul_tiling::ScheduleType::N_BUFFER_33, ...);
cubeTiling.SetMatmulConfigParams(matmulConfigParams);

// Preload
static constexpr MatmulConfig MM_CFG = GetMDLConfig(false, false, preloadMode);
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> matmulPreload;
```

### 3C. Variant Notes

- 不同模板/开关有前置条件和互斥关系。
- 大 block/MDL/Preload 依赖更大 L1/L0 预算。

## Step 4: 约束复核

- 错误组合可能退化
- 小 shape 头开销可能抵消收益
- UnitFlag/Kdim reorder 受模板约束

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "CFG_MDL|CFG_NORM|N_BUFFER_33|Preload" modified_files/op_kernel/*.cpp  # >=1
grep -cE "MatmulConfigParams|SetMatmulConfigParams" modified_files/op_kernel/*.cpp  # >=1
grep -cE "ScheduleType|UnitFlag|base.*block" modified_files/op_kernel/*.cpp  # >=1
grep -cE "GetMDLConfig|preloadMode" modified_files/op_kernel/*.cpp  # >=1
grep -cE "CFG_NORM.*CFG_NORM" modified_files/op_kernel/*.cpp  # ==0（至少一种优化）
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 组合退化 | 按 bottleneck 选模板 |
| 小 shape 慢 | 头开销抵消，回退 Norm |
| 预算不足 | 检查 L1/L0 |
| 模板不支持 | 确认 CANN 版本 |

---

**完成清单**：
```
[P86 Playbook Completion]
Step 1: done (/tmp/p86_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: 错误组合可能退化; 小 shape 头开销可能抵消收益; UnitFlag/Kdim reorder 受模板约束: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
