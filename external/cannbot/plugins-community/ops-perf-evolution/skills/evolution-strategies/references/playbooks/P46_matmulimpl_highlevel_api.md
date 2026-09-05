# P46 Playbook: MatmulImpl 高阶矩阵乘 API

> 本 Playbook 为**强制流程**。采纳 P46 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P46 的核心是**用手写 ~160 行的 Matmul 流水线代码替换为 MatmulImpl 高阶 API（~10 行），获得框架自动优化的 L1 多缓冲、K 维重排等能力**。

## Step 1: 定位关键结构

```bash
grep -n "Mmad|LoadData|Fixpipe|L0A|L0B|L0C|L1_PREFETCH" \
    shared/original/op_kernel/*.cpp > /tmp/p46_locations.txt
grep -n "LoadNdGmToNzL1|LoadNzL1ToZzL0A|FixpipeNzL0cToNdGm" \
    shared/original/op_kernel/*.cpp >> /tmp/p46_locations.txt
grep -n "matmul::MatmulImpl|REGIST_MATMUL_OBJ|IterateAll" \
    shared/original/op_kernel/*.cpp >> /tmp/p46_locations.txt
grep -n "depthA1|depthB1|stepKa|stepKb|MDL" \
    shared/original/op_kernel/*.cpp >> /tmp/p46_locations.txt
grep -n "TPosition::L1|CubeFormat|ND|NZ" \
    shared/original/op_kernel/*.cpp >> /tmp/p46_locations.txt
```

**交付物**：
- Matmul 流水线位置、手写指令序列
- L0/L1 buffer 管理代码
- 当前模板配置、数据格式

## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| Matmul 实现 | `?` (手写/API) | MatmulImpl | `?_kernel.cpp:L?` |
| 代码行数 | `?` (~160) | ~10 | `?_kernel.cpp:L?` |
| L1 缓冲 | `?` (手动管理) | 自动管理 | `?_kernel.cpp:L?` |
| K 维重排 | `?` (无/有) | 自动 | `?_kernel.cpp:L?` |
| 模板 | `?` (Norm/MDL) | `alpha/beta` 见 3A | `?_kernel.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — MatmulImpl + MDL（最常见）**：标准高阶 API，自动管理缓冲和重排。
- **形态 β — 回退手写（极端场景）**：非标准 layout 无法使用 API。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
// 改造前（手写 ~160 行）
// LoadNdGmToNzL1, LoadNzL1ToZzL0A, Mmad, FixpipeNzL0cToNdGm

// 改造后（MatmulImpl ~10 行）
matmul::MatmulImpl<aT, bT, cT, BiasT, CFG_MDL> mm;
REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm, &mmTilingData_);
mm.SetOrgShape(m, n, k);
mm.SetSingleShape(curSingleM, curSingleN, k);
mm.SetTensorA(xGm[xOffset]);
mm.SetTensorB(weightGm[wOffset]);
mm.IterateAll<true>(yGm[yOffset], false);
```

### 3C. Variant Notes

- 形态 β：非标准 layout 时保留手写，但隔离为 fallback 路径。
- 与 P63 协同：P63 的异步 Iterate 配合 MatmulImpl 效果最佳。

## Step 4: 约束复核

- MDL 参数需按 K 维调优
- 仅标准 layout 适用
- 需匹配 CANN 版本

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "MatmulImpl|REGIST_MATMUL_OBJ" modified_files/op_kernel/*.cpp  # >=1
grep -cE "LoadNdGmToNzL1|LoadNzL1ToZzL0A" modified_files/op_kernel/*.cpp  # ==0
grep -cE "IterateAll|Iterate" modified_files/op_kernel/*.cpp  # >=1
grep -cE "CFG_MDL|CFG_NORM" modified_files/op_kernel/*.cpp  # >=1
grep -cE "Mmad.*L0A|Fixpipe.*L0C" modified_files/op_kernel/*.cpp  # ==0
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| API 变更 | 跟进 MatmulImpl 版本 |
| 非标准 layout | 回退手写路径 |
| MDL 参数不对 | 按 K 维大小调优 depthA1/depthB1 |

---

**完成清单**：
```
[P46 Playbook Completion]
Step 1: done (/tmp/p46_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: MDL 参数需按 K 维调优; 仅标准 layout 适用; 需匹配 CANN 版本: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
