# P48 Playbook: 多量化模式编译期分发

> 本 Playbook 为**强制流程**。采纳 P48 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P48 的核心是**通过编译期宏/TilingKey 支持 A8W8/A4W4/A16W8 等多量化路径，消除运行时开销**。

## Step 1: 定位关键结构

```bash
grep -n "int8|int4|half|float|A8W8|A4W4|A16W8|MSD" \
    shared/original/op_kernel/*.cpp > /tmp/p48_locations.txt
grep -n "TILING_KEY_IS|if.*defined|#if|#elif" \
    shared/original/op_kernel/*.cpp >> /tmp/p48_locations.txt
grep -n "Cast.*int32|Mul.*scale|dequant" \
    shared/original/op_kernel/*.cpp >> /tmp/p48_locations.txt
grep -n "using.*aT|using.*bT|using.*cT" \
    shared/original/op_kernel/*.cpp >> /tmp/p48_locations.txt
grep -n "QuantMode|quant.*mode" \
    shared/original/op_kernel/*.cpp >> /tmp/p48_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **量化路径数量**：文件 + 行号
- **当前数据类型**：文件 + 行号
- **TilingKey 配置**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 量化模式 | `?` (单/多) | 多 | `?_kernel.cpp:L?` |
| 分发方式 | `?` (运行时/if) | 编译期宏 | `?_kernel.cpp:L?` |
| TilingKey | `?` (无/有) | 有 | `?_tiling.cpp:L?` |
| 类型别名 | `?` (无/有) | using aT/bT/cT | `?_kernel.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 编译期宏分发**：`#if defined(GMM_A8W8)` 等。
- **形态 β — TilingKey 运行时选择**：TILING_KEY_IS 配合模板。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
#if defined(GMM_A8W8)
    using aT = int8_t; using bT = int8_t; using cT = int32_t;
#elif defined(GMM_A4W4)
    using aT = int4b_t; using bT = int4b_t; using cT = int32_t;
#elif defined(GMM_A16W8)
    using aT = half; using bT = int8_t; using cT = float;
#endif

if (TILING_KEY_IS(2)) { /* A8W8 */ }
if (TILING_KEY_IS(3)) { /* A4W4 */ }
if (TILING_KEY_IS(4)) { /* A16W8 */ }
```

### 3C. Variant Notes

- 与 P49 协同：P49 提供硬件反量化，P48 提供路径分发。
- 与 P70 协同：P70 的 Fixpipe 量化可与分发叠加。

## Step 4: 约束复核

- 每种模式生成独立内核，二进制体积增加
- TilingKey 组合需裁剪
- MSD 需额外中间 buffer

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "defined\(GMM_|TILING_KEY_IS" modified_files/op_kernel/*.cpp  # >=1
grep -cE "using.*aT|using.*bT|using.*cT" modified_files/op_kernel/*.cpp  # >=1
grep -cE "if.*int8|if.*int4|switch.*type" modified_files/op_kernel/*.cpp  # ==0
grep -cE "A8W8|A4W4|A16W8" modified_files/op_kernel/*.cpp  # >=2
grep -cE "QuantMode|quant.*mode" modified_files/op_kernel/*.cpp  # >=1
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 二进制体积大 | 裁剪 TilingKey 组合 |
| MSD buffer 不足 | 分配额外中间 buffer |
| 模式遗漏 | 全路径覆盖测试 |

---

**完成清单**：
```
[P48 Playbook Completion]
Step 1: done (/tmp/p48_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: 每种模式生成独立内核，二进制体积增加; TilingKey 组合需裁剪; MSD 需额外中间 buffer: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
