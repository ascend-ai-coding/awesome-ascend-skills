# P52 Playbook: L2 Cache Hint 优化

> 本 Playbook 为**强制流程**。采纳 P52 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P52 的核心是**在 blockDimM==1 时通过 SetL2CacheHint(DISABLE) 主动禁用 L2，避免多核竞争导致的 cache thrashing**。

## Step 1: 定位关键结构

```bash
grep -n "blockDim|GetBlockNum|coreNum|多核" \
    shared/original/op_kernel/*.cpp > /tmp/p52_locations.txt
grep -n "l2Cache|L2Cache|CacheMode|cache" \
    shared/original/op_kernel/*.cpp >> /tmp/p52_locations.txt
grep -n "SetL2CacheHint|CACHE_MODE_DISABLE" \
    shared/original/op_kernel/*.cpp >> /tmp/p52_locations.txt
grep -n "weightGm|weight|共享.*权重" \
    shared/original/op_kernel/*.cpp >> /tmp/p52_locations.txt
grep -n "DataCopy|LoadData" \
    shared/original/op_kernel/*.cpp >> /tmp/p52_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **blockDimM 值**：文件 + 行号
- **L2 Cache 配置**：文件 + 行号
- **共享权重访问模式**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| blockDimM | `?` | 不变 | `?_tiling.cpp:L?` |
| L2 Cache | `?` (默认) | 条件禁用 | `?_kernel.cpp:L?` |
| 竞争场景 | `?` (有/无) | 检测 | `?_kernel.cpp:L?` |
| 数据量 | `?` | > L2 容量时禁用 | `?_kernel.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 条件禁用（最常见）**：blockDimM==1 时禁用。
- **形态 β — 动态决策**：按数据量动态开关。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
if (blockDimM == 1) {
    weightGmLocal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
}
```

### 3C. Variant Notes

- 与 P61 冲突：P61 开启 L2，P52 禁用 L2。场景不同（P61 PageAttention，P52 blockDimM==1）。

## Step 4: 约束复核

- 仅 blockDimM==1 时触发
- 小数据量可能变慢
- 两行 API 调用

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "SetL2CacheHint|CACHE_MODE_DISABLE" modified_files/op_kernel/*.cpp  # >=1
grep -cE "blockDimM.*==.*1|blockDim.*==.*1" modified_files/op_kernel/*.cpp  # >=1
grep -cE "CACHE_MODE_ENABLE|l2Cache.*true" modified_files/op_kernel/*.cpp  # ==0
grep -cE "weightGm|weightGmLocal" modified_files/op_kernel/*.cpp  # >=1
grep -cE "if.*blockDim" modified_files/op_kernel/*.cpp  # >=1
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 小数据量变慢 | 添加数据量判断 |
| 非 blockDimM==1 | 不触发 |
| DDR 带宽不足 | 监控带宽利用率 |

---

**完成清单**：
```
[P52 Playbook Completion]
Step 1: done (/tmp/p52_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: 仅 blockDimM==1 时触发; 小数据量可能变慢; 两行 API 调用: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
