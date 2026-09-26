# P17 Playbook: CV Parallel Block Sizing

> 本 Playbook 为**强制流程**。采纳 P17 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P17 的核心是**在 Cube-Vector 融合算子中，通过反向推导 mBaseSize 和 s2BaseSize，使 Cube 和 Vector 工作量匹配，最大化 CV 流水掩盖**。

## Step 1: 定位关键结构

```bash
grep -n "mBaseSize|s2BaseSize|baseSize|BLOCK_SIZE" \
    shared/original/op_kernel/*.cpp > /tmp/p17_locations.txt
grep -n "Cube.*Vector|Vector.*Cube|CV|ExecuteTask|SyncAll" \
    shared/original/op_kernel/*.cpp >> /tmp/p17_locations.txt
grep -n "Tiling|CalcBaseSize|tileSize|ubFactor" \
    shared/original/op_host/*_tiling.cpp >> /tmp/p17_locations.txt
grep -n "workspace|中间.*buffer|tempBuf" \
    shared/original/op_kernel/*.cpp >> /tmp/p17_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前 mBaseSize/s2BaseSize 值**：文件 + 行号
- **CV 融合确认**：文件 + 行号
- **workspace 大小**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| mBaseSize | `?` (固定) | 按 sInner 反向推导 | `op_host/*_tiling.cpp:L?` |
| s2BaseSize | `?` (固定) | 按 GQA group 分档 | `op_host/*_tiling.cpp:L?` |
| CV 平衡 | `?` (不均衡) | 耗时匹配 | `op_host/*_tiling.cpp:L?` |
| workspace | `?` | mBase×s2Base ≤ 上限 | `op_host/*_tiling.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整反向推导（Decode 分档 + Prefill 固定）**。
- **形态 β — 仅 s2BaseSize 调整**：GQA group 分档。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
void CalcMBaseSize() {
    if (layout == TilingKeyLayout::TND || layout == TilingKeyLayout::NTD) {
        mBaseSize_ = M_BASE_SIZE_512;
    } else {
        if (s1Size <= S1_SIZE_16) {
            // Decode: sInner 反向推导
            if (sInnerSizeAlign <= 512)       mBaseSize_ = 512;
            else if (sInnerSizeAlign <= 1024) mBaseSize_ = 256;
            else if (sInnerSizeAlign <= 2048) mBaseSize_ = 128;
            else if (sInnerSizeAlign <= 4096) mBaseSize_ = 64;
            else                               mBaseSize_ = 32;
        } else {
            mBaseSize_ = M_BASE_SIZE_512;  // Prefill
        }
    }
}

// s2BaseSize 按 GQA group 分档
uint32_t sInnerSize[3U] = {8192U, 4096U, 2048U};
uint32_t idx = std::min(gSize / 5U, 2U);
sInnerSize_ = sInnerSize[idx];
```

### 3C. Variant Notes

- mBaseSize × sInnerSizeAlign ≈ workspace 上限常数。
- TND/NTD layout 需独立处理。
- Prefill 场景 M 方向给足，Decode 场景反向压缩。

## Step 4: 约束复核

- mBaseSize 过大 → Vector 等待
- s2BaseSize 过大 → Cube 等待 + workspace 膨胀
- 块过小 → 启动开销占比增大

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "mBaseSize|s2BaseSize|sInnerSize" modified_files/op_host/*_tiling.cpp  # >=1
grep -cE "M_BASE_SIZE|S1_SIZE" modified_files/op_host/*_tiling.cpp  # >=1
grep -cE "gSize|GQA|group" modified_files/op_host/*_tiling.cpp  # >=1
grep -cE "Cube.*Vector|ExecuteTask|SyncAll" modified_files/op_kernel/*.cpp  # >=1
grep -cE "constexpr.*=.*256|constexpr.*=.*512" modified_files/op_kernel/*.cpp  # ==0（或改为动态）
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| mBaseSize 过大 | Vector 空等，减 mBaseSize |
| s2BaseSize 过大 | Cube 空等 + workspace OOM，减 s2BaseSize |
| 块过小 | 启动开销占比高，适当放大 |
| CV 不匹配 | 用 profiling 验证双方耗时 |

---

**完成清单**：
```
[P17 Playbook Completion]
Step 1: done (/tmp/p17_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: mBaseSize 过大→Vector 等待; s2BaseSize 过大→Cube 等待+workspace 膨胀; 块过小→启动开销占比增大: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
