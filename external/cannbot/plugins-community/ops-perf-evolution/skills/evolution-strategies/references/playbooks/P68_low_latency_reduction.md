# P68 Playbook: 低延迟归约指令组合

> 本 Playbook 为**强制流程**。采纳 P68 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P68 的核心是**在大规模归约中，用二分累加（Add 指令折半）将数据缩减到 256B 以内，再用 WholeReduceSum 一次完成最终归约，替代纯 WholeReduceSum 方案，降低归约延迟**。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p68_locations.txt`：

```bash
# 1. 归约操作
grep -n "Reduce|reduce|Sum|sum|WholeReduce|BlockReduce|REDUCE_MAX|REDUCE_SUM" \
    shared/original/op_kernel/*.cpp > /tmp/p68_locations.txt
# 2. 数据规模
grep -n "hLength|bsLength|reduce.*count|tileSize|totalNum|elementCount" \
    shared/original/op_kernel/*.cpp >> /tmp/p68_locations.txt
# 3. Add 指令
grep -n "Add\s*(" \
    shared/original/op_kernel/*.cpp >> /tmp/p68_locations.txt
# 4. Mask 设置
grep -n "SetMask|SetVectorMask|MaskMode|COUNTER|ResetMask" \
    shared/original/op_kernel/*.cpp >> /tmp/p68_locations.txt
# 5. 当前归约实现
grep -n "binaryParams|halfNum|DivCeil|DEFAULT_REP_STRIDE|ONE_REPEAT" \
    shared/original/op_kernel/*.cpp >> /tmp/p68_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **归约位置**：所有 `Reduce`/`WholeReduceSum`/`BlockReduceSum` 调用位置
- **数据规模**：归约维度长度、总元素数
- **Add 指令**：当前 Add 使用位置
- **Mask 设置**：是否有 Counter 模式或 Mask 控制
- **当前归约方式**：纯 WholeReduceSum / 纯 BlockReduceSum / 其他

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 归约类型 | `?` (Sum/Max/Mean) | 不变 | `?_kernel.cpp:L?` |
| 归约规模 | `?` (元素数) | 不变 | `?_kernel.cpp:L?` |
| 当前实现 | `?` (纯 WholeReduceSum/BlockReduceSum) | `alpha/beta` 见 3A | `?_kernel.cpp:L?` |
| Mask 模式 | `?` (无/Counter/Norm) | Counter | `?_kernel.cpp:L?` |
| 二分阈值 | `?` (无/256B/512B) | ONE_REPEAT_FLOAT_SIZE | `?_kernel.cpp:L?` |
| Add 参数 | `?` (标量/张量) | 张量折半 | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的归约规模和当前实现，判断你的代码属于以下哪种形态：

- **形态 α — 二分累加 + WholeReduceSum（最常见）**：数据量较大（>256B），先用 Add 指令二分折半到 256B 以内，再用 WholeReduceSum 最终归约。
- **形态 β — BlockReduceSum + WholeReduceSum（超大 shape）**：当数据量极大且跨 block 时，先用 BlockReduceSum 做 block 级归约，再用 WholeReduceSum 做最终归约。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta`

### 3B. Canonical Template（形态 α — 二分累加 + WholeReduceSum）

```cpp
// === 改造前（纯 WholeReduceSum，延迟高）===
__aicore__ inline void ReduceNaive(LocalTensor<float> src,
                                    LocalTensor<float> dst,
                                    uint32_t count) {
    // 直接 WholeReduceSum：count=30000 时约 242 cycles
    WholeReduceSum<float, false>(dst, src, MASK_PLACEHOLDER, 1, ...);
}

// === 改造后（二分累加 + WholeReduceSum）===
__aicore__ inline void ReduceOptimized(LocalTensor<float> src,
                                        LocalTensor<float> dst,
                                        uint32_t count) {
    constexpr uint32_t ONE_REPEAT_FLOAT_SIZE = 256 / sizeof(float);  // 64
    
    SetMaskCount();  // 进入 Counter 模式
    
    LocalTensor<float> srcTmp = src;
    LocalTensor<float> dstTmp = dst;
    uint32_t totalNum = count;
    
    // Step 1: 二分累加，直到 totalNum <= ONE_REPEAT_FLOAT_SIZE
    while (totalNum > ONE_REPEAT_FLOAT_SIZE) {
        uint32_t halfNum = DivCeil(totalNum, 16) * 16;  // 16 对齐
        SetVectorMask<uint8_t, MaskMode::COUNTER>(0, totalNum - halfNum);
        Add<float, false>(dstTmp, srcTmp, srcTmp[halfNum], 
                          MASK_PLACEHOLDER, 1, binaryParams);
        totalNum = halfNum;
        srcTmp = dstTmp;
    }
    
    // Step 2: 最终 WholeReduceSum（totalNum <= 256B，延迟低）
    SetVectorMask<uint8_t, MaskMode::COUNTER>(0, totalNum);
    WholeReduceSum<float, false>(dstTmp, srcTmp, MASK_PLACEHOLDER, 1, ...);
    
    ResetMask();
    SetMaskNorm();  // 恢复 Normal 模式
}
```

### 3C. Variant Notes（若是形态 β）

- **形态 β（BlockReduceSum + WholeReduceSum）**：
  当数据跨多个 block 且需要跨 block 归约时：
  ```cpp
  // Block 内归约
  BlockReduceSum<float, false>(blockDst, src, ...);
  // Block 间最终归约
  WholeReduceSum<float, false>(dst, blockDst, ...);
  ```
  形态 β 适合 attention 的跨 head 归约或超大 batch 场景。

- **与 P67 的协同**：P67（Multi-Shape Migration）处理多 shape 支持，P68 处理归约指令优化。两者可同时存在：P67 保证 shape 适配，P68 优化归约指令序列。
- **与 P69 的协同**：P69（UB Fused Vector Chain）处理连续 Vector 计算的 UB 内融合，P68 处理归约计算的指令级优化。两者可同时存在：P69 优化 Vector 链，P68 优化 Reduce 链。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: 二分累加的 halfNum 必须是 16 的整数倍（Vector 指令对齐要求）
约束 2: Counter 模式下必须 SetMaskCount() 进入，ResetMask() + SetMaskNorm() 退出
约束 3: 最终 WholeReduceSum 的输入 totalNum 必须 <= ONE_REPEAT_FLOAT_SIZE（256B）
约束 4: Add 的 dst 和 src 不能重叠（除非明确使用 in-place 语义）
约束 5: 小数据量（<=64 float）时，二分累加无收益，直接用 WholeReduceSum
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `归约规模 = ?`, `二分次数 = ?`
- `ONE_REPEAT_FLOAT_SIZE = ?`, `halfNum 对齐 = ?`
- `预期延迟 = ? cycles`（对比纯 WholeReduceSum）
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 有 Add 指令用于二分累加
grep -cE "Add\s*\([^,]+,\s*[^,]+,\s*[^,]+\[halfNum\]|Add.*half|Add.*折半" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: 有 WholeReduceSum 或 BlockReduceSum
grep -cE "WholeReduceSum|BlockReduceSum|WholeReduce|BlockReduce" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 3: 有 Mask 控制（Counter 模式）
grep -cE "SetMaskCount|SetVectorMask.*COUNTER|MaskMode::COUNTER|ResetMask" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: 有 while 或循环结构用于二分
grep -cE "while.*totalNum|while.*count|halfNum|DivCeil" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 5: 无纯 WholeReduceSum 在大规模数据上（无二分直接调用）
grep -cE "WholeReduceSum\s*\([^)]*\)[^;]*;[^A]*WholeReduceSum" modified_files/op_kernel/*.cpp
# 期望: == 0（或 note 中说明 "小数据量直接调用"）
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：DivCeil 不存在 | 确认 CANN 版本。旧版本用 `(totalNum + 15) / 16 * 16` 手动计算 |
| 运行时：结果错误 | 检查 Mask 模式。Counter 模式下 SetVectorMask 的 count 参数是否正确。退出时必须 ResetMask |
| 运行时：性能不如预期 | 小数据量（<=64 float）无需二分。检查 totalNum 是否远大于 ONE_REPEAT_FLOAT_SIZE |
| halfNum 未 16 对齐 | halfNum 必须是 16 的倍数。检查 DivCeil 计算是否正确 |
| Mask 未恢复导致后续指令异常 | 必须 `ResetMask(); SetMaskNorm();` 在二分累加结束后恢复 |
| 内存覆盖 | Add 的 dst 和 src 若重叠需确保 in-place 安全。建议用独立 buffer |
| Counter 模式与 Normal 模式混用 | Counter 模式下所有 Vector 指令都受 mask 影响。确保仅在归约段使用 |
| BlockReduceSum 跨 block 不一致 | 形态 β 需确保所有 block 的归约结果一致写入 GM 的同一位置 |
| 数据类型不匹配 | half 的 ONE_REPEAT_FLOAT_SIZE 与 float 不同（256B / 2B = 128）。按实际类型计算 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P68 Playbook Completion]
Step 1: done (/tmp/p68_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta, canonical/variant applied
Step 4: constraints: reduce_count=? halfNum=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
