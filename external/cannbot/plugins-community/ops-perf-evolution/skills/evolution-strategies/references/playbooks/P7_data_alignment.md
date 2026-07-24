# P7 Playbook: 32B 对齐 + DataCopyPad 实操流程

> 本 Playbook 为**强制流程**。采纳 P7 策略的子 agent 必须逐步执行。
> 关联设计：[knowledge-strategy-architecture-v3.2](../../../../../../docs/design/knowledge-strategy-architecture-v3.2.md)

## Step 1: 定位关键结构

```bash
# 所有 DataCopy 调用
grep -n "DataCopy\b\|DataCopyPad\b\|DataCopyExt" op_kernel/*.cpp op_kernel/*.h > /tmp/p7_locations.txt

# tileSize / dataCount 计算（搬运粒度）
grep -n "tileSize\|dataCount\|nElement\|copyLen\|transferSize" op_kernel/*.cpp >> /tmp/p7_locations.txt

# GM 地址 / offset 计算
grep -n "BaseAddr\|GlobalTensor.*Set\|GM\[" op_kernel/*.cpp >> /tmp/p7_locations.txt
```

**交付物**（写入 implementation_note.txt "Playbook Step 1"）：
- DataCopy 调用列表：文件 + 行号 + 搬运量表达式
- 当前 tileSize / 搬运元素数计算公式
- 该算子的 GM 起始地址是否 32B 对齐（hint：基址通常 32B 对齐，关键是 offset 的对齐）

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。需在 `implementation_note.txt` "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 搬运 API | `DataCopy(...)` | `DataCopyPad(..., padParams)` 或对齐后 `DataCopy` | `?_kernel.cpp:L?` |
| tileSize 对齐 | 任意 | 按 32 / sizeof(T) 上取整 | `?_kernel.cpp:L?` |
| GM offset | 任意 | 32B 倍数 | `?_kernel.cpp:L?` |
| 边界条件 | 无 padding | 尾块 + DataCopyPad padParams 处理 | `?_kernel.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 整片对齐**：搬运量恒为 32B 倍数 → 用 `DataCopy`，只需对齐 tileSize
- **形态 β — 尾块非对齐**：N 个 tile 中前 N-1 整对齐，最后一个尾块小 → tail 用 `DataCopyPad`
- **形态 γ — 全非对齐**：每次搬运量都可能非对齐（如 broadcast/stride）→ 全部用 `DataCopyPad`

**必须在 implementation_note.txt "Playbook Step 3A" 声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 β）

```cpp
// === 改造前（直接 DataCopy 可能非对齐）===
constexpr uint32_t TILE_ELEMS = 1000;     // 非 32B 对齐
for (int i = 0; i < loops; i++) {
    DataCopy(ubLocal, gmTensor[i * TILE_ELEMS], TILE_ELEMS);
    Compute(...);
    DataCopy(gmOut[i * TILE_ELEMS], ubLocal, TILE_ELEMS);
}

// === 改造后（对齐 + tail 用 Pad）===
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t ELEMS_PER_BLOCK = BLOCK_BYTES / sizeof(T);
constexpr uint32_t TILE_ELEMS_ALIGNED =
    (1000 + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK * ELEMS_PER_BLOCK;

uint32_t mainLoops = totalElems / TILE_ELEMS_ALIGNED;
uint32_t tailElems = totalElems - mainLoops * TILE_ELEMS_ALIGNED;

// 主循环：对齐 DataCopy
for (int i = 0; i < mainLoops; i++) {
    DataCopy(ubLocal, gmTensor[i * TILE_ELEMS_ALIGNED], TILE_ELEMS_ALIGNED);
    Compute(...);
    DataCopy(gmOut[i * TILE_ELEMS_ALIGNED], ubLocal, TILE_ELEMS_ALIGNED);
}

// 尾块：DataCopyPad（边界由硬件处理）
if (tailElems > 0) {
    DataCopyParams params;
    params.blockCount = 1;
    params.blockLen   = tailElems * sizeof(T);
    params.srcStride  = 0;
    params.dstStride  = 0;
    DataCopyPadParams padParams;
    padParams.isPad    = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = TILE_ELEMS_ALIGNED - tailElems;
    padParams.paddingValue = 0;
    DataCopyPad(ubLocal, gmTensor[mainLoops * TILE_ELEMS_ALIGNED], params, padParams);
    // ... compute ...
    DataCopyPad(gmOut[mainLoops * TILE_ELEMS_ALIGNED], ubLocal, params);
}
```

### 3C. Variant Notes

- **形态 α（全对齐）**：只需把 tileSize 改为 32B 倍数，主循环不变，无 tail；最简单
- **形态 γ（全非对齐如 stride）**：全部 DataCopyPad，padParams 按实际 stride 算；可能需要 P26 stride 搬运策略配套
- **dtype 影响**：fp32 → ELEMS_PER_BLOCK=8，fp16 → 16，int8 → 32；TILE_ELEMS_ALIGNED 跟着变

## Step 4: 约束复核

```
约束 1: TILE_ELEMS_ALIGNED ≥ TILE_ELEMS（不能向下取整丢数据）
约束 2: (mainLoops × TILE_ELEMS_ALIGNED) + tailElems == totalElems
约束 3: ubLocal 容量 ≥ TILE_ELEMS_ALIGNED × sizeof(T)
约束 4: GM 基址必须本身 32B 对齐（一般框架保证）；offset 必须 32B 倍数
```

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检（5 条 grep）
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
# 检查 1: 含 DataCopyPad 或对齐计算
grep -cE "DataCopyPad|ELEMS_PER_BLOCK|BLOCK_BYTES.*32|tileSize.*32" \
    modified_files/op_kernel/*.cpp modified_files/op_kernel/*.h
# 期望: >= 2

# 检查 2: 含 padParams / DataCopyPadParams 使用
grep -cE "DataCopyPadParams|paddingValue|rightPadding|isPad" \
    modified_files/op_kernel/*.cpp
# 期望: >= 1（除非是形态 α 无 tail）

# 检查 3: tail block 处理（if mainLoops/tailElems 分支或 DataCopyPad 调用）
grep -cE "tailElems|mainLoops\b|if\s*\(.*[Tt]ail" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: TILE_ELEMS_ALIGNED / tileSize 32B 对齐计算
grep -cE "ELEMS_PER_BLOCK|32\s*/\s*sizeof|sizeof\(T\)\s*[*]\s*(?:8|16|32)" \
    modified_files/op_kernel/*.cpp modified_files/op_kernel/*.h
# 期望: >= 1

# 检查 5: 没有遗漏的硬编码非对齐 tileSize
grep -E "constexpr\s+(?:uint32_t|int32_t)\s+TILE_(?:SIZE|ELEMS)\s*=\s*\d+" \
    modified_files/op_kernel/*.h | grep -vE "=\s*(?:32|64|128|256|512|1024|2048|4096|8192)\b"
# 期望: 0 行匹配（如果有匹配则说明 hardcode 了非 32B 对齐的 tile）
```

## Step 6: Known Pitfalls

| 现象 | 原因 | 修复 |
|---|---|---|
| 编译通过但精度对不上 | tail block padding 值不是 0（影响后续运算）| `padParams.paddingValue = 0` 显式设 |
| 性能与 baseline 持平 | shape 本来就对齐 → 改造无效 | 检查 totalElems % 32 是否为 0，是则 P7 不适用 |
| `DataCopyPad` 比 `DataCopy` 慢一倍以上 | DataCopyPad 走慢路径 | 主循环用 DataCopy，仅 tail 用 Pad（形态 β）|
| 多个 dtype 测试中 fp16 通过 fp32 挂 | ELEMS_PER_BLOCK 用了固定 8 | 改用 `32 / sizeof(T)` 动态算 |
| Cast 后 dtype 变，对齐失效 | 中间结果 dtype 不同 | Cast 前后都要 align check |

---

**完成清单**：
```
[P7 Playbook Completion]
Step 1: done (/tmp/p7_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: TILE_ELEMS_ALIGNED ≥ TILE_ELEMS; (mainLoops×aligned)+tail==total; ub≥aligned×sizeof(T); offset 32B aligned: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
