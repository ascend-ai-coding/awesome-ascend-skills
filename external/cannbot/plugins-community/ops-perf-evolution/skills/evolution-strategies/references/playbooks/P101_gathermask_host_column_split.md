# P101 Playbook: GatherMask 替代 Host 侧列拆分

> 本 Playbook 为**强制流程**。采纳 P101 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P101 的核心是**用 kernel 内 GatherMask 解交织替代 Host 侧 `.select().contiguous()` 列拆分**，将多路 DMA 降为 1 路 DMA + 4 条 VEC GatherMask 指令。内置 pattern 3/4/5/6 对应多分量分离。

## Step 1: 定位关键结构

```bash
# Host 侧列拆分（本策略要消除的目标）
grep -nE "\.select\(.*\)\.contiguous\(\)" op_host/*.cpp > /tmp/p101_locations.txt

# Interleaved 多分量输入 (N,2)/(N,3)/(N,4)
grep -nE "size\(1\)\s*==\s*[234]" op_host/*.cpp >> /tmp/p101_locations.txt

# Kernel 侧现有 DMA 调用
grep -nE "DataCopyPad|CopyIn|CopyOut" op_kernel/*.cpp >> /tmp/p101_locations.txt

# 当前 TILE_N 和 UB 缓冲区配置
grep -nE "TILE_N|InitBuffer|bufNum|BUFFER_NUM" op_kernel/*.cpp >> /tmp/p101_locations.txt
```

> 注意：op_host/op_kernel 的路径因 pipeline 类型而异。lingxi-evo 为 `kernel/op_host/`、`kernel/op_kernel/`；ops-evo 为 `shared/original/op_host/`、`shared/original/op_kernel/`。以实际目录结构为准。

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **Host 侧 `.select().contiguous()` 调用数量及行号**
- **Kernel 侧当前 CopyIn 数量及行号**
- **当前 TILE_N、UB 缓冲区数量**

## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| Host 列拆分 | N× `.select().contiguous()` | 0（直接传原始 (N,C) 交错格式） | `op_host/*.cpp` |
| GM tensor 数量 | 当前值 | 仅保留原始交错 tensor + output（**含必然推论：被消除 tensor 的计算迁移到 kernel**） | `op_host/*.cpp` + `op_kernel/*.cpp` |
| Host 侧依赖被消除列的预计算 | 若有（如 area = (x2-x1)*(y2-y1)） | **移至 kernel UB 向量计算**，不再作为独立 GM 输入 | `op_host/*.cpp` + `op_kernel/*.cpp` |
| Kernel CopyIn | 多路 DataCopyPad（每分量一路） | 1 路 CopyIn（交错格式）+ 4×GatherMask | `op_kernel/*.cpp` |
| 解交织位置 | 不适用 | **外层循环**（UB 常驻，内层复用） | `op_kernel/*.cpp` |
| TILE_N | 当前值 | 向上填满 UB：`(192KB) / (buf_count × sizeof(T))` | `op_kernel/*.cpp` |

> **必然推论（禁止跳过）**：消除列拆分的 GM tensor 后，任何**基线中依赖这些 tensor 的 Host 侧预计算结果**不再有独立 GM 通道传入 kernel。你必须：
> 1. 列出基线 op_host 中所有 GM 输入（`grep EXEC_KERNEL_CMD op_host/*.cpp`）
> 2. 逐个判定是否依赖被消除的列 → 若是，判定能否在 kernel UB 上重算 → 若不能，评估是否超过 P101 节省的 DMA 开销
> 3. 改造后 kernel `SetGlobalBuffer` 数量应严格 ≤ 改造前 - 被消除的原始分量数

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整 GatherMask 解交织**：1 CopyIn → 4 GatherMask（pattern 3/4/5/6），TILE_N 填到 UB 上限。**解交织必须放外层循环**（解交织后 UB 常驻，内层维度全量复用），放内层则每 tile 都要重 DMA + 重 GatherMask → DMA 复杂度 O(外层×内层/tile²)。
- **形态 β — 保底标量解交织**：GatherMask API 不可用时用标量 loop，但必须在 implementation_note 中标注性能风险（标量 loop 是已知性能杀手，实测比 GatherMask 慢 3-6x）。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta`

### 3B. Canonical Template（形态 α）

```cpp
// ==================================================
// Host 侧：消除所有 .select().contiguous()
// ==================================================
// 改造前：
//   at::Tensor col0 = input.select(1, 0).contiguous();
//   at::Tensor col1 = input.select(1, 1).contiguous();
//   ... // N 路 .select() → N 路 column tensor
// 改造后：
//   直接传原始 (N, C) 交错格式，不做任何列拆分
//   仅保留必要的 dtype 转换（如 FP16→FP32）

// ==================================================
// Kernel 侧：1 路 CopyIn + 4×GatherMask
// ==================================================

// 1. 加载交错 tile（1 次 DMA）
DataCopyExtParams cp{1, static_cast<uint32_t>(tileLen * C * sizeof(float)), 0, 0, 0};
DataCopyPadExtParams<float> pp{false, 0, 0, 0.0f};
DataCopyPad(interleaved, srcGm[tileStart * C], cp, pp);
PipeBarrier<PIPE_ALL>();

// 2. GatherMask 解交织 — Normal 模式，每次 repeat 处理 16 dst 元素
//    每次 repeat 读取 16×C 源元素 = 256B（C=4 时），src0RepeatStride=8 DataBlocks
int32_t alignedLen = (tileLen / 16) * 16;
uint16_t repeatTimes = static_cast<uint16_t>(alignedLen / 16);

// !!! CRITICAL: src0RepeatStride=8 不可写成 1 !!!
// stride=1 会导致连续 repeat 重叠 224B → NaN/Inf
GatherMaskParams gmParams(1, repeatTimes, 8, 0);
uint64_t rsvdCnt = 0;

// Built-in patterns: Normal 模式，每 4 个取第 k 个（0-indexed）
// pattern 3 (00010001): offset=0 → 分量 0
GatherMask(col0, interleaved, static_cast<uint8_t>(3), false, 0, gmParams, rsvdCnt);
// pattern 4 (00100010): offset=1 → 分量 1
GatherMask(col1, interleaved, static_cast<uint8_t>(4), false, 0, gmParams, rsvdCnt);
// pattern 5 (01000100): offset=2 → 分量 2
GatherMask(col2, interleaved, static_cast<uint8_t>(5), false, 0, gmParams, rsvdCnt);
// pattern 6 (10001000): offset=3 → 分量 3
GatherMask(col3, interleaved, static_cast<uint8_t>(6), false, 0, gmParams, rsvdCnt);
PipeBarrier<PIPE_ALL>();

// 3. 标量尾块（tileLen 非 16 倍数时补齐）
for (int32_t j = alignedLen; j < tileLen; ++j) {
    int32_t off = j * C;
    col0.SetValue(j, interleaved.GetValue(off + 0));
    col1.SetValue(j, interleaved.GetValue(off + 1));
    col2.SetValue(j, interleaved.GetValue(off + 2));
    col3.SetValue(j, interleaved.GetValue(off + 3));
}
```

### 3C. Variant Notes

- **stride=8 踩坑警告**：`src0RepeatStride=8` 不能写成 1。stride=1 导致连续 repeat 重叠 224B → 全 case 精度 fail（NaN/Inf）。
- **16 对齐要求**：Normal 模式每次 repeat 输出 16 dst 元素，tileLen 向下对齐到 16 倍数。尾块 0~15 元素走标量补齐。
- **UB 容量公式**：交错 raw buffer 占 `TILE_N × C × sizeof(T)` 字节，C 个分量各占 `TILE_N × sizeof(T)`。总 UB = `TILE_N × 2C × sizeof(T)`（不含其他计算 buffer）。据此反推 TILE_N 上限。
- **CANN 8.5.1 API**：`GatherMaskParams` 必须显式构造对象：`GatherMaskParams(src0BlockStride, repeatTimes, src0RepeatStride, src1RepeatStride)`。不支持初始化列表。

### 3D. 收益与边界

P101 的收益取决于两个维度上的复用程度：
- **消除的 DMA 路数**：原先是每分量一路 CopyIn，改造后是 1 路 DMA + 4 条 VEC GatherMask。分量数 C 越大、原 DMA 路数越多，相对收益越高。
- **On-device 替代 GM 预计算**：原先在 host 侧对拆分后的列做预计算再通过 GM 传入 kernel，改造后直接在 kernel 内向量计算。减少的 GM 读取次数 = 预计算结果的路数。

适用前提：输入为 `(N, C)` 交错格式，C>1，且交错维度上需要读入 UB 后做多次随机、向量或逐元素访问（而非单次逐分量读写）。若每个分量仅被消费一次且不涉及跨分量运算，标量 GM 访问的额外开销远小于 GatherMask 的指令和 UB 开销，此时 P101 反而可能退化。

**必然推论**：若基线有对拆分后各列做的预计算结果作为独立 GM 输入，消除列拆分后该预计算结果失去数据来源 → 必须移至 kernel UB 内用向量指令重算。残留其 GM DMA 会抵消列拆分节省的 DMA 收益，导致 P101 退化。

## Step 4: 约束复核

- **UB 容量**：总 buffer 字节数 < 芯片 UB 上限
- **32B 对齐**：CopyIn 偏移保证 32B 对齐（每次偏移 × C_float = 4×C 字节，双拍 = 2×4×C ≥ 32）
- **精度验证**：GatherMask 解交织结果与标量方式逐位一致（pattern 3/4/5/6 是精确无舍入的映射）
- **接口兼容**：算子签名不变，仅内部实现变化
- **循环顺序**：解交织在外层循环 — 确认 grep 验证

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**。

## Step 5: 编码后自检

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 0. 确认解交织在外层循环（不是内层！此项最优先）
grep -nE "GatherMask|deinterleave|pipe.InitBuffer.*bufRaw" op_kernel/*.cpp
# 目视检查：GatherMask 调用在 for(r=startRow) 之前还是之内？
# 必须在外层循环！

# 1. Host 侧：确认消除列拆分
grep -cE "\.select.*contiguous" op_host/*.cpp  # ==0

# 2. Kernel 侧：确认 GatherMask 接入
grep -cE "GatherMask" op_kernel/*.cpp  # >=4

# 3. 确认关键参数正确
grep -cE "repeatTimes.*16|alignedLen.*16" op_kernel/*.cpp  # >=2
grep -cE "src0RepeatStride.*8|stride.*8" op_kernel/*.cpp  # >=1
grep -cE "static_cast<uint8_t>\([3-6]\)" op_kernel/*.cpp  # >=4

# 4. 确认 CopyIn 减少
grep -cE "DataCopyPad|CopyIn" op_kernel/*.cpp  # 应 < 改造前数量

# 5. 【必然推论】确认 GM tensor 数量减少（预计算结果已迁移到 kernel）
echo "改造前 GM tensor 数: N"  # 从 Step 1 记录的值
echo "改造后:"
grep -cE "SetGlobalBuffer" op_kernel/*.cpp     # 应 ≤ 改造前 - 被消除的原始分量数
# 若相等或接近 → 必然推论未执行，P101 大概率退化
```

## Step 6: Known Pitfalls

| 现象 | 根因 | 修复 |
|---|---|---|
| 全 case 精度 fail（NaN/Inf） | `src0RepeatStride=1` 导致 repeat 重叠 | 改为 `GatherMaskParams(1, r, 8, 0)` |
| 大 case 性能退化严重 | tile 过小 → DMA 次数随 N 线性增长 | 增大 TILE_N 到 UB 上限 |
| 编译错误：GatherMaskParams 构造 | CANN 8.5.1 不支持初始化列表 | 显式构造：`GatherMaskParams gm(1, r, 8, 0)` |
| VECTOR 对齐 trap | 源或目标缓冲区非 32B 对齐 | 确保 TBuf 声明为 `TBuf<TPosition::VECCALC>` |
| UB 溢出 | 缓冲区过多 → TILE_N 超出 UB | 减少不必要的中间 buffer 或复用 |
| 小 N 场景性能无提升 | 仅解交织一个大维度，另一个用标量 GM 访问 | 按"关键修改点 5"原则：只解交织需要复用的大维度 |

---

**完成清单**：
```
[P101 Playbook Completion]
Step 1: done (/tmp/p101_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta, canonical/variant applied
    解交织方向 = 外层循环（确认），tile 填到 UB 上限
Step 4: UB 容量通过; 32B 对齐通过; 精度验证通过; 循环顺序确认: yes/no
Step 5: all 5 grep checks passed（详见下文）
Step 6: no pitfalls triggered / {列出触发的}
```
