# Prologue UB 布局与空间约束专题

本文是本场景 Prologue 层的 UB 内存规划专题。Step 3 用它设计 Prologue delta 中的 UB 布局合同；DESIGN 冻结后，Step 3 用它编译对应 PLAN action；Step 4 只有在 PLAN 将本文绑定到当前 action 时才读取。本文不选择 Blaze 组装方案，也不提供固定 UB recipe。

## 1. 输入与输出合同

输入：`matmul_base_analysis`、Investigation 中已闭合的 UB 容量事实、反量化公式合同和当前 Tiling 参数。输出：

```text
ub_layout_contract
ub_space_constraint
ub_lifecycle_contract
ub_validation_additions
```

每个结论必须引用当前候选组装方案评估、witness 和 evidence IDs。

## 2. UB ping-pong 布局合同

Prologue 在 AIV 侧使用 UB 进行反量化。DESIGN 必须冻结 UB 的双 buffer ping-pong 布局：

| 对象 | 必填合同 |
|---|---|
| bIn | 输入 B 的低比特 dtype（int8, 1B）、每个 buffer 的 N×K 元素数、ping-pong 双 buffer |
| bOut | 反量化后 B_dequant 的 dtype（bf16, 2B）、每个 buffer 的 N×K 元素数、ping-pong 双 buffer |
| scale | perchannel scale 的 dtype、shape `(N,)`、每个 half 各一份的加载策略 |
| offset（可选） | perchannel offset 的 dtype、shape `(N,)`、每个 half 各一份的加载策略 |

DESIGN 必须回答：

1. 每个 half 内 bIn/bOut/scale/offset 的排列顺序和字节范围；
2. scale/offset 的加载时机（首轮 K-L1 迭代加载，后续迭代复用）；
3. ping-pong buffer 的切换时机和与 K-L1 迭代的对应关系；
4. 每个 buffer 内低比特输入和 bf16/fp16 输出的共存约束。

## 3. UB 空间约束合同

DESIGN 必须冻结 UB 空间约束公式。公式必须按当前 B dtype 参数化，不得硬编码 int8 系数：

```text
2 * (weightElemBytes + dequantBElemBytes) * kUbSize * nUbSize + 2 * dequantBElemBytes * nUbSize * (1 + hasOffset) <= UB_SIZE
```

其中：
- `weightElemBytes = sizeof(BType)`：B 权重输入字节数（int8/fp8=1B）
- `dequantBElemBytes = sizeof(DequantBType)`：B_dequant 字节数（bf16/fp16=2B，= scale/offset 字节数）
- `2 * (weightElemBytes + dequantBElemBytes)`：bIn 和 bOut 各 2 个 buffer 的每元素字节数
- `2 * dequantBElemBytes`：scale 的 2 个 buffer；offset 同理（仅 hasOffset=true 时计入）
- `kUbSize`：单 AIV 单 buffer 的 K 方向元素数
- `nUbSize`：单 AIV 单 buffer 的 N 方向元素数

int8/fp8 时系数 = `2×(1+2)=6`。

PLAN 必须验证当前 Tiling 参数满足此约束。约束不满足时调整 kUbSize/nUbSize 或返回 Step 3 重新选择 Tiling 合法值。

## 4. UB 生命周期合同

DESIGN 必须冻结每个 UB 区域的生命周期：

| 区域 | 生产者 | 消费者 | 生命周期 |
|---|---|---|---|
| bIn[buf] | GM→UB DataCopy | VF 反量化 | 从 DataCopy 完成到 VF 消费完成 |
| bOut[buf] | VF 反量化写入 | UB→L1 DataCopy | 从 VF 写入完成到 L1 搬运完成 |
| scale[buf] | GM→UB DataCopy | VF Mul | 从加载到该 half 最后一次 K-L1 迭代 |
| offset[buf] | GM→UB DataCopy | VF Add | 从加载到该 half 最后一次 K-L1 迭代 |

ping-pong buffer 的覆盖前等待通过 HardEvent 同步保证。DESIGN 必须确认同步事件配对闭合，不得依赖 UB 隐式时序。

## 5. 验证门禁

- UB 空间约束公式在当前 Tiling 参数下满足；
- ping-pong buffer 的切换与 K-L1 迭代对齐，无数据竞争；
- scale/offset 的加载时机正确（仅首轮加载，后续复用）；
- hasOffset=false 时 UB 布局正确排除 offset 区域；
- UB 区域的字节边界和对齐满足 DataCopy 要求；
- 官方源码区与 Blaze Asset 原文件保持零改动。

具体 kUbSize、nUbSize、hasOffset 和 UB_SIZE 来自 DESIGN/PLAN 和当前环境事实，不由本文固定。

## 6. UB Bank 冲突优化

### 6.1 Bank 冲突原理

UB 共 16 个 bank，每 32 字节为一个 DataBlock，bank 索引 = `DataBlock索引 % 16`。`StoreAlign<T, DATA_BLOCK_COPY>` 一次写 256 字节 = 8 个 DataBlock。当 `dataBlockStride % 16 == 0` 时，8 个 DataBlock 全映射到同一 bank，需 8 个 cycle 完成而非 1 个 cycle，导致 StoreAlign 吞吐降为 1/8。

### 6.2 修复方法

将 `dataBlockStride`（即 VF 函数的 `outerAxisSize`）从外轴元素数改为**外轴元素数 + 1**，使 8 个 DataBlock 分布到不同 bank：

```text
outerAxisSize = (transB ? nUbLen : kUbLen) + 1
```

### 6.3 Padding 方向

padding 加在外轴方向：

| transB | 分形格式 | 外轴 | padding 方向 | outerAxisSize |
|---|---|---|---|---|
| false | NZ | K | K+1 | kUbLen + 1 |
| true | ZN | N | N+1 | nUbLen + 1 |

### 6.4 UB Buffer 大小调整

`bOutOneBuffer_` 必须包含 padding 空间：

- transB=false: `(kUbSize + 1) * nUbSize * sizeof(DequantBType)`
- transB=true: `kUbSize * (nUbSize + 1) * sizeof(DequantBType)`

`bInOneBuffer_` 不需要 padding（bIn 是低比特输入，不通过 StoreAlign 写出）。

### 6.5 UB→L1 搬运改用 copy_ubuf_to_cbuf

padding 后，UB 中数据的物理 stride 与 NZ/ZN layout 描述的 stride 不一致（layout 的 stride 是 16 对齐的，无法描述 +1）。因此不能用 `Te::Copy(CopyUB2L1)`（它从 layout 自动计算 srcGap），必须用底层 `copy_ubuf_to_cbuf` 显式指定参数：

```text
copy_ubuf_to_cbuf(dst, src, sid, nBurst, lenBurst, srcGap, dstGap)
  dst:      L1 目标地址
  src:      UB 源地址
  sid:      stream id, 固定为 0
  nBurst:   外轴 burst 数（外轴 16-group 数）
  lenBurst: 每个 burst 的 DataBlock 数（内轴连续 DataBlock 数, 32B/个）
  srcGap:   相邻 burst 间在 UB 中跳过的 DataBlock 数（bank conflict padding = 1）
  dstGap:   相邻 burst 间在 L1 中跳过的 DataBlock 数（L1 对齐 padding）
```

NZ（transB=false，外轴=N, 内轴=K）：
- `nBurst = CeilDiv(nUbLen, 16)`，`lenBurst = kUbLen`，`srcGap = 1`，`dstGap = CeilAlign(curKL1, 16) - kUbLen`

ZN（transB=true，外轴=K, 内轴=N）：
- `nBurst = CeilDiv(kUbLen, 16)`，`lenBurst = nUbLen`，`srcGap = 1`，`dstGap = CeilAlign(curN, 16) - nUbLen`

### 6.6 性能影响

bank 冲突修复后，AIV Vec 时间降低 48%~62%（StoreAlign 写入吞吐提升 8 倍）。整体 Task Duration 的改善取决于 AIC MAC 是否为瓶颈——当 AIC MAC 占主导时整体改善有限，小 shape 或 AIV 成为瓶颈时效果更显著。
