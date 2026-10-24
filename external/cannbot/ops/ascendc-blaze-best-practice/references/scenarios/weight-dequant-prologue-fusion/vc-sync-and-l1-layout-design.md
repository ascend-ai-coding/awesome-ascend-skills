# V+C 数据流、CV 同步与 L1 布局专题

本文是本场景 Kernel 层的 V+C 协作、CV 同步和 L1 内存规划专题。Step 3 用它设计 Kernel delta 和 Block delta 中的同步/L1 合同；DESIGN 冻结后，Step 3 用它编译对应 PLAN action；Step 4 只有在 PLAN 将本文绑定到当前 action 时才读取。本文不选择 Blaze 组装方案，也不提供固定同步 recipe。

## 1. 输入与输出合同

输入：`matmul_base_analysis`、Investigation 中已闭合的 CV 同步 API 事实、L1 容量事实和当前 Block/Prologue witness。输出：

```text
vc_dataflow_contract
cv_sync_contract
l1_layout_contract
bias_consistency_contract
sync_validation_additions
```

每个结论必须引用当前候选组装方案评估、witness 和 evidence IDs。

## 2. V+C 数据流合同

本场景的数据流方向为 V+C（AIV 先执行 → AIC 后执行），与 C+V epilogue 融合（AIC 先 → AIV 后）方向相反。DESIGN 必须冻结完整数据流：

| 阶段 | 执行核 | 数据路径 | 必填合同 |
|---|---|---|---|
| Prologue | AIV | GM B(low-bit) → UB → VF 反量化 → UB → L1(bf16/fp16 B_dequant) | 每段搬运的 dtype/layout/extent/offset |
| MMAD | AIC | GM A(bf16) → L1 → L0A; L1 B_dequant → L0B → MMAD → L0C → Fixpipe → GM | BlockMmad 的 K 循环、双缓冲和 L0C 输出 |

DESIGN 必须确认：
1. AIV 写入 L1 的 B_dequant 的 layout 与 AIC BlockMmad 从 L1 读取 B 的 layout 一致；
2. AIC 不执行 B 的 GM→L1 搬运（B 来源从 L1 读取，非 GM）；
3. A 侧搬运路径与官方 MatMul 一致（保留基础合同）。

## 3. L1 布局合同

L1 空间由 AIV 写入的 B_dequant 和 AIC 读取的 A 共享。DESIGN 必须冻结 L1 布局：

| 区域 | 内容 | 必填合同 |
|---|---|---|
| Lower half | B0 + A0 | B_dequant buffer 0 和 A buffer 0 的字节范围、对齐 |
| Upper half | B1 + A1 | B_dequant buffer 1 和 A buffer 1 的字节范围、对齐 |
| Tail（可选） | bias | 64-byte 对齐、`Align(baseN * biasElemBytes, 64)` |

DESIGN 必须回答：
1. B_dequant 和 A 在每个 half 内的排列顺序和字节边界；
2. bias 存在时 L1 尾部空间的计算公式和与 Tiling 的一致性；
3. ping-pong half 的切换与 K-L1 迭代的对应关系；
4. L1 总空间约束：`B_dequant + A + bias <= TOTAL_L1_SIZE`。

## 4. CV 同步合同

AIV 和 AIC 通过 CrossCore/HardEvent 同步。DESIGN 必须冻结同步时序：

| 事件 | 方向 | 时机 | 必填合同 |
|---|---|---|---|
| AIC_SYNC_AIV_FLAG | AIC → AIV | AIC 消费完 L1 B_dequant | flag ID、count、pipe 配对 |
| AIV_SYNC_AIC_FLAG | AIV → AIC | AIV 写完 L1 B_dequant | flag ID、count、pipe 配对 |

DESIGN 必须回答：
1. 首轮 K-L1 迭代是否预置 AIC_SYNC_AIV_FLAG（避免首轮阻塞）；
2. K-loop 内 AIV/AIC 的 wait/set 交替顺序；
3. 循环结束后剩余 flag 的消费（final drain）；
4. 是否使用 CrossCore 同步还是仅 HardEvent（本场景 AIV/AIC 在同一核内，使用 HardEvent 即可，无需 CrossCore）；
5. 预置 flag 的数量与 L1 buffer 数 × AIV sub-block 数的关系。

同步事件 ID、count 和 pipe 必须来自当前 Investigation 的源码事实，不从历史实现泛化。

## 5. Bias 类型一致性合同

Bias 默认为 `float`，也可改为 `bfloat16_t` 或 `half`。当 BiasType 改变时，以下 4 处必须同步修改：

| # | 组件 | 合同要求 |
|---|---|---|
| 1 | BlockMmad `using BiasType` | 与 DESIGN 冻结的 bias dtype 一致 |
| 2 | BlockMmad `tensorBiasL1` | 自动跟随 BiasType |
| 3 | Kernel `biasGmPtr` | 使用 `BiasType*`，不得用 `CType*` |
| 4 | Tiling `biasElemBytes` | `sizeof(BiasType)`，与 BlockMmad 一致 |

DESIGN 必须确认 GM/L1/BT 三处 bias 类型一致，否则 `CopyGM2L1` 的类型检查会编译失败。`biasL1Offset_` 使用 `sizeof(BiasType)` 计算 L1 空间，Tiling 的 `biasElemBytes` 必须与之匹配。

## 6. 对齐约束合同

DESIGN 必须冻结分形轴对齐约束：

| 条件 | 约束 | 原因 |
|---|---|---|
| transB=true | N 必须 16 对齐 | N 是分形轴（ZN），AIV 切分点需 16 边界 |
| transB=false | K 必须 16 对齐 | K 是分形轴（NZ），AIV 切分点需 16 边界 |
| transB=true | 尾轮 N 方向不切分 | `CeilDiv` 不保证 16 对齐，破坏分形边界 |
| M | 任意 | M 不是分形轴 |

Tiling 入口必须校验分形轴对齐，违反时抛出 `std::runtime_error`。

## 7. 验证门禁

- V+C 数据流方向正确（AIV 先 → AIC 后），无反向依赖；
- L1 布局的 B_dequant/A/bias 字节边界与 Tiling 一致；
- CV 同步的 flag 预置、交替和 final drain 闭合，无死锁或残留 flag；
- BiasType 在 GM/L1/BT/Tiling 四处一致；
- 分形轴对齐约束在 Tiling 入口校验；
- L1 总空间约束满足；
- 官方源码区与 Blaze Asset 原文件保持零改动。

具体 flag ID、L1 偏移、bias dtype 和对齐值来自 DESIGN/PLAN 和当前 Investigation，不由本文固定。
