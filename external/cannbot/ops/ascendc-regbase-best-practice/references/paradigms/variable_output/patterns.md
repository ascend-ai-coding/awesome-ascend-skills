# VariableOutput 类算子场景路由

> 本文档用于**场景判定**和**策略选择**。确定场景后，按链接进入对应详细文档。
>
> VariableOutput 是**修饰范式**（与 NumericalStable/DynamicShape 同类，必须与主范式结合）：
> 输出的 shape 由 **device 侧计算结果**决定——非零个数、标签计数、去重个数——
> 执行前不存在，host 侧任何机制都推不出来（输出**个数**则恒由属性/构图决定，列表形态见
> Step 0）。它不接管计算骨架（数据流/切分/buffer 仍由主范式
> 决定），只接管 **shape 定形链路**：outshape 回传通道 + kernel 侧叠加的计数定形结构。相比
> NumericalStable（只改 Compute 公式）和 DynamicShape（只改 host 契约），它是侵入最深的修饰——
> 改 kernel 签名（yshape 形参）、改多核协调（计数 workspace + 前缀和）、加写出阶段（唯一核写
> shape），但依然没有自己的模板体系。机制全链路（四阶段生命周期、outshape 回传协议、kernel
> 计数骨架）见 [knowledge.md](knowledge.md)——**动手设计前必读其 §2~§5**。

---

## 场景判定流程

```
给定: spec.yaml 输入输出 + 语义公式

Step 0 — DYNAMIC 侧别快筛（仅当参数带 DYNAMIC 时）:
  DYNAMIC 在输入侧（列表输入，如 Pack/Concat 的 x）→ 与本范式无关，按主范式处理
  DYNAMIC 在输出侧 → 只说明"输出是列表"，继续 Step 1（列表 ≠ 本范式，Split 即反例）

Step 1 — 输出维长来源判定（逐输出 × 逐场景分支，取层级最大值）:
  对每个输出的每个维长回答"由什么推出"（分支依赖的按最坏分支）:
    ├─ 全部输出 ≤ L1（输入 shape/dtype + 常量 attr，如 Add 广播）→ 普通算子，按主范式 → 结束
    ├─ 全部输出 ≤ L2（输入张量的值，且算子契约保证执行前 host 可读：`ValueDepend(REQUIRED)`
    │    强制常量，或值依赖声明触发 D2H）→ DynamicShape 值依赖范式 → dynamic_shape/patterns.md
    │    ⚠ "常量输入时 host 恰好能推出"只是退化分支，不改归类——契约允许运行期输入
    │      且未声明值依赖时，按最一般执行路径归 L3（BroadcastGradientArgs 即此形态）
    └─ 存在 L3（对输入数据的统计/计算结果，非零个数、标签计数、去重个数，
         执行完才知道）→ VariableOutput → Step 2
  值只影响上界的输入（max_output_size 类）：是否做值依赖收紧 range 是可选设计
  （收紧：StatelessRandomChoiceWithMask；不收紧取输入规模上界：SortedNMS max=N），不影响归属

Step 2 — 输出形态判定:
  输出个数是否运行期/属性驱动的列表（K 个）？
    ├─ 是（如 DynamicPartition 的 K=num_partitions 个 y[k]）
    │    → Output("y").OutputShapeDependOnCompute().ParamType(DYNAMIC)
    │      图模式构图用 create_dynamic_output_y(K)，kernel 用 ListTensorDesc 取各输出
    ├─ 否（个数固定，仅维长未知，如 MaskedSelectV3/NonZero 单输出）
    │    → Output("y").OutputShapeDependOnCompute().ParamType(REQUIRED)
    └─ 多固定输出且子集依赖计算（SparseSlice：y_indices/y_values 动态、y_shape 静态）
         → 依赖计算的输出逐个声明 OutputShapeDependOnCompute，静态输出正常 InferShape
           （Range 可全输出注册、静态份 min=max；kernel 侧静态输出槽位照写，见
             knowledge.md §3.2）

Step 3 — infershape 注册风格:
  输出维数（rank）能否由输入（含 attr）推出？
    ├─ 能 → InferShape(未知维置 -1) + InferShapeRange(min/max)
    │       （MaskedSelectV3 / NonZero / StatelessRandomChoiceWithMask 风格）
    └─ 不能或输出本身是列表 → 仅 InferShapeRange，每个输出复制同一份 range
            （DynamicPartition 风格）

Step 4 — kernel 定形策略:
  是否存在「shape 可提前知道」的退化分支？
    ├─ 有（如 DynamicPartition K=1 → dim0 恒 H）→ tiling 预填 shape，kernel 照抄
    │   （值依赖输入为编译期常量时也可 infershape 期定形，退化分支谱系见 knowledge.md §5.4）
    └─ 无 → kernel 计数定形，走两遍法/重复扫描/定序骨架（见 §kernel 计数骨架）
```

---

## 通用规则

**判定法则：输出某个维长 = 对输入数据值的统计量，必须执行计算才能揭晓 → VariableOutput。**

| 维长来源 | 范式归属 | 实例 |
|---------|---------|------|
| 输入 shape/dtype + attr | 普通（主范式） | Add、Concat、Pack（DYNAMIC 在输入侧，输出 shape 纯计算）、ReduceSum |
| 输入张量的值（契约保证执行前 host 可读） | DynamicShape（值依赖） | Reshape、MirrorPad、Slice |
| 计算结果（执行后才存在） | **VariableOutput（本范式）** | NonZero、MaskedSelectV3、DynamicPartition |
| 输出个数动态但每个 shape 已知 | **非** VariableOutput（仅列表输出） | Split、IdentityN、ShapeN |

> 两个高频混淆点：**DYNAMIC_OUTPUT ≠ VariableOutput**（前者只说明输出是列表，shape 可编译期已知，
> 如 Split 按 attr 切分后每份 shape 全已知；"已知"含 L1 纯计算与 L2 值依赖 host 推出两种，皆非
> L3 即排除——GroupedMatmul 的 y[g] 靠 host 读 group_list 值定形，属 L2 列表输出，仍是反例）；
> **值依赖救不了 VariableOutput**（值依赖是执行**前** host 读输入，本范式信息执行**后**才产生）。
> 值依赖与本范式可组合：输出维长既依赖输入值上界又依赖计算，或算子兼有值依赖输入
> （StatelessRandomChoiceWithMask 的 count）。

## 变长信息的三条去向（设计选型）

"输出 shape / 有效长度依赖输入"的语义，有三种合法消化路线。**设计新算子时主动选型**，
判定既有算子时看它实际选了哪条：

| 路线 | 机制 | 输出 shape | 下游如何拿到有效长度 | 实例 |
|------|------|-----------|-------------------|------|
| A 值依赖（L2） | 声明 ValueDepend + D2H，host 推导 | host 期精确已知 | 框架 shape 传播 | Reshape、MirrorPad |
| B outshape 回传（L3，本范式） | kernel 计数 + 执行后回传 | 按 [min,max] 预分配，执行后刷新 | 框架 shape 传播 | NonZero、DynamicPartition |
| C max-size + count 值输出 | 按最坏情况静态分配，计数进输出**值** | 编译期静态（= 上界） | 下游读 count 值自行切片 | NonZeroWithValue |

选型判据：

- 下游/生态需要框架级 shape 传播（图编排、shape 校验依赖真实 shape）→ A 或 B；shape 信息在
  **输入值**中且可接受 D2H 同步 → A；信息是**计算产物**或要免同步 → B
- 最坏情况内存可控（numel 级上界可接受）且下游是"读 count 切片"风格（PyTorch 生态常见）→ C，
  换来整条 outshape 链路的免除
- 同名输出可走不同路线：UniqueDim 的 count 走 B（shape 承载，输出维长即计数）、
  NonZeroWithValue 的 count 走 C（静态 [1]，值承载）——**路线选择不改变语义，只改变契约**。
  选了 C 的算子输出 shape 静态可推、不属于本范式；其 kernel 仍可借用两遍法，但用途是**数据
  定位**（向 max-size 缓冲 scatter）而非 shape 定形，勿混淆

---

## 四阶段契约（每个阶段各有一条硬约束）

| 阶段 | 契约 | 反例后果 |
|------|------|---------|
| 1 op_def | 输出参数声明 `.OutputShapeDependOnCompute()`（`register/op_def.h` OpParamDef API）；列表输出再加 `.ParamType(DYNAMIC)`；动态能力开 `DynamicShapeSupportFlag(true)` 等（能力开关非范式标识——非本范式算子也可全开）。部分实现在 `IMPL_OP_INFERSHAPE` 注册块同时声明 `.OutputShapeDependOnCompute({输出索引...})`（`register/op_impl_registry.h` 的 OpImplRegisterV2 API，如 unique_dim `{0,1,2}`）——存量两种风格并存：仅 op_def 侧声明已实证充分，IMPL 侧可选（详见 knowledge.md §2.2） | 不声明 → 框架按静态 shape 编排，无 outshape 通道 |
| 2 infershape | 推不出精确 shape，只能给范围：注册 `InferShapeRange`（min/max）；能定 rank 的（含 attr 信息）再加 `InferShape`（未知维 -1）。**max = 运行期预分配内存大小，真实 shape 必须 ⊆ [min, max]** | max 给小 → 运行期越界写；max 给大 → 内存浪费 |
| 3 tiling | **只看输入**（输出 shape 此时不存在）；多核时为跨核计数规划 user workspace（布局按算子设计，如 DynamicPartition 的 `usedCoreCnt × min(K, 4096) × 8`、masked_select_v3 的 offset 表；单核速决则不留计数缓冲，勿死预留）。outshape 缓冲自身占用一个 workspace 参数槽位：按实际槽数 `GetWorkspaceSizes(n)` 对齐、该槽大小置 0（见 knowledge.md §2.3） | tiling 试图读输出 shape → 拿到的是 -1/max；槽位数不匹配 → kernel 参数错位 |
| 4 kernel | 框架在参数表自动附加 `GM_ADDR yshape`（紧跟输出、workspace 之前）；kernel 必须把每个输出的真实 shape 按协议写出；唯一核写出（惯例末核，亦可指定核如 0 号核）；user workspace 一律从 `GetUserWorkspace(workspace)` 起用（裸 workspace 基址前 2MB 是系统保留区） | 不写 → 框架拿不到真实 shape；写 raw 基址 → 踩系统 workspace |

## outshape 回传协议（摘要）

编码（uint64）：`word[0] = dimNum [| B64_FLAG(1<<31)]`、`word[1] = dim0`、
`word[2..dimNum] = 其余各维`。
框架按 word[0] 的 dimNum 解析，其后写多少 word 跟随 dimNum——`SHAPE_GAP=9`（dimNum+8 维）只是
DynamicPartition 的保守整块布局（含填充 1），不是硬协议（masked_select_v3 单维输出只写 2 个
word）。三要素分层：**B64 打法是协议位**（存量写法有分歧：DynamicPartition 仅首个输出打且写完
恢复、UniqueDim 全部输出打、UniqueConsecutive 不打——从 knowledge.md §3.2 已证实写法中任选
其一，全算子一致，不自创）；**word 数与填充值是布局惯例**，可与协议位独立组合（SortedNMS 补 0
与 DynamicPartition 补 1 并存）。注入缓冲按**每个输出一个 SHAPE_GAP 槽位**排布（含固定多输出
与未声明输出的槽位，见 knowledge.md §3.2）。
完整协议与写法分歧记录见 [knowledge.md §3.2](knowledge.md)。

## kernel 计数骨架（两遍法）

多核写同一输出 y[k] 的标准解法（DynamicPartition H 切分主线）：

```
Step 1 各核本地计数:  扫自己段的标签 → 每分区计数写入 user workspace 本核区段
Step 2 SyncAll + 排他前缀和: 每核把排在它前面的核的计数累加 → 本核在各分区的起始行号
Step 3 搬运:        按预定偏移 scatter，计数器边搬边涨
Step 4 末核写 shape: 前缀和副产品——末核计数终值 = 全局总数，写进 yshape（其余核不写）
```

替代与退化：重复扫描免同步（每核扫全量、各核计数天然全局，代价是标签重复读，W 切分用）；
**输出序有语义**（NMS/TopK 贪心类）→ 前缀和不可用，改"多核并行预处理 + 单核按序 append"定序
骨架（knowledge.md §5.3）；数据量极小 → 单核速决（BlockDim=1，唯一写者天然成立、免计数缓冲，
knowledge.md §5.4）；K=1 / shape 可提前知道 → tiling 预填、kernel 照抄；空输入 → 仍要写出
shape（协议性输出：写全 0 shape 或 tiling 拒空二选一，全算子一致；混合输出形态下静态输出
照常写，见 knowledge.md §3.2）。

---

## 与其他范式的关系

- **主范式**：VariableOutput 算子的数据搬运/计算骨架仍由主范式决定（DynamicPartition 的搬运属
  搬运类主范式；MaskedSelectV3 的掩码压缩选择属
  [MaskPredicate](../mask_predicate/patterns.md)）。本范式只接管"shape 定形链路"
- **DynamicShape（值依赖）**：分界见判定法则表与"三条去向"节；组合场景两侧声明都要做，两种
  已证实组合形态——
  1. 值依赖输入参与 range **上界推导**（StatelessRandomChoiceWithMask 的 count：max = max(shapeSize, count)）
  2. 值依赖输入在 **tiling 期被消费**、预填辅助数据并**决定退化分支**（SparseSlice 的
     shape/start/size：TilingInputsDataDependency 读值预填切片边界，空盒判空走空输出模板——
     "tiling 期定形"的提前知道来源恰是值依赖输入）
- 修改清单：op_def / infershape / tiling（workspace）/ kernel（yshape + 计数）四处 ✅ 改；
  Compute 语义部分 ❌ 不改（跟主范式）

---

## 校验清单

**判定为"是"（本范式）时**：

- [ ] `OutputShapeDependOnCompute` 已声明（op_def 输出参数必备；IMPL 注册块声明可选，见 knowledge.md §2.2）
- [ ] infershape 注册了 `InferShapeRange`，且 max 是"可能的最大维长"的真实上界（含 -1 传染处理）
- [ ] 输出维数可推出时（含 attr）`InferShape` 把未知维置 -1（不是随便给个值）
- [ ] tiling 不读输出 shape；多核时 user workspace 已含跨核计数缓冲（单核速决则不留死预留）；
      outshape 的 workspace 槽位按 `GetWorkspaceSizes(n)` 对齐
- [ ] kernel 签名含 `yshape` 形参（紧跟输出、workspace 之前）；计数缓冲从 `GetUserWorkspace(workspace)` 起用
- [ ] yshape 写出编码：B64 打法从 knowledge.md §3.2 已证实写法中选取，word 数/填充值为布局惯例可自便，全算子一致
- [ ] 只有唯一核（惯例末核，或指定核）写 yshape，无写冲突
- [ ] kernel 写出的 shape 与 infershape 的静态结论不矛盾（含各退化分支：哑输出、空输入的 rank；
      双模算子常量分支 host 推导值 = kernel 计数值）
- [ ] 退化分支（K=1、空输入、标量）各自写全 shape；空输入可选"写全 0 shape"或"tiling 拒空"两种策略之一，全算子一致（混合输出形态下静态输出照常写真实 shape）
- [ ] 运行期真实 dim0 ≤ InferShapeRange max（协议一致性，测试必测）
- [ ] 多核写同一输出的偏移互不重叠（前缀和正确性，用可手算用例验证）

**判定为"否"时（反向核验）**：

- [ ] 语义含统计量/有效长度 → 确认变长信息有静态化载体（路线 C：max-size 上界 + count/有效
      长度值输出；或路线 A：值依赖 host 推导），不是"漏建 outshape 通道"的缺陷
- [ ] op_def 声明与 kernel 形参**互相一致**：声明了 DependOnCompute 则 kernel 必有 shapeout
      形参，反之亦然（一侧有一侧无 = 缺陷信号）
- [ ] 下游消费方式与所选路线匹配（值承载时下游读 count 切片，非框架 shape 传播）
- [ ] DYNAMIC 参数的侧别（输入/输出）已核对，未把列表形态当范式证据；动态能力 flag 未被当作
      范式标识

---

## 跨场景参考

| 主题 | 文档 |
|------|------|
| 四阶段机制全链路、outshape 协议细节、kernel 计数技术、案例库 | [knowledge.md](knowledge.md) |
| 范式自包含声明（算子名与案例库仅为举证，分析不依赖检索库上实现） | [knowledge.md §9](knowledge.md) |
| 值依赖范式（执行前 host 读输入值的场景） | [dynamic_shape/patterns.md](../dynamic_shape/patterns.md) |
| 值依赖三处契约摘要 | [common/value-depend-process.md](../common/value-depend-process.md) |
