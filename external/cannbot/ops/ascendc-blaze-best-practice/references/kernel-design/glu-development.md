# GLU / SwiGLU 组件开发指导

本页是所有 GLU-family Scenario 共用的组件级事实源。它定义双分支
`BlockMmad`、Scheduler/Kernel 组装、L0C2UB、GLU Epilogue、Host/Tiling、
参考资产和验证方法，不执行 Scenario 路由，也不是可独立选择的 Scenario。

完整需求必须先在 [Scenario 索引](../scenarios/index.md)唯一命中一个场景，再由该
叶子引用本页。Scenario 之间不得相互依赖；GLU-only 与 GLU+per-token quant
共享本页的组件合同，但分别拥有自己的完整路线、workspace、同步和验证合同。

场景设计必须冻结自己的事实源、证据和 custom 授权边界。GMM 相对 MM 的公共
差异统一遵循 [Grouped MatMul 项目侧差异](group-matmul-delta.md)，本页
不重复定义 group 编码、POD、group 基址和 Scheduler 生命周期。

## 1. GLU 组件合同

### 1.1 数学语义

将两路投影记为 `act` 和 `gate`，输出逻辑宽度记为 `Q`：

```text
act  = MatMul(A, B_act)
gate = MatMul(A, B_gate)
y    = activation(act) * gate
```

常见变体仅说明激活关系：

| 变体 | `activation(act) * gate` |
|---|---|
| GLU | `sigmoid(act) * gate` |
| Bilinear | `act * gate` |
| ReGLU | `ReLU(act) * gate` |
| GEGLU | `GELU(act) * gate` |
| SwiGLU | `Swish(act) * gate` |

`DESIGN.md` 必须从用户合同确认：

- 哪一路是 `act`，哪一路是 `gate`；
- `activation` 的精确公式和近似方式；
- 两路权重是独立 Tensor、按 N 拼接还是交错存储；
- MatMul 累加 dtype、激活计算 dtype、输出 dtype 和转换位置；
- bias 是否存在、两路 bias 的布局，以及逐操作顺序；
- `act/gate/y` 的逻辑 shape、物理布局和 stride。

只有两路各宽 `Q` 且拼接输入宽度明确为 `2Q` 时，才能使用 `Q=N/2`。
本页的 SwiGLU 参考实现固定为：

```text
Swish(x) = x / (1 + exp(-x))
y = Swish(act) * gate
```

它不包含可变 `beta`、dequant 或 scale。需要这些语义时，由选中的完整
Scenario 冻结额外阶段及其先后顺序，不能把额外公式隐式塞入本页的 Epilogue。

所有沿源 N 轴定义的输入或元数据都必须按 `act/gate` 成对解释：

| 输入或元数据 | 双分支合同 |
|---|---|
| `weight[K,2Q]` | `B_act[K,Q]` 与 `B_gate[K,Q]` |
| `bias[2Q]` | `bias_act[Q]` 与 `bias_gate[Q]` |
| MX `b_scale[...,2Q]` | `b_scale_act[...,Q]` 与 `b_scale_gate[...,Q]` |
| 其他 N 轴 metadata | 使用与 weight 相同的 `qOffset/curQ` 成对切分 |

M 轴或 K 轴输入不会仅因 GLU 自动复制。具体算子没有 bias/MX scale 时，不在
Params 中虚构字段；需要时必须把成对地址、layout、stride 和 Block 接口一起
实现并验证。

### 1.1.1 公式与参数语义冻结

一次实际合同审计中，旧 Golden 把 `mm` 留在 group 循环外、签名中的 scale 参数
没有进入公式，同时实现准备使用稳定的 SiLU 等价式。三者必须分别处理：

- Golden 必须逐 group 写回完整输出，并明确 `act/gate` 的分支顺序；
- 没有 scale、dequant 或 quant 阶段的接口，不得从参数名或历史签名推导额外
  计算；
- `F.silu(act)`、`act / (1 + exp(-act))` 或其他近似只有在用户合同允许、dtype
  顺序和非有限值行为已证明等价后才能互换。未获授权时保持 `conflict` 或
  `blocking`，不能用设备 PASS 反向授权公式改写。

该门禁属于 Scenario 设计阶段；具体 beta、近似阈值、scale 顺序和特殊值规则
仍由目标算子的 DESIGN/Golden 冻结，不写成 GLU 通用常量。

### 1.2 L0C2UB 组件边界

使用本组件的 Scenario 以 L0C2UB 连接双分支 BlockMmad 和 GLU Epilogue：

```text
Host/Tiling
    -> complete Params and static entry
Scheduler
    -> group/tile ownership
Kernel
    -> BlockMmad(A, B_act, B_gate)
    -> L0C2UB
    -> GLU Epilogue(act, gate)
    -> y
```

目标版本中与本实现相同物理合同的 Block L0C2UB、Kernel C/V 编排和
Epilogue 接口必须全部闭合。L0C2GM 只保留为 C-direct 诊断边界，不能替代
Scenario 已选择的 L0C2UB 正式接口。

各层只修改以下 GLU delta：

| 层 | 复用 MM/GMM 的内容 | GLU 特有修改 |
|---|---|---|
| Host/Tiling | 基础 tiler、平台资源、静态入口机制 | 校验 `N=2Q`，推导成对 N 轴资源，选择 GLU entry |
| Scheduler | MM/GMM problem 与 M/N tile 规则 | 保证一个逻辑 tile 同时覆盖相同 `qOffset/curQ` 的两路 |
| Kernel | group/tile 枚举、A view、同步框架 | 构造 `B_act/B_gate` 及所有成对 N 轴 view，组装完整类型链 |
| BlockMmad | A/B 搬运、MMAD、L0C 生命周期 | A 复用、双 B staging、成对累加、B32 NoQuant L0C2UB |
| Epilogue | Vector API、tail、GM 写回 | 对局部 `act/gate` 执行精确 GLU 公式 |

GMM 的公共 delta 不在表中重复展开，统一引用前述公共文档。

### 1.3 成立条件/门禁

- 两路投影、激活分支或 weight 布局未确认时保持语义未闭合。
- 基础 MM/GMM Assembly、Scheduler、Kernel、Params、layout 和接口能力由
  当前 Investigation 的 concrete witness 绑定；项目侧 Host TilingData 单独绑定。
- 参考资产只覆盖登记的物理合同；其他公式、layout、dtype 和附加输入必须标记
  `adaptation_required`。

### 1.4 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| 从本页没有 MM Scheduler 资产推导普通 MM 缺少 Scheduler | GLU 资产只登记组件 delta | 复用同版本 Investigation 选定的 MM Scheduler/Kernel |
| 把 GMM Scheduler/Kernel 改名后作为 MM 实现 | group 调度和普通 MM 的 problem 合同不同 | 查询 MM 上游组件；确认为缺口且无精确资产时阻塞 |
| 根据 `SwiGLU` 名称默认 act/gate 顺序或近似公式 | 名称不能固定用户 Golden 的分支和运算顺序 | 在 `DESIGN.md` 逐项冻结公式、dtype 和分支 |

## 2. 组件架构

### 2.1 双分支 BlockMmad

`BlockMmad` 负责两路矩阵乘，不负责激活公式：

```text
同一个 A tile
    -> B_act tile -> act accumulator
    -> B_gate tile -> gate accumulator
```

设计要求：

- 每个 K window 只搬运一次 A，并在两路 B 之间复用；
- `B_act/B_gate` 在 L1/L0B 中有独立且可证明的地址和布局；
- 两路累加器在同一个 L0C 物理 tile 中的排布、间距和 tail 明确；
- 只把 final C 交给 GLU；SplitK/StreamK partial 必须先完成归并；
- 输出 policy 显式区分 L0C2UB 和 L0C2GM，不由调用方猜测。

通用双分支参考资产是
`assets/blaze_custom/block/block_mmad_glu.h`。A8W8 与 FP16 共用同一模板：

| A/B | `CType_` / L0C | B 分支对齐 | Block K-window |
|---|---|---|---|
| `int8_t` | `int32_t` | `C0_ELEMENT<int8_t>` | `128` |
| `half` | `float` | `C0_ELEMENT<half>` | `64` |

类型相关内容必须由模板推导：

```text
L0CType = CType_
B branch alignment = C0_ELEMENT<BType>
Block K-window = 128 / sizeof(AType)
resource bytes = element count * sizeof(actual type)
```

#### MX 双分支追加合同

MM_MX 不能只给普通双分支 Block 增加两个 scale 参数。DESIGN 必须把逻辑
extent、物理 window 和资源 allocation 分开记录：

```text
logicalK      = 当前 GM Slice 实际参与数学计算的 K
physicalK     = selected MX Copy/MMAD contract 对 logicalK 的对齐结果
packedQ       = align(curQ, B branch alignment)
packedFullN   = 2 * packedQ
```

- A/B 的 GM Slice 和 scale 有效范围只消费 `logicalK`，K tail 之外的数据与
  scale 不得进入数学结果；
- L1 layout、offset、allocation、padding fill 和 L1→L0 MX Copy 必须使用
  同一个 `physicalK` owner，不能一部分使用逻辑尾长、另一部分使用对齐长度；
- packed B 的完整物理区域必须先确定性初始化为 MX value `+0`，再覆盖
  act/gate 两个有效 `curQ` view；
- packed ScaleB 的完整物理区域必须先初始化为数学值 `1.0`，再覆盖与
  B 完全相同 `qOffset/curQ` 的两路有效 scale view；
- E8M0 若在目标工具链中只有 pointer/data-plane 语义，先用绑定目标 CANN、
  编译器和 `ops-tensor/tensor_api` revision 的 encode/decode probe 确认
  `1.0` 的物理编码。不得把某次 probe 的 byte 写成跨版本 Skill 常量；
- 具体算子没有 bias 时，九参数模板中的 `BiasType/LayoutBias` 只能作为编译期
  偏特化占位，不得生成 runtime bias Tensor、地址、buffer、BT 或 MMAD 分支。

上述合同必须通过完整 MX Block 调用编译以及 K/Q tail、padding poison 和设备
精度验证；单 MM 的 MX 支持不能自动证明双 B/双 ScaleB GLU 已闭合。

### 2.2 Scheduler

Scheduler 只分配 problem 和 M/N tile，不解释 `act/gate` 业务语义。

| 基础场景 | Scheduler 来源 | GLU 侧要求 |
|---|---|---|
| 普通 MM | Blaze Investigation 选定的上游 Scheduler | 输出可映射到成对 N tile；不修改其基础调度公式 |
| BMM | Blaze Investigation 选定的上游 BMM/QBMM Scheduler | 保留 batch ownership 与 M/N tile 公式；同一 batch 内将完整 N tile 映射为成对 `qOffset/curQ` |
| GMM | 优先使用 Investigation 选定的上游 Grouped Scheduler | 按 group 重置 problem，并保持空组、offset 和逻辑 tile 合同 |
| GMM custom delta | `assets/blaze_custom/block/block_scheduler_group_matmul.h` | 仅在 Investigation 证明缺口且 DESIGN 选中时使用 |

GMM custom Scheduler 仍须完整遵循
[Grouped Scheduler 与 Kernel 合同](group-matmul-delta.md#3-scheduler-与-kernel)；
GLU 只额外要求同一逻辑 tile 的两路使用相同 `qOffset/curQ`。

### 2.3 Kernel

Kernel 是组件组装和生命周期的 owner。普通 MM Kernel 从 Investigation
选定的上游 MM Kernel/CV 组装出发；BMM 从 Investigation 选定的 BMM/QBMM
Kernel 出发并保留其 batch loop、batch decode、broadcast policy 和 GM batch
offset 公式，其中用户合同禁止 broadcast 时 Host 必须拒绝广播输入，不能删掉
Kernel 的 batch ownership 后用 Host 多次 launch 伪装 BMM；GMM 优先使用
Investigation 选定的 Grouped Kernel。只有 DESIGN 允许 custom delta 时，才以
`assets/blaze_custom/kernel/group_matmul_kernel_glu_fused.h` 作为 GMM GLU
控制流参考；若其后还有完整行 V2，则由公式无关的
`group_matmul_kernel_cv1_v2.h` 组合这个已有 C+V1 Kernel。通用组合层不拥有 GLU
group traversal、双分支 view 或 `Q=N/2`。

Concrete entry 先用当前项目生成的
`GET_TILING_DATA_WITH_STRUCT`/`GET_TILING_DATA_MEMBER` 将 GM tiling 解包为
本地 POD，再把其指针写入 Kernel Params。Kernel 负责：

1. 从 entry 传入的完整本地 TilingData 和 runtime Tensor 构造 Block、
   Scheduler 和 Epilogue Params；CV 同步常量由选中的 Kernel specialization
   直接读取，不扩展 `GemmUniversal` 模板参数；
2. 按 Scheduler 枚举 group 和 tile，构造 A、B_act、B_gate、成对 N 轴
   metadata、C 和 y view；
3. 在同一层持有有状态 BlockMmad 与 Epilogue，不依赖临时对象析构完成跨核协议；
4. 管理 C-ready、V-done、slot acquire/release、首轮、空任务和 final drain；
5. 保证 AIV 只消费 final C，并把真实 tile、sub、row pitch 和 tail 传给
   Epilogue；
6. 对 inactive AIV 也闭合协议要求的 wait/release，不能用 early return
   跳过同步。

对 BMM，Kernel 还必须：

- 使用 Base Kernel 已证明的 batch coordinate/stride 事实源，构造同一 batch
  的 A、B、ScaleA、ScaleB 和 y view；
- 保留 Base Scheduler 的 M/N tile schema；GLU 只把源 N tile 解释为同一
  `qOffset/curQ` 的 act/gate 配对，不另造 BMM Scheduler；
- 将逻辑输出行映射为 `rowBase = batchIdx * M + mOffset`（或目标 ABI 中等价的
  batch-local y base），再通过 `TileContext` 传给通用 SwiGLU Epilogue；
- MX BMM 时让 ScaleA 沿 batch/M/K 保持单份 view，只对沿 N 定义的 ScaleB
  构造 act/gate 两路 view。

这属于 `blaze_base_plus_custom_delta`：Base BMM Kernel/Scheduler 继续拥有
batch 调度，custom Kernel delta 只增加 paired view、L0C2UB、C/V 生命周期和
Epilogue adapter。不存在预制 BMM GLU Kernel 文件不等于缺少完整 Scenario。
只有 Base owner 或接口边界无法从目标版本证据确定，或必要 delta 无法精确
描述时，才使用 `extension_missing`。

#### Kernel 构造 `B_act/B_gate`

Scheduler 只提供 group 和 tile 坐标；Kernel 是从原始 weight 构造
`B_act/B_gate` GM Tensor view 的唯一 owner。BlockMmad 接收两个已经切好的
逻辑 view，不在 Block 内重新解释完整 weight；Epilogue 不读取 weight。

当前参考资产覆盖连续 ND/DN、前后半区拼接的逻辑
`weight[E,K,N]`，其中 `N=2Q`。Kernel 从 `BlockMmad::LayoutB` 派生
`FrameLayoutFormat<LayoutB, LayoutTraitDefault<BType>>`；因此逻辑 Slice 坐标
始终按 `[K,N]` 表达，物理连续次序由 ND/DN layout 决定。对专家 `groupIdx`：

```text
weightPtr = reinterpret_cast<BType *>(weightGmAddr)
expertElementOffset = groupIdx * K * N
expertBase = weightPtr + expertElementOffset
expertWeight = Tensor(expertBase, FrameLayoutFormat<LayoutB>(K,N))
```

上述偏移单位是 `BType` 元素。字节地址实现必须按 `sizeof(BType)` 转换，并在
接口处显式标注单位。

Scheduler 返回完整拼接 N 方向的当前 tile。Kernel 必须先绑定所选 Scheduler
的 `BlockCoord/BlockShape` schema，再计算地址；不能把其他 Scheduler 的尾块
offset 槽位带入当前实现。

当前 custom Grouped Scheduler 的合同是：

```text
BlockCoord = [mTile, nTile, 0, 0]
BlockShape = [curM, curFullN, K, reserved]
mOffset    = mTile * baseM
fullNOffset = nTile * baseN
curQ       = curFullN / 2
qOffset    = fullNOffset / 2
```

只有目标版本 Investigation 选中的 Scheduler 明确声明独立 tail offset
字段时，Kernel 才能按该字段修正地址。当前 `BlockShape[2]` 是 K，不是 M tail
offset；`BlockShape[3]` 是 reserved，不是 N tail offset。

Kernel 构造：

```text
B_act  = expertWeight[0:K, qOffset : qOffset + curQ]
B_gate = expertWeight[0:K, Q + qOffset : Q + qOffset + curQ]
```

并调用：

```text
blockMmad(A_tile, B_act, B_gate, accumulator, blockShape)
blockShape = [curM, 2 * curQ, K, ...]
```

构造前必须验证：

- `N % 2 == 0`、`fullNOffset % 2 == 0`、`curFullN % 2 == 0`；
- `curQ > 0` 且 `qOffset + curQ <= Q`；
- `B_act/B_gate` 使用相同 K 范围、`qOffset` 和 `curQ`；
- 专家基址、GM layout 和 row stride 与用户 weight 合同一致；
- act/gate 的顺序与 Golden 一致，不能因参考资产命名而交换。

普通 MMAD 路径允许 `LayoutA/LayoutB` 分别为连续 ND 或 DN；NZ/ZN 在当前
Grouped GLU Kernel 中用 `static_assert` 明确拒绝。开启 FP16 near-zero
refinement 时，AIV 会按原始连续非转置地址重算，因此 `LayoutA/LayoutB` 必须
同时为 ND；任一 DN 均在编译期拒绝。

Kernel 的 GM view 宽度始终是逻辑 `curQ`。`C0_ELEMENT<BType>` 对齐、
`packedHalfN` 和 L1/L0B staging 由 BlockMmad 处理。

其他 weight 合同必须显式适配：

| weight 合同 | Kernel 接口要求 | 当前参考资产状态 |
|---|---|---|
| 连续 ND/DN | 从 `BlockMmad::LayoutB` 构造官方 `FrameLayoutFormat`，保持逻辑 `[K,N]` Slice | 普通 MMAD 支持；refinement 仅支持 ND |
| NZ/ZN | 提供与真实分形物理地址一致的 view/pack adapter | `adaptation_required`，当前 Kernel 静态拒绝 |
| 两个独立 `B_act/B_gate` Tensor | Params 保存两个 GM 地址和各自 layout/stride | `adaptation_required` |
| act/gate 交错存储 | 提供经过证明的 view/pack adapter | `adaptation_required` |

这些形式必须实现对应 adapter，并完成完整调用编译、边界和设备精度验证。

bias 或 MX `b_scale` 的适配遵循同一 N 轴配对规则：Kernel 必须用 weight 的
`qOffset/curQ` 同时构造 act/gate 两路 view，并把两路都传给 Block。当前
`block_mmad_glu.h` 没有双 bias 或双 MX scale 接口，因此这些合同均为
`adaptation_required`，不能因为单 MM Block 支持一个 bias/scale 就宣称 GLU
已支持。

L0C2UB 融合 Kernel 必须显式拥有 AIC→AIV 和 AIV→AIC 生命周期。

### 2.4 GLU Epilogue

Epilogue 只负责一个已经分配好的 `act/gate` tile：

```text
load act/gate
    -> activation(act)
    -> multiply gate
    -> mask/tail
    -> write y
```

参考资产是
`assets/blaze_custom/epilogue/block_epilogue_swiglu.h`。设计要求：

- `Params` 只保存公式、tile 和本地资源所需字段；
- Kernel 通过 `TileContext` 或等价接口传入地址、逻辑范围、row pitch 和 tail；
- Epilogue 的输入是 Kernel 已分配的局部 tile，group 分配和跨核同步仍由
  Kernel 持有；
- `act/gate` 使用同一逻辑坐标和 tail mask；
- 类名表达实际职责：参考实现使用 `BlockEpilogueSwiGlu`，且不承担
  dequant、scale 或 quant。

近零相对误差、累加顺序与静态精度策略统一读取
[GLU 精度诊断](glu-precision-diagnosis.md)，不在组件主线中展开。

GLU 通常包含激活和乘法，是多步依赖链。默认将 RegBase 列为主选；只有目标
API、资源或设备数据证明 RegBase 不成立或更差时才选择 MemBase。两条候选都
必须核对中间 dtype、近似、mask/tail、非有限值和 GM 写回。

RegBase 循环中 `UpdateMask<T>(remaining)` 若按目标版本签名通过引用推进
`remaining`，它就是唯一的循环计数 owner；应以可变局部对象接收需要
`MaskReg&` 的后续 API（不要写成 `const auto mask`），调用后不得再次手工递减。
设备矩阵
至少覆盖 `Q=VL-1`、`VL`、`VL+1` 和 `2*VL+1`，防止后续 VF slice 被跳过。

### 2.5 Host 与 Tiling

Host/Tiling 负责构造完整 ABI 和选择已设计的静态入口，不负责在运行时改变
正式架构。

- 普通 MM 与 GMM 的项目侧 Host Tiling 优先绑定 Investigation 中实际存在的
  关联证据；否则绑定 `DESIGN.md` 选中的目标工程或本 Skill 资产。Blaze
  library 本身不提供 TilingData。
- 当前普通 GMM 登记 `GroupMatmulTilingData` 共享 POD 和
  `MatmulTilingSwat` 的 grouped `GetTilingData` 重载；普通 MM 接口不感知
  grouped 元数据。目标工程调用该重载一次构造完整 POD，调用方不得在函数外
  补写 grouped 字段。Quant/MX GMM 不得从该 POD 推导自身 ABI。
- Concrete `__global__ __aicore__` entry 使用项目生成的 typed tiling
  loader 一次解包 POD，并在其局部对象生命周期内调用 Kernel。Kernel
  Params 只持有该本地 POD 指针；公共资产不重复维护字段列表，不从
  `GM_ADDR` 逐字段标量读取 tiling。
- 建立 `TilingData field -> consumer` 映射。字段存在于 POD 不表示
  Scheduler、Block 和 Kernel 都消费它。
- Custom Block 不读取 `tiling.baseK` 时，不要求它等于 Block 内部 K-window；
  资源 gate 按实际 tile、dtype 字节数、buffer 数和平台容量计算。
- Host 只实例化 DESIGN 已绑定的 L0C2UB GLU entry，不提供 L0C2GM
  workspace fallback。

MX+GLU 必须把基础 MX tiler 的 `baseN` 当作候选，而不是最终可执行结论。
融合后的资源 gate 按 `packedFullN=2*align(baseQ, B alignment)` 重新计算
L1 B/ScaleB、L0B/ScaleB、L0C 和 UB；候选超限时缩小 Q tile 并让正式
Scheduler 多 tile 遍历。不得把某个算子实测得到的固定 `baseQ` 上限推广为
Skill 常量，修改 tile 后还要覆盖 slot reuse 和 final drain。

逻辑 `Q` 不满足 Cube 的最小宽度或对齐时，必须显式区分 `logicalQ` 与
`physicalQ=align(logicalQ, B alignment)`。act/gate 两半分别 padding，物理
顺序为 `[act logical, act padding, gate logical, gate padding]`；Epilogue 的
gate offset 使用 `physicalQ`，mask 和 GM 写回长度仍使用 `logicalQ`。实现可由
BlockMmad 内部 pack，或由冻结 ABI 中的显式 adapter/prepack 完成；后一种必须
保持外部 weight/ScaleB shape 不变、补零 value、同步搬移两半的 ScaleB，并将
逻辑/物理 buffer 字节和消费者写进 crosswalk。不得把 gate 留在 `logicalQ`
偏移后再把整个拼接宽度一次 padding，否则两半语义会错位。

完整字段映射必须落入 `DESIGN.md`，至少包含：

| 字段/资源 | producer | 单位与推导 | consumer | gate |
|---|---|---|---|---|
| `lda` | Host Tensor contract | A 元素；由真实 stride 推导 | Kernel A layout/group base | 覆盖最大 A offset |
| `splitMRows` | Host + Fixpipe contract | 行；`ceil(baseM/2)` 或实际 split 规则 | Kernel/Epilogue | 两个 AIV 的 localRows 均可表达 |
| `cLocalPitch` | Block layout helper | FP32 元素；`packedFullN` | Block/Epilogue | UB row pitch 与 Copy 完全一致 |
| `ubViewBytes` | paired UB resource helper | bytes；dequant Epilogue 的 accumulator 与三段 scale staging 总空间 | Epilogue `Init` | 256B 对齐且不超过实际可用 UB |
| `stageRows` | Host resource helper | 行；由剩余 UB 和 `cLocalPitch` 推导 | Epilogue | 正数且不靠写死 `1` 逃避预算 |
| CV 同步常量 | `cv_sync_constants.h` | flag/slot 编译期常量 | Kernel specialization | ratio、flag 范围、首轮和 final drain 闭合 |

未被实际消费者读取的字段不得保留为“完整性”占位。字段存在、单位正确、资源
gate 通过和消费者实际读取必须分别验证。

`BlockEpilogueDequantSwiGlu` 不新增 ABI 字段。Host 按 256B 对齐顺序推导：

```text
accumulatorBytes = AlignUp(splitMRows * cLocalPitch * sizeof(int32_t), 256)
weightScaleBytes = AlignUp((baseN / 2) * sizeof(float), 256)
xScaleBytes      = AlignUp(splitMRows * sizeof(float), 256)

weightScaleActOffsetBytes  = accumulatorBytes
weightScaleGateOffsetBytes = weightScaleActOffsetBytes + weightScaleBytes
xScaleOffsetBytes          = weightScaleGateOffsetBytes + weightScaleBytes
ubViewBytes                = xScaleOffsetBytes + xScaleBytes
```

Host 必须拒绝未对齐、越界或顺序重叠的结果。Device 再按当前 tile 的
`localRows/packedFullN/curH` 计算 accumulator、act scale、gate scale 和 x scale
四个实际半开区间，确认全部在 `ubViewBytes` 内、256B 起点对齐且两两不重叠后才
加载 scale。同一 dequant 阶段四段数据同时存活，不能互相复用；完成 FP workspace
写回、最后一次 MTE3 drain 和 Scenario 冻结的阶段同步后，per-token quant 可以
复用整块物理 UB。跨阶段预算因此取 `max(dequantUbBytes, quantUbBytes)`，不是求和。

纯 `BlockEpilogueSwiGlu` 没有三段 scale staging，`ubViewBytes` 仍表示它可用的
总 UB view，通常只需覆盖 accumulator 区。

### 2.6 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| 为 FP16 复制一份 Block | 双分支结构相同，复制会让布局和资源规则漂移 | 实例化同一模板，并按实际 dtype 重新验证 |
| 只删 dtype 断言，仍保留 INT32 L0C、固定 32-element 对齐或 K-window 128 | 类型相关参数没有随模板推导 | 从 `CType_`、`C0_ELEMENT<BType>` 和 `sizeof(AType)` 推导 |
| 专家基址同时按元素和字节计算 | offset 单位混用导致专家权重错位 | 在接口处固定单位，字节换算只执行一次 |
| 未经 ABI adapter 就把本地 `packedHalfN` 带入外部 GM Slice 宽度或 stride | 本地搬运布局被误当作逻辑 weight layout | 外部 GM view 保持逻辑 `curQ`；padding 由 BlockMmad 或已冻结且有 crosswalk 的内部 adapter 持有 |
| 整体 padding 拼接后的 `N`，仍用 `logicalQ` 定位 gate | act padding 被当成 gate，双分支错位 | 两半分别 padding；gate offset=`physicalQ`，Vector mask/writeback=`logicalQ` |
| 把连续前后半区 Slice 直接套到独立、交错或 NZ/ZN weight | 这些合同的基址和 stride 不同 | 实现显式 adapter，并重新验证 |
| 让 Epilogue 读取全局 Scheduler 或 weight | 组件 ownership 被打破 | Kernel 构造 view，Epilogue 只消费局部 act/gate tile |
| 只把 weight 分成两路，bias 或 MX `b_scale` 仍按完整 N 传一次 | 所有 N 轴数据都与 act/gate 列一一对应 | 用相同 `qOffset/curQ` 构造两路 view |
| 把 A 的 per-token metadata 也无条件复制 | 它沿 M/K 轴定义，不属于 N 轴配对 | 按原数学 broadcast 合同传递 |
| 把 `BlockShape[2]/[3]` 固定解释为 M/N tail offset | 当前 custom Scheduler 返回的是 `[curM,curN,K,reserved]`，错误解释会把 K 加进 A/y 行地址 | 按所选 Scheduler 的逐字段 schema 建立 consumer 映射；当前 offset 只由 tile coordinate 和 base shape 推导 |

## 3. 实现流程

### 3.1 Capability preflight

编码前按接口实际消费的合同分别验证。不要把 A/B 输入 dtype 的差异带入
L0C2UB 能力判断：

| 能力 | 必须完成的最小穿刺 |
|---|---|
| Block MMAD | 按 A/B dtype、layout 和累加 dtype 实际构造 Tensor 并调用 Block 主计算 |
| L0C2GM | 选择真实 GM Tensor/policy/trait/location，设备对比 C golden |
| B32 L0C2UB | 按 B32、NoQuant、L0C/UB layout 和 stride 实际调用 Copy |
| B32 dual SplitM | 覆盖 `dualDstCtl`、`subBlockId`、odd/even M、tail 和空 sub |
| Kernel C/V | 实际运行 wait/set、slot reuse、空任务和 final drain |

证据状态及判定由当前场景 DESIGN 的 capability 与证据边界记录。
当前两种已登记 MMAD 实例分别为 `int8 x int8 -> int32` 和
`half x half -> float`，其 MMAD 必须分别证明；到 L0C2UB 层时，两者都是
B32、NoQuant 的同一物理搬运合同。A8W8 路径已有的 B32 SplitM 证据可作为
FP16 路径的搬运能力依据，不能再以 A/B dtype 不同把 FP16 L0C2UB 标为
`unsupported` 或 `unverified`。

这份复用不等于完整 FP16 Kernel 已交付。FP16 C-direct 证明 L0C 数值正确后，
若 C-through 或 Full 失败，责任域已经收窄到 L0C/UB layout、SplitM 映射、
有效行、C/V pipe、UB 地址和生命周期；继续重做 MMAD dtype capability 会
绕远。

### 3.2 实现 L0C2UB 融合路线

1. 从 Investigation 绑定的 Block 最终结果点确认 final/partial 语义。
2. 记录 L0C 与 UB 的物理 layout、B32 element、Copy extent、`srcStride`、
   `dstStride`、row pitch、tail 和容量；不得从逻辑 ND shape 猜物理 pitch。
3. 固定 `dualDstCtl=DUAL_DST_SPLIT_M`，证明 `subBlockId` 与两个 AIV 的映射。
   `mSize` 必须偶数对齐，并分别计算两个 AIV 的逻辑起始行和有效
   `localRows`；`localRows=0` 只能跳过 Vector 计算，不能跳过同步。
4. Fixpipe 完成后从实际完成 pipe 发出 C-ready；两个 AIV 按各自物理映射
   wait，再在读写完成后从合同指定的 pipe 发出 V-done。Kernel 负责 slot
   首轮、复用和 final drain。
5. Epilogue 从 Fixpipe 实际写入的 UB 基址和 pitch 读取。Block 构造的目的
   UB 地址、AIV `LocalTensor` 地址和读取时间点必须是同一事实，不得各自写死
   后假定一致。
6. 对 ping-pong 或循环复用 UB，按
   [Blaze 同步模式](../fundamentals/blaze-sync-patterns.md) 建立正向和反向
   依赖。MTE3 完成读取前，同一 UB 不得被下一轮 MTE2 覆盖；需要时使用
   `MTE3_MTE2`。

实现中不得加入 L0C2GM fallback。能力不成立时返回 Step 3 更新设计状态。

当前参考 Block 通过 `AscendC::Te::MakeCopy(CopyL0C2UB{},
CopyL0C2UBTraitMixSplitM{})` 和 `AscendC::Te::Copy` 表达 B32、NoQuant、
row-major UB 与 `DUAL_DST_SPLIT_M`。不得丢弃调用方传入的 Te Tensor 后从固定
UB/L0C offset 重新构造 `LocalTensor` 并直接调用 `Fixpipe`；否则 Tensor 的地址、
layout 与 slice 合同会被绕过。资产必须让 Block 目的 UB 地址与 Epilogue 源地址
来自同一合同，并用 C-through 验证真实物理布局和同步；不能仅凭 trait 名称宣称
完整集成通过。

### 3.3 组装顺序

按以下顺序实现并在每一步建立可验证结果：

1. BlockMmad 完整调用编译和 C-direct；
2. Scheduler 的 group/tile/空组映射；
3. Kernel Params、逻辑索引和静态入口；
4. 正式 C→V 交接路径；
5. Epilogue 公式、dtype、mask/tail 和写回；
6. Host/Tiling ABI、资源 gate、Wrapper 和 Launcher；
7. 分层验证、Full 回归和性能采集。

### 3.4 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| 因 A/B 从 INT8 变为 FP16 就把 B32 L0C2UB 重置为 unverified | A/B dtype 属于 MMAD 合同，B32 NoQuant 搬运不消费它 | 复用 B32 搬运能力，重新闭合 MMAD、消费者语义和完整集成 |
| 类型 alias 或类声明通过就判定 MMAD 或 Full 支持 | 没有实例化真实 MMAD、layout、Kernel 和消费者 | 分层完成调用编译和设备 probe |
| L0C2GM 正确后直接宣称 Full supported | C-direct 没有覆盖布局、SplitM 和 C/V 生命周期 | 直接检查五项交接边界并验证正式 C-through |
| Block、Kernel 和 Epilogue 尚未分层闭合就反复运行 Full | 最早失败边界不可归因 | 按组装顺序建立逐层可验证结果 |

## 4. 参考资产与适配边界

### 4.1 基础组件依赖

基础 Assembly、MM/GMM Scheduler/Kernel、Params、layout、transpose 及目标接口
均由当前 Investigation 和场景 DESIGN 的事实源边界绑定，项目侧 Host
TilingData 单独绑定。本章只登记 GLU custom delta。

### 4.2 GLU custom 参考资产

| GLU-specific 责任 | 参考资产 | 使用前必须重新确认 |
|---|---|---|
| Dispatch policy | `assets/blaze_custom/policy/dispatch_policy.h` | 模板类型链和版本接口 |
| 双分支 BlockMmad | `assets/blaze_custom/block/block_mmad_glu.h` | dtype、layout、Copy 和资源 |
| GMM Scheduler 缺口参考 | `assets/blaze_custom/block/block_scheduler_group_matmul.h` | Investigation 缺口、group 语义和逻辑索引 |
| GMM GLU C/V Kernel specialization | `assets/blaze_custom/kernel/group_matmul_kernel_glu_fused.h` | group traversal、Params、ratio、pipe、slot、final drain，并排除 V1/V2 tuple |
| SwiGLU Epilogue | `assets/blaze_custom/epilogue/block_epilogue_swiglu.h` | 公式、API、dtype 和 tail |
| CV 同步常量 | `assets/blaze_custom/epilogue/cv_sync_constants.h` | MODE、flag、slot、ratio 和目标 Kernel |
| GMM TilingData 扩展参考 | `assets/op_tiling/matmul/blaze_group_matmul_tiling_data.h` | Investigation ABI、完整构造和 consumer |

跨公式共用的 `C+V1 + V2` 组合资产为
`assets/blaze_custom/kernel/group_matmul_kernel_cv1_v2.h`；它不是 GLU-specific
资产，只负责复用已有 C+V1、执行全核交接、按完整逻辑行重分配 AIV 并调用 V2。

目标工程只复制 DESIGN 选择的 custom delta 和必要直接依赖。其他 GLU 变体
标记 `adaptation_required`，并实现对应 Epilogue 与 Golden。

资产覆盖边界必须按责任解释：

- `block_epilogue_swiglu.h` 是局部 FP32 act/gate tile 的公式资产，不拥有
  MM、BMM 或 GMM 调度；Kernel 提供正确 row base、pitch 和 tail 后可复用。
- `block_mmad_glu.h` 当前闭合 A8W8/FP16 双分支结构，不含 MX ScaleA/双
  ScaleB 接口；MX 双分支 Block 是目标工程的 `adaptation_required`。
- `group_matmul_kernel_glu_fused.h` 与 `block_scheduler_group_matmul.h` 含 GLU
  group traversal/group list 合同，只能作为 GMM GLU delta。BMM 必须扩展
  Investigation 绑定的 BMM Kernel/Scheduler。
- `group_matmul_kernel_cv1_v2.h` 只参照 ops-transformer INT8 输入 GMM SwiGLU
  per-token quant 的 `C+V1 -> SyncAll -> V2(realM)` 阶段骨架，用
  `AscendC::Std::tuple<V1,V2>` 占用既有 `BlockEpilogue_` 参数。GLU 的
  group traversal/MMAD/V1 仍只属于 `group_matmul_kernel_glu_fused.h`。其中 `+` 是
  显式 `{PIPE_ALL,PIPE_ALL}` 的文件级 `SyncAllConfig` 全核硬同步，不是本核
  `PipeBarrier<PIPE_ALL>`。
- A8W8 MIX Kernel 只能作为相同 B32 L0C2UB/CV 生命周期的物理合同参考，不能
  证明 MX MMAD、MX scale view 或完整 MX BMM+GLU。

完整类型链必须在一个可编译入口中同时实例化，不能只验证若干 alias：

```text
ProblemShape
  -> MatmulDualBranchGlu<..., RefineNearZeroFp16>
  -> BlockScheduler<ProblemShape>
  -> BlockMmad<MatmulDualBranchGlu<...>, AType, ..., BTypeTuple, ..., CType, ...>
  -> BlockEpilogueSwiGlu
  -> GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>
     selected by BlockMmad::DispatchPolicy::ScheduleType
  -> Kernel::Params
  -> static __mix__ entry
```

若 GLU 后还有完整行 V2，类型链只把 Epilogue 一项改为
`AscendC::Std::tuple<BlockEpilogueGlu, BlockEpilogueV2>`；不得增加 Kernel 模板
参数或新建带完整组合公式名的 Kernel 类。Host selector 为 single-pass 和
two-pass 分别实例化该 tuple，目标工程的静态 entry/Launcher 负责启动选择结果。
GLU C+V specialization 必须排除 `IsCv1V2EpiloguePipeline<BlockEpilogue>`，使
tuple 只由通用组合 specialization 匹配。

`cv_sync_constants.h` 保留所选流水线的参考同步常量；具体
`GemmUniversal` specialization 直接读取这些常量，并私有持有双 AIV flag
range offset。不得把同步常量包装成额外 Policy 后追加到 Kernel 模板参数。

普通 MM 的 Scheduler/Kernel 来自 Investigation；GMM 缺口才适配本页登记的
Grouped 参考资产。两条路线共享 GLU Block/Epilogue delta，不共享错误的
group 控制流。

### 4.3 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| 为了让资产表完整而复制 MM Scheduler、Kernel 或 Tiling | 基础组件已有同版本 Investigation 事实源 | 只登记 GLU-specific delta |
| 用 GMM 参考资产覆盖 MM 的上游组件 | 参考资产的基础场景和接口不匹配 | 保留 Investigation 绑定的唯一 Base Assembly owner |
| 因没有 BMM GLU 专用资产而直接判定 `extension_missing` | Scenario 已覆盖 BMM，且资产按组件责任复用；缺的是项目侧适配而非新场景 | 绑定 BMM Base owner，列出 MX dual Block、paired view、C/V 与 row adapter 的 `adaptation_required` |
| 共用双分支 Block 就宣称所有 GLU 公式已支持 | Block 只完成两路投影，不实现具体激活 | 为变体实现 Epilogue、Golden 和设备验证 |
| 把 GLU traversal、双分支 view 或 `Q=N/2` 放进通用 C+V1+V2 Kernel | 通用阶段组合被 GLU 路线锁死 | GLU 控制流留在 `group_matmul_kernel_glu_fused.h`，通用层只组合已有 C+V1 和 V2 |

## 5. 验证与交付

### 5.1 分层诊断

同一组输入、shape 和 Tiling 使用以下模式：

| 模式 | 证明边界 |
|---|---|
| `C-direct-GM` | C 与真实 L0C2GM；不证明 L0C2UB 和 V |
| `C-through-L0C2UB` | L0C2UB、C-ready、slot 和 identity writeback |
| `V-zero-C` | 零 C 下的公式、broadcast、mask/tail 和写回 |
| `V-known-C` | 已知非零 C 下的 Vector 公式和布局 |
| `Full` | 正式 producer、V、offset、同步和 slot reuse |

正式 L0C2UB 路线必须执行 `C-through-L0C2UB`；GM workspace identity 不属于
使用本组件的 Scenario 验收模式。

C-direct 的持久化输出必须是逻辑全局 `C[M,N]`：act 写
`[0,Q)`，gate 写 `[Q,N)`。不得把单个物理 tile 的
`[act_tile,padding,gate_tile,padding]` 直接交给按全局前后半区解释的
Verifier；若保留物理 dump，必须另存并显式记录 tile、pitch 和 padding。

每个 staged mode 的 `SplitM`、`mSize/nSize`、`subBlockId`、`dualDstCtl`、
`cLocalPitch`、stage rows 以及资源 gate 输入，必须从同一份已选 Tiling/POD 和
当前 tile 计算得到；不得用 `1` 或其他便于编译的常量替代真实值。诊断启动前
先用至少一个非空 group、非零 canary 和已知正确的对齐 case 做 positive control，
并检查实际写入元素数。若输出全零、未创建文件或覆盖数为零，首先判为
“诊断入口/控制数据/资源 gate 未执行”的未闭合证据，不能据此归因 BlockMmad、
L0C2UB 或 Vector 公式。

C-through 的 identity writeback 仍是一个真实 UB→GM consumer。它必须记录
逻辑 block 长度、UB 物理 row pitch、GM 逻辑 row pitch、两个 stride 的单位
及其表示 pitch 还是 block 间隔，并在 UB 复用前等待 MTE3 完成。不能因为它
不做激活计算就跳过 Copy 和生命周期证据；stride 合同未证明时，使用逐行
`blockCount=1` 写回作为诊断基线。

诊断 adapter 也必须闭合真实生命周期：GM→UB 输入先完成 MTE2→首个消费者，
Vector 结果完成 V→MTE3，下一 tile 复用 UB 前等待 MTE3 完成。V-zero 的
Golden 为零时，输出必须先填非零 canary，再证明所有逻辑元素被覆盖；预清零
输出得到全零不能证明 Kernel 执行。

最早失败模式决定责任域：

- C-direct 失败：检查 A/B 搬运、layout、MMAD、归并、Tiling 和 GM Copy；
- C-direct 仅在接近零的严格相对误差点失败：转入
  [GLU 精度诊断](glu-precision-diagnosis.md)，不得在组件主线中
  直接启用静态修正；
- C-direct 通过而对应 C-through 失败：不再按 A/B dtype 重查 L0C2UB
  capability；依次检查 L0C/UB 物理 layout 与 stride、`dualDstCtl/subBlockId`
  和双 AIV 映射、偶数 `mSize` 与各 AIV 有效行、C-ready/V-done pipe、AIV
  UB 地址/pitch/读取时点；
- C-through 通过而 V-zero/V-known 失败：检查 Epilogue；
- 隔离模式通过而 Full 失败：检查真实 offset、RAW、slot reuse 和生命周期。

Dump 只放在目标环境可读的 GM/UB 边界，并记录 tile、slot、sub、stage 和
同位置 golden。定位后删除 debug entry、known-C、诊断 Params、Dump，以及
生成器、manifest、Verifier 和输出 schema 中的诊断字段，并静态检索残留。
后续输入、Golden、构建产物和 Full 回归由测试流程负责；本 Skill
只要求删除诊断字段并登记待验证项。

### 5.2 精度和边界矩阵

精度判据与特殊值分类按当前场景 development guide 编译进 PLAN 的交付规则
执行；本页只增加 GLU 场景矩阵。

至少覆盖：

- `act/gate` 不对称 pattern，检测分支交换和错位；
- 合同允许的最小正偶数 `N`，以及 `Q` 对 Cube 最小宽度/对齐的前一值、等值、
  后一值；设备异常不能通过悄悄缩窄逻辑支持域规避；
- base、M/N/K tail、odd/even M、multi-tile 和 slot reuse；
- Grouped 的空 expert、prefix offset 和 group tail；
- `localRows=0`、alignment 前一值/等值/后一值；
- 激活零点附近、大正负值、tiny 和合同要求的 NaN/Inf；
- 若精确诊断后启用静态精度策略，按
  [GLU 精度诊断](glu-precision-diagnosis.md)执行专项矩阵；
- 有限值精度、特殊值分类和重复执行；
- 正式路线对应的 C-direct、C-through、V-zero、V-known 和 Full。

MX 路线还必须满足：

- Golden 从实际落盘并写给设备的 MX value 与 E8M0 scale bytes 解码，不从
  量化前 FP 输入旁路计算；
- manifest 分别记录 scale 数学坐标、Pattern/Copy 坐标、raw file order、
  Golden-only transform、shape、seed、逻辑/物理 dtype、每个文件 byte 数和
  SHA256，并保存 `C/act/gate/y` 分层 Golden；具体 ScaleB 三层合同按
  [Tensor API 参考](../fundamentals/tensor-api-reference.md)由当前
  specialization 推导；
- Golden 为建立 `[scaleK,N]` 数学视图而使用的 `permute/reshape` 不得直接
  物化为设备 scale 输入；先用 raw-byte 地址公式证明变换是否属于设备 ABI；
- 对 K data tail、unused scale slot 和 packed B/ScaleB padding 使用独立
  poison 证明，poison 不进入正式数学结果；
- 覆盖 MX K-group 和物理 window 对齐的前一值、等值、后一值，以及
  `Q=alignment-1/alignment/alignment+1`；
- 精度指标和特殊值门禁只引用项目锁定的 precision 事实源，本组件指导不
  自建或放宽阈值。

### 5.3 完成门禁

只有以下条件全部满足才能交付：

- `DESIGN.md` 已冻结公式、正式架构和唯一 Base Assembly owner；
- BlockMmad、Scheduler、Kernel、Epilogue、Host/Tiling 的责任和接口均有
  源码或完整调用证据；
- 正式输出路径在目标版本完成设备验证；
- 未用 L0C2GM 代替合同要求的 L0C2UB；
- 分层诊断、边界矩阵和清理后 Full 回归通过；
- 性能结论来自同输入、同 shape、同 Tiling、同公式和同精度标准的设备数据；
  没有 matched baseline 时只报告测量值。

### 5.4 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| 用 `C-through-fusion` 作为模糊模式名 | 无法判断实际验证的是 L0C2UB 还是其他交接 | 使用明确的 `C-through-L0C2UB` |
| C-direct 通过就跳过正式 C-through | C-direct 不覆盖 C/V 交接和同步 | 继续执行 C-through、V 隔离和 Full |
| C-through 批量写回直接把 row pitch 填进 stride | Copy 接口可能要求 block 间隔且 src/dst 单位不同，首行正确不能证明后续行 | 先证明字段语义；未闭合时用逐行 `blockCount=1` 基线，并等待 MTE3 完成 |
| V-zero 从已清零输出得到 PASS | 即使诊断 Kernel 未执行也会得到相同结果 | 先写非零 canary，并检查全部逻辑元素被覆盖 |
| C-direct 把局部 packed tile 当全局 C | padding 和局部分支顺序会造成伪错位 | 分别写入全局 act/gate 半区，物理 dump 使用独立 schema |
| 定位完成后保留 known-C、Dump 或诊断 Params | 诊断代码改变正式入口和证据对象 | 清理后 clean build 并重跑 Full |
