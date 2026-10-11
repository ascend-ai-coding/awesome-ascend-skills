# Grouped MatMul 项目侧 Host/Tiling/ABI 与执行差异

> Blaze library 不定义目标算子的 Host TilingData。本文定义 GMM 相对普通
> MM 的项目侧 Host/Tiling/ABI、执行和验证 delta。Blaze Assembly、layout、
> Scheduler、Kernel、Params 和接口能力仍由同版本 Blaze Investigation
> 绑定；精确 POD 只以目标工程或本 Skill 登记的资产定义为事实源，融合
> Scenario 只描述自身额外差异。

## 1. 数学与运行时合同

### 设计主线

GMM 把 M 轴划分为 `E` 个半开区间 `[start_i,end_i)`。每个 group 使用同一
个 A 逻辑矩阵中的不同行，并选择自己的 B：

```text
C[start_i:end_i, :] =
    A[start_i:end_i, :] @ B[i, :, :]
```

当前普通 GMM 参考资产支持两种编码：

| `groupListType` | 编码 | `start_i` | `end_i` | `totalM` |
|---|---|---|---|---|
| `0` | cumsum | `i == 0 ? 0 : groupList[i-1]` | `groupList[i]` | `groupList[E-1]` |
| `1` | count | `sum(groupList[:i])` | `start_i + groupList[i]` | `sum(groupList)` |

cumsum 中相邻端点相等、count 中元素为 0，都表示合法空 group。
这些枚举值和空组语义是当前参考资产的 ABI，不是 Blaze 通用事实。目标算子
必须先以原型、接口文档和 Golden 确认是否支持两种编码、固定其中一种，或采用
其他编码；不匹配时必须适配 Host、POD、Kernel 和测试，不能只复用表中数值。

### 接口与数据流

Host 必须在启动前验证：

- `E > 0`，`groupList` 长度为 `E`；
- cumsum 非递减、首值非负、末值等于 `M`；
- count 每项非负、累加和等于 `M`；
- 地址、shape、dtype、layout 和 group 编码与 Kernel ABI 一致。

Device 在 Host 已验证的合同下遍历 group，空 group 只推进 prefix，不创建
MM tile。若算子 ABI 面向不可信调用方，需要的额外防御必须作为该算子的显式
合同设计，不能隐式改变公共 GMM 路线。

若目标合同明确允许 `M=0` 的 no-op（并可能同时 `E=0` 或全零 cumsum），上面的
`E>0` 正常执行前置条件只适用于正维度 grouped execution。Host 必须在解析/解引用
group buffer 前短路该 no-op，生成合同规定的空输出并且不 launch；verifier 也要为
空数组定义比较 identity。不能把“空 group 合法”和“任意 E=0 都可执行”混为一谈。

Golden 必须先物化完整的逻辑 `mm[M,N]`：对每个非空 group 写入对应的
`mm[start_i:end_i]` slice，所有 group 完成后再执行一次完整矩阵的后处理或
per-token quant。不能在 group 循环中复用一个局部 `mm` 并在循环外量化，否则
只会保留最后一个非空 group，或让其他行保持未定义；空 group 和不均匀 group
必须由该完整矩阵语义覆盖。

对普通 FP16 GMM + GLU，Golden 的最小可执行主线应保持逐组写回和逐组后处理：

```python
y = torch.empty((M, N // 2), dtype=torch.float32)
for g, end in enumerate(ends):
    start = 0 if g == 0 else ends[g - 1]
    if start == end:
        continue
    c = torch.matmul(x[start:end].float(), weight[g].float())
    act, gate = c[..., :N // 2], c[..., N // 2:]
    y[start:end] = torch.nn.functional.silu(act) * gate
return y
```

`mm`、`act` 和 `gate` 不能只在循环内覆盖最后一个 group；`weight_scale`、
`x_scale` 等未参与冻结公式的参数也不能因为出现在旧函数签名中就被隐式
引入。Golden 后端、累加 dtype、非有限值分类和任何等价公式改写必须与
DESIGN/PLAN 的语义合同一致。

### 成立条件/门禁

- 目标 ABI 的 `groupListType` 是运行时 selector 时，两种编码必须分别验证。
- 固定 cumsum 或固定 count 的算子可以不暴露 selector，但文档和入口名称必须
  明确固定编码。
- `totalM` 的推导必须先于 MM tiling，不能把 cumsum 数组再次求和。

### 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| 对所有 `groupList` 使用 `sum` 得到 `totalM` | cumsum 保存的是端点而非组大小 | cumsum 取末端点，count 才求和 |
| 把相邻相等端点判为非法 | 空 group 是合法运行时状态 | 跳过该 group，并保持后续 prefix 正确 |
| Device 静默修复未排序的 cumsum | 会掩盖 Host 合同错误并改变数学语义 | Host 拒绝非法输入 |
| Golden 在 group 循环外只量化最后一个 `mm` | 没有把分组结果写回完整逻辑矩阵，空组和前序 group 的行被丢失或未定义 | 先分配完整 `mm[M,N]`，逐 group 写 slice，全部 group 完成后再统一后处理 |

## 2. Host、Tiling 与 ABI

### 设计主线

Blaze library 不拥有 TilingData。目标工程必须为 GMM 定义一个 Host/Device
共享 POD，普通 MM POD 也不能因为增加 group 字段而被隐式改写。

当前普通 GMM 参考路线的精确 ABI 唯一由
`assets/op_tiling/matmul/blaze_group_matmul_tiling_data.h` 定义；字段类型、
packing、alignment、`sizeof` 和 `offsetof` 均以该资产为准，本文不复制
`struct`。

Quant/MX GMM 不能从普通 `GroupMatmulTilingData` 推导。若目标路线需要
`GroupQuantMatmulTilingData` 或其他 grouped POD，必须绑定目标工程或本 Skill
已登记的独立定义、Host builder 和 Kernel loader；当前没有可绑定定义时记录
`extension_missing`，不能临时复用普通 GMM POD。

本 Skill 登记共享 POD 以及 `MatmulTilingSwat` 的 grouped `GetTilingData`
重载；普通 MM 的 `GetTilingData(..., MatmulTilingData&)` 接口保持不变。
目标工程必须通过该 grouped overload 作为完整 POD 的唯一 producer，调用方
在进入重载前完成 group 编码校验并推导 `totalM`，且不得在函数外补写
grouped 字段。重载内部：

1. 接收调用方已验证的 `totalM,N,K`；
2. 调用普通 MM 的 tiler 生成 `.matmul`；
3. 写入 `groupListAddr/groupNum/groupListType`；
4. 执行 ABI、资源和字段消费 gate；
5. 返回完整且可直接序列化的 POD。

### 接口与数据流

当前普通 GMM 参考 tiling 复用 MM tiler，group 元数据不参与其 SWAT profile
搜索：

```text
{totalM, N, K}
    -> ordinary MM tiler
    -> MatmulTilingData
    -> wrap with runtime group metadata
    -> current ordinary-GMM project POD
```

每个字段都必须建立 `producer -> unit -> consumer -> gate` 映射。外层 POD
携带字段不表示 MM Scheduler 或 Block 会消费它。

### 成立条件/门禁

- 目标工程不得让 Launcher、Wrapper 或另一 helper 再次补写 group 字段。
- Host 与 Device 必须共享同一 POD 定义，并用 `sizeof/offsetof` 编译断言锁定
  ABI。
- 对完整 POD 执行的资源 gate、拷贝和序列化必须 byte-wise 保留 group 元数据。
- Device 从 `__gm__` 读取完整 POD 时，不得假定地址空间不同的结构体可以通过
  `local = *__gmPtr` 调用隐式复制构造。优先复用项目中已由目标编译器证明的
  loader；否则按共享 POD schema 逐字段读取到 local 对象。字段清单仍由同一
  POD 定义和消费映射产生，不能为 loader 维护第二份 Tiling 公式。
- 首个功能 checkpoint 必须逐字段核对
  `Host producer -> serialized offset/unit -> device loader -> first consumer`，
  不能以 `sizeof/offsetof`、Host 序列化成功或 Kernel 编译成功代替字段消费
  证明。对固定 cumsum 路线，`groupNum`、`m/n/k`、group-list 地址/类型和
  stride/layout 是必查字段；对其他编码，检查其对应的 group metadata。
- 首次设备 smoke 必须使用至少一个非空 group，且 Golden workspace 的
  `absmax` 非零；测试入口应确认 device workspace/scale 没有保持初始化哨兵值。
  若出现全零 workspace 或所有 scale 都停在 clamp 值，先按 control-plane
  失败检查 Tiling loader、dispatch、group validation 和阶段消费者，不得先
  修改 dequant、GLU 或量化公式。
- 不新增 grouped profile 或第二套 MM 搜索公式，除非目标工程证据证明当前
  项目侧 MM tiler 无法表达所需输入并在 `DESIGN.md` 中单独立项。该结论不从
  Blaze library 本身推导。

### 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| 调用方在 grouped `GetTilingData` 返回后补写 `groupNum` | 产生两个 POD 构造事实源 | 由 `MatmulTilingSwat` grouped overload 一次写完全部字段 |
| 把当前 SWAT 设计推广到所有 GMM | group 元数据是否参与 Host 搜索属于具体 tiler 合同 | 只对绑定的参考 tiler 使用 `totalM,N,K` |
| POD 有 `baseK` 就要求 Custom Block 常量与其相等 | 字段存在不证明 Block 消费 | 先建立字段消费映射，再决定 gate |
| 从 `__gm__` 整体解引用复制 POD | Host POD 可序列化不等于 device 地址空间间隐式构造合法 | 使用目标版本已证明的 loader 或逐字段读取，并编译真实 Kernel 调用 |
| Host POD 已写入字段但 device loader 遗漏字段 | 编译、`sizeof` 和 Host 拷贝都可能通过，Kernel 却按默认值提前返回 | 首个功能 checkpoint 做逐字段 loader/consumer 对账，并用非空 group positive-control 验证 |

## 3. Scheduler 与 Kernel

### 设计主线

GMM Scheduler 在普通 MM tile 规则外增加 group problem 生命周期：

```text
for group i:
    derive [start_i,end_i)
    skip empty group
    reset problem shape to [groupM,N,K]
    enumerate the group's M/N tiles
```

Kernel 是 group 地址与 problem 生命周期的 owner。当前参考资产的连续、非转置
ND 基线为：

```text
xGroupBase      = x + start_i * lda
weightGroupBase = weight + i * weightGroupStride
yGroupBase      = y + start_i * ldy
```

这里的 offset 默认以对应 Tensor 元素计。该公式不是 Blaze 通用布局规则；
独立/分离 Tensor、转置、非 ND 或其他 stride 合同必须由目标算子和实际
Scheduler/Kernel 重新绑定。若 ABI 使用字节地址，必须在接口处显式转换一次。

Tensor/Layout 可表达性与 Copy 实际消费能力的区分统一遵循
[Tensor API 参考](../fundamentals/tensor-api-reference.md)。GMM 必须为每个实际
Copy 调用绑定 group 输入的逻辑 layout/stride：支持域内直接连接；支持域外才在
算子边界增加显式 layout adapter，并把目标 group 基址、物理 leading dimension/
pitch、生命周期和同步依赖写入 DESIGN。不得把源 Tensor 的逻辑 stride 继续传给
packed Block，也不得把 materialize 推广为所有 GMM 的固定步骤。

### 3.1 逻辑维度与物理 tile 合同必须分开

一次实际设备故障是把逻辑 `K` 直接写入 A8W8 的物理 `baseK`；`K<` 当前硬件粒度时
设备返回 `507015`。相反，把 `N` 方便地切成多个窄 phase 也会破坏 full-row
per-token quant 的唯一 scale owner，并在 N-tail/多 phase 上挂死。可复用规则是：

- `ProblemShape` 的逻辑 `M/N/K` 与 `baseM/baseN/baseK` 的物理 tile/对齐约束必须
  在 Tiling、Block Params 和 Copy layout 中分别记录；不能因为字段同名就令
  `baseK==logical K` 或令一个窄 `baseN` 代表完整行。
- 物理 K tile 可以大于逻辑 K，但 MMAD 的 K-loop 必须有 source-backed 的逻辑 tail
  clip；物理 N phase 只有在有跨 phase 的 rowAbsMax/完成协议时才允许。否则由一个
  M-owner 遍历完整逻辑 N 后再发布 `yScale/y`，不能让每个 N slice 各自产生 scale。
- 正交回归至少同时覆盖 K tail、N tail、`N` 跨 tile、空 expert 和多 M tile；某个
  shape 通过不能证明逻辑/物理边界已闭合。具体粒度、tile 数和 workaround 只写入
  项目 DESIGN/PLAN，不写成 Skill 常量。

### 3.2 Source-backed assembly 与 MIX writer ownership

生产 GMM Kernel 必须从当前 Investigation 绑定的完整 assembly 出发，保留其
dispatch、Scheduler、Block 生命周期和跨核参与者集合。只调用 `BlockMmad` 的
手写入口可以用于隔离 ABI、数值或同步诊断，但不能作为正式生产路线；它绕过的
Scheduler/ownership 缺口不能由“单次 launch 返回”推断为已闭合。

在 MIX 中，逻辑 tile 的 owner 与物理参与者必须分开记录。AIV 只有在当前源码或
最小 probe 证明任务排列一致时，才能用 `GetBlockIdx()/GetTaskRation()` 恢复逻辑
AIC tile；需要区分 Vector rank 时再结合已证明的 `GetSubBlockIdx()`。
`GetBlockNum()` 的总量/分区语义也必须由同一事实源证明。对于 `baseM=1` 或其他
单行 tile，必须显式指定唯一 sub-block writer；未拥有该 tile 的 lane/sub-block
不得触碰 C、workspace、`y` 或 `yScale`。每一个 writer 都要对应自己的输出
pitch、有效范围和完成通知，不能依赖物理 index 连续或“两个 sub 都会执行”的
假设。

### 接口与数据流

| 层 | 复用普通 MM | GMM delta |
|---|---|---|
| Scheduler | M/N tile 形状和核分配原则 | 按 group 重置 problem、跳过空组、维护 prefix |
| Kernel | Block 调用和单个 tile 数据流 | 遍历 group、构造 group GM view、保持 tile 索引一致 |
| Params | A/B/C、layout、stride、tiling 地址 | group POD 或 runtime group 元数据 |
| 验证 | 单 MM 的 C、tail、multi-tile | 空组、不均匀组、prefix、group tail、编码 selector |

融合 Scenario 只在此基础上增加自身 delta。例如 GLU 增加成对 N 轴 view，
per-token quant 增加完整逻辑行的量化阶段；它们不重新定义 group 编码和基址。

### 成立条件/门禁

- `route=blaze` 时 Scheduler/Kernel 必须使用目标版本 Investigation 绑定的
  MM/GMM 基础组件；`route=scenario` 时 custom Scheduler/Kernel 还必须绑定
  选中资产的精确接口。参考资产缺失不能推导 Blaze 组件缺失。
- `__mix__` 下 AIC/AIV 的物理 block index、task ratio 和 sub-block 值不能仅按
  名称或连续编号解释为逻辑 owner。DESIGN 必须由目标版本源码或设备可见探针
  冻结真实参与者集合、物理到逻辑的映射和每条逻辑行的唯一 consumer。若该映射
  无法证明，选择同一 stream 的独立 producer/consumer entry 或明确完成协议，
  再在依赖正确的候选中用同设备性能数据选择；不得把 MIX 预设为默认最优。
- 每个 group 的 B stride、x/y leading dimension、transpose 和 layout 必须来自
  用户合同，不得由 cumsum 编码推导。
- 启用 layout adapter 时，源逻辑索引、目标物理索引、目标 leading dimension
  和首个 Cube consumer 的可见性必须分别验证。

### 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| 把 GMM Scheduler 改名后当普通 MM Scheduler | problem 生命周期不同 | MM 使用 Investigation 绑定的 MM Scheduler |
| 在每个融合 Scenario 重写 group 公式和三组基址 | 容易产生多份漂移事实 | 链接本文，只保留融合特有 delta |
| AIV 直接用物理 `GetBlockIdx()` 作为完整行 owner | `__mix__` 可能有稀疏/非量化参与者，物理 index 与逻辑 rank 不同 | 证明参与者集合和映射；无法证明时使用同 stream split 或明确完成协议 |
| 生产路径直接手写 `BlockMmad`，没有 source-backed dispatch/Scheduler/ownership 组装 | 直调缺少真实 ABI 和参与者协议，可能在同步点挂起 | 恢复源 assembly；直调只保留为已标注的隔离 probe，并单独记录首个失败边界 |
| 两个 MIX sub-block 都写同一个单行 tile | 物理 sub-block 数不等于逻辑 M owner 数 | 由源码/probe 冻结唯一 writer，并覆盖单行、奇偶 M、tail 和重复运行 |
| Copy 路线未消费某个 stride 就判定 Layout 不可表达 | Layout 能力与 Copy specialization 支持域被混淆 | 先查实际 Routing；不支持时才增加显式 adapter，并改用目标物理 pitch |

## 4. 验证

### 设计主线

除普通 MM 的 dtype、layout、transpose、M/N/K tail 和 multi-tile 外，GMM
至少增加：

- cumsum 与 count 的独立正例；
- 首组、中间组、末组为空；
- 多个小组、不均匀组和 group 内 M tail；
- group 起点、专家 B 选择和 y 写回 offset 的非对称数据；
- 重复执行和非法 Host 输入拒绝。

### 成立条件/门禁

精度标准引用项目唯一事实源，不在本文复制公式。设备证据必须记录目标架构、
CANN、入口、group 编码、shape 和关键返回码。

### 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| 只测总 `M`，不检测专家 B 选择 | 多组可能都错误使用同一 B | 为每个专家使用可辨识 pattern |
| 只测无空组的均匀分组 | 未覆盖 prefix 和跳过逻辑 | 覆盖连续空组及前中后空组 |
| 用最终融合输出代替 GMM 分层诊断 | 无法定位 group、MM 或 Epilogue | 保留可归因的 C 边界验证 |
