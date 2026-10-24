# VariableOutput 机制原理

> 本文是 VariableOutput 范式的原理层：四阶段生命周期、outshape 回传协议、kernel 计数定形技术、
> 与值依赖/列表输出的分界。场景路由与契约清单见 [patterns.md](patterns.md)。
> 本文是**自包含知识**：算子规格（语义 + 输入输出表）+ 本文档即可完成场景判定与四阶段全链路
> 设计，不依赖、也不应查阅任何算子库/存量实现（见 §9）。文中算子名与案例库（§6）仅作形态
> 举证，不是需要打开的"参考实现"。

---

## 1 本质：shape 推导的信息论边界

一个输出维长，按"何时能知道"分三个层级：

| 层级 | 信息可用时刻 | 定形机制 | 实例 |
|------|------------|---------|------|
| L1 | 构图/编译期（shape 演算） | InferShape 直接算 | Add、Concat |
| L2 | 执行前（输入值 host 可读） | 值依赖 + D2H 后读值 | Reshape、MirrorPad |
| L3 | **执行后（计算产物）** | kernel 计数 + outshape 回传 | NonZero、DynamicPartition |

VariableOutput 就是 L3：维长是**对输入数据的统计量**（非零个数、各标签计数、去重个数），
数据不跑完不存在。判定时问一句："给我输入的完整 shape/dtype/attr，不看数据本身，能否算出输出
shape？"能 → 不是本范式；不能、必须数数据 → 是。

两条裁决规则（多输出/多分支/双模算子按此归档）：

- **逐输出、逐分支取最大层级**：多输出算子对每个输出独立回答"维长何时可知"（场景分支依赖的
  按最坏分支，如 GroupedMatmul 不同 group_type 分支的 M 维来源不同），取最大层级——只要有一个
  输出是 L3，算子整体进本范式；全部 ≤ L2 则整体排除
- **按最一般执行路径归类**："host 可读"指**算子契约保证**（`ValueDepend(REQUIRED)` 强制常量，
  或值依赖声明触发运行期 D2H）。常量输入时 host 恰好能推出精确 shape，只是**退化分支**——
  契约允许运行期输入且未声明值依赖时，按运行期路径（值只在 device）归 L3
  （BroadcastGradientArgs：常量分支 infershape 直接读值算出 y 长度，非常量走 kernel 计数写
  outshape，整体按 L3 归本范式）

两个易混概念的排除：

- **DYNAMIC_OUTPUT / `ParamType(DYNAMIC)`** 只表示"输出（或输入）是列表"，个数由属性/构图决定，
  每个输出的 shape 可以编译期已知。DYNAMIC 落在**哪一侧**都不影响结论：Split/IdentityN/
  ShapeN（输出列表）与 Pack/Concat（输入列表）都不是 VariableOutput。
  `OutputShapeDependOnCompute` 才是本范式在 op_def 上的标识（两者独立、可组合）
- **值依赖（DynamicShape 范式）** 是"执行前把输入值搬到 host"；本范式是"执行后把输出 shape 从
  device 搬回框架"。方向相反、时机相反，救不了对方

## 2 四阶段生命周期

以 DynamicPartition（K 个输出 `[m_k] + x.shape[P:]`，m_k = partitions 中值 k 的计数）为例：

```
构图:   create_dynamic_output_y(K)            # 输出个数 = attr，此时已知
  ↓
编译:   InferShapeRange → 每输出 [min,max]      # min=0, max=partitions 元素总数
        输出内存按 max 预分配                    # m_k ≤ max 必须成立（内存契约）
  ↓
tiling: 只读 x/partitions 的 shape/dtype        # 输出 shape 不存在，任何决策只围绕输入
        规划 user workspace（跨核计数缓冲）
  ↓
执行:   kernel 边搬边数 → 末核把真实 shape 写 yshape
  ↓
回传:   框架读回 yshape，刷新各输出真实 shape → 交给下游
```

### 2.1 op_def 声明（阶段 1）

```cpp
// dynamic_partition_def.cpp
this->Output("y")
    .OutputShapeDependOnCompute()   // 标识：shape 由计算决定（OpParamDef API）
    .ParamType(DYNAMIC)             // 列表输出（可选，与上者独立）
    .DataType(xDType)...;
OpAICoreConfig aicoreConfig;
aicoreConfig.DynamicCompileStaticFlag(true)
    .DynamicRankSupportFlag(true)
    .DynamicShapeSupportFlag(true);  // 动态 shape 能力必须开
```

- `OutputShapeDependOnCompute()` 声明在 `register/op_def.h` 的 **OpParamDef**（输出参数链）上，
  不是 IMPL 注册块
- 单输出算子（MaskedSelectV3、NonZero）配 `.ParamType(REQUIRED)`；列表输出（DynamicPartition）配
  `.ParamType(DYNAMIC)`（等价旧 IR 的 `.DYNAMIC_OUTPUT(y, ...)`）
- 老构图接口（op_graph proto）对应写法：`.DYNAMIC_OUTPUT(y, ...)` + attr
- 动态能力 flag 是**能力开关，不是范式标识**：`DynamicShapeSupportFlag` 服务 -1 维编排（本范式
  通常必备）；`DynamicRankSupportFlag`（-2）与 `DynamicCompileStaticFlag` 按是否支持未知
  rank/动静态编译选取。非本范式的算子也可全开（GroupedMatmul、NonZeroWithValue 实证）——范式
  标识只有 `OutputShapeDependOnCompute`

### 2.2 infershape 两种注册风格（阶段 2）

| 风格 | 注册 | 适用 | 实例 |
|------|------|------|------|
| A：仅 Range | `IMPL_OP_INFERSHAPE(Op).InferShapeRange(F)` | rank 也推不出 / 输出是列表 | DynamicPartition |
| B：-1 + Range | `.InferShape(F).InferShapeRange(F2)` | rank 可由输入（含 attr）推出，仅维长未知 | MaskedSelectV3、NonZero、UniqueDim、UniqueConsecutive、StatelessRandomChoiceWithMask |

风格 B 的 InferShape：未知维一律 `-1`，不得给猜测值；rank 已知的维照常填（NonZero 的
`[dim, -1]` / `[-1, dim]` 随 transpose 变化）。

另有第二处注册位：`IMPL_OP_INFERSHAPE` 注册块可同时声明
`.OutputShapeDependOnCompute({输出索引...})`（`register/op_impl_registry.h` 的 OpImplRegisterV2 API）。
存量两种风格并存——UniqueDim/UniqueConsecutive 声明了 `{0,1,2}`（全输出），DynamicPartition/
MaskedSelectV3/NonZero 未声明（只靠 op_def 侧）。框架对两处的生效差异未查实；案例库实证：**仅
op_def 侧声明即可建立 outshape 通道**（DynamicPartition/MaskedSelectV3/NonZero 均如此），IMPL 侧
属可选补充，两处同声亦有存量实证、不产生冲突。**新写算子只声明 op_def 侧即可；是否追加 IMPL 侧
以当前工程惯例为准，不确定时按最简（仅 op_def 侧）取**。

InferShapeRange 的推导要点（按案例库已证实实现归纳）：

- **min**：语义允许空输出就取 0（分区可能一个条目分不到；NonZero 全 false 输出 0 行）
- **max**：真实上界，三个来源层级——
  1. 输入含 -1（unknown dim）→ max 也只能 -1 或按各维上限乘积；
  2. 标签/计数输入的元素总数（`GetShapeSize()`），标量取 1；
  3. 组合值依赖：上界还受输入值影响时（StatelessRandomChoiceWithMask 的 count），
     max = max(shapeSize, count)，count 本身是值依赖输入（`InputsDataDependency({1})`）
- 列表输出的 K 份 range 无信息可区分 → 每份复制同一个 [min, max]
- **契约**：运行期真实 shape 必须 ⊆ [min, max]；max 同时是预分配大小，越界 = 越界写内存

### 2.3 tiling 约束（阶段 3）

- 入口只 `GetInputShape/GetInputDesc`（输出 shape 此刻是 -1/max，读了也没用）
- **user workspace 规划**是本范式 tiling 的特征项（多核时）：跨核计数需要协调缓冲，但布局因
  算子而异——DynamicPartition：`usedCoreCnt × coreWS` 字节，`coreWS = min(K, 4096) × 8`（每分区
  一个 uint64，单核缓冲上限 4096 分区，超出走分批循环）；masked_select_v3：`numBlocks × 64B`
  偏移表 + 各核暂存区；UniqueDim：每核 128B 计数页 + 全局前缀区。共性：**多核写同一输出就必须
  规划协调缓冲**，公式按算子设计取；单核速决（§5.4）则无此需求
- **kernel 侧基址契约**：计数缓冲一律从 `GetUserWorkspace(workspace)` 起用——裸 workspace 基址前
  2MB 是系统保留区（RESERVED_WORKSPACE），直接绑定会踩系统 workspace（存量反例：unique_consecutive
  的多核 kernel 把计数区绑到 raw 基址）
- **outshape 缓冲占用一个 workspace 参数槽位**：框架下发的 outShapeTensor 会占掉
  `GetWorkspaceSizes(n)` 的一个槽——tiling 需按实际槽位数取 n：除 outshape 槽外无其他
  workspace 需求时取 `GetWorkspaceSizes(1)`（BroadcastGradientArgs 单核速决即此场景），该槽
  大小可置 0（槽位数不匹配会导致 kernel 参数表错位）
- **跨核计数缓冲按核数条件化**：多核写同一输出才需要 user workspace 协调缓冲；数据量极小走
  单核速决（见 §5.4 退化表）时**不留**计数缓冲——声明了 kernel 不消费的区段即死预留
- 退化分支在 tiling 期定形：K=1 时 dim0=H 已知，tiling data 直接预填 `dim0` 和
  `outDimsExtFirst[]`（x.shape[P:]），kernel 照抄进 yshape——**能提前知道的，实现就应该提前知道**

### 2.4 kernel 阶段（阶段 4）见 §3、§5

## 3 outshape 回传通道

### 3.1 图模式（apt kernel）的隐藏形参

声明了 `OutputShapeDependOnCompute` 的输出，框架在 kernel 参数表**自动附加**一个 outshape 指针：

```cpp
// dynamic_partition_apt.cpp —— op_def 只有 x/partitions/y 三个 tensor 参数
void dynamic_partition(GM_ADDR x, GM_ADDR partitions, GM_ADDR y, GM_ADDR yshape,
                       GM_ADDR workspace, GM_ADDR tiling)
// masked_select_v3_apt.cpp 同构
void masked_select_v3(GM_ADDR x, GM_ADDR mask, GM_ADDR y, GM_ADDR shapeout,
                      GM_ADDR workspace, GM_ADDR tiling)
```

位置约定：**紧跟最后一个输出之后、workspace 之前**。框架执行完读回这块内存刷新输出 shape。

### 3.2 内存编码协议

```
word[0]          : dimNum [| B64_FLAG]        // B64_FLAG = 1UL << 31（=0x8000'0000）
word[1]          : 真实 dim0（运行期数出来的）
word[2..dimNum]  : 其余各维（编译期已知，来自 tiling data）
```

- 框架按 word[0] 的 **dimNum** 解析，其后写多少 word 跟随 dimNum。`SHAPE_GAP = 9`（dimNum + 最多
  8 维）是 DynamicPartition 的保守整块布局（不足位填 1、按批 DMA），**不是硬协议**：
  masked_select_v3 单维输出只写 2 个 word
- **B64_FLAG 是"本块按 uint64 编码"的信号，存量写法有分歧**：DynamicPartition 只打全局第一个输出
  的 word[0] 且写完立即恢复裸 dimNum；UniqueDim 给全部输出打（常量 `0x80000001`）；
  UniqueConsecutive 不打（裸 dimNum）。框架对不同写法的解析差异未查实——**新写算子从上述三种
  已证实写法中任选其一（规则以本节描述为准），全算子保持一致**，禁止自创第四种 B64 打法
- **三要素分层（协议位 vs 布局惯例）**：word[0] 的 dimNum|B64_FLAG 编码是**协议位**——框架解析
  的唯一依据，打法从已证实写法中选取；word 数（SHAPE_GAP 整块 vs dimNum+1 精确）与填充值
  （补 0/补 1）是**布局惯例**，可与协议位独立组合（SortedNMS：9-word 整块 + B64 打 + 补 0，与
  DynamicPartition 的补 1 并存且均正常工作）
- **唯一写者原则**：只有一个核写 yshape（避免写冲突 + 值正确）。默认末核（DynamicPartition：
  `if (blockIdx_ != usedCoreCnt - 1) return;`——末核前缀和终值恰为全局总数，正确性来源见 §5.1）；
  非前缀和结构可指定其他唯一核（SortedNMS 收敛到 0 号核写出；masked_select_v3、UniqueDim、
  UniqueConsecutive 同为末核写者）
- **多输出的槽位布局**：注入缓冲按**每个输出一个 SHAPE_GAP 槽位**顺序排布——不仅列表输出，
  固定个数的多输出同构（BroadcastGradientArgs 的 y2 槽位在 offset 9；列表输出第 k 个槽位偏移
  `k × SHAPE_GAP`，K > 4096 分批时偏移 = `批号 × 4096 × 9`）。存量实证中**未声明
  DependOnCompute 的输出也占槽并写出**（SparseSlice 为静态的 y_shape 写第 3 槽）；框架按全部
  输出数还是已声明数分配缓冲未查实——**新写算子按全部输出数排布槽位、逐槽写出**（与实证
  一致），声明子集决定框架消费哪些槽
- 空输出 dim0 = 0 是合法编码——**空输入不豁免写 shape**（协议性输出）；也存在另一种合法策略：
  tiling 期直接拒绝空输入（masked_select_v3），二选一并全算子一致。混合输出形态下"空"只作用于
  **依赖计算的输出**（dim0=0），静态输出照常写真实 shape（SparseSlice 空模板：y_indices=[0,
  rank]、y_shape=[rank] 内容照写）——每个输出按自身语义回答，不是全体一起变空

### 3.3 L2（aclnn 单算子）模式的变体

L2 实现不走 op_def 隐藏形参，用 executor 显式机制：

```cpp
// AICORE 路径：AllocTensor 出 shape 缓冲，随 kernel 下发，执行后框架刷新 out  
Shape outShapeShape{8 + 1}; // 9 × int64（= dimNum+8 维上界）
auto outShapeTensor = executor->AllocTensor(outShapeShape, DataType::DT_INT64, ...);
ADD_TO_LAUNCHER_LIST_AICORE(NonZero, OP_INPUT(self), OP_OUTPUT(out),
                            OP_ATTR(...), OP_OUTSHAPE({outShapeTensor, 0}));  // 0 = 输出序号
```

- **多输出形态**：每个输出各挂一条 `OP_OUTSHAPE`，如 UniqueConsecutive L2 用共享的 27×int64 tensor
  + 三条 `OP_OUTSHAPE({outShapeTensor, 0/1/2})`（27 = 3 输出 × 9 word）
- L2 的 out 由调用方按上界传入（PyTorch 风格），执行后由框架按 outshape 刷新真实 shape
- L2 `GetWorkspaceSize` 阶段**不得**同步等待数据来"提前知道 shape"——AICore 主线路径 shape
  一律执行后回传

## 4 kernel 侧输出列表访问（ListTensorDesc）

`ParamType(DYNAMIC)` 输出在 kernel 侧的表现是"一张指针表"（`asc/include/.../kernel_operator_list_tensor_intf.h`）：

```cpp
ListTensorDesc outGMList_(reinterpret_cast<__gm__ void*>(y));   // y = 列表输出形参
auto outK = outGMList_.GetDataPtr<T>(partID);                   // 第 k 个输出的 GM 基址
```

- 输出个数 K 来自 attr/tiling data，kernel 按 k 遍历
- 单输出算子不需要 ListTensorDesc，直接 GlobalBuffer 即可

## 5 kernel 计数定形骨架

### 5.1 两遍法（多核写同一输出的标准解法）

问题：多核并行搬运时，谁写 y[k] 的哪一段？答案：先数数、再做排他前缀和、最后按预定偏移搬运。

```
Step 1 本地计数:  每核扫自己行段的标签（分批进 UB），向量计数:
                  CompareScalar(mask, parts, k) → ReduceSum → 每分区计数
                  写 user workspace 本核区段 ws_[blockIdx * coreWS]
Step 2 SyncAll + 排他前缀和: 每核把 ws_[0..blockIdx) 的计数累加 → ubPartBase[k]
                  = 本核在分区 k 的起始行号（无需归约核，核数少开销可控）
Step 3 搬运:      GatherMask 把命中行压实成行号表（AR 寄存器免费给出命中数），
                  整行块拷到 y[k] 的 baseOffset 处；ubPartBase[k] 边搬边涨
Step 4 末核写 shape: 末核 base 终值 = 前面所有核 + 自己 = 全局总数 m_k → 写 yshape
```

关键不变式（可手算验证）：`Σ m_k = H`（条目守恒）；末核终值恰为全局计数是"末核写 shape"
的正确性来源。

### 5.2 免同步替代（重复扫描）

W 切分：每核**扫全量**标签但只搬自己的列段——每核都见过全量数据，各自的分区计数天然是全局值，
零跨核协调；代价是标签被重复读 usedCoreCnt 遍（tiling 决策给 W 切分设 2 倍惩罚）——以标签
重复读的带宽开销换取零跨核同步。

### 5.3 序敏感输出的定序骨架（并行预处理 + 单核按序 append）

两遍法/前缀和解决的是"无序 scatter"——输出各行的相对顺序无语义，谁先谁后无所谓。若**输出
顺序本身有语义**（NMS/TopK 类按分数序贪心选择，重排即语义错误），前缀和不可用，标准解法换
形态：

```
Step 1 多核并行预处理:  O(N²) 的成对关系预计算成掩码矩阵（如成对 IoU 抑制位图），
                         按全局并行单元切分，写 user workspace
Step 2 SyncAll:         掩码矩阵"产完/消费"的分界
Step 3 单核串行决策:    按序贪心，计数器边选边涨——偏移即计数器，天然单写者按序 append，
                         无需前缀和；终计数 = 输出维长
Step 4 唯一核写 shape:  决策核（如 0 号核）写 yshape
```

判别条件：输出序有语义约束 → 前缀和/两遍法不可用，走本骨架（SortedNMS 实证：多核
BuildPairwiseMasks + 单核贪心，跨核协调仅一个广播字 + SyncAll）。

### 5.4 退化与兜底分支

| 分支 | 条件 | 定形方式 |
|------|------|---------|
| tiling 期定形 | K=1（dim0 恒 H）/ shape 可提前算 | tiling data 预填，kernel 照抄 |
| infershape 期定形 | 值依赖输入为编译期常量（IsConstTensor） | host 读值直接推出精确 shape；kernel 仍照写 outshape（BroadcastGradientArgs 实证。框架如何取舍两处来源未查实——保守做法：kernel 无条件照写，运行期路径不依赖 host 侧结论） |
| 单核速决 | 有效数据量极小（rank 级/个位数元素） | BlockDim=1 顺序计数，唯一写者天然成立，无需 user workspace 计数缓冲（BroadcastGradientArgs：标量路径边扫边数 / 向量路径 Squeeze + AR 寄存器计数） |
| 空输入仍计数 | x 空、partitions 非空 | 无数据可搬但 dim0 仍要数，写 [m_k, ...] |
| 空输入短路 | tiling 期拒绝空输入 | 报错返回（masked_select_v3 策略；与"写全 0 shape"二选一，全算子一致） |
| 全空 | x、partitions 均空 | 计数清零，写全 0 dim0 的 yshape |
| 标量 | x、partitions 均标量 | y[p]=[·] 其余 [0]，单核速决 |
| 小尾块 | 行宽 ≤ 一个 block | 降到元素级 Gather（B8/B16 需 Pack 索引） |
| K > 4096 | 分区数超单核缓冲 | 分批循环，每批独立前缀和 + yshape 按批偏移落位 |

多输出的混合形态（UniqueDim/UniqueConsecutive/SparseSlice 实证）：并非每个输出都依赖计算——
idx 可能静态可推（恒 [numInp]），甚至是不支持的哑输出（kernel 恒写 [1]），也可能三个输出中
两个依赖计算、一个纯静态（SparseSlice 的 y_shape）。**每个输出独立回答"维长何时可知"**
（判定规则见 §1 逐输出裁决），但只要有一个依赖计算，算子整体进本范式；kernel 写出的 shape
不得与 infershape 的静态结论矛盾（哑输出、空输入分支最容易翻车——存量代码里 infershape 说
rank=n、kernel 写 rank=1 的矛盾是真实缺陷）。

### 5.5 与主范式 Compute 的分工

计数/定形链路（本范式）与数据搬运/计算（主范式）在同一 kernel 内并存：主范式决定
CopyIn/Compute/CopyOut 的数据流，本范式叠加"计数器、workspace 前缀和、唯一核 shape 写出"。
UB 规划要同时满足两者（DynamicPartition 用 TBufPool 三池复用：process/after/sync）。

## 6 案例库

> 案例库是对存量实现的**形态举证**（知识本身），用于佐证判定法则、写法分歧与退化分支真实存在。
> "位置"列仅为溯源信息，**分析新算子不需要检索这些路径**——判定依据是 §1 判据与四阶段契约，
> 不是"库里怎么写"（见 §9 自包含声明）。

| 算子 | 位置 | 输出形态 | infershape 风格 | 定形机制 | 备注 |
|------|------|---------|----------------|---------|------|
| DynamicPartition | ops-math/conversion/dynamic_partition | K 个（DYNAMIC） | 仅 Range | 两遍法 + 末核写 yshape + K=1 tiling 预填 | 本范式四阶段全链案例，§2~§5 机制的举证来源 |
| MaskedSelectV3 | ops-math/conversion/masked_select_v3（canndev 有 twin） | 单输出 | -1 + Range | 掩码计数 + shapeout（仅 2 word） | range min=1；tiling 拒空输入 |
| NonZero | canndev/ops/index/non_zero | 单输出（transpose 定 rank 顺序） | -1 + Range | L2 `OP_OUTSHAPE` | L2 模式代表 |
| UniqueDim | ops-nn/index/unique_dim | 3 输出（y/idx/count 全 DependOnCompute） | -1 + Range | 排序去重 + 末核全局前缀和终值写 shapeout | IMPL 侧另声明 `.OutputShapeDependOnCompute({0,1,2})`；idx 静态可推、B64_FLAG 全输出打 |
| UniqueConsecutive | ops-nn/index/unique_consecutive | 3 输出（y/idx/count） | -1 + Range | 相邻比较 + 两遍法（与 §5.1 同构） | idx 为哑输出恒 [1]；L2 多输出 OP_OUTSHAPE 0/1/2；⚠ 存量疑似缺陷：计数绑 raw workspace 基址 |
| StatelessRandomChoiceWithMask | ops-math/random/... | 固定 2 输出 | -1 + Range + InferDataType | 随机选择计数 | **与值依赖组合**：count 是 InputsDataDependency({1})，参与 max 推导 |
| BroadcastGradientArgs | ops-nn/index/broadcast_gradient_args | 固定 2 输出 | -1 + Range（常量分支 host 精确推导） | 单核计数 + kernel 写 out_shape + L2 OP_OUTSHAPE（共享 18×int64，槽位 0/9） | 双模实证：常量 infershape 期定形、非常量走 L3（§1 最一般路径规则）；单核速决；outshape 占 workspace 槽位（GetWorkspaceSizes(1)） |
| SparseSlice | ops-nn/index/sparse_slice | 3 固定输出（y_indices/y_values 依赖计算，y_shape 静态） | 动态输出 -1 + Range、静态输出正常推导（Range 全输出注册，静态份 min=max） | 两遍法 + 末核写 + 空输入写全 0 模板 | 混合输出形态；**与值依赖组合（tiling 期）**：读 shape/start/size 值预填辅助数据、空盒判空走退化分支；未声明输出的槽位照写（§3.2） |
| SortedNMS | ops-cv/objdetect/sorted_nms | 单输出 | -1 + Range | §5.3 定序骨架：多核成对掩码 + 单核贪心按序 append，0 号核写 shape | 序敏感形态实证；max_output_size 值只影响上界（未做值依赖收紧，max=N——值依赖收紧 range 是可选项） |
| Split / IdentityN / ShapeN / Copy | canndev built-in | 列表输出 | 正常 InferShape | 无需定形 | **反例**：DYNAMIC_OUTPUT 但 shape 已知，非本范式；Pack 同为反例但 DYNAMIC 在输入侧（列表输入、单静态输出） |
| NonZeroWithValue | ops-nn/index/non_zero_with_value | 3 静态输出（value/index 按最坏情况、count=[1]） | 正常 InferShape | 无（**max-size + count 值输出**路线） | **反例（第三路线）**：变长信息编码进 count 的值而非 shape，输出静态、整条 outshape 链路免除；kernel 借两遍法仅做数据定位（定位 ≠ 定形）。NonZero 家族路线分裂（non_zero 走 L2 OP_OUTSHAPE）——同名/同家族不构成判定依据 |
| MemSetV2 | ops-math/conversion/mem_set_v2 | 列表输出 | 正常 | 无需定形 | **反例**：DYNAMIC 输入输出但 kernel 无 shapeout 形参（无通道 = 非本范式的决定性判据） |
| Coordinates1DTo2D | ops-math/conversion/coordinates_1d_to_2d | 3 输出 | 正常（输出 shape 复制输入） | 无需定形 | **反例**：注册了 InferShapeRange ≠ 本范式，判定看维长来源 |
| DynamicStitch | ops-math/conversion/dynamic_stitch | 单输出 | 正常 | 无需定形 | **反例**：DYNAMIC 是输入列表（indices/x），输出 shape 由拼接规则推出 |
| ChunkCat | ops-math/conversion/chunk_cat | 单输出 | 正常 | 无需定形 | **反例**：DYNAMIC 在输入 x，y.shape 由列表 shape + attr 纯计算得出 |

## 7 常见误区

1. **DYNAMIC_OUTPUT 当 VariableOutput**——列表输出 ≠ shape 依赖计算；判定看维长来源，不看输出个数。
   决定性判据：kernel 签名有没有框架注入的 shapeout 形参（MemSetV2 无 → 必然不是）
2. **想让 tiling 读输出 shape**——它此时是 -1/max；本范式 tiling 只能围绕输入设计
3. **InferShapeRange max 随手给**——max 是预分配内存大小也是协议上界：给小运行期越界写，给大浪费
   内存；含 -1 输入时 max 的推导要按各维上限乘积，不能直接乘 -1
4. **多个核都写 yshape**——写冲突 + 值错误；唯一写者（惯例末核）是前缀和结构的副产品，不是随意的
5. **自创 shapeout 编码**——B64 打法是协议位，从 §3.2 已证实写法中任选其一并全算子一致；
   word 数与填充值是布局惯例，可独立组合（§3.2 三要素分层）
6. **空输入忘了统一策略**——"写全 0 shape"与"tiling 拒空"都是合法实现，但同一算子内必须一致，
   且空分支的 rank 也要与 infershape 结论一致
7. **L2 里同步等数据定 shape**——AICore 主线不支持也不应阻塞等 device 结果；shape 一律执行后回传
8. **与值依赖混用**——值依赖输入（如 count）参与的是 range 上界推导（编译期/运行期推导前），
   不能替代 kernel 计数；两者职责不同、可共存
9. **忘记 Σ m_k = H 类守恒不变式**——它是多核前缀和正确性的可测断言，测试用例必须覆盖
10. **kernel 写 shape 与 infershape 静态结论矛盾**——哑输出（恒 [1]）、静态可推输出（恒
    [numInp]）、空输入分支最易翻车；逐分支核对两边说的 rank/维长是否同一个故事
11. **计数缓冲绑到裸 workspace 基址**——user workspace 必须从 `GetUserWorkspace(workspace)` 起
    取，raw 基址前 2MB 是系统保留区（存量反例：unique_consecutive）
12. **tiling 预留 kernel 不消费的 workspace 区段**——每个声明区段都应被 kernel 使用，死预留既浪费
    又误导后续维护（单核速决时不要照多核模板预留计数缓冲）
13. **kernel 运行期读输入值做切分/定位，当成本范式**——几乎所有算子的 kernel 都会读输入值
    （GroupedMatmul 读 group_list 推进每组行区间）；范式分界是**输出 shape 何时在 host 定形**：
    GroupedMatmul 的 y[g] 在 infershape/tiling 执行前已定形并完成内存分配，kernel 只是向预分配
    内存写入——"kernel 是否读值"不具判别力，"shape 是否 host 期定形"才是
14. **把动态能力 flag 当范式标识**——DynamicShapeSupportFlag 等是能力开关，非本范式算子也常全开
    （NonZeroWithValue、GroupedMatmul）；范式标识只有 `OutputShapeDependOnCompute`，反向判据是
    kernel 无 shapeout 形参（见误区 1）

## 8 测试要点

- 范围契约：随机数据下真实 dim0 恒 ≤ max、≥ min（协议一致性）
- 守恒不变式：Σ m_k = H（构造可手算的小用例——如两核各分得若干条目——推演各核排他前缀和与
  末核终值恰好等于全局计数）
- 空输出/空输入/全空/标量/K=1/K>4096 各分支的 shape 编码正确性（rank 与 infershape 一致）
- 多核偏移不重叠（数据竞争检测）
- dimNum 与 B64_FLAG 的位组合按所选已证实写法（§3.2）的预期逐字断言
- 哑输出/静态可推输出的 shape 与 infershape 静态结论一致
- 双模算子（常量输入 infershape 期定形 + kernel 照写 outshape）两处结论一致性——同一输入下
  host 推导值与 kernel 计数值必须相同

## 9 自包含声明

- 本范式是**自包含知识**：给定算子规格（语义 + 输入输出表）+ 本文档，即可完成 VariableOutput
  判定与四阶段全链路设计。**不需要、也不应去查阅算子库或任何存量实现**——判定依据是 §1 的
  信息层级判据与 §7 误区清单，不是"库里怎么写的"
- 案例库（§6）与文中出现的算子名（DynamicPartition、MaskedSelectV3、NonZero、UniqueDim 等）仅作
  **形态举证**：佐证判定法则、写法分歧与退化分支的真实存在；"位置"列是溯源信息，不是必读入口。
  禁止把"逐字抄某参考实现"当作设计方法——写法分歧处按本文档的决策规则取舍（IMPL 侧声明默认
  省略，见 §2.2；B64 打法从已证实写法中任选其一，见 §3.2），全算子保持一致
- 若分析中对某算子是否属于本范式拿不准，回到 §1 三层级判定（"不看数据能否算出输出 shape"）
  与 patterns.md 判定法则重新推导，而不是找类似算子的实现来仿写
