# Reduction 类算子 DESIGN.md 生成规范

本文件是 reduction 类算子方案设计的**知识索引入口 + 生成强制规范**。spec-to-design 技能为 reduction 类算子生成 `DESIGN.md` 时，**必须**遵守本文件的规则。

---

## 强制规则（生成前必读）

1. 生成 DESIGN.md 时，必须逐节查阅下方设计规范中的**所有必读文件**，从对应文件中提取结论
2. 必读文件中的"约束"、"规则"、"陷阱"、限制说明，必须在 DESIGN.md 对应章节中**以表格或编号列表显式写出**，不得省略
3. 必读文件中的代码必须写到 DESIGN.md，仅替换算子特有占位符（Reducer/PostElewise/dtype）。代码块内禁止出现 `// ...`、`/* ... */` 等省略占位符，也不得用自然语言描述替代真实 API 调用。容易静默出错的参数必须保留 `⛔` 标注。**凭记忆自创的 Ascend C 代码必然出错——必读文件的写法是唯一正确来源，禁止凭理解自行编写**
4. **不得凭记忆或经验编写**；必须以必读文件中的结论为准，不得自行发挥
5. 若必读文件知识与 `spec.yaml` 或 `REQUIREMENTS.md` 冲突，以 spec 为准，并在对应章节标注差异
6. 所有 API 映射必须引用 `[regbase_api_whitelist.md](../../api/regbase_api_whitelist.md)` 中的验证状态；**不得把"待验证"写成"已验证"**
7. **本规范中的 reduction 专用知识优先于 agent 通用知识**。若两者对同一设计点有不同结论（如数据流结构、API 选型、Buffer 布局），以本规范指向的必读文件为准，不得用通用经验覆盖
8. **源码标注规范**：(附源码) = DESIGN.md 必须包含 ```cpp 代码块，禁止散文替代；(附结构) = DESIGN.md 必须包含 ```cpp 结构体定义；(附公式) = DESIGN.md 必须包含公式块
9. **路线固定 RegBase**：reduction 范式所有算子统一走 RegBase 路线

---

## 参考代码

以下 euclidean_norm 示例是 reduction 范式的参考实现，以 arch35 为例，覆盖 Kernel、Tiling、数据结构三层。设计阶段（DESIGN.md 生成）和开发阶段（代码实现）均可查阅，用于理解文件组织、类结构、入口注册、Tiling 流程等具体写法。

**覆盖场景**：全 reducer 统一 pad 清零路径 + 二分缓存树 + 通用类别（MAX_PATTERN_RANK=9）+ tail-R/tail-A 双路径 + Group 模板 + Empty tensor。

**未覆盖场景**：mean/ArgReduce 变体、All Reduce/单 R 轴类别。这些场景的实现细节以 references/ 文档为准。

| 层级 | 文件路径 | 内容 |
|------|---------|------|
| Kernel | `example/euclidean_norm_package/op_kernel/arch35/euclidean_norm_base.h` | normal 模板 kernel 类 |
| Kernel | `example/euclidean_norm_package/op_kernel/arch35/euclidean_norm_group.h` | group 模板 kernel 类 |
| Kernel | `example/euclidean_norm_package/op_kernel/arch35/euclidean_norm_empty.h` | 空 tensor 模板 kernel 类 |
| Kernel | `example/euclidean_norm_package/op_kernel/arch35/euclidean_norm.cpp` | 入口注册 + 全 TilingKey 实例化 |
| 数据结构 | `example/euclidean_norm_package/op_kernel/arch35/euclidean_norm_tiling_data.h` | TilingData struct |
| 数据结构 | `example/euclidean_norm_package/op_kernel/arch35/euclidean_norm_tiling_key.h` | TilingKey struct |
| Tiling | `example/euclidean_norm_package/op_host/arch35/euclidean_norm_tiling_arch35.cpp` | Host 侧 Tiling 实现 |
| Tiling | `example/euclidean_norm_package/op_host/arch35/euclidean_norm_tiling_arch35.h` | Tiling 头文件 |
| Host 注册 | `example/euclidean_norm_package/op_host/euclidean_norm_infershape.cpp` | InferShape |
| Host 注册 | `example/euclidean_norm_package/op_host/euclidean_norm_def.cpp` | 算子定义 |
| Graph 注册 | `example/euclidean_norm_package/op_graph/euclidean_norm_graph_infer.cpp` | InferDataType（IMPL_OP 注册，与 op_host 的 InferShape 分文件） |

**使用原则**：
1. 各章节的"必读"references/ 文档是**规范真值源**，示例代码是**实现参照**，两者互补
2. 当 references/ 文档的抽象描述与示例代码有出入时，以 references/ 文档为准
3. 示例代码中的 PostElewise、dtype 是算子特有的，开发新算子时替换为对应实现

---

## 范式设计总览

> **进入设计规范前必读**：[reduction-template-overview.md](references/reduction-template-overview.md)
>
> 该文档描述模板划分、TilingKey 划分、TilingData 划分及三者之间的联系。先读此文档建立全局心智模型，再进入下方逐节设计规范。

---

## 设计规范

### 1 模板划分总览

**[1.1] TilingKey 设计与 TPL_SEL** (附结构)

- **必读**：[reduction-template-overview.md](references/reduction-template-overview.md) §3.1
- **指引**：
  - TPL_SEL 拆 3 组合(base×1 + empty×1 + group×1)
  - dtype 不进 key(DTYPE_X 编译期)，axisNum 不进 key(运行时 if/else)
- **产出**：(1) ASCENDC_TPL_ARGS_DECL + ASCENDC_TPL_SEL

**[1.2] 模板划分表与实例数**

- **必读**：[reduction-template-overview.md](references/reduction-template-overview.md) §3.2
- **指引**：模板表覆盖(触发条件/dtype/shape)
- **产出**：(1) 模板与 TilingKey 对应关系表 (2) 实例数

---

### 2 TilingData 结构体

**[2.1] Base / Group TilingData struct 完整定义** (附结构)

- **必读**：[reduction-template-overview.md](references/reduction-template-overview.md) §4.1.1、[reduction-binary-group-tiling.md](references/reduction-template-binary-group/reduction-binary-group-tiling.md) §5.4、[reduction-tiling-preprocess.md](references/reduction-tiling-preprocess.md) §2
- **指引**：MAX_PATTERN_RANK 取值（A/R 类型标记以 A 结尾达 `2rseg+1` / 以 R 结尾为 `2rseg`，全 R 恒 2；标记由算子 reduce 轴语义确定，axes 任意按最坏取 `2rseg+1`，rseg 不定兜底 9）→ 决定 axisShape/axisStride 数组长度；valid/padded 双字段（`rUbFactor`/`rUbFactorAlign`）用途区分
- **产出**：(1) 完整 TilingData struct 定义（含 MAX_PATTERN_RANK）(2) valid/padded 双字段说明（**单独成段**，说明 `rUbFactor`/`rUbFactorAlign` 用途区分及为什么只有 `rUbFactorAlign` 而没有 `aUbFactorAlign`）

**[2.2] Empty TilingData 独立 struct 定义** (附结构)

- **必读**：[reduction-template-overview.md](references/reduction-template-overview.md) §4.1.2、[reduction-empty-tiling.md](references/reduction-template-empty/reduction-empty-tiling.md) §5.4
- **指引**：Empty 模板使用独立的 `ReduceEmptyTilingData`，不复用 Base/Group struct
- **产出**：(1) 完整 Empty TilingData struct 定义

---

### 3 Tiling 公共处理

**[3.1] 值依赖三处契约**（通用，框架前置）

- **必读**：[reduction-tiling-preprocess.md](references/reduction-tiling-preprocess.md) §1.2~§1.3
- **指引**：三处缺一不可，详见必读文档
- **产出**：(1) **输入输出校验约束**（标量处理、GetOptionalInputShape、可选输入空处理、OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON、INFO 日志打印每个参数）(2) op_def 声明 (3) INFERSHAPE 声明 (4) OPTILING 声明

**[3.2] 硬件参数获取** (附源码)（通用）

- **必读**：[reduction-tiling-preprocess.md](references/reduction-tiling-preprocess.md) §1.1
- **指引**：5 个硬件参数全走 platform 接口，每个返回值非 0 校验
- **产出**：(1) `GetPlatformInfo()` 完整源码 (2) 每个返回值非 0 校验
- **约束**：按必读文件抄代码，返回值校验不可省

**[3.3] 合轴预处理** (附源码)（通用，含空 tensor 短路）

- **必读**：[reduction-tiling-preprocess.md](references/reduction-tiling-preprocess.md) §2
- **指引**：axes 归一化（6 步，**空 axes 语义判定必须给出本算子的单一结论**——all reduce 还是 noop，不能罗列两种可能）→ 空 tensor 短路产生 EMPTY_A/EMPTY_R 分支；四步规则：去1→合轴→补leadingA→补R增广，**顺序不可换**
- **产出**：(1) **axes 归一化结论**（本算子的空 axes 语义判定结论 + 归一化后的轴下标集合）(2) **合轴结论**（合轴后的 axisShape/axisNum/是否 tailR/tailA）(3) `HandleEmptyTensor()` 函数签名及核心代码 (4) `DropSizeOneAxes()` / `FuseAxis()` / `PadLeadingOneA()` / `PadRIfPureA()` 四步独立函数签名及核心代码
- **约束**：以必读文件骨架为准，四步函数签名和核心代码不变

---

### 4 Kernel 公共处理

**[4.1] Kernel 入口与文件组织** (附源码)（通用）

- **必读**：[reduction-kernel-entrance.md](references/reduction-kernel-entrance.md) §1
- **指引**：Kernel 入口注册 + 模板分发（base/group/empty 三路径）+ 文件组织（base / group / empty 各一个 .h）
- **产出**：(1) Kernel 入口函数完整源码（模板签名 + 注册宏 + if constexpr 分发 base/group/empty）(2) 三文件组织说明（`<op>_base.h` 含 base / `<op>_group.h` 含 group / `<op>_empty.h` 含 empty / `<op>.cpp` 入口）(3) dtype 走 DTYPE_X 实例化代码 (4) group 模板 InitGroup/ProcessGroup 分支
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写；Reducer 非模板参数直接写死；base / group / empty 各一个 .h 文件； `DTYPE_X` 替换为实际算子的入参

**[4.1a] Tuple Reduce 循环调用模式**（通用）

- **必读**：[reduction-tuple-reduce.md](references/reduction-tuple-reduce.md)
- **指引**：N 次独立归约用 for 循环依次处理（empty/base/group 各分支内）
- **产出**：(1) Tuple Reduce 场景判定结论 (2) for 循环结构代码（preEle 按 processIdx 分发 + CopyOut 写到不同输出指针）(3) Group 模板 for 循环末尾需额外 SyncAll（base/empty 不加 SyncAll，跨过程 WAR 由 [5.10]/[6.10] 全局跳旗规则覆盖）
- **约束**：⚠ 算子为 Tuple Reduce 时，代码必须产出，不能是伪码，不得凭记忆编写；Workspace 和 UB buffer 与标准范式一致，N 次循环复用同一份，禁止在 workspace 公式或 UB 预算中乘以 N；存活节点分析按代码结构画一条流：公共步骤一次、过程独有步骤 if/else 分支并列（分割粒度 = 归约过程，同过程分叉输出在分支内合并），分支互斥取各分支峰值 max（不叠加、不按分支单独记账），其余按存活区间判定

**[4.2] Kernel 代码规则与 VF 编程速查** (附源码)（通用）

- **必读**：[reduction-binary-base-kernel-template.md](references/reduction-template-binary-base/reduction-binary-base-kernel-template.md) 文件开头（范式 kernel 约束 + kernel 规则）、[common/vf-programming-rules.md](../common/vf-programming-rules.md)
- **指引**：Kernel 代码规则 + VF 编程范式 + API 速查
- **产出**：(1) Kernel 代码规则逐条列出 (2) VF 循环范式完整代码块 (3) VF 速查表
- **约束**：⚠ Kernel 代码规则全部列出

---

### 5 空tensor模板

**[5.1] UB 划分（存活节点 + VF 融合 + Buffer 分类 + Double Buffer）**

- **必读**：[reduction-empty-dag-buffers.md](references/reduction-template-empty/reduction-empty-dag-buffers.md) §1；⛔ Tuple Reduce 场景：额外必读 [reduction-tuple-reduce.md](references/reduction-tuple-reduce.md)
- **指引**：计算图从算子公式推导（不依赖合轴）。存活节点分析（L→P）→ VF 融合分析 → 额外 buffer → Buffer 分类汇总表。Double Buffer 当前不开启
- **产出**：
  - (1) **Empty**：计算图 + L 调度 trace + L 结论表 + P_baseline + 可用 API 能力表 + VF 融合穷举分析 + VF 融合方案 + {dtype} 物理 trace + P 结论表（P_post）+ Buffer 分类汇总表 + **Double Buffer 决策结论**
  - ⛔ Tuple Reduce 场景：UB buffer 与标准范式一致，N 次循环复用同一份，禁止乘以 N；存活节点画一条流（公共步骤一次、独有步骤按 processIdx 分支并列，分割粒度 = 归约过程、禁止按 for 展开），分支互斥取 max 不叠加，其余按存活区间判定

**[5.2] Tiling UB 切分** (附源码)（empty 专用）

- **必读**：[reduction-empty-tiling.md](references/reduction-template-empty/reduction-empty-tiling.md) §0（速查）、§1~§5
- **指引**：EMPTY_A `usedCoreNum=0` + `SetBlockDim(1)` 早退；EMPTY_R aUbFactor 4 约束取 min（4KB 下界 / 优先多核 / UB 上限 / aTotal 兜底）→ postBufSize（封顶 64KB）
- **产出**：(1) **EMPTY_A/EMPTY_R UB 切分结论**（aUbFactor 4 约束公式 + postBufSize 公式 + 单 buf 64KB 上限）(2) `HandleEmptyTensor()` 完整源码（EMPTY_A 早退 + EMPTY_R 切分 + SetBlockDim/SetTilingKey + FillAndLogTilingData） + workspace
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写；严禁 `SetBlockDim(0)`

**[5.3] Tiling 多核切分**（empty 专用）

- **必读**：[reduction-empty-tiling.md](references/reduction-template-empty/reduction-empty-tiling.md) §0（速查）、§1~§5
- **指引**：EMPTY_R 大小核均衡 + 多核切分 4 约束
- **产出**：(1) **大小核均衡计算公式** (2) **多核切分 4 约束**
- **约束**：源码见 [5.2] `HandleEmptyTensor()` 完整源码

**[5.4] workspace 分配**（empty 专用）

- **必读**：[reduction-empty-tiling.md](references/reduction-template-empty/reduction-empty-tiling.md) §0（速查）、§1~§5
- **产出**：(1) workspace 结论
- **约束**：源码见 [5.2] `HandleEmptyTensor()` 完整源码

**[5.5] Kernel 计算流程( Init() / Process() 流程代码 )** (附源码)（empty）

- **必读**：[reduction-empty-kernel-template.md](references/reduction-template-empty/reduction-empty-kernel-template.md) §1.1（empty 数据流）、§2（kernel 骨架代码）
- **指引**：EMPTY_A 全核早退；EMPTY_R Init() 分配 postReducePhase buffer，Process() Duplicate 一次 + for 循环 CopyIn_post + PostElewise + CopyOut
- **产出**：(1) **empty 数据流说明**（EMPTY_A 早退流程 + EMPTY_R Duplicate→[CopyIn_post→PostElewise→]CopyOut 调用次序，buffer 个数及复用关系与 [5.1] UB 划分一致）(2) `Init()` 完整源码（**无同步版**，核内流水同步由 [5.10] 产出）(3) `Process()` 完整源码（**无同步版**，含 EMPTY_A 早退 + blockIdx→[aStart,aEnd) + Duplicate + for 循环）
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写；禁止包含 Lock/Unlock 同步，同步由 [5.10] 统一添加

**[5.6] GM 偏移计算** (附源码)（empty）

- **必读**：[reduction-empty-kernel-template.md](references/reduction-template-empty/reduction-empty-kernel-template.md) §3
- **指引**：EMPTY_A 无偏移计算；EMPTY_R blockIdx → [aStart, aEnd) 按元素数切，输出偏移 = aOff（GM 上 A 轴 dense），postReduceInput 偏移 = aOff
- **产出**：(1) **偏移计算公式**（核间 blockIdx→[aStart,aEnd) + 核内 aOff 步进 + 输出/postReduceInput GM 偏移）

**[5.7] CopyIn** (附源码)（empty）

- **必读**：[reduction-empty-kernel-template.md](references/reduction-template-empty/reduction-empty-kernel-template.md) §4
- **指引**：EMPTY_A 无 CopyIn（零计算、零 IO）；EMPTY_R 仅 `CopyIn_post`（有 post_reduce_input 时），从 GM 搬入 postReduceInput。不搬入输入数据（empty 不做 reduce，只 Duplicate 固化值 + 可选 PostElewise）
- **产出**：(1) `CopyIn_post()` 完整源码（DataCopyPad，dense 搬入）(2) **约束**：不搬入输入数据
- **约束**：⚠ has_post_reduce_input=false 时无 CopyIn，声明不适用

**[5.8] Compute 计算逻辑** (附源码) ⛔ 强制产出（empty）

- **必读**：[reduction-empty-kernel-template.md](references/reduction-template-empty/reduction-empty-kernel-template.md) §1.2（实现要点）、§5（Compute 计算逻辑）
- **指引**：EMPTY_R 按 has_post_elewise 分两路径填充输出。`has_post_elewise` / `has_post_reduce_input` 是算子级编译期常量，由算子公式决定，产出代码前必须先判定本算子的取值，只产出对应路径
- **产出**：(1) **实现要点**（见必读文档 §1.2 实现要点）(2) **empty_r_output_value 取值表 + 本算子的 has_post_elewise / has_post_reduce_input 取值结论** + 对应路径的 `DuplicateEmptyROutputVf()` 完整源码 (3) PostElewise 引用 [6.8.5] base PostElewise（有 PostElewise 时）
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写；EMPTY_R 使用 empty_r_output_value 作为 reduce 后的结果

**[5.9] CopyOut** (附源码)（empty）

- **必读**：[reduction-empty-kernel-template.md](references/reduction-template-empty/reduction-empty-kernel-template.md) §6
- **指引**：EMPTY_A 无 CopyOut；EMPTY_R `DataCopyPad` 写 GM，全 `srcStride=0, dstStride=0`（UB gap=0 + GM dense）
- **产出**：(1) `CopyOut()` 完整源码（DataCopyPad，blockLen=aLen×sizeof(D_T)，blockCount=1）(2) **约束**：不需要 `DataCopyPadExtParams`（UB→GM 方向不支持也不需 padParams）
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写

**[5.10] 流水同步方案** (附源码) ⛔ 强制产出（empty）

- **必读**：[common/sync-and-consistency.md](../common/sync-and-consistency.md) §2.2（持有法则；同步以 Mutex Lock/Unlock 段实现）、[reduction-empty-kernel-template.md](references/reduction-template-empty/reduction-empty-kernel-template.md) §7
- **指引**：以 [5.5] 产出的**无同步版 Process 源码**作为基础，用持有法则逐行标注三态 → 推导每个 buffer 的跨流水 RAW/WAR → 将各操作划入对应流水的 Lock/Unlock 段（DataCopyPad → MTE2/MTE3 段，VF/Reduce → V 段），同 id 段链式串行，产出**带同步版**。跨迭代 WAR 由段链顺序天然覆盖（无首末轮跳过技巧；⛔ Tuple Reduce：跨过程 WAR 同样由全局段链串接覆盖）。VF 链中间无 sync（同 V 流水线硬件保证串行）。EMPTY_A 无同步（所有核早退）；EMPTY_R 无跨核同步（不调用 SyncAll）
- **产出**：
  - (1) **持有法则 trace**（empty 一份，每行三态注释 `// 执行前/中/后: 持有=[...]`）。⚠ **必须检查三层 WAR**：①同一迭代内；②跨内层迭代（aOff 下一轮覆写同一 buffer）；③跨外层迭代
  - (2) **同步点推导表**（buffer × crossing 类型(RAW/WAR) × 所属流水段 × 循环内/循环间）。**每行必须标注 buffer 名称**
  - (3) **带同步的 Process 完整源码**（empty，含 Lock/Unlock 段插入，每行保留持有注释）
  - (4) **MutexID 使用说明**（申请/释放位置 × 段划分 × 用途说明）
- **约束**：⚠ 代码必须产出，不能是伪码；Lock/Unlock 必须严格成对（pipe 与 id 完全一致）；MutexID 通过 `AllocMutexID/ReleaseMutexID` 申请释放，不可硬编码；同 id 不得嵌套；禁止凭直觉插入未由持有法则推导出的同步段；⚠ **跨迭代 WAR 必须验证 buffer 身份**；⚠ **三层 WAR 都必须检查**

---

### 6 Base 模板

**[6.1] UB 划分（存活节点 + VF 融合 + Buffer 分类 + Double Buffer）**

- **必读**：[reduction-binary-base-dag-buffers.md](references/reduction-template-binary-base/reduction-binary-base-dag-buffers.md) §1；⛔ Tuple Reduce 场景：额外必读 [reduction-tuple-reduce.md](references/reduction-tuple-reduce.md)
- **指引**：计算图从算子公式推导（不依赖合轴）。存活节点分析（L→P）→ VF 融合分析 → 额外 buffer → Buffer 分类汇总表。Double Buffer 当前不开启
- **产出**：
  - (1) **Base**：计算图 + L 调度 trace + L 结论表 + P_baseline + 可用 API 能力表 + VF 融合穷举分析 + VF 融合方案 + {dtype} 物理 trace + P 结论表（P_pre/P_post）+ isReuseSource + sharedTmpBuffer + Buffer 分类汇总表 + **Double Buffer 决策结论**（当前不开启）+ **preReduceResult 校验**（完全独立 buffer，不与 preReducePhase 内任何节点复用）
  - ⛔ Tuple Reduce 场景：UB buffer 与标准范式一致，N 次循环复用同一份，禁止乘以 N；存活节点画一条流（公共步骤一次、独有步骤按 processIdx 分支并列，分割粒度 = 归约过程、禁止按 for 展开），分支互斥取 max 不叠加，其余按存活区间判定

**[6.2] Tiling UB 切分** (附源码)（base）

- **必读**：[reduction-binary-base-tiling.md](references/reduction-template-binary-base/reduction-binary-base-tiling.md) §0（速查）、§1（含前置依赖 + §1.1 切分原理 + §1.2 切分公式）
- **指引**：定 A 切分（`ComputeAUbFactor`）→ 反解 R 切分（`ComputeRUbFactor`）→ R 全载扩 A（`ExpandAIfRFullyLoaded`）
- **产出**：(1) **UB 切分结论**（aSplitIdx/rSplitIdx 的选取规则 + aUbFactor/rUbFactor/rUbFactorAlign/innerAProdAlign/innerRProdAlign/aUnit 的计算公式）(2) `ComputeAUbFactor()` 完整源码 (3) `ComputeRUbFactor()` 完整源码 (4) `ExpandAIfRFullyLoaded()` 完整源码
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写；⚠ 无条件 FloorAlign 会误判 TILING_FAIL；⚠ R<blockSize 全载时跳过 FloorAlign；⛔ dtype 参数适用边界：`bsElem`（元素数取整）用 `sizeof(D_T)`、`maxDtypeSize`（字节容量）用 `max(sizeof(D_T), sizeof(float))`，禁止混用

**[6.3] Tiling 多核切分** (附源码)（base A方向）

- **必读**：[reduction-binary-base-tiling.md](references/reduction-template-binary-base/reduction-binary-base-tiling.md) §0（速查）、§2（含 §2.1 切分原理 + §2.2 切分公式）
- **指引**：A 方向 fused aLoop 分核 + 大小核均衡
- **产出**：(1) **多核切分结论**（aLoopCntTotal/aSplitChunkCnt/大小核/usedCoreNum 的计算公式）(2) `ComputeFusedALoopSplit()` 完整源码 (3) `ComputeRLoopCnt()` 完整源码
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写

**[6.4] workspace 分配** (附源码)（base）

- **必读**：[reduction-binary-base-tiling.md](references/reduction-template-binary-base/reduction-binary-base-tiling.md) §0（速查）、§5（含 §5.1 UB 大小计算 + §5.2 Workspace 分配）；Tuple Reduce 场景额外必读 [reduction-tuple-reduce.md](references/reduction-tuple-reduce.md)
- **指引**：base `ws[0] = sysWorkspaceSize`
- **产出**：(1) `SetWorkspaceSize()` 完整源码（base）(2) ⛔ Tuple Reduce 场景：N 次循环复用同一份 workspace，禁止乘 N
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写；必须设置 workspace，即使 base 不使用也要显式设置 sysWorkspaceSize

**[6.5] Kernel 计算流程( Init() / Process() 流程代码 )** (附源码) ⛔ 强制产出（base）

- **必读**：[reduction-binary-base-kernel-template.md](references/reduction-template-binary-base/reduction-binary-base-kernel-template.md) §1.1（base 数据流）、§2
- **指引**：保持必读文件的类结构/调用次序完全不变，仅替换 Reducer/PostElewise/dtype 占位符；⛔ Tuple Reduce 场景：Process 新增 processIdx 形参、内部按 processIdx 分发（见 [reduction-tuple-reduce.md](references/reduction-tuple-reduce.md) 与 [4.1a]），其余类结构不变
- **产出**：(1) **数据流说明**（Phase A/B 配对逻辑 + 二分树在 Process 中的位置 + buffer 个数及复用关系与 [6.1] UB 划分一致）(2) `Init()` 完整源码 (3) `Process()` 完整源码（**无同步版**，核内流水同步由 [6.10] 产出）(4) `ProcessOneRChunk()` 完整源码（无同步版）
- **约束**：⛔ 代码必须产出，不能是伪码，不得凭记忆编写；⛔ 严禁用自然语言描述替代源码块，禁止 `// ...` `/* ... */` 省略；Reducer/PostElewise/dtype 三位占位符是唯三允许修改的地方；Init()、Process() 必须有各自独立完整的源码块；禁止包含 Lock/Unlock 同步，同步由 [6.10] 统一添加

**[6.6] 外层循环映射与 GM 偏移计算** (附源码)（base）

- **必读**：[reduction-binary-base-tiling.md](references/reduction-template-binary-base/reduction-binary-base-tiling.md) §2.2、[reduction-binary-base-kernel-template.md](references/reduction-template-binary-base/reduction-binary-base-kernel-template.md) §3
- **指引**：blockIdx→aLoop 大小核映射 + UnravelALoop 解码 + R 外层循环调度（UnravelR + Phase A/B）+ GM 偏移（输入偏移 = 外层索引 + aSplit chunk + rSplit chunk；输出偏移 = dense）
- **产出**：(1) **偏移计算公式**（输入 GM 偏移 = 外层索引 + aSplit chunk + rSplit chunk；输出偏移 = dense）(2) 大小核映射源码 (3) `UnravelALoop()` 完整源码 (4) `UnravelR()` 完整源码 (5) R 外层循环 + Phase A/B 调度源码 (6) 输入 GM 偏移计算代码 (7) 输出 GM 偏移计算代码
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写

**[6.7] CopyIn** (附源码) ⛔ 强制产出（base）

- **必读**：[reduction-binary-base-kernel-template.md](references/reduction-template-binary-base/reduction-binary-base-kernel-template.md) §4（K 分级 + BuildUBAxes + UBAxisDesc + DoCopyInTile 完整骨架）、[common/datacopypad-rules.md](../common/datacopypad-rules.md)
- **指引**：BuildUBAxes 构建 UB 轴描述 → DoCopyInTile 执行数据搬运（DataCopyPad + LoopMode）。
- **产出**：(1) **CopyIn 原理**（UB 轴描述构建逻辑 + DataCopyPad 参数推导 + K 分级）(2) **搬入 UB 后的数据排布**（tail-R/tail-A 各场景下 UB 内 `[A_bundle, R_bundle]` 或 `[R_bundle, A_bundle]` 的轴顺序及 padded 对齐）(3) **pad 总结表**（BurstPad/ExtensionPad 的 tail-R/tail-A 置零规则）(4) `BuildUBAxes()` 完整源码 (5) `DoCopyInTile()` 完整源码 (6) K 分级说明（K=2 / K∈[3,4] / K≥5 三档）
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写

**[6.8] Compute 计算逻辑** (附源码) ⛔ 强制产出（base）

- **必读**：[reduction-binary-base-kernel-template.md](references/reduction-template-binary-base/reduction-binary-base-kernel-template.md) §5
- **指引**：Compute 阶段从 PreElewise 到 PostElewise 的完整计算链，含 5 个子步骤。子步骤按计算次序排列，每个子步骤独立产出源码
- **产出**：见 [6.8.1]~[6.8.5]

**[6.8.1] PreElewise** (附源码)（base）

- **必读**：[reduction-binary-base-kernel-template.md](references/reduction-template-binary-base/reduction-binary-base-kernel-template.md) §5.2、[common/cast-rules.md](../common/cast-rules.md)
- **指引**：CopyIn 后、Reduce 前的 elementwise 阶段。regbase cast 指令（LoadAlign + Cast）
- **产出**：(1) PreElewise VF 完整源码（b16→fp32 LoadAlign + Cast + 算子专属 elementwise，同一 asc_vf_call）
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写

**[6.8.2] 二分缓存树算法** (附源码)（base）

- **必读**：[reduction-binary-base-kernel-template.md](references/reduction-template-binary-base/reduction-binary-base-kernel-template.md) §5.1.1、[reduction-binary-sum.md](references/reduction-binary-sum.md) §六
- **指引**：所有 reduction 算子统一使用二分缓存树（GetCacheID + DoCaching + FindNearestPower2 + CalLog2）
- **产出**：(1) **二分缓存树原理**（二分树设计目的 + GetCacheID 位运算原理 + DoCaching 吸收逻辑）(2) **cacheBuf 布局结论**（levelStride 对齐约束 + 树根位置公式 + 不需预清零声明）(3) `GetCacheID()` 完整源码 (4) `DoCaching()` 完整源码 (5) `FindNearestPower2()` 完整源码 (6) `CalLog2()` 完整源码 (7) kernel 侧变量与 reduction-binary-sum.md 的映射表
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写

**[6.8.3] pad 清零与 garbage 隔离** (附源码)（base）

- **必读**：[reduction-binary-base-kernel-template.md](references/reduction-template-binary-base/reduction-binary-base-kernel-template.md) §5.1.2 + §5.1.3
- **指引**：搬运不指定 paddingValue（isPad=false，pad 为脏数据），R 方向 pad 恒清零（tail-R 行内 BurstPad + partial chunk ExtensionPad）+ A 方向 garbage 不需主动清零（A 各 lane 独立，天然隔离）
- **产出**：
  - (1) **pad_value 取值表 + R 方向 pad 清零决策表**（tail 类型 × rChunk × rSplitIdx → 调用哪些 VF）
  - (2) `ClearChunkExtensionVf()` 完整源码（tail-R / tail-A 两个分支都必须有实现，禁止留空，未清零会经 PreElewise 污染 ReduceSum 结果）
  - (3) `ClearInnerBurstTailPadVf` 触发条件表（tail-R 且 burst 尾轴非对齐恒触发 + rSplit<LastR / rSplit==LastR 组合）
  - (4) **按 (3) 触发条件表结论决定是否产出**。`ClearInnerBurstTailPadVf()` 完整源码（tail-R 路径，含 StoreAlign 起点 FloorAlign + mask 三段）
  - (5) `MergeTmpBufVf()` 完整源码
  - (6) 操作位置=preReduceResult(fp32)，时机=Reduce 之前/PreElewise 之后
  - (7) A 方向 garbage 隔离机制说明 + "禁止额外 A 方向清零"声明
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写；禁止额外 A 方向清零操作

**[6.8.4] Reduce 高阶api调用** (附源码)（base）

- **必读**：[reduction-binary-base-kernel-template.md](references/reduction-template-binary-base/reduction-binary-base-kernel-template.md) §5.1.4、[reduction-highlevel-api.md](references/reduction-highlevel-api.md)
- **指引**：调用 AscendC::ReduceXxx API，根据 tail 类型选择 Pattern（AR/RA）
- **产出**：(1) **dst 写入位置公式**（cacheBuf[cacheID × levelStride]，levelStride = CeilAlign(laneA, 8)，laneA = aUbFactor × innerAProdAlign）(2) ReduceXxx 调用代码（Pattern + srcShape + sharedTmpBuffer=preReduceResultTail）。srcShape 即 [6.7] 产出的 UB 内数据排布维度 (3) isReuseSource 取值（**本算子的单一取值 + 理由**）(4) ReduceXxx 6 条硬约束表 (5) **精度说明**：ReduceSum 内部采用多路并行二叉树折叠，累加精度高
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写；srcInnerPad=true（无论 tail-R 或 tail-A）；src/dst 不重叠

**[6.8.5] PostElewise** (附源码)（base）

- **必读**：[reduction-binary-base-kernel-template.md](references/reduction-template-binary-base/reduction-binary-base-kernel-template.md) §5.1.5（PostElewise 伪码）、[common/cast-rules.md](../common/cast-rules.md)
- **指引**：PostElewise 将算子专属 elewise + 缩位 Cast 合并在同一 VF 内完成
- **产出**：(1) **PostElewise 原理**（VF 融合逻辑 + Cast 缩位规则 + padded 整行处理 vs valid 拷出的区别）(2) `PostElewise()` 完整源码（含 PostElewise + 缩位 Cast，同一 asc_vf_call）
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写

**[6.9] CopyOut 三条路径** (附源码) ⛔ 强制产出（base）

- **必读**：[reduction-binary-base-kernel-template.md](references/reduction-template-binary-base/reduction-binary-base-kernel-template.md) §6（含 §6.1 UB 输出布局 + §6.2 CopyOut 指令 + 4典型例）
- **指引**：CopyOut 根据 tail 类型和 aSplit 位置选择三条路径
- **产出**：(1) **UB 输出布局表**（Reduce 后 UB 上 A_bundle 排布：tail-R dense / tail-A+LastA 单根 / tail-A+!LastA 多 burst）(2) `CopyOut()` 完整源码（三条路径）(3) 路径决策表 (4) 4 个典型例代码 (5) "禁路径3套tail-R"声明
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写；禁止用"CopyOut 用 DataCopyPad"之类的一句话概括

**[6.10] 流水同步方案** (附源码) ⛔ 强制产出（base）

- **必读**：[common/sync-and-consistency.md](../common/sync-and-consistency.md) §2.2（持有法则；同步以 Mutex Lock/Unlock 段实现）、[reduction-binary-base-kernel-template.md](references/reduction-template-binary-base/reduction-binary-base-kernel-template.md) §7
- **指引**：以 [6.5] 产出的**无同步版 Process 源码**作为基础，用持有法则逐行标注三态 → 推导每个 buffer 的跨流水 RAW/WAR → 将各操作划入对应流水的 Lock/Unlock 段（DataCopyPad → MTE2/MTE3 段，VF/Reduce → V 段），同 id 段链式串行，产出**带同步版**。跨迭代 WAR 由段链顺序天然覆盖（无首末轮跳过技巧；⛔ Tuple Reduce：跨过程 WAR 同样由全局段链串接覆盖）。VF 链中间无 sync（同 V 流水线硬件保证串行）
- **产出**：
  - (1) **持有法则 trace**（base 一份，每行三态注释 `// 执行前/中/后: 持有=[...]`）。⚠ **必须检查三层 WAR**：①同一迭代内（如 Phase A 主块→尾块 preInBuf 覆写）；②跨内层迭代（rIdx 下一轮覆写同一 buffer）；③跨外层迭代（aLoopIdx 下一轮覆写同一 buffer）。同一 buffer 被上轮 V 读后下轮 MTE2 覆写，构成 V→MTE2 WAR，需要反向同步
  - (2) **同步点推导表**（buffer × crossing 类型(RAW/WAR) × 所属流水段 × 循环内/循环间）。**每行必须标注 buffer 名称**，确认跨迭代 crossing 的两个操作作用在同一个 buffer 上才构成 WAR；不同 buffer 不构成 WAR，不得插入同步段
  - (3) **带同步的 Process 完整源码**（base，含 Lock/Unlock 段插入，每行保留持有注释）
  - (4) **MutexID 使用说明**（申请/释放位置 × 段划分 × 用途说明）。无对应 WAR 的流水段标注"不需要"，并附理由（如"outBuf 与 preInBuf 不复用，跨迭代无 WAR"）
- **约束**：⚠ 代码必须产出，不能是伪码；Lock/Unlock 必须严格成对（pipe 与 id 完全一致）；MutexID 通过 `AllocMutexID/ReleaseMutexID` 申请释放，不可硬编码；同 id 不得嵌套；禁止凭直觉插入未由持有法则推导出的同步段；⚠ **跨迭代 WAR 必须验证 buffer 身份**——上轮消费者读的 buffer 与下轮生产者写的 buffer 是同一个物理 buffer 才构成 WAR，段链须保证相应段序；⚠ **三层 WAR 都必须检查**：①同一迭代内（Phase A 主块→尾块）；②跨内层循环（rIdx）；③跨外层循环（aLoopIdx）。⚠ 注意各循环内 `continue`/`break` 可能导致的 Lock/Unlock 不成对，必须逐路径检查配对完整性

---

### 7 Group 模板

**[7.1] UB 划分（存活节点 + VF 融合 + Buffer 分类 + Double Buffer）**

- **必读**：[reduction-binary-base-dag-buffers.md](references/reduction-template-binary-base/reduction-binary-base-dag-buffers.md) §1、[reduction-binary-group-dag-buffers.md](references/reduction-template-binary-group/reduction-binary-group-dag-buffers.md) §1（Phase 2 差异）；⛔ Tuple Reduce 场景：额外必读 [reduction-tuple-reduce.md](references/reduction-tuple-reduce.md)
- **指引**：Phase 1 preReducePhase / postReducePhase 的 VF 融合分析与 Base 完全一致。Phase 2 差异：Phase 2 CopyIn 复用 Phase 1 的 preReduceResult 物理槽 + cacheBuf 写[0] + workspace GM buffer 新增
- **产出**：
  - (1) 复用 [6.1] 全部产出（计算图 + L 调度 trace + L 结论表 + P_baseline + 可用 API 能力表 + VF 融合穷举分析 + VF 融合方案 + {dtype} 物理 trace + P 结论表 + Buffer 分类汇总表 + Double Buffer 决策结论 + preReduceResult 校验）
  - (2) **Group Phase 2 差异**（Phase 2 CopyIn 复用 Phase 1 的 preReduceResult 物理槽 + cacheBuf 写[0] + workspace GM buffer `[rGroupCnt, aTotal]` fp32 dense），见 group dag-buffers §1.1~§1.3
  - ⛔ Tuple Reduce 场景：UB buffer 与标准范式一致，N 次循环复用同一份，禁止乘以 N；存活节点画一条流（公共步骤一次、独有步骤按 processIdx 分支并列，分割粒度 = 归约过程、禁止按 for 展开），分支互斥取 max 不叠加，其余按存活区间判定

**[7.2] Tiling UB 切分**

- **必读**：[reduction-binary-base-tiling.md](references/reduction-template-binary-base/reduction-binary-base-tiling.md) §0（速查）、§1
- **指引**：Group 模板 Phase 1 完全复用 Base 的 UB 切分三步算法。Phase 2 复用 Phase 1 UB 物理槽，不重新计算
- **产出**：复用 [6.2] 全部产出

**[7.3] Tiling 多核切分** (附源码)（group 2D分核 + Group 触发判定）

- **必读**：[reduction-binary-base-tiling.md](references/reduction-template-binary-base/reduction-binary-base-tiling.md) §0（速查）、§2（含 §2.1 切分原理 + §2.2 切分公式）、[reduction-binary-group-tiling.md](references/reduction-template-binary-group/reduction-binary-group-tiling.md) §0（速查）、§2
- **指引**：Group 触发判定 `ShouldUseGroup()`（A 用不满核且 R 有并行度时触发）→ 2D 分核 `ComputeGroupSplit()` + `SetScheduleMode(1)`
- **产出**：(1) **Group 多核切分结论**（rGroupCnt 的计算公式 + aPerCore=1 恒成立）(2) `ShouldUseGroup()` 完整源码 (3) `ComputeGroupSplit()` 完整源码 (4) `SetScheduleMode(1)` 调用代码及原因
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写

**[7.4] workspace 分配** (附源码)（group）

- **必读**：[reduction-binary-base-tiling.md](references/reduction-template-binary-base/reduction-binary-base-tiling.md) §0（速查）、§5（含 §5.1 UB 大小计算 + §5.2 Workspace 分配）、[reduction-binary-group-tiling.md](references/reduction-template-binary-group/reduction-binary-group-tiling.md) §0（速查）、§5；Tuple Reduce 场景额外必读 [reduction-tuple-reduce.md](references/reduction-tuple-reduce.md)
- **指引**：group `ws[0] = sysWorkspaceSize + usrWorkspaceBytes`，usrWorkspaceBytes = `rGroupCnt × aTotal × sizeof(fp32)`，布局 `[rGroupCnt, aTotal]` fp32 dense 行优先
- **产出**：(1) `SetWorkspaceSize()` 完整源码（group）(2) group workspace 布局说明（`[rGroupCnt, aTotal]` fp32 dense 行优先）(3) ⛔ Tuple Reduce 场景：N 次循环复用同一份 workspace，布局仍为 `[rGroupCnt, aTotal]`，禁止乘 N
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写；必须设置 workspace

**[7.5] Group Phase 1：部分复用 Base 写 Workspace** (附源码) ⛔ 强制产出（group）

- **必读**：[reduction-binary-group-kernel-template.md](references/reduction-template-binary-group/reduction-binary-group-kernel-template.md) §1.1（group 数据流）、§2~§7；⛔ Tuple Reduce 场景：额外必读 [reduction-tuple-reduce.md](references/reduction-tuple-reduce.md)、[common/sync-and-consistency.md](../common/sync-and-consistency.md) §2.2
- **指引**：Phase 1 部分复用 Base 模板，差异在二分缓存树局部化 + CopyOut 改写 workspace + PostElewise 跳过。复用对照：

| 步骤 | 复用 base | 差异 |
|------|----------|------|
| Process 骨架 | 6.5 | CopyOut→workspace, PostElewise 跳过 |
| 外层循环+GM偏移 | 6.6 | 同 base |
| CopyIn | 6.7 | 同 base |
| PreElewise | 6.8.1 | 同 base |
| 二分缓存树 | 6.8.2 | rCount 局部化 + cacheBuf 树根用 localRoot |
| pad 清零+garbage | 6.8.3 | 同 base |
| Reduce API | 6.8.4 | 同 base（局部 rCount） |
| PostElewise | 6.8.5 | 跳过 |
| CopyOut | 6.9 | 改为写 workspace fp32 |

- **产出**：(1) **group Phase 1 数据流说明**（须显式说明：每核 rCount 个原始 R chunk 经核内二分树累加为 1 个 partial reduce 结果，写 workspace 第 rChunkIdx 行（核的分组下标 ∈ [0, rGroupCnt)）；workspace 行数 = rGroupCnt）(2) **2D 坐标计算公式**（blockIdx → aChunkIdx/rChunkIdx + R 方向大小核式均匀分配 rStart/rEnd/rCount（无空组）+ aPerCore=1 退化）(3) `Phase1Process()` 完整源码（含 2D 坐标计算 + CopyOut 到 workspace + SyncAll，**无同步版**）
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写；⛔ R 方向必须大小核式均匀分配（rGroupCnt ≤ rLoopCntTotal 恒成立，见 group-kernel-template §3.1），禁止 CeilDiv 截断式分配；⛔ Phase 1 CopyOut 参数与 normal CopyOut 三条路径完全一致，仅 sizeof(D_T) → sizeof(float)：
  - tail-R：blockLen = aLen × innerAProd_ × sizeof(float)，blockCount = 1，srcStride = 0
  - tail-A + aSplit==lastA：blockLen = aLen × sizeof(float)，blockCount = 1，srcStride = 0
  - tail-A + aSplit!=lastA：blockLen = lastASize × sizeof(float)，blockCount = aLen × innerAProd_ / lastASize，srcStride = (lastASizeAlign - lastASize) × sizeof(float) / 32（0 或 1）
  - dstStride = 0（GM dense）
  - 偏移 = wsGm_[rChunkIdx * aTotal_ + chunkOutOff]
  - src = cache 树根（fp32），必须用本核 rCount 局部计算 rootOff，禁止用全局 cacheCount_

**[7.6] Group Phase 2：RA Mini-Kernel** (附源码) ⛔ 强制产出（group）

- **必读**：[reduction-binary-group-kernel-template.md](references/reduction-template-binary-group/reduction-binary-group-kernel-template.md) §1.1（group 数据流）、§2~§7、[common/sync-and-consistency.md](../common/sync-and-consistency.md) §2.2
- **指引**：Phase 2 是 RA mini-kernel，R 全载于 UB，kernel 侧反推 aUbFactorP2。复用对照：

| 步骤 | 复用 base | 差异 |
|------|----------|------|
| Process 骨架 | 不复用 | 独立 RA mini-kernel，R 全载单 chunk |
| 外层循环+GM偏移 | 不复用 | 从 workspace shape 单独计算切分 |
| CopyIn | 6.7 | 从 workspace 搬入 fp32（参数不同） |
| PreElewise | 6.8.1 | 无 |
| 二分缓存树 | 6.8.2 | 无（cacheBuf 写[0]） |
| pad 清零+garbage | 6.8.3 | 无 |
| Reduce API | 6.8.4 | RA Pattern，R 全载单 chunk，dst 写 cacheBuf[0] |
| PostElewise | 6.8.5 | 同 base |
| CopyOut | 6.9 | 走 base 三条路径中的 tail-A+LastA 一条（blockCount=1） |

- **产出**：(1) **group Phase 2 数据流说明** (2) **aUbFactorP2 反推公式 + a_len / a_len_ub 双字段**（preBufSize/rGroupCnt → aUbFactorP2 + postBufSize/aTotal 钳制 + 大小核现算公式；aUbFactorP2 切分因子≤aTotal，a_off=aSplitChunkIdx×aUbFactorP2，a_len=min(aUbFactorP2, aTotal−a_off) valid 当前块长，a_len_ub=CeilAlign(a_len, BS_FP32) padded UB 行步长）(3) `Phase2Process()` 完整源码（含 aUbFactorP2 反推 + 大小核均衡 + CopyIn workspace + Reduce RA + PostElewise + CopyOut，**无同步版**）
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写；⛔ Phase 2 CopyIn 参数：
  - src = wsGm_[a_off]（workspace，fp32）
  - blockLen = a_len × sizeof(float)（valid）
  - blockCount = rGroupCnt_（全部 R 分组，不是 1）
  - srcStride = aTotal_ × sizeof(float) - blockLen（workspace 行间 gap，不是 0）
  - dstStride = 0（块间 gap=0，HW 自动按 CeilAlign(blockLen, 32B) 放置下一个块）
  - padParams：isPad=false，rightPad=0（行宽对齐由 HW 自动保证，A 方向 pad 不进 reduce 结果）
  ⛔ Phase 2 ReduceXxx 参数：
  - Pattern = `Pattern::Reduce::RA`（tail-A）
  - dst = cacheBuf[0]（R 全载单 chunk，无二分树层级）
  - src = preReduceResult（CopyIn 搬入的 UB buffer）
  - sharedTmpBuffer = preReduceResultTail
  - srcShape = `{rGroupCnt, a_len_ub}`    ← ⚠ **必须用 a_len_ub（padded）**：ReduceSum 按 a_len_ub 行步长读
  - srcInnerPad = true
  - isReuseSource = true

**[7.7] 流水同步方案** (附源码) ⛔ 强制产出（group）

- **必读**：[common/sync-and-consistency.md](../common/sync-and-consistency.md) §2.2（持有法则；同步以 Mutex Lock/Unlock 段实现）、[reduction-binary-group-kernel-template.md](references/reduction-template-binary-group/reduction-binary-group-kernel-template.md) §7
- **指引**：以 [7.5]/[7.6] 产出的**无同步版 Phase1Process()/Phase2Process() 源码**作为基础，分别用持有法则逐行标注三态 → 推导每个 buffer 的跨流水 RAW/WAR → 将各操作划入对应流水的 Lock/Unlock 段（DataCopyPad → MTE2/MTE3 段，VF/Reduce → V 段），同 id 段链式串行，产出**带同步版**。Group Phase 1→Phase 2 用 SyncAll（已有，不重复）。跨迭代 WAR 由段链顺序天然覆盖（无首末轮跳过技巧）。VF 链中间无 sync（同 V 流水线硬件保证串行）
- **产出**：
  - (1) **持有法则 trace**（Phase 1 一份 + Phase 2 一份，每行三态注释 `// 执行前/中/后: 持有=[...]`）。⚠ **必须检查三层 WAR**：①同一迭代内；②跨内层迭代（rIdx 下一轮覆写同一 buffer）；③跨外层迭代（aLoopIdx 下一轮覆写同一 buffer）
  - (2) **同步点推导表**（buffer × crossing 类型(RAW/WAR) × 所属流水段 × 循环内/循环间）。**每行必须标注 buffer 名称**
  - (3) **带同步的 Phase1Process() 完整源码** + **带同步的 Phase2Process() 完整源码**（含 Lock/Unlock 段插入，每行保留持有注释）
  - (4) **MutexID 使用说明**（申请/释放位置 × 段划分 × 用途，Phase 1 + Phase 2 合并）。Phase 1 与 Phase 2 共用同一 MutexID：SyncAll 含 fence 语义，Phase 1 末段 Unlock 后 Phase 2 首段 Lock 自然衔接
- **约束**：⚠ 代码必须产出，不能是伪码；Lock/Unlock 必须严格成对（pipe 与 id 完全一致）；MutexID 通过 `AllocMutexID/ReleaseMutexID` 申请释放，不可硬编码；同 id 不得嵌套；禁止凭直觉插入未由持有法则推导出的同步段；⚠ **跨迭代 WAR 必须验证 buffer 身份**——上轮消费者读的 buffer 与下轮生产者写的 buffer 是同一个物理 buffer 才构成 WAR，段链须保证相应段序；⚠ **三层 WAR 都必须检查**：①同一迭代内；②跨内层循环（rIdx）；③跨外层循环（aLoopIdx）。⚠ 注意各循环内 `continue`/`break` 可能导致的 Lock/Unlock 不成对，必须逐路径检查配对完整性

---

### 8 Tiling 合并后代码架构

**[8.1] 填 TilingData + 设 TilingKey + 工程约束** (附源码)（base/group + empty 分述）

- **必读**：[reduction-tiling-preprocess.md](references/reduction-tiling-preprocess.md) §1.1、[reduction-template-overview.md](references/reduction-template-overview.md) §3.1/§3.2/§4.1.1、[reduction-binary-base-tiling.md](references/reduction-template-binary-base/reduction-binary-base-tiling.md) §0（速查）、§5、[reduction-empty-tiling.md](references/reduction-template-empty/reduction-empty-tiling.md) §5.4
- **指引**：base/group 填 `ReduceGenericTilingData` 全字段 + OP_LOGI 全量打印 + SetBlockDim/SetTilingKey；empty 填 `ReduceEmptyTilingData`（见 [5.2]）
- **产出**：(1) **Buffer 大小公式**（maxDtypeSize、aUnit、preBufSize、postBufSize、cacheBufUbSize + UB 预算不等式）(2) base/group `FillAndLogTilingData()` 完整源码（全字段赋值 + OP_LOGI 打印全量 TilingData 字段）(3) SetBlockDim/SetTilingKey 调用代码 (4) 工程约束逐条列出
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写；⚠ TilingData 所有字段必须通过 OP_LOGI 打印；axisShape/axisStride 等数组字段可拼接为字符串打印

**[8.2] Tiling 合并后代码架构** (附源码)（通用）

- **必读**：[reduction-binary-base-tiling.md](references/reduction-template-binary-base/reduction-binary-base-tiling.md) §4（整体切分伪码）
- **指引**：将 [3.1]~[3.3]、[5.2]~[5.4]、[6.2]~[6.4]、[7.3]~[7.4]、[8.1] 各节产出的函数按必读文档 §4 的调用次序组装为 `TilingFunc` 主函数。empty 短路在合轴之前，非空走完整 tiling 流程
- **产出**：Tiling 完整源码
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写

---

### 9 API 映射及验证

**[9.1] API 映射总表**

- **必读**：[reduction-binary-base-kernel-template.md](references/reduction-template-binary-base/reduction-binary-base-kernel-template.md) §4（CopyIn）、§5（Compute）、§6（CopyOut）、[reduction-empty-kernel-template.md](references/reduction-template-empty/reduction-empty-kernel-template.md) §5.1（EMPTY_R Duplicate）
- **指引**：计算步骤 → Ascend C API → 关键参数 → 约束
- **产出**：API 映射总表（覆盖 CopyIn / Cast 扩位 / Reduce / partial 清零 / DoCaching / 行 pad 清零 / Cast 缩位 / CopyOut / EMPTY_R Duplicate）

**[9.2] ReduceXxx 约束 + Pattern/srcShape + isReuseSource** (附结构)

- **必读**：[reduction-highlevel-api.md](references/reduction-highlevel-api.md)、[reduction-binary-base-kernel-template.md](references/reduction-template-binary-base/reduction-binary-base-kernel-template.md) §5.1.4
- **指引**：ReduceXxx 调用约束 + Pattern/srcShape 选择 + isReuseSource 取值（sharedTmpBuffer 固定传 preReduceResultTail）
- **产出**：(1) ReduceXxx 完整签名 (2) AR/RA srcShape 计算公式 (3) ReduceXxx 6 条硬约束表 (4) isReuseSource 取值（引用 [6.8.4] 的结论）(5) **buffer 份数固定 3 份的理由**（preReduceResultTail 兼作 sharedTmpBuffer，Reduce 时已空闲，复用不增加 UB 开销）
- **约束**：⚠ srcInnerPad=true（无论 tail-R 或 tail-A）；src/dst 不重叠

**[9.3] DataCopyPad 参数/陷阱 + Loop 迭代重叠约束** (附结构)

- **必读**：[common/datacopypad-rules.md](../common/datacopypad-rules.md)
- **指引**：DataCopyPad 参数结构 + padding 陷阱 + stride 单位 + Loop 迭代重叠约束
- **产出**：(1) DataCopyExtParams 完整结构 (2) LoopModeParams 完整结构 (3) 9 条陷阱表 (4) stride 单位表 (5) Loop 迭代重叠约束公式
- **约束**：⚠ stride 单位区分 GM 侧(byte) 和 UB 侧(datablock=32B)，方向决定 src/dst 对应哪一侧（MTE2/MTE3）

**[9.4] Cast 规则** (附结构)

- **必读**：[common/cast-rules.md](../common/cast-rules.md)
- **指引**：扩位 b16→fp32 + 缩位 fp32→b16 的 CastTrait 配置 + DIST 配对
- **产出**：(1) 扩位 CastTrait 完整结构 (2) 缩位 CastTrait 完整结构 (3) DIST_UNPACK_B16 / PACK_B32 配对说明 (4) 8 条编译错误对照表 (5) int8/uint8 ↔ fp32 两步转换模式

**[9.5] API 验证总表 + 未验证项**

- **必读**：[regbase_api_whitelist.md](../../api/regbase_api_whitelist.md)
- **指引**：记录 API 的可信来源、验证状态
- **产出**：(1) API 验证总表（API 名称/白名单路径/验证状态/备注）(2) 待验证项清单
- **约束**：⚠ 验证状态必须准确，禁止把"待验证"写成"已验证"

---

### 10 内存管理

各模板数据流详见各模板 Kernel 计算流程章节（[5.5] empty、[6.5] base、[7.5]/[7.6] group）。

UB 划分预分析 + Buffer 分类详见 [6.1]（base）、[7.1]（group）、[5.1]（empty）。Empty postBufSize + workspace（ws[0]=sysWorkspaceSize）详见 [5.2]。Group Workspace 布局详见 [7.4]。Buffer 大小公式详见 [8.1]。

---

### 11 UB 容量验证

**[11.1] UB 预算不等式** (附公式)

- **必读**：[reduction-binary-base-tiling.md](references/reduction-template-binary-base/reduction-binary-base-tiling.md) §5.1
- **指引**：`(P_pre + P_pre_ext) × preBufSize + P_post × postBufSize ≤ ubAvailable`，其中 `ubAvailable = ubSize − cacheBufUbSize`（先扣固定 16KB）。P_pre/P_pre_ext(固定 1)/P_post 由 [6.1] Buffer 分类汇总表传入。
- **产出**：(1) 预算不等式公式块 (2) 按实际芯片代入值 (3) 验证结论

**[11.2] cacheBuf 硬约束** (附公式)

- **必读**：[reduction-binary-base-tiling.md](references/reduction-template-binary-base/reduction-binary-base-tiling.md) §1.2（Step 3 钳制）、§5.1（容量结论）
- **指引**：所有 reduction 算子：cacheCount×CeilAlign(aUbFactor×innerAProdAlign, 8)×sizeof(float)≤cacheBufUbSize(16KB)，cacheCount=CalLog2(FindNearestPower2(rLoopCntTotal))+1——fp32/All Reduce 恒满足；b16/bf16 需 rLoopCntTotal ≥ 2^32 才违反，实际不可达，不设校验。R 全驻（cacheCount=1）时由 base-tiling.md §1.2 Step 3 钳制 aUnit≤4096
- **产出**：公式 + 校验结论（本算子是否处于触发窗口 + 校验代码）

---

### 12 确定性保证

- **必读**：spec.yaml
- **指引**：固定 aSplit/rSplit + 固定二分缓存树累加顺序 → bitwise 可复现
- **产出**：确定性策略说明
