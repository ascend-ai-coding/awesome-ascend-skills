# Broadcast 类算子 DESIGN.md 生成规范

本文件是 broadcast 类算子方案设计的**知识索引入口 + 生成强制规范**。 broadcast 类算子生成 `DESIGN.md` 时，**必须**遵守本文件的规则。

---

## 强制规则（生成前必读）

1. 生成 DESIGN.md 时，必须逐节查阅下方设计规范中的**所有必读文件**，从对应文件中提取结论。一次只处理一个子节，完成当前子节前不得前进（每节依赖上一节的决策：3.2 TilingData 依赖 3.1 模板计划；3.3 Host Tiling 依赖 3.2；3.4 Kernel 依赖 3.3）。禁止先通读全文再凭记忆回填、跳节、或先读后节再补前节——这会基于尚未推导的假设产出错误结果
2. 必读文件中的"约束"、"规则"、"陷阱"、限制说明，必须在 DESIGN.md 对应章节中**以表格或编号列表显式写出**，不得省略
3. 必读文件中的代码必须写到 DESIGN.md，仅替换算子特有占位符。代码块内禁止出现 `// ...`、`/* ... */` 等省略占位符，也不得用自然语言描述替代真实 API 调用。容易静默出错的参数必须保留 `⛔` 标注。**凭记忆自创的 Ascend C 代码必然出错——必读文件的写法是唯一正确来源，禁止凭理解自行编写**
4. **不得凭记忆或经验编写**；必须以必读文件中的结论为准，不得自行发挥。生成的 Tiling 代码对齐参考源 Tiling，生成的 Kernel 代码对齐参考源 Kernel——先读参考源再写，禁止凭空推导
5. 若必读文件知识与 `spec.yaml` 或 `REQUIREMENTS.md` 冲突，以 spec 为准，并在对应章节标注差异
6. 所有 API 映射必须引用 `[regbase_api_whitelist.md](../../api/regbase_api_whitelist.md)` 中的验证状态；**不得把"待验证"写成"已验证"**
7. **本规范中的 broadcast 专用知识优先于 agent 通用知识**。若两者对同一设计点有不同结论（如数据流结构、API 选型、Buffer 布局），以本规范指向的必读文件为准，不得用通用经验覆盖
8. **源码标注规范**：(附源码) = DESIGN.md 必须包含 ```cpp 代码块，禁止散文替代；(附结构) = DESIGN.md 必须包含 ```cpp 结构体定义；(附公式) = DESIGN.md 必须包含公式块
9. **路线固定 RegBase**：Broadcast 范式所有算子统一走 RegBase 路线，使用 AscendC TilingKey 模板化（`ASCENDC_TPL_*` 宏）。TilingKeyIS 在本范式中无效
10. **禁止使用 add_example 作为 Broadcast 算子的代码模板**。`add_example` 是 Elementwise 范式（输入输出 shape 相同），使用 TQue + flat DataCopyPad + 线性循环，与 Broadcast 范式不兼容。Broadcast 算子必须使用：
    - **TBuf**（禁止 TQue）
    - **NDDMA `DataCopy<T, NDDMA_DIMS, cfg>`**（禁止 flat DataCopyPad 作为 CopyIn）
    - **多维坐标系统**（GetCoreRange / FlatToEffectiveCoord / CalcOffset）
    - **NDDMA stride 参数**（loopSize / loopSrcStride / loopDstStride）

    违反上述约束的代码不可交付
11. **章节标题与代码块规范**：章节标题必须与 DESIGN.md.templ 一致（`## X` / `### X.y`，X 为章节号 5–11）；参考示例用另一套编号（`## 1./## 2.`），不得照抄。不得重排、跳过、改名或自造子节。代码块语言标签：pseudocode（algorithm/scheduling）、c++（struct/template）、regbase（AscendC API），不得混用。本规范覆盖 §5–§11，§1/§2 从 DESIGN.md.templ 填写，§4 见本规范末尾，§12/§13 简要填写

---

## 参考代码

以下 adam_apply_one_assign 示例是 broadcast 范式的参考实现（以 arch35 为例），覆盖 Kernel、Tiling、数据结构三层。设计阶段（DESIGN.md 生成）和开发阶段（代码实现）均可查阅，用于理解文件组织、类结构、入口注册、Tiling 流程等具体写法。

**覆盖场景**：Standard 路径（NDDMA broadcast + 多维坐标 + TBuf）。

| 层级 | 文件路径 | 内容 |
|------|---------|------|
| Kernel | `example/examples-adam-code/op_kernel/arch35/adam_apply_one_assign_kernel.h` | adam Kernel 参考（含 CopyInBrc / CopyOut / Init / Process / Sync + 四个泛型辅助函数与搬运量计算函数） |
| Kernel | `example/examples-adam-code/op_kernel/adam_apply_one_assign.cpp` | 入口注册 + TilingKey 模板实例化（RANK=4/8 分发；入口文件名须与 opFile.value 一致，不带 _apt 等后缀） |
| 数据结构 | `example/examples-adam-code/op_kernel/arch35/adam_apply_one_assign_tiling_struct.h` | adam TilingData struct |
| 数据结构 | `example/examples-adam-code/op_kernel/arch35/adam_apply_one_assign_struct.h` | adam TilingKey struct |
| Tiling | `example/examples-adam-code/op_host/arch35/adam_apply_one_assign_tiling_arch35.cpp` | adam Host 侧 Tiling 参考（FindSplitAxis / MultiCoreSplit / TilingFunc） |
| Tiling | `example/examples-adam-code/op_host/arch35/adam_apply_one_assign_tiling_arch35.h` | Tiling 头文件（AdamTiling 类 + 辅助函数声明） |
| Host 注册 | `example/examples-adam-code/op_host/adam_apply_one_assign_def.cpp` | 算子定义（OP_ADD + AICore 配置） |

**使用原则**：
1. 各章节的"必读"references/ 文档是**规范真值源**，示例代码是**实现参照**，两者互补
2. 当 references/ 文档的抽象描述与示例代码有出入时，以 references/ 文档为准
3. 示例代码中的算子特有部分（TilingKey 取值、TilingData 字段、dtype）开发新算子时替换为对应实现

---

## 范式设计总览

> **进入设计规范前必读**：[broadcast-template-overview.md](references/broadcast-template-overview.md)
>
> 该文档描述模板划分、TilingKey 模板化、TilingData 划分及三者之间的联系。先读此文档建立全局心智模型，再进入下方逐节设计规范。

---

## 设计规范

> 按 DESIGN.md 章节顺序排列。每个 H4 必须出现且内容非空。

### §3.1 模板划分总览

**[3.1.1] TilingKey 模板化** (附结构)

- **必读**：[broadcast-template-overview.md](references/broadcast-template-overview.md) §3.1、§5（代码模板）
- **指引**：broadcast-template-overview.md §3 用 `ASCENDC_TPL_*` 模板参数替代传统 `TILING_KEY_IS` 数值分支，Broadcast 范式统一走模板化路径，不使用 TilingKeyIS。追问：本算子怎么用 TilingKey 模板化的方式划分 TilingKey
- **产出**：(1) 不同 TilingKey 覆盖的范围 (2) 按 [broadcast-template-overview.md](references/broadcast-template-overview.md) §5 的 TilingKey 模板（原 op_struct.h.templ，已内联）完成 C++ 代码
- **约束**：Broadcast TilingKey 只与 RANK 有关，与数据类型无关；参考源码建议 Rank=4、Rank=8（Rank=4 是常态，Rank=8 是保底）

---

### §3.2 TilingData 结构体

**[3.2.1] TilingData 字段定义** (附结构)

- **必读**：[broadcast-template-overview.md](references/broadcast-template-overview.md) §4.1、§5（代码模板）
- **指引**：本算子如何效仿 adam_apply_one_assign 构建 TilingData 结构体
- **产出**：(1) 参考 [broadcast-template-overview.md](references/broadcast-template-overview.md) §5 的 TilingData 模板（原 tiling_data.h.templ，已内联），实现 TilingData C++ 代码
- **约束**：以必读文件模板为准，仅替换算子特有占位符

---

### §3.3 Host 侧 Tiling 方案

**[3.3.1] 逻辑存活节点（L）**

- **必读**：[broadcast-standard-dag-buffers.md](references/broadcast-template-standard/broadcast-standard-dag-buffers.md) §1.1、§3
- **指引**：从 spec.yaml 公式推导 S1..Sn 并标注分叉点；以持有法则逐操作 trace。L trace 仅基于纯数学拓扑，不考虑 VF、融合、寄存器。调度 trace 格式示例：

  ```pseudocode
  // S1: DataCopy x（第一步，执行前为空）
  // 执行前: 持有=[]
  // 执行中: 持有=[x]
  // 执行后: 持有=[x]               ← x 后续 S2 会读，保留
  // S2: out = abs(x)
  // 执行前: 持有=[x]
  // 执行中: 持有=[x, out]          ← 峰值 2
  // 执行后: 持有=[out]             ← x 无未来消费者，释放
  ```

- **产出**：
  (1) `##### 计算图` — 从 spec.yaml 公式推导 S1..Sn，标注分叉点。标 pseudocode。格式：`S<N>: <操作> [<输入> → <输出>]`
  (2) `##### 调度` — 以持有法则逐操作 trace。标 pseudocode。每步标注 执行前/中/后: 持有=[...]，消费者推理，峰值标 ← 峰值
  (3) `##### 结论` — 表格：L 值 / 峰值出现步骤 / 瓶颈 / 不可降原因
- **约束**：对照 [broadcast-standard-dag-buffers.md](references/broadcast-template-standard/broadcast-standard-dag-buffers.md) §3 自查

---

**[3.3.2] 物理存活节点（P）**

- **必读**：[broadcast-standard-dag-buffers.md](references/broadcast-template-standard/broadcast-standard-dag-buffers.md) §1.1、§1.2、§3，[broadcast-standard-kernel-template.md](references/broadcast-template-standard/broadcast-standard-kernel-template.md) §5.2、§5.3
- **指引**：P = f(计算流, API 能力, 算法实现)。计数域为 UB buffer，寄存器不计入。P ≥ L 一般成立，但 VF 寄存器链融合可打破，此时 P < L 合法。融合穷举方法论见 [broadcast-standard-dag-buffers.md](references/broadcast-template-standard/broadcast-standard-dag-buffers.md) §1.2
- **产出**：
  (1) `##### 核心原则` — P = f(计算流, API 能力, 算法实现)。计数域为 UB buffer，寄存器不计入。P ≥ L 一般成立，但 VF 寄存器链融合可打破，此时 P < L 合法
  (2) `##### 可用 API 能力` — 表格：API | 能力 | 对物理节点的意义 | 来源。必须检查：NDDMA 随路 broadcast、RegBase 寄存器指令集（打开 `../../api/regbase_api_whitelist.md` 逐条核对存在性）
  (3) `##### VF 融合穷举分析` — 列出所有相邻 Vector 操作对，逐对判据：RegBase 指令存在？非 Cast？链长 ≤ 7？多消费者不打断链，但融合后峰值 ≤ 目标 P 方可全融，超 P 则拆分。Cast 不参与 VF 融合，作为断点
  (4) `##### {dtype} 物理 trace` — 每种 dtype 独立 trace（从 spec.yaml dtype_policy 提取）。从 CopyIn S1 到 CopyOut Sn 每步展开，禁止写总结。每步标注 执行前/中/后: 持有=[B0,B1,...] + 消费者推理，峰值标 `← P_{dtype}=N 峰值`。Buffer 用 B0/B1/... 命名。标 regbase
  (5) `##### 结论表` — 汇总各 dtype 路径下的 P 值。含 perBufBytes、perBufElems
- **约束**：对照 [broadcast-standard-dag-buffers.md](references/broadcast-template-standard/broadcast-standard-dag-buffers.md) §3 自查

---

**[3.3.3] 计算流伪码**

- **必读**：[broadcast-standard-compute-flow.md](references/broadcast-template-standard/broadcast-standard-compute-flow.md) 全文，[broadcast-standard-kernel-template.md](references/broadcast-template-standard/broadcast-standard-kernel-template.md) §1.2、§8
- **指引**：本节非必须。agent 自行判断是否需要——计算流简单（≤3 步链、单输出、无 Cast+VF 混合）时可跳过，P trace 已充分表达计算步骤时也可跳过。复杂算子（链长 >5、含分支/并行路径、多输出）建议执行
- **产出**：（如决定执行）(1) `##### {dtype} 计算流伪码` — 每种 dtype 路径分别写，代码块标 ` ```pseudocode ` (2) `##### 关键设计决策表`
- **约束**：⚠ agent 自行判断是否执行，简单算子可跳过

---

**[3.3.4] 物理计算流验证**

- **必读**：harness 见 [broadcast-standard-tiling.md](references/broadcast-template-standard/broadcast-standard-tiling.md) §7
- **指引**：本节非必须。agent 自行判断是否执行——计算逻辑复杂（链长 >5、含分支/并行路径、多输出）、不确定正确性时执行；简单算子（≤3 步链、单输出、无 Cast + VF 混合）可跳过
- **产出**：（如决定执行，按以下 5 步）
  (1) `#### 物理计算流验证` — golden 来源、opt 映射、verify_results.txt 原文
  - 步骤 1：Write `golden.py` — spec 公式的逐行翻译
  - 步骤 2：Write `opt.py` — 计算流伪码的逐行翻译，独立实现
  - 步骤 3：Write `verify.py` — 从 [broadcast-standard-tiling.md](references/broadcast-template-standard/broadcast-standard-tiling.md) §7 拷贝 harness，取消 `from golden import ...` / `from opt import ...` 注释
  - 步骤 4：Bash `python3 verify.py`，exit 必须为 0
  - 步骤 5：Bash `cat verify_results.json` 确认 `total >= 180` 且 `failed == 0`
- **约束**：⚠ agent 自行判断是否执行；golden 与 opt 必须独立实现

---

**[3.3.5] 输入预处理** (附源码)

- **必读**：[broadcast-standard-tiling-preprocess.md](references/broadcast-template-standard/broadcast-standard-tiling-preprocess.md) §2
- **指引**：基于参考实现 `HasNonPositiveDim` / `PadAndSqueeze` / `CheckBroadcastShape`，替换入参/出参数目和 shape 来源为本算子的值，其余逻辑原样保留
- **产出**：
  (1) `#### 输入预处理` — 基于 [broadcast-standard-tiling-preprocess.md](references/broadcast-template-standard/broadcast-standard-tiling-preprocess.md) §2 中的 `HasNonPositiveDim`（空 shape 防御，任一维 <= 0 报错）+ `PadAndSqueeze` 参考实现，替换入参/出参数目和 shape 来源为本算子的值，其余逻辑原样保留。代码块标 ` ```c++ `
  (2) `#### 校验` — 基于 [broadcast-standard-tiling-preprocess.md](references/broadcast-template-standard/broadcast-standard-tiling-preprocess.md) §2 中的 `CheckBroadcastShape` 参考实现（输入同维非 1 大小一致 + 输出稠密——广播输出多核写-写竞争，Host 侧拒绝），替换入参/出参数目为本算子的值，其余逻辑原样保留。代码块标 ` ```c++ `
- **约束**：对照 [broadcast-standard-tiling-preprocess.md](references/broadcast-template-standard/broadcast-standard-tiling-preprocess.md) §3 自查

---

**[3.3.6] UB 切分** (附源码)

- **必读**：[broadcast-standard-tiling.md](references/broadcast-template-standard/broadcast-standard-tiling.md) §1
- **指引**：以 `FindSplitAxis` 函数为模板（签名含 `ubBlockSize` 参数，来自 `Ops::Base::GetUbBlockSize(ctx_)`，不要写 32），替换 perBufElems 来源和坐标系参数为本算子的值，其余逻辑原样保留
- **产出**：(1) `#### UB 切分` — `FindSplitAxis` 模板替换后的 C++ 代码。代码块标 ` ```c++ `
- **约束**：对照 [broadcast-standard-tiling.md](references/broadcast-template-standard/broadcast-standard-tiling.md) §6 自查

---

**[3.3.7] 多核切分** (附源码)

- **必读**：[broadcast-standard-tiling.md](references/broadcast-template-standard/broadcast-standard-tiling.md) §2
- **指引**：以 `MultiCoreSplit` 函数为模板，替换 totalTiles/usedCoreNum 来源为本算子的值，其余逻辑原样保留
- **产出**：(1) `#### 多核切分` — `MultiCoreSplit` 模板替换后的 C++ 代码。代码块标 ` ```c++ `
- **约束**：对照 [broadcast-standard-tiling.md](references/broadcast-template-standard/broadcast-standard-tiling.md) §6 自查

---

**[3.3.8] Host 侧注册** (附源码)

- **必读**：[broadcast-standard-tiling.md](references/broadcast-template-standard/broadcast-standard-tiling.md) §8
- **指引**：以 `IMPL_OP_OPTILING({OP}).Tiling(TilingFunc{OP}).TilingParse<{OP}CompileInfo>(TilingPrepareFor{OP})` 为模板，替换算子名为本算子的值（CompileInfo 为空结构、TilingPrepare 为空实现，平台参数在 Tiling 阶段直取）
- **产出**：(1) `#### Host 侧注册` — 注册宏模板替换后的 C++ 代码。代码块标 ` ```c++ `
- **约束**：仅替换算子名，注册结构不变

---

**[3.3.9] Tiling 整体结构**

- **必读**：[broadcast-standard-tiling.md](references/broadcast-template-standard/broadcast-standard-tiling.md) §4、§8
- **指引**：回答三个问题：入口函数是什么？按什么分叉（对应 §3.1）？每条分叉里按什么顺序调前面定义的函数？
- **产出**：(1) `#### Tiling 整体结构` — 三问回答。代码块标 ` ```pseudocode `
- **约束**：分叉依据必须与 §3.1 的 TilingKey 划分一致

---

### §3.4 Kernel 侧计算方案

**[3.4.1] Kernel 入口** (附源码)

- **必读**：[broadcast-template-overview.md](references/broadcast-template-overview.md) §5（代码模板）、[broadcast-kernel-entrance.md](references/broadcast-kernel-entrance.md) §1
- **指引**：先写**开发指导推导**（永久保留在文档中，供开发阶段参考），再放代码。针对当前算子，推导 Key 是怎样从 §3.1 一路传到 kernel 模板实例化的，回答：
  1. 本算子的 Key 是什么、有几个分叉？回到 §3.1 确认
  2. 为什么 TilingData 必须按 Key 分两档 using？（提示：看 TilingData 的模板参数是什么，那个参数的值从哪来）
  3. DTYPE 从哪个 input 来？为什么是这个 input？（提示：看 spec.yaml 的第一个 input 叫什么，入口 .cpp 里对应的宏怎么写的）
- **产出**：
  (1) `#### Kernel 入口` — 开发指导推导（自然段落，不用表格）+ 代码
  - **代码**：以 [broadcast-template-overview.md](references/broadcast-template-overview.md) §5 的 Kernel 入口模板（原 apt.cpp.templ，已内联）为准，替换 8 个占位符（{OP_SNAKE}/{OP}/{OP_UPPER}/{PRIMARY_INPUT_NAME}/{INPUT_GMADDRS}/{OUTPUT_GMADDRS}/{INPUT_ARRAY}/{OUTPUT_ARRAY}）。模板中的 `{{` `}}` 为 C++ 字面花括号，复制时转为单 `{` `}`。**代码块必须标 ` ```c++ `**
- **约束**：⚠ 确认开发指导推导回答了三个问题；确认所有占位符已替换

---

**[3.4.2] Kernel 类声明** (附源码)

- **必读**：[broadcast-template-overview.md](references/broadcast-template-overview.md) §5（代码模板）、[broadcast-standard-kernel-template.md](references/broadcast-template-standard/broadcast-standard-kernel-template.md) §2
- **指引**：先写**开发指导推导**，再按模板写代码。这个类的成员里有几个数字不是凭空定的，逐一追问：
  1. `PHYS_NODES` 为什么是这个值？它和 §3.3 有什么关系？
  2. `MAX_INPUT_SLOTS` / `MAX_OUTPUT_SLOTS` 为什么是这个数？来源是什么？
  3. `NDDMA_DIMS` 的公式为什么是 `min(RANK, 5)`？NDDMA 的硬件限制是什么？
- **产出**：
  (1) `#### Kernel 类声明` — 开发指导推导（写出当前算子对应的具体值）+ 代码
  - **代码**：以 [broadcast-template-overview.md](references/broadcast-template-overview.md) §5 的 Kernel 类声明模板（原 kernel.h.templ，已内联）为准，替换 5 个占位符（{OP_SNAKE}/{OP}/{PHYS_NODES}/{MAX_INPUTS}/{MAX_OUTPUTS}）。模板中的 `{{` `}}` 为字面花括号，复制后转为单 `{` `}`。**代码块必须标 ` ```c++ `**
- **约束**：⚠ 确认推导段列出了 PHYS_NODES/MAX_INPUT_SLOTS/NDDMA_DIMS 的来源；确认所有占位符已替换

---

**[3.4.3] Kernel 调度框架** (附源码)

- **必读**：[broadcast-standard-kernel-template.md](references/broadcast-template-standard/broadcast-standard-kernel-template.md) §3、[broadcast-standard-kernel-helpers.md](references/broadcast-template-standard/broadcast-standard-kernel-helpers.md) §1、§2
- **指引**：四个泛型辅助函数（坐标/偏移类）与搬运量计算函数 `CalcTransferCount` 直接抄模板，只改类型名；调度链路用 pseudocode 写出 Process 的调用序列
- **产出**：
  (1) **四个泛型辅助函数 + 搬运量计算函数**（共 5 个，标 ` ```c++ `）：`GetCoreRange`、`GetUBSplitRange`、`FlatToEffectiveCoord`、`CalcOffset`（以上四个见 [broadcast-standard-kernel-helpers.md](references/broadcast-template-standard/broadcast-standard-kernel-helpers.md) §1）+ `CalcTransferCount`（见同文档 §2；CalcOffset / CalcTransferCount 输入与输出规则相同，共用一份实现，不区分 CalcInputOffset / CalcOutputOffset）
  (2) **当前算子的调度链路**（标 ` ```pseudocode `）：从多核切分结果 → `GetCoreRange` 拿到 [start,end) → flat loop → `GetUBSplitRange` 取本段大小 → `FlatToEffectiveCoord` 算坐标 → 接下来每个输入 `CopyInBrc(coord, ...)`，讲清每一步在当前算子语境下的作用
- **约束**：⚠ 确认 5 个函数（四个泛型辅助 + CalcTransferCount）已写入且类型名已适配；确认调度链路覆盖从 `GetCoreRange` 到 `CopyInBrc` 的完整序列；对照 [broadcast-standard-kernel-template.md](references/broadcast-template-standard/broadcast-standard-kernel-template.md) §8 自查

---

**[3.4.4] Init** (附源码) ⛔ 强制产出

- **必读**：[broadcast-standard-kernel-template.md](references/broadcast-template-standard/broadcast-standard-kernel-template.md) §2、§4，[broadcast-standard-kernel-helpers.md](references/broadcast-template-standard/broadcast-standard-kernel-helpers.md) §3
- **指引**：写函数定义、参数意义、算法描述。必须包含：GM 绑定（`SetGlobalBuffer`）、TBuf 初始化（`InitBuffer`）、NDDMA 参数预计算（loopSize/loopSrcStride/loopDstStride/loopLpSize=0/loopRpSize=0 五字段）。NDDMA 结构体（`NdDmaLoopInfo`/`NdDmaParams`/`NdDmaConfig`）及三原则见 kernel-template §4，`nddmaOuterIters_` 计算见 kernel-helpers §3。代码质量标准参照必读文档
- **产出**：(1) `#### Init` — 函数定义、参数意义、算法描述。**代码块必须标 ` ```regbase `**（含 DataCopy/SetGlobalBuffer/TBuf Init）
- **约束**：⚠ 代码必须产出，不能是伪码，不得凭记忆编写；对照 [broadcast-standard-kernel-template.md](references/broadcast-template-standard/broadcast-standard-kernel-template.md) §8 自查

---

**[3.4.5] Process** (附源码) ⛔ 强制产出

- **必读**：[broadcast-standard-dag-buffers.md](references/broadcast-template-standard/broadcast-standard-dag-buffers.md) §1.1、[broadcast-standard-kernel-template.md](references/broadcast-template-standard/broadcast-standard-kernel-template.md) §2、§5.3
- **指引**：基于 §3.3 P trace（VF 融合后）写 Kernel 代码。每条 dtype 路径分别写，每行三态注释（执行前/中/后: 持有=[...]），持有数 ≤ P，峰值处标注 `← P_{dtype}=N 峰值`。禁止把融合链拆成多个操作——一条 `asc_vf_call` 吃掉一组 VF 链。多输出处理：若算子有 >1 个输出（{MAX_OUTPUTS} > 1），Process 中每个输出需独立 `CopyOutOne` 调用，按 P trace 中写明的 buffer 和顺序执行；多个 CopyOut 可连续排列、放在同一段 `Mutex::Lock/Unlock<PIPE_MTE3>` 临界区内（MTE3 顺序执行）；输出 buffer 在 CopyOut 前不可被覆写。Buffer 复用推导：从 P trace 的持有法则分析中识别每个 buffer 的角色变化点（何时释放、何时重新分配），代码中 buffer 角色变更须标注 `// UBx 释放` / `// UBx 重新分配为{新角色}`
- **产出**：(1) `#### {dtype} Process` — 基于 P trace 的 Kernel 代码。**代码块必须标 ` ```regbase `**（含 CopyInBrc/Cast/asc_vf_call/CopyOut）
- **约束**：⛔ 代码必须产出，不能是伪码，不得凭记忆编写；逐行核对三态注释（执行前/中/后: 持有=[...]）+ 持有数 ≤ P，任一行超出即回退 §3.3 P trace 重新推导；对照 [broadcast-standard-kernel-template.md](references/broadcast-template-standard/broadcast-standard-kernel-template.md) §8 自查

---

**[3.4.6] Sync** (附源码) ⛔ 强制产出

- **必读**：[broadcast-standard-kernel-template.md](references/broadcast-template-standard/broadcast-standard-kernel-template.md) §7、[broadcast-standard-dag-buffers.md](references/broadcast-template-standard/broadcast-standard-dag-buffers.md) §1.1
- **指引**：取 Process 代码，从**全局持有法则**出发识别跨流水线 RAW/WAR 依赖，用 Mutex 临界区串行化（范式统一机制，禁止 SetFlag/WaitFlag 事件对）：Process 开头 `uint8_t mutexId = AscendC::AllocMutexID();`、循环结束后 `AscendC::ReleaseMutexID(mutexId);`。每段 CopyIn（MTE2 操作）包在 `AscendC::Mutex::Lock<PIPE_MTE2>(mutexId)` / `Unlock<PIPE_MTE2>(mutexId)` 内；每段计算（Mul/Div/Cast/`asc_vf_call` 等 V 操作）包在 `Lock<PIPE_V>(mutexId)` / `Unlock<PIPE_V>(mutexId)` 内；全部 CopyOut 包在同一段 `Lock<PIPE_MTE3>(mutexId)` / `Unlock<PIPE_MTE3>(mutexId)` 内。分析：哪些 buffer 被 MTE2 写入后 V 读取（RAW MTE2_V）、哪些被 V 写入后 MTE3 读取（RAW V_MTE3）、哪里发生跨迭代覆写（WAR V_MTE2 / MTE3_MTE2）——同一 mutexId 的互斥一次性覆盖四类依赖。不是每条 CopyInBrc 都单独一段——相邻同流水线操作可合并进同一临界区。同步代码每操作上方同样标注 执行前/中/后: 持有=[...]，与 Process 格式一致
- **产出**：(1) `#### Sync` — 带 Mutex 临界区的代码。**代码块必须标 ` ```regbase `**
- **约束**：⚠ 代码必须产出，不能是伪码；Mutex Lock/Unlock 必须成对（同一对同 PIPE）且覆盖持有法则推导出的全部跨流水线 buffer 访问，`AllocMutexID` 后必须 `ReleaseMutexID`；对照 [broadcast-standard-kernel-template.md](references/broadcast-template-standard/broadcast-standard-kernel-template.md) §8 自查

---

**[3.4.7] CopyInBrc / CopyOut** (附源码) ⛔ 强制产出

- **必读**：[broadcast-standard-kernel-template.md](references/broadcast-template-standard/broadcast-standard-kernel-template.md) §4、§6、[broadcast-standard-kernel-helpers.md](references/broadcast-template-standard/broadcast-standard-kernel-helpers.md) §2、§3
- **指引**：`CopyInBrc` 核心是 NDDMA 5 维上限：`if constexpr (RANK <= MAX_NDDMA_DIMS)` 单次 DataCopy 覆盖所有维，`else` 分支通过 `nddmaOuterIters` 逐段搬运超出部分。`CopyOut` 使用 `DataCopyPad` 而非普通 `DataCopy`——DataCopyPad 不要求 blockLen 32B 对齐，硬件只写有效字节，blockLen = count × sizeof(T)
- **产出**：
  (1) `#### CopyInBrc` — 当前算子的 `CopyInBrc`，含 `NdDmaConfig` 配置结构体和 `DataCopy<T, NDDMA_DIMS, cfg>` 标准调用写法。代码块标 ` ```regbase `
  (2) `#### CopyOut` — 当前算子的 `CopyOut`，使用 `DataCopyPad`，blockLen = count × sizeof(T)。代码块标 ` ```regbase `
- **约束**：⚠ 代码必须产出，不能是伪码；⛔ 确认 `CopyInBrc` 含 `if constexpr (RANK <= MAX_NDDMA_DIMS)` 分叉；确认 `CopyOut` 使用 `DataCopyPad` 且 blockLen = count × sizeof(T)

---

### §3.5 API 映射

**[3.5.1] VF 函数签名** (附源码)

- **必读**：[broadcast-standard-kernel-template.md](references/broadcast-template-standard/broadcast-standard-kernel-template.md) §5.3
- **指引**：写 Kernel 代码、参数意义、算法描述
- **产出**：(1) `#### VF 函数签名` — Kernel 代码、参数意义、算法描述。**代码块必须标 ` ```regbase `**
- **约束**：⚠ 代码必须产出，不能是伪码

---

### §3.6 API 验证记录

**[3.6.1] API 验证总表**

- **必读**：[broadcast-standard-kernel-template.md](references/broadcast-template-standard/broadcast-standard-kernel-template.md) §8、[regbase_api_whitelist.md](../../api/regbase_api_whitelist.md)
- **指引**：从 Sync 代码（§3.4）中提取每步操作对应的 AscendC API，逐项查白名单。结论判定规则：API 在白名单中且平台支持且约束已处理 → "通过"；不在白名单或平台不支持 → "不通过"
- **产出**：(1) `#### API 验证` — 表格：Sync 步骤 | API | 白名单 | dtype 支持 | 约束 | 结论
- **约束**：⚠ 验证状态必须准确，禁止把"待验证"写成"已验证"；若任一 API 不通过 → 退回 §3.3 P trace，换 API 重新推导 P 值 → 重走 Process → 重走 Sync → 重查本节，直到全部通过

---

### §3.7 数据流设计

**[3.7.1] 数据流（由 §3.4 覆盖）**

详见 §3.4（Init → Process → Sync → CopyInBrc → CopyOut），无需重复填充。

---

### §3.8 内存管理

**[3.8.1] UB 内存分配方案**

- **指引**：⛔ Broadcast 算子一律用 TBuf 管理 UB 内存，禁止 TQue 和裸指针。当前算子分配方案：buffer 数量（= P 值，§3.3 结论表）、每个 buffer 大小（= perBufBytes）、角色分配（哪块 B? 对应 P trace 中的哪个 buffer，哪个角色可复用）
- **产出**：(1) **约束声明**：`⛔ Broadcast 算子一律用 TBuf 管理 UB 内存，禁止 TQue 和裸指针。` (2) **当前算子分配方案**：buffer 数量 / 每个 buffer 大小 / 角色分配
- **约束**：⛔ 禁止使用 TQue 或裸指针；`### 3.8 内存管理` 下直接写入，不另起标题

---

### §3.9 UB 容量验证

> 模板 L3 不可删。暂无内容。

---

## 4. 性能优化

> 模板 L3 不可删。暂无内容。
