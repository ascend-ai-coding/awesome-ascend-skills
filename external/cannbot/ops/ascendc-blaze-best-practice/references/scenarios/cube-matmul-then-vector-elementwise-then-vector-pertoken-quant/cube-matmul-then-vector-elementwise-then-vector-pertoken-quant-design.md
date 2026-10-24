# Cube MatMul -> Vector Elementwise -> Vector Per-token Quant

本 Scenario 是 shape-preserving unary/pointwise DAG 的受控扩展开发模板，
描述一条不可拆分的复杂扩展路线：

```text
Cube MatMul (MM / BMM / GMM / MM_MX)
    -> optional declared bias / scale / dequant
    -> Vector: one shape-preserving elementwise DAG
    -> FP workspace [R, N]
    -> Vector: per-token quant
    -> y [R, N] + yScale [R]
```

本页在 [Scenario 索引](../index.md)唯一命中后读取，只描述 Cube
MatMul、shape-preserving elementwise 和 per-token quant 的组合 delta。
GMM 相对 MM 的 group 编码、完整 POD、Scheduler/Kernel 生命周期和地址规则
统一遵循
[Grouped MatMul 项目侧差异](../../kernel-design/group-matmul-delta.md)。

## 0. Step 3 输入与源码前提

进入正文前必须具备 `requirements_contract`、`operator_interface_contract`、
`native_gaps`、`matmul_base_analysis` 及其 source-backed `abi_bindings[]`，以及
同次 Investigation 对 MatMul final output、目标 elementwise/broadcast API、
FP workspace、阶段完成依赖、完整行 quant、Tiling/Params 和静态 entry 的事实。

缺少决定性事实时只允许向 Step 2 提出一次无场景名补充问题，记录受影响
requirement IDs、待确认的公式/物理接口/完成时机、有限源码前沿和阻塞原因；
补充后仍缺失则停止，不生成 PLAN。MemBase/RegBase 选择前分别加载实际需要的
依赖 Skill 根入口，并记录 API、限制和真实参考证据。

DESIGN 必须区分 consumed/preserved/added contracts，并用
`abi_crosswalk_delta` 只追加 broadcast operand、workspace、量化参数、entry、
输出和同步接线；每个 custom 文件都要记录来源和授权边界。

## 1. 精确匹配条件

### 1.1 设计主线

`scenario_id`:
`cube-matmul-then-vector-elementwise-then-vector-pertoken-quant`。

仅当需求同时满足以下条件时选择本 Scenario：

- 基础计算是 MM、BMM、GMM 或 MM_MX；对应的 batch/group/MX scale、layout、
  dtype 和地址合同已确认。
- elementwise 阶段只有一个主 Tensor 输入；每个输出元素只依赖同位置元素，
  以及用户声明的 scalar、row 或 column broadcast 输入。
- 将全部逻辑行轴展平为 `R` 后，elementwise 输出保持逻辑 shape `[R,N]`，
  不包含跨行或跨列归约；MM/GMM 通常 `R=M`，BMM 通常 `R=B*M`。
- 公式 DAG、逐操作顺序、近似方式和 dtype 已由用户合同或 Golden 冻结。
- GELU、ReLU 以及其他满足上述同位置依赖和 shape-preserving 条件的 unary
  DAG 均进入本 Scenario；没有现成公式资产时标记 `adaptation_required`，
  仍由本 Scenario 约束完整联合路线。
- elementwise 输出写入 FP workspace，per-token quant 对每个完整逻辑行
  `[0,N)` 生成唯一 scale。
- bias、scale 和 dequant 仅在用户语义要求时存在，其 broadcast、顺序和 dtype
  已确认。

### 1.2 成立条件/门禁

以下需求不匹配：

- GLU、GEGLU、SwiGLU 等双投影/双分支公式不匹配；返回
  [Scenario 索引](../index.md)选择 GLU-family 完整路线。
- Softmax、LayerNorm 或其他包含跨元素归约、归一化或 shape 改变的计算。
- Cube MatMul 后没有本页定义的 elementwise、而是直接执行 quant 时不匹配；
  返回 [Scenario 索引](../index.md)重新选择。
- 输入只是已存在的 FP workspace、没有 Cube MatMul producer，不属于本 Skill
  的 Scenario。

基础组件、证据状态和 `integration_mode` 由当前 Investigation 的 concrete
witness 和本场景 DESIGN 绑定。

### 1.3 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| 把 GLU 当作 unary elementwise | GLU 需要双投影和 paired-axis 语义 | 选择 Cube MatMul→Vector GLU→Vector quant Scenario |
| 把包含归约或 shape 改变的 Vector 图登记为本场景 | 本场景只覆盖同位置依赖并保持 `[R,N]` | 登记精确的新 Scenario |
| 只根据激活名称选择近似 API | 同名 API 可能使用不同公式和 dtype 顺序 | 以用户 Golden 冻结 DAG |

## 2. 不可拆分契约

### 2.1 设计主线：Cube MatMul 与 elementwise 阶段

`DESIGN.md` 使用用户原型参数名写出完整标量公式。例如 GELU-tanh 按 Golden
冻结的常量、运算顺序和 dtype 实现。MatMul accumulation policy 单独冻结；
Cube accumulation 与 Golden accumulation 的差异必须有合同或输入域证据。

目标公式可选择 MemBase 或 RegBase，并按条件 Skill 给出当前 SDK API、
mask/tail 和真实参考证据。没有公式匹配的通用资产时，在目标工程实现并验证
公式专用 Epilogue。

### 2.2 接口与数据流：Workspace 交接

elementwise 生产者与 quant 消费者共享以下合同：

| 项 | 必须冻结 |
|---|---|
| 逻辑 Tensor | FP workspace `[R,N]` |
| 物理布局 | dtype、row pitch、alignment 和实际 GM 容量 |
| elementwise 数据路径 | 公式执行位置（producer VF/UB 或独立 GM pass）、源/目的地址、行步长、对齐、padding、Copy specialization 和有效 tail |
| 完成条件 | 所有 workspace 写入完成且本地 C/V 协议 final drain 完成 |
| 阶段边界 | 单一 `__mix__` entry 内所有 producer/consumer 完成已选同步协议 |
| 所有权切换 | 交接后由 Kernel 按完整逻辑行重新分配 AIV |

Kernel 根据逻辑 AIV rank 切出连续行 Tensor slice；quant Epilogue 只处理传入
slice。量化公式、UB 容量、tail 和 selector 合同见
[Per-token Quant 组件开发指导](../../kernel-design/per-token-quant-development.md)。

### 2.3 成立条件/门禁

- elementwise 输出保持 `[R,N]`，每个元素只依赖合同声明的同位置与 broadcast
  输入。
- workspace 的 shape、dtype、row pitch、alignment 和容量在 producer 与
  quant consumer 间一致。
- 全部 workspace 写入和 final drain 完成后，Kernel 才按完整行重新分配。
- MatMul accumulation、elementwise 逐操作 dtype 和近似均由合同冻结。

### 2.4 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| 把双分支 GLU 资产改名后用于 unary activation | 数据流、输入数和 paired-axis 合同均不匹配 | 实现公式专用 Epilogue |
| quant 沿用 elementwise 的局部列 ownership | 无法覆盖完整 `[0,N)` 归约域 | 交接后由 Kernel 按行重新分配 |
| 把逻辑连续的 `[R,N]` workspace 直接交给未证明的额外 GM/UB elementwise pass | 当 `N*sizeof(dtype)` 不是目标 DMA 对齐粒度的倍数时，下一行的物理起始地址、pitch 或 padding 可能与逻辑 Tensor view 不一致 | 先冻结物理地址、行步长、Copy specialization 并用同设备正例验证；无法证明时将 elementwise 保留在 producer VF/UB 或使用已证明的物理搬运路径 |
| 为规避最终 INT8 的 half-tie mismatch 反复替换等价 elementwise 表达式 | 改变了用户 Golden、逐操作 dtype 或数值路径，且不能证明最早错误边界 | 先比较 MMAD/dequant workspace、elementwise workspace 和 yScale；只有在正式验收合同重新批准后才改变数值接受规则 |
| 最终输出是 INT8 就忽略 accumulation 差异 | quant 不能消除上游数值语义差异 | 单独冻结并验证 accumulation policy |

## 3. 参考资产与适配

### 3.1 设计主线与接口

本扩展模板不登记第二份公共资产表：

- GMM Scheduler、Kernel、TilingData 和 group 合同只从
  [Grouped MatMul 项目侧差异](../../kernel-design/group-matmul-delta.md)选择。
- quant Epilogue、UB 公式和 Host selector 只从
  [Per-token Quant 组件开发指导](../../kernel-design/per-token-quant-development.md)选择。
- elementwise 公式根据冻结的 DAG 选择目标版本 MemBase/RegBase API；没有匹配
  资产时实现公式专用 Epilogue，并保持本叶子的 workspace 和 `__mix__` 合同。

目标工程只复制 `DESIGN.md` 选中的资产及必要直接依赖。

### 3.2 成立条件/门禁

- 目标 MM/BMM/GMM/MM_MX 基础 Assembly、Scheduler、Params、Block 输出和
  Kernel Cube→Vector 接口由同版本 Investigation 绑定；项目侧 Host TilingData
  按关联证据或 `DESIGN.md` 单独绑定。
- 正式路线只允许单一 `__mix__` entry，不得拆成独立 elementwise producer 和
  quant consumer entry。
- elementwise 公式所需的 MemBase/RegBase API 必须有目标版本证据。
- `unknown/conflict` 时阻塞；`not_found` 只允许进入扩展，不证明 delta 成立。
- Grouped Block/Kernel 资产仅在目标为 GMM 且 group 合同和接口兼容时适配；
  MM、BMM、MM_MX 不得引入无关的 Grouped delta。

### 3.3 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| 把 dequant+SwiGLU 资产当作 unary activation 证据 | 资产的输入和公式不匹配 | 为目标 DAG 选择或实现专用 Epilogue |
| `not_found` 后直接宣称 custom 路线可用 | 上游缺失不证明扩展接口成立 | 继续完成 delta 的 capability 和设备证据 |
| 复制未被 DESIGN 选择的资产 | 扩大符号和依赖面 | 只复制选中的 delta 与直接依赖 |

## 4. 联合验证

### 4.1 场景验证矩阵

除每个阶段的独立验证外，至少覆盖：

- elementwise workspace 与最终 `y`/`yScale` 的分阶段 Golden；
- 公式近似、逐操作 dtype、dequant broadcast 和运算顺序；
- MatMul accumulation policy 的合同一致性或已证明输入域；
- M/N/K tail 和 multi-tile；BMM 增加 batch 边界，GMM 增加 empty expert 与
  不均匀 group，MM_MX 增加其 scale/block 合同边界；
- absmax 位于列 64 之后、全零行 `scaleMin` 与编译期 clamp 顺序、量化边界和 saturation；
- single-pass 容量边界前/等于/后，two-pass 首个宽度、多 chunk 和 chunk tail；
- final drain、阶段交接、inactive AIV 和交接后完整行重分配；
- 叶子正式输出路径对应的 C-direct、C-through、V-zero、V-known 和 Full；
- 相同输入重复运行至少 10 次，清理诊断代码后的 Full 回归。

### 4.2 成立条件/门禁

- elementwise workspace 和最终 `y/yScale` 使用分阶段 Golden。
- 诊断模式必须命名实际交接边界，不能使用模糊的 `C-through-fusion`。
- 通用精度、同步和性能结论按通用规则导航的唯一事实源执行。

### 4.3 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| 只比较最终 INT8 | 无法定位 Cube MatMul、elementwise 或 quant 阶段 | 同时验证 FP workspace、scale 和输出 |
| 用 `C-through-fusion` 隐藏输出路径 | 诊断证据无法对应具体接口 | 写明实际 C-through 路径 |
| 清理前的诊断 PASS 作为最终证据 | 诊断 entry 改变了正式工程 | 清理后 clean build 并重跑 Full |
