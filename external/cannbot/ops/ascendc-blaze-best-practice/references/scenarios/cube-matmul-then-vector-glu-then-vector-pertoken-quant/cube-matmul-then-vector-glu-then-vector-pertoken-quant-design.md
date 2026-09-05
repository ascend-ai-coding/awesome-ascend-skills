# Cube MatMul -> Vector GLU -> Vector Per-token Quant

本 Scenario 描述一条不可拆分的复杂扩展路线：

```text
Cube MatMul (MM / BMM / GMM / MM_MX)
    -> optional bias / scale / dequant
    -> Vector: specified GLU-family formula
    -> FP workspace [R, Q]
    -> Vector: per-token quant
    -> y [R, Q] + yScale [R]
```

本页在 [Scenario 索引](../index.md)唯一命中后读取，只描述 Cube
MatMul、GLU 和 per-token quant 不可拆分组合的场景 delta。
GMM 相对 MM 的 group 编码、完整 POD、Scheduler/Kernel 生命周期和地址规则
统一遵循
[Grouped MatMul 项目侧差异](../../kernel-design/group-matmul-delta.md)。

## 0. Step 3 输入与源码前提

进入正文前必须具备 `requirements_contract`、`operator_interface_contract`、
`native_gaps`、`matmul_base_analysis` 及其 source-backed `abi_bindings[]`，以及
同次 Investigation 对基础 Assembly、双分支 L0C2UB、GLU producer、FP
workspace、全局完成依赖、完整行 consumer、Tiling/Params 和静态 entry 的事实。

缺少决定性事实时只允许向 Step 2 提出一次无场景名补充问题，记录受影响
requirement IDs、待确认的 producer-completion-consumer 物理关系、有限源码
前沿和阻塞原因；补充后仍缺失则停止，不生成 PLAN。普通 AscendC API 与
RegBase/VF 的依赖 Skill 按实际实现路线加载根入口。

DESIGN 必须分别冻结 MatMul/GLU、workspace 交接和 quant 的
consumed/preserved/added contracts；`abi_crosswalk_delta` 只追加成对 operand、
workspace、量化参数、entry、输出和同步接线，并为每个 custom 文件记录授权。

## 1. 精确匹配条件

### 1.1 设计主线

`scenario_id`: `cube-matmul-then-vector-glu-then-vector-pertoken-quant`。

仅当需求同时满足以下条件时选择本 Scenario：

- 基础计算是 MM、BMM、GMM 或 MM_MX；对应的 batch/group/MX scale、layout、
  dtype 和地址合同已确认。
- 用户明确给出 GLU-family 公式，例如 GLU、Bilinear、ReGLU、GEGLU 或
  SwiGLU。
- 将全部逻辑行轴展平为 `R` 后，GLU 输出写入逻辑 `[R,Q]` 的 FP workspace；
  MM/GMM 通常 `R=M`，BMM 通常 `R=B*M`。
- per-token quant 对每个完整逻辑行 `[0, Q)` 生成唯一 scale。
- bias、scale 和 dequant 仅在用户语义要求时存在，其计算顺序和 dtype 已确认。

只包含 GLU，或 Cube MatMul 后直接执行 per-token quant 时，返回
[Scenario 索引](../index.md)按完整需求选择唯一场景。

### 1.2 成立条件/门禁

- 基础计算变体语义、GLU 公式、workspace `Q` 和 per-token quant 域必须同时闭合。
- bias、scale、dequant 的存在、顺序和 dtype 均以用户合同为准。
- 基础组件、证据状态和 `integration_mode` 由当前 Investigation 的 concrete
  witness 和本场景 DESIGN 绑定。

### 1.3 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| 将简单 GLU 与 quant 两个叶子自由拼接 | 两阶段 workspace、同步和所有权切换尚未联合设计 | 选择本不可拆分 Scenario |
| 从参考类名默认选择 SwiGLU | 用户公式可能是其他 GLU 变体 | 按 Golden 冻结 activation 和分支 |
| GLU 后仍按原始拼接宽度 `N` 做 quant | quant 域没有跟随 GLU 输出 | 对完整逻辑输出 `[0,Q)` 生成 scale |

## 2. 不可拆分契约

### 2.1 设计主线：GLU 阶段

两路投影记为 `act` 和 `gate`：

```text
workspace = activation(act) * gate
```

`activation`、两路命名、权重排布、bias/scale/dequant、计算 dtype 和输出
`Q` 均来自用户契约。只有两路各宽 `Q` 且输入拼接宽度明确为 `2Q` 时，才能
使用 `Q = N / 2`。

GLU 的双分支 Block、公式、布局和组件 delta 见
[GLU / SwiGLU 组件开发指导](../../kernel-design/glu-development.md)。

### 2.2 接口与数据流：Workspace 交接

GLU 生产者与 quant 消费者共享同一份合同：

| 项 | 必须冻结 |
|---|---|
| 逻辑 Tensor | FP workspace `[R,Q]` |
| 物理布局 | dtype、row pitch、alignment 和实际 GM 容量 |
| 完成条件 | 所有 workspace 写入完成且本地 C/V 协议 final drain 完成 |
| 阶段边界 | 单一 `__mix__` entry 内所有 producer/consumer 完成已选同步协议 |
| 所有权切换 | 交接后由 Kernel 按完整逻辑行重新分配 AIV |

Kernel 根据逻辑 AIV rank 切出连续行 Tensor slice；quant Epilogue 只处理传入
slice。完整量化、Host selector 和边界合同见
[Per-token Quant 组件开发指导](../../kernel-design/per-token-quant-development.md)。

### 2.3 成立条件/门禁

- GLU producer 写入的逻辑 shape、dtype、row pitch、alignment 和容量必须与
  quant consumer 完全一致。
- workspace 全部写入并完成本地 final drain 后，才能切换到完整行 ownership。
- quant Epilogue 只消费 Kernel 切出的连续行 slice。
- 两阶段的同步协议按通用规则导航的同步事实源设计和验证。

### 2.4 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| quant 继承 GLU 的局部列 tile ownership | consumer 不能看到完整 `[0,Q)` | 交接后由 Kernel 按完整逻辑行重新分配 |
| 使用 chunk-local absmax | 每行产生多个局部 scale | 对完整 workspace 行归约唯一 scale |
| workspace 写完前开始行重分配 | producer/consumer 生命周期未闭合 | 先证明全部写入和 final drain |

## 3. 参考资产与适配

### 3.1 设计主线

本组合 Scenario 不登记第二份 GLU 或 quant 资产表：

### 3.2 接口与数据流

- GLU Block、Epilogue、Grouped delta 和同步资产只从
  [GLU / SwiGLU 组件开发指导](../../kernel-design/glu-development.md)选择。
- quant Epilogue、UB 公式和 Host selector 只从
  [Per-token Quant 组件开发指导](../../kernel-design/per-token-quant-development.md)选择。
- 本叶子只拥有 GLU FP workspace→quant 的联合 shape、pitch、完成条件、
  ownership 重分配和单一 `__mix__` 生命周期合同。

目标工程只复制 `DESIGN.md` 选中的文件及必要直接依赖。

### 3.3 成立条件/门禁

- 目标 MM/BMM/GMM/MM_MX 基础 Assembly、Scheduler、Params 和 Cube→Vector
  接口由同版本 Investigation 绑定；项目侧 Host TilingData 按关联证据或
  `DESIGN.md` 单独绑定。
- 对 BMM，保留 Investigation 绑定的 Base BMM/QBMM batch ownership、
  Scheduler 和地址公式；GLU 阶段按公共 GLU 指导实现 paired view、双分支
  Block、L0C2UB/CV 和 row adapter，quant 阶段只增加本叶子定义的完整行
  ownership 交接。缺少预制 BMM GLU 或 BMM GLU+quant Kernel 资产本身不构成
  `extension_missing`，上述精确缺口应标记 `adaptation_required`。
- 正式路线只允许单一 `__mix__` entry，不得拆成独立 GLU producer 和 quant
  consumer entry。
- Grouped Block/Kernel/Tiling 资产仅在目标为 GMM 且 group 合同兼容时适配；
  MM、BMM、MM_MX 不得引入无关的 Grouped delta。
- 关键接口为 `unknown/conflict` 时阻塞；custom 资产必须有明确 delta 边界。
- 非 SwiGLU 变体标记 `adaptation_required`，并冻结对应 Epilogue 公式、API、
  mask/tail 和 Golden。

### 3.4 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| 复制整个 custom 目录 | 未选资产和间接依赖会扩大事实源与符号面 | 只复制 DESIGN 选中的 delta 和直接依赖 |
| 参考资产存在就判定其他 dtype 或版本支持 | 资产不等于目标 capability | 重新编译并验证目标实例与输出路径 |
| 共用双分支 Block 就跳过非 SwiGLU Epilogue 适配 | Block 不拥有 activation 公式 | 实现变体 Epilogue 与 Golden |
| 因没有 BMM GLU+quant 专用资产而改用 GMM Kernel 或直接阻塞 | Base BMM、局部 GLU 和完整行 quant 分属不同责任层 | 保留 BMM Base，并按本叶子联合合同实现精确 custom delta |

## 4. 联合验证

### 4.1 场景验证矩阵

除两个阶段各自的验证外，至少覆盖：

- 不对称 `act/gate` pattern 和用户指定的 GLU 公式；
- GLU workspace 与最终 INT8/scale 的分阶段 Golden；
- M/Q tail、multi-tile、single-pass 容量边界和 two-pass 大 Q；BMM 增加 batch
  边界，GMM 增加 empty group，MM_MX 增加其 scale/block 合同边界；
- final drain、阶段交接、inactive AIV 和交接后行重分配；
- 相同输入重复运行，以及 Host 实际选择的静态 quant entry；
- 清理诊断代码后的完整 Cube MatMul + GLU + quant 回归。

### 4.2 成立条件/门禁

- GLU workspace 和最终量化结果分别与分阶段 Golden 对比。
- 设备证据覆盖正式输出路径、阶段交接、静态 selector 和 Full 回归。
- 通用精度、同步和性能结论按通用规则导航的唯一事实源执行。

### 4.3 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| 只比较最终 INT8 输出 | 无法区分 GLU producer 与 quant consumer 误差 | 同时比较 FP workspace、scale 和最终输出 |
| 两个简单阶段各自 PASS 就宣称组合 PASS | 联合同步、row pitch 和 ownership 未覆盖 | 运行组合边界矩阵和清理后的 Full |
