# Cube MatMul -> Vector Per-token Quant

本 Scenario 描述 MM、BMM、GMM 或 MM_MX 的 Cube MatMul 结果经过正式交接后，
由 Vector 阶段执行 per-token quant 的完整路线：

```text
Cube MatMul (MM / BMM / GMM / MM_MX)
    -> FP workspace [R, N]
    -> Vector: per-token quant
    -> y [R, N] + yScale [R]
```

本页在 [Scenario 索引](../index.md)唯一命中后，使用
[Per-token Quant 组件开发指导](../../kernel-design/per-token-quant-development.md)实现
完整行归约、single/two-pass、量化参数、阶段交接和精度诊断。本页只描述
Cube MatMul 直接连接 per-token quant 时的场景边界和集成 delta。

GMM 相对 MM 的 group 编码、完整 POD、Scheduler/Kernel 生命周期和地址规则
统一遵循
[Grouped MatMul 项目侧差异](../../kernel-design/group-matmul-delta.md)。

## 0. Step 3 输入与源码前提

进入正文前必须具备 `requirements_contract`、`operator_interface_contract`、
`native_gaps`、`matmul_base_analysis` 及其 source-backed `abi_bindings[]`，以及
同次 Investigation 对 MatMul final FP 输出、Fixpipe/GM workspace、Kernel
参与者、完成依赖、Tiling/Params 和静态 entry 接线的事实。

缺少决定性事实时只允许向 Step 2 提出一次无场景名补充问题，记录受影响
requirement IDs、待确认的 workspace 物理合同/完成时机、有限源码前沿和阻塞
原因；补充后仍缺失则停止，不生成 PLAN。quant 代码实际使用普通 AscendC API
或 RegBase/VF 时，分别先加载对应依赖 Skill 根入口。

DESIGN 必须区分 consumed/preserved/added contracts，并用
`abi_crosswalk_delta` 只追加 workspace、量化参数、静态 entry、输出和同步接线；
每个 custom 文件都要记录来源、首个修改点、保持不变量和验证门禁。

## 1. 精确匹配条件

### 1.1 设计主线

`scenario_id`: `cube-matmul-then-vector-pertoken-quant`。

仅当需求同时满足以下条件时选择本 Scenario：

- 基础计算是 MM、BMM、GMM 或 MM_MX；对应的 batch/group/MX scale、layout、
  dtype 和地址合同已确认。
- Cube MatMul 的逻辑输出展平为 `[R,N]`；MM/GMM 通常 `R=M`，BMM 通常
  `R=B*M`。
- Cube MatMul 与 quant 之间没有 GLU、elementwise、独立 Vector dequant 或
  其他未命名计算。量化 MatMul 在 Cube/Fixpipe 输出侧完成的 accumulator→FP
  转换属于 MatMul 最终输出合同，不增加独立 Vector 阶段。
- per-token quant 对每个完整逻辑行 `[0,N)` 生成唯一 scale。
- scale、零行、舍入、clamp、饱和范围和 NaN/Inf 行为均已由用户合同或 Golden
  冻结。

若 MatMul 与 quant 之间还存在 GLU-family 或 shape-preserving elementwise，
必须分别选择对应的完整组合 Scenario，不能把本 Scenario 与其他叶子自由拼装。

### 1.2 成立条件/门禁

- `integration_mode`、唯一 Base Assembly owner 和目标版本绑定必须由
  Investigation 证据确认。
- Cube 输出的 accumulation dtype、workspace dtype、逻辑 `[R,N]`、物理 row
  pitch 和完成条件必须闭合。
- quant 组件的输入必须是最终 MatMul 结果，不得是尚未归并的 SplitK/StreamK
  部分和。
- 参考资产只是适配起点，不证明目标版本或目标工程已经支持本 Scenario。

### 1.3 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| 已有任意 FP workspace 就选择本 Scenario | 缺少本 Scenario 必需的 Cube MatMul producer | 纯 Vector 请求不属于本 Skill 的 Scenario |
| MatMul 后还有 elementwise 仍选择本 Scenario | 名称没有表达实际中间阶段 | 选择对应的完整组合 Scenario |
| 将 quant 组件文档登记为 Scenario | 组件事实源被误当成完整路由 | 注册表只登记从 Cube 开始的完整路线 |

## 2. 接口与数据流

### 2.1 Cube 输出

Cube 阶段必须产生逻辑 `[R,N]` 的最终 FP 结果。若 Fixpipe 能直接写 GM
workspace，优先使用正式直写路径；不得为了接入 quant 人为增加没有语义的
identity Vector Epilogue。

`DESIGN.md` 必须冻结：

- MM/BMM/GMM/MM_MX 的基础变体及其输入、layout 和 scale 合同；
- accumulator 到 FP workspace 的转换顺序和 dtype；
- `R`、逻辑 `N`、物理 row pitch 和 workspace 容量；
- 每条逻辑行的 Cube producer 集与 Vector consumer 集；
- workspace 完成信号、final drain 和 quant 前的 ownership 重分配。

### 2.2 Quant 组件

quant 阶段的数学合同、Params、single/two-pass 静态选择、同步依赖、行切片和
RNE 边界诊断统一由
[Per-token Quant 组件开发指导](../../kernel-design/per-token-quant-development.md)定义。
目标算子必须按 Golden 显式选择饱和范围，不得由 Skill 资产替用户决定。

## 3. 组件与资产适配

### 3.1 设计主线

- Cube Block、Scheduler、Kernel 和 Tiling 以目标版本 Investigation 报告为
  事实源。
- quant Epilogue 和 Host selector 以组件开发指导登记的资产为适配起点。
- Kernel 负责 producer→consumer 交接、workspace Tensor slice 和逻辑行 ownership；
  quant Epilogue 只消费已切分的 slice。
- DESIGN 必须冻结实际拓扑：可以是已证明 ownership 的单一 `__mix__` entry，也
  可以是同一 ACL stream 上的独立 AIC producer 与 AIV consumer。MIX 的物理 block
  index、task ratio 和参与者集合必须有源码或设备证据；没有证据时不得假设稠密
  logical rank。候选方案先证明 workspace 依赖和完整行 ownership，再用同设备
  性能数据选择，不能预先规定 MIX 或 split 永远最优。

### 3.2 成立条件/门禁

- 不得用 GMM 专用 Scheduler、POD 或 selector 证明 MM/BMM/MM_MX 已支持。
- Host、Tiling、Kernel、Epilogue 和 Launcher 必须共同消费同一逻辑 `N`、pitch
  和量化参数。
- 选定拓扑的参与者集合、完成依赖、inactive AIV、final drain 和行重分配必须由
  目标工程设备证据闭合；同 stream split 还必须证明 producer kernel 完成后
  consumer kernel 才读取 workspace。

## 4. 验证矩阵

必须同时覆盖：

- MM/BMM/GMM/MM_MX 中目标需求实际包含的基础变体边界；
- base、M/N/K tail、multi-tile、空 group/零行及重复运行；
- quant single-pass 容量边界和 two-pass 大 `N`/chunk tail；
- 全零、tiny、大正负值、舍入半整数附近和 INT8 saturation；
- Cube workspace、`yScale`、最终 `y` 的分阶段 Golden；
- 宽范围随机端到端输入，以及只用于隔离 quant 的精确二进制小数输入；
- 实际命中的静态 kernel entry 和同步方案性能证据。

任何 RNE mismatch 必须按组件开发指导导出 workspace、scale 并分阶段归因；
位级合同未闭合时不得通过缩窄输入分布声明通过。
