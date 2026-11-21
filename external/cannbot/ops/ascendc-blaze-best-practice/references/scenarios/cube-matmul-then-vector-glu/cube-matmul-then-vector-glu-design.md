# Cube MatMul -> Vector GLU

本 Scenario 描述 MM、BMM、GMM 或 MM_MX 的 Cube MatMul 后执行 GLU-family
计算的完整路线：

```text
Cube MatMul (MM / BMM / GMM / MM_MX)
    -> L0C2UB
    -> Vector: specified GLU-family formula
    -> y [R, Q]
```

本页在 [Scenario 索引](../index.md)唯一命中后读取
[GLU / SwiGLU 组件开发指导](../../kernel-design/glu-development.md)，只描述
不含后续 quant 的 GLU-only 路线及其联合合同，不作为其他 Scenario 的事实源。
GMM 相对 MM 的 group 编码、完整 POD、Scheduler/Kernel 生命周期和地址规则统一
遵循 [Grouped MatMul 项目侧差异](../../kernel-design/group-matmul-delta.md)。

## 0. Step 3 输入与源码前提

进入正文前必须具备 `requirements_contract`、`operator_interface_contract`、
`native_gaps`、`matmul_base_analysis` 及其 source-backed `abi_bindings[]`，以及
同次 Investigation 对基础 Assembly、双分支输出、L0C2UB、Kernel/Epilogue
adapter、Tiling/Params 和 C/V 生命周期的事实。GMM/BMM/MM_MX 只在需求激活时
追加对应 group/batch/MX 事实，不从本场景资产推导。

缺少决定性事实时只允许向 Step 2 提出一次无场景名补充问题，记录受影响
requirement IDs、待确认的物理接口/完成时机、有限源码前沿和阻塞原因；补充后
仍缺失则停止，不生成 PLAN。使用普通 AscendC API 或 RegBase/VF 时，分别先
加载对应依赖 Skill 根入口。

DESIGN 必须区分 consumed/preserved/added contracts，并用
`abi_crosswalk_delta` 只追加 paired operand、输出、Params、entry 和同步接线；
每个 custom 文件都要有来源、首个修改点、保持不变量和验证门禁。

## 1. 精确匹配条件

### 1.1 设计主线

`scenario_id`: `cube-matmul-then-vector-glu`。

仅当需求同时满足以下条件时选择本 Scenario：

- 基础计算是 MM、BMM、GMM 或 MM_MX；对应的 batch/group/MX scale、layout、
  dtype 和地址合同已确认。
- 用户明确给出 GLU-family 公式和两路分支语义，例如 GLU、Bilinear、ReGLU、
  GEGLU 或 SwiGLU。
- 将全部逻辑行轴展平为 `R` 后，GLU 输出逻辑 shape 为 `[R,Q]`。
- 需求不包含后续独立的 per-token quant 或其他未命名 Vector 阶段。

若 GLU 后还需要 per-token quant，返回 [Scenario 索引](../index.md)选择已登记
的完整组合路线；不能把两个叶子自由拼装。

### 1.2 成立条件/门禁

- `act/gate` 分支、activation 精确公式、两路 weight/bias/scale 布局、逐操作
  dtype 和输出 `Q` 必须由用户合同或 Golden 冻结。
- 只有两路各宽 `Q` 且输入拼接宽度明确为 `2Q` 时，才能使用 `Q=N/2`。
- `integration_mode`、唯一 Base Assembly owner 和目标版本绑定由 Investigation
  证据确认。
- 对 BMM，若同版本 Investigation 已绑定可复用 Base BMM/QBMM
  Scheduler/Kernel，且缺口可精确写成双分支 Block、paired N metadata view、
  L0C2UB/CV 生命周期和 Epilogue row adapter，则选择
  `blaze_base_plus_custom_delta`，这些缺口为 `adaptation_required`。Scenario
  不要求预先存在一个专用 BMM GLU Kernel 资产。
- 基础组件缺失且本 Scenario 或公共组件指导没有登记精确 delta 时，记录
  `extension_missing` 并阻塞。

### 1.3 常见误用

| 误用 | 原因 | 正确处理 |
|---|---|---|
| 根据 `SwiGLU` 类名默认分支顺序或近似公式 | 名称不能替代用户 Golden | 在 `DESIGN.md` 冻结公式、dtype 和分支 |
| 从 GMM 参考资产推导普通 MM 缺少 Scheduler/Kernel | custom 资产不是上游能力表 | 从同版本 Investigation 绑定普通 MM 基础组件 |
| 因资产目录没有 BMM GLU Kernel 就阻塞 | BMM Base 与局部 GLU 公式属于不同责任层 | 复用同版本 BMM Base，目标工程适配 paired view、MX Block、C/V 和 row base |
| 将 GLU-only 叶子与 Quant 叶子组合 | 两阶段 workspace、同步和 ownership 未联合设计 | 选择已登记的 GLU+Quant Scenario |

## 2. 不可拆分合同

### 2.1 数学合同

将两路投影记为 `act` 和 `gate`：

```text
act  = MatMul(A, B_act)
gate = MatMul(A, B_gate)
y    = activation(act) * gate
```

所有沿源 N 轴定义的输入和元数据必须使用相同 `qOffset/curQ` 成对切分。M 轴或
K 轴输入不会仅因 GLU 自动复制。GLU 公式、双分支 view、BlockMmad、Epilogue
和 dtype 适配的详细合同统一由
[GLU / SwiGLU 组件开发指导](../../kernel-design/glu-development.md)定义。

### 2.2 唯一正式架构

本 Scenario 的正式架构只有 L0C2UB：

```text
Host/Tiling
    -> selected Base Assembly and GLU delta
Scheduler
    -> problem/group/tile ownership
Kernel
    -> BlockMmad(A, B_act, B_gate)
    -> L0C2UB
    -> GLU Epilogue(act, gate)
    -> y
```

L0C2GM 只用于 C-direct 诊断，不属于正式实现或 fallback。任一正式接口为
`unsupported` 或 `unverified` 时阻塞，不得临时改成 Cube 写 GM workspace、
第二个 Vector Kernel 执行 GLU。

### 2.3 组件责任

- Scheduler 只分配 problem/group/tile，不解释 `act/gate` 公式。
- Kernel 构造 `B_act/B_gate` 和成对 N 轴 Tensor view，管理 C/V 生命周期。
- BMM Kernel 继续拥有 batch loop/decode/offset；通用 Epilogue 只接收映射后的
  batch-local tile 或展平 row base，不读取 batch Scheduler。
- BlockMmad 复用 A tile 完成两路矩阵乘，只交付 final C。
- GLU Epilogue 只消费 Kernel 已切出的局部 `act/gate` tile。
- Host/Tiling 绑定完整 ABI、资源门禁和静态 `__mix__` entry。

## 3. 资产适配

本 Scenario 不复制公共资产表。GLU Block、Epilogue、Grouped delta、同步 Policy
和 TilingData 的唯一登记位置是
[GLU / SwiGLU 组件开发指导](../../kernel-design/glu-development.md)。

目标工程只复制 `DESIGN.md` 选择的 custom delta 及必要直接依赖。连续 ND/DN
weight 由当前 Grouped GLU Kernel 从 `BlockMmad::LayoutB` 派生；NZ/ZN、独立或
交错 weight、双 bias、双 MX scale 以及非 SwiGLU 公式均按公共指导标记
`adaptation_required`，并由目标工程完成对应接口、Golden、编译和设备验证。

## 4. 验证

至少覆盖：

- `act/gate` 不对称 pattern 和用户指定的 GLU 公式；
- base、M/N/K tail、odd/even M、multi-tile 和 slot reuse；
- GMM 的空 group、prefix offset 和 group tail；
- MM_MX 的 K/scale window、padding poison 和 Q alignment 边界；
- 正式 L0C2UB 路线对应的 C-direct、C-through-L0C2UB、V-zero-C、V-known-C
  和 Full；
- 清理诊断代码后的 clean build、重复运行和正式输出路径。

精度异常按
[GLU 精度诊断](../../kernel-design/glu-precision-diagnosis.md)定位；通用同步、精度
指标和交付门禁按 Scenario 通用规则导航的唯一事实源执行。
