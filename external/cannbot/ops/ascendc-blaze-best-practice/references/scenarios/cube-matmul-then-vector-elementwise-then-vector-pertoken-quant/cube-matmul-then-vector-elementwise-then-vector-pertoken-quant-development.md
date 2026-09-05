# Cube MatMul -> Vector Elementwise -> Vector Per-token Quant 开发指导

本文只在同目录 design 已冻结、`implementation_route=blaze_custom` 且
`selected_scenario=cube-matmul-then-vector-elementwise-then-vector-pertoken-quant`
后用于编译项目 PLAN。本场景明确排除双分支 GLU 和跨元素 reduction。

## 1. PLAN 输入和阅读

PLAN 必须绑定 shape-preserving elementwise/broadcast DAG、逐操作 dtype 和近似、
FP workspace `[R,N]` 及 pitch、producer/consumer 完成依赖、完整行 quant
合同、Base MatMul ABI 和 static selector。按 DESIGN 读取：

- [Per-token Quant 组件指导](../../kernel-design/per-token-quant-development.md)；
- GMM 时读取 [Grouped MatMul delta](../../kernel-design/group-matmul-delta.md)；
- 选定的 MemBase/RegBase 依赖 Skill、当前 API/公式参考实现；
- concrete Block/Kernel、同步、Tiling 和 Launcher 来源。

## 2. 有序动作

1. 只读核对 Base Assembly 和 ABI，选择一个已经在 DESIGN 冻结的公式实现。
2. 适配公式专用 Epilogue，保持 `[R,N]`；不得把 GLU 资产改名成 unary
   Epilogue，也不得把 reduction 隐藏在 elementwise DAG 中。
3. Epilogue 写 FP workspace；已有 C+V1 Kernel 完成 final drain 后，可由通用
   `group_matmul_kernel_cv1_v2.h` 在同一 `__mix__` entry 内完成全局交接，再按
   完整行重新分配 AIV。该组合层不拥有具体 elementwise 公式。
4. 复制并适配 quant single-pass、two-pass 和共享 UB selector；quant
   Epilogue 只消费 Tensor slice。
5. 将额外 broadcast operand、workspace、量化参数、entry 和输出映射到
   `abi_crosswalk_delta`，并生成 workspace/scale/y 分阶段 Golden。

每项动作写明准确目标文件、来源、前置、checkpoint、rollback 和清理。Step 4
发现公式、物理 layout 或同步事实缺失时回 Step 2/3，不扩展场景范围。

## 3. 分阶段诊断可信度

DESIGN 若冻结 staged diagnostic，每个 mode 都必须写清：

```text
executed_stages
stopped_or_bypassed_stages
output dtype/shape/layout/row pitch
output file and lifecycle
matched Golden
verifier and pass criteria
lane ownership and synchronization
```

模式名称不构成证据。一个名为 `c-direct` 的模式若没有实际写出 raw C，或者
verifier 只返回 blocked，就不能证明 Cube 输出。诊断路径如果改成单一 AIV
处理全部 M、改变 Split-M、row pitch、barrier 或 GM store 方式，也不能与正式路径
比较并归因。

每个 staged diagnostic 必须先在一个已知正确、对齐且 full-tile 的 case 上校准，
证明输出和 verifier 能识别正确结果，再用于失败 case。推荐按以下边界逐层推进：

```text
raw MatMul C
  -> dequant/elementwise 后 FP workspace
  -> per-token yScale 和 y
```

每一层都保持与正式 MIX entry 相同的 AIC/AIV lane 所有权、Split-M、物理 layout、
同步和输出生命周期，只改变 DESIGN 明确冻结的观察点。校准失败时先修复或废弃诊断，
不得据此判定 BlockMmad、Epilogue 或 Quant。

### 3.1 Elementwise 物理数据路径门禁

Elementwise 若在 producer 的 VF/UB 中直接消费 accumulator 或 dequant 结果，必须
记录其寄存器/UB layout、masked tail 和最终 workspace store。若设计为独立的 GM→UB
→GM pass，必须另外冻结并验证：

```text
source/destination GM address
logical row pitch and byte stride
UB layout and padding
DataCopy/DataCopyPad specialization
valid element and tail range
lane ownership and completion dependency
```

逻辑 Tensor 能表达 `[R,N]` 不等于该 Copy 路径能消费所有物理行步长。特别是
`N*sizeof(dtype)` 不是设备搬运对齐粒度倍数时，必须使用同设备、同地址形态的
正例证明下一行起始地址和有效范围；没有证明时应回到 producer VF/UB 融合或已证明
的物理搬运路径。若 MMAD/dequant workspace 已通过而最终结果失败，首要检查该
elementwise 数据路径的地址、pitch、padding 和 Copy specialization，不要先修改
elementwise 等价公式。

定位 shape 问题时采用正交矩阵：每次只改变 M/N/K/group 中一个变量，并比较多个
失败 case 的共同因素。case 标签或最显眼的 tail 不能替代因果证据。

## 4. 验证和交付

覆盖公式近似和 broadcast 映射、M/N/K tail、multi-tile、single-pass 容量边界、
two-pass 大 N/chunk tail、RNE/saturation、inactive AIV，以及需求激活的
batch/group/MX 边界。分别比较 FP workspace、yScale 和 y；清理诊断后 clean
build 并重跑 Full。任何 staged diagnostic 的结论必须先通过第 3 节可信度门禁。

发生 elementwise 或 quant mismatch 时，必须按 `workspace -> yScale -> y` 的顺序
保留中间结果。只有 workspace 已经与 Golden 对齐，才能把问题缩窄到 quant 的
division/cast/RNE；只有 yScale 也已对齐，才可讨论最终 INT8 half-tie。不得通过
替换用户合同中的等价 GELU 表达式来掩盖未闭合的地址或数值证据。

最终交付列出公式、正式 entry、selector 命中、支持/拒绝边界和证据状态。
未上板组合写 `unverified`，无 matched baseline 时写 `NOT_EVALUATED`。
