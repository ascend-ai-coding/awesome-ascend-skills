# Cube MatMul -> Vector GLU 开发指导

本文只在同目录 design 已冻结、`implementation_route=blaze_custom` 且
`selected_scenario=cube-matmul-then-vector-glu` 后用于编译项目 PLAN。Step 4
只执行 PLAN，不重新选择 GLU 公式、Base Assembly、MemBase/RegBase 或资产。

## 1. PLAN 输入和阅读

PLAN 必须引用已闭合的 `matmul_base_analysis`、基础 ABI、GLU 公式与 paired-axis
合同、L0C2UB 接口、C/V 生命周期、customization scope 和验证合同。按 DESIGN
登记以下阅读：

- [GLU 组件指导](../../kernel-design/glu-development.md)及其中实际激活的资产；
- GMM 时读取 [Grouped MatMul delta](../../kernel-design/group-matmul-delta.md)；
- 选用普通 AscendC API 或 RegBase/VF 时，分别先加载对应依赖 Skill 根入口；
- 实际使用的同步、Tiling 和 Launcher 方法，以及 Investigation 绑定的源码位置。

缺少 concrete Kernel/Block/Epilogue signature、Params 字段、物理 layout 或同步
事实时回 Step 2/3，不让 Step 4 猜测。

## 2. 有序动作

1. 只读核对当前 Blaze witness 和基础 ABI。
2. 复用 Base MM/BMM/GMM/MM_MX Scheduler/Kernel owner；仅复制 DESIGN 授权的
   GLU delta 和直接依赖到项目 `blaze_custom`。
3. 适配双分支 BlockMmad、paired N metadata、L0C2UB 和指定 GLU Epilogue；
   Block 不实现激活，Epilogue 不读取全局 group/batch 状态。
4. 在 concrete Kernel 中完成 Tensor view、C-ready/V-done、slot、inactive AIV
   和 final drain；不得用单独 Vector launch 替代正式 `__mix__` 路线。
5. 将 TilingData/Params/Wrapper/entry/Launcher 的新增或替换字段逐项映射到
   `abi_crosswalk_delta`。
6. 生成与冻结公式、dtype 和转换顺序一致的数据与 Golden。

当 `paired N` 或其他 GLU 物理 view 需要由逻辑 weight/scale 物化时，必须额外
生成独立 physical witness；Launcher 从逻辑输入重新执行同一份已冻结的 Host
adapter，并在 H2D 前对 physical value、ScaleB、shape、字节数和 gate offset 做
逐字节闭合检查。检查失败属于 Host ABI 边界失败，不得启动设备或用 Full 精度
结果掩盖。Golden 读取该次实际 H2D 的 physical bytes；具体 paired 宽度和 padding
仍由 DESIGN/PLAN 冻结。

每个 action 必须写明目标文件、来源、前置、checkpoint 和 rollback。资产原文件
与三个官方源码区始终只读。

## 3. 验证和交付

先构建真实类型链，再按 C-direct、C-through-L0C2UB、V-zero-C、V-known-C、
Full 定位最早失败边界。覆盖分支不对称、M/N/K tail、multi-tile、slot reuse，
以及需求激活的 batch/group/MX 边界。若逻辑 Q 小于 Cube 最小宽度或未对齐，
必须验证 act/gate 两半独立 padding、physical gate offset、逻辑 mask/writeback
和外部 ABI 未改变；合同允许任意正偶数 N 时必须上板最小 N，不能用实现约束
缩窄合同。清理所有诊断 entry、known-C 和 Dump 后 clean build 并重跑 Full。

交付记录必须列出最终 entry/symbol、复制并适配的资产、目标版本、支持边界、
构建/设备验证状态。未上板的组合写 `unverified`，不得从资产存在推导支持。
