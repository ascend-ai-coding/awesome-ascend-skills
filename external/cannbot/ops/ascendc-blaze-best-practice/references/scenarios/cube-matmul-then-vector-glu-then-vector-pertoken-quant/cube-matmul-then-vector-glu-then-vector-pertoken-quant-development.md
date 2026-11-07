# Cube MatMul -> Vector GLU -> Vector Per-token Quant 开发指导

本文只在同目录 design 已冻结、`implementation_route=blaze_custom` 且
`selected_scenario=cube-matmul-then-vector-glu-then-vector-pertoken-quant`
后用于编译项目 PLAN。两个简单场景的 PLAN 不得在 Step 4 临时拼接成本路线。

## 1. PLAN 输入和阅读

除基础 MatMul ABI 外，PLAN 必须闭合指定 GLU 公式、可选 bias/scale/dequant
顺序、FP workspace `[R,Q]` 及 pitch、全部 producer 完成条件、行 ownership
交接、完整行 quant 合同和 static selector。必读：

- [GLU 组件指导](../../kernel-design/glu-development.md)；
- [Per-token Quant 组件指导](../../kernel-design/per-token-quant-development.md)；
- GMM 时读取 [Grouped MatMul delta](../../kernel-design/group-matmul-delta.md)；
- DESIGN 激活的 API/RegBase Skill、同步、Tiling、Launcher 与 concrete source。

## 2. 有序动作

1. 核对 Base Assembly 和基础 ABI，仅复制 DESIGN 授权的 GLU/quant delta。
2. 适配双分支 BlockMmad、paired view 和 GLU Epilogue。纯 SwiGLU 使用
   `BlockEpilogueSwiGlu`；用户合同包含 dequant 时才使用并适配
   `BlockEpilogueDequantSwiGlu`，二者都不执行 quant。
3. GLU Epilogue 写唯一 FP workspace `[R,Q]`；冻结 dtype、pitch、alignment、
   capacity 和逐操作顺序。
4. GMM GLU custom 路线先适配
   `group_matmul_kernel_glu_fused.h` 的 GLU C+V1，再由通用
   `group_matmul_kernel_cv1_v2.h` 组合 V2：以
   `AscendC::Std::tuple<GluEpilogue, QuantEpilogue>` 占用既有
   `BlockEpilogue_` 参数，在同一 `__mix__` entry 内按 ops-transformer INT8
   输入 GMM SwiGLU quant 的阶段骨架完成 C+V1、一次 final drain、AIV-only
   `SyncAll<true, config>()` 和基于 `realM` 的完整行重分配。GLU group traversal、
   双分支 view 和 `Q=N/2` 只属于前者，通用组合层不得复制。`+` 的文件级 config
   显式指定 trigger/wait 均为 `PIPE_ALL`，不得替换成本核
   `PipeBarrier<PIPE_ALL>`；不得用同 stream 的两个 entry 替代。
5. 复制并适配 single-pass、two-pass 和共享 UB selector，为两个 quant V2
   分别实例化静态 tuple Kernel；quant Epilogue 只读 Kernel 切出的 FP
   workspace slice。
6. 逐项接线 Params、TilingData、Wrapper、entry、Launcher 和输出 buffer，
   追加 `abi_crosswalk_delta`。
7. 生成 GLU workspace、scale、INT8 输出三层 Golden。

## 3. 验证和交付

先验证 C-direct、C-through-L0C2UB、V-zero/V-known 和 GLU workspace，再验证
阶段交接、single/two-pass selector、yScale 和最终 y。覆盖 act/gate 不对称、
M/Q/K tail、multi-tile、inactive AIV、empty group/batch/MX 激活边界，以及
重复执行。只比较最终 INT8 不能作为通过证据。

清理诊断代码后 clean build 并重跑 Full。交付必须枚举唯一正式 entry、实际
selector 命中、workspace 合同和验证状态；未匹配基线时性能写
`NOT_EVALUATED`。
