# Cube MatMul -> Vector Per-token Quant 开发指导

本文只在同目录 design 已冻结、`implementation_route=blaze_custom` 且
`selected_scenario=cube-matmul-then-vector-pertoken-quant` 后用于编译项目 PLAN。
Step 4 不重新决定量化公式、完整行宽度、UB 路线或同步协议。

## 1. PLAN 输入和阅读

PLAN 必须绑定最终 FP workspace `[R,N]`、物理 pitch、producer/consumer 集、
完成依赖、完整行 scale 公式、RNE/clamp/saturation 合同、static entry selector
和基础 MatMul ABI。必读：

- [Per-token Quant 组件指导](../../kernel-design/per-token-quant-development.md)；
- GMM 时读取 [Grouped MatMul delta](../../kernel-design/group-matmul-delta.md)；
- Investigation 指定的 Base Kernel/Fixpipe、同步、Tiling 和 Launcher 来源；
- quant 实现实际使用的 AscendC 或 RegBase 依赖 Skill 根入口及必要叶子。

## 2. 有序动作

1. 核对 Base MatMul final output；Fixpipe 可直接写 FP workspace 时不增加
   identity Vector Epilogue。
2. 按 DESIGN 冻结的依赖选择并实现 `__mix__` 或同一 ACL stream 上的
   producer/consumer split；先闭合 producer 完成、final drain、消费者可见性和
   完整逻辑行 ownership，再进行性能选择。
3. 复制并适配 single-pass、two-pass 及共享 selector。`MakeUbLayout()` 是 UB
   容量唯一事实源；Host 选择独立静态 entry，不修改现有序列化 Tiling ABI。
4. Kernel 切出带真实 pitch 的 workspace/y/yScale Tensor slice；quant
   Epilogue 只使用 slice-relative row。
5. 把量化参数、workspace 字节、entry、拓扑参与者和 Launcher buffer 映射到
   `abi_crosswalk_delta`；Host 和 device 消费同一合同。
6. 生成同源 Golden 中间值和最终结果；half-tie 分类使用生成 Golden `y` 的
   同一个内存 `normalized` tensor。

## 3. 验证和交付

覆盖 single-pass 容量边界前/等于/后、two-pass 大 N、多 chunk/tail、零行、
tiny、饱和、RNE 和需求规定的非有限值行为；每个 case 记录实际 entry。RNE
mismatch 必须依次比较 workspace、yScale 和 quant 重算结果。相同输入重复运行
次数由 DESIGN 冻结；对 MIX 还要记录实际 physical-to-logical owner 映射，
对 split 还要记录同 stream 顺序和 workspace 可见性；清理诊断后 clean build
并重跑 Full。

交付分别声明 single-pass 与 two-pass 的正确性和性能状态。没有 matched
baseline 时性能写 `NOT_EVALUATED`；不得以 single-pass 优化删除任意宽度的
two-pass 能力。
