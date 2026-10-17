# `two-pass-amax-cast`

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

## 解决的问题

同一份 UB 输入先做 amax/scale，再做量化输出。`vregDupScale` 在两个 VF pass 之间传递，load 阶段融合 B16→FP32 和可选 smooth，输出阶段把 Div、Cast 和 Pack 连在寄存器中。

## 为什么这样写

- scale 依赖整行 amax/min-max，所以量化不能在统计量完成前开始；两遍扫描是数据依赖决定的，不是无效重复。
- 两遍放在同一个 `__VEC_SCOPE__`，用 `vregDupScale/vregDupOffset` 直接传值；写 UB 再读回只会增加搬运。
- 对称量化的无效 lane 可由 abs/max 初始化控制；非对称 min/max 的尾拍必须单独 masked reduce，不能让补零参与最小值。
- int8、int4、FP8 的 CastTrait 和 Pack 链不同，它们定义实际舍入与物理布局，不能统一成凭记忆写出的单次 Cast。

## 怎么使用

- 本地注解代码：[`code/two_pass_amax_cast_vf.h`](code/two_pass_amax_cast_vf.h)
- 重点函数：`DataCopyInputVF`、`ComputeScaleVF`、`ComputeYVF`、`ComputeVF`
- 源构建契约：`unverified`；本地摘录不是独立类，当前无最终编译命令

1. 先匹配目标的 scale 公式、对称分支、smooth 语义和输出 dtype。
2. 离线直接读取本地摘录，按 `DataCopyInputVF → ComputeScaleVF → ComputeYVF → ComputeVF` 的整体关系迁移；从目标工程补齐 CastTrait、dtype 常量、Tiling 和队列，不猜默认值。
3. 保持两个 pass 位于同一 `__VEC_SCOPE__`，并保留尾拍 mask、非对齐 scale 回写的 post 调用和目标 dtype 的 Cast/Pack 链。
4. 先做单 dtype 最小编译，再覆盖零 scale、极值、尾拍、smooth 开关及各输出 dtype 的精度测试。

## 前置条件

- Arch35 / DAV_3510。
- scale 公式、对称/非对称分支、目标 dtype 的 CastTrait 和 Pack 布局必须与目标接口一致。
- amax 累积的 full mask、最后一拍的 tail mask 和 scale 的零值行为必须做精度验证。

这是 `annotated_excerpt`：关键真实指令链已随库提供，网络受限时仍可分析和实施；但它不是独立可编译类。必须从目标工程补齐上下文并重新验证，不能把注释中的省略分支当成完整实现。
