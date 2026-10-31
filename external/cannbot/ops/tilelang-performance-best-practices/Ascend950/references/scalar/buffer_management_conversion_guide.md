# PTO Buffer 管理

UB 使用 T.alloc_shared，寄存器/线程局部使用 T.alloc_fragment，Cube 路径使用 T.alloc_l1、T.alloc_l0a/b/c。T.copy 与 T.dual_copy 负责目标仓库已验证的层级搬运。

流水 tile 的物理字节为元素数×dtype 字节×版本数；再加入 padding、resident 数据与安全余量。常驻权重和 reduction 状态不参与多版本。只有 lifetime 不重叠的 buffer 才允许依赖后端 merge/reuse，并通过生成代码确认。

尾部 GM 只访问 valid，UB 仍按完整寄存器 footprint 分配。每个 task 重新初始化可变状态。
