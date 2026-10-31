# TileLang/PTO 热路径编码原则

1. 用 Python factory 生成模式专用 T.prim_func，消除运行时 mode 分支。
2. 用局部标量缓存重复的 task offset、tile id 和 stride 运算。
3. 把主循环与尾块拆开，主循环保持静态完整 tile。
4. 只有固定小循环使用 T.Unroll；其余使用 T.serial 或 T.Pipelined。
5. 缩短 fragment、局部向量和标量的活跃范围，降低 spill。
6. 连续规则计算放入 T.SimdVF；离散索引、复杂分支和 reduction 使用目标仓库已验证的 T.SimtVF。
7. 常驻 buffer 单版本，流水 tile 显式 annotate_buffer_versions。
8. 64 位总偏移不得为了省指令截断；只有证明范围后才降位。
9. 每次结构修改检查生成源码、精度和端到端延迟。
