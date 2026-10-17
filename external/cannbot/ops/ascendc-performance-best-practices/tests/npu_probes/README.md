# AscendC 实机代码探针

独立构建选定的知识模板，调用真实 NPU kernel，并逐元素比较 CPU 参考结果。
使用 dav-3510 的 CANN 开发环境和逻辑设备 0；运行前加载目标 SDK：

```bash
source /path/to/cann/set_env.sh
python3 ops/ascendc-performance-best-practices/tests/npu_probes/run_all.py
```

`run_all.py` 每次从当前仓库重新物化模板并编译。工程不改写 SDK、驱动或知识源文件。
平台查询返回 UB、AIV/AIC 核数、系统 workspace 和 TopK 所需 SDK 临时空间，传入实际 Host tiler。
同一时间仅运行一个设备程序。

| 探针 | 检查内容 |
|------|----------|
| Softmax | 六条路径，FP32/FP16/BF16，稳定 CPU softmax、归一化、跨 UB 重算、输出哨兵；无法装入最小 tile 的配置须被 tiler 拒绝 |
| Scan | 流式扫描与 Sklansky，非 2 次幂行数、FP32/INT32；确定性输入与 CPU 精确值比较 |
| Reg 链 | FP16/BF16、中间舍入开关、逐步 CPU 参考、尾部哨兵；paired segment 的三种 dtype |
| Conv3d DW | 当前模板的三种 dtype、Stream-K 模式、CPU 数值参考及重复运行逐位比较 |
| TopK | 七种 dtype、largest/smallest、公共/小轴/兜底路径、INT64 极值与非对齐尾部；值、索引合法性/唯一性和哨兵 |
| Gather | 末轴与 GA 全载，四种索引类型、六种末轴 payload、负索引和 I64 越界值、行列分核尾部；CPU 字节级对照及哨兵 |
| Conversion | 小形状 SIMT transpose 的 2–8 维排列，以及 tensor move；七种 dtype、CPU 字节级对照及哨兵 |
| Host tiler | Softmax / Conv 的资源约束与无效配置拒绝 |

搬运和整数路径要求精确匹配。Softmax 使用量化后输入的 double 精度稳定参考，
比较量化后的输出，容差为 `1e-7 + rtol * abs(ref)`，FP32/FP16/BF16 的
`rtol` 分别为 `3e-4/0.002/0.016`，另检查归一化误差不超过 `0.02`。
其他浮点算子的容差及舍入语义定义在各自探针源码中。

输出 `manifest.json` 记录仓库 commit、工作区 patch 哈希及物化源文件 SHA256；
`results.jsonl` 记录命令、退出码、耗时及日志路径。构建失败、精度超差、越界哨兵变化，
或 TopK SDK 查询无法解析时，整个流程失败。

本工程覆盖表中列出的路径与输入集。`prepare.py` 物化文件不表示该文件已实例化验证；
未覆盖的模板、注释摘录、所有 shape/dtype 组合和性能收益需另外验证。
