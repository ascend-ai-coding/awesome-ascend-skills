# 参考算子阶梯路由（金标准四算子）

本参考在设计选型与性能优化节点按 implementation_route=tensor_api_custom 条件加载。四算子各按「入门 → 中级 → 高级/高阶」逐级优化，是官方「金标准」参考路径；按「算子类型 → 当前阶段 → 下一步优化」路由。

## 1. 阶梯总览

| 算子 | 类型 | 入门 | 中级 | 高级/高阶 |
|---|---|---|---|---|
| add_vector | 基础矢量（MemBase） | 功能 + 调试（printf/DumpTensor/msOpProf） | 标量→向量、多核均匀切分+大块搬运、双缓冲 | L2 Cache bypass、消除 UB bank conflict |
| gelu_reg_vector | RegBase 矢量 | RegBase 落地（寄存器逐步计算） | RegBase + VF 融合 | `#pragma unroll` 循环展开 |
| matmul_matrix | Cube 基础 API | 数据通路：MTE2 搬 GM→L1、MTE1 搬 L1→L0、Cube 算、Fixpipe 写回 | L1/L0 双缓冲、细粒度流水同步、多核切分、先搬 B | 大包搬运、LoadData3D、常量 Tiling、UnitFlag、L2Cache |
| matmul_leakyrelu_fusion | Cube+Vector 融合 | 融合落地 | CV 融合 GM 中转（Scenario 1）+ Matmul/Gelu 各自优化 | CV 融合 UB 直通（Scenario 2，**仅 A5**）+ Matmul 高阶 |

## 2. 优化手段 → 去向路由

按手段归类，指向 Skill 已有 reference：

| 优化手段 | 阶段 | 去向 |
|---|---|---|
| 标量→向量指令 | add 中级 | [vector-performance-patterns.md](vector-performance-patterns.md) §1 |
| RegBase + VF 融合 | gelu 中级 | [vector-performance-patterns.md](vector-performance-patterns.md) §4 |
| 循环展开 `#pragma unroll` | gelu 高阶 | [vector-performance-patterns.md](vector-performance-patterns.md) §3 |
| 多核均匀/通用切分、单核 Tiling、常量 Tiling | add/matmul | [tiling-strategy.md](tiling-strategy.md) |
| 双缓冲（L1/L0 Ping-Pong、搬算重叠） | add/matmul | [vector-performance-patterns.md](vector-performance-patterns.md) §6、[sync-and-lifecycle.md](sync-and-lifecycle.md) |
| L2 Cache bypass / L2Cache 分块 / 大包搬运 / 先搬 B | add/matmul 高级 | [api-dataflow-layout.md](api-dataflow-layout.md) |
| UB bank conflict 消除 | add 高级 | [api-dataflow-layout.md](api-dataflow-layout.md) |
| MDL、UnitFlag（MMAD-FIXPIPE 流水重叠）、LoadData3D | matmul 高级 | [performance-patterns.md](performance-patterns.md) |
| CV 融合（GM 中转 → UB 直通） | fusion | [fusion-design.md](fusion-design.md)、[performance-patterns.md](performance-patterns.md) |

## 3. A5 差异化（选型必读）

- **CV 融合 Scenario 2（UB 直通）仅 A5（Ascend 950PR）支持**：AIC 侧 Fixpipe 将 L0C 结果直写 UB，AIV 侧直接读 UB 算激活，省去 GM 中转读写。实测 `aiv_mte2_time` 从 57.145µs 降至 0.005µs（几乎归零），总耗时 2606.91 → 2573.584µs。
- A2/A3 不支持 Fixpipe 直写 UB，融合只能用 Scenario 1（GM 中转）。**在本路线（A5）下优先评估 Scenario 2**，它是融合算子的差异化收益点。
- Matmul 高阶序列（MDL → +L1Cache → +L2Cache → +Constants Tiling → +UnitFlag）是 Cube 极限打磨的固定递进，按序叠加、逐步实测，勿跳步。

## 4. 使用纪律

1. 先按算子类型找到对应阶梯，判断当前停在「入门/中级/高级」哪一级，再取「下一级」的手段——不要上来就上高级手段（如 gelu 直接循环展开，缺 VF 融合底座则收益不显）。
2. 每级优化后都要回归数值（见 [verification-and-debug.md](verification-and-debug.md) 功能精度排障），负收益也记录。
3. 官方阶梯文档的接口名与手段名（DataCopy/LoadData/Mmad/Fixpipe、UnitFlag/MDL/LoadData3D、常量 Tiling）以传统 AscendC 语境表达；本 Skill 落地用 Tensor API 侧的 `copy`/`mmad` 及其 trait/模板参数，映射见 [tensor-api-reference.md](tensor-api-reference.md)。

## 事实源

- `$ASC_DEVKIT_ROOT/docs/zh/guide/operator_practice/typical_operator_cases/simd_operator_impl/{add_vector_operator, gelu_reg_vector_operator, matmul_matrix_operator, matmul_leakyrelu_fusion_operator}/`

`$ASC_DEVKIT_ROOT` 为 ensure_docs.py 输出的 `selected_root`（默认 `~/asc-devkit`，被非 master 分支占用时为 `~/asc-devkit-master` 镜像），缺失时运行 `../scripts/ensure_docs.py` 自动获取。
