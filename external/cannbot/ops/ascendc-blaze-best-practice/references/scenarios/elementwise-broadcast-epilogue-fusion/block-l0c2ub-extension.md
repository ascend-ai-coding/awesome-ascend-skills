# Block层 L0C 输出与扩展

本文是本场景 Block层专题。Step 3 用它设计 Block delta；DESIGN 冻结后，Step 3 用它编译对应 PLAN action；Step 4 只有在 PLAN 将本文绑定到当前 action 时才读取。本文不选择 MatMul Blaze 组装方案，也不默认要求 custom Block。

## 1. 输入与输出合同

输入：`matmul_base_analysis`、Investigation 中已闭合的相关源码事实、公式/broadcast 合同和当前 Block witness。输出：

```text
block_output_contract
block_copy_contract
final_partial_contract
selected_implementation_boundary
block_validation_additions
```

每个结论必须引用当前 候选组装方案评估、witness 和 evidence IDs。历史 `source_observed`/`device_verified` 只有在 Investigation、Blaze 组装方案、构建和测试范围一致时才能复用；否则标记 `unverified`。

## 2. 三种范围必须分开

| 对象 | 必填事实 |
|---|---|
| L0C 源 | dtype、location、layout、逻辑/物理 shape、有效范围、Copy extent |
| UB 目的 | dtype、layout、row pitch、alignment、slot 起点/容量、sub 可见性 |
| GM 输出 | final `curM/curN`、full shape、offset/stride 单位、有效写回范围 |

Copy extent、UB pitch 和 GM 有效列是三个不同量。任何 Slice、Pad、alignment 或 view 都必须说明作用于源还是目的；不能从相邻 MX/Quantized/Grouped Block 推导。

## 3. 输出生命周期

DESIGN 必须回答：

1. 当前 tile 是否已完成全部 K、workspace 和跨核归并；
2. final/partial 的判定位置和完成时机；
3. L0C2GM、L0C2UB 或其他目的路径的实际选择条件；
4. SplitM/非 SplitM 使用的 concrete Copy API/trait 和物理 view；
5. Kernel 构造的 UB Tensor 与 Epilogue 本地 C 起点是否使用相同 offset 单位和范围；
6. C-direct-GM 使用同一真实 GM 分支还是等价纯 MatMul输出路径。

普通 Vector Epilogue 只能消费 DESIGN 已证明为 final 的 MatMul 输出。Partial/reduction 路径没有闭合时阻塞。

## 4. Official/Custom 决策

优先直接使用当前 witness 的官方 Block。只有全部满足时才授权 custom：

- 目标输出合同由 Investigation 证明缺失或冲突；
- 最早失败已经定界到 Block Copy/输出边界；
- 候选修改只改变一个可说明的 Block层变量；
- DESIGN 记录来源、项目目标、独立符号/namespace、首个修改点和保留不变量；
- PLAN 能形成原实现负向、单变量正向及最终 Full 回归。

原实现与候选都通过时只能记录“已测范围等价”，不能宣称候选必要或是根因。不得修改官方源码或依赖 include 顺序覆盖 specialization。

## 5. PLAN 动作要求

若使用官方 Block，PLAN 只登记 concrete 绑定、ABI/Copy 核对和 checkpoint。若授权 custom，PLAN 必须显式登记：

1. 只读来源与项目副本路径；
2. 保持 Mmad、调度、bias/scale、K 循环、final/partial 和未变输出路径的动作；
3. 仅修改 DESIGN 列出的 trait/view/extent/分支；
4. Policy、Params、Kernel include/type chain 的接线；
5. 结构检查、构建、隔离模式、单变量对照和清理回归。

任何新增文件或接线若需要越过项目根或 DESIGN forbidden scope 时返回 Step 3；项目根内的实现补充由 Step 4 处理并记录。

## 6. 验证门禁

- C-direct-GM 与 C-through-fusion 使用同一 MatMul 语义、输入和 Tiling；
- row/column-encoded 数据能区分 M/N、pitch 和 offset 错位；
- 覆盖 DESIGN 声明的 alignment/tail、odd/even、single/multi-tile 和 slot reuse；
- SplitM 只在 DESIGN 激活时执行其 empty sub 和地址矩阵；
- 根因声明同时具备稳定负向、单变量正向和清理后 Full 回归；
- 官方源码区与 Blaze Asset 原文件保持零改动。

具体 shape、ratio、slot 和重复次数来自 DESIGN/PLAN，不由本文固定。
