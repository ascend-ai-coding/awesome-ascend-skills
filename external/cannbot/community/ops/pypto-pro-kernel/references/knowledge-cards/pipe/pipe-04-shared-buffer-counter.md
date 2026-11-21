---
type: "PyPTO Performance Optimization Card"
title: "共享地址 Tile Group 用显式共享计数器管理 Buffer 下标"
description: "多组 make_tile_group 共享同一物理地址空间时，用每个地址空间一个显式递增计数器替代各组独立的 .next() 游标，消除不同流水线阶段调用次数不一致导致的游标错相与 buffer 冲突。"
status: "stable"
tags: ["pypto-pro", "pipeline", "tile-group", "scheduling", "double-buffer"]
item_id: "pipe-04"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/crosscore-sync.md#keep-slot-selection-visible-to-the-stage-scanner"
      scope: "共享地址空间的显式计数器要与实际槽位选择一致，并让阶段扫描器能够追踪被选择的 Tile。"
      boundary: "目标的可追踪性不证明计数器相位正确；本卡每空间单游标、递增点、depth 与跨 task 生命周期仍需人工核验。"
bound_hint: "PIPELINE"
applicability: "同一地址空间（相同 addrs 与 mutex_ids）被多组 tile group 共享，且不同流水线阶段（如 QK 与 PV、drain 扩展轮、按位图跳块）对这些组的调用次数不一致"
target_api_gate: "仅限 Ascend 950PR 或 950DT；仅使用 pl.make_tile_group 的下标取用 group[idx] 与 Python 标量计数器，已在当前工具链 kernel 中核验；其它 SoC 重新查表并验证"
---
# 技术卡片 pipe-04：共享地址 Tile Group 用显式共享计数器管理 Buffer 下标

- **适用 bound**：VEC/Cube 联合流水线的 Tile 分配（L0A/L0B/L0C 等共享地址空间）
- **一句话**：多组 make_tile_group 共享同一物理地址空间时，同一空间用一个显式递增计数器管理所有组的 buffer 下标，取代各组独立的 `.next()`。

## 何时用（诊断特征）

- 源码中多组 `make_tile_group` 定义了相同的 `addrs` 列表与相同 `mutex_ids`（如 L0A/L0B 的 `[0, 32768]`、L0C 的 `[0, 65536, 131072, 196608]`），且分别服务不同流水线阶段（如 QK 用一组、PV 用另一组）。
- 各阶段对这些组的分配调用次数天然不一致：drain 扩展轮、按 block 位图跳过部分块、PV 相对 QK 延迟一步等软件流水结构。
- 现象：偶发结果错误或数据被覆盖，与时序/负载分布相关、难以稳定复现；mutex 串行化保证了访问互斥，但槽位选择行为不可预期。

## 何时不适用

- 某地址空间只有一组 tile group 使用（独占），独立 `.next()` 游标不存在错相问题，无需改造。
- 不能用 `task_id % depth` 直接取余：QK 与 PV 在同一 task 内、`task_id` 相同，两组会选中同一槽位。
- 每个地址空间必须拥有独立的计数器：不同地址空间是相互独立的物理资源，槽位消耗节奏各自独立；即使 depth 相同也不能共用同一计数值，否则一个空间的分配会推进另一个空间的游标相位。各空间按各自空间的 depth 取模，与深度是否相同无关。

## 原理

`.next()` 每调用一次游标 +1。同一地址空间的多组 tile group 各自独立前进，当 QK 与 PV 的调用次数因 drain 等流水线扩展阶段不同步时，游标相位差随时间累积，无法保证同一时刻两组选中不同的 buffer 槽——mutex 只约束访问时序，不约束槽位选择。

修复：同一地址空间的所有 tile group 共用一个显式计数器 `buf_idx`，每次分配后按该空间深度递增取模（`buf_idx = (buf_idx + 1) % depth`）。同空间所有组的槽位选择绑定到单一游标：同一次取用相位一致、跨阶段严格错开，与历史调用次数无关。计数器按"每个地址空间一个"划分，而不是按组或按深度划分。

附带收益：流水线对齐不再需要插入空的 `.next()` 同步调用；空 `.next()` 会占用 MTE1 队列造成 stall。

## 怎么改（before / after）

以下为嵌入片段：`group[idx]` 下标取用写法已在当前工具链 kernel 中核验，组定义参数省略。

错误写法（各自 `.next()`，游标独立，错相后可能选中同一 buffer）：

```python
left_db = pl.make_tile_group(..., addrs=[0, 32768], mutex_ids=[6, 7])
left2   = pl.make_tile_group(..., addrs=[0, 32768], mutex_ids=[6, 7])

# QK 阶段
qk_left = left_db.next()
# PV 阶段（调用次数可能与 QK 不同）
pv_left = left2.next()
```

正确写法（共享计数器，显式递增）：

```python
buf_a = 0  # L0A (Left)：left_db / left2，depth 2

# QK 阶段
qk_left = left_db[buf_a]
buf_a = (buf_a + 1) % 2

# PV 阶段
pv_left = left2[buf_a]
buf_a = (buf_a + 1) % 2
```

多地址空间完整示例：

```python
buf_a = 0  # L0A (Left)：left_db / left2，depth 2
buf_b = 0  # L0B (Right)：right_db / right2，depth 2
buf_c = 0  # L0C (Acc)：acc_db / acc2，depth 4

# QK 阶段
qk_left = left_db[buf_a];   buf_a = (buf_a + 1) % 2
qk_right = right_db[buf_b]; buf_b = (buf_b + 1) % 2
qk_acc = acc_db[buf_c];     buf_c = (buf_c + 1) % 4

# PV 阶段
pv_left = left2[buf_a];     buf_a = (buf_a + 1) % 2
pv_right = right2[buf_b];   buf_b = (buf_b + 1) % 2
pv_acc = acc2[buf_c];       buf_c = (buf_c + 1) % 4
```

`buf_a` 与 `buf_b` 深度相同但相互独立——计数器按地址空间划分，与深度无关。

计数器在 task 循环外初始化、每次分配后配对递增；软件流水中 QK 与 PV 槽位错开靠"两次取用之间各递增一次"实现，改造时必须保持调用与递增的先后顺序。

## 性能与验证指标

- 静态检查：grep 确认共享地址组不再出现 `.next()`，例如
  `grep -nE "left_db\.next|left2\.next|right_db\.next|right2\.next|acc_db\.next|acc2\.next" <kernel.py>`（无输出即通过）。
- 可选 AST 检查：遍历函数体，确认同一变量名的所有 `.next()` 绑定形状一致；共享地址的组全部改为 `[buf_idx]` 显式下标。
- 正确性：在 flex_attention（QK/PV 软件流水 + drain 扩展 + block 位图跳块）上，matrix 15 case 全部 PASS，未再出现游标错相类偶发数据覆盖（当前算子已复现的验证）。
- 性能：机制上消除了空 `.next()` 同步调用的 MTE1 stall；独立性能数字待实测，不在此填写收益结论。

## 技术限制与风险

- 计数器递增必须与取用严格配对：漏递增或多递增都会重新错相，且不会报错。
- 取模的 depth 必须与该空间 `addrs` 槽位数一致。
- 计数器为 Python 标量，作用域须正确：task 循环外初始化、循环内跨 task 累加；跨 task 相位由累计调用次数自然同步，不得在 task 边界重置。
- 软件流水（PV 延迟一步）中，同一迭代内 QK 与 PV 取到不同槽位依赖两次取用之间的一次递增；调整流水结构时需同步审视递增点。

## 参考资料

- PyPTO 仓库：`docs/zh/api/pro_api/SIMD-API/resource_management/make_tile_group.md`（group[i] 运行时索引）
- 落地实现：xllm-ops 仓 `xllm_ops/flex_attention/op_kernel/flex_attention.py:1649-1654`（计数器初始化）、`:1739-1759`（QK/PV 调用点配对递增）、`:1095-1097`/`:1126-1128`（组下标取用）
