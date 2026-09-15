# Tiling 选择与 Scheduler Params 合同

本文是 Step 3 设计、场景 PLAN 编译和 Step 4 执行时按需读取的 Tiling 方法。它不提供固定 SWAT、MX、Grouped 或融合 recipe；当前合法值和支持域必须来自本次 Investigation 报告与 DESIGN。

## 1. Tiling 的输入边界

每个 候选组装方案评估 必须先有：

```text
TilingData/Params fields and types
Scheduler/Policy/Block consumers
field units and legal predicates
shape/core/tail dependencies
cross-field constraints
Kernel entry/grid/workspace ABI
partition coverage
```

Step 2 闭合 device 侧“需要什么”；Step 3 决定 host 侧“怎样产生”。缺少 Blaze 源码 host tiling 实现本身不是阻塞，缺少 device 字段语义、合法域、单位或 ABI 才阻塞。

## 2. 逐字段兼容审查

把现有项目 Engine 或 [Blaze skill Tiling Asset](../../assets/op_tiling/) 作为可选结构起点，逐字段比较：

| 范围 | 必须一致 |
|---|---|
| Identity | 当前 candidate、partition、Kernel/Policy/Block/Scheduler chain |
| Shape | M/N/K、batch/group、tail、alignment、动态轴和物理 shape |
| Params | 字段名/类型、单位、consumer、TilingData mapping、默认/合法域 |
| Resources | tile、core、L1/L0/UB、workspace、归并和 final timing |
| ABI | Kernel entry、GM args、grid/usedCore、dispatch flags、输出路径 |
| Layout/numeric | dtype、transpose、ND/NZ/packing、scale、broadcast、转换顺序 |

文件名相似、字段名相同或 example 默认值不构成兼容。Asset 原文件始终只读，复制后必须重新适配和验证。

### 2.1 硬件 base shape 与逻辑 tail

`baseM/baseN/baseK` 等字段若用于 Block、Copy 或片上资源实例化，就是硬件基形，
必须满足所选当前源码 specialization 的粒度、对齐、容量和交叉约束。
`curM/curN/curK`、Scheduler block shape 或等价字段才表达当前逻辑 tail。

禁止用 `min(logical_shape, preferred_base)` 将硬件 base 缩成一个不合法的小 tail。
正确合同是：

```text
legal hardware base shape
  + scheduler/current logical extent
  + Block/Copy 对 tail 的已证明支持
```

具体对齐值不能从单个算子样例推广。必须从本次选定的 Block/Copy/Scheduler
源码或 capability witness 推导，并在 DESIGN 中记录 consumer 和合法谓词。
如果当前 specialization 不支持该 tail，应重划 partition 或阻塞，不能通过非法
base shape 强行表达。

当多个失败 case 分别被命名为 K-tail、N-tail，但共享同一个小 M 时，不能按 case
名称直接归因。应一次只改变一个轴，并比较所有失败 case 的共同变量。

## 3. 三种结果

### 3.1 复用兼容 Engine

当字段语义、单位、合法域、交叉约束、资源和 ABI 全部一致时，DESIGN 可选择复制/最小适配既有 Engine。PLAN 必须记录来源、目标文件、字段审计和边界 checkpoint。

### 3.2 固定合法值 Engine

Blaze 源码未提供 host Engine，但 device 合同完整时，项目 Tiling Engine 可对一个已冻结 demand partition 返回固定合法控制值。固定值元组必须绑定：

```text
partition_id
legal predicate
shape/tail/core assumptions
all consumer fields
cross-field constraints
validation checkpoint
```

一个元组不能覆盖未证明的 shape/dtype/layout/group/quant 分区。需要增加 specialization 或重划 partition 时返回 Step 2 记录 amendment，再重做 Step 3。

### 3.3 Blocking

以下情况不得猜默认：Scheduler 参数语义、单位、合法域、TilingData->Params 映射、workspace/final 关系、group metadata 或 output offset 不完整；候选保持 partial/unknown，PLAN 不生成可执行 Tiling action。

## 4. MatMul 与定制场景

纯 MatMul 只复用所选官方 MatMul Blaze 组装方案的 Tiling；Batch、Grouped、Quantized、MX 是需求分区，不是固定 Engine 名称。

Elementwise/Broadcast Epilogue 不建立独立 Vector tiling engine。其场景 DESIGN 必须消费 MatMul Tiling/Params 合同，并将额外输入、broadcast mapping、Epilogue staging、slot、同步和输出资源纳入同一预算。未来场景是否消费 MatMul 由其自身 design guide 决定。

只有 StreamK/SplitK/partial 等明确由 Tiling 与 Kernel 表达、并在 final/归并后启动 Epilogue 时，才能进入融合设计；不能把 partial tile 直接交给普通逐 tile Vector。

## 5. 需求分区与组合

对每个 demand partition 单独记录：

- Tiling 字段和合法谓词；
- candidate specialization、grid/core 和 workspace；
- runtime/compile-time dispatch；
- Group metadata、scale/packing 或 broadcast operand（需求激活时）；
- shape/alignment/tail 和拒绝范围。

不同 Scheduler、数值模式或 topology 的 Tiling 事实不能交叉借用。Grouped Plain/Quantized/MX 只有同一 exact Blaze 组装方案 witness 闭合后，才可各自进入 Step 3；不以 `totalM` 或任意旧字段替代实际 ABI。

## 6. PLAN 接线与验证

PLAN 依次登记：

```text
host input -> Tiling Engine -> TilingData
  -> Scheduler/Policy/Block/Epilogue Params
  -> Kernel/Wrapper -> grid/workspace/output
```

每项动作绑定 DESIGN refs、source refs、目标文件、checkpoint 和 rollback。验证至少覆盖 DESIGN 声明的 tail/alignment、multi-tile、dispatch、workspace/final、slot/Epilogue staging（适用时）和不支持范围。对每个有硬件粒度约束的轴，至少包含小于粒度、恰好等于粒度、`base+1` 和 multi-tile；Grouped 场景还覆盖空分组。未验证组合保持 `unverified`，不能通过默认值隐式扩大。
