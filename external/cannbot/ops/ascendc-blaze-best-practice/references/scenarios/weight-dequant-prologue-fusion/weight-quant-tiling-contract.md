# Weight-Quant Tiling 参数合同专题

本文是本场景的 Tiling 方法专题。Step 3 用它设计 Tiling/资源合同并编译 PLAN；Step 4 以 PLAN 初始 Tiling 基线为起点，按实际失败证据读取本文并执行诊断。本文不提供固定 Tiling recipe，当前合法值和支持域必须来自本次 Investigation 报告与 DESIGN。

## 1. 输入与输出合同

输入：`matmul_base_analysis`、Investigation 中已闭合的 Scheduler/Block/Prologue Params 语义事实、当前 Tiling Engine witness。输出：

```text
tilingdata_field_contract
scheduler_params_contract
prologue_params_contract
alignment_and_space_constraint
tiling_validation_additions
```

每个结论必须引用当前候选组装方案评估、witness 和 evidence IDs。

## 2. TilingData 字段语义合同

DESIGN 必须逐字段冻结 TilingData 的语义、单位、合法域和 consumer。当前参考字段（以 Asset Engine 为结构起点，实际字段由 Investigation 确认）：

| 字段 | 语义 | 单位 | consumer | 合法域/约束 |
|---|---|---|---|---|
| `usedCoreNum` | 启动核数 | 核数 | kernel launch `<<<usedCoreNum>>>` | 正整数，受平台核数限制 |
| `baseM` / `baseN` / `baseK` | L0 tile 颗粒 | 元素数 | BlockMmad::Init | baseM×baseN×4×DB ≤ l0cSize |
| `kL1` | K 方向 L1 chunk | 元素数 | BlockMmad K 循环 + Prologue K 循环 | 分形轴 16 对齐 |
| `nUbSize` | 单 AIV 单 buffer N 方向元素数 | 元素数 | Prologue UB 规划 | UB 空间约束 |
| `kUbSize` | 单 AIV 单 buffer K 方向元素数 | 元素数 | Prologue UB 规划 | UB 空间约束 |
| `mTailCnt` / `nTailCnt` | 尾块切分份数 | 份数 | Scheduler 尾轮 | transB=true 时 nTailCnt 不切分 |
| `mBaseTailSplitCnt` / `nBaseTailSplitCnt` | 边块合并组数 | 组数 | Scheduler 边块 | 非负整数 |
| `mTailMain` / `nTailMain` | 合并组主块尺寸 | 元素数 | Scheduler 边块 | 与 baseM/baseN 关系闭合 |
| `l0cDB` | L0C 双缓冲级数 | 级数 | BlockMmad FIX_M flag | 1 或 2 |
| `transB` | B 转置标志 | bool | Kernel 布局选择 + Prologue 切分轴 | true/false |
| `hasOffset` | 有 offset | bool | Prologue VF 分支 + UB 布局 | true/false |
| `hasBias` | 有 bias | bool | BlockMmad BT 路径 + L1 空间 | true/false |

PLAN 必须逐字段验证复制后的 Asset Tiling Engine 与当前 Scheduler/Block/Prologue Params 合同兼容。只有全部语义/单位/合法域一致才复用或最小适配。

## 3. Tiling Engine 合同

当前 Asset 提供的 `WeightQuantTilingSwat` 是 SWAT 流式路径的 Tiling Engine。DESIGN 必须确认：

1. Engine 入口签名和参数语义与 Investigation 确认的源码一致；
2. Engine 内部分形轴对齐校验逻辑闭合（transB=true: `n%16==0`; transB=false: `k%16==0`）；
3. Engine 返回的 TilingData 字段覆盖全部 Scheduler/Block/Prologue consumer；
4. Engine 不支持的 partition 需要项目 Tiling 返回固定合法控制值时，固定值元组必须绑定 partition_id、legal predicate 和 consumer fields。

不能复用但 device 合同已闭合时，PLAN 可令项目 Tiling Engine 返回经过证明的固定合法控制值。需要新的 Params、specialization 或合法域时返回 Step 2/3。

## 4. 对齐校验合同

Tiling 入口必须自动校验分形轴对齐：

- transB=true：`n % 16 == 0`
- transB=false：`k % 16 == 0`
- 违反时抛出 `std::runtime_error`

PLAN 的 Tiling action 必须验证此校验逻辑在复制后的 Engine 中保留。若项目修改了 Engine，必须重新验证校验闭合。

## 5. Tiling 与 BlockMmad 空间一致性合同

DESIGN 必须冻结 Tiling 计算与 BlockMmad/Prologue 使用的空间一致性。UB 空间约束必须按当前 B dtype 参数化，不得硬编码 int8 系数：

| 空间 | Tiling 计算 | BlockMmad/Prologue 使用 | 一致性要求 |
|---|---|---|---|
| L1 bias 尾部 | `Align(baseN * biasElemBytes, 64)` | `TOTAL_L1_SIZE - CeilAlign(baseN * sizeof(BiasType), 64)` | biasElemBytes == sizeof(BiasType) |
| L0C 双缓冲 | `baseM * baseN * 4 * DB_SIZE <= l0cSize` | `HALF_L0C_SIZE = TOTAL_L0C_SIZE / 2` | DB_SIZE 与 l0cDB 一致 |
| UB | `2*(weightElemBytes+dequantBElemBytes)*kUbSize*nUbSize + 2*dequantBElemBytes*nUbSize*(1+hasOffset) <= UB_SIZE` | Prologue Init UB 布局 | weightElemBytes 与 BType 一致 |
| L1 总空间 | B_dequant + A + bias | L1 布局合同 | 与 kL1/baseM/baseN 关系闭合 |

参数化 UB 约束公式：设 `weightElemBytes = sizeof(BType)`，`dequantBElemBytes = sizeof(DequantBType)`（通常为 2），UB 系数 = `2 × (weightElemBytes + dequantBElemBytes)`。int8/fp8=1B → 系数 6。当前 Asset Tiling Engine 中 `elemBytesB()` 硬编码为 `DATA_SIZE_INT8`，切换 dtype 时必须参数化。

## 6. 验证门禁

- TilingData 全部字段有闭合的语义/单位/合法域/consumer 映射；
- 分形轴对齐校验在 Tiling 入口保留；
- Tiling 与 BlockMmad/Prologue 的空间一致性公式满足；
- Engine 不支持的 partition 有固定合法值或 blocking 处置；
- transB=true 时尾轮 N 方向不切分；
- 官方源码区与 Blaze Asset 原文件保持零改动。

具体 baseM/baseN/baseK/kL1/kUbSize/nUbSize 值来自 DESIGN/PLAN 和当前 shape/资源，不由本文固定。
