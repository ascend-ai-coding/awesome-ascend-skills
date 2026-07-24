# Blaze Tiling 选择指南

> 适用范围：Ascend 950 / DAV_3510 上使用 Blaze / tensor_api 路径开发普通 MatMul、MX MatMul、Grouped MatMul 以及 C+V 融合算子。
>
> 本 skill 只提供 SWAT 流式 tiling。Full-load、StreamK、4-buffer 和 grouped 专用 tiling 不属于默认路径。

---

## 1. 边界

Blaze 路径的 tiling 选择由本 skill 维护。Ascend 950 / DAV_3510 的 Blaze MatMul 场景不要引用 AscendC API MatmulImpl / MatmulApiTiling 文档。

本文不做：

- 不推导 `MatmulTilingSwat` / `QuantMatmulTilingSwat` 内部算法。
- 不为 C+V 的 Vector 部分新增独立 tiling engine。
- 不为 Grouped MatMul 新增 grouped 专用 tiling data。

---

## 2. 场景路由

| Blaze 场景 | Tiling engine | Tiling shape | 说明 |
|------------|---------------|--------------|------|
| 普通 MatMul | `MatmulTilingSwat` | `{M,N,K}` | A/B 两输入、可选 bias、输出 C |
| 普通 MatMul + C+V | `MatmulTilingSwat` | `{M,N,K}` | V 部分只消费剩余 UB |
| 普通 Grouped MatMul | `MatmulTilingSwat` | `{totalM,N,K}` | `groupNum/groupList` 独立传给 grouped kernel |
| 普通 Grouped MatMul + C+V | `MatmulTilingSwat` | `{totalM,N,K}` | grouped 与 epilogue 参数均不进入 tiling data |
| MXFP8/MXFP4 MatMul | `QuantMatmulTilingSwat<AType,BType>` | `{M,N,K}` | A/B/ScaleA/ScaleB，可选 bias，输出 C |
| MXFP8/MXFP4 MatMul + C+V | `QuantMatmulTilingSwat<AType,BType>` | `{M,N,K}` | V 部分只消费剩余 UB |
| MX Grouped MatMul | `QuantMatmulTilingSwat<AType,BType>` | `{totalM,N,K}` | group 参数独立传入 kernel |
| MX Grouped MatMul + C+V | `QuantMatmulTilingSwat<AType,BType>` | `{totalM,N,K}` | 不新增 grouped tiling |

---

## 3. 调用方式

普通 MatMul：

```cpp
MatmulTilingData tilingData;
MatmulTilingSwat tiling;
tiling.GetTilingData(
    m, n, k, inputElemBytes, tilingData,
    transA, transB, isANz, isBNz, hasBias);
```

MX MatMul：

```cpp
QuantMatmulTilingData tilingData;
QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN> tiling;
tiling.GetTilingData(
    m, n, k, tilingData,
    transA, transB, isANz, isBNz, hasBias);
```

Grouped 场景只把 `m` 替换为 `totalM = sum(groupList)`。`groupList` 内容不参与 tiling engine 计算。

---

## 4. 支持范围

| Tiling | dtype | format | transpose | bias |
|--------|-------|--------|-----------|------|
| `MatmulTilingSwat` | fp16/bf16/fp32/int8 等，按 `inputElemBytes` 区分 | A/B ND 或 NZ | A/B 4 种组合 | 支持 L1 预算预留 |
| `QuantMatmulTilingSwat` | A/B 为 MXFP8 或 packed MXFP4 | A/B ND 或 NZ | A/B 4 种组合 | 支持 L1 预算预留 |

平台信息（AIC/AIV 核数、UB/L1/L0A/L0B/L0C/L2/BT 容量）必须通过 `platform_ascendc::PlatformAscendCManager` 获取，禁止在 tiling 中写死容量。

---

## 5. 常见误用

| 误用 | 正确做法 |
|------|----------|
| 为 Grouped MatMul 新增 `groupNum/groupList` tiling data 字段 | group 参数独立传给 grouped kernel，tiling 使用 `{totalM,N,K}` |
| C+V 为 V 部分新增独立 tiling engine | V 部分只在 Cube tiling 后的剩余 UB 中规划 stage |
| 在本 skill 中切换 full-load、StreamK 或 4-buffer | 默认只走 SWAT；这些变体需要单独设计和验证 |
| MX format 参数传给旧的 5 参数接口 | 使用包含 `isANz/isBNz/hasBias` 的新接口 |
