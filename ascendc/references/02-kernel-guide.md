# AscendC Kernel 开发与 Matmul/Cube 指引

包含：Kernel 通用模式（GlobalTensor/TQue、CopyIn/Compute/CopyOut）、Matmul/Cube 调用模板、GMM 权重转置说明。**参考以 CANN 官方 API 与编程指南为准，本页仅做要点归纳。**

## 官方文档参考（示例与规范来源）

- **基础数据结构**：[Ascend C API 列表](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html) — [LocalTensor](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0006.html)、[GlobalTensor](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0007.html)、TQue 等
- **TQue 与流水**：CANN 文档「内存管理与同步控制」中 TQue 简介（CopyIn/Compute/CopyOut 流水、AllocTensor/EnQue/DeQue/FreeTensor）
- **Matmul 高阶 API**：Ascend C API 列表中数学库/Matmul 相关接口（SetOrgShape、SetSingleShape、SetTensorA/B、Iterate、GetTensorC 等）
- **多核 / 工程样例**：[Gitee Ascend samples - operator/ascendc](https://gitee.com/ascend/samples/tree/master/operator/ascendc) 中矩阵类、多核 Tiling 等示例

---

## 一、GlobalTensor 与 TQue 的职责

| 类型 | 职责 | 常用接口 |
|------|------|----------|
| **GlobalTensor\<T\>** | 表示 GM 上的张量，用于与 GM 读写 | `SetGlobalBuffer(ptr, length)`、`tensor[offset]`（与 DataCopy 配合） |
| **TQue\<QuePosition, BUFFER_NUM\>** | 双缓冲流水队列，管理 LocalTensor | `AllocTensor<T>()`、`EnQue(local)`、`DeQue<T>()`、`FreeTensor(local)` |

**GlobalTensor 没有** AllocTensor/EnQue/DeQue/FreeTensor；**TQue 不直接表示 GM**，需与 GlobalTensor 配合完成 GM↔片上数据交换。

### Init 阶段

- **GlobalTensor**：仅设置 GM 视图，如 `inputGM.SetGlobalBuffer((__gm__ T*)ptr + blockOffset, blockLength);`
- **TQue**：用 pipe 初始化，如 `pipe.InitBuffer(inputQueueX, BUFFER_NUM, tileLength * sizeof(T));`，不要对 GlobalTensor 调用 InitBuffer。

### CopyIn / Compute / CopyOut

1. **CopyIn**：TQue 上 `AllocTensor<T>()` → `DataCopy(local, inputGM[progress * tileLength], tileLength)` → 对输入 TQue `EnQue(local)`。
2. **Compute**：输入 TQue `DeQue<T>()` → 输出 TQue `AllocTensor<T>()` → 计算 → 输出 TQue `EnQue(local)` → 输入 TQue `FreeTensor(local)`。
3. **CopyOut**：输出 TQue `DeQue<T>()` → `DataCopy(outputGM[progress * tileLength], local, tileLength)` → 输出 TQue `FreeTensor(local)`。

**参考示例**：官方文档中 GlobalTensor、LocalTensor、TQue 的接口说明及「内存管理与同步控制」中的 CopyIn/Compute/CopyOut 流水示例；工程内同类型 SIMD/向量算子可作实现参考。

---

## 二、Matmul/Cube 调用模板

适用于在 AIC 上用 Cube 做矩阵乘（如 GMM、MoE finalize routing）。

### 1. 类型与 MatmulImpl

约定：A（输入 x）GM+ND；B（权重 w）GM+NZ；C（中间/输出）GM+ND；Bias 可选 GM+ND。

```cpp
using aT    = MatmulType<TPosition::GM, CubeFormat::ND, int8_t>;
using bT    = MatmulType<TPosition::GM, CubeFormat::NZ, int8_t>;
using BiasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using cT    = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using MT = matmul::MatmulImpl<aT, bT, cT, BiasT, CFG_MDL>;
```

### 2. 单块调用流程

```cpp
// 1. 绑定 GM（Init 阶段）
xGm.SetGlobalBuffer(...); weightGm.SetGlobalBuffer(...); cGm.SetGlobalBuffer(...);

// 2. 配置整体与当前 block 形状
mm.SetOrgShape(M, N, K);           // 整体 (M,N,K)
mm.SetSingleShape(curM, curN, K);   // 当前 block

// 3. 绑定 A/B，K 轴迭代写 C
mm.SetTensorA(xGm[xOffset]);
mm.SetTensorB(weightGm[weightOffset]);
uint64_t cOffset = workspaceOffset;
while (mm.Iterate()) {
    mm.GetTensorC(cGm[cOffset], 0, true);
    cOffset += (baseM * baseN);
}
```

- `SetOrgShape`：整体矩阵，用于 tail 与性能模型。
- `SetSingleShape`：当前一次计算的子块。
- `Iterate()`：按 tiling 的 stepKa/stepKb 沿 K 分段；每次后 `GetTensorC(..., true)` 累加写 C。

### 3. 分块与多组（Grouped Matmul）

Host tiling 提供 baseM/baseN/baseK、mList/kList/nList、groupNum 等；Device 侧根据 blockIdx 算 (mIdx, nIdx)、当前 group 的 (m,k,n)、A/B/C 在 GM/workspace 的偏移，再按上节调用 MatmulImpl。**参考示例**：官方 Matmul 高阶 API 与 [Gitee ascend/samples - operator/ascendc](https://gitee.com/ascend/samples/tree/master/operator/ascendc) 中多核/矩阵类样例。

### 4. grouped_matmul_finalize_routing 特例

- **workspace 分片**：Cube 只写 int32 到 workspace，不同 core/并行度用 disjoint offset。
- **AIC/AIV 协作**：AIC 做 MMCompute + CrossCoreSetFlag；AIV 做 CrossCoreWaitFlag → DataCopyMMOut → AscendDequant → VectorAtomicProcess 写回。
- **确定性**：tiling->deterministicFlag 时用 mmQuantOutGm + FRDeterministic 按行分配与原子加汇总。新算子可直接复用该整体设计。

---

## 三、GMM 权重 W 的转置（逻辑转置）

- 算子内**不做**显式转置（不搬运整张 W 做转置写回）；转置通过 **Matmul 逻辑转置标记 + offset 计算** 完成。
- Host：Tiling 中 `transpose_weight` 决定权重逻辑是否转置，并调整 K/N 解析与 tiling key，选择 `MatmulType<..., isTrans>`。
- Kernel：`transposeW = mmType::BT::isTrans`；取权重子块时只改**读偏移**（如非转置 ND：`wOffset = tailN`；转置 ND：`wOffset = tailN * k`），再 `mm.SetTensorB(weightGmLocal, transposeW)`。Matmul 内部按标记访问，无新转置矩阵。
- ND2NZ 是数据布局转换（适配 Cube），不等于用户语义的转置，也无显式转置写回。
