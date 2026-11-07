# 计算层：AscendC::Matmul 与 HCCL 流水耦合（910B / A2）

本文只补 hccl-matmul 路线特有的融合用法。`AscendC::Matmul` 的 API 签名、dtype、Tiling 类、A2 限制一律引用 `ascendc-api-best-practices` 的 [`api-matmul.md`](../../../../ascendc-api-best-practices/references/api-matmul.md)，**本文件不复述**。

> 蓝本：[`all_gather_matmul/`](all_gather_matmul)。本期不支持 quant。Blaze 为 950 专属，本路线禁用。

## 1. 本路线怎么用 Matmul

- 头文件：`lib/matmul/matmul_intf.h`。官方类名 `AscendC::Matmul`，现网别名 `MatmulImpl`。
- 公式：`C = A * B + Bias`。A2 下 MMAD 在 AIC/Cube，经 Fixpipe 从 L0C 直出 GM（不经 UB；A2 无 L0C→UB 直连）。
- 蓝本不裸调 `AscendC::Matmul`，而是经 `mc2_common/op_kernel/mc2_matmul_compute.h` 的 `MatmulCompute`（封装 `MatmulImpl<…,CFG_MDL>` + `Iterate`/`GetTensorC`）。两种写法都合法：直调更贴近官方文档，wrapper 多了 DEFAULT/L2CACHE 分块。框架解剖见 [`op_architecture.md`](op_architecture.md) §4.2。

## 2. 与 HCCL 的耦合点

官方直调写法（Wait 之后再算）：

```cpp
AscendC::Matmul<aType,bType,cType,biasType> mm;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm);
mm.Init(&cubeTiling); mm.SetOrgShape(...); mm.SetTensorA(gmA); mm.SetTensorB(gmB);
for (i in tileCnt) {
    hccl.Wait(handleId);
    while (mm.Iterate()) { mm.GetTensorC(gmC); }
}
mm.End();
```

蓝本（[`x_all_gather_matmul_full_mesh.h`](all_gather_matmul/x_all_gather_matmul/op_kernel/x_all_gather_matmul_full_mesh.h)）：`hccl_.Wait(handleId)`（`:143`）→ 跳过本 rank（`:149-150`）→ `mm.Compute(index)`（`:161`）。wrapper 内部才是 `Init→SetSingleShape/SetTensorA/B→Iterate→GetTensorC`。

高阶 API 不暴露 `SetFlag<HardEvent::FIX_M>`；蓝本 wrapper 里的 HardEvent 是其内部实现，不是官方用法。

## 3. 本路线额外约束（A2）

| 约束 | 理由 |
|------|------|
| 禁止 Blaze | 950 专属，910B 官方文档无此路径 |
| 不调 `SetLocalWorkspace` | A2 不支持 |
| 手动 `CrossCoreSetFlag` 勿占用 `[0, 2N-1]` | 与 Matmul 内部 flagId 冲突 |
| 无 FP8/MXFP8 | 950-only |
| 不引入 quant 路径 | 本期范围选择（`SetQuantScalar/Vector` 在 A2 硬件上可用，但本路线不覆盖） |

## 4. 排错

| 现象 | 排查 |
|------|------|
| 编译报 Blaze 未定义 | 删 Blaze，改 `AscendC::Matmul` |
| 结果全零 | 检查 `Init→SetTensor→Iterate→GetTensorC` 顺序 |
| `SetLocalWorkspace` 报错 | 删除该调用 |
| flagId 冲突 | 让出 `[0, 2N-1]`，Finalize 同步用 EVENT_ID_6（蓝本） |
| API 签名/dtype/Tiling 细节 | 回 `api-matmul.md` |
