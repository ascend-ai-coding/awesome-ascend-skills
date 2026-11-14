# hccl-matmul 路线代码审查验收条件

> Reviewer 在设计评审与阶段三验收逐项检查。违反任意红线项 = FAIL。本路线仅覆盖 Ascend 910B（A2 / dav-2201）。

## 红线项

| # | 检查项 | 验收条件 |
|---|--------|---------|
| R1 | 架构=910B(A2) | `grep -i "ascend910b\|dav-2201"` 命中；无 A3/910_93/950 |
| R2 | 通信走 HCCL 高阶 V2 | 含 `Hccl<HCCL_SERVER_TYPE_AICPU>`、`InitV2`、`SetCcTilingV2`、`AllGather`/`AllReduce`/`Commit`/`Wait`/`Finalize`；V1 `Init`/`SetCcTiling` 应为空 |
| R3 | Matmul 走 `AscendC::Matmul` | 含 `lib/matmul/matmul_intf.h` 或 `MatmulImpl`/`MatmulCompute`；无 Blaze |
| R4 | 无 SHMEM/UDMA | `grep -rn "aclshmem"` 应为空 |
| R5 | 无 Blaze | `grep -rni "blaze"` 应为空 |
| R6 | AIC-only + Finalize 前跨核同步 | `ASCEND_IS_AIC` 命中；Finalize 前有 `CrossCoreSetFlag`/`CrossCoreWaitFlag`（非 `SyncAll`） |
| R7 | 无 quant | `grep -rni "quant\|qbmm_mx\|copy_scale"` 应为空 |
| R8 | 仅注册 / aclnn | 无 `<<<>>>` 直调入口；host 有 `MC2().HcclGroup("group")` |

## 精度门禁

| 融合方向 | 判据 |
|----------|------|
| 先通后算（AllGather+MM） | 误差元素比例 `<1e-2`，逐元素 `np.isclose`（fp16 `5e-3`，bf16 `1e-2`） |
| 先算后通（MM+ReduceScatter/AllReduce） | 整体相对误差 `‖y_npu−y_fp32‖₂/‖y_fp32‖₂ ≤ 4ε`；不可照抄逐元素 `atol` |
| scatter 类（各卡输出不同） | **逐 rank** 比对，不可只查 rank-0 |

详见 [`workflow_integration.md`](workflow_integration.md) §1.4.1。

## 常见 FAIL 原因

| 现象 | 根因 | 修复方向 |
|------|------|----------|
| R2 命中 `hcomm_`/`aclshmem` | 误用 SHMEM/HCOMM | 改回 `Hccl<HCCL_SERVER_TYPE_AICPU>` V2 |
| R5 命中 Blaze | 误抄 950 模板 | 换 `AscendC::Matmul` |
| `Wait` 卡死 | Finalize 前缺跨核同步 | 加 `CrossCoreSetFlag/CrossCoreWaitFlag`（参考 `full_mesh.h:224-225`） |
| 误用 `AlltoAllV` | A2 不支持 | 改 `AlltoAll`（等长） |
| R7 命中 quant | 引入量化路径 | 删除 quant 文件 |
| bf16 稳定超差 2%~3%（先算后通） | 判据不适用，非算子问题 | 换整体相对误差 `≤4ε` |
| scatter 类精度"全过"但上层结果错 | 只校验了 rank-0 | 逐 rank 比对 |
| 改了 kernel 仍跑旧结果 | `.run` 装成 `vendors/custom_opp/vendors/custom_opp` 嵌套，或 op_type 与内置算子撞名 | 改 `VENDOR_NAME` / `--install-path=`；op_type 加自定义前缀（蓝本用 `X`） |
