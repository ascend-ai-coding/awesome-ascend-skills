# 跨代际迁移 Fixpipe 参数速查（FixpipeParamsArch3510）

> **适用**：DAV_2201（A2/A3）→ DAV_3510（950）迁移时 L0C 回写参数体系的切换。V220 与 Arch3510 两套参数**同名不同义、单位不同**，照搬旧参数必错。
> **参数真源**：官方文档 `$DEVKIT_PATH/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_store/`（Fixpipe / FixpipeParamsArch3510 全字段）。
> 迁移决策（何时弃高阶走低阶直跑、Fixpipe 与 Mmad 的 unitFlag 配合）见迁移技能 `ascendc-cross-gen-port-light` 的 cube-migration-guide 改动 3。

## FixpipeConfig（输出格式配置，950 增强）

```cpp
struct FixpipeConfig {
    CO2Layout format;   // NZ=0(保持NZ) / ROW_MAJOR(开启NZ2ND→ND) / COLUMN_MAJOR(仅950，NZ2DN→DN)
    bool isToUB;        // 目的地址是否在 UB（L0C→UB 通路）
};
```

## FixpipeParamsArch3510 关键字段（与 V220 的差异点是易错区）

| 字段 | 说明 |
|---|---|
| nSize | N 方向大小；不开启 channelSplit 时**必须为 16 的倍数**（channelSplit 时为 8 的倍数） |
| mSize | M 方向大小；mSize=0 表示 NOP |
| srcStride | 相邻 Z 排布起始地址偏移，单位 **C0_SIZE**（16×sizeof(T)），填 mSize 对 16 向上取整 |
| dstStride | 开启 NZ2ND/NZ2DN 时 = 目的**每一行元素个数（单位 element）**；不开转换时 = 相邻 Z 排布偏移。**注意：与 V220 的 dstStride datablock（32B）单位不同！** |
| params | 随路格式转换参数：NZ（普通搬运）→ `Nz2NdParams{ndNum, srcNdStride, dstNdStride}` / 950 新增 `Nz2DnParams{dnNum, srcNzMatrixStride, dstDnMatrixStride, srcNzC0Stride}` |
| dualDstCtrl / subBlockId | **仅 L0C→UB 通路有效**（双目标 / 子块模式） |
| unitFlag | Mmad 与 Fixpipe 细粒度并行（每算完一个分形立即搬出）；开启时 Mmad 和 Fixpipe 的 unitFlag 须同为 2 或 3 |

> **易错点速记**：`dstStride` 单位是 **element**（V220 是 datablock/32B）；`nSize` 不开启 channelSplit 时须为 **16 的倍数**；`dualDstCtrl` / `subBlockId` **仅 L0C→UB 通路**有效。
