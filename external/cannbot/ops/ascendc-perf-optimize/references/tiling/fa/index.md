# FA 族 Tiling 算法路由

> FA 族（FlashAttention、FusedInferAttentionScore、QSFA、SparseFlashAttention、SCFA）Tiling 算法入口。
>
> 先按目标代际路由；arch22 已有 FIA 源码级建模知识，arch35 fallback 当前仍是占位，不能跨代互相补位。

---

## 1. 已注册算法

| 优先级 | 算法/知识入口 | 选择条件 | 目录 |
|--------|------|---------|------|
| 100 | arch22 FIA 目标模型 | DAV_2201 / 220x；FIA nonquant/MLA，或已证明共享同一 arch22 数据流的变体 | [A2/A3 FA 指南](../../../../ascendc-performance-best-practices/references/flash_attention/arch22/guide.md) |
| 0（兜底） | arch35 官方参考 | DAV_3510；当前为占位，返回参考缺口 | [fallback/](fallback/) |

arch22 FIA 路线联合求解 `mBaseSize`、`sInnerSize`、核数、L1/L0/UB buffer、事件和 normal/FlashDecode workspace，详见 [Tiling、buffer 与同步](../../../../ascendc-performance-best-practices/references/flash_attention/arch22/tiling_pipeline_design.md)。QSFA、SparseFlashAttention、SCFA 等变体未证明共享 arch22 ABI/模板时返回变体级参考缺口。A3 没有目标实测时保留 `A3_not_run`，不得迁移 A5 的 SCALAR/SCALARLDST 阈值、Reg/VF、SIMT 或 `__ssbuf__` 路线。

---

## 2. 贡献新算法

在 `fa/` 下创建新目录，在此文件的路由表中注册；选择条件必须包含目标架构和具体 FA 变体。
