# Ascend NPU 内存体系参考（调优用）

> 核心真源：《昇腾950 NPU架构白皮书》（表3-1 SKU 规格、表4-2 Buffer 物理容量、§4.3 内存层次）
> 用途：TileShape 设置、UB/L1 容量规划、泳道图搬运分析、Stitch/合图调优时的硬件约束查询。表内为**族系典型值**，具体 SKU 可能裁剪，运行时必须以接口返回值为准。

## 一、AI Core 片上 Buffer 层级（内存金字塔）

| Buffer | 910B2 (DAV_2201) | 950PR (DAV_3510) | 用途 |
|---|---|---|---|
| **L1** | 512 KB | 512 KB | Cube 输入缓存（矩阵左右操作数驻留） |
| **L0A / L0B** | 64 KB / 64 KB | 64 KB / 64 KB | Cube 左/右矩阵操作数 |
| **L0C** | 128 KB | **256 KB（翻倍）** | Cube 输出，决定基本块 Tiling 上限 |
| **UB** | 192 KB | **248 KB（每 AIV 可用）** | Vector 工作区，每 AIV 独立一份，SIMT/SIMD 共享 |

### UB 容量口径说明

- 950 白皮书表 4-2 标注 "UB 512 KB per AI Core"，实际按 **AI Core 组（1 AIC + 2 AIV）** 计物理容量：
  - 单 AIV 物理 256 KB × 2 = 组级 512 KB
  - 单 AIV 用户可用 = 256 KB − 8 KB 预留 = **248 KB**（INI `ub_size = 253952`，即 `GetCoreMemSize(UB)` 返回值）
- SIMT 场景还需为 DCache 让位 ≥ 32 KB
- 同一代架构内各 SKU 的 L1/L0/UB/BT 通常一致；L2 与 Memory 可能因子型号/形态而异

## 二、L2 与全局 Memory（因子型号/形态而异）

| 参数 | 910B2 | 950PR PCIE | 950PR Server |
|---|---|---|---|
| Cube 核数 | 24 | 28 | 32 |
| 频率 | 1.8 GHz | 1.65 GHz | 1.65 GHz |
| **L2** | 192 MB | 112 MB | 128 MB |
| **Memory 容量** | 64 GB | 112 GB | 128 GB |
| **Memory 带宽** | 1.6 TB/s（经验值） | 1.4 TB/s（白皮书证实） | 1.6 TB/s（白皮书证实） |

白皮书 §4.3 补充：

- 950DT：Memory 144 GB / 4 TB/s
- L2 为 UMA 架构，512B CacheLine、4×128B Sector，支持 L2 Hint 与 CMO

**调优含义**：带宽是搬运算子（纯 Vector、GM↔UB 密集）的屋顶线；L2 容量决定大权重/激活驻留策略（如 weight-none-l2-cacheable 场景）。

## 参考信息

- [昇腾950 NPU架构白皮书](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf)（DAV_3510 官方规格与微架构真源）
- [npu-arch SKILL.md](https://gitcode.com/cann/cannbot-skills/blob/master/ops/npu-arch/SKILL.md)
- [npu-arch-guide.md](https://gitcode.com/cann/cannbot-skills/blob/master/ops/npu-arch/references/npu-arch-guide.md)
- CANN 安装包：`${ASCEND_HOME_PATH}/<arch>/asc/include/utils/tiling/platform/platform_ascendc.h`
