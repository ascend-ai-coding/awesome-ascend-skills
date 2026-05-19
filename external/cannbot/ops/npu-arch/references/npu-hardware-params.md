# NPU 硬件参数真值参考

> **数据源**：`${ASCEND_HOME_PATH}/<arch>/data/platform_config/*.ini`（arch 如 `aarch64-linux`、`x86_64-linux`、`arm64-linux`），参数值以运行时 `PlatformAscendC` 接口返回为准。
>
> **数据可信度约定**：本文档中「INI 字段」/「实测」均指从 ini 读取到的静态配置值。**ini 文件是发现硬件参数的必要不充分条件** — 并非每个 ini 都对应真实量产芯片，部分配置可能来自工程样片、仿真平台或规划中的 SKU。文档聚焦架构级常量（§1、§2）和关键差异速查（§3），变化参数仅以典型量产 SKU 举例。

---

## 0. 产品映射表

按 NpuArch 列出 Ascend NPU 全代际的产品系列、SocVersion 与具体芯片型号。**本表为 SKILL.md 与 `npu-arch-guide.md` 共用的真源，需要修改芯片清单时仅在此处更新。**

| 产品系列 | SocVersion | NpuArch | __NPU_ARCH__ | 芯片型号 |
|---------|-----------|---------|:---:|---------|
| Atlas 训练系列 | ASCEND910 | DAV_1001 | 1001 | Ascend910 |
| Atlas 推理系列 | ASCEND310P | DAV_2002 | 2002 | Ascend310P1, Ascend310P3, Ascend310P5, Ascend310P7 |
| Atlas A2 训练/推理 | ASCEND910B | DAV_2201 | 2201 | Ascend910B1~B4, Ascend910B2C |
| Atlas A3 训练/推理 | ASCEND910B | DAV_2201 | 2201 | Ascend910_93 |
| Atlas 200I/500 A2 推理 | ASCEND310B | DAV_3002 | 3002 | Ascend310B1~B4 |
| Atlas A5 训练 | ASCEND950 | DAV_3510 | 3510 | Ascend950DT (Decode) |
| Atlas A5 推理 | ASCEND950 | DAV_3510 | 3510 | Ascend950PR (Prefill) |

> **一对多关系**：一个 NpuArch 可对应多个 SocVersion / 芯片型号。例如 `DAV_2201` 对应 Ascend910B1~B4、Ascend910B2C、Ascend910_93。
>
> **运行时映射**：`Ascend910_93` 的 SocVersion 字符串在运行时映射到 `SocVersion::ASCEND910B`（非独立枚举值）。源码中虽存在 `SocVersion::ASCEND910_93` 枚举值（platform_ascendc.h），但 convertMap 中 `"Ascend910_93"` 映射到的是 `ASCEND910B`，该枚举仅在少数内部模块使用。NpuArch 同为 `DAV_2201`。

---

## 1. 跨架构一致参数

以下参数在所有已验证的 Ascend NPU 架构上保持一致：

| 参数 | INI 字段 | 值 | 说明 |
|------|---------|:---:|------|
| L0A | `[AICoreSpec] l0_a_size` | 64 KB (65536) | Cube 左矩阵操作数 |
| L0B | `[AICoreSpec] l0_b_size` | 64 KB (65536) | Cube 右矩阵操作数 |
| Cube MAC 阵列 | `cube_m_size / cube_k_size / cube_n_size` | 16×16×16 | 一个周期完成 4096 次 MAC |

> **注意**：即便以上参数通常一致，代码中仍应通过 `GetCoreMemSize` 获取，避免硬编码。

---

## 2. 各架构参数（子型号间通常一致）

通常同 NpuArch 的子型号中，以下参数值一致：

### 2.1 DAV_1001 — Ascend910 系列

| 参数 | INI 字段 | 值 |
|------|---------|:---:|
| NpuArch | `NpuArch` | 1001 |
| L1 | `l1_size` | 1 MB (1048576) |
| L0C | `l0_c_size` | 256 KB (262144) |
| UB | `ub_size` | 256 KB (262144) |
| L2 | `l2_size` | 32 MB (33554432) |
| BT | `bt_size` | — (不存在) |
| 稀疏 | `sparsity` | — (不存在) |
| 核心类型 | `core_type_list` | `AICore`（无独立 VectorCore） |
| 核间关系 | — | VectorCore 不存在，ai_core 内聚 Cube+Vector 功能 |

### 2.2 DAV_2002 — Ascend310P 系列

| 参数 | INI 字段 | 值 |
|------|---------|:---:|
| NpuArch | `NpuArch` | 2002 |
| L1 | `l1_size` | 1 MB (1048576) |
| L0C | `l0_c_size` | 256 KB (262144) |
| UB | `ub_size` | 256 KB (262144) |
| L2 | `l2_size` | 16 MB (16777216) |
| Memory | `memory_size` | 24 GB (24000000000) |
| BT | `bt_size` | — (不存在) |
| 稀疏 | `sparsity` | — (不存在) |
| 核心类型 | `core_type_list` | `AICore,VectorCore`（无独立 CubeCore） |
| 核间关系 | — | Cube 功能集成在 AICore 内，AICore 与 VectorCore 非 1:2 关系 |

### 2.3 DAV_2201 — Ascend910B / Ascend910_93 系列

| 参数 | INI 字段 | 值 |
|------|---------|:---:|
| NpuArch | `NpuArch` | 2201 |
| L1 | `l1_size` | 512 KB (524288) |
| L0C | `l0_c_size` | 128 KB (131072) |
| UB | `ub_size` | 192 KB (196608) |
| BT | `bt_size` | 1 KB (1024) |
| 稀疏 | `sparsity` | 1（支持 4:2） |
| 核心类型 | `core_type_list` | `CubeCore,VectorCore` |
| 核间关系 | — | CubeCore : VectorCore = 1 : 2 |

### 2.4 DAV_3002 — Ascend310B 系列

| 参数 | INI 字段 | 值 |
|------|---------|:---:|
| NpuArch | `NpuArch` | 3002 |
| L1 | `l1_size` | 1 MB (1048576) |
| L0C | `l0_c_size` | 128 KB (131072) |
| UB | `ub_size` | 248 KB (253952) |
| BT | `bt_size` | 1 KB (1024) |
| L2 | `l2_size` | 4 MB (4194304) |
| 稀疏 | `sparsity` | 1（支持 4:2） |
| 核心类型 | `core_type_list` | `AICore,VectorCore,CubeCore`（三合一） |
| 核心数 | `ai_core_cnt / cube_core_cnt / vector_core_cnt` | 均为 1（单核集成） |

### 2.5 DAV_3510 — Ascend950DT / Ascend950PR 系列

| 参数 | INI 字段 | 值 |
|------|---------|:---:|
| NpuArch | `NpuArch` | 3510 |
| L1 | `l1_size` | 512 KB (524288) |
| L0C | `l0_c_size` | 256 KB (262144) |
| UB | `ub_size` | 248 KB (253952) |
| BT | `bt_size` | 4 KB (4096) |
| 稀疏 | `sparsity` | 0（不再支持 4:2） |
| 核心类型 | `core_type_list` | `CubeCore,VectorCore` |
| 核间关系 | — | CubeCore : VectorCore = 1 : 2 |

> **关于 UB 容量**：表内值为 INI `ub_size` 字段，即 `GetCoreMemSize(CoreMemType::UB, ...)` 返回的用户可用容量。**运行时始终以该接口返回值分块**，禁止硬编码。
>
> **FB（Fix Buffer）**：FixPipe 量化 scale 存储区。INI 字段 `fb0_size` / `fb1_size` / `fb2_size` / `fb3_size`；`GetCoreMemSize(FB)` 返回 `fb0_size`。

---

### 子型号变化参数

> **同一 NpuArch 内，核数、频率、L2、Memory 等参数可能随子型号不同而变化。**
>
> `platform_config/*.ini` 涵盖多种配置（含工程样片、仿真配置），ini 文件是发现硬件参数的必要不充分条件，因此**不在此处枚举"参数范围"**。算子代码**必须通过 `PlatformAscendC` 接口运行时获取**。
>
> 作为参考，以下为有 archXX 特化开发路径的架构典型量产 SKU 示例（数据来源：对应 ini 文件）：
>
> | 架构 | 示例型号 | CubeCore | VectorCore | 频率 | L2 | Memory |
> |------|---------|:---:|:---:|:---:|:---:|:---:|
> | DAV_2201 | Ascend910B2 | 24 | 48 | 1.8 GHz | 192 MB | 64 GB |
> | DAV_3510 (PCIE) | Ascend950PR_957b | 28 | 56 | 1.65 GHz | 112 MB | 112 GB |
> | DAV_3510 (Server) | Ascend950PR_9589 | 32 | 64 | 1.65 GHz | 128 MB | 128 GB |
>
> > 注：DAV_2201 / DAV_3510 中 CubeCore : VectorCore = 1 : 2。
> >


---

## 3. 架构关键差异速查

| 特征 | DAV_1001 | DAV_2002 | DAV_2201 | DAV_3002 | DAV_3510 |
|------|:--:|:--:|:--:|:--:|:--:|
| 核心类型 | AICore | AICore+VectorCore | CubeCore+VectorCore | AICore+VectorCore+CubeCore | CubeCore+VectorCore |
| Cube:Vec 比例 | N/A (无Vec) | 非 1:2 | 1:2 | N/A (单核) | 1:2 |
| L1 | 1 MB | 1 MB | 512 KB | 1 MB | 512 KB |
| L0C | 256 KB | 256 KB | 128 KB | 128 KB | 256 KB |
| UB | 256 KB | 256 KB | 192 KB | 248 KB | 248 KB |
| BT | — | — | 1 KB | 1 KB | 4 KB |
| 稀疏 4:2 | — | — | 支持 | 支持 | 不支持 |
| Cube 16³ | ✓ | ✓ | ✓ | ✓ | ✓ |
| L0A/B 64KB | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## 4. 基于公开资料与经验值的规格

以下信息无法从当前安装的 INI 直接推导，数据来源于公开资料或工程经验积累：

> **位置标注约定**：本表"文档位置"列使用章节锚点（如 `Guide §5`），不使用行号，以避免文档行号漂移导致引用失效。

| # | 内容 | 文档位置 | 信息来源 |
|---|------|---------|---------|
| 1 | SIMT Register File 128KB | Guide §5 SIMT vs SIMD | 公开资料 / 经验值 |
| 2 | SIMT DCache 最大 128KB | Guide §5 SIMT vs SIMD | 同上 |
| 3 | SSBuffer 256KB | Guide §3 Buffer 容量 / SKILL.md §DAV_3510 关键变化 | 同上 |
| 4 | CV 直通通路：L0C→UB、UB→L1、SSBuffer 消息 | Guide §4 关键数据通路改动 / SKILL.md §DAV_3510 关键变化 | 同上 |
| 5 | L1→GM 和 GM→L0A/L0B 通路已删除 | Guide §4 关键数据通路改动 | 同上 |
| 6 | BufferID 取代 set/wait 同步 | Guide §4 指令序列与 BufferID 同步 / SKILL.md §DAV_3510 关键变化 | 同上 |
| 7 | 多核同时访问 GM 同地址性能优化 | Guide §4 MTE 数据搬运引擎 | 同上 |
| 8 | SIMD-Regbase：OOO 指令双发 | Guide §6 SIMD-Regbase | 同上 |
| 9 | Warp Scheduler 每 AIV 4 个 | Guide §5 SIMT vs SIMD | 同上 |
| 10 | NDDMA + ND-DMA Cache 规格 | Guide §8 NDDMA 高维 DMA | 同上 |
| 11 | CCU 三种通信范式及 KFC 调度变化 | Guide §9 CCU 通算融合 | 同上 |
| 12 | Vector 算力 950PR Server=54T / PCIE=47T：基准 27T/23.7T × 双发 2，但并非所有 Vector 指令均支持双发 | Guide §3 算力与系统规格 → Vector 算力推导 | 公开资料 / 经验值 |
| 13 | Memory 带宽 Server 1.6 TB/s / PCIE 1.4 TB/s | Guide §3 算力与系统规格 | 同上 |
