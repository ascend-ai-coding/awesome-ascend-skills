# SHMEM 算子测试用例规模标准

本文定义 SHMEM 算子测试用例的规模分档、边界条件、dtype 覆盖和最小 case 数要求。所有 case matrix 生成必须参照此标准。

## 1. 规模分档

### 1.1 通信算子（以每 PE 搬运字节数为主轴）

| 分档 | 每 PE 数据量 | 典型 shape（fp16） | 用途 |
| --- | --- | --- | --- |
| XS (极小) | < 64KB | (64, 32), (128,), (256,) | 边界测试、启动开销、最小功能验证 |
| S (小) | 64KB ~ 1MB | (256, 128), (512, 256), (1024, 256) | smoke 正确性验证 |
| M (中) | 1MB ~ 64MB | (1024, 1024), (2048, 2048), (4096, 4096) | 常规性能测试、中等规模正确性 |
| L (大) | ≥ 64MB（全 PE 总量 ≥ 256MB） | (8192, 8192), (16384, 4096), (32768, 2048) | 性能达标测试 |

**数据量计算**：`per_pe_bytes = elements_per_pe × sizeof(dtype)`

**L 档要求**：集合通信总数据量 ≥ 256MB（即 `per_pe_bytes × n_pes ≥ 256MB`）。

### 1.2 通算融合算子（MatMul/GEMM + 跨 PE 通信）

以 **MatMul 规模（M×N×K）** 与 **通信 payload** 双轴划分：

| 分档 | MatMul 规模（fp16 示例） | 通信 payload（每 PE） | 用途 |
| --- | --- | --- | --- |
| XS | M,N,K ≤ 128 | < 64KB | 边界、smoke |
| S | M=N=K=512（~0.5M elements/矩阵） | 64KB ~ 1MB | **Phase 6 小数据量必采**；端到端时延、小消息行为 |
| M | M=N=2048, K=1024 量级 | 1MB ~ 64MB | 中等规模正确性、辅性能 |
| L | M=N=4096, K=2048 及以上（≥2G FLOPs 量级） | ≥ 64MB/PE 或 design `min_scale` | **Phase 6 大数据量必采**；端到端 e2e / 可选诊断分段 |

**Phase 6 门禁**（见 [performance-eval-guide.md §4](../../shmem-ops-performance-eval/references/performance-eval-guide.md)）：
- **MUST** 覆盖 **S 档 + L 档** 各至少 1 个 performance case
- 通算融合 performance case 在 `cases.json` 中 `category: performance`，correctness matrix 可 SKIP，但 Phase 6 **MUST** 单独执行
- PE 数推荐 8PE，资源受限时可 4PE/2PE（**MUST** 在报告标注）


## 2. PE 数覆盖

| PE 数 | 用途 | 必测级别 |
| --- | --- | --- |
| 2 | 最小拓扑，smoke 和功能验证 | 必测 |
| 4 | 中等规模，覆盖非 2-PE 分片逻辑 | 推荐 |
| 8 | 单机 8 卡标准配置，性能推荐 | 性能推荐（非强制；可降级 4/2PE） |

case matrix 至少覆盖 2 PE；性能 case 推荐 8 PE（资源受限可按 [performance-eval-guide.md §4.3](../../shmem-ops-performance-eval/references/performance-eval-guide.md) 降级为 4/2PE）。4 PE 推荐覆盖，尤其是存在 PE 不整除分片时。

## 3. 边界条件（SHMEM 特有）

以下边界条件必须在 case matrix 中至少各有 1 个 case 覆盖：

| 编号 | 边界类别 | 描述 | 典型构造方式 |
| --- | --- | --- | --- |
| B1 | chunk 不整除 | shape 不整除 chunk_size，产生 tail chunk | 如 chunk=190KB，数据 200KB → 余 10KB |
| B2 | PE 不整除 | shape 不整除 PE 数，产生不均匀分片 | 如 N=1000, n_pes=8 → 每 PE 125，无余 |
| B3 | 非 2 幂次维度 | shape 含质数或非对齐维度 | (7, 13), (3, 5, 11), (127, 63) |
| B4 | 单元素/单行退化 | 退化到极小值的维度 | (1,), (1, 256), (1, 1, 64) |
| B5 | UB 容量边界 | 单次搬运接近 UB 容量（~192KB） | 构造 per_chunk_bytes ≈ 190KB ± 5KB |
| B6 | signal/state 复用 | 多轮 repeat 验证 epoch/magic/清零逻辑 | repeat=3 或 repeat=5 |
| B7 | 最小 PE（2PE）+ 最大 shape | 2 PE 下运行 L 档 shape | 验证大数据量下 2 PE 正确性 |
| B8 | 最大 PE + 最小 shape | 8 PE 下运行 XS 档 shape | 验证多 PE 下极小数据的正确分发 |

### dtype 覆盖

### 4.1 通信算子

按 design.md 支持的 dtype 全覆盖。常见 dtype：

| dtype | sizeof | 典型场景 |
| --- | --- | --- |
| float16 | 2B | 主路径，必测 |
| float32 | 4B | 高精度路径 |
| bfloat16 | 2B | 训练场景 |
| int32 | 4B | 索引/metadata 搬运 |
| int8 | 1B | 量化场景 |

### 4.2 覆盖规则

- **每个规模分档 × 每个支持的 dtype** 至少 1 个 case
- 边界 case 使用主 dtype（通常 fp16）即可，不需全 dtype 交叉
- 性能 case（L 档）至少覆盖主 dtype

## 5. Case Matrix 模板

### 5.1 通信算子（以 AllGather 为例）

| case_id | category | scale | n_pes | dtype | shape (per PE) | 特殊条件 | 用途 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| smoke_2pe_fp16 | functional | XS | 2 | fp16 | (128, 64) | - | 最小功能验证 |
| smoke_2pe_fp32 | functional | XS | 2 | fp32 | (128, 64) | - | fp32 路径验证 |
| smoke_2pe_int32 | functional | XS | 2 | int32 | (128, 64) | - | 整型路径验证 |
| small_2pe_fp16 | functional | S | 2 | fp16 | (512, 256) | - | 正确性 |
| small_2pe_fp32 | functional | S | 2 | fp32 | (512, 256) | - | 正确性 |
| small_2pe_int32 | functional | S | 2 | int32 | (512, 256) | - | 正确性 |
| medium_2pe_fp16 | functional | M | 2 | fp16 | (2048, 1024) | - | 中等规模 |
| medium_2pe_fp32 | functional | M | 2 | fp32 | (2048, 1024) | - | 中等规模 |
| medium_2pe_int32 | functional | M | 2 | int32 | (2048, 1024) | - | 中等规模 |
| large_8pe_fp16 | performance | L | 8 | fp16 | (8192, 8192) | - | 性能达标 |
| large_8pe_fp32 | performance | L | 8 | fp32 | (8192, 4096) | - | 性能达标 |
| large_8pe_int32 | performance | L | 8 | int32 | (8192, 4096) | - | 性能达标 |
| tail_chunk_2pe | boundary | S | 2 | fp16 | (513, 257) | B1: 不整除 chunk | 边界 |
| tail_pe_4pe | boundary | M | 4 | fp16 | (1000, 1024) | B2: 不整除 PE | 边界 |
| unaligned_2pe | boundary | XS | 2 | fp16 | (7, 13) | B3: 非 2 幂次 | 边界 |
| degenerate_2pe | boundary | XS | 2 | fp16 | (1, 256) | B4: 单行 | 边界 |
| ub_boundary_2pe | boundary | S | 2 | fp16 | (根据 UB 计算) | B5: UB 边界 | 边界 |
| repeat_3x_2pe | stress | S | 2 | fp16 | (512, 256) | B6: repeat=3 | signal 复用 |
| large_2pe_fp16 | boundary | L | 2 | fp16 | (16384, 4096) | B7: 2PE+大 shape | 边界 |
| xs_8pe_fp16 | boundary | XS | 8 | fp16 | (64, 32) | B8: 8PE+极小 shape | 边界 |
| medium_4pe_fp16 | functional | M | 4 | fp16 | (2048, 1024) | 4 PE 覆盖 | 正确性 |

**统计**：21 case，覆盖 3 dtype × 4 scale + 8 boundary + 1 stress

### 5.2 通算融合算子（以 MatMul + AllToAll 为例）

| case_id | category | scale | n_pes | dtype | shape (M,N,K) | 通信说明 | 用途 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| smoke_4pe_fp16 | functional | XS | 4 | fp16 | (128,128,128) | 小 payload | smoke |
| func_4pe_fp16 | functional | S | 4 | fp16 | (512,512,512) | S 档通信量 | 正确性 |
| medium_4pe_fp16 | functional | M | 4 | fp16 | (2048,1024,1024) | M 档 | 正确性 |
| perf_s_8pe_fp16 | performance | S | 8 | fp16 | (512,512,512) | S 档；Phase 6 必采 | 小数据量性能 |
| perf_l_8pe_fp16 | performance | L | 8 | fp16 | (4096,4096,2048) | L 档通信 payload | 大数据量性能 |
| tail_k_4pe | boundary | S | 4 | fp16 | (513,257,128) | K 不整除 | 边界 |
| repeat_3x_4pe | stress | S | 4 | fp16 | (512,512,512) | repeat=3 | signal 复用 |

**统计**：通算融合 case matrix **≥ 12**（含 2 个 performance S/L）；Phase 6 **MUST** 执行 `perf_s_*` 与 `perf_l_*`。

## 6. 最小 case 数要求

**总 case 数 ≥ 20**

分类最低要求：

| 类别 | 最低数量 | 说明 |
| --- | --- | --- |
| functional | ≥ 4 shape × len(dtype) | XS + S + M + L 各至少 1 shape，每 shape 覆盖所有 dtype |
| boundary | ≥ 4 | chunk tail, PE tail, 非对齐, UB 边界各至少 1 |
| stress | ≥ 1 | repeat 多轮验证 signal/state 复用 |
| performance | ≥ 1 | L 档 shape + 8 PE |
| 多 PE 覆盖 | ≥ 2 种 PE 数 | 至少覆盖 2 PE 和 8 PE |

**计算公式**：`total_cases = len(SCALE_SHAPES) × len(SUPPORTED_DTYPES) + len(BOUNDARY_CASES) + len(STRESS_CASES)`

如果 `total_cases < 20`，需追加 shape × dtype 交叉或增加边界 case。

## 7. 与其他 Skill 的关系

| Skill | 使用的 case 档次 | 要求 |
| --- | --- | --- |
| shmem-ops-testcase-gen | 全部（XS~L + 边界 + 压力） | 生成完整 case matrix |
| shmem-ops-correctness-eval | XS~M + 边界 | 正确性验证，不要求 L 档性能 |
| shmem-ops-performance-eval | M + L | 性能采集必须包含 L 档，不能只用 XS/S |
| shmem-ops-performance-optim | L（主）+ M（辅） | 优化迭代以 L 档为主指标 |
