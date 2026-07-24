# msprof op CSV 字段完整参考

数据来源: MindStudio 8.3.0 官方文档。适用于 Atlas A2/A3 训练系列产品和推理系列产品。

> 字段名中 `ai*` 表示 `aic`（Cube Core）或 `aiv`（Vector Core），具体取决于算子类型。

---

## 目录

1. [OpBasicInfo.csv](#1-opbasicinfocsv)
2. [PipeUtilization.csv](#2-pipeutilizationcsv)
3. [ArithmeticUtilization.csv](#3-arithmeticutilizationcsv)
4. [Memory.csv](#4-memorycsv)
5. [MemoryL0.csv](#5-memoryl0csv)
6. [MemoryUB.csv](#6-memoryubcsv)
7. [L2Cache.csv](#7-l2cachecsv)
8. [ResourceConflictRatio.csv](#8-resourceconflictratiocsv)

---

## 1. OpBasicInfo.csv

算子基本信息，首先查看此文件了解全局情况。

| 字段名 | 含义 | 关注点 |
|--------|------|--------|
| Op Name | 算子名称 | — |
| Op Type | 算子类型 | — |
| Task Duration(us) | Task 总耗时（含调度+执行+响应），单位 us | **核心指标**，与理论耗时对比判定性能 |
| Block Dim | Task 运行核数（开发者设置的逻辑核数） | 检查是否用满可用核数 |
| Mix Block Dim | 混合算子从核 blockDim（非 Mix 算子显示 N/A） | 仅 A2/A3 |
| Device ID | 运行使用的 NPU ID | — |
| PID | 进程号 | — |
| Current Freq | 当前运行频率 | **与 Rated Freq 对比**，确认是否满频 |
| Rated Freq | 理论额定频率 | Current < Rated 说明 DVFS 降频 |

---

## 2. PipeUtilization.csv

**最重要的文件**。各流水线单元的耗时和占比，用于判定瓶颈类型。

每一行对应一个核（由 block_id + sub_block_id 标识）。

### 公共字段

| 字段名 | 含义 |
|--------|------|
| block_id | Task 运行切分数量（核数） |
| sub_block_id | 每个 block 内的 Vector/Cube 核名称和序号 |
| aic_time(us) | 每个 Cube Core 执行时间 |
| aic_total_cycles | 每个 Cube Core 执行 cycle 总数 |
| aiv_time(us) | 每个 Vector Core 执行时间 |
| aiv_total_cycles | 每个 Vector Core 执行 cycle 总数 |

### 流水线单元耗时和占比

| 字段名 | 含义 | 瓶颈判定阈值 |
|--------|------|-------------|
| aiv_vec_time(us) | Vector 指令耗时 | — |
| aiv_vec_ratio | Vector 指令 cycle 占比 | >50% = VEC Bound |
| aic_cube_time(us) | Cube 指令耗时 | — |
| aic_cube_ratio | Cube 指令 cycle 占比 | >40% 正常(MatMul)；VEC 算子应≈0 |
| ai*_scalar_time(us) | Scalar 指令耗时 | — |
| ai*_scalar_ratio | Scalar 指令 cycle 占比 | >30% = SCALAR Bound |
| aic_fixpipe_time(us) | FixPipe 指令（L0C→GM/L1）耗时 | — |
| aic_fixpipe_ratio | FixPipe cycle 占比 | >15% = **地址未对齐** |
| aic_mte1_time(us) | MTE1（L1→L0A/L0B）耗时（不含等待） | — |
| aic_mte1_ratio | MTE1 cycle 占比 | — |
| ai*_mte2_time(us) | MTE2（GM→AICORE）耗时 | — |
| ai*_mte2_ratio | MTE2 cycle 占比 | >50% = MTE2 Bound |
| ai*_mte3_time(us) | MTE3（AICORE→GM）耗时 | — |
| ai*_mte3_ratio | MTE3 cycle 占比 | — |
| ai*_icache_miss_rate | ICache 缺失率 | >15% 需关注 |

### 活跃带宽（仅 A2/A3）

| 字段名 | 含义 |
|--------|------|
| aiv_mte2_active_bw(GB/s) | Vector 核 MTE2 活跃带宽 |
| aiv_mte3_active_bw(GB/s) | Vector 核 MTE3 活跃带宽 |
| aic_mte1_active_bw(GB/s) | Cube 核 MTE1 活跃带宽（需 MemoryDetail） |
| aic_mte2_active_bw(GB/s) | Cube 核 MTE2 活跃带宽（需 MemoryDetail） |
| aic_mte3_active_bw(GB/s) | Cube 核 MTE3 活跃带宽 |
| aic_fixpipe_active_bw(GB/s) | Cube 核 FixPipe 活跃带宽 |

---

## 3. ArithmeticUtilization.csv

Cube 和 Vector 类型指令的详细占比和计算量。

### 公共字段

同 PipeUtilization.csv（block_id, sub_block_id, aic_time, aiv_time 等）。

### Cube 指令详情

| 字段名 | 含义 |
|--------|------|
| aic_cube_ratio | Cube 指令 cycle 占比 |
| aic_cube_fp16_ratio | Cube fp16 指令占比 |
| aic_cube_int8_ratio | Cube int8 指令占比 |
| aic_cube_fops | Cube 浮点运算总数（用于计算理论利用率） |
| aic_cube_total_instr_number | Cube 指令总条数 |
| aic_cube_fp_instr_number | Cube fp 类型指令条数 |
| aic_cube_int_instr_number | Cube int 类型指令条数 |

### Vector 指令详情

| 字段名 | 含义 |
|--------|------|
| aiv_vec_ratio | Vec 指令 cycle 占比 |
| aiv_vec_fp32_ratio | Vec fp32 指令占比 |
| aiv_vec_fp16_ratio | Vec fp16 指令占比 |
| aiv_vec_int32_ratio | Vec int32 指令占比 |
| aiv_vec_int16_ratio | Vec int16 指令占比 |
| aiv_vec_misc_ratio | Vec misc 类型指令占比 |
| aiv_vec_fops | Vector 浮点运算总数 |

---

## 4. Memory.csv

内存读写带宽和数据搬运量。

### 带宽速率

| 字段名 | 含义 |
|--------|------|
| aiv_gm_to_ub_bw(GB/s) | GM→UB 写入带宽 |
| aiv_ub_to_gm_bw(GB/s) | UB→GM 写入带宽 |
| aic_l1_read_bw(GB/s) | L1 读带宽 |
| aic_l1_write_bw(GB/s) | L1 写带宽 |
| ai*_main_mem_read_bw(GB/s) | 主存储器读带宽 |
| ai*_main_mem_write_bw(GB/s) | 主存储器写带宽 |

### 指令统计

| 字段名 | 含义 |
|--------|------|
| aic_mte1_instructions | MTE1 指令条数 |
| aic_mte1_ratio | MTE1 cycle 占比 |
| ai*_mte2_instructions | MTE2 指令条数 |
| ai*_mte2_ratio | MTE2 cycle 占比 |
| ai*_mte3_instructions | MTE3 指令条数 |
| ai*_mte3_ratio | MTE3 cycle 占比 |

### 数据搬运量

| 字段名 | 含义 |
|--------|------|
| read_main_memory_datas(KB) | 读主存储器总量 |
| write_main_memory_datas(KB) | 写主存储器总量 |
| GM_to_L1_datas(KB) | GM→L1 搬运量 |
| L1_to_GM_datas(KB)(estimate) | L1→GM 搬运量（估算） |
| L0C_to_L1_datas(KB) | L0C→L1 搬运量 |
| L0C_to_GM_datas(KB) | L0C→GM 搬运量 |
| GM_to_UB_datas(KB) | GM→UB 搬运量 |
| UB_to_GM_datas(KB) | UB→GM 搬运量 |

### 带宽利用率

| 字段名 | 含义 | 达标标准 |
|--------|------|---------|
| GM_to_L1_bw_usage_rate(%) | GM→L1 带宽利用率 | >60% 良好 |
| L1_to_GM_bw_usage_rate(%)(estimate) | L1→GM 带宽利用率 | >60% 良好 |
| L0C_to_L1_bw_usage_rate(%) | L0C→L1 带宽利用率 | — |
| L0C_to_GM_bw_usage_rate(%) | L0C→GM 带宽利用率 | — |
| GM_to_UB_bw_usage_rate(%) | GM→UB 带宽利用率 | >60% 良好 |
| UB_to_GM_bw_usage_rate(%) | UB→GM 带宽利用率 | >60% 良好 |

---

## 5. MemoryL0.csv

L0A/L0B/L0C 读写带宽，主要关注 Cube 场景。

| 字段名 | 含义 |
|--------|------|
| aic_l0a_read_bw(GB/s) | L0A 读带宽 |
| aic_l0a_write_bw(GB/s) | L0A 写带宽 |
| aic_l0b_read_bw(GB/s) | L0B 读带宽 |
| aic_l0b_write_bw(GB/s) | L0B 写带宽 |
| aic_l0c_read_bw_cube(GB/s) | Cube 从 L0C 读带宽 |
| aic_l0c_write_bw_cube(GB/s) | Cube 向 L0C 写带宽 |

---

## 6. MemoryUB.csv

Vector 和 Scalar 对 UB 的读写带宽。

| 字段名 | 含义 |
|--------|------|
| aiv_ub_read_bw_vector(GB/s) | Vector 从 UB 读带宽 |
| aiv_ub_write_bw_vector(GB/s) | Vector 向 UB 写带宽 |
| aiv_ub_read_bw_scalar(GB/s) | Scalar 从 UB 读带宽 |
| aiv_ub_write_bw_scalar(GB/s) | Scalar 向 UB 写带宽 |

---

## 7. L2Cache.csv

L2 Cache 命中率数据。

### Atlas A2/A3

| 字段名 | 含义 | 达标标准 |
|--------|------|---------|
| ai*_write_cache_hit | 写 cache 命中次数 | — |
| ai*_write_cache_miss_allocate | 写 cache 缺失重新分配次数 | — |
| ai*_r*_read_cache_hit | 读 r* 通道 cache 命中次数 | — |
| ai*_r*_read_cache_miss_allocate | 读 r* 通道缺失重新分配次数 | — |
| ai*_write_hit_rate(%) | 写 cache 命中率 | >80% |
| ai*_read_hit_rate(%) | 读 cache 命中率 | >80% |
| ai*_total_hit_rate(%) | 读/写 cache 总命中率 | **>80% 良好，<50% 需优化** |

---

## 8. ResourceConflictRatio.csv

UB 上的 bank group、bank conflict 和资源冲突占比。**上板独有数据，仿真无法获取。**

### 核心冲突指标

| 字段名 | 含义 | 达标标准 |
|--------|------|---------|
| aiv_vec_total_cflt_ratio | Vector 指令被阻塞的总占比 | **<5% 良好，>15% 严重** |
| aiv_vec_bankgroup_cflt_ratio | 被 bankgroup 冲突阻塞占比（block stride 不合理） | <3% |
| aiv_vec_bank_cflt_ratio | 被 bank 冲突阻塞占比（读写指针地址不合理） | <3% |
| aiv_vec_resc_cflt_ratio | 被计算单元资源冲突阻塞占比 | <5% |
| aiv_vec_mte_cflt_ratio | 被 MTE 冲突阻塞占比 | <3% |

### 等待指标

| 字段名 | 含义 |
|--------|------|
| aic_cube_wait_ratio | Cube 单元被阻塞占比 |
| aiv_vec_wait_ratio | Vector 单元被阻塞占比 |
| ai*_mte1_wait_ratio | MTE1 被阻塞占比 |
| ai*_mte2_wait_ratio | MTE2 被阻塞占比 |
| ai*_mte3_wait_ratio | MTE3 被阻塞占比 |

### Bank Conflict 解读

- **bankgroup 冲突**: UB 中每个 bank group 包含多个 bank。Vector 指令的 `block_stride` 设置不合理导致多次访问落入同一 bank group → 调整 block stride
- **bank 冲突**: Vector 指令操作数的读写指针地址不合理，多个操作数命中同一 bank → 调整 UB 地址分配，添加 padding
- **资源冲突**: 多个计算单元竞争同一执行单元 → 调整流水编排，错开不同计算单元的调度
- **MTE 冲突**: Vector 和 MTE 操作竞争共享资源 → 调整数据搬运和计算的时序

### 优化建议

| 冲突类型 | 优化方向 |
|---------|---------|
| bankgroup_cflt 高 | 修改 block_stride（Vector API 的 repeatStride/blockStride 参数） |
| bank_cflt 高 | 修改 UB tensor 起始地址，添加 padding 使不同操作数落入不同 bank |
| resc_cflt 高 | 错开 Vector 和 Cube 调度，增加流水并行度 |
| mte_cflt 高 | 错开 MTE2/MTE3 和 Vector 操作的执行时序 |
