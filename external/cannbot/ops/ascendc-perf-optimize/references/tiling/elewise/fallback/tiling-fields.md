# Elementwise 族 — TilingData 字段语义

> Elementwise 族 TilingData 字段定义。

---

## 1. 字段定义

| 字段 | 类型 | 含义 | 推导出处 |
|------|------|------|---------|
| `dim0` | int64 | 元素总数量 | 所有维度相乘 |
| `dtype` | str | 数据类型 | 输入参数 |
| `elem_bytes` | int | 单元素字节数 | dtype映射 |
| `core_num` | int32 | 实际使用的核数 | `min(ceil(dim0*bits/MIN_TILING_BITS), maxCores)` |
| `block_former` | int64 | 每个核的基础元素数(512对齐) | `AlignUp(CeilDiv(dim0, coreNum), 512)` |
| `block_num` | int64 | 虚拟block数量 | `CeilDiv(dim0, blockFormer)` |
| `block_tail` | int64 | 尾block的元素数 | `dim0 - (blockNum-1)*blockFormer` |
| `ub_former` | int64 | UB每次处理的元素数(256B对齐) | UB容量 / bufferDivisor 向下对齐 |
| `ub_loop_former` | int64 | 首block的UB循环次数 | `CeilDiv(blockFormer, ubFormer)` |
| `ub_tail_former` | int64 | 首block的尾块大小 | `blockFormer - (loopFormer-1)*ubFormer` |
| `ub_loop_tail` | int64 | 尾block的UB循环次数 | `CeilDiv(blockTail, ubFormer)` |
| `ub_tail_tail` | int64 | 尾block的尾块大小 | `blockTail - (loopTail-1)*ubFormer` |
| `use_fp32_cast` | bool | 是否启用FP32升精度 | dtype + op_type判定 |
| `K_extra_buffers` | int | 升精度额外FP32 buffer份数 | API别名约束 |

## 2. 约束常量

| 常量 | 值 | 说明 |
|------|----|------|
| `MIN_TILING_BITS` | 32768 (4KB) | 每核最小数据量 |
| `ELEM_ALIGN_FACTOR` | 512 | 多核切分元素对齐因子 |
| `ALIGN_256` | 256 | UB对齐字节数 |

## 3. 经验规则

| 规则 | 说明 |
|------|------|
| 最小粒度 | 每核≥4KB，否则不值得开核 |
| 多核对齐 | 元素数对齐到512 |
| UB对齐 | 按256B对齐确保Vector指令效率 |
| 跨核偏移 | `blockFormer * blockIdx * sizeof(dtype)` |
