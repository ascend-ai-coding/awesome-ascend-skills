# 生态算子开源精度标准

## 误差指标

当前该标准主要用来衡量生态贡献中的计算类算子是否达标，采用平均相对误差和最大相对误差指标来判断，计算公式如下：

1. 平均相对误差（Mean Relative Error，MERE）：采样点中相对误差的平均值。

$$
\text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden) +\text{1e-7}})
$$

计算相对误差的时候引入小值1e-7 以避免golden出现除0风险。

2. 最大相对误差（Max Relative Error, MARE）：采样点中相对误差的最大值。

$$
\text{MARE} = \text{max}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden) +\text{1e-7}})
$$

## 通过标准

**单标杆比对**：与更高精度的实现的单一精度标杆（CPU、GPU或者昇腾小算子拼接）直接比较。

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|---------|---------|----------|---------|-----------|-------------|-------------|
| **通过阈值 (Threshold)** | 2⁻¹⁰ | 2⁻⁷ | 2⁻¹³ | 2⁻¹¹ | 2⁻³ | 2⁻² |

**通过标准**：当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 判断为通过。

## 迁移场景特殊考虑

### 950 新增数据类型

| 数据类型 | Threshold | 说明 |
|---------|-----------|------|
| FP8 E4M3FN | 2⁻³ | 950 新增，L2 迁移时常见 |
| FP8 E5M2 | 2⁻² | 950 新增，L2 迁移时常见 |
| HiFloat8 | 2⁻³ | 950 新增，L2 迁移时常见 |
| INT8 | 0 | 量化类型，要求精确匹配（反量化后对比） |

### 量化算子（Quant）精度测试设计规范

量化算子（输出为 INT8/INT4 等整型的算子）的精度测试需要注意以下要点。

#### 核心认知

1. **quantScale 取值不需要限制**：即使 quantScale 很小导致大量值溢出 INT8 范围，NPU 和 CPU 两边都会 clamp 到 ±127，溢出区域结果一致
2. **CPU 参考实现必须加 clamp**：`round(pre_quant / quantScale).clamp(-128, 127).to(torch.int8)`，与 NPU 算子行为对齐
3. **精度差异来源**：仅来自非溢出区域的 FP16/BF16 vs FP32 中间计算差异，round 后最多差 1

#### CPU 参考实现模板

所有低精度输出类型（整型 + 低精度浮点）**MUST** 在类型转换前加 `clamp`，确保与 NPU 算子行为一致：

| 输出类型 | clamp 范围 | 代码 |
|---------|-----------|------|
| INT8 | [-128, 127] | `.clamp(-128, 127).to(torch.int8)` |
| UINT8 | [0, 255] | `.clamp(0, 255).to(torch.uint8)` |
| INT4（存储为 INT8） | [-8, 7] | `.clamp(-8, 7).to(torch.int8)` |
| FP4 E2M1（存储为 UINT8/INT8） | [-6.0, 6.0] | `.clamp(-6.0, 6.0)` 后按算子文档存储方式转换 |
| FP8 E4M3FN | [-448.0, 448.0] | `.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)` |
| FP8 E5M2 | [-57344.0, 57344.0] | `.clamp(-57344.0, 57344.0).to(torch.float8_e5m2)` |
| FP16 | [-65504.0, 65504.0] | `.clamp(-65504.0, 65504.0).to(torch.float16)` |
| BF16 | [-3.389e38, 3.389e38] | `.clamp(-3.389e38, 3.389e38).to(torch.bfloat16)` |

```python
# ✅ 正确：round 后 clamp 再转目标类型
cpu_out = torch.round(silu_out / quant_scale).clamp(-128, 127).to(torch.int8)

# ❌ 错误：不加 clamp，溢出时 PyTorch 行为未定义
cpu_out = torch.round(silu_out / quant_scale).to(torch.int8)
```

#### INT8 精度判定标准

NPU 使用 FP16/BF16 计算 pre-quant 值，CPU 使用 FP32，浮点精度差异在 round 后可能差 1。

**完整精度判定矩阵**（官方标准）：

| 输入类型 \ 输出类型 | 整型输出（INT4/INT8/INT16 等） | 浮点输出（FP4/FP8/FP16/BF16 等） |
|:---|:---|:---|
| 整型输入（INT4/INT8 等） | N/A | 参考通用浮点精度标准 |
| 浮点输入（FP4/FP8/FP16/BF16 等） | **MaxAbsErr ≤ 1** | 参考通用浮点精度标准 |

**注意**：
- 如果 MaxAbsErr > 1，应先排查输入数据是否包含 NaN（`torch.rand(..., device='npu')` 可能产生 NaN），再排查 quant_scale 取值是否合理（是否导致 INT8 溢出），而非直接判定为算子 bug。
- 以上阈值为实测经验值，非官方标准。不同算子可能需要根据实际情况调整。

### NPU 随机数生成注意事项

`torch.rand(..., device='npu')` 在 NPU 设备状态异常时可能产生 NaN（偶发，设备重置后恢复），导致精度测试假性失败。

```python
# ✅ 推荐：CPU 生成再传 NPU，更稳定
quant_scale = (torch.rand(C, dtype=torch.float32) * 1.0 + 0.001).to('npu:0')

# ⚠️ 可用但不推荐：直接在 NPU 上生成
quant_scale = torch.rand(C, dtype=torch.float32, device='npu:0') * 1.0 + 0.001
```

**排查方法**：当精度测试出现异常大的 MaxAbsErr（如 >10）时，首先检查输入数据是否包含 NaN：
```python
assert not torch.isnan(x).any(), "Input contains NaN!"
assert not torch.isnan(quant_scale).any(), "quant_scale contains NaN!"
```

### 跨架构精度对比

迁移验证时，950 NPU 输出应与 CPU 参考实现（PyTorch float32 计算）对比。如有 910b 基线数据，应额外进行 950 vs 910b 对比，确保迁移未引入精度退化。

### L2 MicroAPI 重写后的精度验证

L2 重写涉及 CastTrait 替换原 Cast+RoundMode，需特别关注：
- SatMode 选择是否正确（量化 SAT，反量化 NO_SAT）
- 三步量化中间类型是否正确（FP32→INT16→FP16→INT8）
- 溢出模式控制（GetCtrlSpr/SetCtrlSpr）是否正确保存/恢复

### L3 SIMT 优化后的精度验证

L3 涉及多线程并行，需特别关注：
- 多线程写同一地址是否使用 AtomicAdd
- SIMT 线程间数据竞争是否导致结果不确定
- 多次运行结果是否一致（确定性验证）
