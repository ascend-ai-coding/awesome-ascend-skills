---
type: reference
status: experimental
platforms:
- '950'
sources:
- resource: asc-devkit master include/tensor_api/experimental
  role: primary
- resource: 'cannbot-knowledge: ops/ascendc/apis/simd_api/tensor_api/reg_vector_compute/reg_tensor/reg_tensor.md'
  role: supporting
updated_at: '2026-10-29'
---

# Experimental Reg Vector Compute

> 入口：`tensor_api/experimental/vector_ops.h`。命名空间：`asc::te::experimental`。当前公开实现主要面向 `__NPU_ARCH__ == 3510`；非 3510 目标必须逐 API 查看当前头文件注释和实现。

## 定位

Experimental Reg Vector Compute 提供基于 `reg_tensor` 的向量计算、寄存器 load/store、gather、mask、数据重排和同步能力。它是实验性接口，稳定性弱于基础 Tensor API；用于设计前必须确认目标架构、dtype、mask 语义和官方样例证据。

## reg_tensor 与 mask

`reg_tensor<T>` 是寄存器 Tensor。部分操作会携带或更新 active-lane mask。使用时必须区分：

- 数值结果。
- active-lane mask。
- masked-out 元素的值是否被定义为零、保持原值或未定义。

## 新增 API

### move_mask

```cpp
template <int16_t Offset, typename T>
__simd_callee__ inline reg_tensor<bool> move_mask(const reg_tensor<T>& src);
```

- 从 `src` 的指定 data block 中抽取 bit，并扩展为 register mask。
- `T` 仅支持 `uint16_t` / `uint32_t`。
- `Offset` 有效范围：`uint16_t` 为 `[0,15]`，`uint32_t` 为 `[0,31]`。

### move

```cpp
template <typename T>
__simd_callee__ inline void move(reg_tensor<T>& dst, const reg_tensor<T>& src);
```

- 复制 `src.mask` 选中的 active 元素。
- `dst.mask` 更新为 `src.mask`。
- 支持 `bool`、`uint8_t`、`int8_t`、`uint16_t`、`int16_t`、`half`、`bfloat16_t`、`uint32_t`、`int32_t`、`float`。

### mem_bar

```cpp
template <mem_type src, mem_type dst>
__simd_callee__ inline void mem_bar();
```

- 同步 SIMD VF 内两个 pipeline 的 UB 访问。
- 源 pipeline 的前序 UB 访问完成后，目标 pipeline 的后续 UB 访问才继续。
- 不同步 MTE pipeline，不跨 NPU core。
- 只允许当前头文件文档声明的 source/destination 组合。

## Mask API

| API | 作用 |
|---|---|
| `make_mask<Pattern, DataType>()` | 构造指定 pattern 的 register mask |
| `all_mask<DataType>()` | 全 lane 激活 |
| `none_mask<DataType>()` | 全 lane 关闭 |
| `update_mask<DataType>(remain)` | 为当前迭代构造 mask，并更新剩余元素数 |
| `interleave<DataType>(src0, src1)` | 交织两个 mask |
| `deinterleave<DataType>(src0, src1)` | 反交织两个 mask |

支持 b8/b16/b32 元素宽度。

## Load / Store

- `load(tensor)` / `load(tensor, coord)`：从 UB Tensor 载入 register tensor。
- `load_broadcast(tensor)` / `load_broadcast(tensor, coord)`：载入并按 broadcast mode 广播。
- `load_gather(tensor, index, mask)`：按 element 或 datablock mode gather。
- `store(tensor, value)` / `store(tensor, coord, value)`：写回 UB Tensor。
- `store_scatter` 支持 scatter 写回。
- `unalign_load` / `unalign_store` 维护非对齐访问的硬件状态，必须按头文件状态机调用 prepare / load / store。

## 能力模块

| 模块 | 能力 |
|---|---|
| basic_arithmetic | add/sub/mul/fma 等 |
| compare_and_select | compare/select |
| composite_compute | 复合计算 |
| data_padding | fill/pad |
| data_reorder | interleave/deinterleave/pack/unpack |
| gather_compute | gather |
| histogram_compute | frequency/cumulative histogram |
| logical_compute | and/or/xor/not/shift |
| mask_reg_compute | mask 构造和更新 |
| reduction_compute | reduce sum/min/max |
| reg_compute_sel | high/low part 选择 |
| reg_index_compute | iota/index |
| type_conversion | dtype cast |
| reg_sync_control | `mem_bar` |

## 官方样例

见 [sample-router.md](sample-router.md) 的 Experimental Reg Vector 样例表。

## 门禁

1. 必须声明 experimental 路线并记录 devkit commit。
2. 每个 API 使用前核对当前头文件注释、模板约束和架构条件。
3. mask 语义必须写入 DESIGN；masked-out 值不得参与 golden。
4. `mem_bar` 只解决 VF 内 UB 访问同步，不替代 MTE/Cube/Vector 的事件同步。
5. 无官方样例覆盖时，先用最小编译探针和设备用例验证。
