> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/data_move/DataCopyGM2L1.md

# DataCopyGM2L1

## 产品支持情况

|产品|是否支持|
|--|:-:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

DataCopyGM2L1提供数据搬运功能，该接口支持Global Memory到L1 Buffer之间的数据搬运，同时支持随路数据格式转换。支持以下数据格式转换场景：

- ND2ND：ND格式到ND格式的数据搬运
- ND2NZ：ND格式到NZ格式的数据搬运
- ND2ZN：ND格式到ZN格式的数据搬运
- DN2NZ：DN格式到NZ格式的数据搬运
- DN2ZN：DN格式到ZN格式的数据搬运
- NZ2NZ：NZ格式到NZ格式的数据搬运
- ZN2ZN：ZN格式到ZN格式的数据搬运
- Scale相关格式转换：ScaleAND2ZZ、ScaleADN2ZZ、ZZ2ZZ、ScaleBND2NN、ScaleBDN2NN、NN2NN

### 不支持坐标偏移的接口

该接口提供基础的数据搬运功能，直接从源操作数的起始地址搬运数据到目的操作数的起始地址。

**使用场景**：

- 当需要从源数据的起始位置开始搬运时
- 数据搬运的起始位置和大小完全由Tensor的Layout定义

### 支持坐标偏移的接口

该接口在基础数据搬运功能的基础上，支持通过坐标偏移参数指定源操作数上的偏移位置，实现从源数据的指定位置开始搬运。

**使用场景**：

- 当需要从源数据的中间位置开始搬运时
- 当需要分批次搬运源操作数的不同部分时
- 当需要实现数据的分块处理时

## 函数原型

- 不支持坐标偏移的接口

  ```cpp
  template <const DataCopyTrait& trait = DEFAULT_DATA_COPY_TRAIT, typename T, typename U>
  __aicore__ inline typename void DataCopy(const T& dst, const U& src)
  ```

- 支持坐标偏移的接口

  ```cpp
  template <const DataCopyTrait& trait = DEFAULT_DATA_COPY_TRAIT, typename T, typename U, typename Coord>
  __aicore__ inline void DataCopy(const T& dst, const U& src, const Coord& coord)
  ```

## 参数说明

**表 1**  模板参数说明

| 参数名 | 描述 |
| -- | -- |
| DataCopyTrait | 预留参数，保持默认值即可。 |
| T | 目的操作数的数据类型，通过MakeTensor构造的[LocalTensor类型](../struct/tensor/LocalTensor.md)，存储位置支持L1，数据格式支持ND、NZ、ZN、ZZ和NN。 |
| U | 源操作数的数据类型，通过MakeTensor构造的[LocalTensor类型](../struct/tensor/LocalTensor.md)，存储位置支持GM，数据格式仅支持ND、DN、NZ、ZN、ZZ、NN、ScaleAND、ScaleADN、ScaleBND和ScaleBDN。 |
| Coord | 偏移坐标的数据类型，通过MakeCoord构造的[Coord类型](../struct/coord/Coord.md)。 |

**表 2**  参数说明

| 参数名 | 输入/输出 | 含义 |
| -- | -- | -- |
| dst | 输出 | 目的操作数。 |
| src | 输入 | 源操作数。 |
| coord | 输入 | 源操作数上的偏移坐标。 |

## 数据类型和数据格式支持

**表 3**  GM->L1数据类型和数据格式支持

| 源操作数数据格式 | 目的操作数数据格式 | 源操作数/目的操作数数据类型 |
| -- | -- | -- |
| ND | ND | fp4x2_e2m1_t/fp4x2_e1m2_t/int8_t/uint8_t/hifloat8_t/fp8_e5m2_t/fp8_e4m3fn_t/int16_t/uint16_t/half/bfloat16_t/int32_t/uint32_t/float/int64_t/uint64_t |
| ND | NZ | fp4x2_e2m1_t/fp4x2_e1m2_t/int8_t/uint8_t/hifloat8_t/fp8_e5m2_t/fp8_e4m3fn_t/int16_t/uint16_t/half/bfloat16_t/int32_t/uint32_t/float |
| ND | ZN | int8_t/uint8_t/hifloat8_t/fp8_e5m2_t/fp8_e4m3fn_t/int16_t/uint16_t/half/bfloat16_t/int32_t/uint32_t/float |
| DN | NZ | int8_t/uint8_t/hifloat8_t/fp8_e5m2_t/fp8_e4m3fn_t/int16_t/uint16_t/half/bfloat16_t/int32_t/uint32_t/float |
| DN | ZN | fp4x2_e2m1_t/fp4x2_e1m2_t/int8_t/uint8_t/hifloat8_t/fp8_e5m2_t/fp8_e4m3fn_t/int16_t/uint16_t/half/bfloat16_t/int32_t/uint32_t/float |
| NZ | NZ | fp4x2_e2m1_t/fp4x2_e1m2_t/int8_t/uint8_t/hifloat8_t/fp8_e5m2_t/fp8_e4m3fn_t/int16_t/uint16_t/half/bfloat16_t/int32_t/uint32_t/float/int64_t/uint64_t |
| ZN | ZN | fp4x2_e2m1_t/fp4x2_e1m2_t/int8_t/uint8_t/hifloat8_t/fp8_e5m2_t/fp8_e4m3fn_t/int16_t/uint16_t/half/bfloat16_t/int32_t/uint32_t/float/int64_t/uint64_t |
| ScaleAND | ZZ | fp8_e8m0_t |
| ScaleADN | ZZ | fp8_e8m0_t |
| ZZ | ZZ | fp8_e8m0_t |
| ScaleBND | NN | fp8_e8m0_t |
| ScaleBDN | NN | fp8_e8m0_t |
| NN | NN | fp8_e8m0_t |

## 返回值说明

无

## 约束说明

- 地址重叠约束：源操作数与目的操作数位于不同物理位置，不存在地址重叠场景。
- 地址对齐约束：源操作数起始地址要求按元素位宽对齐。目的操作数起始地址要求满足32字节对齐。
- 参数组合约束：无约束。
- 环境影响约束：无约束。
- 特殊数据类型约束：无约束。
- 性能约束：无约束。
- 异常和边界值处理：无约束。
- Tensor Layout相关约束：
  - Shape、Stride只支持四维，针对不同的数据格式和数据类型，四个维度的配置均有不同的约束，部分维度为固定值，不可配置。详见[层次化表达法](../Layout和层次化表述法.md)。
  - Shape、Stride具体维度的数据，仅支持size_t和Std::Int类型。
  - 支持坐标偏移的接口中，coord需要满足对应源操作数分形的对齐要求。
  - ND2ND场景：
    - 一维数据格式的场景下，源操作数Shape中的ShapeRow1或ShapeColumn1需要设置为Std::Int<1>{}。
    - 二维数据格式的场景下，当目的操作数列数大于源操作数列数时，目的操作数列步长需满足32字节对齐；当目的操作数列数等于源操作数列数时无此约束。使用支持坐标偏移的接口时，源操作数列数按偏移后计算。

## 流水类型

PIPE_MTE2

## 调用示例

```cpp
...
// 以GM->L1，half类型为例
using namespace AscendC::Te;
using SrcT = half;
using DstT = half;
size_t m = 128;
size_t k = 64;

auto srcLayout = MakeNDLayout<SrcT>(m, k);
auto dstLayout = MakeNZLayout<DstT>(m, k);

__gm__ half gmAAddr[256*256*2];
__cbuf__ half l1Addr[256*256*2];
auto gmAPtr = MakeGMmemPtr(gmAAddr);
auto l1Ptr = MakeL1memPtr(l1Addr);
auto gmABuff = gmAPtr + 0;
auto l1Buff = l1Ptr + 0;

auto srcTensor = MakeTensor(gmABuff, srcLayout);
auto dstTensor = MakeTensor(l1Buff, dstLayout);

// 调用方式1
Copy(CopyAtom<CopyTraits<CopyGM2L1, DataCopyTraitDefault>>{}, dstTensor, srcTensor);

// 调用方式2
CopyAtom<CopyTraits<CopyGM2L1, DataCopyTraitDefault>>{}.Call(dstTensor, srcTensor);

// 调用方式3
auto atomCopyGM2L1 = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});
atomCopyGM2L1.Call(dstTensor, srcTensor);
...
```

完整样例请参考[TensorAPI样例代码](../../../../../examples/01_simd_cpp_api/02_features/05_tensor_api/matmul_quant_relu/matmul_quant_relu.asc)。
