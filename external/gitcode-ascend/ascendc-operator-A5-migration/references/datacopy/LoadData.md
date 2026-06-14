> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/data_move/LoadData.md

# LoadData

## 产品支持情况

|产品|是否支持|
|--|:-:|
|Ascend 950PR/Ascend 950DT|√|


## 功能说明

LoadData接口支持Local Memory内部的数据搬运，能够完成L1 Buffer -> L0A Buffer或L1 Buffer -> L0B Buffer的矩阵数据搬运。
该接口通过L1 Buffer与L0A Buffer、L0B Buffer数据排布格式，自动完成是否转置的推导。

## 函数原型

- 不支持坐标偏移的接口

  ```cpp
  template<const LoadDataTrait& trait = DEFAULT_LOAD_DATA_TRAIT, typename T, typename U>
  __aicore__ inline void LoadData(const T& dst, const U& src)
  ```

- 支持坐标偏移的接口
  ```cpp
  template<const LoadDataTrait& trait = DEFAULT_LOAD_DATA_TRAIT, typename T, typename U, class Coord>
  __aicore__ inline void LoadData(const T& dst, const U& src, const Coord& coord)
  ```

## 参数说明

**表 1**  模板参数说明

|参数名|描述|
|--|--|
|LoadDataTrait|LoadData相关配置参数结构体，类型为LoadDataTrait：<br>struct LoadDataTrait {<br> &emsp;\_\_aicore__ constexpr LoadDataTrait() {}<br>&emsp;\_\_aicore__ constexpr LoadDataTrait(bool transposedIn) : transposed(transposedIn) {}<br> &emsp;\_\_aicore__ constexpr LoadDataTrait(const LoadDataTrait& trait, bool transposedIn) : transposed(transposedIn) {}<br>&emsp;bool transposed = false;<br>};<br>具体参考[表3](#table3)。DEFAULT_LOAD_DATA_TRAIT为默认取值。|
|T|目的操作数的数据类型，通过MakeTensor构造的[LocalTensor类型](../struct/tensor/LocalTensor.md)。|
|U|源操作数的数据类型，通过MakeTensor构造的[LocalTensor类型](../struct/tensor/LocalTensor.md)。|
|Coord|坐标偏移的数据类型，通过MakeCoord构造的[Coord类型](../struct/coord/Coord.md)。|

**表 2**  参数说明

|参数名|输入/输出|含义|
|--|--|--|
|dst|输出|目的操作数。|
|src|输入|源操作数。|
|coord|输入|源操作数上的偏移坐标。|

**表 3<a id="table3"></a>**  LoadDataTrait参数说明

|参数名|含义|
|--|--|
|transposed|是否启用转置功能，对每个分形矩阵进行转置，默认为false，该参数无需设置，接口内部自动推导，具体参考[数据排布支持](#format)。|

## 数据类型支持

**表 4** L1->L0A/L0B数据类型支持

|源操作数类型|目的操作数类型|是否支持|
|--|--|:-:|
|half|half|√|
|bfloat16_t|bfloat16_t|√|
|uint32_t|uint32_t|√|
|int32_t|int32_t|√|
|float|float|√|
|uint8_t|uint8_t|√|
|int8_t|int8_t|√|
|fp8_e4m3fn_t|fp8_e4m3fn_t|√|
|fp8_e5m2_t|fp8_e5m2_t|√|
|hifloat8_t|hifloat8_t|√|
|fp4x2_e2m1_t|fp4x2_e2m1_t|√|
|fp4x2_e1m2_t|fp4x2_e1m2_t|√|
|uint16_t|uint16_t|√|
|int16_t|int16_t|√|

## 数据排布支持<a id="format"></a>

**表 5** L1->L0A数据排布支持

|源操作数排布|目的操作数排布|是否转置|
|--|--|:-:|
|NZ|NZ|x|
|ZN|NZ|√|

**表 6** L1->L0B数据排布支持

|源操作数排布|目的操作数排布|是否转置|
|--|--|:-:|
|NZ|ZN|√|
|ZN|ZN|x|

## 约束说明

- 地址重叠约束：源操作数与目的操作数位于不同物理位置，不存在地址重叠场景。
- 地址对齐约束：源操作数需要满足32字节对齐，目的操作数需要512字节对齐。
- 参数组合约束：无特殊约束。
- 环境影响约束：无约束。
- 特殊数据类型约束：L1->L0A和L1->L0B通路使能转置时，即ZN2NZ或NZ2ZN，支持的数据类型约束如下：
  - b32数据类型要求源矩阵Shape在K轴方向16对齐。
  - b8数据类型要求源矩阵Shape在M轴方向32对齐。
  - b4数据类型要求源矩阵Shape在M轴方向64对齐。
- 性能约束：无特殊约束。
- 异常和边界值处理：无约束。
- Tensor Layout相关约束：
  - Shape、Stride仅支持四维，针对不同的物理存储位置，四个维度的配置均有不同的约束，部分维度为固定值，不可配置。详见[层次化表述法](../Layout和层次化表述法.md)。
  - Shape、Stride具体维度的数据，仅支持size_t和Std::Int类型。
  - LoadData要求L1 Buffer上Layout排布格式为NZ或ZN，L0A Buffer上Layout排布为NZ，L0B Buffer上Layout排布为ZN。

## 返回值说明

无

## 流水类型

PIPE_MTE1

## 调用示例

```cpp
// 以L1->L0A，half类型为例
using namespace AscendC::Te;
using SrcT = half;
using DstT = half;
size_t m = 128;
size_t k = 64;

auto dstLayout = MakeNZLayout<DstT>(m, k);
auto srcLayout = MakeNZLayout<SrcT>(m, k);

__cbuf__ half l1Addr[256*256*2];
__ca__ half l0AAddr[128*128*2];
auto l1Ptr = MakeL1memPtr(l1Addr);
auto l0APtr = MakeL0AmemPtr(l0AAddr);
auto l1Buff = l1Ptr + 0;
auto l0ABuff = l0APtr + 0;

auto srcTensor = MakeTensor(l1Buff, srcLayout);
auto dstTensor = MakeTensor(l0ABuff, dstLayout);

// 调用方式1
Copy(CopyAtom<CopyTraits<CopyL12L0, LoadDataTraitDefault>>{}, dstTensor, srcTensor);

// 调用方式2
CopyAtom<CopyTraits<CopyL12L0, LoadDataTraitDefault>>{}.Call(dstTensor, srcTensor);

// 调用方式3
auto atomCopyL1A2L0A = MakeCopy(CopyL12L0{}, LoadDataTraitDefault{});
atomCopyL1A2L0A.Call(dstTensor, srcTensor);
```
完整样例请参考[TensorAPI样例代码](../../../../../examples/01_simd_cpp_api/02_features/05_tensor_api/matmul_quant_relu/matmul_quant_relu.asc)。
