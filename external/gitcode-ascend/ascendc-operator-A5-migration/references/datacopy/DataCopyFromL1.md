> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/data_move/DataCopyFromL1.md

# DataCopyFromL1

## 产品支持情况

|产品|是否支持|
|--|:-:|
|Ascend 950PR/Ascend 950DT|√|


## 功能说明

DataCopyFromL1提供数据搬运功能，支持L1 Buffer到BiasTable Buffer和Fixpipe Buffer的数据搬运。

### 不支持坐标偏移的接口

该接口提供基础的数据搬运功能，支持直接从源操作数的起始位置开始搬运。

**使用场景**：

- 需要从源操作数的起始位置开始搬运
- 数据搬运大小完全由Tensor的Layout决定

### 支持坐标偏移的接口

该接口提供带偏移的数据搬运功能，支持通过坐标偏移参数指定源操作数上的偏移位置开始搬运。

**使用场景**：

- 需要从源数据的指定位置开始搬运
- 需要分批次搬运源操作数的不同部分

## 函数原型

- 不支持坐标偏移的接口

    ```cpp
    template <const DataCopyTrait& trait = DEFAULT_DATA_COPY_TRAIT, typename T, typename U>
    __aicore__ inline void DataCopy(const T& dst, const U& src)
    ```

- 支持坐标偏移的接口

    ```cpp
    template <const DataCopyTrait& trait = DEFAULT_DATA_COPY_TRAIT, typename T, typename U, typename Coord>
    __aicore__ inline void DataCopy(const T& dst, const U& src, const Coord& coord)
    ```


## 参数说明

**表 1**  模板参数说明

|参数名|描述|
|--|--|
|DataCopyTrait|预留参数，保持默认值即可。|
|T|目的操作数的数据类型，通过MakeTensor构造的[LocalTensor类型](../struct/tensor/LocalTensor.md)，逻辑存储位置支持BiasTable Buffer和Fixpipe Buffer，数据格式支持ND。|
|U|源操作数的数据类型，通过MakeTensor构造的[LocalTensor类型](../struct/tensor/LocalTensor.md)，逻辑存储位置支持L1 Buffer，数据格式支持ND。|
|Coord|坐标偏移的数据类型，通过MakeCoord构造的[Coord类型](../struct/coord/Coord.md)。| 


**表 2**  参数说明

|参数名|输入/输出|含义|
|--|--|--|
|dst|输出|目的操作数。|
|src|输入|源操作数。|
|coord|输入|源操作数上的偏移坐标。|

## 返回值说明

无

## 约束说明  
- 地址重叠约束：无约束。
- 参数组合约束：无约束。
- 环境影响约束：无约束。
- 特殊数据类型约束：无约束。
- 性能约束：无约束。
- 异常和边界值处理：无约束。 
- Tensor Layout相关约束：
    - Shape、Stride只支持四维，针对不同的存储位置，四个维度的配置均有不同的约束，部分维度为固定值，不可配置。详见[层次化表达法](../Layout和层次化表述法.md)。
    - Shape、Stride具体维度的数据，仅支持基础size_t和Std::Int类型。
    - 支持坐标偏移的接口中，coord需要满足地址对齐要求。

## 流水类型

PIPE_MTE1(L1 Buffer -> BiasTable Buffer)或PIPE_FIX(L1 Buffer -> Fixpipe Buffer)

## 数据通路说明

**表 3**  L1 Buffer -> BiasTable Buffer数据通路说明

|项目|内容|
|--|--|
|模板参数T|数据类型为LocalTensor|
|模板参数U|数据类型为LocalTensor|
|目的操作数Hardware要求|逻辑存储位置BiasTable Buffer|
|源操作数Hardware要求|逻辑存储位置L1|
|目的操作数的数据类型|half、bfloat16_t、float、int32_t|
|源操作数的数据类型|float、int32_t|
|源操作数和目的操作数数据类型组合要求|half、bfloat16_t、float->float, int32_t->int32_t|
|目的操作数数据对齐要求|起始地址和空间大小要求64字节对齐|
|源操作数数据对齐要求|起始地址和空间大小要求32字节对齐| 
|数据格式要求|源操作数和目的操作数数据格式均为ND| 
|搬运数据量要求|每次最多搬运4KB数据|


**表 4**  L1 Buffer -> Fixpipe Buffer数据通路说明

|项目|内容|
|--|--|
|模板参数T|数据类型为LocalTensor|
|模板参数U|数据类型为LocalTensor|
|目的操作数Hardware要求|逻辑存储位置Fixpipe Buffer|
|源操作数Hardware要求|逻辑存储位置L1|
|目的操作数的数据类型|uint64_t、LocalTensor|
|源操作数的数据类型|uint64_t、LocalTensor|
|源操作数和目的操作数数据类型要求|类型一致 |
|目的操作数数据对齐要求|起始地址和空间大小要求128字节对齐|
|源操作数数据对齐要求|起始地址和空间大小要求32字节对齐| 
|数据格式要求|源操作数和目的操作数数据格式均为ND| 
|搬运数据量要求|每次最多搬运4KB数据|

## 调用示例

```cpp
...
    using namespace AscendC::Te;

    constexpr uint32_t m = 64;
    constexpr uint32_t n = 64;
    constexpr uint32_t k = 64;
    using _1 = AscendC::Std::Int<1>;

    // 创建GM上tensor的layout
    auto layoutA = MakeLayout(MakeShape(m, k), MakeStride(k, _1{}));
    auto layoutB = MakeLayout(MakeShape(k, n), MakeStride(n, _1{}));
    auto layoutC = MakeLayout(MakeShape(m, n), MakeStride(n, _1{}));
    auto layoutQuant = MakeLayout(MakeShape(_1{}, n), MakeStride(n, _1{}));
    auto layoutBias = MakeLayout(MakeShape(_1{}, n), MakeStride(n, _1{}));

    // 创建GM上tensor的迭代器
    auto gmPtrA = MakeGMmemPtr(gm_x);
    auto gmPtrB = MakeGMmemPtr(gm_y);
    auto gmPtrC = MakeGMmemPtr(gm_z);
    auto gmPtrQuant = MakeGMmemPtr(gm_quant);
    auto gmPtrBias = MakeGMmemPtr(gm_bias);

    // 创建GM上tensor
    auto globalA = MakeTensor(gmPtrA, layoutA);
    auto globalB = MakeTensor(gmPtrB, layoutB);
    auto globalC = MakeTensor(gmPtrC, layoutC);
    auto globalQuant = MakeTensor(gmPtrQuant, layoutQuant);
    auto globalBias = MakeTensor(gmPtrBias, layoutBias);

    // 创建L1上tensor的layout
    auto l1ALayout = MakeNzLayout<half>(m, k);
    auto l1BLayout = MakeNzLayout<half>(k, n);
    auto l1QuantLayout = MakeLayout(MakeShape(_1{}, n), MakeStride(n, _1{}));
    auto l1BiasLayout = MakeLayout(MakeShape(_1{}, n), MakeStride(n, _1{}));

    // 创建L0上tensor的layout
    auto l0ALayout = MakeNzLayout<half>(m, k);
    auto l0BLayout = MakeZnLayout<half>(k, n);
    auto l0CLayout = MakeL0CLayout(m, n);

     // 创建BT上tensor的layout
    auto btLayout = MakeLayout(MakeShape(_1{}, n), MakeStride(n, _1{}));

    constexpr int32_t  L1_MAX = 128*128;
    constexpr int32_t  L0_MAX = 128*128;
    constexpr int32_t  L0C_MAX = 128*128;

    // 数组方式静态分配内存
    __cbuf__ half l1AAddr[L1_MAX];
    __cbuf__ half l1BAddr[L1_MAX];
    __cbuf__ uint64_t l1QuantAddr[128];
    __cbuf__ float l1BiasAddr[128];

    __ca__ half l0AAddr[L0_MAX];
    __cb__ half l0BAddr[L0_MAX];
    __cc__ float l0CAddr[L0C_MAX];
    __biasbuf__ float btAddr[L0_MAX];

    // 创建L1上tensor的迭代器
    auto l1APtr = MakeL1memPtr(l1AAddr);
    auto l1BPtr = MakeL1memPtr(l1BAddr);
    auto l1QuantPtr = MakeL1memPtr(l1QuantAddr);
    auto l1BiasPtr = MakeL1memPtr(l1BiasAddr);

    // 创建L0上tensor的迭代器
    auto l0APtr = MakeL0AmemPtr(l0AAddr);
    auto l0BPtr = MakeL0BmemPtr(l0BAddr);
    auto l0CPtr = MakeL0CmemPtr(l0CAddr);
    // 创建BT上tensor的迭代器
    auto biasPtr = MakeBiasmemPtr(btAddr);

    // 创建L1上tensor
    auto l1ATensor = MakeTensor(l1APtr, l1ALayout);
    auto l1BTensor = MakeTensor(l1BPtr, l1BLayout);
    auto l1QuantTensor = MakeTensor(l1QuantPtr, l1QuantLayout);
    auto l1BiasTensor = MakeTensor(l1BiasPtr, l1BiasLayout);

    // 创建L0上tensor
    auto l0ATensor = MakeTensor(l0APtr, l0ALayout);
    auto l0BTensor = MakeTensor(l0BPtr, l0BLayout);
    auto l0CTensor = MakeTensor(l0CPtr, l0CLayout);
    // 创建BT上tensor
    auto biasTensor = MakeTensor(biasPtr, btLayout);

    auto atomCopyGM2L1 = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});
    // GM->L1的数据搬运，内部调用DataCopy
    atomCopyGM2L1.Call(l1ATensor, globalA);
    atomCopyGM2L1.Call(l1BTensor, globalB);
    atomCopyGM2L1.Call(l1QuantTensor, globalQuant);
    atomCopyGM2L1.Call(l1BiasTensor, globalBias);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    auto atomCopyL12L0 = MakeCopy(CopyL12L0{}, LoadDataTraitDefault{});
    // L1->L0A的数据搬运，内部调用DataCopy
    atomCopyL12L0.Call(l0ATensor, l1ATensor);
    // L1->L0B的数据搬运，内部调用DataCopy
    atomCopyL12L0.Call(l0BTensor, l1BTensor);

    auto atomCopyL12BT = MakeCopy(CopyL12BT{}, LoadDataTraitDefault{});
    // L1->BT的数据搬运，内部调用DataCopy
    atomCopyL12BT.Call(biasTensor, l1BiasTensor);
    
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    // 执行Mad运算
    MmadParams para;
    para.m = m;
    para.n = n;
    para.k = k;
    para.cmatrixInitVal = true;
    Mmad(MmadAtom<MmadTraits<MmadOperation, MmadTraitDefault>>{}, l0CTensor, l0ATensor, l0BTensor, biasTensor, para);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);    

    // L0C->GM的数据拷贝，量化系数L1到FIXBUF的搬运内部调用DataCopy实现
    auto atomCopyL0C2GM = MakeCopy(CopyL0C2GM{}, FixpipeTraitDefault{});
    atomCopyL0C2GM.Call(globalC, l0CTensor, l1QuantTensor);
...
```
完整样例请参考[TensorAPI样例代码](../../../../../examples/01_simd_cpp_api/02_features/05_tensor_api/matmul_quant_relu/matmul_quant_relu.asc)。
