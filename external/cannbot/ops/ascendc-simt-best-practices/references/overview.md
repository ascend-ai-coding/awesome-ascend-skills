# SIMT C API 总览

SIMT API基于AI Core硬件能力，通过 `AscendC::Simt::VF_CALL` 启动SIMT VF子任务。每32个线程组成一个Warp，Warp内每个线程称为Lane（编号0~31）。

## 头文件包含

```cpp
#include "simt_api/asc_simt.h"       // 通用（非half/bfloat16/fp8类型）
#include "simt_api/asc_fp16.h"       // half/half2类型API
#include "simt_api/asc_bf16.h"       // bfloat16/bfloat16x2_t类型API
#include "simt_api/asc_fp8.h"        // hifloat8x2_t/float8_e4m3x2_t/float8_e5m2x2_t类型API
```

## API分类

| 类别 | 功能 | 头文件 | 索引文件 |
|------|------|--------|----------|
| 核函数定义 | 启动SIMT VF子任务 | common_functions.h | `misc/asc_vf_call.md` |
| 同步函数 | 线程同步与内存可见性 | device_sync_functions.h | `01_同步函数.md` |
| 数学函数 | 三角/指数/对数/幂等运算 | math_functions.h | `02_数学函数.md` |
| 精度转换 | 取整(rint/round/floor/ceil) | math_functions.h | `03_精度转换.md` |
| 比较函数 | 判断有限数/NaN/Inf | math_functions.h | `04_比较函数.md` |
| 原子操作 | UB/GM上的原子读写 | device_atomic_functions.h | `05_原子操作.md` |
| Warp函数 | Warp内数据交换/归约 | device_warp_functions.h | `06_Warp函数.md` |
| 类型转换 | float/half/bf16/int间转换 | device_functions.h | `07_类型转换.md` |
| 向量构造 | make_int2/float2等 | vector_functions.h | `08_向量构造函数.md` |
| Cache Hints | 带缓存提示的Load/Store | device_functions.h | `09_Cache_Hints.md` |
| 调测接口 | printf/assert/trap | asc_simt.h | `10_调测接口.md` |
| 内置宏 | 特殊值常量和数学常数 | asc_simt.h/asc_fp16.h/asc_bf16.h | `11_内置宏.md` |

---

## 核函数定义与启动

### asc_vf_call

在SIMD与SIMT混合编程场景，启动SIMT VF（Vector Function）子任务，通过参数配置，启动指定数目的线程，执行指定的SIMT核函数。

> 注意：asc_vf_call启动SIMT VF子任务时，子任务函数不能是类的成员函数，推荐使用普通函数或类静态函数，且入口函数必须使用\_\_simt\_vf\_\_修饰宏。传递的参数只支持裸指针和常见基本数据类型，不支持传递结构体、数组等。

**函数原型**:

```cpp
template <auto funcPtr, typename... Args>
__aicore__ inline void asc_vf_call(dim3 threadNums, Args &&...args)
```

**模板参数**:

| 参数名 | 描述 |
|--------|------|
| funcPtr | 用于指定SIMT入口核函数 |
| Args | 定义可变参数，用于传递实参到SIMT入口核函数 |

**参数说明**:

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| threadNums | 输入 | dim3结构{dimx,dimy,dimz}，指定SIMT线程块内线程数量。线程总数=dimx*dimy*dimz，必须<=2048 |
| args | 输入 | 可变参数，传递实参到SIMT入口核函数 |

**返回值**: 无

**需要包含的头文件**: `#include "simt_api/common_functions.h"`

**调用示例**:

```cpp
__simt_vf__ __launch_bounds__(2048) inline void SimtCompute(
    __gm__ float* dst, __gm__ float* src0, __gm__ float* src1, int count) const
{
    for(int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < count; idx += gridDim.x * blockDim.x)
    {
        dst[idx] = src0[idx] + src1[idx];
    }
}

__global__ __aicore__ void SimtComputeShell(__gm__ float* x, __gm__ float* y, __gm__ float* z, const int size)
{
    __gm__ float* dst = x;
    __gm__ float* src0 = y;
    __gm__ float* src1 = z;
    asc_vf_call<SimtCompute>(dim3{1024, 1, 1}, dst, src0, src1, size);
}
```

---

## SIMD与SIMT混合编程辅助函数

以下函数用于SIMD与SIMT混合编程场景中的辅助操作。

### GetRuntimeUBSize

获取运行时UB空间的大小，单位为byte。开发者根据UB的大小来计算循环次数等参数值。

**函数原型**: `__aicore__ inline uint32_t GetRuntimeUBSize()`

**返回值**: 运行时UB空间的大小（字节）。Ascend 950PR/Ascend 950DT架构下，SIMD与SIMT混合场景中UB大小上限为216KB，非混合场景返回固定值248KB。

**调用示例**:

```cpp
uint32_t totalLength = 126976;
uint32_t tileLength = AscendC::GetRuntimeUBSize() / sizeof(half) / 2;
uint32_t tileNum = totalLength / tileLength;
```

### BlockReduceMax / BlockReduceMin / BlockReduceSum

对每个datablock内所有元素分别求最大值、最小值、求和。这些是SIMD层面的归约指令，在SIMD与SIMT混合编程场景中使用。

- **BlockReduceMax**: 对每个datablock内所有元素求最大值
- **BlockReduceMin**: 对每个datablock内所有元素求最小值
- **BlockReduceSum**: 对每个datablock内所有元素求和（二叉树方式两两相加）

支持的数据类型: half, float

**函数原型** (mask连续模式):

```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const int32_t repeatTime, const int32_t mask,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
```

> 更多详细的参数说明（mask模式、stride语义等），请参考 AscendC SIMD API 文档。
