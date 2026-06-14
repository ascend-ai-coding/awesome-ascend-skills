> **原始文档路径**: asc-devkit/docs/api/context/c_api/cache_ctrl/asc_dcci.md

# asc_dcci

## 产品支持情况

| 产品 | 是否支持  |
| :-----------| :------: |
|<cann-filter npu_type = "950"> Ascend 950PR/Ascend 950DT | √</cann-filter> |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

在AI Core内部，Scalar单元和DMA单元都可能对Global Memory进行访问。

![](../figures/dcci.png "DataCache内存层次示意图") \\

如上图所示：

DMA搬运单元读写Global Memory，数据通过asc_copy_ub2gm[](../vector_datamove/asc_copy_ub2gm.md)和asc_copy_gm2ub[](../vector_datamove/asc_copy_gm2ub.md)等接口在UB等Local Memory和Global Memory间交互，没有Cache一致性问题；
Scalar单元访问Global Memory，首先会访问每个核内的Data Cache，因此存在Data Cache与Global Memory的Cache一致性问题。
该接口用来刷新Cache，保证Cache的一致性，使用场景如下：

读取Global Memory的数据，但该数据可能在外部被其余核修改，此时需要使用asc_dcci接口，直接访问Global Memory，获取最新数据；
用户通过Scalar单元写Global Memory的数据，希望立刻写出，也需要使用asc_dcci接口。

Scalar单元访问UB数据时，该接口需配合[asc_set_ctrl()](../sys_var/asc_set_ctrl.md)接口使用，将CTRL[49]设置为1'b1，开启datacache模式。

<cann-filter npu_type = "950">

针对Ascend 950PR/Ascend 950DT，不支持asc_dcci_entire_ub接口。

</cann-filter>

## 函数原型

   ```cpp
   // 刷新传入地址所在的单个Cache Line
   __aicore__ inline void asc_dcci_single(__gm__ void* dst)

   // 刷新传入的UB地址空间地址所在的单个Cache Line
   __aicore__ inline void asc_ub_dcci_single(__ubuf__ void* dst)

   // 刷新所有Cache Line
   __aicore__ inline void asc_dcci_entire_all()
   __aicore__ inline void asc_dcci_entire_ub()
   __aicore__ inline void asc_dcci_entire_out()
   __aicore__ inline void asc_dcci_entire_atomic()
   ```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输入 | 需要刷新缓存的向量的起始地址。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
// 示例1：SINGLE_CACHE_LINE 模式，假设mmAddr_为0x40（64B对齐）
// total_length 指参与计算的数据长度
constexpr uint64_t total_length = 32;
__gm__ int16_t src[total_length];
for( int i = 0; i < total_length; i++) {
   src[i] = i;
}
// 由于首地址64B对齐，调用asc_dcci指令后，会立刻刷新前32个数
asc_dcci_single_out(reinterpret_cast<__gm__ uint64_t*>(src))
```
