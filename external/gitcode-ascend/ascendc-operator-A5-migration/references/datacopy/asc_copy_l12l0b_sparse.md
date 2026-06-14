> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_datamove/asc_copy_l12l0b_sparse.md

# asc_copy_l12l0b_sparse

## 产品支持情况

| 产品 | 是否支持  |
| :----------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

用于搬运存放在L1 Buffer里的512B大小的稠密权重矩阵到L0B Buffer里，同时读取128B大小的索引矩阵用于稠密矩阵的稀疏化。

索引矩阵在一个int8_t的地址中的排布是逆序排布的，例如：索引矩阵1 2 0 1 0 2 1 0，在地址中的排布为1 0 2 1 0 1 2 0，其中1 0 2 1（对应索引矩阵前四位1 2 0 1）拼成一个int8_t数据，0 1 2 0（对应索引矩阵后四位0 2 1 0）拼成一个int8_t数据。

## 函数原型

- 高维切分搬运
  ```c++
  __aicore__ inline void asc_copy_l12l0b_sparse(__cb__ int8_t* dst, __cbuf__ int8_t* src, __cbuf__ int8_t* index, uint16_t start_index, uint8_t repeat)
  ```

- 同步搬运
  ```c++
  __aicore__ inline void asc_copy_l12l0b_sparse_sync(__cb__ int8_t* dst, __cbuf__ int8_t* src, __cbuf__ int8_t* index, uint16_t start_index, uint8_t repeat)
  ```

## 参数说明

表1 参数说明
| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的操作数在L0B Buffer的起始地址。 |
| src | 输入 | 源操作数在L1 Buffer的起始地址。 |
| index | 输入 | 索引矩阵在L1 Buffer的起始地址。 |
| start_index | 输入 | 分形矩阵ID，说明搬运起始位置为源操作数中第几个分形（0为源操作数中第1个分形矩阵）。取值范围：[0, 65535]。单位：512B。 |
| repeat | 输入 | 迭代次数，每个迭代可以处理512B数据。取值范围：[1, 255]。 |

## 返回值说明

无

## 流水类型

PIPE_MTE1

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。
- dst、src的起始地址需要32字节对齐。
- repeat为0表示不执行。
- start_index不能小于0。
- 不支持转置功能。

## 调用示例

```cpp
__cb__ int8_t dst[256];
__cbuf__ int8_t src[256];
__cbuf__ int8_t index[64];
uint16_t start_index = 1;
uint8_t repeat = 8;
asc_copy_l12l0b_sparse(dst, src, index, start_index, repeat);
```