> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_set_l0c2gm_quant_post.md

# asc_set_l0c2gm_quant_post

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
| Ascend 950PR/Ascend 950DT |    √     |

## 功能说明

对QUANT_POST寄存器中的值进行设置, QUANT_POST是一个64bit的寄存器，存储Fixpipe后处理阶段量化信息。
其中各bit含义如下，其中各种量化模式的说明可见[asc_copy_l0c2l1](../cube_datamove/asc_copy_l0c2l1/asc_copy_l0c2l1_arch_3510.md)中对各种量化模式的说明：

| bit范围 |                                                           含义                                                           |
| :-----------|:----------------------------------------------------------------------------------------------------------------------:|
| 4:0 |                                在QS162S4_POST，QF162S4_POST和SHIFT2S4_POST量化模式中表示为s5中的偏移量。                                |
| 8:0 | 在QS162B8_POST，QF162B8_POST和SHIFT2B8_POST量化模式中表示s9中的偏移量，或在QS162S16_POST，QF162S16_POST和SHIFT2S16_POST量化模式中表示s17偏移量中的低9位。 |
| 9 |                             在QS162B8_POST，QF162B8_POST和SHIFT2B8_POST量化模式中用于表示量化结果是否为有符号的。                              |
| 31:13 |         在QS162B8_POST，QF162B8_POST，QS162S4_POST，QF162S4_POST，QS162S16_POST，QF162S16_POST量化模式中表示M3。不能为INF/NAN。          |
| 39:32 |                           在QS162S16_POST，QF162S16_POST，SHIFT2S16_POST和SHIFT2S32_POST量化模式中表示s17偏移量中的高8位. |
| 44:40 | 在SHIFT2S4_POST，SHIFT2B8_POST，SHIFT2S16_POST和SHIFT2S32_POST量化模式中表示位移值。 |


## 函数原型

```cpp
__aicore__ inline void asc_set_l0c2gm_quant_post(uint64_t config)
```

## 参数说明

|参数名|输入/输出| 描述        |
| :------ | :---  |:----------|
|config   |输入   | 待设置的寄存器值。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例
```cpp
const uint64_t MASK_8_0 = 0x1FF;
const uint64_t MASK_BIT9 = 0x200;
const uint64_t MASK_31_13 = 0xFFFFE000;

uint64_t field_8_0 = 0x1AB;
uint64_t bit9 =1;
uint64_t field_31_13 = 0x000EF;

uint64_t config = 0;

config = (config & ~MASK_8_0) | (field_8_0 & MASK_8_0);
config = (config & ~MASK_BIT9) | ((bit_9<<9) & MASK_BIT9);
config = (config & ~MASK_31_13) | (field_31_13 & MASK_31_13);
asc_set_l0c2gm_quant_post(config);
```