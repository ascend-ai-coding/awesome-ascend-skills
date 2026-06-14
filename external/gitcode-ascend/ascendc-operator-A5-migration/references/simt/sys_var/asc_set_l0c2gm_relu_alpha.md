> **原始文档路径**: asc-devkit/docs/api/context/c_api/sys_var/asc_set_l0c2gm_relu_alpha.md

# asc_set_l0c2gm_relu_alpha

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
| Ascend 950PR/Ascend 950DT |    √     |

## 功能说明

对RELU_ALPHA寄存器中的值进行设置。这是一个64bit的寄存器，存储在fixpipe或cube指令中进行Scalar ReLU时使用的alpha值。

其中各bit含义如下：

| bit范围 |                                           含义                                           | 
|:------|:--------------------------------------------------------------------------------------:|
| 31:13 | 表示ReLU_PRE中Scalar ReLU的M2值，只有Scalar ReLU时生效。硬件将以1位符号位，8位指数位和10位尾数位的格式用于计算，不能是INF/NAN。  |
| 63:45 | 表示ReLU_POST中Scalar ReLU的M2值，只有Scalar ReLU时生效。硬件将以1位符号位，8位指数位和10位尾数位的格式用于计算，不能是INF/NAN。 |

## 函数原型

```cpp
__aicore__ inline void asc_set_l0c2gm_relu_alpha(uint64_t config)
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
uint32_t pre_m2 = (0<<18)|(127<<10)|0;
uint32_t post_m2 = (0<<18)|(124<<10)|0;
uint64_t config = ((uint64_t)pre_m2<<13)|((uint64_t)post_m2<<45);
asc_set_l0c2gm_relu_alpha(config);
```