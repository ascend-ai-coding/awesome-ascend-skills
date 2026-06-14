> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_compute/asc_mmad_sparse.md

# asc_mmad_sparse

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

完成矩阵乘加操作，传入的左矩阵A为稀疏矩阵，右矩阵B为稠密矩阵。对于矩阵A，在asc_mmad_sparse计算时完成稠密化；对于矩阵B，在计算执行前的输入数据准备时自行完成稠密化（按照下文中介绍的稠密算法进行稠密化），所以输入本接口的矩阵B为稠密矩阵。稠密矩阵B需要通过[asc_copy_l12l0b_sparse](../cube_datamove/asc_copy_l12l0b_sparse.md)载入，同时加载索引矩阵，索引矩阵在矩阵B稠密化的过程中生成，再用于矩阵A的稠密化。

## 函数原型

- 常规计算
```cpp
__aicore__ inline void asc_mmad_sparse(__cc__ int32_t* c, __ca__ int8_t* a, __cb__ int8_t* b, uint16_t m, uint16_t k, uint16_t n, uint8_t unit_flag, bool cmatrix_source, bool cmatrix_init_val)
```

- 同步计算
```cpp
__aicore__ inline void asc_mmad_sparse_sync(__cc__ int32_t* c, __ca__ int8_t* a, __cb__ int8_t* b, uint16_t m, uint16_t k, uint16_t n, uint8_t unit_flag, bool cmatrix_source, bool cmatrix_init_val)
```


## 参数说明

| 参数名       | 输入/输出 | 描述               |
| --------- | ----- | ---------------- |
| c       | 输出    | 	目的操作数，结果矩阵C，起始地址需要1024字节对齐。            |
| a       | 输入    |   源操作数，左矩阵A，起始地址需要512字节对齐。|
| b       | 输入    | 	源操作数，右矩阵B，起始地址需要512字节对齐。      |
| m       | 输入    | 	左矩阵Height，取值范围：m∈[0,4095]。      |
| k       | 输入    | 	左矩阵Width、右矩阵Height，取值范围：k∈[0,4095]。      |
| n       | 输入    | 	右矩阵Width，取值范围：n∈[0,4095]。      |
| unit_flag       | 输入    | 	unit_flag是一种Mmad指令和Fixpipe指令细粒度的并行，使能该功能后，硬件每计算完一个分形，计算结果就会被搬出，该功能不适用于L0C Buffer累加的场景。取值说明如下： <br> 0：保留值。 <br> 2：使能unit_flag，硬件执行完指令后，不会关闭unit_flag功能。<br> 3：使能unit_flag，硬件执行完指令后，会关闭unit_flag功能。<br> 使能该功能时，Mmad指令的unit_flag在最后一个分形设置为3，其余分形计算设置为2即可。     |
| cmatrix_source       | 输入    | 	配置C矩阵初始值是否来源于BiasTable Buffer（存放Bias的硬件缓存区）。取值说明如下：  <br>&bull; true：来源于BiasTable Buffer。 <br>&bull; false：来源于L0C Buffer。    |
| cmatrix_init_val       | 输入    | 	配置C矩阵初始值是否为0。取值说明如下：  <br>&bull; true：C矩阵初始值为0。 <br>&bull; false：C矩阵初始值通过cmatrix_source参数进行配置。     | 


## 返回值说明

无

## 流水类型

PIPE_MTE1

## 约束说明

- 原始稀疏矩阵B每4个元素中应保证最多2个非零元素，如果存在3个或更多非零元素，则仅使用前2个非零元素。
- 当m、k、n中的任意一个值为0时，该指令不会被执行。

## 稠密算法说明

假设原始稀疏矩阵B的每4个元素中至少有2个零，稠密化后的矩阵B是一个在每4个元素中过滤掉2个零的稠密矩阵。矩阵B稠密化的过程中生成索引矩阵，过程如下：对于稀疏矩阵B中的每4个元素，将在index矩阵中生成2个2位索引，并按照以下规则进行编码。索引必须在{0, 1, 2}范围内。
<br>&bull; 第一个索引用于指示前3个元素中第1个非零元素的相对位置。
<br>&bull; 第二个索引用于指示第2个非零元素在后3个元素中的相对位置。
<br>具体可参考下表。其中，“-” 表示算法不关心该位置上的值，因为其会被过滤。
<table border="1" style="border-collapse: collapse; width: 100%; text-align: center;">
  <!-- 第1行：不合并 -->
  <tr>
    <td style="padding: 8px; border: 1px solid #333;">示例</td>
    <td style="padding: 8px; border: 1px solid #333;">ele0</td>
    <td style="padding: 8px; border: 1px solid #333;">ele1</td>
    <td style="padding: 8px; border: 1px solid #333;">ele2</td>
    <td style="padding: 8px; border: 1px solid #333;">ele3</td>
    <td style="padding: 8px; border: 1px solid #333;">Index_a[i]</td>
    <td style="padding: 8px; border: 1px solid #333;">Index_b[i]</td>
  </tr>

  <!-- 第2-7行：第一列合并 (共6行) -->
  <tr>
    <td rowspan="6" style="padding: 8px; border: 1px solid #333; vertical-align: middle;">Two non-zero elements</td>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">X</td>
    <td style="padding: 8px; border: 1px solid #333;">Y</td>
    <td style="padding: 8px; border: 1px solid #333;">2'b10</td>
    <td style="padding: 8px; border: 1px solid #333;">2'b10</td>
  </tr>
  <tr>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">X</td>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">Y</td>
    <td style="padding: 8px; border: 1px solid #333;">2'b01</td>
    <td style="padding: 8px; border: 1px solid #333;">2'b10</td>
  </tr>
  <tr>
    <td style="padding: 8px; border: 1px solid #333;">X</td>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">Y</td>
    <td style="padding: 8px; border: 1px solid #333;">2'b00</td>
    <td style="padding: 8px; border: 1px solid #333;">2'b10</td>
  </tr>
  <tr>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">X</td>
    <td style="padding: 8px; border: 1px solid #333;">Y</td>
    <td style="padding: 8px; border: 1px solid #333;">-</td>
    <td style="padding: 8px; border: 1px solid #333;">2'b01</td>
    <td style="padding: 8px; border: 1px solid #333;">2'b01</td>
  </tr>
  <tr>
    <td style="padding: 8px; border: 1px solid #333;">X</td>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">Y</td>
    <td style="padding: 8px; border: 1px solid #333;">-</td>
    <td style="padding: 8px; border: 1px solid #333;">2'b00</td>
    <td style="padding: 8px; border: 1px solid #333;">2'b01</td>
  </tr>
  <tr>
    <td style="padding: 8px; border: 1px solid #333;">X</td>
    <td style="padding: 8px; border: 1px solid #333;">Y</td>
    <td style="padding: 8px; border: 1px solid #333;">-</td>
    <td style="padding: 8px; border: 1px solid #333;">-</td>
    <td style="padding: 8px; border: 1px solid #333;">2'b00</td>
    <td style="padding: 8px; border: 1px solid #333;">2'b00</td>
  </tr>

  <!-- 第8-11行：第一列合并 (共4行) -->
  <tr>
    <td rowspan="4" style="padding: 8px; border: 1px solid #333; vertical-align: middle;">One non-zero element</td>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">X</td>
    <td style="padding: 8px; border: 1px solid #333;">2'b00</td>
    <td style="padding: 8px; border: 1px solid #333;">2'b10</td>
  </tr>
  <tr>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">X</td>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">2'b10</td>
    <td style="padding: 8px; border: 1px solid #333;">2'b00</td>
  </tr>
  <tr>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">X</td>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">2'b01</td>
    <td style="padding: 8px; border: 1px solid #333;">2'b00</td>
  </tr>
  <tr>
    <td style="padding: 8px; border: 1px solid #333;">X</td>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">2'b00</td>
    <td style="padding: 8px; border: 1px solid #333;">2'b00</td>
  </tr>

  <!-- 第12行：不合并 -->
  <tr>
    <td style="padding: 8px; border: 1px solid #333;">All zero</td>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">0</td>
    <td style="padding: 8px; border: 1px solid #333;">2'b00</td>
    <td style="padding: 8px; border: 1px solid #333;">2'b00</td>
  </tr>
</table>
该索引矩阵用于A矩阵的稠密化，根据索引矩阵从MatrixA中的4个元素中选择2个元素参与计算，如下图所示：

![ ](../figures/asc_mmad_sparse_稠密算法说明.png)

## 调用示例

```cpp
uint32_t total_length = 128;
__ca__ int8_t a[total_length];
__cb__ int8_t b[total_length];
__cc__ int32_t c[total_length];
uint16_t m = 2;
uint16_t k = 2;
uint16_t n = 1;
uint8_t unit_flag = 2;
bool cmatrix_source = true;
bool cmatrix_init_val = true;
asc_mmad_sparse(c, a, b, m, k, n, unit_flag, cmatrix_source, cmatrix_init_val);
```
