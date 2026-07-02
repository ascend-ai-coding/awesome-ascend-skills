## 调试诊断

### 调试打印

TileLang-ascend 引入了新的调试接口：T.printf 和 T.dump_tensor。目前支持 Ascend 端的全量转储功能。基本类型、指针、ub_buffer、l1_buffer、l0c_buffer 和 global_buffer 均可打印。
注意：T.printf 和 T.dump_tensor 是设备端的调试工具；对于主机端，请直接使用 Python 内置的 print 即可。

#### 1 T.printf

**接口定义**：

````
```python
def printf(format_str: str, *args)
```
````

- format_str是用于打印字符串、变量、地址等信息的格式字符串，通过格式说明符%控制转换类型，支持字符串、十进制数、十六进制数、浮点数和指针的输出。

- *args 是一个可变长度的参数列表，参数类型可以不同：根据不同的格式字符串，函数可能需要一系列额外的参数。每个参数中包含一个将被插入的值，用于替换格式参数中指定的每个 % 占位符。参数的数量应与 % 占位符的数量相匹配。
  - 格式说明符函数
    - %d/%i: 输出十进制整数
    - %f: 输出浮点数
    - %x: 输出十六进制整数（可用于输出地址信息）
    - %s: 输出字符串
    - %p: 输出指针地址（**建议直接使用%x输出地址**）

**举例**：

```
# Supports variable arguments
T.printf("fmt %s %d\n", "string", 0x123)
```

#### 2 T.dump_tensor

用于转储指定Tensor的内容，同时支持打印自定义的附加信息（仅支持uint32_t数据类型），例如打印当前行号等。

**接口定义**：

```
def dump_tensor(tensor: Buffer, desc: int, dump_size: int, shape_info: tuple=())
```

- 该张量是需要转储的张量，支持ubuffer、l1_buffer、l0c_buffer和global_buffer，这些类型无需区分，只需输入张量的名称即可。
- desc为用户自定义的附加信息（行号或其他有意义的数字）。
- dump_size 是指需要转储的元素数量。
- shape_info是输入张量的shape信息，可用于格式化打印输出。
  - 当shape size大于dump_size指定的元素个数时，按照shape_info的顺序输出元素，其中缺失的dump数据则显示为“-”。
  - 当shape size小于等于dump_size指定的元素个数时，按照shape_info的描述打印元素，超出shape维度的dump数据则不会显示。

**举例**：

```
## ub_buffer、l1_buffer、l0c_buffer、global_buffer
T.printf("A_L1:\n")
T.dump_tensor(A_L1, 111, 64) # l1_buffer

T.printf("B_L1:\n")
T.dump_tensor(B_L1, 222, 64) # l1_buffer

T.printf("C_L0C:\n")
T.dump_tensor(C_L0C, 333, 64) # l0c_buffer

T.printf("a_ub:\n")
T.dump_tensor(a_ub, 444, 64) # ub_buffer

T.printf("A_GLOBAL:\n")
T.dump_tensor(a_global, 555, 64) # global_buffer

## Using shape_info for clearer dumping

T.printf("A_L1:\n")
T.dump_tensor(A_L1, 111, 64, (8, 8)) # l1_buffer

T.printf("B_L1:\n")
T.dump_tensor(B_L1, 222, 64, (8, 9)) # l1_buffer

T.printf("C_L0C:\n")
T.dump_tensor(C_L0C, 333, 64, (8, 7)) # l0c_buffer

T.printf("a_ub:\n")
T.dump_tensor(a_ub, 444, 64, (8, 8)) # ub_buffer

T.printf("A_GLOBAL:\n")
T.dump_tensor(a_global, 555, 64, (8, 8)) # global_buffer
```

DumpTensor的打印结果会在开头自动输出高度详细的信息，包括：

- CANN software package version details
- Timestamp of the CANN software package release
- Kernel type information
- Operator details
- Memory information
- Data type
- Location information

```
输出信息示例：
opType=AddCustom, DumpHead: AIV-0, CoreType=AIV, block dim=8, total_block_num=8, block_remain_len=1046912, block_initial_space=1048576, rsv=0, magic=5aa5bccd
CANN Version: XX.XX, TimeStamp: XXXXXXXXXXXXXXXXX
DumpTensor: desc=111, addr=0, data_type=float16, position=UB, dump_size=32
```
