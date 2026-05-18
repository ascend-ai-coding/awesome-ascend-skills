1. # 代码审查技能文件 - 数值运算安全

本文档例举C++安全编码规范中数值运算安全相关条款, 为Ascend C 代码检视过程提供编码规范指导


# 二、C++安全编码规范 - 数值运算安全

数值运算安全涉及整数溢出、回绕、除零错误等关键安全问题


### 2.1 确保有符号整数运算不溢出

**【描述】**
有符号整数溢出是未定义的行为。出于安全考虑，对外部数据中的有符号整数值在如下场景中使用时，需要确保运算不会导致溢出：

- 指针运算的整数操作数(指针偏移值)
- 数组索引
- 变长数组的长度(及长度运算表达式)
- 内存拷贝的长度
- 内存分配函数的参数
- 循环判断条件

在精度低于int的整数类型上进行运算时，需要考虑整数提升。程序员还需要掌握整数转换规则，包括隐式转换规则，以便设计安全的算术运算。

1)加法

**【错误代码示例】**(加法)

如下代码示例中，参与加法运算的整数是外部数据，在使用前未做校验，可能出现整数溢出。

```
int num_a = ... // 来自外部数据
int num_b = ... // 来自外部数据
int sum = num_a + num_b;
...
```

**【正确代码示例】**(加法)

```
int num_a = ... // 来自外部数据
int num_b = ... // 来自外部数据
int sum = 0;
if (((num_a > 0) && (num_b > (INT_MAX - num_a))) ||
	((num_a < 0) && (num_b < (INT_MIN - num_a)))) {
  	... // 错误处理
}
sum = num_a + num_b;
...
```

2)减法

**【错误代码示例】**(减法)

如下代码示例中，参与减法运算的整数是外部数据，在使用前未做校验，可能出现整数溢出，进而造成后续的内存复制操作出现缓冲区溢出。

```
unsigned char  *content = ... // 指向报文头的指针
size_t content_size = ... // 缓冲区的总长度
int total_len = ... // 报文总长度
int skip_len = ... // 从消息中解析出来的需要忽略的数据长度
// 用total_len - skip_len 计算剩余数据长度，可能出现整数溢出
(void)memmove(content, content + skip_len, total_len - skip_len);
...
```

**【正确代码示例】**(减法)

如下代码示例中，重构为使用size_t类型的变量表示数据长度，并校验外部数据长度是否在合法范围内。

```
unsigned char *content = ... //指向报文头的指针
size_t content_size = ... // 缓冲区的总长度
size_t total_len = ... // 报文总长度
size_t skip_len = ... // 从消息中解析出来的需要忽略的数据长度
if (skip_len >= total_len || total_len > content_size) {
	... // 错误处理
}

(void)memmove(content, content + skip_len, total_len - skip_len);
...
```

3)乘法

**【错误代码示例】**(乘法)

如下代码示例中，内核代码对来自用户态的数值范围做了校验，但是由于opt是int类型，而校验条件中错误的使用了ULONG_MAX进行限制，导致整数溢出。

```
int opt = ... // 来自用户态
if ((opt < 0) || (opt > (ULONG_MAX / (60 * HZ)))) { // 错误的使用了ULONG_MAX做上限校验
	return -EINVAL;
}

... = opt * 60 * HZ; // 可能出现整数溢出
...
```

**【正确代码示例】**(乘法)

一种改进方案是将opt的类型修改为unsigned long类型，这种方案适用于修改了变量类型更符合业务逻辑的场景。

```
unsigned long opt = ... // 将类型重构为 unsigned long 类型。
if (opt > (ULONG_MAX / (60 * HZ))) {
	return -EINVAL;
}
... = opt * 60 * HZ;
...
```

另一种改进方案是将数值上限修改为INT_MAX。

```
int opt = ... // 来自用户态
if ((opt < 0) || (opt > (INT_MAX / (60 * HZ)))) { // 修改使用INT_MAX作为上限值
	return -EINVAL;
}
... = opt * 60 * HZ;
```

4)除法

**【错误代码示例】**(除法)

如下代码示例中，做除法运算前只检查了是否出现被零除的问题，缺少对数值范围的校验，可能出现整数溢出。

```
int num_a =  ... // 来自外部数据
int num_b =  ... // 来自外部数据
int result = 0;

if (num_b == 0) {
	... // 对除数为0的错误处理
}

result = num_a / num_b; // 可能出现整数溢出
...
```

**【正确代码示例】**(除法)

如下代码示例中，按照最大允许值进行校验，防止整数溢出，在编程时可根据具体业务场景做更严格的值域校验。

```
int num_a = ... // 来自外部数据

int num_b = ... // 来自外部数据

int result = 0;

// 检查除数为0及除法溢出错误

if ((num_b == 0) || ((num_a == INT_MIN) && (num_b == -1))) {

  ... // 错误处理

}

result = num_a / num_b;
...
```

5)求余数

**【错误代码示例】**(求余数)

```
int num_a = ... // 来自外部数据
int num_b = ... // 来自外部数据
int result = 0;
if (num_b == 0) {
	... // 对除数为0的错误处理
}

result = num_a % num_b; // 可能出现整数溢出
...
}
```

**【正确代码示例】**(求余数)

如下代码示例中，按照最大允许值进行校验，防止整数溢出。在编程时可根据具体业务场景做更严格的值域校验。

```
int num_a =  ... // 来自外部数据
int num_b =  ... // 来自外部数据
int result = 0;

// 检查除数为0及除法溢出错误
if ((num_b == 0)  || ((num_a == INT_MIN) && (num_b == -1))) {
	... // 错误处理
}

result = num_a % num_b;
...
}
```

6)一元减

当操作数等于有符号整数类型的最小值时，在二进制补码一元求反期间会发生溢出。

**【错误代码示例】**(一元减)

如下代码示例中，计算前未校验数值范围，可能出现整数溢出。

```
int num_a = ... // 来自外部数据
int result = -num_a; // 可能出现整数溢出
...
```

**【正确代码示例】**(一元减)

如下代码示例中，按照最大允许值进行校验，防止整数溢出。在编程时可根据具体业务场景做更严格的值域校验。

```
int num_a =  ... // 来自外部数据
int result = 0;

if (num_a == LNT_MIN) {
	... // 错误处理
}

result = -num_a;
...
```

### 2.2 确保无符号整数运算不回绕

**【描述】**

涉及无符号操作数的计算永远不会溢出，因为超出无符号整数类型表示范围的计算结果会按照（结果类型可表示的最大值 + 1）的数值取模。

这种行为更多时候被非正式地称为无符号整数回绕。

在精度低于int的整数类型上进行运算时，需要考虑整数提升。程序员还需要掌握整数转换规则，包括隐式转换规则，以便设计安全的算术运算。

出于安全考虑，对外部数据中的无符号整数值在如下场景中使用时，需要确保运算不会导致回绕：

- 指针运算的整数操作数(指针偏移值)
- 数组索引
- 变长数组的长度(及长度运算表达式)
- 内存拷贝的长度
- 内存分配函数的参数
- 循环判断条件

1)加法

**【错误代码示例】**(加法)

如下代码示例中，校验下一个子报文的长度加上已处理报文的长度是否超过了整体报文的最大长度，在校验条件中的加法运算可能会出现整数回绕，造成绕过该校验的问题。

```
size_t total_len =  ... // 报文的总长度
size_t read_len = 0 // 记录已经处理报文的长度
...
size_t pkt_len = parse_pkt_len(); // 从网络报文中解析出来的下一个子报文的长度
if (read_len + pkt_len > total_len) { // 可能出现整数回绕
	... // 错误处理
}

...
read_len += pkt_len;
...
```

**【正确代码示例】**(加法)

由于read_len变量记录的是已经处理报文的长度，必然会小于total_len，因此将代码中的加法运算修改为减法运算，导致条件绕过。

```
size_t total_len = ... // 报文的总长度
size_t read_len = 0; // 记录已经处理报文的长度
...

size_t pkt_len = parse_pkt_len(); // 来自网络报文
if (pkt_len > total_len - read_len) {
	... // 错误处理
}

...
read_len += pkt_len;
...
```

2)减法

**【错误代码示例】**(减法)

如下代码示例中，校验len合法范围的运算可能会出现整数回绕，导致条件绕过。

```
size_t len = ... // 来自用户态输入
if (SCTP_SIZE_MAX - len < sizeof(SctpAuthBytes)) { // 减法操作可能出现整数回绕
	... // 错误处理
}

... = kmalloc(sizeof(SctpAuthBytes) + len, gfp); // 可能出现整数回绕
...
```

**【正确代码示例】**(减法)

如下代码示例中，调整减法运算的位置（需要确保编译期间减法表达式的值不翻转），避免整数回绕问题。

```
size_t len = ... // 来自用户态输入
if (len > SCTP_SIZE_MAX - sizeof(SctpAuthBytes)) { // 确保编译期间减法表达式的值不翻转
	... // 错误处理
}

... = kmalloc(sizeof(SctpAuthBytes) + len, gfp);
...
```

3)乘法

**【错误代码示例】**（乘法）

如下代码示例中，使用外部数据计算申请内存长度时未校验，可能出现整数回绕。

```
size_t width =  ... // 来自外部数据
size_t hight =  ... // 来自外部数据
unsigned unsigned char  *buf = (unsigned char  *)malloc(width  * hight);
```

无符号整数回绕可能导致分配的内存不足。

**【正确代码示例】**（乘法）

如下代码是一种解决方案，校验参与乘法运算的整数数值范围，确保不会出现整数回绕。

```
size_t width =  ... // 来自外部数据
size_t hight =  ... // 来自外部数据
if (width == 0 || hight == 0) {
	... // 错误处理
}

if (width  > SIZE_MAX / hight) {
	... // 错误处理
}

unsigned char  *buf = (unsigned char  *)malloc(width  * hight);
```

**【例外】**
为正确执行程序，必要时无符号整数可能表现出模态（回绕）。建议将变量声明明确注释为支持模数行为，并且对该整数的每个操作也应明确注释为支持模数行为。

**【相关软件CWE编号】** CWE-190

### 2.3 确保除法和余数运算不会导致除以零的错误(被零除)

**【描述】**

整数的除法和取余运算的第二个操作数值为0会导致程序产生未定义的行为，因此使用时要确保整数的除法和余数运算不会导致除零错误(被零除，下同)。

1)除法

**【错误代码示例】**(除法)

有符号整数类型的除法运算如果限制不当，会导致溢出。

如下示例对有符号整数进行的除法运算做了防止溢出限制，确保不会导致溢出，但不能防止有符号操作数num_a和num_b之间的除法过程中出现除零错误：

```
int num_a_a =  ... // 来自外部数据
int num_b_b =  ... // 来自外部数据
int result = 0;

if ((num_a_a == INT_MIN) && (num_b_b == -1)) {
	... // 错误处理
}

result = num_a_a / num_b_b; // 可能出现除零错误
...
```

**【正确代码示例】**(除法)

如下代码示例中，添加num_b_b是否为0的校验，防止除零错误。

```
int num_a_a =  ... // 来自外部数据
int num_b_b =  ... // 来自外部数据
int result = 0;

if ((num_b_b == 0)  | | ((num_a_a == INT_MIN) && (num_b_b == -1))) {
	... // 错误处理
}

result = num_a_a / num_b_b;
...
```

2)取余

**【错误代码示例】**(求余数)

如下代码，同除法的错误代码示例一样，可能出现除零错误，因为许多平台以相同的指令实现求余数和除法运算。

```
int num_a_a =  ... // 来自外部数据
int num_b_b =  ... // 来自外部数据
int result = 0;

if ((num_a_a == INT_MIN) && (num_b_b == -1)) {
	... // 错误处理
}

result = num_a_a % num_b_b; // 可能出现除零错误
...
```

**【正确代码示例】**(求余数)

如下代码示例中，添加num_b_b是否为0的校验，防止除零错误。

```
int num_a_a =  ... // 来自外部数据
int num_b_b =  ... // 来自外部数据
int result = 0;

if ((num_b_b == 0)  | | ((num_a_a == INT_MIN) && (num_b_b == -1))) {
	... // 错误处理
}

result = num_a_a % num_b_b;
...
```


### 2.4 特殊数值处理

**【描述】**
必须处理NaN/Inf/-Inf/±0等边界值。nan/inf/-inf/±0等特殊值易被忽视，导致与竞品有差距。计算时存在：inf - inf = = nan; 0 * nan = nan 等情况。

**【风险】**
特殊值运算（如Inf-Inf=NaN）导致精度与竞品不一致

**【核心要求】**
对无穷值、非数值做清零/掩码处理，消除脏数据

**【正确代码示例】**
```cpp
// weightMaskBuf_作tmpBuf用，和weight无关
LocalTensor<uint8_t> maskUb = weightMaskBuf_.Get<uint8_t>(MASK_UB_SIZE);
LocalTensor<T> tmpUb = inputXYFPBuf_.Get<T>();
// +INF/-INF/NAN 场景下，+INF/-INF * 0 = NAN，消INF
Muls(tmpUb, iXFpUb, (float)(0.0), CAL_H_W_BLOCK);
pipe(pipe_barrier(PIPE_V);
// NAN eq NAN = FALSE，maskUb是NAN的mask
Compare(maskUb, tmpUb, tmpUb, CMPMODE::EQ, CAL_H_W_BLOCK);
pipe_barrier(PIPE_V);
// 对上一步mask的位置置0，即+INF/-INF/NAN 全置0
CoordinatesSelectScalar(iXFpUb, iXFpUb, maskUb, 0.0f, CAL_H_W_BLOCK);
pipe_barrier(PIPE_V);
```
