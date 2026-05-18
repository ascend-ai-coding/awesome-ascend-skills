# 代码审查技能文件 - 输入验证

本文档例举C++安全编码规范中输入验证相关条款, 为Ascend C 代码检视过程提供编码规范指导


# 二、C++安全编码规范 - 输入验证

输入验证涉及外部输入合法性校验、缓冲区溢出防护等关键安全问题


### 2.19 外部输入作为内存操作相关函数的复制长度时，需要校验其合法性

**【描述】**
将数据复制到容量不足以容纳该数据的内存中会导致缓冲区溢出。为了防止此类错误，必须根据目标容量的大小限制被复制的数据大小，或者必须确保目标容量大小以容纳要复制的数据。

**【错误代码示例】**
外部输入的数据不一定会直接作为内存复制长度使用，还可能会间接参与内存复制操作。
如下代码示例中，inputTable->count来自外部报文，虽然没有直接作为内存复制长度使用，而是作为for循环体的上限使用，间接参与了内存复制操作。由于没有校验其大小，可造成缓冲区溢出：

```
typedef struct {
	size_t count;
	int val[MAX_num_bERS];
} ValueTable;

ValueTable *value_table_dup(const ValueTable *input_table)
{
	ValueTable *output_table = ... // 分配内存
	...
	for (size_t i = 0; i  < input_table->count; i++) {
		output_table->val[i] = input_table->val[i];
	}
  	...
}
```

**【正确代码示例】**
如下代码示例中，对input_table->count做了校验。

```
typedef struct {
size_t count;
int val[MAX_num_bERS];
}ValueTable;

ValueTable *value_table_dup(const ValueTable *input_table)
{
	ValueTable *output_table = ... // 分配内存
	...

	/ *
	- 根据应用场景，对来自外部报文的循环长度input_table->count
	- 与output_table->val数组大小做校验，避免造成缓冲区溢出
	*/
	if (input_table->count  > sizeof(output_table->val) / sizeof(output_table->val[0]){
		return NULL;
	}

	for (size_t i = 0; i  < input_table->count; i++) {
		output_table->val[i] = input_table->val[i];
	}

	...
}
```

### 2.20 外部输入数据（如函数入参/通信消息/寄存器数据等）需要做合法性校验

**【描述】**

* 1、外部输入数据需要做合法性校验且确保校验范围正确
* 2、边界接口需要对传入的地址做合法性校验避免任意地址读写
* 3、需要对入参进行合法性校验避免数组越界
* 4、需要对地址偏移校验避免任意地址读写
* 5、外部传入指针需要判空后使用（模块内调用确认不会传入空指针下不需要判空）
* 6、外部入参参与循环、递归条件的运算，必须严格校验边界和终止条件
* 7、文件路径来自外部数据时，必须对其做合法性校验
