# 二分对比方法

通过在 kernel 函数中添加检查点 tensor 作为输入参数进行原地修改，对比中间结果的精度，定位导致精度差异的具体 op。

## 核心原理

1. **检查点tensor作为输入参数**：在kernel函数定义中添加检查点tensor作为输入参数，支持原地修改
2. **shape和dtype声明**：检查点tensor需要声明shape和dtype
3. **检查点对比**：与golden对应位置的tensor进行对比
4. **二分定位**：先从单个关键计算点开始，然后继续二分进行精度对比，直到找到精度不对的op

## 工作流程

### 步骤 1：分析代码结构，确定检查点

分析kernel和golden代码，确定需要检查的关键计算节点。选择在关键计算节点之后的位置，确保检查点的结果有明确的含义，优先选择有明显边界的位置（如matmul、softmax之后）。

### 步骤 2：修改kernel函数，添加检查点tensor

在kernel函数定义中添加检查点tensor作为输入参数，每个对应一个检查点。

**重要原则**：
- 检查点tensor作为输入参数直接在 kernel 函数中声明
- 在测试函数中用torch.empty()初始化检查点tensor
- kernel函数内部不需要return，输出tensor通过out参数传入，使用out.move()或pypto.assemble写入
- 算子和golden的检查点必须完全一致，shape和dtype都要相同
- 对于循环内的变量，可以在循环外创建大的tensor，在循环内使用view, assemble赋值
- 如果检查点的变量在循环中计算的，用pypto.assemble把中间结果搬运到要输出的大tensor里面
- 对于多层循环实现很复杂的算子，且算子和golden实现一致，可以不通过搬运比较assemble后的大tensor，只比较循环内的临时变量即可（（注：pypto的最后一块数据在非对齐情况下会带有脏数据，导致assert误判，此时可选第一块数据进行对比）

**shape推导方法**：
- 从变量定义推导：找到产生该变量的赋值语句，分析等号右边的操作对shape的变换
- 从权重/输入推导：找到相关的权重tensor shape和输入tensor shape，根据matmul/view等操作规则推导
- 从循环tile推导：循环内变量的第一维 = tile_batch，向上追溯到原始输入的shape

**kernel函数声明（新前端推荐）**：
- 直接使用 `@pypto.frontend.jit()` 装饰器
- 检查点tensor作为输入参数直接在 kernel 函数中声明
- 输出tensor也作为输入参数，使用 `out.move()` 或 `pypto.assemble` 写入结果
- 不使用 return 语句返回结果

### 步骤 3：修改golden函数，增加返回值

**重要原则**：golden函数的检查点必须与kernel函数的检查点完全一致，包括检查点的数量、shape和dtype、顺序、计算逻辑相同。

**返回值数量和变量意义匹配**：
- 确保golden函数的返回值数量与kernel函数的检查点数量一致
- 确保每个返回值的意义与kernel函数中对应的检查点变量意义一致
- 如果golden函数有多个子函数调用，需要确保每个子函数的返回值数量和意义都正确匹配

**检查点对等性原则**：
- 检查点必须在计算流程中处于相同的状态才能进行对比，如果实现上有细微差别，修正对齐后才可以进行对比
- 需要仔细分析kernel和golden代码中每个中间变量的计算路径
- 如果某个变量在golden中被后续操作修改（如乘法、加法等），kernel中对应的检查点也必须是修改后的版本
- 对比的必须是同一计算节点的结果，而不是相同名称但不同计算阶段的变量

修改返回类型注解，增加检查点对应的tensor类型。

### 步骤 4：修改测试函数，对比所有结果

创建检查点tensor，执行kernel（kernel内部会原地修改checkpoint），执行golden，对比最终结果和所有检查点。
注意：如果结果相差过大，或者shape不匹配，重新检查添加的检查点代码；device输出为0，可能是卡冲突了或者检查点添加有误。

### 步骤 5：运行测试并分析结果

根据对比结果，定位到第一个精度不匹配的检查点，问题就在该检查点之前或该检查点的计算中。

### 步骤 6：二分定位精度问题

先从个别关键的计算点开始，每次添加一个检查点即可，然后继续二分进行精度对比，直到找到精度不对的op。

## 关键技巧

### 处理循环输出

当检查点在循环内时，需要使用切片或者view, assemble赋值。

### shape 对齐

确保jit和golden的输出shape一致。

### dtype 转换

如果dtype不一致，进行转换。golden使用float32，jit使用bf16，统一转换为float32对比。

### assemble变量名不能相同

使用pypto.assemble时，输入和输出变量名不能相同，否则会报错"mix assemble and common operation for same output"。使用不同的变量名解决。

### assemble的dtype一致性

pypto.assemble要求输入输出tensor的dtype一致，否则编译报错"Source dtype must be same with dst dtype!"。使用cast转换dtype解决。

### 理解检查点的shape

添加检查点时，需要准确理解检查点变量的shape。查看kernel中检查点变量的实际shape（可通过中间变量计算得到），查看golden中对应变量的shape，两者必须一致才能对比。

## 关键模式

1. 在kernel函数输入参数中添加检查点tensor（声明shape和dtype，作为输入参数）
2. 在测试函数中用torch.empty()创建检查点tensor并传递给kernel
3. 在kernel函数内部使用pypto.assemble或切片赋值保存中间结果到检查点tensor
4. kernel函数内部不需要return，输出tensor通过out参数传入，使用out.move()或pypto.assemble写入
5. 在golden函数中对应保存中间结果并返回
6. 在测试函数中对比所有检查点
7. 先从个别关键计算点开始，然后二分定位

## 检查清单

- [ ] 分析代码结构，确定检查点
- [ ] 修改kernel函数，添加检查点tensor作为输入参数（声明shape和dtype）
- [ ] 修改golden函数，增加返回值
- [ ] 修改测试函数，对比所有结果
- [ ] 运行测试并分析结果
- [ ] 二分定位精度问题，直到找到具体op

## 常见问题

### 输出参数shape不匹配

检查golden和kernel的shape是否一致，不一致检查（检查点输出拼接。

### 循环内结果拼接

使用切片赋值或者view, assemble赋值，根据代码逻辑选择合适的方法。

### dtype不一致

统一转换为float32对比。

### LSP提示"Tuple expression not allowed in type expression"

检查是否在 -> 之后声明了返回值。根据正确的实现方式，检查点tensor应该作为输入参数，而不是返回值。

### 如何在正确处理返回值

不需要return。调用kernel函数时添加中间变量在输出列表，不需要接收返回值，因为会原地修改变量。
