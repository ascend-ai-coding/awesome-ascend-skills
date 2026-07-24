# {算子名称}

## 模板须知

本模板用于编写开源算子README文档（gitcode仓），您可以参考本文提供的**简单模板**，也支持您基于**模板拓展**。

- 文档格式：推荐Markdown文件格式，要求语法满足markdown或Html语法规范。
- 写作规范：满足[CANN文档写作规范](https://gitcode.com/cann/community/blob/master/contributor/docs/document_writing_specs.md)、[markdownlint规范](https://github.com/DavidAnson/markdownlint/blob/main/doc/Rules.md)。
- 文档作用：阐述清楚算子功能、实现原理、参数规格以及算子调用方式等。
- 文档标题：采用算子名，样式需满足大驼峰格式。
- 文档章节：优先使用模板章节名（如功能说明、参数说明等），标题层级为##，如有特殊情况层级请按序增加。支持章节标题自定义，“可选”的章节视情况呈现，但不能与已有章节重复或矛盾。
- 文档内容：每章内容的写作目标和规范要求请参考下文详细描述。
- 文档存放路径：`operators/{operator_name}/README.md`
- 在线参考：<https://gitcode.com/cann/ops-math/blob/master/math/add/README.md。>

## 产品支持情况

> [!NOTE]
>
> **写作规范**：推荐表格形式
>
> - 列全产品支持度，支持打√不支持打×
> - 产品名称不变、顺序不变、需要增加<term>产品</term>注释对
> - 若新增产品，需所有算子全量适配
> - 产品形态介绍参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

> [!NOTE]
>
> **写作目标**：阐明算子功能、计算原理、参数规格、使用场景等。
>
> **写作规范**：推荐无序列表形式，一般包括如下维度：
>
> - 算子功能（必选）：以一句话形式简洁明了阐述功能
> - 版本演进（可选）：若算子有V2、V3...版本等，需额外提供版本差异说明，讲清新增什么功能
> - 计算公式（可选）：复杂功能可借助公式介绍实现原理或不同场景下的计算过程
> - 其他维度（可选）：支持无序列表拓展，如计算示例、流程图等

- 算子功能：完成加法计算。
- 计算公式：

  $$
  y = x1 + alpha \times x2
  $$

## 参数说明

> [!NOTE]
>
> **写作目标**：阐明算子定义的参数含义、作用、规格等信息。
>
> **写作规范**：采用表格形式，一般包括如下维度：
>
> - 参数名：解释算子定义文件(即 `op_host/{op}_def.cpp` )中的参数，参数顺序保持一致
> - 输入/输出/属性：明确参数定位，默认是必选，若为可选一般为可选输入/可选输出/可选属性
> - 描述：提供参数含义、功能、使用场景等介绍，包括与公式变量的映射关系
> - 数据类型：参数支持的data type，可不带 `DT_` 前缀
> - 数据格式：参数支持的数据排布方式，可不带 `FORMAT_` 前缀
> - 其他维度（可选）：支持表格字段扩展，如shape规格等
>
> **芯片差异**：表格里罗列所有芯片描述的并集，差异化描述在表格外以"产品1、产品2：xx描述"形式组织

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>x1</td>
    <td>输入</td>
    <td>公式中的输入x1。</td>
    <td>FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>输入</td>
    <td>公式中的输入x2。</td>
    <td>FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>alpha</td>
    <td>可选属性</td>
    <td><ul><li>alpha的描述xxx。</li><li>默认值为1.0。</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>公式中的y。</td>
    <td>FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

- <term>Atlas 训练系列产品</term>：不支持BFLOAT16。

## 约束说明

> [!NOTE]
>
> **写作目标**：阐明算子使用过程中的注意事项，例如参数组合约束、适用场景、对业务影响、算子性能或精度等。
>
> **写作规范**：
>
> - 若无约束本章写“无”；若有约束，请采用无序列表形式。
> - 算子原生语义的通用约束不写
> - 从使用场景、硬件/软件资源、对系统/网络性能或精度影响等角度说明
> - 列出该算子在昇腾硬件上实现时的特有限制

无

## 调用说明

> [!NOTE]
>
> **写作目标**：提供算子调用方法，尽量是可直接拷贝运行的示例代码，方便快速验证。
>
> **写作规范**：推荐表格形式，如果内容复杂可采用其他形式。
>
> - 调用方式（必选）：支持aclnn API、GE图模式、PyTorch API等方式调用，请提供至少一种方式。
> - 样例代码（可选）：若提供了示例代码则跳转，若无示例代码则填写-（表示不涉及）。以aclnn API方式为例，示例代码跳转到`examples/test_aclnn_add_example.cpp`。
> - 说明（必选）：不同调用方式的补充说明，例如调用场景、调用原理、编译运行指导等，请根据实际情况自定义。
>   - aclnn API：要求跳转到aclnnXxx.md接口介绍文档
>   - GE图模式：要求跳转到算子IR proto.h定义文件
>   - PyTorch API：要求跳转到torch_extension目录下的接口介绍文档
>
> **注意**：只要样例代码 `examples/cpp` 有变化，请同步修改算子README

| 调用方式   | 调用样例            | 说明                          |
|-----------|--------------------|----------------------------------------------|
| aclnn API | [test_aclnn_add_n](examples/test_aclnn_add_example.cpp) | 通过[aclnnAddExample](跳转aclnnmd)方式调用AddExample算子                    |
| GE图模式 | [test_geir_add_n](examples/test_geir_add_example.cpp)   | 通过[算子IR](跳转add_example_proto.h)构图方式调用AddExample算子                                      |
| PyTorch API | - | 通过[add_example](跳转torch_extension目录下接口介绍md)接口调用AddExample算子                    |

## 参考资源（可选）

> [!NOTE]
>
> **写作目标**：提供除算子功能、规格、调用外的其他补充介绍，如算子设计文档（Tiling/Kernel 设计）、参考文献等。
>
> **写作规范**：本章为可选，若无参考资源可不呈现本章内容；若有请采用无序列表形式。
>
> - 文档位置：放在与 aclnn md 同层级目录下，`docs/xxx算子设计文档.md`
> - 图片建议放在最外仓 `docs/zh/figures` 目录下

- [算子设计原理](./docs/{算子名称}设计文档.md) <!-- 待开发阶段补充 -->

<!-- ⚠️ 以下「质量检查清单」仅供文档写作过程中自检使用，禁止输出到最终交付文档中 -->

## 质量检查清单（仅供自检，不输出）

> 完成文档后逐项检查规范性、完整性、正确性、可读性、实用性。**此清单仅用于写作自检，不得写入最终交付的{算子名称}.md文件。**

### 格式规范性

- [ ] 大纲层级和章节标题正确
- [ ] md/HTML 语法无误（列表、表格、图片、链接、代码块、公式等）
- [ ] 检查md语法是否满足规则<https://github.com/DavidAnson/markdownlint/blob/main/doc/Rules.md>
- [ ] 检查html标签是否配对且闭合，反例：<ul>xx<ul>
- [ ] 检查写作内容元素规范<https://gitcode.com/cann/community/blob/master/contributor/docs/document_writing_specs.md>：文件命名、标题、字体样式、图片、代码块、列表、注释符号、链接、锚点、表格
- [ ]  检查标点符号：
  - [ ] 多余空格：中文与中文、中文与英文、中文与单位符号（例如MB）、数字与单位符号之间不允许有空格（除了产品名称允许有空格，正文其他描述不允许有空格）
  - [ ] 多余标点符号：检查是否出现连续出现两个或两个以上一样的标点符号
  - [ ] 顿号和逗号：中文文档多个数据类型之间用中文顿号分隔（正例：支持INT8、INT32），英文文档多个数据类型之间用英文逗号分隔（正例：It supports INT8, INT32）
- [ ] 检查md编码格式是否为utf-8，不允许出现其他格式，否则渲染不佳
- [ ] 检查产品名称是否符合规范，要求空格、大小写完全一样，不允许出现其他不在范围的产品名描述。产品名枚举值如下：
  - <term>Ascend 950PR/Ascend 950DT</term>
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>
  - <term>Atlas 200I/500 A2 推理产品</term>
  - <term>Atlas 推理系列产品</term>
  - <term>Atlas 训练系列产品</term>

### 内容完整性

- [ ] 产品支持情况：产品是否罗列完整
- [ ] 功能说明：作用、使用场景、计算公式、V版本差异、芯片差异
- [ ] 参数说明：
  - [ ] 参数是否齐全，要求与`op_host/{op}_def.cpp`定义的参数对应
  - [ ] 参数表5个字段是否齐全
- [ ] 约束说明：场景化限制、芯片差异
- [ ] 调用说明：1. 是否有调用方式 2. 表格3个字段是否齐全

### 内容正确性

- [ ] 产品支持度与软件匹配
- [ ] 功能描述、公式表达、变量、芯片差异描述与软件是否匹配
- [ ] 参数说明：
  - [ ] 参数名是否为小写+下换线形式
  - [ ] 输入/输出/属性是否与`op_host/{op}_def.cpp`定义一致
  - [ ] 数据类型是否为大写

### 内容可读性

- [ ] 语句通顺，符合中文语境
- [ ] 检查单md内容冗余性：避免标题/段落/表格/列表/语句/参数/代码段等信息重复或冗余表达
- [ ] 检查中英文语境表达是否正常
  - [ ] 1. 内容易理解、无歧义、无模糊表达
  - [ ] 2. 句式结构清晰，无语法错误（比如病句中缺少主语等）
  - [ ] 3. 内容无冗余表述，不要太冗长
  - [ ] 4. 内容无口语化表达，具有一定专业性
  - [ ] 5. 术语第一次出现，要有拼写全称和中文释义
- [ ] 无歧义、前后一致、不冗余
- [ ] 检查章节大纲是否合理
  - [ ] 1. 章节标题是否有重复或含义冲突。
  - [ ] 2. 文档内目录层级原则上不超过4层（即最多####）

### 内容实用性

- [ ] 具备操作指导性，新用户无断点
