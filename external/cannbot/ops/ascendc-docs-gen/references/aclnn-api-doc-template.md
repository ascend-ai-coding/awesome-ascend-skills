# aclnn{OperatorName}

> 本模板用于编写开源aclnn API接口文档（gitcode仓），以aclnnAdd为示例。
> 文档存放路径：`operators/{operator_name}/docs/aclnn{OperatorName}.md`
> 格式依据：[ops-transformer Wiki](https://gitcode.com/cann/ops-transformer/wiki) 的 aclnn API 文档模板。下列支持度和参数仅为格式示例，须按目标接口源码核验。

## 产品支持情况

> **写作规范：阐述接口适用的产品型号。**
>
> 1. 产品范围：当前产品形态名请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。
> 2. 写作样式：①必须无序列表，表格废弃不再使用。②每个产品支持情况单行呈现，产品名顺序不变，最新产品在最上层位置。③按“产品：支持/不支持”样式写作
> 3. 本领域API文档要求产品名需加“term”标记对，且符合html/md语法规范。

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id6 -->

## 功能说明

> **写作规范：旨在帮助理解接口功能、V版本差异、使用场景和计算流程等。**
>
> 1. 接口功能（必选）：
>    - ①先一句话整体介绍接口使用场景和具体功能。
>    - ②如果涉及多个子接口配套使用，请阐述调用场景和配合使用的过程。
>    - ③如果功能在不同产品上有差异，请分别描述功能差异点。
> 2. 版本演进（可选）：aclnnXxx有多个V版本时，必须介绍最新版本能力差异。例如：本接口兼容aclnnXxxxV2功能，在其基础上本接口新增支持xxx功能。
> 3. 计算公式/计算流程（可选）：①大部分算子均需提供公式，介绍API实现原理或计算过程。②若无法用公式呈现，可采用mermaid语法绘制流程图。
> 4. 示例（可选）：复杂算子根据实际需要，可通过示例阐述计算逻辑。
> 5. 其他维度（可选）：支持无序列表拓展，请根据实际情况自定义，比如计算示例、流程图等。
> 6. 公式要求：行内公式用`$`包裹，跨行公式用`$$`包裹。缩进2个空格，行内公式与上下文字之间必须加空行，`$` 后不能有空字符。

- 接口功能：完成任意两个张量的矩阵乘计算。
- 版本演进：本接口兼容aclnnXxxxV2功能，在其基础上新增支持xxx功能。
- 计算公式：

  - 场景1：先计算xx，再计算xxx，输出xxx。

    $$
    out_i = self_i+alpha \times other_i
    $$

  - 场景2：先计算xx，再计算xxx，输出xxx。

    $$
    out_i = self_i+alpha
    $$

## 函数原型

> **写作规范**：
>
> 1. 两段式接口介绍（必选）：阐述算子两段式接口的调用逻辑，参考样例话术。
> 2. 原型定义（必选）：代码块呈现，要求
>     - ①必须与`op_api/aclnn_xx.h`定义的参数名、参数类型、参数顺序完全一致；
>     - ②参数名需要为驼峰格式；
>     - ③代码段里末尾不带分号；
>     - ④参数单行呈现，不用添加参数注释;
>     - ⑤参数垂直缩进位置对齐；
>     - ⑥标明语言类型cpp。

每个算子分为[两段式接口](https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/two_phase_api.md)，必须先调用`aclnn{Op}GetWorkspaceSize`接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用`aclnn{Op}`接口执行计算。

**示例**：

```cpp
aclnnStatus aclnn{Op}GetWorkspaceSize(
  const aclTensor  *self,
  const aclTensor  *other,
  const aclScalar  *alpha,
  aclTensor        *out,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnn{Op}(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnn{Op}GetWorkspaceSize

- **参数说明**

  > **写作规范**（第一段接口参数说明表）：
  >
  > 1. 呈现形式：
  >    - 参数说明必须为html表格，列宽总和不超过1550px
  >    - 表格固定有8个字段：参数名、输入/输出、描述、使用说明、数据类型、数据格式、维度(shape)、非连续Tensor
  > 2. 参数表 > 参数名（必选）：填写`参数名（参数类型）`，要求中文括号，`*` 前后不带空格，参数名不能换行。参数与[函数原型](#函数原型)定义完全一致。
  > 3. 参数表 > 输入/输出（必选）：只允许填写“输入、输出、输入/输出、输入|输出、可选输出、可选输入、输入和输出”其中一种。
  > 4. 参数表 > 描述（必选）：指明参数含义+与公式变量的映射关系，不要写约束。
  > 5. 参数表 > 使用说明（必选）：
  >    - 填写单参数的简单约束：①是否支持空Tensor；②带Optioal后缀的参数，传空指针时代表的含义；③基础类型参数的取值范围或枚举值范围以及含义。
  >    - 默认值说法：针对C接口里基础类型的参数不允许出现默认值说法，例如bool、int类型等；针对C接口里指针类型的参数，若空指针有特殊含义的请明确其含义。
  >    - 复杂参数的约束可在[约束说明](#约束说明)章节按场景化维度阐述，例如xx场景下多个参数dtype需组合使用。
  > 6. 参数表 > 数据类型（必选）：
  >    - 参数值支持的数据类型范围（例如INT8、与xx数据类型保持一致），dtype范围必须属于[dtype规范](https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/data_type.md)。
  >    - 仅 `aclXxx` 类型参数需要填写（如 aclTensor/aclTensorList/aclScalar/aclIntArray等）；C自带参数类型（char\*/bool/uint64等）不需要填写
  > 7. 参数表 > 数据格式（必选）：对于Tensor类型，必须填写支持的数据格式（例如ND/NZ、与xx数据格式保持一致），format范围必须属于[format规范](https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/data_format.md)。
  > 8. 参数表 > 维度(shape)（必选）：对于Tensor类型，填写参数维度要求或具体shape形状要求（例如1-8维、[M,N,K]）。
  > 9. 参数表 > 非连续Tensor（必选）：对于Tensor类型参数需指出是否支持非连续性，支持打√，不支持打×，不涉及打-。
  > 10. 参数表格式要求：单元格不允许为空，若不涉及请用中横线“-”。
  > 11. 区分产品差异：
  >     - 参数说明表里不允许出现产品差异化描述，例如“Atlas A2系列产品：仅支持取1”。
  >     - 参数表里阐述所有产品的并集描述，差异化内容请在参数表就近下方或[约束说明](#约束说明)章节阐述，样式一般以“产品1、产品2：数据类型不支持xxx。”呈现，差异化描述尽量采用反向“不支持”的描述。
  >     - 产品差异化描述必须打产品标签，即差异化内容必须被起始标签（`<!-- npu="npu值" id序号 -->`） 和结束标签（`<!-- end id序号 -->`）包裹。

  <table style="table-layout: fixed; width: 1500px">
  <colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 250px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>self（const aclTensor*）</td>
      <td>输入</td>
      <td>表示xxx含义，对应公式中self。</td>
      <td>
      <ul>
      <li>支持空Tensor。</li>
      <li>数据类型与other的数据类型需满足数据类型推导规则（参见<a href="https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/deduction_relationship.md">互推导关系</a>）。</li>
      <li>shape需要与other满足<a href="https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/broadcast_relationship.md">broadcast关系</a>。</li>
      </ul>
      </td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other（const aclTensor*）</td>
      <td>输入</td>
      <td>表示xxx含义，对应公式中other。</td>
      <td><ul>
      <li>支持空Tensor。</li>
      <li>shape需要与self满足<a href="https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/broadcast_relationship.md">broadcast关系</a>。</li>
      </ul></td>
      <td>数据类型与self保持一致。</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alpha（const aclScalar*）</td>
      <td>输入</td>
      <td>表示xxx含义，对应公式中alpha。</td>
      <td>预留参数，当前暂不支持，传入空指针即可。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>表示xxx含义，对应公式中out。</td>
      <td><ul>
      <li>不支持空Tensor。</li>
      <li>数据类型需要是self与other推导之后可转换的数据类型（参见<a href="https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/deduction_relationship.md">互推导关系</a>）。</li>
      <li>shape需要是self与other broadcast之后的shape。</li>
      </ul></td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="910" id7 -->
  - <term>Atlas训练系列产品</term>：不支持BFLOAT16。
  <!-- end id7 -->

- **返回值**

  > **写作规范**（第一段接口返回值说明）：
  >
  > 1. aclnn返回码（必选）：话术固定，参考下方示例。
  > 2. 一阶段参数校验（必选）：一阶段接口必须进行参数异常拦截，异常场景需罗列清楚，要求如下。
  >    - 样式必须为表格，列宽总长固定为1000px
  >    - 表格要求：仅罗列该算子aclnn层独有的异常问题，通用异常（例如框架异常或所有算子通用问题无需罗列，直接查看aclnn返回码，例如ACLNN_ERR_RUNTIME_ERROR）
  >    - 表格字段名固定，要求给出返回值、错误码、描述（指出异常可能的原因，例如空指针、dtype、shape、format、取值异常等）。
  >    - 写大颗粒原因，不要写细节，提供定位方向即可。描述列里不要区分产品
  >    - 注意异常描述需与软件实现一致。

  aclnnStatus：返回状态码，具体参见[aclnn返回码](https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1000px">
  <colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>self、other、alpha、out存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>self、other或alpha的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self、other或alpha的数据类型不匹配。</td>
    </tr>
    <tr>
      <td>self或other的shape维度不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self、other的shape不满足broadcast规则。</td>
    </tr>
  </tbody>
  </table>

## aclnn{Op}

- **参数说明**

  > **写作规范**（第二段接口参数说明表）：
  >
  > - 字段固定：参数名、输入/输出、描述
  > - 表格中无内容用 `-` 填充，"参数名"列内容不可换行显示
  > - 列宽总长1000px

  <table style="table-layout: fixed; width: 1000px">
  <colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnn{Op}GetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/aclnn_return_code.md)。

## Inplace接口（如适用）

> **写作规范**：
>
> - Inplace接口的第一段接口参数说明要求同非原地算子
> - selfRef参数标记为"输入/输出"
> - Inplace接口的第二段接口参数说明要求同非原地算子

若算子存在Inplace版本（如aclnnInplace{Op}），参照上述格式分别编写 `aclnnInplace{Op}GetWorkspaceSize` 和 `aclnnInplace{Op}` 两节内容。

**与非Inplace版本的关键差异**：

- selfRef参数同时作为输入和输出
- 广播后的shape必须等于selfRef的shape

## 约束说明

> **写作规范**：
>
> - 单个参数约束就近写在参数说明表格里
> - 多个参数的组合约束写在此章节
> - 算子原生语义的通用约束不写，不要重写第一段接口参数说明表的内容，也不要语义矛盾
> - 列出该算子在昇腾硬件上实现时的特有限制
> - 若无约束填写"无"，不允许空章节
> - 产品差异化描述（可选）：按软件实际情况，区分产品描述。
>   - 按软件实现，以键值对形式写作，key表示产品名/value表示差异点。中文：key+中文冒号+value，英文：key+英文冒号+空格+value。
>   - 必须打产品标签，产品内容必须被起始标签（`<!-- npu="npu值" id序号 -->`）和结束标签（`<!-- end id序号 -->`）包裹。
>
> **必选项**：
>
> - 确定性计算必须按产品平台分述，不得使用没有平台边界的全局结论：
>   - `<term>平台A</term>、<term>平台B</term>：aclnnXxx默认确定性实现。`
>   - `<term>平台C</term>：aclnnXxx默认非确定性实现，支持通过aclrtSetSysParamOpt开启确定性。`
>   - `<term>平台D</term>：aclnnXxx默认非确定性实现，不支持通过aclrtSetSysParamOpt开启确定性。`
> - 确定性结论必须从对应平台的 Tiling/Kernel 实现分支取证；无法确认时标记“待确认”。
>
> **可选项**：
>
> - Batch一致性说明（三选一）：
>   - 默认Batch一致性实现：aclnnXxx在xxx条件/xxx场景下满足。
>   - 默认非Batch一致性实现，支持通过aclrtSetSysParamOpt开启Batch一致性：aclnnXxx在xxx条件/xxx场景下满足。
>   - 默认非Batch一致性实现，不支持通过aclrtSetSysParamOpt开启Batch一致性。
>
> **可选项**：
>
> - 输入 shape 限制、输入值域限制、输入属性限制、输入数据类型限制
> - 按产品区分描述，格式：`<term>产品1</term>、<term>产品2</term>：xxx`
> - 约束较多时可使用 `<details><summary>` 折叠展示

- 确定性计算：

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：aclnnXxx默认确定性实现。
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT系列产品</term>：aclnnXxx默认非确定性实现，支持通过aclrtSetSysParamOpt开启确定性。
  <!-- end id9 -->

## 调用示例

> **写作规范**：
>
> 1. 编译与运行（必选）：话术固定，不要轻易修改。建议跳转算子仓工程通用的[编译与运行样例](https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/compile_and_run_sample.md)（以math仓为例）。
> 2. Demo要求：
>    - 提供可直接运行的调用示例代码，不允许伪代码。
>    - 若不同产品的demo不同，请以“产品1、产品2：xx”形式呈现，中文顿号分割多个产品名。
>    - 代码段里必须附上必要的代码注释，辅助理解。
>    - 代码段必须标明语言类型（cpp）。

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/compile_and_run_sample.md)。

```cpp
// 示例代码占位，根据实际算子填写
```

## 源码生成与关键格式规范（仅供写作，不输出）

### 资料与源码核验

- 开发者提供规格、需求或设计文档时，先提取设计意图，再用源码逐项核验；不得只读取其中一类输入。
- 没有开发者资料时，直接从源码提取。
- 接口声明和参数顺序以 `op_host/op_api/aclnn_*.h/.cpp` 为准；ParamType、dtype、format 和平台注册以 `*_def.cpp`、config 为准；shape 以 `*_infershape.cpp` 为准；约束、错误码和确定性行为以 CheckParam、Tiling 和 Kernel 实现为准。
- 资料与源码冲突时，当前实现按源码描述，同时记录资料结论、源码结论、证据位置和待确认事项；不得把尚未实现的设计目标写成已经支持。

### 多接口与废弃接口

- 扫描全部 `aclnn_*.h` 中的所有 `ACLNN_API` 声明，不得按文件数量推断接口数量。
- 每个 aclnn 接口独立生成文档，并分别核验签名、参数、平台支持、约束、示例和废弃状态。
- 单个接口在某平台废弃或不支持时，只更新该接口的产品支持情况，并注明替代接口；不得扩大为整个算子不支持。
- Inplace 接口继续使用本模板的“Inplace 接口”章节，并核验输入/输出复用与 shape 约束。

### 参数表格式

- 参数名必须包含与函数原型一致的类型，例如 `query（const aclTensor*）`，保留声明中的 `const`。
- 输入/输出列只允许：输入、输出、输入/输出、输入|输出、可选输出、可选输入、输入和输出。
- Tensor 参数的使用说明使用 `<ul><li>` 包裹；非 Tensor 参数用纯文本描述；不涉及填写“-”，不得留空。
- 数据类型不带 `DT_` 前缀，多种类型使用中文顿号分隔；非 Tensor 参数按模板规则填写或使用“-”。
- 非连续 Tensor：支持标“√”，不支持标“×”，不涉及标“-”。
- 产品差异化描述放在参数表下方，产品名使用 `<term>` 标签。

### 调用示例格式

- 优先从对应 `examples/test_aclnn_*.cpp` 提取，确保接口名和参数顺序与源码一致。
- 示例包含 CHECK_RET/LOG_PRINT、初始化、Tensor 构造、两段式调用、同步取结果和资源释放的完整流程。
- 根据算子格式选择 ND 或 NCHW Tensor 构造；包含 `aclIntArray` 参数时必须展示 `aclCreateIntArray`。
- 多接口算子为每个接口提供对应示例或明确的示例链接。

---

<!-- ⚠️ 以下「质量检查清单」仅供文档写作过程中自检使用，禁止输出到最终交付文档中 -->

## 质量检查清单（仅供自检，不输出）

> 基于 aclnn 文档评分规则，完成文档后逐项检查。**此清单仅用于写作自检，不得写入最终交付的 aclnn{OperatorName}.md 文件。**

### 模板须知

无论是新贡献的算子或是版本正式发布的算子，请开发者根据实际情况判断是否提供算子的aclnn API文档。本文提供了API文档**简单模板**，也支持您**基于模板拓展**。

- 文档格式：推荐Markdown文件格式，支持使用原生或Html语法，请确保所有语法符合官方规范，整体要求满足[CANN文档写作规范](https://gitcode.com/cann/community/blob/master/contributor/docs/document_writing_specs.md)。
- 文档功能：提供算子对应的C API供调用，aclnn前缀API。本文主要阐述aclnn API功能、计算原理、两段式接口参数规格、约束说明以及调用示例等。
- 文档位置：①aclnn API总列表在docs/zh/op_api_list.md；②每个aclnn API文档分散在对应算子的docs目录下。
- 文档命名：采用API命名，一般为aclnn{算子名}md，样式需满足大驼峰格式，例如aclnnAdd.md。
- 文档标题：与文档命名一致，采用API命名为标题，例如# aclnnAdd。
- 文档章节：优先使用模板章节名（必选标题：产品支持情况、功能说明、约束说明、调用示例），标题层级为##，如有特殊情况层级请按序增加。支持章节标题自定义，但不能与已有章节重复或矛盾。
- 配套md修改：新增/废弃aclnn API，需同步刷新仓内如下文件（以zh目录为例）：
  - `docs/zh/menu_aclnn_api.md`（aclnn索引大纲文件，总入口）
  - `docs/zh/op_api_list.md`（aclnn接口简介和列表）
  - `{算子分类}/{算子目录}/README.md`（调用说明章节中的“aclnn API调用方式”）
- 文档样例：完整文档示例可参考开源仓在线文档，例如[aclnnAbs](https://gitcode.com/cann/ops-math/blob/master/math/abs/docs/aclnnAbs.md)。
- 产品异化描述：参数表里不允许区分产品，否则无法打产品标签。建议表格里罗列所有产品合集描述，差异化描述就近表下方或约束说明章节阐述。差异化样式一般以“产品1、产品2：xx描述”形式呈现，同时打上产品标签。
- 产品标签(商发文档必选)：aclnn md里有产品差异化描述的位置，必须增加产品标签（表格和有序列表不支持，请注意调整格式），标签格式：起始`<!-- npu="npu值" id序号 -->`和结束`<!-- end id序号 -->`，注意必须配对注释且空格严格一致，npu取值固定如下，id序号单md内不允许重复。
      > - 950  ->  Ascend 950PR&950DT系列产品
      > - A3   -> Atlas A3系列产品
      > - 910b  -> Atlas A2系列产品
      > - 910  -> Atlas训练系列产品
      > - 310p  -> Atlas推理系列产品
      > - 310b  -> Atlas 200I/500 A2推理产品

- 算子前置知识库学习：[aclnn 基本概念](https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/basic_concept.md)，例如数据类型、数据格式、量化模式、确定性计算、Batch一致性等

### 质量要求

1. 完成文档写作后，先检查是否满足[模板须知](#模板须知)。
2. 然后逐项检查规范性、完整性、正确性、可读性、实用性。
3. 每次检查完上述接口文档后，必须提供如下提示：新增aclnn/废弃aclnn/aclnn功能变更等场景，需同步刷新本仓 `docs/zh/menu_aclnn_api.md`（aclnn索引大纲文件）、`docs/zh/op_api_list.md`（aclnn接口列表）、`{算子分类}/{算子目录}/README.md`（调用说明章节中“aclnn API调用方式”）。

#### 格式规范性

- [ ] 检查章节大纲规范：
  - 不允许超过####四级标题
  - 章节标题是否与上述##标题命名相似，其中产品支持情况、功能说明、函数原型、参数说明、约束说明、调用示例为必选章节名，其他章节名支持自定义。
  - md里每个章节里不允许空章节，比如“## 约束说明”章节里没有文字描述。
- [ ] 检查md语法是否满足[markdownlint规则](https://github.com/DavidAnson/markdownlint/blob/main/doc/Rules.md)
- [ ] 检查html标签是否配对且闭合，反例：`<ul>xx<ul>`
- [ ] 检查是否符合[CANN社区文档写作规范](https://gitcode.com/cann/community/blob/master/contributor/docs/document_writing_specs.md)：主要包括文件命名、标题、字体样式、图片、代码块、列表、注释符号、链接、锚点、表格等元素规范
- [ ] 检查标点符号：
  - [ ] 多余空格：中文与中文、中文与英文、中文与单位符号（例如MB）、数字与单位符号之间不允许有空格（除了产品名称允许有空格，正文其他描述不允许有空格）
  - [ ] 多余标点符号：检查是否出现连续出现两个或两个以上一样的标点符号
  - [ ] 顿号和逗号：中文文档多个数据类型之间用中文顿号分隔（正例：支持INT8、INT32），英文文档多个数据类型之间用英文逗号分隔（正例：It supports INT8, INT32）
- [ ] 检查link有效性：检查相对路径（例如`../../docs/zh/context`）和绝对路径（例如`https://gitcode.com/cann/ops-nn`）是否存在，网页响应是否正常
- [ ] 检查单词拼写是否符合英文单词词汇
- [ ] 检查md编码格式是否为纯utf-8，不允许出现其他格式（包括变种utf-8-bom），否则渲染不佳
- [ ] 检查敏感词：
  - 不允许出现分裂国家的词汇或语句
  - 不允许出现暴力、犯罪、吸毒、死亡、军队等相关描述
  - 不允许使用这些敏感词：使能、HBM、图象、好象、阀值、按装、监听、监控、追踪、跟踪、超连接
  - 不允许使用夸张词汇：世界第一、全球第一、市场等一等
- [ ] 检查术语规范：
  - 检查英文术语首次出现时是否有拼写全称和中文释义（如CANN（Compute Architecture for Neural Networks，神经网络计算架构））
  - 算子领域相关术语和概念：需满足①[算子领域术语表](https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/guide/technical_appendix/concepts_and_terms/glossary.md)；②[aclnn 基本概念](https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/basic_concept.md)
- [ ] 检查代码段规范：
  - 使用```xxx```包裹的代码段，必须要有指定语言类型，例如cpp、python、bash等
  - 代码段内语法规范是否匹配指定的“语言类型”，例如python里函数定义时，参数默认值没有空格，形如def fun(a=1, b=2)
- [ ] 检查产品差异化描述是否符合规范
  - 产品差异化样式要求：一般以“产品1、产品2：xx描述”或者“产品1仅支持xxx”形式呈现，例如“Atlas A2系列产品：xx参数不支持FLOAT16类型”、“Atlas A2系列产品时，xx参数不支持FLOAT16类型”。
  - 产品差异化位置要求：
    - 不允许出现在md或html语法的表格里
    - 一般建议在[参数说明](#aclnnopgetworkspacesize)表下方或者[约束说明](#约束说明)
- [ ] 检查产品名称是否符合规范，要求空格、大小写完全一样，不允许出现其他不在以下范围的产品名。正确的产品名范围如下：
  - Ascend 950PR&950DT系列产品
  - Atlas A3系列产品
  - Atlas A2系列产品
  - Atlas 200I/500 A2推理产品
  - Atlas推理系列产品
  - Atlas训练系列产品
- [ ] 检查产品标签是否符合规范：
  - 位置规范：①不支持表格描述里打产品标签，无论是md原生语法表格还是html语法表格。②不支持有序列表（1. xx  2. xx）打产品标签
  - 格式规范：标签格式：起始（`<!-- npu="npu值" id序号 -->`）和结束（`<!-- end id序号 -->`），注意必须配对注释且空格严格一致，npu取值固定如下，id序号单md内不允许重复。

    - 950 -> Ascend 950PR&950DT系列产品
    - A3 -> Atlas A3系列产品
    - 910b -> Atlas A2系列产品
    - 910 -> Atlas训练系列产品
    - 310p -> Atlas推理系列产品
    - 310b -> Atlas 200I/500 A2推理产品

#### 内容完整性

检查必选章节的写作内容是否符合每章要求的“写作目标”、“写作规范”要求。

- [ ] 必选章节是否缺失：产品支持情况、功能说明、函数原型、参数说明、约束说明、调用示例
- [ ] 功能说明：是否缺少功能说明、使用场景、计算公式、V版本差异、产品差异等描述
- [ ] 参数说明：
  - [ ] 参数是否齐全，要求与`op_host/{op}_def.cpp`定义的参数对应
  - [ ] 一阶段参数表8个字段、二阶段参数表3个字段是否齐全
  - [ ] 是否缺少aclnn返回码跳转
  - [ ] 一阶段接口是否缺少参数异常拦截表
- [ ] 约束说明：
  - [ ] 是否缺少aclnnXxx确定性说明
  - [ ] 是否缺少aclnnXxx Batch一致性说明
- [ ] 调用示例：是否缺少直接复制可运行的代码段，不允许出现伪代码
- [ ] 检查产品标签是否漏打：扫描md里所有含“产品名”的关键字，检查是否被起始标签（<!-- npu="npu值" id序号 -->）和结束标签（<!-- end id序号 -->）包裹。

#### 内容正确性

检查必选章节的写作内容是否符合每章要求的“写作目标”、“写作规范”要求。

- [ ] 必选章节的顺序是否正确：产品支持情况、功能说明、函数原型、参数说明、约束说明、调用示例
- [ ] 产品支持度与软件是否匹配
- [ ] 功能描述、公式表达、变量、产品差异描述与软件是否匹配
- [ ] 函数原型：是否与`op_api/aclnn_xx.h`定义的参数名、参数类型、参数类型完全一致
- [ ] 参数说明：
  - [ ] 参数名是否为小写+下换线形式
  - [ ] 输入/输出/属性是否与`op_host/{op}_def.cpp`定义一致
  - [ ] 数据类型是否为大写
  - [ ] 一阶段接口的参数异常拦截表是否与软件拦截一致
- [ ] 约束说明：

#### 内容可读性

- [ ] 检查内容表达是否专业，避免口语化表达，例如"咱们"、"咋样"、"搞定"等，应保持专业性。
- [ ] 检查内容是否易理解：是否存在歧义表达（如"这个参数可以设置"未说明具体设置方式）、模糊表达（如"可能需要"未说明具体条件），避免使用模糊词语例如“比较”、“相对”。
- [ ] 检查句式结构清晰性：句子结构是否清晰，是否存在语法错误（如病句、缺少主语、缺少谓语、语序混乱、标点符号、断句不正确等）。
- [ ] 检查语义赘述且篇幅冗长，核心不明确（非文字重复，而是语义重复）：
  - 同义词堆叠（连续使用多个含义相同的词），例如道路崎岖不平---崎岖（包含不平）
  - 修饰词冗余（修饰词的含义已包含在被修饰词中），例如非常巨大---巨大（包含非常）
  - 重复主语/宾语（用代词或名词重复指代同一事物），例如FA算子，它是一种（它冗余）
  - 副词冗余（使用可省略的强调或程度副词），例如“可能也许”、“基本上算是”
  - 一句话里含义重复：例如”通过分块计算的方式进行分块计算“
  - 一段话里两句话含义重复：句子级别语义重复
  - 信息回退：说完新信息后，又回退旧信息重述：例如“它避免存储完整矩阵，也就是说它不需要把整个矩阵保存下来”
  - 解释过度：对简单概念或常识做不必要解释，例如”HBM即高带宽显存，是一种高带宽的显存“
  - 否定式重复：先说肯定再说否定，信息无增量，例如“能减少显存占用，也就是说不会占用太多显存”
- [ ] 检查单md内容冗余性：检查标题/段落/表格/列表/语句/参数/代码段等信息重复或冗余表达，例如文字重复“的的”
- [ ] 检查中英文语境表达是否正常
  - 内容易理解、无歧义、无模糊表达
  - 句式结构清晰，无语法错误（比如病句中缺少主语等）
  - 内容无冗余表述，不要太冗长
  - 内容无口语化表达，具有一定专业性
  - 术语第一次出现，要有拼写全称和中文释义
- [ ] 检查内容表述前后是否一致
  - 单md内参数名、术语、概念、变量、单位写法前后保持是否一致
  - 单md内是否有未解释的参数名或变量
- [ ] 检查写作格式和风格：要求单md内上下文的写作风格和逻辑要连贯
  - 人称与视角：不要相互矛盾，使用开发者/您/用户表示阅读者
  - 中文/英文标点符号习惯：code注释/表格/ul无序列表/ol有序列表里，语句末尾标点符号是否有，跟随大部分场景
- [ ] 检查代码段易理解：
  - 是否有注释：代码段超过10行，需要有注释
  - 注释是否通顺、语义清晰
  - 注释与代码关联性分析：检查注释是否能表达上下行code的含义，避免出现`inputTensor()`代码段解释为“输出Tensor”
  - [ ] 检查章节大纲是否合理
  - [ ] 章节标题是否有重复或含义冲突。
  - [ ] 文档内目录层级原则上不超过4层（即最多####）

#### 内容实用性

- [ ] 具备操作指导性，新用户无断点
- [ ] demo是否满足主流场景
