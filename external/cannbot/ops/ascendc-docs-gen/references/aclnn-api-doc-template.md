# aclnn{OperatorName}

> 本模板用于编写开源 aclnnAPI 接口文档（gitcode 仓），以 aclnnAdd 为示例。
> 文档存放路径：`operators/{operator_name}/docs/aclnn{OperatorName}.md`

## 产品支持情况

> **写作规范**：
> - 按照如下顺序列出所有产品，支持打 √，不支持打 ×
> - 按产品实际支持情况填写
> - 若新增产品，需所有 aclnn 全量适配
> - 产品与芯片映射关系参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)
> - aclnn 不涉及麒麟芯片

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

> **写作规范**：
> - 简要介绍接口功能，增加 V 版本之间功能对比
> - 罗列计算公式
> - 多个 aclnn 接口写在一个文档内时，需简要介绍各接口间的区别
> - Inplace 和非 Inplace 的接口区别默认已知不写
>
> **内容要素**：
> 1. 算子接口功能（必选）：讲清楚接口功能、计算逻辑、使用场景
> 2. 计算公式（可选）：复杂算子建议有公式，公式变量需给出解释
> 3. 示例（可选）：复杂算子可通过示例阐述计算逻辑
> 4. 公式要求：缩进 2 个空格，行内公式与上下文字之间必须加空行，`$` 后不能有空字符
> 5. 若功能在不同芯片上有差异，请分别描述

**示例**：

- 接口功能：完成加法计算。
- 计算公式：

  $$
  out_i = self_i+alpha \times other_i
  $$

## 函数原型

> **写作规范**：
> - 每个接口一个单独的代码块，每行一个参数，参数缩进位置对齐
> - 标明 Cpp 语言类型，不添加用于解释参数的注释信息
> - 算子分为[两段式接口](../../../docs/context/两段式接口.md)

每个算子分为两段式接口，必须先调用"aclnn{Op}GetWorkspaceSize"接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用"aclnn{Op}"接口执行计算。

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

> **写作规范**（参数说明表）：
> - 表格固定呈现 8 个字段：参数名、输入/输出、描述、使用说明、数据类型、数据格式、维度(shape)、非连续Tensor
> - 参数名格式：`参数名（参数类型）`，中文括号，`*` 前后不带空格，参数名不能换行
> - 描述：给出参数含义 + 公式变量映射，不要写约束
> - 数据类型：仅 `aclXxx` 类型参数需要填写（如 aclTensor/aclTensorList/aclScalar/aclIntArray 等）；C 自带参数类型（char\*/bool/uint64 等）不需要填写
> - 数据格式：ND、FRACTAL_NZ(NZ) 等
> - 非连续 Tensor：不涉及打 `-`，涉及的参数若支持非连续打 `√`、不支持打 `×`
> - 使用说明：空 Tensor 支持度（必选）+ Optional 后缀需显式讲传空指针场景 + 其他约束
> - 区分芯片：表里提供所有产品支持的并集，差异描述放在表下方，以"产品1、产品2：xx描述"形式组织
> - 表格中无内容用中横线 `-` 填充
> - 中文语境用中文符号，中英文混合不能有空格
> - 列宽总和不超过 1550px，"参数名"列和"错误码"列内容不可换行显示

- **参数说明**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
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
    </tr></thead>
  <tbody>
    <tr>
      <td>self（aclTensor*）</td>
      <td>输入</td>
      <td>表示xxx含义，对应公式中self。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与other的数据类型需满足数据类型推导规则（参见<a href="../../../docs/context/互推导关系.md">互推导关系</a>）。</li><li>shape需要与other满足<a href="../../../docs/context/broadcast关系.md">broadcast关系</a>。</li></ul></td>
      <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、COMPLEX128、COMPLEX64、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>other（aclTensor*）</td>
      <td>输入</td>
      <td>表示xxx含义，对应公式中other。</td>
      <td><ul><li>支持空Tensor。</li><li>shape需要与self满足<a href="../../../docs/context/broadcast关系.md">broadcast关系</a>。</li></ul></td>
      <td>数据类型与self保持一致。</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alpha（aclScalar*）</td>
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
      <td><ul><li>不支持空Tensor。</li><li>数据类型需要是self与other推导之后可转换的数据类型（参见<a href="../../../docs/context/互推导关系.md">互推导关系</a>）。</li><li>shape需要是self与other broadcast之后的shape。</li></ul></td>
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

  - <term>Atlas 训练系列产品</term>：不支持BFLOAT16。

> **写作规范**（返回值表）：
> - 列出返回值 string 名称、错误码、导致错误可能原因描述
> - 写大颗粒原因，不要写细节，提供定位方向即可
> - 校验表里原因不要区分芯片
> - 列宽总长固定为 1000px，保证"错误码"内容不换行显示

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1000px"><colgroup>
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
  </tbody></table>

## aclnn{Op}

> **写作规范**（第二段接口参数说明）：
> - 字段固定：参数名、输入/输出、描述
> - 表格中无内容用 `-` 填充，"参数名"列内容不可换行显示
> - 列宽总长 1000px

- **参数说明**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
  </thead>
  <tbody>
    <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnn{Op}GetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## Inplace 接口（如适用）

> **写作规范**：
> - Inplace 接口的第一段接口参数说明要求同非原地算子
> - selfRef 参数标记为"输入/输出"
> - Inplace 接口的第二段接口参数说明要求同非原地算子

若算子存在 Inplace 版本（如 aclnnInplace{Op}），参照上述格式分别编写 `aclnnInplace{Op}GetWorkspaceSize` 和 `aclnnInplace{Op}` 两节内容。

**与非 Inplace 版本的关键差异**：
- selfRef 参数同时作为输入和输出
- 广播后的 shape 必须等于 selfRef 的 shape

## 约束说明

> **写作规范**：
> - 单个参数约束就近写在参数说明表格里
> - 多个参数的组合约束写在此章节
> - 算子原生语义的通用约束不写，不要重写使用说明里的内容
> - 列出该算子在昇腾硬件上实现时的特有限制
> - 若无约束填写"无"，不允许空章节
>
> **必选项**：
> - 确定性说明（三选一）：
>   - aclnnXxx默认确定性实现。
>   - aclnnXxx默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。
>   - aclnnXxx默认非确定性实现，不支持通过aclrtCtxSetSysParamOpt开启确定性。
>
> **可选项**：
> - 输入 shape 限制、输入值域限制、输入属性限制、输入数据类型限制
> - 按产品区分描述，格式：`<term>产品1</term>、<term>产品2</term>：xxx`
> - 约束较多时可使用 `<details><summary>` 折叠展示

- 确定性说明：aclnnXxx默认确定性实现。

## 调用示例

> **写作规范**：
> - 编译运行话术固定，不要轻易修改
> - 若不同芯片 demo 不同，请区分芯片写作，格式：`<term>产品名</term>：`
> - 多个芯片 demo 一致时合并呈现，中文顿号分割芯片
> - 代码段必须设置语言类型 `Cpp`

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```Cpp
// 示例代码占位，根据实际算子填写
```

---

<!-- ⚠️ 以下「质量检查清单」仅供文档写作过程中自检使用，禁止输出到最终交付文档中 -->

## 质量检查清单（仅供自检，不输出）

> 基于 aclnn 文档评分规则，完成文档后逐项检查。**此清单仅用于写作自检，不得写入最终交付的 aclnn{OperatorName}.md 文件。**

### 格式规范性
- [ ] 大纲层级和章节标题正确
- [ ] md/HTML 语法无误（列表、表格、图片、链接、代码块、公式等）

### 内容完整性
- [ ] 功能说明：作用、使用场景、计算公式、V 版本差异、芯片差异
- [ ] 函数原型：接口定义无遗漏，参数/参数类型完整
- [ ] 参数说明：参数名与原型对应，8 个字段齐全，区分芯片，第 1 段参数拦截校验
- [ ] 约束说明：确定性说明、场景化限制、芯片差异
- [ ] 调用示例：区分芯片

### 内容正确性
- [ ] 产品支持度与软件匹配
- [ ] 功能描述、公式表达、变量、芯片差异描述准确
- [ ] 函数原型参数名/类型正确，代码段语言类型标记
- [ ] 参数拦截校验准确
- [ ] 确定性描述正确
- [ ] demo 可编译运行、结果正确

### 内容可读性
- [ ] 语句通顺，符合中文语境
- [ ] 无歧义、前后一致、不冗余

### 内容实用性
- [ ] 具备操作指导性，新用户无断点
- [ ] demo 满足主流场景
