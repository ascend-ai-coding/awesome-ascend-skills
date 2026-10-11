# {算子名称}

> 本模板用于编写开源算子 README 文档（gitcode 仓）。
> 文档存放路径：`operators/{operator_name}/README.md`
> 在线参考：https://gitcode.com/cann/ops-math/blob/master/math/add/README.md

## 产品支持情况

> [!NOTE]
>
> **写作规范**：推荐表格形式，罗列支持的产品型号并打 √。
> - 按产品实际支持情况填写
> - 产品顺序不要改变
> - 若新增产品，需所有算子全量适配
> - 产品形态介绍参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)
> - 仅算子支持麒麟芯片，aclnn 不涉及

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | √ |
| <term>Kirin xxx</term> | √ |

## 功能说明

> [!NOTE]
>
> **写作目标**：阐明算子功能、计算原理、参数规格、使用场景等。
>
> **写作规范**：推荐无序列表形式，一般包括如下维度：
> - 算子功能（必选）：以一句话形式简洁明了阐述功能
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
> - 参数名：解释算子定义文件中的参数，顺序保持一致（如 `op_host/{op}_def.cpp` 或 `op_graph/{op}_proto.h`）
> - 输入/输出/属性：明确参数定位，默认是必选，若为可选一般为可选输入/可选输出/可选属性
> - 描述：提供参数含义、功能、使用场景等介绍，包括与公式变量的映射关系
> - 数据类型：参数支持的 data type，可不带 `DT_` 前缀
> - 数据格式：参数支持的数据排布方式，可不带 `FORMAT_` 前缀
> - 其他维度（可选）：支持表格字段扩展，如 shape 规格等
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

## 约束说明（可选）

> [!NOTE]
>
> **写作目标**：阐明算子使用过程中的注意事项，例如参数组合约束、适用场景、对业务影响、算子性能或精度等。
>
> **写作规范**：
> - 本章为可选，若无约束可不呈现本章内容；若有请采用无序列表形式
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
> - 调用方式：支持 aclnn、图模式等方式调用算子，请提供至少一种方式
> - 样例代码：请在算子 `examples` 目录提供调用示例代码，文件命名规则为 `test_${invoke_mode}_${op_name}`
> - 说明：针对不同调用方式提供补充介绍
>
> **注意**：只要样例代码 `examples/cpp` 有变化，请同步修改算子 README

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn调用</td>
    <td><a href="../examples/{op_name}/examples/test_aclnn_{op_name}.cpp">test_aclnn_{op_name}</a></td>
    <td rowspan="2">参见<a href="./zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
  <tr>
    <td>图模式调用</td>
    <td><a href="../examples/{op_name}/examples/test_geir_{op_name}.cpp">test_geir_{op_name}</a></td>
  </tr>
</tbody></table>

## 参考资源（可选）

> [!NOTE]
>
> **写作目标**：提供除算子功能、规格、调用外的其他补充介绍，如算子设计文档（Tiling/Kernel 设计）、参考文献等。
>
> **写作规范**：本章为可选，若无参考资源可不呈现本章内容；若有请采用无序列表形式。
> - 文档位置：放在与 aclnn md 同层级目录下，`docs/xxx算子设计文档.md`
> - 图片建议放在最外仓 `docs/zh/figures` 目录下

* [算子设计原理](./docs/{算子名称}设计文档.md) <!-- 待开发阶段补充 -->
