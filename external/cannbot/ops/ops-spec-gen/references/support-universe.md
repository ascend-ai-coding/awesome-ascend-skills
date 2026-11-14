# 官方 dtype / format 全集契约

`huawei-cann-types-2026-10-15` 是用户在 2026-10-15 提供的华为文档快照标识，不代表 CANN 版本。
源文档分别保存在 [DataType](datatype.md) 和 [Format](format.md)，原始引用为
`metadef/res/docs/api/DataType_res.md` 和 `metadef/res/docs/api/Format_res.md`。
`schemas/op-spec.json` 是结构化全集的统一来源：`$defs.official_dtype.enum` 和
`$defs.official_format.enum` 分别定义完整 dtype / format 列表，枚举顺序也由此确定。
`x-support-catalog` 保存官方名称映射、来源和上下文限制；快照标识取自
`properties.support_universe.const`。生成器和校验器通过共享读取逻辑使用同一份 schema，
一致性检查核对别名映射、限制列表、兼容词汇和芯片引用，无需维护另一份全集文件。

## 全集边界

包含文档中每个具名枚举成员：42 个 DataType、57 个 Format。保留状态和边界标记，
但它们不能被声明为有效 Tensor 支持项。DataType 保留编号 5 没有具名成员，不创建虚构类型。

| 原始名称 | spec 中的名称 / 处理 |
|---|---|
| DT_FLOAT / DT_DOUBLE / DT_BF16 | float32 / float64 / bfloat16 |
| 其他 DT_* | 去前缀，改小写，例如 DT_FLOAT6_E3M2 → float6_e3m2 |
| FORMAT_* | 去前缀，保留大小写，例如 C1HWNCoC0 |
| 旧名称 NZ | 输入时归一化为 FRACTAL_NZ；全集中只计一次 |
| DT_UNDEFINED / DT_MAX | undefined / max，固定不支持 |
| FORMAT_RESERVED / ALL / NULL / END / MAX | 保留枚举成员，固定不支持 |
| uint4 / CSR / COO / BSR | 不属于本快照，不能放入新的全量契约 |

全量枚举登记不代表 numpy / oracle / NPU 已具备这些类型的计算或测试能力。
特殊类型若声明支持，还需要真实 dtype_rule、组合表、oracle 和相应运行环境，不能静默转成 float32。

## 生成和校验

新生成文件始终带 `support_universe`。输入的 `dtype_set` 来自完整需求，输出 `dtype_set`
取组合表中对应输出的投影；输入、输出都有 `supported_layouts`，默认 `[ND]`，可使用
`--supported-formats "input.x:ND,NCHW;output.y:ND"` 分别指定。输入格式变体也归入正向集合。
生成器对每个接口计算 `unsupported = universe - supported`，顺序固定为官方枚举顺序。

**标记值固定排除是支持集合的前置约束**：全集保留枚举边界/状态标记，但它们不是有效 Tensor 支持项。
`DT_UNDEFINED` / `DT_MAX` 归一化为 `undefined` / `max`，固定写入每个接口的 `unsupported_dtypes`；
`FORMAT_RESERVED` / `ALL` / `NULL` / `END` / `MAX` 归一化为 `RESERVED` / `ALL` / `NULL` / `END` / `MAX`，
固定写入每个接口的 `unsupported_layouts`。不能把它们放入支持集合，也不能从官方全集删除。
`x-support-catalog.non_tensor_dtypes` / `non_tensor_formats` 是此约束的结构化来源：
`support_catalog.partition()` 拒绝支持标记值，stage 2 使用 `support_partition.non_tensor` 报错。

完整支持范围仅为 float32 / ND 时，每个输入和输出分别写 `dtype_set: [float32]`、`supported_layouts: [ND]`；
不支持列表为其余 41 个 dtype / 56 个 format，包含上述全部标记。这仍覆盖 42 / 57 项完整全集。
解释全量生成时，既要说明互斥补集，也要明确说明标记值固定排除；补集公式本身不能替代这条支持约束。

允许用户额外传入部分排除断言，但这些只是断言；最终输出的排除列表仍是完整补集。
互斥、并集覆盖、无全集外值均需成立。删除任何排除条目都会造成覆盖失败。
`layout` 必须在支持格式列表中，组合表的 dtype 必须在对应支持列表中，输出支持 dtype
必须恰好等于其组合表投影。

支持列表是本次生成契约的穷尽定义，不是“官方文档只举例提到了这些值”。确认支持范围后
才能交付；不能凭全量枚举清单推断一个实际算子支持哪些类型。

## 条件和上下文

文档列出的八种低精度浮点 dtype 和五种 FRACTAL_NZ_C0_* 格式仅支持 Ascend950PR / Ascend950DT；
全量契约声明这些值为支持项时，必须显式给出相应芯片范围。芯片范围不会缩小官方全集，
其他芯片不可用的值仍保留在对应契约的不支持列表中。

Format 文档的“IR 构图不支持输入”限制不是所有 kernel 输入/输出的限制。结构化表保留
这一上下文清单，Agent 在确实生成 IR 构图输入契约时应结合它确认完整支持范围。
仅有边际支持列表不能证明所有 dtype × format 或多输入组合均合法；条件限制仍需另行描述。

## 旧规格迁移

不带 `support_universe` 的文件继续采用原有部分排除语义；缺失值不自动视为不支持。
迁移时先确认完整输入 dtype、输出组合表和每个接口的物理格式，归一化别名，然后计算
各自补集并添加快照标识。不能只加版本字段，保留旧的部分排除列表。
Python 调用方需要给 `GenInput.support_universe` 设置上述快照标识才能启用全量生成；
保留空值仅用于兼容历史调用方。CLI 和交互式入口始终生成新契约。
