# DynamicShape 值依赖机制原理

> 本文是 DynamicShape 范式的原理层，回答三个问题：
> 1. **怎么分析**——给定算子规格（语义 + 输入输出表），如何判定哪些输入是值依赖（§2）；
> 2. **为什么**——值依赖机制的数据驻留模型与不声明契约的失败根因（§3~§5）；
> 3. **怎么表示**——某输入判定为值依赖后，在昇腾算子代码各层的完整表示（§6）。
>
> 本文是**自包含知识**：规格 + 本文档即可完成全部分析与代码表示，不依赖、也不应查阅任何算子
> 库/存量实现（见 §12）。文中只描述昇腾算子框架（CANN exe_graph / register 接口）的通用机制，
> 不绑定任何特定代码仓；涉及同名工具函数的多版本差异时，给出按当前工程环境自适应的选择规则。
> 场景路由见 [patterns.md](patterns.md)。

---

## 1 为什么叫 DynamicShape

多数算子的输出 shape 只由输入的 **shape/dtype** 决定，编译期即可推出。另一类算子的输出 shape 或
tiling 决策由某个输入的**数值**决定——`Reshape(x, shape_tensor)` 的输出形状是 `shape_tensor`
里的值，`MirrorPad(x, paddings)` 的输出形状是 `paddings` 里的值。这些值只在运行期才确定（甚至只在
device 上存在），输出 shape 因此是"动态"的。使这类算子在框架里跑得通的机制就是**值依赖**：
把 device 上 / 运行期才有的值，在 host 侧 infershape/tiling 执行前搬到 host 可读。

## 2 值依赖分析方法（怎么分析）

**判定问题**：对每个输入回答一个问题——*host 侧（infershape / tiling / 框架公共 tiling 逻辑）
是否需要读取该输入的**数值**（而不止 shape/dtype）？* 是 → 值依赖输入；否 → 非值依赖。

### 2.1 语义判定（依据算子规格）

**核心思想实验——"变值不变形"**：固定该输入的 shape/dtype 不变，仅改变它的元素值，问：

| 问题 | 变化点 | 是 → |
|------|--------|------|
| 输出的 **shape**（维数/维长）会变吗？ | 推 shape 的量 | 值依赖（infershape 读值） |
| host 侧**切分/分支决策**会变吗？（tiling 参数、有效计算量、mask 窗口） | 推 tiling 的量 | 值依赖（tiling 读值） |
| 只有 kernel 的**计算结果**会变吗？ | 逐元素数据 | **非**值依赖（普通数据张量） |

前两问任一为"是"即值依赖。示例：
- `MirrorPad(x, paddings)`：paddings 的值决定输出各维长度 → infershape+tiling 都读 → 值依赖
- `Trilu(x, k, upper)`：k 的值决定三角带位置 → tiling 按三角带定有效数据量 → tiling 读 → 值依赖
- `Add(x, y)`：x/y 的值只改输出元素值，输出 shape 与切分不变 → 非值依赖

**读值侧的确定（第二问的细化）**：tiling 需要的量有两条获取路线，属**设计选择**，两条都正确：

- 路线一（直读值）：tiling 直接解引用该输入的值，或 tiling 复用公共 tiling 模板 / auto-tiling
  （值由框架代码消费，算子代码看不到读值点也计入）→ tiling 是读值侧，须声明
- 路线二（读输出 shape）：该量已体现在输出 shape 中 → 运行期执行链（D2H → 重跑 infershape 补齐
  -1 → 跑 tiling，§7）保证 tiling 执行时 `TilingContext::GetOutputShape(idx)`（返回的
  StorageShape 含运行时 shape）已拿到真值 → tiling **不是**读值侧，免声明（省一次 D2H 同步）

**路线按输入逐个选择**：同一算子的多个值依赖输入可以各走各的路线——体现在输出 shape 中的量走
路线二，不在输出 shape 中的量走路线一（如切片语义的算子：size/目标 shape 的值即输出各维长 →
路线二；offsets/begin 只决定源窗口位置、不在输出 shape → 路线一，tiling 侧只声明后者）。

同名算子在不同工程里可能选不同路线（Reshape 选 infershape-only，MirrorPad 选双侧），判定时以
当前规格/设计意图为准；无论选哪条，**声明集合必须与实际读值集合一致**。

**参数角色分类（快速佐证，不能替代思想实验）**：

| 角色 | 典型名 | 值依赖倾向 |
|------|--------|-----------|
| 数据张量（kernel 逐元素消费） | x、weight、grad | ❌ |
| 轴/维度描述 | axis、axes、dim | ✅ |
| 目标 shape/大小描述 | shape、size、dims、crop_size、output_shape | ✅ |
| 窗口/padding/切片参数 | paddings、ksize、strides、begin/end、offsets | ✅ |
| 数量/长度参数 | depth、num_segments、repeats、minlength | ✅ |
| 只被 kernel 消费的控制量 | mask（kernel 内判断）、epsilon | 视 host 是否读：一般 ❌ |
| mask/权重被 tiling 消费 | mask（tiling 定有效行数）、alpha/beta | ✅ |

**属性 vs 张量输入的表示决策**（分析阶段就要定）：该参数在语义上是否**允许是运行期张量**？
- 允许 → 必须建为算子的**张量 Input** + 值依赖契约（属性是编译期常量，接不住运行期值）
- 恒为编译期常量且框架生态要求属性形态 → 可建为属性，但若 L0/L2 接口已是张量形态，仍按张量输入
  + `ValueDepend(REQUIRED)` 表示，保持接口统一（SliceV2 的 offsets/size 即此取向）

### 2.2 分析结论的四元组

每个值依赖输入的分析结论记为一个四元组，后续表示完全由它决定：

```
(输入名, 输入索引, 档位 OPTIONAL/REQUIRED, 读值侧 {infershape, tiling, 两侧})
```

档位判定见 §5 选择依据：infershape 读值的看"输出能否回退 -1/-2"；仅 tiling 读值的看"参数是否
需要支持运行期张量"（不需要则可收紧 REQUIRED）。

## 3 数据驻留模型

| 张量类别 | 数据位置 | host 侧何时可读 |
|---------|---------|----------------|
| 图编译期常量（const tensor） | host 常量区 | infershape/tiling 天然可读（`IsConstTensor` 为真） |
| 运行期张量（非常量） | NPU device 内存 | **默认不可读**；声明值依赖后，框架在执行 infershape/tiling 前做 D2H 拷贝 |
| 主数据张量 input0 | device | 永远不读它的值（host 只读 shape/dtype） |

- infershape / tiling 跑在 **host**；算子数据跑在 **device**。框架默认只把 shape/dtype 传给
  infershape/tiling，**不**传数据
- "input0 不需要值依赖"的本质：input0 按惯例是主数据张量，host 侧逻辑从不解引用它的值。值依赖
  的判定依据是"host 是否读值"，与输入下标无关（`AvgPool3DGrad` 的 orig_input_shape 就在下标 0，
  `StatelessUniform` 的 shape 参数也在下标 0）

## 4 失败模式（不声明契约的后果）

| 路径 | 未声明依赖时读值 | 症状 |
|------|----------------|------|
| gert（`IMPL_OP_INFERSHAPE`/`IMPL_OP_OPTILING` 注册，新写算子一律此路径） | `GetInputTensor(idx)` 返回的 tensor 数据地址为 **nullptr**（SDK `exe_graph/runtime/tiling_context.h`、`infer_shape_context.h` 接口注释明文） | 判空缺失时空指针解引用崩溃；判空走错分支 |

症状的共同根因：**host 侧访问了不在 host 的数据**。修复方式不是加判空兜底，而是补全值依赖契约
（见 §6 的作用域边界）。（另有历史的 FE/`ge::Operator` 路径，未声明读值可能直接解引用 device
地址段错误——仅作机制解读，新写算子不涉及。）

## 5 OPTIONAL vs REQUIRED 确切语义

**分层事实（先记住这条，再谈契约）**：

- **编译期常量输入**：数据本来就在 host 常量区，infershape/tiling **天然可读**（`IsConstTensor`
  为真），不声明 DataDependency 也能读到——存量代码里存在大量"tiling 读值但不声明"的算子，
  正是只按常量路径工作的实例。**原则：机制逻辑是规范——host 读值就应配齐契约（def ValueDepend +
  读值侧 DataDependency）；存量中不符合该逻辑的写法属历史遗留（只在常量路径成立、动态路径拿
  nullptr，或以判空兜底掩盖），理解成因即可，不作为范式依据、不效仿**
- **DataDependency 声明的真实作用域**：让**运行期（非常量）输入**在该侧执行前完成 D2H 拷贝。
  声明不是为了"能读到常量"，而是为了"非常量时也能读到"

| 档位 | 图层语义 | 配套要求 |
|------|---------|---------|
| `ValueDepend(OPTIONAL)` | 该输入可以是图编译期常量，**也可以是运行期张量**。运行期场景由框架在 host 侧推导执行前做 D2H 同步 | 实际读值侧必须声明 DataDependency（否则非常量路径拿不到数据）；infershape 必须写 `IsConstTensor` 非常量回退分支 |
| `ValueDepend(REQUIRED)` | **强制该输入为常量**。非常量输入在构图/编译期即报错，不留运行期路径 | 常量数据天然可读，无需 D2H；代码可直接读值，无需回退分支 |

选择依据按读值侧分两种情况：

- **infershape 读值**（输出 shape 依赖该值）：值未知时输出能否先回退未知维（-1/-2）等运行期补齐
  ——能则 OPTIONAL（`MirrorPad` paddings、`ReduceSum` axes、`Reshape` shape）；不能（值未知则
  连输出维数语义都不成立，或生态强制常量）则 REQUIRED（`SliceV2` offsets/size、`Col2im`
  output_size）
- **仅 tiling 读值**（输出 shape 不依赖该值）：OPTIONAL 机械上总是可行（运行期 D2H 先于 tiling，
  tiling 拿真值出切分方案）。若该参数在生态上**恒为构图期常量**、无需支持运行期张量，可收紧为
  REQUIRED——构图期对非常量直接报错，tiling 代码直接读值且**免**读值侧 DataDependency 声明
  （强制常量天然可读）。两条路线都正确：选 OPTIONAL 保动态能力（多一次运行期 D2H），选
  REQUIRED 换实现简洁（`PasteSubImg` 的贴图坐标即 REQUIRED 取向，`Trilu` 的 k 即 OPTIONAL 取向）。
  取舍依据是**当前工程/规格是否要求该参数支持运行期张量**——要求则 OPTIONAL（REQUIRED 会构图期
  拒绝工程要求支持的输入）；确认恒为构图期常量才可收紧 REQUIRED。**工程级"面向动态 shape"与
  算子级"生态恒常量"约束冲突时，以算子级规格为准**（工程级是默认基线，算子级是显式例外）

**新写算子的建议**：即使当前图里该输入恒为常量，只要语义上允许动态输入，就按 OPTIONAL + 读值侧
DataDependency 声明齐全（动态 shape 是主线场景）。

## 6 值依赖的代码表示（input_x 判定为值依赖后怎么写）

设分析结论为：输入 `p_x`（索引 `i`，档位 OPT/REQ，读值侧 S）。它在昇腾算子各层的**完整表示**：

### 6.1 表示总览（改哪里、不改哪里）

| 层 | 是否改 | 表示 |
|----|--------|------|
| `*_def.cpp` 输入定义 | ✅ | 张量 Input + `.ValueDepend(OPTIONAL/REQUIRED)` |
| `IMPL_OP_INFERSHAPE` 注册块（infershape ∈ S） | ✅ | `.InputsDataDependency({i})` |
| `IMPL_OP_OPTILING` 注册块（tiling ∈ S） | ✅ | `.TilingInputsDataDependency({i})` |
| infershape 实现（infershape ∈ S） | ✅ | `GetInputTensor(i)` + `IsConstTensor` 分支 + 未知维回退 |
| tiling 实现（tiling ∈ S） | ✅ | `GetInputTensor(i)` + `GetData<T>()` 参与切分决策 |
| **kernel（Ascend C）** | ❌ **不改** | 对 kernel 而言它就是普通 tensor 输入，照常 GM 读 |
| L2/aclnn（若该参数在 L2 是宿主类型） | ✅ | 进 L0 前 `ConvertToTensor`（见 §9） |
| `OpAICoreConfig`（按需） | ⚠️ | OPTIONAL + (-1/-2) 回退时开 `DynamicShapeSupportFlag` / `DynamicRankSupportFlag`（§11） |

**表示决策表（四元组 → 代码动作）**：

| 分析结论 | 代码动作 |
|---------|---------|
| infershape 读值 | def `.ValueDepend` + `.InputsDataDependency({i})` + IsConstTensor 回退实现（模板 A.1） |
| tiling 读值（直读值路线） | def `.ValueDepend` + `.TilingInputsDataDependency({i})` + tiling 读值实现（模板 A.2） |
| tiling 读输出 shape 路线 | tiling 侧**不声明**、不解引用输入值；量从 `GetOutputShape` 取（§2.1 读值侧路线二） |
| 两侧读值 | 两侧注册块都声明，两侧实现都读 |
| 仅 kernel 读值 | 无值依赖表示，按主范式处理 |
| REQUIRED 档 | 只需 def `.ValueDepend(REQUIRED)`；读值侧**免** DataDependency 声明（强制常量天然可读），代码直接读值、无需回退分支 |
| 值依赖输入同时是可选输入（ParamType(OPTIONAL)） | 读值用 `GetOptionalInputTensor(i)`，返回 nullptr 表示未提供 → 按默认值语义处理（不得当错误）；依赖声明照常按索引挂 |
| L2 宿主参数（aclIntArray*/aclScalar*） | L2 实现内 `ConvertToTensor` 后再进 L0 |

### 6.2 def 层表示

```cpp
// 该参数必须是张量 Input（属性接不住运行期值），ValueDepend 标记图层语义
this->Input("paddings").ParamType(REQUIRED)
    .DataType({ge::DT_INT32, ge::DT_INT64}).Format({ge::FORMAT_ND})
    .ValueDepend(OPTIONAL);   // 或 REQUIRED：强制常量，非常量构图期报错
```

### 6.3 infershape 侧表示（读值 + 回退）

```cpp
static ge::graphStatus InferShapeForXxx(gert::InferShapeContext* context) {
  const gert::Shape* x_shape = context->GetInputShape(0);
  gert::Shape* y_shape = context->GetOutputShape(0);
  const gert::Tensor* p_tensor = context->GetInputTensor(i);   // 值依赖输入
  OPS_CHECK_NULL_WITH_CONTEXT(context, p_tensor);

  if (IsUnknownRank(x_shape)) {
    return SetUnknownRank(y_shape);
  }
  if (IsConstTensor(p_tensor)) {          // 常量：直接读值推 shape
    switch (p_tensor->GetDataType()) {    // 值依赖输入按 dtype 模板化，覆盖 INT32/INT64
      case ge::DT_INT32:
        return InferShapeWithTensor<int32_t>(context, x_shape, p_tensor, y_shape);
      case ge::DT_INT64:
        return InferShapeWithTensor<int64_t>(context, x_shape, p_tensor, y_shape);
      default:
        return ge::GRAPH_FAILED;
    }
  }
  return SetAllUnknownDim(x_shape->GetDimNum(), y_shape);  // 非常量：回退 -1，运行期补齐
}

IMPL_OP_INFERSHAPE(Xxx).InferShape(InferShapeForXxx).InputsDataDependency({i});
```

### 6.4 tiling 侧表示

```cpp
static ge::graphStatus Tiling4Xxx(gert::TilingContext* context) {
  const gert::Tensor* p_tensor = context->GetInputTensor(i);   // 值依赖输入
  OPS_CHECK_NULL_WITH_CONTEXT(context, p_tensor);
  const int32_t* p_value = p_tensor->GetData<int32_t>();     // 已声明依赖，host 可读
  ...
}

IMPL_OP_OPTILING(Xxx).Tiling(Tiling4Xxx).TilingInputsDataDependency({i});
```

- 可选输入改用 `context->GetOptionalInputTensor(i)`：返回 nullptr = 未提供，按默认值处理（如
  Trilu 的 k 未传 → k=0），判空分支走默认值而非报错
- tiling 侧声明了依赖即保证 host 可读（常量天然可读、非常量已 D2H），**无需**再写 IsConstTensor
  回退——回退分支是 infershape（编译期首跑可能拿到非常量）的要求
- 若 Tiling 函数只是**转发进公共 tiling 模板**（如 `DoTilingImpl` / TilingRegistry 模板库），
  读值发生在模板内部——依赖声明仍挂在**算子自身**的 `IMPL_OP_OPTILING` 注册块上
- 共用 tiling 文件服务多个算子时（一个文件注册多个 `IMPL_OP_OPTILING`），依赖契约**按算子分别
  声明**——契约的主体是算子，不是文件

注册形态约束：挂依赖的注册块必须能链式调用 `InputsDataDependency` / `TilingInputsDataDependency`
（SDK `register/op_impl_registry.h` 的 `OpImplRegisterV2` 接口，`IMPL_OP_INFERSHAPE` /
`IMPL_OP_OPTILING` 直链宏展开即返回该类）。若当前工程的公共 tiling 头在此基础上另提供 LEGACY /
封装宏，先确认其展开保留依赖链式入口，再决定能否使用。

## 7 UNKNOWN_DIM(-1) 与运行期补齐

OPTIONAL 档位的完整执行链：

```
编译期 infershape: p_x 非常量 → 输出回退未知维，运行期补齐
                   p_x 是常量 → 直接读值推 shape
运行期:           框架 D2H 拷贝声明了依赖的输入 → 重跑 infershape（拿到真值，补齐未知维）→ 跑 tiling
```

**两种回退档位（按信息量取低者）**：

| 回退 | API | 语义 | 适用 |
|------|-----|------|------|
| 未知维长 | `SetAllUnknownDim(rank)`（-1） | 维数已知、各维长度未知 | MirrorPad paddings 非常量 |
| 未知维数 | `SetUnknownRank`（-2，shape=[-2]） | 连输出是几维都不知道 | RepeatInterleave repeats 非常量（重复后输出 rank 依赖 repeats 元素个数） |

- infershape 里 `x_shape->GetDim(i) == -1` 的维要传染给输出（`MirrorPad` 的写法：输入维未知 →
  输出维未知），不能拿 -1 参与算术
- `IsUnknownRank(x_shape)` 时输出直接 `SetUnknownRank`，不做任何读值推导。若输出维数与输入**解耦**
  （如 CropAndResize 输出恒 4 维、不随 x 的 rank 变），可回退 `SetAllUnknownDim(输出维数)` 保留
  rank 信息（按信息量取低者）；两种写法均可，保守传染 -2 也正确
- 回退粒度按信息量取低者：**只有值依赖的维置 -1，能静态推出的维保留**（部分维 -1）。如
  UnsortedSegmentSum 仅第 0 维依赖 num_segments 值（其余维传染 x）、MaxPoolV2 仅 H/W 两维依赖
  ksize/strides 值（N/C 保留）——不要用 `SetAllUnknownDim` 把已知维也抹掉
- tiling 阶段声明了 DataDependency 的输入，`GetInputTensor` 拿到的就是拷到 host 的数据，可安全
  `GetData<T>()`；tiling 需要的输出维长也可从 `GetOutputShape`（含运行时 shape）获取（§2.1 路线二）

## 8 读值 API 汇总

| 场景 | API | 行为差异 |
|------|-----|---------|
| infershape / tiling 取 tensor | `context->GetInputTensor(idx)` | 未声明依赖且非常量 → 数据地址为 nullptr |
| 读裸数据 | `tensor->GetData<T>()` / `GetShapeSize()` / `GetDataType()` | — |
| 单值读取（版本一） | `Ops::Base::GetConstInt(context, idx, value)` | 纯 dtype switch，**不做**常量判定；数据为空时返回 false |
| 单值/数组读取（版本二） | `ops::GetConstInt` / `ops::GetConstIntToShape` | 内含 `IsConstTensor` 前置检查，非常量直接返回 false |
| 常量判定 / -1 回退 | `IsConstTensor` / `SetAllUnknownDim` / `SetUnknownRank` | — |

**版本选择规则（按工程环境自适应，与仓无关）**：两版同名不同 namespace、行为不同——

- 当前工程提供哪版头文件就用哪版，且与工程内既有算子代码保持一致
- 用**不做**常量判定的版本（`Ops::Base::`）时，非常量输入会与"dtype 不支持/读失败"混在同一类
  false 返回里——OPTIONAL 档位必须自己先做 `IsConstTensor` 分支再调它，否则回退分支永远走不到
- 两版均仅支持 INT32/INT64/UINT32/UINT64，其余 dtype 返回 false，调用处必须检查返回值
- **浮点标量值依赖输入**（alpha/beta、iou_threshold 类）不能用 `GetConstInt`（仅整型）——手写
  `GetData<T>()` 按 dtype 模板化，覆盖 def 声明的全部档位 + default 报错
- 手写读值必须按 dtype 模板化并覆盖 INT32/INT64 两档（值依赖输入的标准 dtype 组合），且带 default
  报错分支（"INT32 之外的 else 一律按 INT64"的写法属于弱化版，不建议）
- `TilingInputsDataDependency` 有带 `TilingPlacement`（TILING_ON_HOST/TILING_ON_AICPU）的重载，
  默认 host tiling 场景不需要

## 9 L2→L0：ConvertToTensor 机制（host→device 反方向）

L2（aclnn）与 L0（kernel/Ascend C）对同一参数的类型要求不同：

- L2 接口签名面向用户，小参数用宿主类型：`aclIntArray* pad`、`aclScalar* value`
- L0 算子执行到 kernel 时**所有输入必须是 tensor**

弥合机制（`aclOpExecutor::ConvertToTensor`，声明于 `aclnn/opdev/op_executor.h`）：

```cpp
// 模板版实现：host 值包装成 aclTensor，托管进 executor
const aclTensor* ConvertToTensor(const T* value, uint64_t size, op::DataType dataType) {
    tensor = new aclTensor(value, size, dataType);      // placement = host
    allocatedTensorList_.push_back(tensor);             // executor 管理生命周期
    return tensor;
}
```

- 重载覆盖 `aclIntArray/aclBoolArray/aclFloatArray/aclFp16Array/aclBf16Array/aclScalar/T*`
- tensor 数据仍在 host，launch 阶段由框架统一做 **H2D 拷贝**进 kernel 可见的 device 内存
- 典型调用：`PadV3` 的 `pad` 数组、`value` 标量、`StridedSlice` 的 begin/end/strides 全部
  `ConvertToTensor` 后才进 `l0op::`
- 排查技巧：**L2 头文件里某参数不是 `aclTensor*`，则 L2 实现里必有一处 `ConvertToTensor`**（或
  L0 算子该参数声明为 scalar 输入）。反之，用户传 device tensor 而实现里直接 `*value` 读 host 值，
  即为方向搞反的 bug
- 反方向（L2 读 device tensor 值）无通用机制：L2 `GetWorkspaceSize` 运行在 host，禁止解引用
  `aclTensor` 的 device 地址。确需 device 值参与 L2 决策时，把逻辑下沉到 infershape/tiling（值依赖）
  或拆成两个阶段

## 10 输出 shape 依赖计算结果（边界场景，勿混用）

另一类"动态"：输出 shape 由 **device 侧计算结果**决定（`Unique`、`NonZero`、`Sort` 的有效长度等）。
host 侧任何值依赖声明都救不了它——值在 kernel 执行完才产生。机制上走
`OpParamDef::OutputShapeDependOnCompute()`（`register/op_def.h`，声明在 **op_def 输出参数**上），
由框架在 kernel 执行后回填输出 shape。该场景由 **VariableOutput 范式**接管
（`paradigms/variable_output/patterns.md`），不要与值依赖混用。

**分界**：输入张量的值、host 执行前可读 → DynamicShape（本文）；device 计算结果、执行后才存在 →
VariableOutput。

## 11 动态 shape 能力声明（OpAICoreConfig）

`*_def.cpp` 的 AICore 配置块与动态 shape 相关的开关：

```cpp
OpAICoreConfig aicoreConfig;
aicoreConfig.DynamicCompileStaticFlag(true)   // 支持动静态编译
    .DynamicRankSupportFlag(true)             // 支持动态维数（rank 未知，shape=[-2]）
    .DynamicShapeSupportFlag(true);           // 支持动态 shape（维数已知、维长未知，-1）
this->AICore().AddConfig("ascend950", aicoreConfig); // 以 arch35 为例
```

值依赖 OPTIONAL 档位 + `-1` 回退的算子，通常需要同时打开 `DynamicShapeSupportFlag`，否则回退产出的
未知维过不了构图校验；`-2` 回退对应 `DynamicRankSupportFlag`。

## 12 自包含声明

- 本范式是**自包含知识**：给定算子规格（语义 + 输入输出表）+ 本文档，即可完成全部值依赖分析与
  昇腾代码表示。**不需要、也不应该去查阅算子库或任何存量实现**——分析依据是规格语义（§2.1
  思想实验）与本文档的机制规则，不是"别人怎么写的"
- 文中出现的算子名（MirrorPad、Trilu、Reshape 等）仅作为**语义角色与取向的举例**，不是"参考
  实现"；禁止据此到库上检索、对照或抄写其代码
- 若分析中对某输入的判定拿不准，回到 §2.1 思想实验与 §5 选择依据重新推导，而不是找类似算子的
  实现来仿写
- 档位/读值侧取向速查见 [patterns.md](patterns.md) 通用规则表（按参数角色组织，同样自包含）

## 13 常见误区

1. **按输入下标判值依赖**——判定依据是"host 是否解引用其值"（§2），input0 惯例上只是恰好不需要
2. **把"声明依赖"当读值的无条件前提**——常量输入天然可读；声明服务于运行期（非常量）输入的 D2H
   （见 §5 分层事实）。新写算子仍应声明齐全：漏声明的症状不在常量路径爆发，而在动态输入路径拿
   nullptr
3. **漏写 IsConstTensor 回退**——OPTIONAL 档位编译期 infershape 就可能拿到非常量 tensor，无回退
   分支直接构造失败
4. **-1 参与算术**——`x_shape` 含 -1 时 `x + pad` 产出非法维长，未知维只能传染不能计算
5. **L2 里读 device tensor**——`ConvertToTensor` 是 host→device 方向；反方向不存在，需要 device 值
   就下沉到 infershape/tiling
6. **乱声明依赖**——每条 `DataDependency` 都附带一次运行期 D2H 同步，"顺手声明"拖慢所有调用。声明
   集合应与实际读值集合精确一致
7. **dtype 只按 INT64 读**——值依赖输入 INT32/INT64 两档都要支持，`GetConstInt` 报 false 就走报错
   或回退，不要硬转
8. **混淆两版 GetConstInt**——带 `IsConstTensor` 检查与不带检查的两版同名 API 行为不同，按当前
   工程可用头文件选择并用对配套写法（§8）
9. **框架代读值漏声明**——tiling 复用公共 tiling 模板 / auto-tiling 时，值由框架代码消费，算子
   代码里看不到 `GetData`，但依赖声明仍必须由算子注册块承担
10. **把 kernel 读值当值依赖**——只有 kernel（device 侧）消费数值的输入不是值依赖，kernel 侧也
    不需要任何值依赖表示
11. **给 REQUIRED 档补读值侧声明**——REQUIRED 强制常量、天然可读，DataDependency 声明是多余的；
    声明服务于"非常量输入的 D2H"，而 REQUIRED 档不存在非常量路径
12. **对未提供的可选输入按错误处理**——`ParamType(OPTIONAL)` 输入未提供时
    `GetOptionalInputTensor` 返回 nullptr，应按默认值语义继续，不是 GRAPH_FAILED
13. **拿不准时去找类似算子的实现来仿写**——判定拿不准应回到 §2.1 思想实验与 §5 选择依据重新
    推导；本范式自包含，不依赖库上实现（§12）
