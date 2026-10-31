# DynamicShape 类算子场景路由

> 本文档用于**场景判定**和**策略选择**。确定场景后，按链接进入对应详细文档。
>
> DynamicShape 是**修饰范式**：算子的输出 shape 或 tiling 参数依赖输入张量的**数值**（而非
> shape/dtype），即"动态 shape"。核心机制是**值依赖**——运行期（非常量）输入的数据在 device 上，
> 不声明契约就在 host 侧（infershape/tiling）读它的值，拿到的是空指针或 device 地址，解引用直接
> 崩溃。
>
> 使用本文档回答两个问题：
> 1. **怎么分析**——某输入是不是值依赖、档位如何、哪侧读值（§场景判定流程）；
> 2. **怎么表示**——分析结论落到昇腾代码各层怎么写（路径 A/B 的契约与模板）。
>
> 机制原理、值依赖分析方法、失败根因见 [knowledge.md](knowledge.md)。范式是**自包含知识**：
> 规格 + 范式文档即可完成全部分析与代码表示，不需要也不应查阅算子库/存量实现（knowledge.md §12）。

---

## 场景判定流程

```
给定: 算子规格（spec/README：语义 + 输入输出表）

Step 1 — 值依赖输入识别:
  逐输入做"变值不变形"思想实验——固定该输入 shape/dtype、仅改其值:
    ├─ 输出 shape 会变，或 host 侧 tiling 决策会变（含 tiling 复用公共模板/框架代读
    │  消费的输入——算子代码看不到读值点也计入）→ 值依赖输入 → Step 2
    └─ 只有 kernel 计算结果变 → 普通数据张量，非值依赖 → 按主范式处理 → 结束

Step 2 — 常量强度判定（决定 ValueDepend 档位，按读值侧分两种情况）:
  infershape 读值（输出 shape 依赖该值）:
    值未知时输出 shape 能否回退未知维（-1/-2）、由运行期补齐？
      ├─ 能回退（paddings/axes/multiples 类）→ ValueDepend(OPTIONAL) + IsConstTensor 回退分支
      └─ 不能（值未知则维数语义不成立/生态强制常量, 如 SliceV2 offsets/size）→ ValueDepend(REQUIRED)
  仅 tiling 读值（输出 shape 不依赖该值）:
    OPTIONAL 机械上总是可行（运行期 D2H 先于 tiling, tiling 拿真值出方案）
      ├─ 需支持运行期张量 → ValueDepend(OPTIONAL) + tiling 侧声明
      └─ 生态上恒为构图期常量、无需动态 → 可收紧 ValueDepend(REQUIRED):
         构图期强制常量, tiling 直接读值且免读值侧声明

Step 3 — 声明侧别（哪侧读值就声明哪侧）:
  infershape 读值 → .InputsDataDependency({idx})
  tiling 读值     → .TilingInputsDataDependency({idx})
  两侧都读       → 两侧都声明
  注意: 声明的作用是让"运行期(非常量)输入"在该侧可读(触发 D2H);
        常量输入天然可读。要支持动态输入的算子,读值侧声明必须齐全——
        漏一侧则该侧在非常量路径 GetInputTensor 拿到 nullptr
  tiling 需要的量若已体现在输出 shape 中,可实现为"tiling 读补齐后的输出 shape"
    （TilingContext::GetOutputShape 含运行时 shape）→ tiling 侧免声明（设计选择,
      与直读值路线二选一,声明集合与实际读值集合一致即可）
  声明集合按"所有图可能读值的并集"取——注册块声明是静态的,无法按属性值分支最小化
    （如 SAME 模式只需 strides,仍按 {ksize,strides} 声明）
  注册形态: 需挂依赖必须用能链式返回 OpImplRegisterV2 的直链注册宏
        (IMPL_OP_INFERSHAPE / IMPL_OP_OPTILING); 工程公共头另提供的
        封装/LEGACY 宏需先确认展开后保留依赖链式入口;
        Tiling 函数转发进公共模板(DoTilingImpl 等)时,声明仍挂算子自身注册块

Step 4 — 特殊子场景:
  值由公共框架代读(auto-tiling/compile_info 消费, 如 MaxPoolV2 的 ksize/strides)
    → 算子代码看不到 GetData, 依赖声明同样不能省
  L2 aclnn 层宿主参数（aclIntArray/aclScalar）要喂给只吃 tensor 的 L0 kernel
    → ConvertToTensor 宿主值转张量（见路径 B）
  输出 shape 由 device 侧计算结果决定（Unique/NonZero 类）
    → OutputShapeDependOnCompute（op_def 输出参数声明），不走路径 A/B，见 knowledge.md §10
```

**路径归属**：Step 1 判定为 YES 的输入走 **路径 A（host 读 device 值，契约声明 + 读值落地）**；
L2 宿主值下发走 **路径 B（ConvertToTensor）**。两者方向相反，经常在同一算子的不同层同时出现。

每个值依赖输入的分析结论汇总为四元组 `(输入名, 索引, 档位, 读值侧)`，代码表示由四元组唯一决定
（knowledge.md §6.1 表示决策表）。

---

## 通用规则

**判定法则：infershape 或 tiling 阶段（含框架代读）要解引用该张量的 `GetData<T>()` → 必须值依赖。**
反之（只读 shape/dtype、仅 kernel 读值）→ 不需要。

| 输入语义 | 是否值依赖 | 档位 | 实例 |
|---------|----------|------|------|
| 主数据张量 `x` | ❌ | — | Add 的 x/y（host 只读其 shape） |
| 轴/维度描述 `axes`/`axis`/`dim` | ✅ | OPTIONAL | ReduceSum axes、Cumsum axis、GatherV2 axis、ReverseV2 axis |
| 目标 shape 描述张量 | ✅ | OPTIONAL | Reshape 的 shape、BroadcastTo 的 shape、Fill 的 dims、Expand 的 multiples |
| padding/切片窗口参数 | ✅ | OPTIONAL 或 REQUIRED | MirrorPad paddings(OPT)、SliceV2 offsets/size(REQ)、StridedSlice begin/end/strides |
| 数量/长度参数 | ✅ | OPTIONAL | OneHot depth、RepeatInterleave repeats、Bincount minlength |
| 标量权重（仅 tiling 读） | ✅ | OPTIONAL | AddR alpha/beta（浮点标量不用 GetConstInt，手写模板读） |
| mask/有效规模描述（仅 tiling 读） | ✅ | tiling-only | IndexPutV2 的 mask / indexed_sizes |
| 对角线/窗口偏移（仅 tiling 读） | ✅ | OPTIONAL 或 REQUIRED | Trilu k(OPT)、PasteSubImg 坐标(REQ) |

> 表中档位为该类输入的常见取值。**仅 tiling 读值**（输出 shape 不依赖该值）的输入，OPTIONAL 与
> REQUIRED 均可行（前者支持运行期张量，后者生态恒常量、免读值侧声明），选择依据见 knowledge.md §5。

> 值依赖与输入序号无关。惯例上 input0 是主数据张量（host 永远不读它的值），小下标参数张量
> （input1/2/...）才是值依赖高发区——但判定依据始终是"host 侧是否读值"，不是下标（
> AvgPool3DGrad 的 orig_input_shape、StatelessUniform 的 shape 都在下标 0）。

---

## 路径 A：契约声明（host 读 device 值）

| # | 位置 | 声明 | 作用层 |
|---|------|------|--------|
| 1 | `*_def.cpp` 输入定义 | `.ParamType(REQUIRED).ValueDepend(OPTIONAL/REQUIRED)` | 图层：标记该输入的值参与 shape/tiling 决策 |
| 2 | `IMPL_OP_INFERSHAPE` 注册块 | `.InputsDataDependency({idx})` | 运行期（非常量）输入在 infershape 执行前 D2H 到 host；常量输入天然可读 |
| 3 | `IMPL_OP_OPTILING` 注册块 | `.TilingInputsDataDependency({idx})` | 运行期（非常量）输入在 tiling 执行前 D2H 到 host |

- `idx` = 该输入在 Input 链中的索引（从 0 开始）。例：`x` 在 0、`paddings` 在 1 → 填 `{1}`
- 声明的真实作用域是**非常量运行期输入**：常量输入天然可读，声明服务于非常量路径的 D2H；支持
  动态输入的算子必须读值侧声明齐全，否则非常量路径 `GetInputTensor` 拿 nullptr
- `IMPL_OP_OPTILING` 块内的 `.InputsDataDependency({...})` 是等价历史写法（同一注册类提供两个入口），
  新代码统一用 `.TilingInputsDataDependency`
- 不要为"保险"乱声明：每处依赖都意味着运行期一次 D2H 同步拷贝
- 共用 tiling 文件服务多个算子时，依赖契约按算子分别声明（契约的主体是算子，不是文件）
- **kernel 侧不改**：值依赖输入对 kernel 就是普通 tensor 输入

### A.1 infershape 读值模板（IsConstTensor + 回退）

```cpp
static ge::graphStatus InferShapeForXxx(gert::InferShapeContext* context) {
  const gert::Shape* x_shape = context->GetInputShape(0);
  gert::Shape* y_shape = context->GetOutputShape(0);
  const gert::Tensor* p_tensor = context->GetInputTensor(1);   // 值依赖输入
  OPS_CHECK_NULL_WITH_CONTEXT(context, p_tensor);

  if (IsUnknownRank(x_shape)) {
    return SetUnknownRank(y_shape);
  }
  if (IsConstTensor(p_tensor)) {          // 常量：直接读值推 shape
    switch (p_tensor->GetDataType()) {    // dtype 覆盖 INT32/INT64 + default 报错
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

IMPL_OP_INFERSHAPE(Xxx).InferShape(InferShapeForXxx).InputsDataDependency({1});
```

### A.2 tiling 读值模板

```cpp
static ge::graphStatus Tiling4Xxx(gert::TilingContext* context) {
  const gert::Tensor* p_tensor = context->GetInputTensor(1);   // 值依赖输入
  OPS_CHECK_NULL_WITH_CONTEXT(context, p_tensor);
  const int32_t* p_value = p_tensor->GetData<int32_t>();     // 已声明依赖，host 可读
  ...
}

IMPL_OP_OPTILING(Xxx).Tiling(Tiling4Xxx).TilingInputsDataDependency({1});
```

- 可选输入（`ParamType(OPTIONAL)`）改用 `context->GetOptionalInputTensor(1)`：返回 nullptr =
  未提供，按默认值语义继续（如 Trilu 的 k 未传 → k=0），判空不走报错
- tiling 侧声明了依赖即保证 host 可读（常量天然可读、非常量已 D2H），无需 IsConstTensor 回退——
  回退分支是 infershape 侧的要求
- Tiling 函数转发进公共模板（框架代读）时，读值在模板内部，声明仍挂算子注册块（见 Step 4）

### A.3 推荐辅助 API（免手写 dtype switch）

| API | 用途 | 行为 |
|-----|------|------|
| `Ops::Base::GetConstInt(context, idx, value)` | 读单值 | 纯 dtype switch，**不做**常量判定；数据为空返回 false |
| `Ops::Base::GetConstIntToShape(context, idx, shape)` | 读整型数组拼成 `gert::Shape` | 同上，不做常量判定 |
| `ops::GetConstInt` / `ops::GetConstIntToShape` | 同名另一版 | **内含 IsConstTensor 检查**（非常量返回 false） |
| `IsConstTensor(tensor)` / `SetAllUnknownDim` / `SetUnknownRank` | 常量判定与未知维回退 | — |

选择规则（按当前工程环境，与仓无关）：工程头文件提供哪版就用哪版，并与工程内既有代码一致。
用**不做**常量判定的版本时，OPTIONAL 档位必须自己先 `IsConstTensor` 分支再调它，否则"非常量"
与"读失败"混在同一类 false 里，回退分支走不到。两版均仅支持 INT32/INT64/UINT32/UINT64，返回
false 必须处理。非常量回退按信息量取低者：维长未知 → `SetAllUnknownDim`(-1)，连维数都未知 →
`SetUnknownRank`(-2)（RepeatInterleave repeats 非常量时输出 rank 不定，即 -2 回退）。

---

## 路径 B：L2 宿主值下发（ConvertToTensor）

L2 aclnn 接口的宿主参数（`aclIntArray* pad`、`aclScalar* value` 等）不是 tensor，而 L0 算子
（kernel）只接受 tensor 输入。L2 实现中调 L0 前必须转换：

```cpp
// aclnnConstantPadNdGetWorkspaceSize 内
const aclTensor* padTensor = executor->ConvertToTensor(noneNegPad, DataType::DT_INT32);   // 数组→tensor
const aclTensor* valueTensor = executor->ConvertToTensor(value, self->GetDataType());    // 标量→tensor
auto result = l0op::PadV3(selfCasted, padTensor, valueCasted, MODE, true, executor);
```

- `ConvertToTensor` 在 host 侧 new 一个包装该值的 `aclTensor`（placement=host），加入 executor
  托管列表；launch 时由框架做 H2D 拷贝，生命周期由 executor 管理
- L2 头文件参数类型与 L0 输入类型的差异，正是靠这个机制弥合：**L2 声明非 tensor 类型 ≠ kernel
  不需要 tensor，而是 L2 内部必然调用了 ConvertToTensor**
- 支持重载：`aclIntArray/aclBoolArray/aclFloatArray/aclFp16Array/aclBf16Array/aclScalar/T*`
  （`aclnn/opdev/op_executor.h`）
- 反方向（L2 GetWorkspaceSize 里读 device tensor 的值）**没有**等价的通用机制，禁止在 L2 host 侧
  解引用 `aclTensor` 的 device 地址

---

## 与主范式融合

DynamicShape 是修饰范式，与主范式共同声明：

```yaml
paradigms: [主范式, DynamicShape]   # 如 [Reduction, DynamicShape]、[Broadcast, DynamicShape]
```

### 修改清单

| 章节 | 修改 | 说明 |
|------|------|------|
| Kernel 入口 / CopyIn / Compute / CopyOut | ❌ 不改 | 跟主范式保持一致；值依赖输入对 kernel 是普通 tensor |
| **op_def 输入定义** | ✅ 改 | 值依赖输入追加 `.ValueDepend(...)`；该参数必须是张量 Input 而非属性 |
| **infershape / tiling 注册块** | ✅ 改 | 追加 `.InputsDataDependency` / `.TilingInputsDataDependency` |
| **infershape 实现** | ✅ 改 | 读值推导 + `IsConstTensor` 未知维回退分支（-1/-2） |
| **tiling 实现** | ✅ 改 | `GetInputTensor` 读值参与切分决策 |
| TilingData / TilingKey | ⚠️ 可选 | 值参与的分支若影响切分策略，才新增 key |

> 仅声明 `DynamicShape` 一项时无默认主范式，必须回 spec 确认计算主范式。

---

## 校验清单

**分析阶段**：

- [ ] 逐输入完成"变值不变形"判定，结论记为四元组（输入名/索引/档位/读值侧）
- [ ] 仅 kernel 读值的输入没有被误判为值依赖
- [ ] tiling 读值侧结论明确了获取路线（直读值 / 读补齐后输出 shape），声明与路线一致
- [ ] 值由公共框架代读（auto-tiling/公共模板/compile_info 消费）的输入未漏判
- [ ] 值依赖参数建为张量 Input 而非属性（若允许运行期值）
- [ ] 可选输入（ParamType(OPTIONAL)）的读值用 GetOptionalInputTensor + 默认值分支

**表示阶段**：

- [ ] 每个被 host 侧读值的输入都完成契约声明（def `ValueDepend` + 读值侧注册块 DataDependency）
- [ ] 声明侧别与读值行为精确对应（infershape 读/tiling 读/两侧都读），支持动态输入则读值侧齐全
- [ ] `ValueDepend(REQUIRED)` 的输入确属"值未知则无法决策"或"生态恒常量"，且**未**多余地补
      DataDependency 声明（REQUIRED 强制常量天然可读）
- [ ] OPTIONAL 档位 infershape 写了 `IsConstTensor` → 未知维回退分支（-1 或 -2 按信息量取低，
      只有值依赖的维置 -1、能静态推出的维保留）
- [ ] 读值使用 `GetConstInt/GetConstIntToShape`（版本与工程环境一致）或完整 dtype switch（INT32/INT64 + default 报错）
- [ ] 未读值的输入未被顺手声明依赖（避免无谓 D2H 同步）
- [ ] kernel 侧未引入任何值依赖特判
- [ ] L2 实现中所有宿主参数进 L0 前都过了 `ConvertToTensor`
- [ ] 空 tensor 路径：值依赖输入为空时 infershape 不崩溃，按 0 元素或报错语义处理

---

## 跨场景参考

| 主题 | 文档 |
|------|------|
| 值依赖分析方法（语义判定）、代码表示总览 | [knowledge.md](knowledge.md) §2/§6 |
| 数据驻留模型、失败根因、OPTIONAL/REQUIRED 语义 | [knowledge.md](knowledge.md) §3~§5 |
| Reduction 范式的值依赖应用（axes tensor，本范式的特例） | ascendc-regbase-best-practice skill: `references/paradigms/reduction/patterns.md` §3.1 |
