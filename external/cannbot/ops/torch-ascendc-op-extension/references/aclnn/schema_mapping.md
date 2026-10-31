# 契约对齐：aclnn 声明 ↔ torch schema ↔ C++ 形参 ↔ converter

PTA 层本质上是一层"名字与顺序的搬运"。四份契约中任何一处对不齐，症状都是编译通过、运行报错或静默算错。本文给出逐项对齐规则与完整样例。

## 1. 四份契约的权威来源

| 契约 | 权威文件 | 决定什么 |
|------|---------|---------|
| aclnn 形参**顺序与类型** | `op_host/op_api/aclnn_<op>.h` | `EXEC_NPU_CMD_V1` 的实参顺序 |
| GE **op_type / Input / Output / Attr 名** | `op_host/<op>_def.cpp`（OpDef） | converter 的 `custom_op` 第一参、`inputs`/`attrs` 的 key、`outputs` 顺序 |
| 输出 **shape 推导** | `op_host/<op>_proto.cpp`（infershape） | Meta impl 是否需要自己推导 |
| torch **调用形态** | 自定义（schema） | Python 侧参数名、可选性、默认值 |

> **aclnn 形参名与 OpDef 名经常不一致**（camelCase vs snake_case，`output` vs `y`）。不要假设同名，逐项核对。

## 2. 类型映射总表

| aclnn 形参 | torch schema | C++ 形参 | converter | ConvertType 支持 |
|-----------|-------------|---------|-----------|-----------------|
| `const aclTensor*`（必选输入） | `Tensor x` | `const at::Tensor& x` | `inputs["x"]` | ✅ |
| `const aclTensor*`（可选输入） | `Tensor? x` | `const c10::optional<at::Tensor>& x` | 非 None 才放进 `inputs` | ✅（None→nullptr） |
| `const aclTensor*`（输出，预分配） | `Tensor! out` | `at::Tensor& out` | 出现在 `outputs` 列表 | ✅ |
| `const aclTensor*`（可选输出） | `Tensor? out=None` | `const c10::optional<at::Tensor>& out` | 非 None 才进 `outputs` | ✅ |
| `const aclTensorList*` | `Tensor[] xs` | `at::TensorList xs` | — | ✅ |
| `const char*` | `str s` | `c10::string_view s` | `attr.Str(s)` | ⚠️ 须先 `const_cast<char*>(s.data())` |
| `int64_t` | `int v` | `int64_t v` | `attr.Int(v)` | ✅ |
| `bool` | `bool v` | `bool v` | `attr.Bool(v)` | ✅（模板透传） |
| `float` / `double` | `float v` | `double v` | `attr.Float(v)` | ⚠️ aclnn 收 `float` 时须显式 `static_cast<float>(v)` |
| `const aclIntArray*` | `int[] v` | `at::IntArrayRef v` | `attr.ListInt(v)` | ✅ |
| `const aclIntArray*`（可选） | `int[]? v=None` | `c10::optional<at::IntArrayRef> v` | 同上 | ✅ |
| `const aclScalar*` | `Scalar v` | `const at::Scalar& v` | — | ✅ |
| `aclDataType` | `ScalarType t` | `at::ScalarType t` | — | ✅ |

`ConvertType` 无重载的类型（`c10::string_view`、`std::string`、枚举等）必须在 impl 里手工转成上表中的形态，否则报 "no matching function for call to ConvertType"。

## 3. 完整样例（XAllGatherMatmul，九参数 MC2 算子）

挑这个算子做样例，是因为它同时覆盖了可选输入、字符串属性、多输出、可选输出、以及 aclnn 名与 OpDef 名不一致这五种情况。

**① aclnn 声明**（`aclnn_x_all_gather_matmul.h`）：

```cpp
aclnnStatus aclnnXAllGatherMatmulGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const char* group,
    int64_t gatherIndex, int64_t commTurn, int64_t streamMode,
    const aclTensor* output, const aclTensor* gatherOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);
```

**② schema**：形参顺序与 aclnn 一一对应，末尾两个 out 参数不出现在 schema 里。

```
npu_x_all_gather_matmul(Tensor x1, Tensor x2, Tensor? bias, str group,
                        int gatherIndex, int commTurn, int streamMode,
                        Tensor! output, Tensor? gatherOut=None) -> (Tensor output, Tensor? gatherOut)
```

**③ NPU impl**：`EXEC_NPU_CMD_V1` 的实参顺序 = aclnn 形参顺序（去掉末尾 `workspaceSize` / `executor`，由宏补齐）。

```cpp
char *group_ptr = const_cast<char *>(group.data());
EXEC_NPU_CMD_V1(aclnnXAllGatherMatmul, x1, x2, bias, group_ptr,
                gatherIndex, commTurn, streamMode, output, gatherOut);
return {output, gatherOut};
```

**④ converter**：key 取自 OpDef，不是 aclnn 形参名。

```cpp
// OpDef 声明
this->Input("x1"); this->Input("x2"); this->Input("bias");      // OPTIONAL
this->Output("y"); this->Output("gather_out");
this->Attr("group").String();
this->Attr("gather_index").Int(0);  this->Attr("comm_turn").Int(0);
OP_ADD(XAllGatherMatmul);
```

对应 converter 应写成（注意 `gatherIndex` → `gather_index`、`output` → `y` 这两处改名，以及 aclnn 才有的 `streamMode` 不出现在 attrs 里）：

```python
inputs = {"x1": x1, "x2": x2}
if bias is not None:
    inputs["bias"] = bias
attrs = {"group": attr.Str(group), "gather_index": attr.Int(gatherIndex),
         "comm_turn": attr.Int(commTurn)}
outputs = ["y", "gather_out"] if gatherOut is not None else ["y"]
return torchair.ge.custom_op("XAllGatherMatmul", inputs=inputs, attrs=attrs, outputs=outputs)
```

> ⚠️ **不要照抄现成 converter**：MC2 通算融合类算子的 PTA 常常只验证了 eager 路径、根本没跑过入图，其 converter 里 attr key 直接沿用 aclnn 形参名（`gatherIndex` / `commTurn`）、`outputs` 沿用 aclnn 输出名（`["output", "gatherOut"]`）、甚至混入 OpDef 中不存在的属性（`streamMode`）的情况很常见。这类写法在 eager 下不会暴露，一入图就报 attr/output 不存在。以自己的 OpDef 为准逐项核对，或干脆跳过 Step 4 只做 eager。

## 4. schema 语法要点

| 写法 | 含义 | 常见错误 |
|------|------|---------|
| `Tensor!` | in-place / 会被写入的可变张量 | 输出漏了 `!`：torch 认为是纯函数，`torch.compile` 下可能被 CSE 消掉 |
| `Tensor?` | 可选（可传 `None`） | 可选输入写成 `Tensor`：Python 传 `None` 报类型错误 |
| `=None` / `=0` | 默认值，只能给尾部参数 | 中间参数带默认值：`m.def` 解析直接抛异常 |
| `-> (Tensor a, Tensor? b)` | 多返回值，名字仅作文档 | 返回值个数与 C++ `std::tuple` 元素数不一致：dispatch 时崩溃 |
| `Tensor[]` | TensorList | — |

schema 的参数名就是 Python 关键字参数名，改名即破坏调用方兼容性。

## 5. 输出策略：in-place 预分配 vs PTA 层分配

**默认：in-place 预分配**——调用方 `torch.empty(...)` 分配好传进来，schema 用 `Tensor!`，Meta impl 直接透传。适合 shape 由业务侧算好、或多输出/可选输出复杂的场景。

**改为 PTA 层分配**时三处同改：

```cpp
// schema：输出不再是入参
"npu_x_custom_op(Tensor x1, Tensor x2) -> Tensor"

// NPU impl：自己 empty 出输出（形状按算子语义推导）
at::Tensor npu_x_custom_op(const at::Tensor& x1, const at::Tensor& x2) {
    auto output = at::empty({x1.size(0), x2.size(1)}, x1.options());
    EXEC_NPU_CMD_V1(aclnnXCustomOp, x1, x2, output);
    return output;
}

// Meta impl：必须做同样的 shape 推导（否则 torch.compile 下 shape 对不上）
at::Tensor npu_x_custom_op_meta(const at::Tensor& x1, const at::Tensor& x2) {
    return at::empty({x1.size(0), x2.size(1)}, x1.options());
}
```

Meta impl 与 NPU impl 的 shape/dtype 推导逻辑必须严格一致——这是 in-place 方案能"Meta 只透传"而分配方案不能的原因。分配用 `at::empty`，不要用 `at::zeros`（多一次无谓的 NPU 下发）。

## 6. 对齐自检清单

改完后逐条核对：

- [ ] `EXEC_NPU_CMD_V1` 实参个数 == aclnn `GetWorkspaceSize` 形参个数 − 2
- [ ] C++ impl 形参顺序/类型逐个匹配 schema（含可选性与默认值）
- [ ] `m.impl` 里的算子名字符串与 `m.def` 中的名字完全一致（写错不报错，只是 dispatch 找不到实现）
- [ ] Meta impl 与 NPU impl 签名完全一致
- [ ] converter 函数签名与 schema 逐参对齐，末尾有 `*, meta_outputs=None`
- [ ] converter 的 `custom_op` 第一参 == OpDef 的 `OP_ADD(...)` 名
- [ ] converter 的 `inputs` / `attrs` key == OpDef 的 `Input` / `Attr` 名
- [ ] converter 的 `outputs` 顺序 == OpDef 的 `Output` 声明顺序
- [ ] `xops/__init__.py` 的 `from .converter import (...)` 列全了所有算子
