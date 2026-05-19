# Ascend C TOPK 问题清单

> **适用场景**：算子开发代码检视高频问题
>
> **说明**：TOPK 问题是检视实践中发现的高频风险点，需重点关注。条款标注适用范围：`[适用: All]` / `[适用: Host]` / `[适用: Kernel]`

## 快速索引

### 两者都适用 `[适用: All]`（5 条）

| 序号 | 问题类型 | 类别 | 严重级别 |
|-----|---------|------|---------|
| 3 | 生命周期内使用局部变量指针，避免野指针 | 内存安全 | 高 |
| 6 | 必须考虑nan/inf/+0/-0等特殊值和边界值处理 | 数值安全 | 高 |
| 8 | gm内存偏移或大小必须用int64表示 | 内存安全 | 高 |
| 10 | 可整数计算时不允许转浮点数计算 | 数值安全 | 中 |
| 12 | 宏定义中临时变量命名不能和外部变量冲突 | 编码规范 | 中 |

### 仅 Host 侧适用 `[适用: Host]`（7 条）

| 序号 | 问题类型 | 类别 | 严重级别 |
|-----|---------|------|---------|
| 1 | 必须校验函数返回值 | 编码规范 | 高 |
| 2 | 使用GetInputDesc获取Dtype，context获取Shape | API使用 | 高 |
| 4 | 属性从context获取，禁止CompileInfo传递 | API使用 | 高 |
| 5 | 属性获取类型需与ir原型一致 | 类型安全 | 高 |
| 7 | 融合规则/InferShape/Tiling外部输入校验 | 输入验证 | 高 |
| 13 | dlopen管理的so禁用thread_local | 并发安全 | 高 |

### 仅 Kernel 侧适用 `[适用: Kernel]`（2 条）

| 序号 | 问题类型 | 类别 | 严重级别 |
|-----|---------|------|---------|
| 9 | atomic累加需src(ub)与dst(gm)清零处理 | 内存安全 | 高 |
| 11 | 通信算子融合需核间同步 | 并发安全 | 高 |

---

## 详细规范

### 1. 必须校验函数返回值 `[适用: Host]`

**问题说明**

Host侧代码调用函数时，必须校验函数返回值，确保操作成功或正确处理错误情况。未校验返回值可能导致程序在错误状态下继续执行，引发后续问题。

**正确示例**

```cpp
auto ret = context->GetAttrPointer<int64_t>(ATTR_INDEX);
if (ret == nullptr) {
    return ge::GRAPH_FAILED;
}
int64_t attrValue = *ret;
```

**错误示例**

```cpp
auto ret = context->GetAttrPointer<int64_t>(ATTR_INDEX);
int64_t attrValue = *ret;  // 未校验 ret 是否为 nullptr，可能解引用空指针
```

---

### 2. 使用GetInputDesc获取Dtype，context获取Shape `[适用: Host]`

**问题说明**

Host侧代码不要使用 `GetInputTensor` 获取 Shape 和 Dtype。应使用 `GetInputDesc` 获取的对象来获取 Dtype，使用 context 获取对应的 Shape。`GetInputTensor` 在某些场景下可能返回无效数据。

**正确示例**

```cpp
// 获取 Dtype
auto inputDesc = context->GetInputDesc(0);
ge::DataType dtype = inputDesc->GetDataType();

// 获取 Shape
auto shape = context->GetInputDesc(0)->GetShape().GetDims();
```

**错误示例**

```cpp
// 错误：使用 GetInputTensor 获取 Dtype
auto tensor = context->GetInputTensor(0);
ge::DataType dtype = tensor->GetDataType();  // 不推荐
```

---

### 3. 生命周期内使用局部变量指针，避免野指针 `[适用: All]`

**问题说明**

在生命周期内使用局部变量指针时，必须确保指针指向的对象在作用域内有效。返回局部变量指针或在作用域外使用局部变量指针会导致野指针问题。

> **Kernel 侧说明**：Ascend C 的 `LocalTensor` 由 `TQue`/`TBuf` 管理生命周期，通过 `AllocTensor`/`FreeTensor` 配对使用。如果在 `FreeTensor` 后继续使用，或在不同核函数间传递，可能导致野指针问题。

**正确示例**

```cpp
// Host 侧
int64_t ComputeLength(const std::vector<int64_t>& shape) {
    int64_t length = 1;
    for (auto dim : shape) {
        length *= dim;
    }
    return length;  // 返回值，而非指针
}

// Kernel侧
AscendC::LocalTensor<float> xLocal = inputQueue.AllocTensor<float>();
AscendC::DataCopy(xLocal, inputGM[0], tileLength_);
inputQueue.EnQue(xLocal);
// 在 FreeTensor 前使用完毕
AscendC::LocalTensor<float> yLocal = inputQueue.DeQue<float>();
AscendC::Add(zLocal, yLocal, xLocal, tileLength_);
inputQueue.FreeTensor(yLocal);  // 使用后释放
```

**错误示例**

```cpp
// Host侧：返回局部变量指针
int64_t* GetLengthPointer(const std::vector<int64_t>& shape) {
    int64_t length = 1;
    for (auto dim : shape) {
        length *= dim;
    }
    return &length;  // 返回局部变量指针，函数返回后变为野指针
}

// Kernel侧：FreeTensor 后继续使用
AscendC::LocalTensor<float> xLocal = inputQueue.AllocTensor<float>();
inputQueue.FreeTensor(xLocal);
AscendC::Add(zLocal, xLocal, yLocal, tileLength_);  // 已释放，野指针
```

---

### 4. 属性从context获取，禁止CompileInfo传递 `[适用: Host]`

**问题说明**

Host侧代码获取属性必须从 context 获取，不允许使用 CompileInfo 传递算子属性值。CompileInfo 主要用于传递编译期确定的配置信息，而非运行时属性。

**正确示例**

```cpp
auto attrs = context->GetAttrs();
auto* attrValue = attrs->GetAttrPointer<int64_t>(ATTR_INDEX);
```

**错误示例**

```cpp
// 错误：通过 CompileInfo 传递属性
compileInfo->GetAttrValue(ATTR_INDEX);  // 不推荐
```

---

### 5. 属性获取类型需与ir原型一致 `[适用: Host]`

**问题说明**

Host侧代码属性的获取，要用 ir 原型对应的类型去获取。`GetAttrPointer<T>` 的类型参数必须与 ir 原型定义的属性类型一致，否则会导致类型错误。

**示例说明**

假设 `swiglu_mx_quant_def.cpp` 文件中属性定义如下：
- `activate_dim` 是 `Int` → 使用 `int64_t`
- `activate_left` 是 `bool` → 使用 `bool`
- `clamp_limit` 是 `float` → 使用 `float`
- `round_mode` 是 `string` → 使用 `char`

**正确示例**

```cpp
auto attrs = context->GetAttrs();
auto* attrActivateDim = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_ACTIVATE_DIM);   // Int → int64_t
auto* attrActivateLeft = attrs->GetAttrPointer<bool>(INDEX_ATTR_ACTIVATE_LEFT);     // bool → bool
auto* attrClampLimit = attrs->GetAttrPointer<float>(INDEX_ATTR_CLAMP_LIMIT);        // float → float
const char* attrRoundMode = attrs->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);     // string → char
```

**错误示例**

```cpp
auto attrs = context->GetAttrs();
auto* attrActivateDim = attrs->GetAttrPointer<int32_t>(INDEX_ATTR_ACTIVATE_DIM);   // 错误：Int 应为 int64_t
auto* attrActivateLeft = attrs->GetAttrPointer<int>(INDEX_ATTR_ACTIVATE_LEFT);      // 错误：bool 应为 bool
```

---

### 6. 必须考虑nan/inf/+0/-0等特殊值和边界值处理 `[适用: All]`

**问题说明**

算子设计时必须考虑 `nan`、`inf`、`-inf`、`+0`、`-0` 等特殊值和边界值处理。这些特殊值在计算中可能产生非预期结果，导致算子输出异常。

> **Kernel 侧说明**：Ascend C 计算指令对特殊值的处理可能与 CPU 不同，需特别注意 FP16/BF16 精度下的边界值行为。

**关注场景**

- 除法运算：除数为0、nan、inf
- 比较运算：nan 与任何值比较结果
- 类型转换：FP32 转 FP16 时的溢出（超出 FP16 范围变成 inf）
- 归约运算：输入包含 nan 时输出结果

---

### 7. 融合规则/InferShape/Tiling外部输入校验 `[适用: Host]`

**问题说明**

融合规则、InferShape、Tiling 的外部输入使用时必须进行合法性校验。外部输入可能包含非法值，未校验可能导致后续计算错误或程序崩溃。

**校验内容**

- Shape 维度是否合法（不为负数、不超过限制）
- Dtype 是否支持
- 属性值是否在有效范围内
- 指针是否为 nullptr

---

### 8. gm内存偏移或大小必须用int64表示 `[适用: All]`

**问题说明**

涉及 GM（Global Memory）内存偏移或者大小必须用 `int64_t` 表示。GM 地址空间可能很大，使用 `int32_t` 可能导致溢出。

> **Kernel 侧说明**：Kernel 中使用 `GM_ADDR` 和 `GlobalTensor`，偏移量计算需用 `int64_t` 防止大地址溢出。

**典型溢出场景**

多维张量的 GM 偏移量在大模型场景下极易超过 `uint32_t` 上限（~4GB）：

- batch=32, heads=32, seqLen=8192, headDim=128, FP16 → 32×32×8192×128×2 = **54GB**

**隐蔽错误**

即使最终变量声明为 `int64_t`，若右侧乘法的操作数全为 `uint32_t`，乘法先以 `uint32_t` 计算并溢出，再转换为 `int64_t` 也于事无补。

**错误示例**

```cpp
// ❌ 错误：int32_t 可能溢出
int32_t totalLength = shape[0] * shape[1];  // 大 shape 可能溢出

// ❌ 隐蔽错误：赋给 int64_t，但右侧先以 uint32_t 溢出
int64_t offset = batchIdx * numHeads * seqLen * headDim;
//               ↑ 四个 uint32_t 相乘，先 overflow 再赋值，结果仍错
```

**正确示例**

```cpp
// ✅ Host侧 Tiling：使用 int64_t
int64_t totalLength = shape[0] * shape[1] * shape[2];

// ✅ Kernel侧：类成员变量使用 int64_t
int64_t blockLength_ = 0;
inputGMX.SetGlobalBuffer((__gm__ T*)x + blockLength_ * AscendC::GetBlockIdx(), blockLength_);

// ✅ 多维偏移：强转第一个操作数，后续自动提升
int64_t offset = (int64_t)batchIdx * numHeads * seqLen * headDim;

// ✅ 等效：声明变量时直接用 int64_t
int64_t batchOffset = (int64_t)batchIdx * numHeads;
int64_t offset = batchOffset * seqLen * headDim;
```

**检视规则**

表达式中有 2 个及以上维度相乘时，检查第一个操作数是否已显式转换为 `int64_t`。

---

### 9. atomic累加需src(ub)与dst(gm)清零处理 `[适用: Kernel]`

**问题说明**

atomic 累加指令需要 src（UB）与 dst（GM）做清零处理。未清零可能导致累加结果包含初始垃圾值，影响计算精度。

**正确做法**

```cpp
// 清零 src (UB)
AscendC::SetTensor<float>(srcLocal, 0, srcLength);
// 清零 dst (GM) 或确保初始值为0
AscendC::DataCopy(dstGM[0], srcLocal, dstLength);  // 先清零
// 再执行 atomic 累加
AscendC::AtomicAdd(dstGM[0], srcLocal, srcLength);
```

---

### 10. 可整数计算时不允许转浮点数计算 `[适用: All]`

**问题说明**

可整数计算时不允许转浮点数计算。整数计算更精确且性能更好，必要时应转更高精度类型而非浮点数。

> **Kernel 侧说明**：Ascend C 的浮点运算单元与整数运算单元不同，不必要的浮点转换会增加计算开销和精度损失风险。

**正确示例**

```cpp
// 累加元素数量，使用整数
int64_t totalElements = batchSize * seqLen * headNum;  // 整数计算
```

**错误示例**

```cpp
float totalElements = (float)batchSize * (float)seqLen * (float)headNum;  // 不必要的浮点转换
```

---

### 11. 通信算子融合需核间同步 `[适用: Kernel]`

**问题说明**

涉及到和通信算子融合时，多轮计算和集合通信之间需要增加核间同步。不同核的执行速度可能不同，未同步可能导致数据竞争或结果不一致。

**同步方式**

```cpp
// 使用 Ascend C 提供的同步 API
AscendC::SyncAll();  // 核间全同步
```

---

### 12. 宏定义中临时变量命名不能和外部变量冲突 `[适用: All]`

**问题说明**

宏定义中临时变量命名不能和外部变量冲突。宏展开后，临时变量可能与作用域内其他变量同名，导致意外行为。

**正确示例**

```cpp
#define COMPUTE_SUM(a, b) ({ \
    int64_t _macro_sum_result = (a) + (b); \
    _macro_sum_result; \
})
```

**错误示例**

```cpp
#define COMPUTE_SUM(a, b) ({ \
    int64_t sum = (a) + (b);  // sum 可能与外部变量冲突 \
    sum; \
})
```

---

### 13. dlopen管理的so禁用thread_local `[适用: Host]`

**问题说明**

由 `dlopen`、`dlclose` 管理的 so，禁止使用 `thread_local` 申明变量。动态加载/卸载的 so 中使用 `thread_local`，在 `dlclose` 后可能导致访问已释放内存，引发崩溃。

**原因**

`thread_local` 变量的生命周期与线程绑定，而 `dlclose` 会卸载 so 的内存。如果线程仍在运行，后续访问 `thread_local` 变量时会访问已释放的内存区域。

**正确做法**

避免在会被 `dlopen/dlclose` 管理的动态库中使用 `thread_local` 变量。