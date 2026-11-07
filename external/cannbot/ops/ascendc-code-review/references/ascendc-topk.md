# Ascend C TOPK 问题清单

<适用>
语言: C++
侧别: All
领域: true
触发: 必须触发
默认启用: true

适用场景: Ascend C 算子开发中实际高频出现的编码问题，来源于生产环境经验总结
介绍: Ascend C TOPK 高频问题清单，12 条条款覆盖算子开发中最常触发的错误模式
类别(All): 函数返回值处理、野指针(局部变量指针生命周期)、特殊值(nan/inf/+0/-0边界处理)、整数与浮点计算选择、宏定义命名冲突
类别(Host): Dtype/Shape获取(GetInputDesc+context)、属性获取(context获取禁止CompileInfo传递)、外部输入校验(融合规则/InferShape/Tiling)、dlopen管理(thread_local禁用)、SyncAll与BATCH_MODE_SCHEDULE绑定
类别(Kernel): atomic累加(src(ub)与dst(gm)双清零)、全核同步禁止BlockDim限制
> **说明**：TOPK 问题是检视实践中发现的高频风险点，需重点关注。条款标注适用范围：`[适用: All]` / `[适用: Host]` / `[适用: Kernel]`。条款保留原编号，删除条款的编号不复用。
</适用>


## 快速索引

### 两者都适用 `[适用: All]`（5 条）

| 序号 | 问题类型 | 类别 | 严重级别 |
|-----|---------|------|---------|
| TOPK-1 | 必须按语义处理函数返回值 | 编码规范 | 高 |
| TOPK-3 | 生命周期内使用局部变量指针，避免野指针 | 内存安全 | 高 |
| TOPK-6 | 必须考虑nan/inf/+0/-0等特殊值和边界值处理 | 数值安全 | 高 |
| TOPK-9 | 可整数计算时不允许无依据地转浮点数计算 | 数值安全 | 中 |
| TOPK-11 | 宏定义中临时变量命名不能和外部变量冲突 | 编码规范 | 中 |

### 仅 Host 侧适用 `[适用: Host]`（5 条）

| 序号 | 问题类型 | 类别 | 严重级别 |
|-----|---------|------|---------|
| TOPK-2 | 使用GetInputDesc获取Dtype，context获取Shape | API使用 | 高 |
| TOPK-4 | 属性从context获取，禁止CompileInfo传递 | API使用 | 高 |
| TOPK-7 | 融合规则/InferShape/Tiling外部输入校验 | 输入验证 | 高 |
| TOPK-12 | dlopen管理的so禁用thread_local | 并发安全 | 高 |
| TOPK-13 | 在使用SyncAll的场景，需要使用SetScheduleMode设置BATCH_MODE_SCHEDULE | 并发安全 | 高 |

### 仅 Kernel 侧适用 `[适用: Kernel]`（2 条）

| 序号 | 问题类型 | 类别 | 严重级别 |
|-----|---------|------|---------|
| TOPK-8 | atomic累加需src(ub)与dst(gm)清零处理 | 内存安全 | 高 |
| TOPK-14 | 调用全核同步API（SyncAll、CrossCoreWaitFlag）时禁止做BlockDim的限制 | 并发安全 | 高 |

---

## 详细规范

### TOPK-1. 必须按语义处理函数返回值 `[适用: All]`

**严重级别**：高

> **交叉引用**：指针返回值的使用要求，参见 ascendc-red-line.md 条例6。

**问题说明**

Host 和 Kernel 侧调用函数时，必须按照接口契约处理表示执行状态、有效性、实际处理数量或资源获取结果的返回值，避免操作失败后继续使用无效结果。处理方式不局限于显式 `if` 判断，也可以是向调用方传播错误、由统一错误处理机制消费状态，或按接口约定执行重试、回退和清理。

**具体要求**

- 对状态码、布尔成功标记、可空指针、可失败的句柄等，必须在依赖该结果的操作前处理失败情形。
- 对返回实际读取、写入或处理数量的接口，必须处理不足预期的情况，不能只确认返回值非零。
- 允许包装函数直接返回状态，由调用方处理；仅赋值、强制转换为 `void` 或记录日志后继续执行依赖成功结果的操作，不构成有效处理。
- 返回 `void` 的接口不适用返回值检查；普通计算结果无需机械增加错误判断，但使用结果时仍须满足数值范围、索引和内存访问要求。Kernel 侧按实际接口契约处理，不为没有状态返回值的指令虚构检查。

**场景示例**

- 合规：资源获取失败后停止依赖该资源的操作并清理已获得的资源；包装函数将失败状态返回，由上层阻止后续执行。
- 不合规：接口返回失败或仅完成部分操作，调用方忽略该结果，继续读取未生成的数据或使用无效资源。

---

### TOPK-2. 使用GetInputDesc获取Dtype，context获取Shape `[适用: Host]`

**严重级别**：高

**问题说明**

Host 侧应使用 `GetInputDesc` 返回的输入描述获取 Dtype，使用当前阶段 context 提供的对应 Shape 接口获取 Shape。不得通过 `GetInputTensor` 获取这些元数据；不能假设在融合规则、InferShape、Tiling 等不同阶段都存在可用的输入 Tensor 数据。

**具体要求**

- 按当前 context 的接口契约选择输入索引及获取方式，区分必选、可选和动态输入；在使用返回对象前处理缺失或无效结果。
- 获取 Shape 时确认当前计算需要的是逻辑 Shape 还是存储 Shape，不能混用。不同阶段使用对应 context 接口，不假设所有 context 的接口和返回类型一致。
- 获取元数据与读取 Tensor 内容分开处理；仅当业务确实需要输入值且当前阶段支持读取时，才使用相应的数据访问接口。

**场景示例**

- 合规：先从 `GetInputDesc` 获取并确认输入描述有效，再读取 Dtype；通过当前 context 的 Shape 接口取得符合当前计算语义的 Shape，并处理其有效性。
- 不合规：为了取得 Dtype 或 Shape 而调用 `GetInputTensor`，并假定所有编译或推导阶段都能取得有效 Tensor。

---

### TOPK-3. 生命周期内使用局部变量指针，避免野指针 `[适用: All]`

> **交叉引用**：未初始化导致的野指针检视策略，参见 ascendc-red-line.md 条例5；指针判空策略参见 ascendc-red-line.md 条例6。

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

### TOPK-4. 属性从context获取，禁止CompileInfo传递 `[适用: Host]`

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

### TOPK-6. 必须考虑nan/inf/+0/-0等特殊值和边界值处理 `[适用: All]`

> **交叉引用**：除零要求参见 ascendc-red-line.md 条例1。

**问题说明**

Host 和 Kernel 侧必须按照算子契约处理 `NaN`、正负无穷、正负零以及目标数据类型的边界值。处理结果应与算子规定的传播、忽略、饱和、舍入或报错语义一致，不得一律拒绝特殊值，也不能默认所有硬件指令与参考实现行为相同。

**关注场景**

- 比较、排序及最值归约：明确 `NaN` 的传播或忽略规则；涉及相等元素或正负零时，遵守结果和索引的选择规则，不能用普通大小比较代替完整语义。
- 算术及数学函数：覆盖零分母、无穷参与运算，以及开方、对数等定义域和边界；区分契约允许产生的特殊结果与错误输入。
- 类型转换：检查有限值超出目标范围、`NaN`/无穷转整数及精度缩减时的行为，确保符合目标接口的转换、舍入和饱和约定。
- 精度与极值：关注最大有限值、最小正常值及非正规数；若算法依赖正负零的符号或非正规数，确认所用指令和数据类型满足该要求。

仅对会影响本算子结果的特殊值建立处理与验证要求；若契约明确排除了某类输入，应能找到对应的输入约束或上游保证。

---

### TOPK-7. 融合规则/InferShape/Tiling外部输入校验 `[适用: Host]`

> **交叉引用**：除零要求参见 ascendc-red-line.md 条例1；内存操作长度要求参见 cpp-secure.md SEC-2.1。

**问题说明**

融合规则、InferShape 和 Tiling 必须在使用外部输入前，按当前阶段和算子契约验证输入及其组合约束；校验失败后不得继续执行依赖该输入的推导、计算或配置。

**校验内容**

- 输入存在性与有效性：必选输入是否齐全、可选输入缺失是否被处理、输入索引是否有效，以及描述、Shape、属性等对象是否可用。
- Shape 与 Dtype：秩、维度范围、类型及格式是否受支持；多输入的广播、维度对应和类型组合是否满足要求。
- 阶段约束：未知秩、未知维度、零维 Tensor 和含零长度维度按当前阶段契约处理。不能将空 Shape 一律视为非法，也不能将推导阶段允许的未知维直接用于必须依赖具体尺寸的分配或 Tiling 计算。
- 属性与组合条件：轴、数量、步长、模式等是否在有效范围内；属性与 Shape、多个输入及所选实现分支之间是否一致。
- 派生数值：元素总数、字节数、偏移、分块数和工作区大小的计算是否存在除零、溢出、窄化或越界；通过单项输入检查不能替代对派生值的检查。

允许使用能够证明适用于当前路径的上游校验结果；若输入经过转换或分支处理，须确认原有保证仍然成立。融合条件不满足时应放弃该融合，推导或 Tiling 条件不满足时按接口约定返回失败或采取合法回退。

---

### TOPK-8. atomic累加需src(ub)与dst(gm)清零处理 `[适用: Kernel]`

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

### TOPK-9. 可整数计算时不允许无依据地转浮点数计算 `[适用: All]`

**问题说明**

元素数量、索引、偏移、字节数、分块数等要求精确整数语义的计算，应使用能够容纳输入和中间结果的整数类型，并检查溢出。不得为简化取整、绕过整数溢出或未经验证的性能收益而转为浮点数计算后再转回整数。

**允许的例外**

- 数学定义要求实数运算，如比例、归一化、插值或带指定舍入规则的实数计算；使用浮点数时仍须遵守算子精度与特殊值契约。
- 所调用接口明确要求浮点参数或结果；在接口边界转换，并验证可表示范围及所需精度，不将该需求扩散到无关的整数计算。
- 针对目标硬件的浮点优化已有可核验依据：证明覆盖的输入范围、所有中间结果、舍入误差及转回整数的行为满足要求，并提供性能收益依据。对索引、分配尺寸等精确结果，必须证明与规定的整数结果一致；近似相等不足以构成例外。

**边界要求**

使用更宽整数类型时须在运算发生前完成必要的类型提升，不能在窄类型已经溢出后再赋给宽类型。浮点转整数前应保证值有限且处于目标类型可表示范围，并满足所需的取整方式。

**场景示例**

- 合规：使用经过范围验证的整数运算计算元素总数；按算子数学定义使用浮点数计算缩放比例。
- 不合规：将尺寸转换为 `float` 后相乘再转为整数，未证明精度与范围；用浮点除法和取整计算分块数，未证明其在全部合法尺寸上与精确整数结果一致。

---

### TOPK-11. 宏定义中临时变量命名不能和外部变量冲突 `[适用: All]`

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

### TOPK-12. dlopen管理的so禁用thread_local `[适用: Host]`

**问题说明**

项目禁止在由 `dlopen`、`dlclose` 管理的 so 中使用 `thread_local` 变量。检视应覆盖动态库构建和实例化后实际包含的声明，而非仅搜索当前源文件的全局变量。

**适用范围**

- 命名空间或全局作用域的 `thread_local` 变量、函数内线程局部变量、类的 `static thread_local` 成员。
- 模板实例、宏展开、头文件定义及内联函数中引入的线程局部变量；根据实际构建配置判断是否进入目标 so。
- 本项目可控制并编入或链接到目标动态加载单元的实现代码。依赖库中发现相关声明时，应核实加载关系及责任边界，不能仅因存在依赖就认定目标 so 违规，也不能遗漏本项目可控制的封装和实例化代码。

**生命周期风险**

线程局部对象的生命周期与线程关联，其析构函数、注册的清理回调以及对象持有的指针或资源可能依赖 so 中的代码和数据。动态库关闭、线程退出及后续回调之间若缺少可靠的生命周期协调，可能产生悬空访问或清理顺序问题。

`dlclose` 成功不保证动态库立即卸载；是否实际卸载还受引用和运行时状态等影响。不得以“每次 `dlclose` 都会立即卸载”解释本规则，也不得以某次运行未卸载作为解除项目禁令的依据。

**正确做法**

- 使用显式传入、所有权清晰的上下文或状态对象，明确其创建、使用及销毁责任。
- 关闭动态库前，停止新的调用，等待已有调用及异步任务结束，并按约定清理相关状态、注销回调，确保后续线程退出或回调不会再进入待卸载代码。
- 不得简单将 `thread_local` 改成共享全局或静态变量；替代实现须同时满足隔离、并发安全及生命周期要求。

**判定要求**

确认目标 so 包含上述 `thread_local` 声明或实例时判定 `FAIL`；构建归属或依赖责任不明确时判定 `SUSPICIOUS`，注明需要核实的构建与加载证据。


**参考**：[Linux dlclose 接口说明](https://www.man7.org/linux/man-pages/man3/dlclose.3.html)。项目禁令按本条执行，接口说明用于理解卸载语义。

---

### TOPK-13 在使用SyncAll的场景，需要使用SetScheduleMode设置BATCH_MODE_SCHEDULE `[适用: Host]`

**【描述】**

工程化算子中，若某次 Kernel 启动实际可能执行 `SyncAll`，对应 Host 侧 Tiling 成功路径必须通过 `SetScheduleMode` 设置 `BATCH_MODE_SCHEDULE`。该配置必须作用于同一次启动所使用的上下文；缺少或错误的调度配置可能在多流场景导致卡死。

**【检视范围】**

- 从实际 Kernel 入口追踪 tiling key、模板参数及运行时分支，确认哪些启动路径可能执行 `SyncAll`；同时检查辅助函数、公共封装、宏及模板实例中的调用。
- 将这些 Kernel 路径映射到对应 Host 侧 Tiling 分支及启动上下文，不能以其他算子、其他分支或其他上下文中存在一次配置作为证据。
- 覆盖所有能够成功返回并启动相关 Kernel 的路径，包括辅助函数返回后的分支、提前成功返回及公共收尾路径。
- 确认设置在启动前生效，且未被后续默认值设置、上下文替换或其他分支覆盖。
- 按当前接口约定处理设置操作的返回结果；接口提供失败状态时，失败路径不得继续按配置成功启动 Kernel。不得虚构接口返回类型或错误码。

**【判定要求】**

- 已确认某个可执行 `SyncAll` 的启动路径未设置 `BATCH_MODE_SCHEDULE`、设置在错误上下文中、被覆盖，或设置失败后仍继续启动，判定 `FAIL`。
- 所有相关成功启动路径均已配置正确模式，且配置在启动时有效，判定 `PASS`。
- 无法确定入口、分支、封装行为或上下文对应关系时，判定 `SUSPICIOUS`，明确缺少的证据；不能仅凭源码中同时出现两个 API 判定通过。
- 经实际构建及分支条件证明不会执行 `SyncAll` 的路径，不因未设置该模式而触发本条。

**【调用方式说明】**

本条的 `SetScheduleMode` 要求适用于工程化算子的 Host 侧 Tiling 流程。直调 Kernel 场景应按对应版本接口文档使用 `__schedmode__` 配置所需调度模式，不为直调路径虚构 `TilingContext` 或套用不存在的设置流程。

调度模式正确不能替代全核同步参与要求；同步参与核及提前退出问题另见第 14 条。


**【Host 配置示意】**

```cpp
// 放在相关 Tiling 分支的成功返回之前；context 已确认有效。
// 此处是项目命名常量，值 1 表示 batchmode，不假设 SDK 提供同名符号。
constexpr uint32_t BATCH_MODE_SCHEDULE = 1U;
const auto ret = context->SetScheduleMode(BATCH_MODE_SCHEDULE);
if (ret != ge::GRAPH_SUCCESS) {
    return ret;
}
// 后续流程不得覆盖此设置；该 context 必须对应将执行 SyncAll 的 Kernel。
```

**参考**：[SetScheduleMode 接口说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/maintenref/basicdataapi/atlasopapi_07_00248.html)、[SyncAll 接口说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta1/API/ascendcopapi/atlasascendc_api_07_0204.html)。具体调用方式应与项目使用的 CANN 版本匹配。

---

### TOPK-14 调用全核同步API（SyncAll、CrossCoreWaitFlag）时禁止做BlockDim的限制 `[适用: Kernel]`

**问题说明**

全核同步不得被 `BlockDim`、有效核数、核索引或其他只让部分参与核进入的条件包围。实际启动核数大于处理数据的有效核数时，无数据核也必须按同步协议参与，不能在同步前直接返回。

- 检查入口、辅助函数、条件分支和循环，确保同步域内各参与核按协议完成对应的同步轮次；限制数据计算的分支不能跳过所需同步。
- 检查 `CrossCoreWaitFlag` 时，应结合对应的置位操作、同步模式、标志及参与核范围确认配对；不能只把 wait 移到分支外而遗漏发送端。
- 核范围限制应放在数据计算或搬运上。全核同步的参与范围按实际接口协议确定；不能把特定核间握手机械改成所有核执行相同调用。

**示意：数据分支与全核同步分开**

```cpp
// usedCoreNum 为需要处理数据的核数；实际启动的参与核可能更多。
// 此处假定所有参与核均执行这一轮 SyncAll，Host 调度配置见第 13 条。
if (AscendC::GetBlockIdx() < usedCoreNum) {
    // 完成本轮数据计算或搬运，并满足相应流水同步要求。
}
AscendC::SyncAll<false>();
// 无数据核不得在上面的 SyncAll 前提前返回。
```
