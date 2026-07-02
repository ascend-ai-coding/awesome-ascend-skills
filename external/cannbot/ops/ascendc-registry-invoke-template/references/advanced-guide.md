# AI Core算子高级开发指南

> 本指南提取高级开发知识点，完整内容请参考官方文档。

## 核心概念

### 概念关系图

```
算子原型 (OpDef)
    ├── Tiling实现 → 生成 TilingData + TilingKey + BlockDim
    ├── Kernel实现 → 使用 TilingData 执行计算
    ├── 图模式适配 → IR定义
    └── aclnn接口 → API动态库
```

### 关键术语表

| 术语 | 说明 | 交付件文件 |
|------|------|------------|
| **算子原型** | 算子输入输出属性定义 | `op_host/{op}_def.cpp` |
| **Tiling** | 数据切分、分块计算过程 | `op_host/arch*/{op}_tiling.cpp` |
| **TilingData** | 切分算法参数的数据结构 | `op_kernel/arch*/{op}_tiling_data.h` |
| **TilingKey** | 区分不同kernel实现分支的标识 | `op_kernel/arch*/{op}_tiling_key.h` |
| **BlockDim** | 核函数执行的核数 | Tiling中设置 |
| **Workspace** | Global Memory工作内存 | Tiling中设置 |
| **核函数** | AI Core上执行的函数 | `op_kernel/{op}_arch22.cpp` |
| **代际隔离** | 不同芯片架构的代码隔离 | `arch22/` `arch35/` |
| **L0 API** | 算子内部实现接口 | `op_api/{op}.cpp` `op_api/{op}.h` |
| **L2 API** | 对外暴露的ACLNN接口 | `op_api/aclnn_{op}.cpp` `op_api/aclnn_{op}.h` |

---

## 算子定义进阶

### 多硬件平台差异化注册

通过 `AddConfig` 为不同芯片配置不同参数。

**完整示例参考**：`references/add_example/op_host/add_example_def.cpp`（ascend910b/ascend950配置）

### 生效规则

- OpAICoreConfig未配置的继承OpDef定义
- OpAICoreConfig定义的会覆盖OpDef定义

---

## Tiling模板编程

> 替代传统TilingKey编程，减少对数值标识的依赖。

### 步骤1：定义模板参数头文件

在 `{op}_tiling_key.h` 中定义TilingKey：

```c++
#define ELEMENTWISE_TPL_SCH_MODE_0 0
#define ELEMENTWISE_TPL_SCH_MODE_1 1
```

**完整定义参考**：`references/add_example/op_kernel/arch22/add_example_tiling_key.h`

### 步骤2：Host侧自动配置TilingKey

使用 `GET_TPL_TILING_KEY` 宏：

```c++
context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));
```

**完整实现参考**：`references/add_example/op_host/arch22/add_example_tiling.cpp`

### 步骤3：Kernel侧模板实现

通过 `if constexpr` 实现编译期分支：

```c++
if constexpr (schMode == static_cast<uint32_t>(AddExampleTilingKey::TILING_KEY_EXAMPLE_FLOAT)) {
    NsAddExample::AddExample<float> op;
    op.Process();
}
```

**完整实现参考**：`references/add_example/op_kernel/add_example_arch22.cpp`

### 模板参数宏

| 宏 | 功能 |
|----|------|
| `ASCENDC_TPL_ARGS_DECL(op, ...)` | 定义模板参数 |
| `ASCENDC_TPL_DATATYPE_DECL(name, ...)` | DataType参数 |
| `ASCENDC_TPL_UINT_DECL(name, bw, mode, ...)` | UINT参数 |
| `ASCENDC_TPL_BOOL_DECL(name, ...)` | Bool参数 |
| `ASCENDC_TPL_SEL(...)` | 模板参数组合 |
| `ASCENDC_TPL_SEL_PARAM(context, ...)` | 自动配置TilingKey |

### UINT模式

| 模式 | 说明 |
|------|------|
| `ASCENDC_TPL_UI_RANGE` | 范围：`{0, 2}` → {0,1,2} |
| `ASCENDC_TPL_UI_LIST` | 穷举：`10, 12, 13` |
| `ASCENDC_TPL_UI_MIX` | 混合：范围+穷举 |

---

## Tiling结构体定义

### 标准C++语法（推荐）

```c++
// op_kernel/arch*/{op}_tiling_data.h
struct AddExampleTilingData {
    int64_t totalNum = 0;      // 总元素数量
    int64_t blockFactor = 0;   // 每个核处理的元素数量
    int64_t ubFactor = 0;      // 每次 UB 循环处理的元素数量
    bool enableFlag = false;   // ✅ 支持bool
    uint32_t shapeInfo[2] = {0}; // ✅ 支持数组
};
```

### Host/Kernel侧使用

**完整使用示例参考**：
- 定义：`references/add_example/op_kernel/arch22/add_example_tiling_data.h`
- Host侧：`references/add_example/op_host/arch22/add_example_tiling.cpp`
- Kernel侧：`references/add_example/op_kernel/add_example_arch22.cpp`

### 使用约束

| 约束类型 | 说明 |
|----------|------|
| ❌ 成员函数 | 不支持（含__aicore__修饰符） |
| ❌ 指针/引用 | Host无法传递到Device |
| ❌ 虚函数/虚继承 | 仅支持POD类型 |
| ❌ 模板类 | 编译问题 |
| ✅ POD类型 | 基本数据类型、数组 |

---

## BlockDim设置

### 取值范围

[1, 65535]

### 设置模式

| 模式 | 建议值 |
|------|--------|
| **耦合模式** | `GetCoreNumAiv()` 或 `GetCoreNumAic()` |
| **分离模式-Vector算子** | Vector核数（如40） |
| **分离模式-Cube算子** | Cube核数（如20） |
| **分离模式-融合算子** | 组合数（不超过物理组合核数） |

---

## Workspace设置

```c++
auto workspaceSizes = context->GetWorkspaceSizes(1);
workspaceSizes[0] = sysWorkspaceSize + usrWorkspaceSize;
```

- **sysWorkspaceSize**：通过`GetLibApiWorkSpaceSize`获取
- **usrWorkspaceSize**：算子实现使用

---

## 代际隔离

### 芯片架构映射

**芯片架构映射参考**：`npu-arch` 技能

### 隔离位置

| 位置 | 是否隔离 |
| ---- | -------- |
| ACLNN接口 | ❌ 共用 |
| IR定义 | ❌ 共用 |
| CMakeLists.txt | ✅ 芯片号列表 |
| op_host/arch22 | ✅ 隔离 |
| op_host/arch35 | ✅ 隔离 |
| op_kernel/arch22 | ✅ 隔离 |
| op_kernel/arch35 | ✅ 隔离 |

### Kernel入口配置

通过 `ExtendCfgInfo` 配置不同架构的Kernel入口：

```c++
.ExtendCfgInfo("opFile.value", "add_example_arch22");
```

**完整配置参考**：`references/add_example/op_host/add_example_def.cpp`

### 文件对应关系

| 配置值 | Kernel入口文件 | TilingData/TilingKey目录 |
| ------ | -------------- | ------------------------ |
| `{op_name}_arch22` | `{op_name}_arch22.cpp` | `op_kernel/arch22/` |
| `{op_name}_arch35` | `{op_name}_arch35.cpp` | `op_kernel/arch35/` |

> **重要**：
> - 高架构可参考低架构代码，低架构**不能**照抄高架构代码！
> - arch35 以上才支持 MicroAPI 微指令编程
> - TilingData 和 TilingKey 头文件需放在对应架构目录下

---

## 图模式适配

### IR定义

定义算子在图模式下的输入输出规格。

**完整实现参考**：`references/add_example/op_graph/add_example_proto.h`

### 接口说明

| 接口 | 说明 |
|------|------|
| `.INPUT(x, type)` | 定义输入 |
| `.OPTIONAL_INPUT(x, type)` | 可选输入 |
| `.DYNAMIC_INPUT(x, type)` | 动态输入 |
| `.OUTPUT(x, type)` | 输出 |
| `.REQUIRED_ATTR(x, type)` | 必备属性 |
| `.ATTR(x, type, default)` | 可选属性 |

### TensorType类型

```c++
TensorType::ALL();              // 所有类型
TensorType::NumberType();       // 数值类型
TensorType::RealNumberType();   // 实数类型
TensorType::IntegerDataType();  // 整数类型
TensorType::FloatingDataType(); // 浮点类型
```

---

## aclnn接口

### 自动生成配置

```cmake
# CMakeLists.txt
ACLNNTYPE aclnn
```

### 生成接口

```c++
aclnnStatus aclnn{OpName}GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnn{OpName}(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

---

## 常见问题

### Q1: Tiling结构体放在哪个目录？

**A:** 必须放在`op_kernel/`目录下，否则在线编译会失败。

### Q2: 核函数参数顺序可以调整吗？

**A:** 不可以，固定顺序：**输入 → 输出 → workspace → tiling**

### Q3: TilingData获取后需要初始化吗？

**A:** 是的，`GetTilingData<T>()`获取的结构体不包含初值，必须显式赋值。

### Q4: 算子输入输出同名怎么处理？

**A:** 输出参数加`ref`后缀，如输入`x`，输出参数为`x_ref`。

---

## 约束条件汇总

### Tiling结构体约束

| 约束项 | 要求 |
|--------|------|
| 成员函数 | ❌ 不支持 |
| 指针/引用 | ❌ 不支持 |
| 虚函数/虚继承 | ❌ 不支持 |
| 模板类 | ❌ 不支持 |
| 数据类型 | 仅POD类型 |

### 核函数约束

| 约束项 | 要求 |
|--------|------|
| 参数顺序 | 固定：输入→输出→workspace→tiling |
| 修饰符 | 必须包含 `extern "C" __global__ __aicore__` |
| 文件位置 | 必须在 `op_kernel/` 目录下 |

### BlockDim约束

| 约束项 | 要求 |
|--------|------|
| 取值范围 | [1, 65535] |
| 融合算子 | 不超过物理组合核数 |

---

## 快速参考

关键API列表（详细用法见各章节）：

**算子定义**：
- Input(), Output() - 输入输出定义
- AddConfig() - 芯片配置
- ExtendCfgInfo() - Kernel文件名映射

**Tiling**：
- SetTilingKey() - 设置调度模式
- SetBlockDim() - 设置核数
- GetTilingData() - 获取TilingData

**Kernel**：
- REGISTER_TILING_DEFAULT - 注册TilingData
- GET_TILING_DATA_WITH_STRUCT - 获取TilingData

**完整API文档**：参考官方API文档
