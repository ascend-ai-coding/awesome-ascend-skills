# AI Core算子基础开发指南

> 本指南提取核心知识点，完整内容请参考官方文档。以 `AddExample` 算子为例说明。

## 工程创建

### 目录创建命令

参照 `references/add_example/` 的完整模板，按以下目录结构创建算子工程。
将模板中的 `add_example` 替换为算子的 snake_case 名，`AddExample` 替换为大驼峰类名。
各文件的实现细节参见本指南后续章节。

### 标准目录结构

```
${op_name}/
├── CMakeLists.txt              # 构建配置（必需）
├── README.md                   # 算子说明
├── op_host/                    # Host侧实现（必需）
│   ├── {op}_def.cpp           # 算子定义（必需）
│   ├── {op}_infershape.cpp    # Shape推导
│   ├── CMakeLists.txt
│   ├── arch22/                 # Ascend910B架构
│   │   └── {op}_tiling.cpp    # Tiling实现
│   └── arch35/                 # Ascend950架构
│       └── {op}_tiling.cpp
├── op_kernel/                  # Kernel侧实现（必需）
│   ├── CMakeLists.txt
│   ├── {op}_arch22.cpp        # Kernel入口（arch22）
│   ├── {op}_arch35.cpp        # Kernel入口（arch35）
│   ├── arch22/                 # Ascend910B实现
│   │   ├── {op}.h             # Kernel类定义
│   │   ├── {op}_tiling_data.h # TilingData结构体
│   │   └── {op}_tiling_key.h  # TilingKey定义
│   └── arch35/                 # Ascend950实现
│       ├── {op}.h
│       ├── {op}_tiling_data.h
│       └── {op}_tiling_key.h
├── op_api/                     # ACLNN接口
│   ├── aclnn_{op}.cpp         # L2 API实现
│   ├── aclnn_{op}.h           # L2 API头文件
│   ├── {op}.cpp               # L0 API实现
│   └── {op}.h                 # L0 API头文件
├── op_graph/                   # 图模式适配
│   └── {op}_proto.h
└── tests/                      # 测试代码
    ├── ut/                     # 单元测试
    │   ├── common/             # 公共测试工具
    │   ├── op_host/            # Host侧UT
    │   └── op_api/             # API侧UT
    └── st/                     # 系统测试
```

---

## 算子定义

### 完整示例

算子定义涉及以下关键配置：
- 输入输出参数定义（Input/Output）
- 芯片配置（AddConfig）
- ExtendCfgInfo（Kernel文件名映射）

**完整实现参考**：`references/add_example/op_host/add_example_def.cpp`

> **注意**：
> - `ExtendCfgInfo("opFile.value", ...)` 配置的值对应 kernel 入口文件名（不含 `.cpp` 后缀）
> - Tiling 和 InferShape 函数在独立文件中通过 `IMPL_OP_OPTILING` 和 `IMPL_OP_INFERSHAPE` 注册

### 输入输出参数

| 参数 | 说明 | 取值 |
|------|------|------|
| ParamType | 参数类型 | REQUIRED（必选）、OPTIONAL（可选）、DYNAMIC（动态） |
| DataType | 数据类型 | ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32 等 |
| Format | 数据格式 | ge::FORMAT_ND 等 |

### 简化写法：Follow接口

```c++
// 输出跟随输入属性
this->Output("y")
    .Follow("x");  // dtype、format、shape与x一致
```

### 简化写法：DataTypeList/FormatList

```c++
// 某个输入支持所有组合
this->Input("x")
    .DataTypeList({ge::DT_FLOAT16})  // 与其他输入所有dtype组合
    .FormatList({ge::FORMAT_ND});    // 与其他输入所有format组合
```

---

## Tiling实现

### Tiling简介

将输入张量切分为多个小块（Tile），逐块进行计算的过程。Tiling实现计算切分参数，传递给Kernel使用。

### 交付件

1. `{op}_tiling.cpp` - Tiling主要逻辑（op_host目录）
2. `{op}_tiling_key.h` - TilingKey定义（op_kernel目录）
3. `{op}_tiling_data.h` - TilingData结构体（op_kernel目录）

### TilingData结构体定义

```c++
// op_kernel/arch*/{op}_tiling_data.h
struct AddExampleTilingData {
    int64_t totalNum = 0;     // 总元素数量
    int64_t blockFactor = 0;  // 每个核处理的元素数量
    int64_t ubFactor = 0;     // 每次UB循环处理的元素数量
};
```

### TilingFunc实现

Tiling流程包括：
1. 获取平台信息（coreNum, ubSize）
2. 获取输入信息（shape, dtype）
3. 计算切分参数（blockFactor, ubFactor）
4. 设置TilingData、BlockDim、TilingKey

**关键配置示例**：
```c++
context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));
```

**完整实现参考**：`references/add_example/op_host/arch22/add_example_tiling.cpp`

### TilingKey定义

```c++
// op_kernel/arch*/{op}_tiling_key.h
#include "ascendc/host_api/tiling/template_argument.h"

#define ELEMENTWISE_TPL_SCH_MODE_0 0  // FLOAT类型
#define ELEMENTWISE_TPL_SCH_MODE_1 1  // INT32类型

ASCENDC_TPL_ARGS_DECL(
    AddExample,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, 
                          ELEMENTWISE_TPL_SCH_MODE_0, ELEMENTWISE_TPL_SCH_MODE_1));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, 
                             ELEMENTWISE_TPL_SCH_MODE_0, ELEMENTWISE_TPL_SCH_MODE_1)));
```

---

## Kernel实现

### 核函数模板

核函数参数顺序固定：
```c++
template <uint32_t schMode>
__global__ __aicore__ void {op_name}(
    GM_ADDR input1,   // 输入参数
    GM_ADDR input2,
    GM_ADDR output,   // 输出参数
    GM_ADDR workspace, // workspace（固定）
    GM_ADDR tiling     // tiling（固定）
)
```

**完整实现参考**：`references/add_example/op_kernel/add_example_arch22.cpp`

> **注意**：
> - 参数顺序固定为：**输入 → 输出 → workspace → tiling**
> - 使用 `REGISTER_TILING_DEFAULT` 注册 TilingData 结构体
> - 使用 `GET_TILING_DATA_WITH_STRUCT` 获取 Tiling 数据

### Kernel类结构

包含Init()和Process()方法：
- **Init()**: 初始化输入输出tensor、tiling data
- **Process()**: 执行计算逻辑

**完整实现参考**：
- Kernel类：`references/add_example/op_kernel/arch22/add_example.h`
- 调用示例：`references/add_example/op_kernel/add_example_arch22.cpp`

### Kernel执行流程

```
核函数定义 → Init初始化 → Process主循环
                          ↓
               CopyIn → Compute → CopyOut（循环执行）
```

### InferShape实现

根据输入shape推导输出shape。

**完整实现参考**：`references/add_example/op_host/add_example_infershape.cpp`

---

## 编译部署

> 详细指南见 `references/build-deploy-guide.md`

### 常用命令

```bash
bash build.sh --soc=ascend910b -j8   # 编译指定芯片
bash build.sh -u --soc=ascend910b    # 编译 + UT测试
bash build.sh -s --soc=ascend910b    # 编译 + ST测试
bash build.sh --make_clean           # 清理构建目录
```

### 安装

```bash
./build/custom_opp_ubuntu_aarch64.run
```

---

## UT验证

### 目录结构

```
tests/ut/
├── common/                        # 公共测试工具
│   ├── tiling_context_faker.h
│   ├── tiling_case_executor.h
│   ├── infershape_context_faker.h
│   └── infershape_case_executor.h
├── op_host/
│   ├── test_{op}_tiling.cpp       # Tiling UT
│   ├── test_{op}_infershape.cpp   # InferShape UT
│   └── test_op_host_main.cpp      # UT入口
└── op_kernel/
    └── test_{op}.cpp              # Kernel UT
```

### Tiling UT

验证Tiling逻辑正确性：
- 平台信息获取
- 切分参数计算
- TilingKey设置

**完整实现参考**：`references/add_example/tests/ut/op_host/`

### Kernel UT

验证Kernel计算正确性：
- 数据搬运
- 计算结果
- 精度验证

**完整实现参考**：
- UT框架：`references/add_example/tests/ut/common/`
- 测试用例：`references/add_example/tests/ut/op_kernel/`

### 运行命令

```bash
# 从项目根目录
bash build.sh -u

# 从 tests/ut 目录
cd tests/ut && ./run.sh
```

### run.sh 选项

| 选项 | 说明 |
|------|------|
| `-c, --clean` | 清理后编译（默认） |
| `--clean false` | 增量编译 |
| `-v, --verbose` | 详细输出 |

---

## 命名规范

### 算子类型与文件名转换

| 算子类型（大驼峰） | 实现文件名/核函数名（下划线） |
|-------------------|------------------------------|
| AddCustom | add_custom |
| BatchNorm | batch_norm |
| Conv2DBackprop | conv2d_backprop |
| ReduceMax | reduce_max |

### 转换规则

1. 首字符大写转小写：`Abc` → `abc`
2. 大写前为小写/数字，插入下划线：`AbcDef` → `abc_def`
3. 大写前为大写且后为小写，插入下划线：`AbcAAc` → `abc_a_ac`
