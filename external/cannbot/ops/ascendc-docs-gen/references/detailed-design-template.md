# {算子名称} 设计文档

## 1. 概述

### 1.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | {算子名称} |
| 算子类别 | {Reduction / Elementwise / Broadcast / Conversion / MatMul / ...} |
| 支持数据类型 | {fp16 / fp32 / bf16 / ...} |
| **目标芯片** | {Ascend910B / Ascend910_93 / Ascend950DT / Ascend910PR} |
| **目标架构** | {arch22 / arch35} |

### 1.2 算子功能
{描述算子的功能}

### 1.3 数学公式
{数学公式定义}

---

## 2. 架构设计

### 2.1 逻辑视图

**模块职责**：

| 模块 | 职责 | 核心文件 |
|------|------|---------|
| **op_api** | ACLNN 接口层：对外暴露算子调用接口，处理输入校验、类型转换、内存管理 | `aclnn_${op}.h`, `aclnn_${op}.cpp`, `${op}.h`, `${op}.cpp` |
| **op_host** | Host 侧逻辑：算子定义、Tiling 切分、Shape 推导 | `_def.cpp`, `_tiling.cpp`, `_tiling.h`, `_infershape.cpp` |
| **op_kernel** | Kernel 侧实现：NPU 计算逻辑 | `.cpp`, `.h`, `_tiling_key.h`, `_tiling_data.h` |
| **op_graph** | 图模式适配，定义算子 IR 原型和类型推导 | `_proto.h`, `_graph_infer.cpp` |

**模块依赖**：

```
op_api (ACLNN接口层)
├── aclnn${Op}GetWorkspaceSize() ──▶ op_host (Tiling)
└── aclnn${Op}() ──▶ op_kernel (Kernel执行)
```

### 2.2 开发视图

```
${op_name}/
├── CMakeLists.txt              # 构建配置
├── README.md                   # 算子说明
├── examples/                   # 调用示例
│   └── test_aclnn_${op_name}.cpp
├── op_host/                    # Host侧实现
│   ├── ${op_name}_def.cpp     # 算子原型注册
│   ├── ${op_name}_infershape.cpp # Shape推导
│   ├── arch22/                 # Ascend910B/Ascend910_93 (DAV_2201)
│   │   ├── ${op_name}_tiling.cpp  # Tiling实现
│   │   ├── ${op_name}_tiling.h    # Tiling数据结构定义
│   └── arch35/                 # Ascend950DT/Ascend910PR (DAV_3510)
├── op_kernel/                  # Kernel侧实现
│   ├── ${op_name}_arch22.cpp  # Kernel入口
│   ├── ${op_name}_arch35.cpp  # Kernel入口
│   ├── arch22/                 # Ascend910B/Ascend910_93 (DAV_2201)
│   │   ├── ${op_name}.h
│   │   ├── ${op_name}_tiling_key.h
│   │   └── ${op_name}_tiling_data.h
│   └── arch35/                 # Ascend950DT/Ascend910PR (DAV_3510)
├── op_api/                     # ACLNN接口
│   ├── aclnn_${op_name}.h
│   ├── aclnn_${op_name}.cpp
│   ├── ${op_name}.h
│   └── ${op_name}.cpp
├── op_graph/                   # 图模式适配（可选）
│   ├── ${op_name}_proto.h
│   └── ${op_name}_graph_infer.cpp
└── tests/                      # 测试代码
    ├── ut/                     # 单元测试
    └── st/                     # 集成测试
```

### 2.3 运行视图

**数据流**：

```
GM (Global Memory)
  │
  │ DataCopy (GM → UB)
  ▼
UB (Unified Buffer)
  │
  │ 计算 (ALU/Vector/Cube)
  ▼
UB (Unified Buffer)
  │
  │ DataCopy (UB → GM)
  ▼
GM (Global Memory)
```

**执行流程**：

```
aclnn${OpName}GetWorkspaceSize()
     │
     ├──▶ op_host/_def.cpp: 获取算子定义
     ├──▶ op_host/_tiling.cpp: 计算 Tiling 参数 → 生成 TilingData
     │
     ▼
aclnn${OpName}()
     │
     ├──▶ 加载 Kernel 二进制
     ├──▶ 传递 TilingData 到 Device
     │
     ▼
NPU Device 执行
     ├──▶ 读取 TilingData
     ├──▶ GM → UB 数据搬运
     ├──▶ 执行计算逻辑
     └──▶ UB → GM 结果写回
```

### 2.4 用户视图

| 调用方式 | 说明 |
|---------|------|
| **ACLNN 调用** | 通过 aclnn 接口直接调用 |
| **图模式调用** | 通过 GE 图引擎调用 |

**典型使用流程**：

```
ACLNN调用: aclnn${OpName}GetWorkspaceSize() → 分配workspace → aclnn${OpName}()执行Kernel
图模式调用: 定义IR图 → 编译优化 → 执行计算图
```

---

## 3. 实现方案

### 3.1 模板划分总览

> **TilingKey 机制**：参考 `ascendc-custom-op-enhance` 技能的 `advanced-guide.md` 第47-103行

**模板参数定义**：

| 参数 | 类型 | 取值范围 | 说明 |
|-----|------|---------|------|
| D_T_X | DataType | {half, float} | 输入X数据类型 |
| TILE_NUM | UINT | {1, 4, 8} | 切分数量 |
| IS_SPLIT | BOOL | {false, true} | 是否切分 |

**模板划分表**：

| 模板 | 触发条件 | 模板参数 | 适用场景 |
|-----|---------|---------|---------|
| 模板一 | {条件描述，如: float + small_shape} | `D_T_X=float, TILE_NUM=1, IS_SPLIT=false` | 基础计算路径 |
| 模板二 | {条件描述，如: half + large_shape} | `D_T_X=half, TILE_NUM=8, IS_SPLIT=true` | 优化计算路径 |
| 模板三 | {条件描述} | {...} | 特殊场景A |
| ... | ... | ... | ... |

### 3.2 TilingData 结构体

**文件位置**：`op_kernel/arch*/${op}_tiling_data.h`

```cpp
struct ${op_name}TilingData {
    uint64_t totalLength = 0;   // 总数据长度
    uint32_t tileLength = 0;    // 单次处理长度
    // {其他字段}
};
```

---

### 3.3 模板一：{模板名称，如：基础计算路径}

#### 3.3.1 触发条件
{条件描述，如: input.dtype == float && totalLength < UB_THRESHOLD}

#### 3.3.2 Host 侧 Tiling

**文件**: `op_host/arch*/${op_name}_tiling.cpp`

```cpp
#include "tiling_key_${op_name}.h"

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 获取输入信息
    auto inputDescX = context->GetInputDesc(0);
    ge::DataType dtype_x = inputDescX->GetDataType();
    uint64_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    
    // 计算模板参数
    uint32_t D_T_X = static_cast<int>(dtype_x);
    uint32_t TILE_NUM = 1;
    uint32_t IS_SPLIT = 0;
    
    // 设置 TilingKey
    ASCENDC_TPL_SEL_PARAM(context, D_T_X, TILE_NUM, IS_SPLIT);
    
    // 设置 TilingData, 空间由kernel侧的REGISTER_TILING_DEFAULT决定
    auto tiling = context->GetTilingData<${op_name}TilingData>();
    tiling->totalLength = totalLength;
    tiling->tileLength = TILE_LENGTH;
    
    return ge::GRAPH_SUCCESS;
}
```

#### 3.3.3 Kernel 侧模板实例化

**文件**: `op_kernel/${op_name}_arch*.cpp`

```cpp
#include "tiling_key_${op_name}.h"

template<typename D_T_X, int TILE_NUM, int IS_SPLIT>
__global__ __aicore__ void ${op_name}_template(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(${op_name}TilingData);
    GET_TILING_DATA_WITH_STRUCT(${op_name}TilingData, tilingData, tiling);
    
    Kernel${Op}<D_T_X> op;
    op.Init(x, y, z, tilingData.totalLength, TILE_NUM);
    
    if constexpr (std::is_same_v<D_T_X, float>) {
        op.Process1();
    } else if constexpr (std::is_same_v<D_T_X, half>) {
        if constexpr (IS_SPLIT == 0) { op.Process1(); }
        else { op.Process2(); }
    }
}
```

#### 3.3.4 API 映射

| 计算步骤 | Ascend C API | 参数签名 | 平台验证 | 约束说明 | 替代方案 |
|---------|-------------|---------|---------|---------|---------|
| 数据搬入 | DataCopyPad\<T\> | (dst, src, DataCopyExtParams, DataCopyPadExtParams) | ✅ {DAV_2201/DAV_3510} | GM→UB: stride=bytes; 32B对齐 | - |
| {计算步骤} | {API\<T, Template\>} | (完整参数列表) | ✅/❌ {DAV_2201/DAV_3510} | {对齐/类型/尺寸约束} | {不可用时的替代} |
| 数据搬出 | DataCopyPad\<T\> | (dst, src, DataCopyExtParams) | ✅ {DAV_2201/DAV_3510} | UB→GM 无 padding 参数 | - |

> **填写说明**：
> - **参数签名**：必须包含完整的参数列表和模板参数
> - **平台验证**：标注 ✅ 或 ❌，以及适用的芯片平台（DAV_2201/DAV_3510）
> - **约束说明**：记录对齐要求、数据类型限制、尺寸限制等
> - **替代方案**：当 API 不可用时的替代实现方式

#### 3.3.5 API 验证记录

> ⚠️ **以下每个 API 必须已通过 `/ascendc-docs-search` skill 查阅 `{API名称}` API 文档验证。**
> 验证方法: 使用 `/ascendc-docs-search` skill 搜索 API 名称，或在 `$ASC_DEVKIT_DIR` 下使用通配符搜索 `find "$ASC_DEVKIT_DIR/docs/api/" -name "{APIName}*.md" -type f`

| API 名称 | 官方文档路径 | 通配符搜索结果 | 验证状态 | 备注 |
|---------|-------------|---------------|---------|------|
| {API名称} | docs/api/ 下的 API 文档（find 搜索） | {文件列表} | ✅ 已验证 / ❌ 不可用 / ⚠️ 有条件 | {说明} |

**验证检查清单**：
- [ ] 已用通配符搜索 API 所有变体文件
- [ ] 已确认 API 在目标芯片架构（{arch}）上可用
- [ ] 已确认 API 支持所需的数据类型（{dtypes}）
- [ ] 已确认参数签名与官方文档一致
- [ ] 已确认 tmpBuffer/对齐等约束条件
- [ ] 如 API 不可用，已确定替代方案

#### 3.3.6 数据流设计

```
1. DataCopy: GM(x) → UB(inputX)
2. Compute:  UB(inputX) → UB(output)
3. DataCopy: UB(output) → GM(y)
```

#### 3.3.7 内存管理

| 内存区域 | 大小计算 | 说明 |
|---------|---------|------|
| **输入 UB** | `tileLength * sizeof(T)` | 单次搬入数据块 |
| **输出 UB** | `tileLength * sizeof(T)` | 单次计算结果块 |
| **临时缓冲区** | {大小} | {用途说明} |
| **Workspace** | {大小} | Global Memory 工作区 |

#### 3.3.8 UB 容量验证

> ⚠️ **所有 Buffer 大小之和必须 ≤ 实际可用 UB 空间**

| 平台 | 总 UB | 向量可用 | 系统保留 | 验证公式 |
|------|-------|---------|---------|---------|
| DAV_2201 (Ascend910B/Ascend910_93) | 192 KB | 184 KB | 8 KB | `sum(buffers) ≤ 184 * 1024` |
| DAV_3510 (Ascend950DT/Ascend910PR) | 248 KB | 248 KB | 0 KB | `sum(buffers) ≤ 248 * 1024` |

**查询方法**：`GetUBSizeInBytes()` 编译时常量，`GetRuntimeUBSize()` 运行时值

**Buffer 总和验证**：{列出所有 InitBuffer 调用及其大小} = {总计} KB {✅/❌}

---

### 3.4 模板二：{模板名称，如：优化计算路径}

#### 3.4.1 触发条件
{条件描述，如: input.dtype == half && totalLength >= UB_THRESHOLD}

#### 3.4.2 Host 侧 Tiling

```cpp
// op_host/arch*/${op_name}_tiling.cpp
uint32_t TILE_NUM = 8;
uint32_t IS_SPLIT = 1;
ASCENDC_TPL_SEL_PARAM(context, D_T_X, TILE_NUM, IS_SPLIT);
```

#### 3.4.3 Kernel 侧模板实例化

```cpp
// op_kernel/${op_name}_arch*.cpp
if constexpr (std::is_same_v<D_T_X, half> && IS_SPLIT == 1) {
    op.Process2();  // 优化的切分计算
}
```

#### 3.4.4 API 映射

| 计算步骤 | Ascend C API | 参数签名 | 平台验证 | 约束说明 | 替代方案 |
|---------|-------------|---------|---------|---------|---------|
| {步骤1} | {API\<T, Template\>} | (完整参数列表) | ✅/❌ {DAV_2201/DAV_3510} | {对齐/类型/尺寸约束} | {不可用时的替代} |

#### 3.4.5 数据流设计

```
{数据流描述}
```

#### 3.4.6 内存管理

| 内存区域 | 大小计算 | 说明 |
|---------|---------|------|
| **输入 UB** | {大小} | {说明} |

**UB 容量验证**：{同 3.3.8，验证 Buffer 总和 ≤ 可用 UB}

---

### 3.5 模板三：{模板名称}

> 按相同结构填写...

---

## 4. 性能优化

### 4.1 并行策略
{描述并行计算策略，如任务划分、多核协同}

### 4.2 流水线设计
{描述双缓冲、事件同步等优化手段}

---

## 5. 风险评估

### 5.1 API 风险
{描述可能存在的 API 兼容性问题}

### 5.2 精度风险
{描述可能存在的精度问题}

### 5.3 应对措施
{描述针对上述风险的应对方案}

---

## 6. 交付件清单

**必需**：`op_host/`, `op_kernel/`, `tests/`, `CMakeLists.txt`

**可选**：`op_graph/`（图模式需要）

---

## 7. 迭代规划

| 迭代 | 目标 | 代码开发 | UT开发 | ST用例 |
|------|------|---------|--------|-------|
| 迭代一 | 骨架搭建 | 单TilingKey + 预埋骨架 + 单dtype | 核心路径用例 | L0标准用例（基础shape + 单dtype） |
| 迭代二 | 策略完善 | 多TilingKey实现 + 单dtype | Tiling分支覆盖 | 多shape用例（单dtype） |
| 迭代三 | 规格完整 | 多TilingKey + 全dtype + 边界处理 | 全覆盖用例 | 全dtype + 边界 + 广播 |
