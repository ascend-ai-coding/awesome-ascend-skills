# API 文档索引

基于 `$ASC_DEVKIT_DIR/docs/zh/api/` 的完整 API 文档索引。

---

## 文档位置

```
$ASC_DEVKIT_DIR/docs/zh/api/  — Ascend C API 文档根目录
```

> 子目录结构随 CANN 版本演进有变化（如 `context/` 扁平结构 → `SIMD-API/SIMT-API` 层级结构 → 目录与文件名英文化：`基础API/`→`basic_api/`、`高阶API/`→`adv_api/`、`Ascend-C-API列表.md`→`docs/zh/api/api_list.md`）。
> **不要假设具体的子目录名**，统一用 `find "$ASC_DEVKIT_DIR/docs/zh/api/" -name "{APIName}*.md"` 搜索。

---

## 一、基础数据结构

| API | 说明 |
|-----|------|
| `LocalTensor` | 存放 AI Core 中 Local Memory 的数据 |
| `GlobalTensor` | 存放 Global Memory 的全局数据 |
| `Coordinate` | 表示张量在不同维度的位置信息 |
| `Layout` | 描述多维张量内存布局的基础模板类 |
| `TensorTrait` | 描述 Tensor 相关信息的基础模板类 |

---

## 二、基础 API

### 表2：标量计算 API
| API | 说明 |
|-----|------|
| `ScalarGetCountOfValue` | 获取标量值计数 |
| `ScalarCountLeadingZero` | 计数前导零 |
| `ScalarCast` | 标量类型转换 |

### 表3：矢量计算 API
| 类别 | API |
|-----|-----|
| 算术运算 | `Add`、`Sub`、`Mul`、`Div`、`Abs` |
| 三角函数 | `Sin`、`Cos`、`Tan`、`Asin`、`Acos`、`Atan` |
| 指数对数 | `Exp`、`Log`、`Sqrt`、`Rsqrt` |
| 比较运算 | `Greater`、`Less`、`Equal`、`NotEqual` |
| 逻辑运算 | `And`、`Or`、`Not`、`Xor` |

### 表4：数据搬运 API
| API | 说明 | 对齐要求 |
|-----|------|---------|
| `DataCopy` | 数据拷贝 | 512 字节 |
| `DataMove` | 数据移动 | 32 字节 |

### 表5：资源管理 API
| API | 说明 |
|-----|------|
| `MemAlloc` | 内存分配 |
| `MemFree` | 内存释放 |

### 表6：同步控制 API
| API | 说明 |
|-----|------|
| `Sync` | 同步等待 |
| `Barrier` | 屏障同步 |

### 表7：缓存处理 API
| API | 说明 |
|-----|------|
| `Cache` | 缓存控制操作 |

### 表8：系统变量访问 API
| API | 说明 |
|-----|------|
| `GetSysVar` | 获取系统变量 |

### 表9：原子操作接口
| API | 说明 |
|-----|------|
| `AtomicAdd`、`AtomicSub`、`AtomicMin`、`AtomicMax` | 原子算术操作 |

### 表10：调试接口
| API | 说明 |
|-----|------|
| `Debug` | 调试相关操作 |

### 表11：工具函数接口
| API | 说明 |
|-----|------|
| 通用工具函数 | 各种辅助函数 |

### 表12：Kernel Tiling 接口
| API | 说明 |
|-----|------|
| `GetTilingKey` | 获取 Tiling Key |
| `SetTilingKey` | 设置 Tiling Key |

### 表13：ISASI 接口
| API | 说明 |
|-----|------|
| 硬件体系结构相关接口 | 底层硬件访问 |

---

## 三、高阶 API

> 实际目录 `docs/zh/api/SIMD-API/adv_api/` 子目录索引见下表。HCCL 通信类详见[第七章](#七hccl-通信-api)。

| 子目录 | 类别 | 典型 API |
|--------|------|---------|
| `math_compute/` | 三角/双曲/位运算/类型转换 | `Acos`、`Acosh`、`Cos`、`Cast`、`BitwiseAnd`、`BitwiseOr`、`Addcdiv`、`Addsub` |
| `quantization/` | 量化/反量化 | 量化相关操作 |
| `reduction_operations/` | 归约 | `ReduceMax`、`ReduceSum` |
| `sort_operations/` | 排序 | `Sort`、`TopK` |
| `tensor_transform/` | 张量重排 | `Broadcast`、`Transpose` |
| `normalization/` | 归一化 | `LayerNorm` 相关 |
| `activation_functions/` | 激活 | `Relu`、`Sigmoid`、`Gelu` |
| `cube_compute/` | 矩阵 | `Mmad` 相关 |
| `convolution_compute/` | 卷积 | 卷积相关 |
| `index_compute/` | 索引 | 索引相关 |
| `data_filter/` | 过滤 | 数据过滤相关 |
| `random_functions/` | 随机 | 随机数生成 |
| `data_structures/` | 高阶 API 数据结构 | `TensorShape`、`TensorDataType` |
| `experimental/` | 实验性接口 | `BesselI0`、`Ndtri` 等 |
| `HCCL_communication/` | 集合通信 | 详见[第七章](#七hccl-通信-api) |

---

## 四、Utils API

> 实际目录 `docs/zh/api/Utils-API/`，含调测接口（printf、asc_dump）等。用 `find "$ASC_DEVKIT_DIR/docs/zh/api/Utils-API/" -name "*.md"` 查阅。

---

## 五、AI CPU API

> 实际目录 `docs/zh/api/AI-CPU-API/`。用 `find "$ASC_DEVKIT_DIR/docs/zh/api/AI-CPU-API/" -name "*.md"` 查阅。

---

## 六、C API

| 类别 | 说明 |
|-----|------|
| `atomic/` | 原子操作 C API |
| `cache_ctrl/` | 缓存控制 C API |
| `cube_compute/` | Cube 计算 C API |
| `cube_datamove/` | Cube 数据搬运 C API |
| `vector_compute/` | 矢量计算 C API |
| `vector_datamove/` | 矢量数据搬运 C API |
| `reg_compute/` | 寄存器矢量计算 C API |
| `scalar_compute/` | 标量计算 C API |
| `sync/` | 同步控制 C API |
| `utils/` | 工具函数 C API（sys_init/sys_misc/sys_var） |
| `defs/` | 常量/枚举/类型定义 |
| `spr/` | 特殊寄存器访问 |

---

## 七、HCCL 通信 API

HCCL（集合通信）API 文档位于 `docs/zh/api/SIMD-API/adv_api/HCCL_communication/`（另有总览 `HCCL_communication.md`），子目录如下：

| 子目录 | 内容 | 典型 API |
|--------|------|---------|
| `HCCL_Kernel/` | Kernel 侧通信原语 | `Hccl::InitV2`、`Hccl::AlltoAllV`、`Hccl::Wait`、`Hccl::Finalize` |
| `HCCL_Tiling/` | Host 侧 Tiling 配置 | `Mc2CcTilingConfig`、`SetCcTilingV2` |
| `HCCL-Context/` | 通信上下文 | `GetHcclContext`、`SetHcclContext` |

**建议先读使用说明**，获取完整调用流程和代码示例，再按需查阅单个 API 文档：
- `HCCL_Kernel/HCCL_usage.md` — Kernel 侧标准调用流程（InitV2 → SetCcTilingV2 → Prepare → Commit → Wait → Finalize），含完整 Kernel 代码示例
- `HCCL_Tiling/HCCL_Tiling_usage.md` — Tiling 侧配置流程（创建 Mc2CcTilingConfig → Set 系列配置 → GetTiling），含代码示例
- `HCCL-Context/HCCL_Context_intro.md` — 通信上下文 GetHcclContext/SetHcclContext 说明

查找命令：
```bash
find "$ASC_DEVKIT_DIR/docs/zh/api/SIMD-API/adv_api/HCCL_communication/" -name "*.md"
```

HCCL 头文件另见 `$ASC_DEVKIT_DIR/include/adv_api/hccl/`（`hccl.h`、`hccl_common.h`、`hccl_tiling.h`、`hccl_tilingdata.h`）。

---

## 使用建议

1. **API 文档查找**：
   ```bash
   find "$ASC_DEVKIT_DIR/docs/zh/api/" -name "${APIName}*.md"
   ```
   （不依赖具体子目录结构）

2. **查阅 API 文档时注意**：
   - **Restriction 章节**：查看使用限制和对齐要求
   - **Parameters 章节**：确认参数类型和范围
   - **Returns 章节**：了解返回值含义
   - **Example 章节**：参考使用示例

3. **常见对齐要求**：
   - 大多数操作：32 字节对齐
   - DataCopy：512 字节对齐
   - 某些特殊 API：64/128 字节对齐

---

## 相关资源

- [示例代码目录](example-catalog.md)
- [环境兼容性表](compatibility.md)
