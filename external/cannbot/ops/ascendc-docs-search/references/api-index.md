# API 文档索引

基于 `asc-devkit/docs/api/` 的完整 API 文档索引。

---

## 文档位置

```
asc-devkit/docs/api/
├── context/           # 基础数据结构（LocalTensor、GlobalTensor 等）
├── basic_api/         # 基础 API（表2-13）
├── adv_api/           # 高阶 API（表14-15）
├── utils/             # 公共辅助函数
├── aicpu_api/         # AI CPU API
└── c_api/             # C API
```

---

## 一、基础数据结构（context/）

| API | 说明 |
|-----|------|
| `LocalTensor` | 存放 AI Core 中 Local Memory 的数据 |
| `GlobalTensor` | 存放 Global Memory 的全局数据 |
| `Coordinate` | 表示张量在不同维度的位置信息 |
| `Layout` | 描述多维张量内存布局的基础模板类 |
| `TensorTrait` | 描述 Tensor 相关信息的基础模板类 |

---

## 二、基础 API（basic_api/）

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

## 三、高阶 API（adv_api/）

### 表14：数学计算 API
| 类别 | API |
|-----|-----|
| 三角函数 | `Acos`、`Acosh`、`Asin`、`Asinh`、`Atan`、`Atanh`、`Cos`、`Cosh`、`Sin`、`Sinh`、`Tan`、`Tanh` |
| 双曲函数 | `Sinh`、`Cosh`、`Tanh` |
| 位运算 | `BitwiseAnd`、`BitwiseNot`、`BitwiseOr`、`BitwiseXor` |
| 类型转换 | `Cast` |
| 复合运算 | `Addcdiv`、`Addsub` 等 |

### 表15：量化操作 API
| API | 说明 |
|-----|------|
| 量化相关操作 | 量化/反量化操作 |

---

## 四、Utils API（utils/）

公共辅助函数，提供通用工具支持。

---

## 五、AI CPU API（aicpu_api/）

AI CPU 处理器相关 API。

---

## 六、C API（c_api/）

| 类别 | 说明 |
|-----|------|
| `atomic/` | 原子操作 C API |
| `cache_ctrl/` | 缓存控制 C API |
| `cube_compute/` | Cube 计算 C API |
| `vector_compute/` | 矢量计算 C API |

---

## 使用建议

1. **API 文档查找优先级**：
   ```
   asc-devkit/docs/api/context/  →  基础数据结构
   asc-devkit/docs/api/basic_api/ →  基础 API
   asc-devkit/docs/api/adv_api/   →  高阶 API
   ```

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
