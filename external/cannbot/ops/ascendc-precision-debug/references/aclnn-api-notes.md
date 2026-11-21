# ACLNN 接口层笔记（缓存机制 / 宏速查 / 调用流程）

Step 3/4 涉及 aclnn 接口层（GetWorkspaceSize / executor / tiling）时阅读。本文档随团队踩坑持续补充。

## ACLNN 缓存机制

### 工作原理

- cache key 由 API 名 + op config + `DFX_IN` 参数 + `DFX_OUT` 参数的 hash 构成（源码：`op_cache.h:GetFromCache`）；
- 命中缓存时直接返回已有的 `aclOpExecutor`（含已计算的 TilingData），跳过 Tiling 函数；
- 缓存按 LRU 策略管理，默认上限 10000 条（`K_CACHE_LIMIT`）。

### 关键影响

- 如果 `DFX_IN` 遗漏了某个影响 TilingData 的参数（如算子属性 beta/threshold 等 scalar，或 int64_t/float/bool/const char* 等普通 C++ 类型参数），该参数不参与 cache key hash，同 shape 不同参数值的调用会命中缓存、复用旧 TilingData，导致计算结果错误；
- **解决方案**：确保 `DFX_IN` 包含所有输入参数（aclTensor* + aclScalar* + aclXXArray* + 普通 C++ 数据类型），使不同参数值产生不同 cache key（详见 patterns/executor-cache-scalar-attr.md）。

### 关闭缓存（调试用）

```bash
export DISABLE_L2_CACHE=1
```

调试精度问题时可用此环境变量关闭缓存，排除缓存干扰：**批量失败在关 cache 后消失 = cache 相关的强证据**。

## 常用宏速查

### L2 接口宏（op_dfx.h）

| 宏 | 用途 | 关键约束 |
| --- | --- | --- |
| `L2_DFX_PHASE_1(APIName, IN, OUT)` | 一阶段接口入口：时延统计 + 入参打印 + 缓存查找 | DFX_IN/OUT 必须与接口签名严格一致 |
| `L2_DFX_PHASE_2(APIName)` | 二阶段接口入口：时延统计 | 在 aclnnXxx 函数最前方调用 |
| `DFX_IN(...)` | 打包所有输入参数为 tuple | 必须包含 aclTensor* + aclScalar* + aclXXArray* + 普通 C++ 类型等全部输入 |
| `DFX_OUT(...)` | 打包所有输出参数为 tuple | 只含输出 tensor |

### L0 接口宏（make_op_executor.h / op_arg_def.h / op_dfx.h）

| 宏 | 用途 | 关键约束 / 来源 |
| --- | --- | --- |
| `CREATE_EXECUTOR()` | 创建 UniqueExecutor（aclOpExecutor 生成工厂） | make_op_executor.h |
| `OP_TYPE_REGISTER(KernelName)` | 注册 L0 算子类型 | op_dfx.h |
| `L0_DFX(OpName, ...)` | L0 接口入参打印 + profiling | op_dfx.h |
| `ADD_TO_LAUNCHER_LIST_AICORE(KernelName, OP_INPUT(...), OP_ATTR(...), OP_OUTPUT(...))` | 创建 AICore 执行任务并加入队列 | make_op_executor.h |
| `OP_INPUT(...)` | 打包输入 aclTensor | op_arg_def.h |
| `OP_ATTR(...)` | 打包算子属性（scalar 值传递给 Kernel） | **OP_ATTR 中每个 scalar 必须同时在 L2 的 DFX_IN 中出现** |
| `OP_OUTPUT(...)` | 打包输出 aclTensor | op_arg_def.h |
| `OP_OPTION(...)` | 打包精度模式 | op_arg_def.h |
| `OP_WORKSPACE(...)` | 打包显式 workspace | op_arg_def.h |
| `CHECK_COND` / `CHECK_RET` | 条件检查宏 | opdev/op_log.h（经 op_error_check.h 传递可见） |

### L0 接口基础张量操作（aclnn_kernels/）

| 接口 | 用途 | 头文件 |
| --- | --- | --- |
| `l0op::Contiguous(tensor, executor)` | 非连续 → 连续 | aclnn_kernels/contiguous.h |
| `l0op::ViewCopy(src, dst, executor)` | 连续 tensor → 连续/非连续 tensor | aclnn_kernels/contiguous.h |
| `l0op::Cast(self, dstDtype, executor)` | 数据类型转换 | aclnn_kernels/cast.h |
| `l0op::Reshape(tensor, shape, executor)` | shape 重构 | aclnn_kernels/reshape.h |

## 两段式调用流程

```
用户调用 aclnnXxxGetWorkspaceSize(input, ..., output, &workspaceSize, &executor)
  ↓
L2_DFX_PHASE_1 → 查缓存（GetFromCache）
  ├─ 命中 → 直接返回缓存的 executor + workspaceSize
  └─ 未命中 → 继续
  ↓
CREATE_EXECUTOR() → 创建 UniqueExecutor
  ↓
CheckParams → Contiguous → L0 接口（ADD_TO_LAUNCHER_LIST_AICORE）→ ViewCopy
  ↓
uniqueExecutor->GetWorkspaceSize() → 计算 workspace 大小
  ↓
存入缓存（AddOpExecCache）→ 返回 executor + workspaceSize
  ↓
用户申请 workspace 内存 → 调用 aclnnXxx(workspace, workspaceSize, executor, stream)
  ↓
L2_DFX_PHASE_2 → 框架执行 executor 中的任务队列 → NPU 计算
```
