# 多核数据交换与同步

## 多核之间交换数据

如果需要多核之间交换数据，需要使用 **workspace**：

1. **tiling 侧**：申请 workspace 大小
2. **kernel 侧**：直接使用 workspace 指针

### 防止时序问题

- kernel 处需要调用 `SyncAll()` 接口
- tiling 需要设置 `context->SetScheduleMode(1)`

```cpp
// Tiling 侧
context->SetScheduleMode(1);  // 同步模式
size_t workspaceSize = ...;   // 计算所需 workspace 大小
context->SetWorkspaceSize(workspaceSize);

// Kernel 侧
__gm__ float* ws = (__gm__ float*)workspace;
// ... 使用 workspace 交换数据 ...
AscendC::SyncAll();  // 全核同步
```

## 多个 VF_CALL 之间的同步

1. 同一个 kernel 内多个 `Simt::VF_CALL` 之间**仅能保证 block 内 VF 串行**
2. **不能保证全核同步**
3. 如果 `Simt::VF_CALL` 之间需要全核同步：
   - 在同步的位置加上 `SyncAll()`
   - tiling 设置 `context->SetScheduleMode(1)`

## 头文件

```cpp
#include "simt_api/device_sync_functions.h"
```