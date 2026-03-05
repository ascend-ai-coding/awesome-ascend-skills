# CANN aclnn 示例模板与用法

生成 `aclnn_*` 示例时：从 op_host/op_kernel 提取输入输出与属性，再按本模板填充分支；保持 CHECK_RET、成对分配/释放、Init → CreateAclTensor → GetWorkspaceSize → 执行 → 同步 → 打印/清理 的顺序。**参考以 CANN 单算子 API 执行文档与官方示例为准。**

## 官方文档参考（示例与规范来源）

- **单算子 API 执行**：CANN 应用开发文档中「单算子调用」「单算子API执行」— 两段式接口（GetWorkspaceSize → 申请 workspace → 执行）、aclInit/SetDevice/CreateStream 与资源释放顺序
- **调用示例**：官方文档中「调用 NN/融合算子接口示例代码」或同版本单算子示例（如 Add 等）
- **头文件与库**：`aclnnop/aclnn_*.h`、AscendCL API 说明（见 CANN 应用开发接口文档）

## 通用 main 流程

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_[operator_name].h"
#define CHECK_RET(cond, return_expr) do { if (!(cond)) { return_expr; } } while (0)

int Init(int32_t deviceId, aclrtStream *stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, ...);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, ...);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, ...);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape,
    void **deviceAddr, aclDataType dataType, aclTensor **tensor) {
  // aclrtMalloc → aclrtMemcpy H2D → 计算 strides → aclCreateTensor(..., ACL_FORMAT_ND, ...)
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, ...);

  // 2. 按 op_host 构造 inputs
  // 3. 构造 outputs
  // 4. aclnn[OperatorName]GetWorkspaceSize(..., &workspaceSize, &executor);
  // 5. if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ...);
  // 6. aclnn[OperatorName](workspaceAddr, workspaceSize, executor, stream);
  // 7. aclrtSynchronizeStream(stream);
  // 8. PrintOutResult / copy back
  // 9. aclDestroyTensor(...); aclrtFree(...); aclrtDestroyStream; aclrtResetDevice; aclFinalize();
  return 0;
}
```

## 占位示例

```cpp
// Input - 按算子定义 FILL IN
aclTensor *input1 = nullptr;
void *input1DeviceAddr = nullptr;
std::vector<int64_t> input1Shape = {1, 1, 1, 1}; // FILL IN
std::vector<Kernel_dtype> input1HostData(GetShapeSize(input1Shape), 0.0f);
ret = CreateAclTensor(input1HostData, input1Shape, &input1DeviceAddr, Acl_dtype, &input1);
```

## 生成步骤

1. 从 op_host 读取 Input/Output 名、DataType、Format、Attr。
2. 从 op_kernel 确认参数结构与张量个数。
3. 替换模板中的 operator_name、Kernel_dtype/Acl_dtype、输入输出构造与 aclnnXxxGetWorkspaceSize/aclnnXxx 调用。
4. 保持 CHECK_RET、成对释放；缺失处用 FILL IN 注释标出。

接口、dtype、format 需与 op_host/op_kernel 精确对齐。**官方参考**：CANN 单算子调用流程与官方示例代码（以当前使用 CANN 版本文档为准）。
