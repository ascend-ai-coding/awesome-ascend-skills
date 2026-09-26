// 每个算子一个 impl 文件：xops/csrc/npu_x_custom_op.cpp
// 占位符：npu_x_custom_op（torch 算子名）、aclnnXCustomOp（aclnn 导出符号，不带 GetWorkspaceSize 后缀）
#include <torch/library.h>

#include "ops_common.h"

namespace custom {

// step2, 为NPU设备实现前向接口：透传给 aclnnXCustomOp
// 形参顺序/类型必须与 ops_def_registration.cpp 的 schema 完全一致
at::Tensor npu_x_custom_op(const at::Tensor& x1, const at::Tensor& x2, at::Tensor& out)
{
    // 实参顺序必须与 aclnnXCustomOpGetWorkspaceSize 的形参顺序完全一致；
    // 末尾的 workspaceSize / executor 由宏自动补齐，此处不要写。
    EXEC_NPU_CMD_V1(aclnnXCustomOp, x1, x2, out);
    return out;
}

// step3, 为META设备实现前向接口：仅返回调用方预分配的输出
at::Tensor npu_x_custom_op_meta(const at::Tensor& x1, const at::Tensor& x2, at::Tensor& out)
{
    return out;
}

}  // namespace custom

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_x_custom_op", &custom::npu_x_custom_op);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_x_custom_op", &custom::npu_x_custom_op_meta);
}
