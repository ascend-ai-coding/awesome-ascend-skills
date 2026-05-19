# aclnn API 返回码完整表

本表列出调用 aclnn API 时常见的接口返回码。

> **遇到未列出的错误码？** 参见 SKILL.md [未知错误码处理](../SKILL.md#未知错误码处理) 流程：
> 1. 使用 `aclGetRecentErrMsg()` 获取详细错误信息
> 2. 使用 `ascendc-docs-search` 技能搜索官方文档
> 3. 搜索社区资源（CANN Issue / 昇腾论坛）

## 基本状态码

| 状态码名称 | 状态码值 | 说明 |
|-----------|---------|------|
| ACLNN_SUCCESS | 0 | 成功 |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 参数校验错误，参数中存在非法的nullptr |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数校验错误，如输入的两个数据类型不满足输入类型推导关系 |
| ACLNN_ERR_RUNTIME_ERROR | 361001 | API内部调用npu runtime的接口异常 |

## NPU 运行时错误码（5xxxxx）

以下错误码由 NPU 硬件/驱动层产生，非 aclnn 框架错误码，通过 aclGetRecentErrMsg() 或 plog 日志查看。

| 错误码值 | 错误信息 | 说明 |
|---------|---------|------|
| 507035 | 向量核异常 (vector core exception) | NPU 向量核执行异常，通常由 DMA 对齐错误 / UB 溢出触发。通过 `aclrtStreamSynchronize()` 报告给用户 |

> 507035 排查详见 [debug_workflow.md#507035-向量核异常](debug_workflow.md#507035-向量核异常)

## 内部异常状态码（561xxx）

### 核心错误

| 状态码名称 | 状态码值 | 说明 |
|-----------|---------|------|
| ACLNN_ERR_INNER | 561000 | 内部异常：API发生内部异常 |
| ACLNN_ERR_INNER_INFERSHAPE_ERROR | 561001 | 内部异常：输出shape推导错误 |
| ACLNN_ERR_INNER_TILING_ERROR | 561002 | 内部异常：Tiling时发生异常 |
| ACLNN_ERR_INNER_FIND_KERNEL_ERROR | 561003 | 内部异常：查找npu kernel异常（可能算子二进制包未安装） |

### 执行器错误

| 状态码名称 | 状态码值 | 说明 |
|-----------|---------|------|
| ACLNN_ERR_INNER_CREATE_EXECUTOR | 561101 | 内部异常：创建aclOpExecutor失败（操作系统异常） |
| ACLNN_ERR_INNER_NOT_TRANS_EXECUTOR | 561102 | 内部异常：未调用uniqueExecutor ReleaseTo |
| ACLNN_ERR_INNER_NULLPTR | 561103 | 内部异常：出现nullptr异常 |

### 属性错误

| 状态码名称 | 状态码值 | 说明 |
|-----------|---------|------|
| ACLNN_ERR_INNER_WRONG_ATTR_INFO_SIZE | 561104 | 内部异常：算子属性个数异常 |
| ACLNN_ERR_INNER_ATTR_NUM_OUT_OF_BOUND | 561114 | 内部异常：属性个数超过json中指定 |
| ACLNN_ERR_INNER_ATTR_LEN_NOT_ENOUGH | 561115 | 内部异常：属性个数少于json中指定 |

### 配置错误

| 状态码名称 | 状态码值 | 说明 |
|-----------|---------|------|
| ACLNN_ERR_INNER_KEY_CONFILICT | 561105 | 内部异常：kernel匹配hash key冲突 |
| ACLNN_ERR_INNER_INVALID_IMPL_MODE | 561106 | 内部异常：算子实现模式参数错误 |
| ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND | 561107 | 内部异常：未检测到ASCEND_OPP_PATH环境变量 |

### JSON/配置文件错误

| 状态码名称 | 状态码值 | 说明 |
|-----------|---------|------|
| ACLNN_ERR_INNER_LOAD_JSON_FAILED | 561108 | 内部异常：加载算子信息json失败 |
| ACLNN_ERR_INNER_JSON_VALUE_NOT_FOUND | 561109 | 内部异常：json文件某字段失败 |
| ACLNN_ERR_INNER_JSON_FORMAT_INVALID | 561110 | 内部异常：json的format非法值 |
| ACLNN_ERR_INNER_JSON_DTYPE_INVALID | 561111 | 内部异常：json的dtype非法值 |
| ACLNN_ERR_INNER_OP_FILE_INVALID | 561113 | 内部异常：加载json字段时异常 |
| ACLNN_ERR_INNER_INPUT_NUM_IN_JSON_TOO_LARGE | 561116 | 内部异常：输入个数超过32限制 |
| ACLNN_ERR_INNER_INPUT_JSON_IS_NULL | 561117 | 内部异常：json信息描述缺失 |

### 二进制包错误

| 状态码名称 | 状态码值 | 说明 |
|-----------|---------|------|
| ACLNN_ERR_INNER_OPP_KERNEL_PKG_NOT_FOUND | 561112 | 内部异常：未加载算子二进制kernel库 |
| ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID | 561118 | 内部异常：静态二进制json中workspace异常 |
| ACLNN_ERR_INNER_STATIC_BLOCK_DIM_INVALID | 561119 | 内部异常：静态二进制json中核数信息异常 |

## 错误码分类速查

### 按错误来源

| 类别 | 错误码范围 | 典型代表 |
|-----|----------|---------|
| **参数错误** | 161xxx | PARAM_NULLPTR, PARAM_INVALID |
| **Runtime错误** | 361xxx | RUNTIME_ERROR |
| **内部错误** | 561xxx | TILING_ERROR, FIND_KERNEL_ERROR |

### 按错误类型

| 类型 | 常见错误码 | 排查方向 |
|-----|----------|---------|
| **参数校验** | 161001, 161002 | 检查参数类型/shape/dtype |
| **Tiling问题** | 561002 | 检查Tiling函数逻辑 |
| **Kernel问题** | 561003, 561112 | 检查算子安装、vendor_name |
| **环境配置** | 561107 | 检查环境变量 |
| **JSON配置** | 561108-561119 | 检查算子信息文件 |
