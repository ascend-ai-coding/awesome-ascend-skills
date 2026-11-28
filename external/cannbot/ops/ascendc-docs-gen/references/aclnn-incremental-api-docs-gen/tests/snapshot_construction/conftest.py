# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Pytest fixtures for snapshot_construction tests."""

import pytest


@pytest.fixture
def sample_md_content():
    """Sample operator markdown content for cann_doc_struct tests.

    Contains all 6 sections: support, summary, function_proto,
    sub_function_hash, constraint, example.
    """
    return """# aclnnAbs

## 产品支持情况
<table>
<thead>
<tr><th>产品</th><th>版本</th></tr>
</thead>
<tbody>
<tr><td>Atlas 800</td><td>A2</td></tr>
<tr><td>Atlas 300I</td><td>T2</td></tr>
</tbody>
</table>

## 功能说明
- 接口功能：计算输入张量的绝对值。

- 计算公式
$$y = |x|$$

## 函数原型
两段式接口说明文字

```cpp
aclnnStatus aclnnAbsGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclnnOpExecutor **executor)
```

## aclnnAbsGetWorkspaceSize

<table>
<thead>
<tr><th>参数</th><th>类型</th></tr>
</thead>
<tbody>
<tr><td>self</td><td>输入张量</td></tr>
</tbody>
</table>

产品限制说明

**返回值**
aclnnStatus 返回值参见 aclnn 返回码。

第一段接口会完成入参校验，出现以下场景时报错：
<table>
<thead>
<tr><th>错误码</th><th>描述</th></tr>
</thead>
<tbody>
<tr><td>ACLNN_ERR_PARAM</td><td>参数错误</td></tr>
</tbody>
</table>

## aclnnAbs

<table>
<thead>
<tr><th>参数</th><th>类型</th></tr>
</thead>
<tbody>
<tr><td>out</td><td>输出张量</td></tr>
</tbody>
</table>

**返回值**
aclnnStatus 返回值参见 aclnn 返回码。

## 约束说明
输入张量的数据类型支持 float16、float32。

## 调用示例
示例代码描述

```cpp
// 示例代码
aclnnAbsGetWorkspaceSize(addr, out, &workspaceSize, &executor);
aclnnAbs(addr, executor, stream);
```
"""
