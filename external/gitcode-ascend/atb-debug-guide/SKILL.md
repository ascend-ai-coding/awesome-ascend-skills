---
name: external-gitcode-ascend-atb-debug-guide
description: 'ATB 调试指南技能。当用户遇到 ATB 算子测试问题、需要分析错误原因、或需要了解 ATB 环境配置时调用此技能。 覆盖：环境配置问题、ABI版本不匹配、内存错误、CSV测试失败、ACLNN接口问题等常见场景。

  '
keywords:
- atb
- debug
- error
- troubleshooting
- 昇腾
- NPU
- 调试
- aclnn
- moe
metadata:
  author: ascend-transformer-boost-team + Claude Code + opus4.7
  version: 1.1.1
  created: '2026-04-19'
  updated: '2026-04-28'
  skill-type: debug
original-name: atb-debug-guide
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# ATB Debug Guide

## 功能概述

提供 ATB 调试指南，覆盖环境配置问题、ABI版本不匹配、内存错误、CSV测试失败、ACLNN接口问题等常见场景。

## 调用时机

在以下情况下调用此技能：
- 用户遇到 ATB 算子测试问题
- 用户需要分析错误原因
- 用户需要了解 ATB 环境配置

## ⚠️ 路径约束（必须执行）

执行此技能前，**必须从用户处获取以下路径**（如未提供则使用默认路径）：
- `<CANN_PATH>`: CANN 安装路径（如 `/usr/local/Ascend/cann-9.0.0-beta.2`）
- `<ATB_REPO_PATH>`: ATB 仓库路径

---

## 索引

| 问题类型 | 快速定位 |
|----------|----------|
| [ABI版本不匹配](#1-abi版本不匹配错误) | undefined symbol, shared object file |
| [内存错误](#2-内存错误) | ERROR_OUT_OF_HOST_MEMORY |
| [Tensor维度错误](#3-tensor维度错误) | ERROR_INVALID_TENSOR_DIM |
| [数据类型错误](#4-数据类型错误) | KeyError: 'float32', ERROR_INVALID_TENSOR_INI_MATCH |
| [ACLNN函数签名错误](#5-aclnn函数签名不匹配) | 161001, 161002, undefined symbol |
| [ACLNN MoE场景不匹配](#6-aclnn-moe场景不匹配) | {xxx}Quant 等非MoE算子 |
| [日志分析方法](#7-日志分析) | plog日志, atb日志 |

---

## 常见错误分类

### ABI 版本不匹配错误

#### 错误信息
```
OSError: /path/to/atb/output/atb/cxx_abi_0/lib/libatb_test_framework.so: cannot open shared object file
```

或

```
undefined symbol: _ZN3atb13OpParamToJson...
```

#### 原因
ATB 输出目录有两个 ABI 版本：
- `cxx_abi_0`: 旧版本（g++/libstdc++abi_0）
- `cxx_abi_1`: 新版本（g++/libstdc++abi_1）

测试框架库和依赖库必须使用相同 ABI 版本。

#### 解决方案

**方案A：使用正确的 set_env.sh 参数**
```bash
source <CANN_PATH>/set_env.sh
source <ATB_REPO_PATH>/output/atb/set_env.sh --cxx_abi=1
echo $ATB_HOME_PATH
```

**方案B：创建软链接（不推荐）**
```bash
ln -s <ATB_REPO_PATH>/output/atb/cxx_abi_1 <ATB_REPO_PATH>/output/atb/cxx_abi_0
```

#### 验证方法
```bash
ls -la $ATB_HOME_PATH/lib/libatb_test_framework.so
ldd $ATB_HOME_PATH/lib/libatb_test_framework.so | grep libatb
```

---

### 内存错误

#### 错误信息
```
ActualError: S:ERROR_OUT_OF_HOST_MEMORY
```

#### 可能原因
1. **输入输出 tensor 数量不匹配**：Runner 中定义的 IN_TENSOR_NUM/OUT_TENSOR_NUM 与接口规格不一致
2. **Shape 计算错误**：输出 shape 计算有误导致内存分配失败
3. **数据太大**：测试用例的 shape 超出设备内存限制

#### 解决方案
1. 检查 Runner 中 IN_TENSOR_NUM 和 OUT_TENSOR_NUM 是否与接口规格匹配
2. 检查 InferShapeImpl 中的 shape 推导逻辑
3. 减小测试用例的 shape 大小

#### 接口规格检查点
```cpp
// 检查接口规格中的输入输出定义
// 例如 SwigluQuant:
// - 输入：1个 [ntokens, 2*hidden_size]
// - 输出：2个 [ntokens, hidden_size] + [ntokens]
// Runner 必须匹配这些规格
```

---

### Tensor 维度错误

#### 错误信息
```
ActualError: I:ERROR_INVALID_TENSOR_DIM_NUM
ActualError: I:ERROR_INVALID_TENSOR_DIM
```

#### 检查点
1. 输入 tensor 维度数量是否符合预期（通常为 2D）
2. 特定维度值是否在有效范围内
3. 维度是否能被指定值整除（如 hidden size 必须能被 2 整除）

---

### 数据类型错误

#### 错误信息
```
ActualError: I:ERROR_INVALID_TENSOR_INI_MATCH
KeyError: 'float32'
```

#### 原因
1. 输入数据类型不支持（如 float32 不在 data_generation.py 的 dtype_dict 中）
2. 输出数据类型与期望不符

#### 解决方案
在 CSV 中使用 data_generation.py 支持的数据类型：
- `float` (对应 float32)
- `float16` / `half`
- `bf16` / `bfloat16`
- `int8` / `char`
- `int32` / `int`
- `int64` / `long`

---

### ACLNN 函数签名不匹配

#### 错误信息
```
undefined symbol: aclnn{xxx}QuantGetWorkspaceSize
ActualError: S:ERROR_CANN_ERROR (161001, 161002)
```

#### 原因
1. 函数名拼写错误
2. 函数参数数量/类型与 ACLNN API 不匹配
3. 动态库加载失败

#### 解决方案
```bash
# 检查函数符号
nm -D <ATB_PATH>/output/atb/cxx_abi_1/lib/libatb.so | grep -i {xxx}

nm -D $ASCEND_TOOLKIT_HOME/lib64/libopapi_nn.so | grep -i {xxx}
```

#### ACLNN 错误码
| 错误码 | 名称 | 含义 |
|--------|------|------|
| 161001 | ACLNN_ERR_PARAM_NULLPTR | 参数为空指针（如 x/yOut 为空） |
| 161002 | ACLNN_ERR_PARAM_INVALID | 参数类型/维度/取值无效 |

---

### ACLNN MoE 场景不匹配 ⭐️

#### 问题描述
**问题**: ATB 算子为非 MoE 场景设计，但 ACLNN 接口仅支持 MoE 场景。

#### 典型案例
**{xxx}Quant**:
- ATB 设计: 非 MoE Per-Token 动态量化
- ACLNN: 仅支持 MoE 分组动态量化，需要 `groupIndex` 和 `smoothScales`

#### 错误信息
```
[ERROR] errno[161001] Check yOut != nullptr failed
```

实际上由于 `smoothScalesOptional` 和 `groupIndexOptional` 不支持空指针（当前版本）。

#### 解决方案

**方案A：回退到 OPS 实现（推荐）**
```cpp
// 禁用 ACLNN 路径
std::shared_ptr<Runner> XxxOperation::CreateRunner(Context &context) const
{
    // TODO: 待 ACLNN 支持非 MoE 场景后启用
    return std::make_shared<XxxOpsRunner>(param_);
}
```

**方案B：构造 MoE 参数模拟非 MoE（复杂，不推荐）**
- 构造单分组 `groupIndex`
- 构造全 1 `smoothScales`

#### 如何识别
查阅 ACLNN 文档：
> "当前 xxx 仅支持 MoE 场景"

---

### 日志分析方法

#### 日志位置

| 日志类型 | 路径 | 内容 |
|----------|------|------|
| ATB 日志 | `<MY_DIR>/log/<log_name>/atb/atb_*.log` | ATB 算子调度、infershape、setup |
| 下层组件日志 | `<MY_DIR>/log/<log_name>/debug/plog/plog-*.log` | GE、RUNTIME、OP、ASCENDCL |

#### ATB 日志关键字
```bash
# 查找 ACLNN 加载
grep -i "aclnn\|LoadMethod\|workspace" atb_*.log

# 查找错误
grep -i "error\|fail\|ERROR" atb_*.log

# 查找测试用例执行
grep -i "Case\|times" atb_*.log
```

#### 下层组件日志关键字
```bash
# 查找 ACLNN 底层错误
grep -i "161001\|161002\|errno" plog-*.log

# 查找 kernel 执行
grep -i "kernel\|execute" plog-*.log

# 查找内存分配
grep -i "malloc\|aclrtMalloc" plog-*.log
```

#### 典型日志分析流程

1. **ATB 层** (atb_*.log): 检查算子创建、Runner 选择、shape 推导
2. **接口层** (plog-*.log OP/GE): 检查 ACLNN 接口调用、参数校验
3. **运行时层** (plog-*.log RUNTIME): 检查 kernel 执行、内存分配

---

## ATB 环境检查流程

### Step 1: 检查 NPU 设备
```bash
npu-smi info
```

### Step 2: Source 环境
```bash
source <CANN_PATH>/set_env.sh
source <ATB_REPO_PATH>/output/atb/set_env.sh
# 或指定 ABI 版本
source <ATB_REPO_PATH>/output/atb/set_env.sh --cxx_abi=1
```

### Step 3: 验证测试框架
```bash
echo "ATB_HOME_PATH=$ATB_HOME_PATH"
ls -la $ATB_HOME_PATH/lib/libatb_test_framework.so
```

### Step 4: 运行 CSV 测试
```bash
cd <ATB_REPO_PATH>/tests/framework/python/CsvOpsTestTool
python3 atb_csv_ops_test.py -i <CSV_FILE_PATH>
```

---

## 调试检查表

| 检查项 | 命令 | 预期结果 |
|--------|------|----------|
| NPU 设备 | `npu-smi info` | 显示可用设备 |
| ATB_HOME_PATH | `echo $ATB_HOME_PATH` | 指向 cxx_abi_0/1 目录 |
| 测试框架库 | `ls $ATB_HOME_PATH/lib/libatb_test_framework.so` | 文件存在 |
| 依赖库版本 | `ldd $ATB_HOME_PATH/lib/libatb_test_framework.so` | 所有依赖可解析 |
| ACLNN 符号 | `nm -D $ASCEND_TOOLKIT_HOME/lib64/libopapi_nn.so \| grep aclnn` | 符号存在 |

---

<details>
<summary>记录模板（点击展开）</summary>

### 新问题记录

```markdown
## 问题名称

**错误信息：**
```
[粘贴错误信息]
```

**日志分析：**
- ATB 日志: [关键行]
- 下层日志: [关键行]

**根本原因：**
[分析原因]

**解决方案：**
[解决方案]

**验证方法：**
[验证命令]
```
</details>

---

*版本: 1.1.0*  
*更新: 2026-04-25 - 格式标准化*
