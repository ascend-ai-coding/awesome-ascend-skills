---
name: external-gitcode-ascend-atb-aclnn-operator-migration
description: '自动执行 ATB 算子到 ACLNN 的迁移操作，在 910B/950 设备上启用 ACLNN 加速。 支持参数映射、ACLNN Runner
  实现、设备检测切换和功能/性能验证全流程。

  '
keywords:
- atb
- aclnn
- migration
- operator
- 昇腾
- ascend
- 910b
- 950
metadata:
  author: ascend-transformer-boost-team + Claude Code + opus4.7
  version: 1.2.0
  created: '2026-04-17'
  updated: '2026-04-28'
  skill-type: migration
hooks:
  PreToolUse:
  - matcher: Write|Edit|Bash
    hooks:
    - type: command
      command: ([ -z "$CANN_PATH" ] || [ ! -f "$CANN_PATH/set_env.sh" ]) && echo '[PATH
        CHECK] CANN_PATH 未设置或无效，请先向用户获取 CANN 路径' >&2; ([ -z "$ATB_REPO_PATH" ] ||
        [ ! -d "$ATB_REPO_PATH" ]) && echo '[PATH CHECK] ATB_REPO_PATH 未设置或无效，请先向用户获取
        ATB 路径' >&2 || true
original-name: atb-aclnn-operator-migration
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# ATB到ACLNN算子迁移工具

## 功能概述

该工具帮助开发者将ATB中的算子迁移到ACLNN实现，特别是在 910B/950 设备上获得更好的兼容性优化。

## 调用时机

在以下情况下调用此技能：
- 用户需要将 ATB 算子迁移到 ACLNN 实现以提升 910B/950 设备性能
- 用户已生成算子替换设计文档（建议先运行 `atb-aclnn-operator-replacement-designer`）
- 用户需要在 ATB 源码上应用 ACLNN Runner 改造

## ⚠️ 路径约束（必须执行）

执行此技能前，**必须从用户处获取以下路径**：
- `<CANN_PATH>`: CANN 安装路径（如 `/usr/local/Ascend/ascend-toolkit/latest`）
- `<ATB_REPO_PATH>`: ATB 仓库路径（如 `{your working path}ascend-transformer-boost`）

**若用户未提供或路径无效，立即停止并报错。**

---

## 前置条件

- CANN 环境已安装（`ASCEND_TOOLKIT_HOME` 已设置）
- ATB 源码仓库已克隆至本地
- 算子对应的 ATB 接口文档和 ACLNN 接口文档已获取
- 建议已完成 `atb-aclnn-operator-replacement-designer` 生成设计文档

<details>
<summary>检查点系统（点击展开）</summary>

每个操作步骤末尾的 **检查点** 标记了验证标准。执行完成后请在 [执行结果](#执行结果) 章节填写实际状态。

| 检查点 ID | 描述 | 验证命令 |
|-----------|------|---------|
| Step0-ENV | CANN 环境 sourced | `echo $ASCEND_TOOLKIT_HOME` |
| Step0-DIR | ATB 源码目录存在 | `[ -d <ATB_REPO_PATH> ]` |
| Step1-DIR | 原有 Operation 源文件定位 | `ls <op>_operation.cpp` |
| Step1-DIR | 现有 ACLNN runner 参考定位 | `ls softmax_aclnn_runner.cpp` |
| Step2-FILE | `<op>_aclnn_runner.h` 创建 | `[ -f <op>_aclnn_runner.h ]` |
| Step2-FILE | `<op>_aclnn_runner.cpp` 创建 | `[ -f <op>_aclnn_runner.cpp ]` |
| Step2-CODE | `BuildAclnnVariantPack` 实现 | `grep "BuildAclnnVariantPack" <op>_aclnn_runner.cpp` |
| Step2-CODE | `SetAclNNWorkspaceExecutor` 实现 | `grep "SetAclNNWorkspaceExecutor" <op>_aclnn_runner.cpp` |
| Step2-CODE | `LaunchAclnnKernel` 实现 | `grep "LaunchAclnnKernel" <op>_aclnn_runner.cpp` |
| Step2-CODE | `LoadMethod` 实现 | `grep "LoadMethod" <op>_aclnn_runner.cpp` |
| Step3-MOD | CreateRunner 包含 910B/950 分支 | `grep "Is910B\|Is950\|910B\|950" <op>_operation.cpp` |
| Step4-BUILD | 编译成功无错误 | `make ... 2>&1` |
| Step5-TEST | 非 910B/950 设备功能正常 | `./apitest ...` |
</details>

## 迁移步骤

### 步骤1: 分析ATB算子实现

1. **理解ATB算子参数结构**
   - 查看 `infer_op_params.h` 中的参数结构
   - 确认参数类型和约束

2. **分析现有实现**
   - 查看 `<operator>_operation.cpp` 了解操作逻辑
   - 分析 `<operator>_ops_runner.cpp` 了解现有runner实现

3. **研究ACLNN接口**
   - 了解对应ACLNN算子的接口
   - 分析参数映射关系

### 步骤2: 创建ACLNN Runner

1. **创建头文件** (`<operator>_aclnn_runner.h`)
   - 继承自 `AclnnRunner` 基类
   - 定义必要的成员变量和方法
   - 声明ACLNN函数指针

2. **实现源文件** (`<operator>_aclnn_runner.cpp`)
   - 实现 `BuildAclnnVariantPack` 方法处理张量转换
   - 实现 `SetAclNNWorkspaceExecutor` 方法配置工作空间
   - 实现 `LaunchAclnnKernel` 方法执行计算
   - 实现 `LoadMethod` 方法动态加载ACLNN函数

### 步骤3: 修改Operation实现

1. **更新头文件包含**
   - 添加 `#include "<operator>_aclnn_runner.h"`

2. **修改CreateRunner方法**
   - 添加设备检测逻辑
   - 在910B/950设备上使用 `<Operator>AclnnRunner`
   - 在其他设备上保持原有 `<Operator>OpsRunner`

3. **更新设备支持**
   - 修改 `CreateOperation` 函数
   - 支持910B/950训练设备
   - 保持对其他设备的支持

### 步骤4: 构建和验证

1. **编译验证**
   - 确保代码通过编译
   - 检查是否有编译错误或警告

2. **功能验证**
   - 在910B/950设备上验证ACLNN实现
   - 在其他设备上验证原有实现
   - 确保功能正确性

3. **性能验证**
   - 对比ATB和ACLNN实现的性能
   - 确保在910B/950设备上获得性能提升

<details>
<summary>技术要点（点击展开）</summary>

### 核心实现

1. **参数映射**
   - ATB参数到ACLNN参数的映射
   - 处理输入输出张量的转换

2. **设备检测**
   - 使用 `GetSingleton<Config>().Is910B()` 检测 910B 设备类型
   - 使用 `GetSingleton<Config>().Is950()` 检测 950 设备类型
   - 根据设备类型选择合适的 runner

3. **内存管理**
   - 自动管理ACLNN工作空间
   - 合理处理张量数据指针

4. **错误处理**
   - 完善的日志记录
   - 错误码处理和返回

### 性能优化

1. **ACLNN加速**
   - 利用ACLNN的硬件优化
   - 减少内存访问开销

2. **条件编译**
   - 根据设备类型选择最优实现
   - 保持代码兼容性

### 注意事项

1. **设备兼容性**
   - 确保在不同设备上的正确运行
   - 处理设备特定的参数和约束

2. **接口一致性**
   - 保持原有ATB接口不变
   - 确保上层应用无需修改

3. **测试覆盖**
   - 编写针对不同设备的测试用例
   - 验证性能和功能正确性

4. **错误处理**
   - 完善的错误处理机制
   - 详细的日志记录
</details>

## 示例使用

### 步骤1: 准备环境

```bash
# 确保CANN环境正确配置
# 确保ATB代码库已更新
```

### 步骤2: 应用迁移

```bash
# 在ATB代码库中应用迁移
git apply <operator>_aclnn_migration.patch
```

### 步骤3: 编译验证

```bash
# 编译ATB代码
cd ascend-transformer-boost
mkdir build && cd build
cmake ..
make -j
```

### 步骤4: 功能测试

```bash
# 运行算子测试
./tests/apitest/opstest/python/operations/<operator>/test_<operator>.py
```

### 步骤5: 性能测试

```bash
# 对比ATB和ACLNN实现的性能
python performance_benchmark.py
```

<details>
<summary>常见问题（点击展开）</summary>

### Q: 编译失败怎么办？
**A:** 检查CANN版本是否兼容，确保ACLNN接口可用。

### Q: 910B/950设备上性能没有提升？
**A:** 检查ACLNN函数是否正确加载，验证工作空间配置。

### Q: 其他设备上功能异常？
**A:** 确保设备检测逻辑正确，验证原有实现未受影响。
</details>

## 总结

通过该工具，开发者可以轻松将ATB算子迁移到ACLNN实现，在910B/950设备上获得更好的性能，同时保持对其他设备的兼容性。该实现遵循项目代码规范，与现有ACLNN runner实现保持一致，确保代码质量和可维护性。

## 执行结果

> ⚠️ **执行后填写**：技能执行完成后，参照下方格式填写实际执行结果。

### 检查点检查表

| 步骤 | 检查点描述 | 状态 |
|------|-----------|------|
| 1 | ATB 源码目录存在 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 1 | 原有 Operation 源文件定位 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 1 | 现有 ACLNN runner 参考定位 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 2 | `<op>_aclnn_runner.h` 创建 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 2 | `<op>_aclnn_runner.cpp` 创建 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 2 | `BuildAclnnVariantPack` 实现 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 2 | `SetAclNNWorkspaceExecutor` 实现 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 2 | `LaunchAclnnKernel` 实现 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 2 | `LoadMethod` 实现 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 3 | CreateRunner 包含 910B/950 分支 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 4 | 编译成功 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 5 | 非 910B/950 设备功能正常 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 5 | 910B 设备 ACLNN 路径正常 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |
| 5 | 950 设备 ACLNN 路径正常 | ✅ PASS / ❌ FAIL / ⏭️ SKIP |

**VERDICT: ✅ SUCCESS / ⚠️ PARTIAL / ❌ FAILED / ⏭️ SKIPPED**

### 问题列表（若有）

| 等级 | 检查点 | 问题描述 | 建议 |
|------|--------|---------|------|
| 🔴 CRITICAL | - | - | - |
| 🟡 WARNING | - | - | - |

### 执行摘要

- **执行时间**：
- **执行环境**：
- **算子名称**：
- **持续时间**：
- **通过率**：
