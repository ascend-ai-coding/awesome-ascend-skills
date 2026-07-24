# PIPE_ALL 同步问题定位流程

## 概述

当确认精度问题由数据同步失败导致时，通过二分插入 PIPE_ALL 定位缺少同步的 CCE 文件。

## 前置条件

1. 所有 Pass 校验通过但精度仍有问题
2. 已通过 `enableDebug_{true}` 验证：插入全量 PIPE_ALL 后精度通过

## 定位流程

### 步骤一：环境配置

**修改配置文件**：`framework/src/interface/configs/tile_fwk_config.json`

```json
{
    "global": {
        "codegen": {
            "parallel_compile" : 1,
            "fixed_output_path": true,
            "force_overwrite": false
        }
    }
}
```

**关键配置**：
- `fixed_output_path=true`：CCE 固定生成在 `./kernel_aicore/`
- `force_overwrite=false`：不覆盖已修改的 CCE 文件

**重新编译安装**：

```bash
python3 -m pip install . --verbose
```

### 步骤二：运行算子生成 CCE 文件

在算子目录下,执行算子测试用例，生成 CCE 文件：

```bash
python3 test_operator.py
```

执行后会在当前目录生成 `kernel_aicore/` 目录，包含所有 CCE 文件。

### 步骤三：执行二分定位

脚本自动定位：文件级二分 → 行级二分 → 具体代码行。

#### 参数说明

| 参数 | 说明 |
|-----|------|
| `--kernel-dir` | kernel_aicore 目录路径（必选） |
| `--test-cmd` | 测试命令（必选，不支持 shell 管道） |
| `--run-dir` | 测试运行目录，默认当前目录 |
| `--info` | 查看所有 CCE 文件 |
| `--cce-file` | 指定文件，仅行级二分 |
| `--no-line-search` | 仅文件级，跳过行级 |

#### 常用命令

```bash
# 查看文件信息
python3 scripts/binary_pipeall_sync.py --kernel-dir kernel_aicore --info

# 定位到文件（仅文件级）
python3 scripts/binary_pipeall_sync.py --kernel-dir kernel_aicore --test-cmd "python3 test.py" --no-line-search

# 定位到代码行（完整流程）
python3 scripts/binary_pipeall_sync.py --kernel-dir kernel_aicore --test-cmd "python3 test.py"

# 已知文件，定位行（仅行级）
python3 scripts/binary_pipeall_sync.py --kernel-dir kernel_aicore --cce-file kernel_aicore/problem.cpp --test-cmd "python3 test.py"
```

#### 定位逻辑

文件级：测试文件 [left, mid] 插入 PIPE_ALL
- PASS → 问题在已插入部分
- FAIL → 问题在未插入部分

行级：测试操作 [left, mid] 插入 PIPE_ALL
- PASS → 问题在已插入部分
- FAIL → 问题在未插入部分

#### 输出结果

```
Found: file [5] → kernel_aicore/Tensor_xxx.cpp
Found: line [85] → TLoad(ubTensor_14, gmTensor_15, ...)
```

## 工作原理

### PIPE_ALL 插入逻辑

脚本在每个重要操作后插入 `pipe_barrier(PIPE_ALL)`：

| 操作类型         | 正则匹配                 | 示例                         |
| ---------------- | ------------------------ | ---------------------------- |
| 阶段标记         | `SUBKERNEL_PHASE`        | `SUBKERNEL_PHASE1`           |
| 计算操作         | `^\s*T[A-Z]`             | `TCast/TMul/TAdd/TDiv/TExp`  |

**排除操作**：变量声明、注释、空行、`}`。

### 编译和测试流程

```
插入 PIPE_ALL 到 CCE 文件
  ↓
备份原文件（.bak）
  ↓
编译：make -f kernel_aicore/Makefile_*.compile
  ↓
运行测试：python3 test.py --run_mode npu
  ↓
解析输出判断 PASS/FAIL
  ↓
恢复原文件（删除 .bak）
```

### 二分收敛保证

脚本每轮结束后自动恢复原始 CCE 文件，确保：
- 每轮从干净状态开始
- 不同范围的测试不会相互干扰

## 结果分析

定位到问题 CCE 文件后：

### 步骤一：查看文件内容

```bash
cat kernel_aicore/<cce_file>
```

**关键位置**：
- `SUBKERNEL_PHASE` 切换处
- `TCast/TLoad/TStore` 等数据操作后
- `set_flag/wait_flag` 同步对之间

### 步骤二：定位具体同步问题行（自动化）

脚本自动执行行级二分：

**定位原理**：

1. 提取问题文件中所有可插入操作的位置（SUBKERNEL_PHASE、T[A-Z]）
2. 在操作位置列表中二分：测试前半段位置插入 PIPE_ALL
3. PASS → 问题在已插入部分（缺少同步被修复）
4. FAIL → 问题在未插入部分
5. 收敛到单个操作位置

**关键代码行特征**：
- 数据搬运操作后缺少同步：`TLoad/TStore` 后
- 计算操作后缺少同步：`TAdd/TMul` 后
- PHASE 切换处缺少同步：`SUBKERNEL_PHASE1` 后

### 步骤三：映射到前端源代码

定位到问题代码行后，使用 locate_source_line.py 映射到前端代码：

```bash
python3 pypto/.agents/skills/pypto-aicore-error-locator/scripts/locate_source_line.py \
    <cce_file> \
    <program_json_path> \
    <problem_line_number>
```

**参数说明**：
- `<cce_file>`：问题 CCE 文件路径
- `<program_json_path>`：program.json 文件路径（在 output/output_xxx/ 目录下）
- `<problem_line_number>`：CCE 文件中的问题行号

### 步骤四：分析并修复

根据映射结果分析问题：

| 映射结果 | 处理方案 |
|---------|---------|
| 成功映射到前端代码 | 检查前Pass 中添加同步逻辑是否有误或反馈框架开发 |
| 无法映射 | 手动分析 CCE 代码逻辑，确定缺少同步的具体位置 |

## 注意事项

1. 测试命令必须输出明确的 PASS/FAIL 标记
2. `force_overwrite=false` 只影响 JIT 编译，手动修改会被保留
3. 脚本自动恢复 .cpp 文件，每轮干净状态
4. 编译在 `kernel_aicore` 的父目录执行

## 相关文档

| 文档                                                         | 内容                |
| ------------------------------------------------------------ | ------------------- |
| [insert_sync.cpp](../../../../../framework/src/passes/block_graph_pass/insert_sync.cpp) | enableDebug 实现    |
| [tile_fwk_config.json](../../../../../framework/src/interface/configs/tile_fwk_config.json) | 编译配置            |
| [precision-pass/SKILL.md](../SKILL.md)                       | Pass 精度校验主流程 |