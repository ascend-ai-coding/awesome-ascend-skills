---
name: external-cannbot-ops-pypto-precision-compare-precision-pass
description: Pass精度校验子技能。开启PreCheck/PostCheck进行全链路Pass校验，通过pass校验定位报错Pass，使用pass_compare逐Op对比定位具体问题Op，支持动态shape上板数据打印验证和IR图分析辅助定位。
  当需要通过 Pass 校验（PreCheck/PostCheck）定位报错 Pass 或具体问题 Op 时使用此技能。
original-name: precision-pass
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-02'
synced-commit: 342679f37ec1b052a8b64d6c9fd59f67e0c89073
license: UNKNOWN
---

## 快速诊断

| 场景 | 错误码 | 处理方法 |
|-----|-------|---------|
| 前端问题 | `0xB4001U` | 调用 `precision-verify` 子技能 |
| OP 报错 | `0xB200FU` | 检查 IR 图，动态shape需打印验证 |
| Pass精度问题 | `0xB4001U` | PreCheck/PostCheck → pass_compare |
| 无报错但精度异常 | 无 | 调用 `precision-binary-search` 子技能 |

```
精度问题 → 查看验证日志 → 按错误码选择处理流程：
├─ 0xB4001U (tensor_graph) → 前端问题 → precision-verify
├─ 0xB200FU (OP报错) → IR图分析 → 动态shape则打印验证
├─ 0xB4001U + Pass名 → Pass精度 → PreCheck/PostCheck
└─ 无报错 → 二分前端 → precision-binary-search
```

---

## 目录

1. [简介](#简介)
2. [环境与配置](#环境与配置)
3. [操作步骤](#操作步骤)
4. [错误码速查表](#错误码速查表)
5. [问题处理流程](#问题处理流程)
6. [打印上板信息](#打印上板信息)
7. [IR图分析](#ir图分析)
8. [注意事项](#注意事项)

---

## 简介

验证 PyPTO Pass 侧精度问题，定位问题来源（前端/Pass/Codegen/Machine）。

> `tensor_graph Verify FAIL` → 前端问题，直接调用 `precision-verify` 子技能。

---

## 环境与配置

### 环境变量配置

| 环境变量 | 设置时机 | 说明 |
|---------|---------|------|
| `ASCEND_WORK_PATH` | 运行测试前必须设置 | 组件日志输出目录 |
| `ASCEND_GLOBAL_LOG_LEVEL` | 建议设为 0（DEBUG） | 获取详细调试信息 |
| `TILE_FWK_DEVICE_ID` | NPU 模式运行前必须设置 | 指定 NPU 设备 ID |

**设置示例**：
```bash
# 必需：设置工作目录
export ASCEND_WORK_PATH="/path/to/work/directory"

# 必需：设置日志级别（0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR）
export ASCEND_GLOBAL_LOG_LEVEL=0

# 可选：设置设备 ID（NPU 模式）
export TILE_FWK_DEVICE_ID=0
```

**验证环境变量**：
```bash
echo "ASCEND_WORK_PATH: $ASCEND_WORK_PATH"
echo "ASCEND_GLOBAL_LOG_LEVEL: $ASCEND_GLOBAL_LOG_LEVEL"
echo "TILE_FWK_DEVICE_ID: $TILE_FWK_DEVICE_ID"
```

---

## 操作步骤

### 步骤一：配置校验开关

#### verify_options 配置

在 PyPTO 算子实现文件中配置：

```python
verify_options = {
    "enable_pass_verify": True,            # 启用Pass验证（必须）
    "pass_verify_pass_filter": ["all"],    # 验证所有Pass
    "pass_verify_save_tensor": True,       # 保存中间数据
}

@pypto.frontend.jit(verify_options=verify_options)
def your_kernel(
    input0: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    input1: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    output: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    output[:] = input0 + input1
```

| 配置项 | 说明 | 默认值 |
|-------|------|-------|
| `enable_pass_verify` | 启用 Pass 验证 | `False` |
| `pass_verify_pass_filter` | 过滤要验证的 Pass | `[]` |
| `pass_verify_save_tensor` | 保存 Pass 中间数据 | `False` |

> **注意**：Shape 必须具体，不能使用占位符；返回类型必须使用输出参数形式。

#### tile_fwk_config.json 配置

配置文件位置：`framework/src/interface/configs/tile_fwk_config.json`

```json
{
    "global": {
        "pass": {
            "default_pass_configs": {
                "print_graph": true,
                "dump_graph": true,
                "pre_check": true,    // Pass精度问题时开启
                "post_check": true    // Pass精度问题时开启
            }
        }
    }
}
```

| 配置项 | 说明 | 使用场景 |
|-------|------|---------|
| `pre_check` | Pass 执行前校验 | Pass精度问题时开启 |
| `post_check` | Pass 执行后校验 | Pass精度问题时开启 |
| `dump_graph` | 保存 IR 图 | 分析 Pass 处理结果 |
| `print_graph` | 打印 IR 图 | 快速查看图结构变化 |

> 大数据量时（Shape T>1000）：缩小shape参数、删除LOOP的unrolllist参数、增大tileshape（cube_tile_shapes、vec_tile_shapes）。

### 步骤二：编译运行

```bash
python3 -m pip install . --verbose
python3 your_test_case.py
```

输出目录：`./output/output_*`（验证数据）、`$ASCEND_WORK_PATH/log/`（日志）

### 步骤三：分析验证结果

错误码定义：`framework/src/interface/interpreter/verify_error.h`

> **判断标准**：只看 CodegenPreproc Pass（最后一个 Pass）是否通过。中间 Pass 报错（如 ReplaceTensor、SplitK 等出现 `VERIFY_RESULT_MISMATCH`）忽略，只要 CodegenPreproc PASS 即表示精度正确。只有 CodegenPreproc FAIL 时才需要用 `pass_compare.py` 进一步定位。

---

## 错误码速查表

| 错误码 | 名称 | 阶段 | 处理方法 |
|-------|------|------|---------|
| `0xB4001U` | VERIFY_RESULT_MISMATCH | 前端/Pass | 参考[问题处理流程](#问题处理流程) |
| `0xB200FU` | RUNTIME_EXCEPTION | Pass | 检查 OP 属性，参考 IR 图 |
| `0xB0001U` | VERIFY_NOT_ENABLE | 环境 | 检查 `torch >= 2.1.0` |
| 其他 | — | 未知 | 联系开发人员 |

---

## 问题处理流程

### 情况一：tensor_graph FAIL → 调用 `precision-verify`

### 情况二：Pass级别FAIL

> **前置配置**（必须）：按照 [操作步骤-步骤一](#步骤一配置校验开关) 配置 `verify_options` 和 `tile_fwk_config.json`。

**2.1 OP报错**：对比 Before/After IR，确认是否误报。

动态shape场景：IR显示符号变量（如 `sym_15_dim_0`）→ 参考 [docs/trouble_shooting/machine.md](../../../../docs/zh/trouble_shooting/machine.md) 进行排查。

**2.2 精度问题**：

> **重要判断标准**：只看最后一个 Pass（`Pass_36_CodegenPreproc`）是否通过。中间 Pass报错但 CodegenPreproc PASS → 属于误报，可忽略。

**处理流程**：
```
配置PreCheck/PostCheck → 编译运行 → 检查interpreter.log
    ├─ CodegenPreproc PASS → pass层精度正确，无需进一步调试
    └─ CodegenPreproc FAIL → 根据日志确定失败pass，使用pass_compare定位问题Pass → pass_compare定位问题Op
```

---

**pass_compare.py定位问题Op**：

定位到问题Pass后，使用 `pass_compare.py` 进一步定位具体Op：

```bash
python3 tools/verifier/pass_compare.py \
    --p <FailedPass> <GoldenPass> \
    --verify_path=/path/to/verify_data 
```

| 参数 | 必选 | 说明 |
|------|------|------|
| `--p` | 是 | 两个 Pass 名称，空格分隔。第一个是问题 Pass，第二个是 golden Pass（前一个通过的 Pass） |
| `--verify_path` | 是 | verify 数据目录。可传1个（两个 Pass 在同一目录）或2个（分别对应两个 Pass） |
| `--func` | 否 | 指定对比的函数名，可传多个，默认对比所有函数 |
| `--atol` | 否 | 绝对容差，默认 1e-3 |
| `--rtol` | 否 | 相对容差，默认 1e-3 |
| `--topk` | 否 | 失败时打印前 k 个差异元素，默认 1000 |

> **定位结果**：pass_compare.py 生成 `verify_graph_result_cmp~Pass_xx~PassA~Pass_yy_PassB~timestamp.csv`，逐 Op 记录对比结果（PASS/FAIL/Skip），失败 Op 即为问题 Op。

**自动分析脚本**：运行完 pass_compare.py 后，直接分析生成的 CSV 定位问题 Op：

```bash
python3 -c "
import pandas as pd
cmp = pd.read_csv('<生成的CSV文件路径>')
total = len(cmp)
passed = sum(str(r).strip() == 'True' for r in cmp['AB>RESULT'])
failed = sum(str(r).strip() == 'False' for r in cmp['AB>RESULT'])
skipped = sum(str(r).strip() == 'Skip' for r in cmp['AB>RESULT'])
print(f'Total: {total}, Pass: {passed} ({passed/total*100:.1f}%), Fail: {failed}, Skip: {skipped}')
print()
if failed > 0:
    fails = cmp[cmp['AB>RESULT'].astype(str).str.strip() == 'False']
    print('=== Fail 按 opcode 分布 ===')
    for opcode, cnt in fails['B>:opcode'].value_counts().items():
        sub = fails[fails['B>:opcode'] == opcode]
        max_mae = sub['AB>mae'].astype(float).max()
        print(f'  {opcode:25s}  {cnt:4d}次  maxMAE={max_mae:.6f}')
    print()
    print('=== Fail 按 symbol 分布（最可能的问题 Op）===')
    print(fails[':symbol'].value_counts().to_string())
"
```

**分析结果判断**：

| 结果 | 含义 | 后续动作 |
|------|------|---------|
| 存在 Fail Op | Fail 的 Op 即为精度问题来源 | 针对该 Op 检查实现逻辑、数据类型、shape 处理等 |
| 全部 Pass 但精度仍有问题 | Pass 层未检出差异，问题在上板执行阶段 | 调用 [precision-binary-search](../precision-binary-search/SKILL.md) 进行上板二分定位 |

**移除 Pass 校验配置**

Pass 校验完成后，**移除校验配置**，避免影响后续调试：

**移除 verify_options 配置**：

```python
# 移除 Pass 校验配置
@pypto.frontend.jit()  # 移除 verify_options 参数
def your_kernel(...)
```
恢复`tile_fwk_config.json`文件中pre_check，post_check配置

### 情况三：Pass级别都Pass，精度仍有问题 → 检查同步问题

如果所有 Pass 校验都通过但算子精度仍有问题，可能是 pipeline 同步不及时导致的数据竞争。

**快速验证同步问题**：

1. 修改 `framework/src/passes/block_graph_pass/insert_sync.h`，将 `bool enableDebug_{false}` 改为 `bool enableDebug_{true}`
2. 重新编译安装 pypto：`python3 -m pip install . --verbose`
3. 重新执行算子。若精度通过 → 说明数据同步不足导致精度失败，需定位具体缺少同步的位置

若确认数据同步导致精度失败，执行以下定位流程：

详细步骤请参考：**[references/pipe_all.md](references/pipe_all.md)**

**定位流程概览**：

1. 二分定位问题 CCE 文件（使用 `binary_pipeall_sync.py`）
2. 在问题 CCE 文件内手动二分插入 `pipe_barrier(PIPE_ALL)`，定位具体问题行
3. 使用 `locate_source_line.py` 映射问题行到前端源代码
4. 分析并修复同步问题

## 打印上板信息

**前置条件**：已通过 `precision-verify` 或 `precision-binary-search` 定位到具体Op。

用于：打印上板tensor数据、验证动态shape/offset值、定位AICORE执行异常。

详细排查方法请参考：**[docs/trouble_shooting/machine.md](../../../../docs/zh/trouble_shooting/machine.md)**

### 打印环境配置

**tile_fwk_config.json 配置**：

```json
{
    "global": {
        "codegen": {
            "fixed_output_path": true,    // 固定CCE输出路径
            "force_overwrite": false,     // 不覆盖已修改的CCE文件
            "parallel_compile": 1         // 单线程编译
        }
    }
}
```

| 配置项 | 正确值 | 说明 |
|-------|-------|------|
| `fixed_output_path` | `true` | CCE固定生成在 `./kernel_aicore/` |
| `force_overwrite` | `false` | 不覆盖手动修改的CCE文件 |
| `parallel_compile` | `1` | 单线程编译，便于调试 |

**aicore_print.h 打印开关**：

确保 `framework/src/interface/machine/device/tilefwk/aicore_print.h` 中：
```c
#define ENABLE_AICORE_PRINT 1   // 必须为 1
```

### 可打印内容

| 内容 | 方法 | 说明 |
|-----|------|------|
| GM tensor数据 | `AiCorePrintGmTensor` | DDR/GM上的tensor |
| UB tensor数据 | `AiCorePrintUbTensor` | UB上的tensor |
| Shape变量值 | `AicoreLogF` | 动态shape实际值 |
| Offset值 | `AicoreLogF` | 动态offset实际值 |
---

## IR图分析

判断误报、辅助定位。详见 `pypto/.agents/skills/pypto-pass-error-locator/references/ir-analysis-guide.md`

```bash
# 查询OP详情
python3 pypto/.agents/skills/pypto-pass-error-locator/scripts/get_op_info.py \
    --ir-file <IR文件> --op-magic <ID>

# 列出所有OP
python3 pypto/.agents/skills/pypto-pass-error-locator/scripts/get_op_info.py \
    --ir-file <IR文件> --list-ops
```

---

## 注意事项

1. Pass精度判断：只看 CodegenPreproc 是否通过
2. tensor_graph FAIL → 调用 precision-verify
3. 无报错但精度异常 → 调用 precision-binary-search
4. 动态shape验证/AICORE异常排查：参考 [docs/trouble_shooting/machine.md](../../../../docs/zh/trouble_shooting/machine.md)
5. 打印配置：`fixed_output_path=true`, `force_overwrite=false`
6. 打印限制：元素数量 ≤ 80
7. 配置备份：修改配置前建议备份原文件
8. 校验完成后移除配置：移除 `verify_options` 参数和 `tile_fwk_config.json` 中的校验开关，重新编译安装

---

## 相关文档

| 文档 | 内容 |
|------|------|
| [docs/trouble_shooting/machine.md](../../../../docs/zh/trouble_shooting/machine.md) | MACHINE组件错误码与排查指南 |
| [pypto-aicore-error-locator](../../pypto-aicore-error-locator/SKILL.md) | AICORE错误定位Skill |