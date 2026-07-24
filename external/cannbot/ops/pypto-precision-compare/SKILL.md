---
name: external-cannbot-ops-pypto-precision-compare
description: PyPTO 算子精度问题调试技能。提供两种精度对比方法：文件保存方法（使用 pypto.pass_verify_save 和 torch.save）和二分对比方法（使用检查点
  tensor）。当需要调试 PyPTO 算子精度、定位精度差异来源、进行中间结果对比时使用此技能。
original-name: pypto-precision-compare
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# PyPTO 算子精度对比技能

提供两种精度对比方法，用于快速定位 PyPTO 算子中导致精度问题的具体 op。

## 执行逻辑

当用户调用本技能时，首先检查用户是否指定了 `mode` 参数：

### 情况一：关键词触发模式（手动干预）

如果用户描述中包含以下**关键词**，则**跳过所有判断**，直接执行对应的子技能：

| 关键词 | 触发模式 | 执行子技能 |
|--------|---------|-----------|
| **精度工具**、**精度对比**、**文件保存**、`pass_verify_save` | `mode="verify"` | `precision-verify` |
| **Pass精度**、**Pass校验**、**PreCheck**、**PostCheck** | `mode="pass"` | `precision-pass` |
| **二分**、**上板二分**、**检查点tensor** | `mode="binary"` | `precision-binary-search` |

**示例触发语句**：
- "用精度工具方式调试" → 直接调用 `precision-verify`
- "用Pass精度校验" → 直接调用 `precision-pass`
- "用二分方式定位" → 直接调用 `precision-binary-search`
- "用文件保存方法对比" → 直接调用 `precision-verify`

> 注意：此时假设用户已经明确知道问题所在，不需要再执行 tensor_graph 校验流程。

### 情况二：未指定模式（全自动）

如果用户未指定 `mode`（或 `mode="auto"`），则执行**标准决策树流程**：

**步骤 1：开启 tensor_graph 校验**

**⚠️ 编译安装（关键步骤，必须严格执行）**：

执行算子之前，**必须**先进行编译：

```bash
python3 -m pip install . --verbose
```

在 PyPTO 算子实现文件中配置 `verify_options`，开启 tensor_graph 校验：

```python
verify_options = {
    "enable_pass_verify": True,
    "pass_verify_save_tensor": True,  # 保存中间tensor数据用于分析
    "pass_verify_pass_filter": []     # tensor_graph 校验中跳过pass校验
}

@pypto.frontend.jit(verify_options=verify_options)
def your_kernel(...) 
```

设置 Golden 数据（**必须严格按照计算 golden → 设置 golden → 执行算子顺序**）：

```python
# 计算 golden
torch_output = torch.add(input_data0, input_data1)

# 设置 golden（必须放在 pypto 算子运行前设置！此处torch的数据必须在cpu上） 
pypto.set_verify_golden_data(goldens=[None, None, torch_output.cpu()])

# 执行 pypto 算子
pypto_output = your_kernel(input_data0, input_data1)
```

**步骤 2：查看 tensor_graph 校验结果**

运行测试后，查看校验结果（output 目录会在执行命令的当前路径下生成）：

```bash
# 查看最新 verify 目录
ls -lt output/output_*/verify_* | head -n 1

# 查看 tensor_graph 校验结果
cat output/output_*/verify_*/interpreter.log
```

打开 `interpreter.log` 文件，查看校验状态。

如果执行遇到以下报错：

| 问题现象 | 可能原因 | 解决方案 |
|---------|---------|---------|
| **aicore error / MPU 地址访问无效** | NPU 卡资源冲突或状态异常 | 1. 更换 `TILE_FWK_DEVICE_ID` 到空闲卡<br>2. 重新执行 |
| **verify 日志无内容** | Golden 数据未设置或顺序错误 | 确保"计算 golden → 设置 golden → 执行算子"顺序，检查set_verify_golden_data输入输出是否匹配 |

**移除 tensor_graph 校验配置**

确认 tensor_graph 有校验结果后，**移除校验配置**：

```python
# 移除 verify_options 配置
@pypto.frontend.jit()  # 移除 verify_options 参数
def your_kernel(...)

# 移除 golden 数据设置代码
# 删除: pypto.set_verify_golden_data(...)
```

**步骤 3：根据校验结果选择调试路线**

```
查看 Tensor Graph 校验结果
    │
    ├── Tensor Graph FAIL → 使用精度工具对比法保存中间结果比对，找出首个失败的 op
    │                       参考 → precision-verify/SKILL.md
    │
    └── Tensor Graph PASS → 进行 Pass 校验，找到首个出错的 Pass，dump Pass 数据对比
                        │
                        ├── Pass 校验 FAIL → 定位出错的 Pass/OP
                        │                   参考 → precision-pass/SKILL.md
                        │
                        └── Pass 校验 PASS → 上板二分定位，找到首个出错的 op
                                            参考 → precision-binary-search/SKILL.md
```

#### 情况 A：tensor_graph Verify FAIL → 前端代码问题

| 项目         | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| **日志特征** | `[VERIFY]:ErrCode: FB4001! tensor_graph Verify for 1 data view list index 0 result FAILED` |
| **错误码**   | `0xB4001U`（VERIFY_RESULT_MISMATCH）                         |
| **原因**     | 前端代码书写问题，构图阶段已出错                             |
| **处理**     | 使用 **精度工具对比法** 定位首个失败的 op                    |

**操作步骤**：

1. 阅读 [precision-verify/SKILL.md](precision-verify/SKILL.md) 了解详细步骤
2. 在 kernel 中添加 `pypto.pass_verify_save()` 调用
3. 在 golden 中添加 `torch.save()` 调用
4. 运行测试生成数据
5. 使用对比工具分析结果，找出首个不匹配的检查点

#### 情况 B：tensor_graph Verify PASS → Pass 精度校验

| 项目         | 说明                                                  |
| ------------ | ----------------------------------------------------- |
| **日志特征** | tensor_graph 验证 PASS，需进一步定位精度问题来源      |
| **原因**     | 问题可能在 Pass、Codegen 或 Machine 执行阶段          |
| **处理**     | 使用 Pass 精度校验定位问题 Pass，如未定位则上板二分   |

**操作步骤**：

1. 阅读 [precision-pass/SKILL.md](precision-pass/SKILL.md) 了解详细步骤
2. 配置 PreCheck/PostCheck 开启 Pass 级别验证
3. 分析验证结果，定位问题 Pass
4. 若 Pass 校验未定位问题，继续使用上板二分方法

#### 情况 C：Pass 校验未定位问题 → 上板二分定位

| 项目         | 说明                                                  |
| ------------ | ----------------------------------------------------- |
| **日志特征** | 前两种方式定位未发现明显问题，但上板结果与 golden 不一致  |
| **原因**     | 问题可能在 Codegen 或 Machine 执行阶段                |
| **处理**     | 添加中间变量的返回输出，二分对比上板数据找到首个出错的 op |

**操作步骤**：

1. 阅读 [precision-binary-search/SKILL.md](precision-binary-search/SKILL.md) 了解详细步骤
2. 修改 kernel 函数签名，添加检查点 tensor 参数
3. 修改 golden 函数，返回检查点数据
4. 修改测试函数，创建检查点 tensor 并对比
5. 从关键计算点开始，二分定位精度问题，直到找到具体出错的 op

**步骤 4：子技能执行完成后的处理**

- 如果子技能返回"已定位到问题 op/Pass" → 结束，汇报问题位置
- 如果子技能返回"需要继续 Pass 校验" → 调用 `precision-pass`
- 如果子技能返回"需要继续二分" → 调用 `precision-binary-search`

## 方法对比

| 特性         | 精度工具对比法                                               | Pass 精度校验法                                   | 二分对比法                                       |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------- | ------------------------------------------------ |
| **实现方式** | 使用 `pypto.pass_verify_save()` 保存到文件，使用 `torch.save()` 保存 golden | 使用 PreCheck/PostCheck、pass_compare 和上板比对 | 使用检查点 tensor 作为输入参数，在内存中直接对比 |
| **适用场景** | Tensor Graph 校验失败，需要快速定位首个失败的 op             | Tensor Graph 校验通过，定位问题 Pass             | Tensor Graph 和 Pass 校验通过，需上板真实数据    |
| **定位粒度** | Op 级别                                                      | Pass 级别                                         | Op 级别                                          |
| **代码修改** | 不需要修改 kernel 函数签名                                   | 配置验证开关                                      | 需要修改 kernel 函数签名，添加检查点参数         |
| **使用难度** | 简单，只需添加检查点调用                                     | 中等，需配置和上板比对                            | 较复杂，需要管理检查点 tensor                    |


## 参考资料

- PyPTO API: `pypto/docs/zh/api/`
- pass_verify_save API: `pypto/docs/zh/api/others/pypto-pass_verify_save.md`