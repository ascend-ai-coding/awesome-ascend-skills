# 文件保存方法

使用 `pypto.pass_verify_save()` 和 `torch.save()` 保存中间结果到文件，然后使用对比工具分析。

## 核心原理

1. **数据对齐**：golden 和 kernel 的计算逻辑、切块方式、数据维度必须完全一致，如果实现不一致，改写golden函数
2. **kernel 保存**：使用 `pypto.pass_verify_save()` 保存中间结果（循环场景使用 `cond=(idx == 0)`）
3. **golden 保存**：使用 `torch.save()` 保存中间结果（循环场景使用 `if idx == 0:`）
4. **数据对比**：使用对比工具检查 kernel 的 `.data` 文件和 golden 的 `.pt` 文件
5. **全量对比**：一次性开启所有关键检查点，对比所有中间结果，定位第一个计算结果不同的 op
6. **数据类型**：保存什么类型就读取什么类型，kernel 和 golden 的数据类型和 shape 必须完全一致，不一致则直接比对失败

## 核心原则

### 原则 1：插入检查点（kernel 函数）

```python
# 必须启用验证选项
verify_options = {
    "enable_pass_verify": True,
    "pass_verify_save_tensor": True,
    "pass_verify_pass_filter": []
}

@pypto.frontend.jit(verify_options=verify_options)
def kernel(inputs, outputs):
    # 基础场景
    temp1 = pypto.compute_op1(inputs[0])
    pypto.pass_verify_save(temp1, "1_after_op1")

    # 循环场景：只保存 idx=0 的数据
    # 多层循环使用 cond=((idx1 == 0) * (idx2 == 0))
    for idx in range(batch_size):
        temp = pypto.compute_op(inputs[idx])
        pypto.pass_verify_save(temp, "2_loop_result", cond=(idx == 0))
```

### 原则 2：保存 golden 中间结果

```python
import os
import torch

operator_dir = f"{operator}"
os.makedirs(operator_dir, exist_ok=True)
output_dir = os.path.join(operator_dir, "golden_data")
os.makedirs(output_dir, exist_ok=True)

def golden(inputs, outputs):
    # 基础场景
    temp1 = compute_op1(inputs[0])
    torch.save(temp1, f"{output_dir}/golden_1_after_op1.pt")

    # 循环场景：只保存 idx=0 的数据
    for idx in range(batch_size):
        temp = compute_op(inputs[idx])
        if idx == 0:
            torch.save(temp, f"{output_dir}/golden_2_loop_result.pt")
```

**文件命名约定**：
- golden 文件：`golden_{序号}_{检查点名称}.pt`
- jit 文件：`{序号}_{检查点名称}_{number}.data`（自动生成）
- 检查点名称必须按计算顺序添加数字前缀，确保按顺序对比

### 原则 3：全量对比策略

一次性开启所有关键计算节点的检查点，对比所有中间结果，定位第一个计算结果不同的 op。

## 易错点及修正方案

### 1. 执行路径问题

**问题**：Output 文件和 pt 文件生成在执行路径下

**修正**：用 `-w` 参数指定工作目录
```bash
python3 .agents/skills/pypto-precision-compare/scripts/compare_accuracy.py -w /path/to/operator -v
```

### 2. 数据类型读取问题

**问题**：读取数据时要根据保存的类型读取（BF16/FP32/INT32 等）

**修正**：对比工具会根据 CSV 文件中的 dtype 自动判断数据类型：
- dtype=8: BF16 格式（2字节）
- dtype=7: FP32 格式（4字节）

### 3. 检查点插入位置问题

**原则**：检查点位置要一一对应
- 如果 kernel 在 A→B→C 三个步骤后都插入检查点，golden 也需要在对应步骤保存
- 保持两边计算逻辑和保存时机完全一致

### 4. 切块计算问题

**原则**：保持保存的数据维度一致

**解决思路**：
1. **golden 保存对应的切片数据**：golden 一次性计算完整数据，只保存与 kernel 对应的切片
2. **golden 改写为和 kernel 实现完全一致**（推荐）：golden 模拟 kernel 的循环结构和分块策略

### 5. 精度标准问题

**修正**：对比工具根据数据类型自动设置容差：

| 数据类型 | dtype | rtol | atol |
|---------|-------|------|------|
| INT16 | 2 | 0 | 0 |
| INT32 | 3 | 0 | 0 |
| INT64 | 4 | 0 | 0 |
| FP8 | 5 | 1e-1 | 1e-2 |
| FP16 | 6 | 1e-3 | 1e-3 |
| FP32 | 7 | 1e-3 | 1e-4 |
| BF16 | 8 | 5e-3 | 5e-2 |

### 6. 量化场景容差设置

**修正**：使用 `--rtol` 和 `--atol` 参数指定自定义容差：
```bash
python3 .agents/skills/pypto-precision-compare/scripts/compare_accuracy.py \
    --rtol 0.0078125 \
    --atol 0.001 \
    -v
```

### 7. 对比逻辑问题

**修正**：对比工具使用 `torch.isclose` 统计不匹配个数：
- 判断条件：不匹配个数 < 总数 * max(rtol, atol)

## 完整工作流程

### 步骤 1：插入检查点

在 jit 和 golden 函数中插入对应的检查点（参考原则 1 和 2）。

**循环场景关键点**：
- 使用 `cond=(idx == 0)` 只保存一批数据
- 确保 kernel 和 golden 保存相同的 idx 数据
- 在检查点名称中包含 idx 信息

### 步骤 2：运行测试生成数据

```bash
python3 test_operator.py
```

### 步骤 3：对比检查点

```bash
python3 .agents/skills/pypto-precision-compare/scripts/compare_accuracy.py -v
```

### 步骤 4：分析对比结果

根据对比结果定位问题：
- 查看哪些检查点不匹配
- 不匹配的检查点就是导致精度问题的位置
- 分析不匹配率、最大差异等指标

**⚠️ 重要检查事项**

1. **检查点数据类型一致性**
    - kernel 和 golden 的数据类型和 shape 必须完全一致，不一致则直接比对失败
    - 在 kernel 和 golden 中使用相同的数据类型保存

2. **审查 log 文件内容**
    - 查看生成的 `*_verify_result.log` 文件
    - 如果在量化类场景下中间结果略超过容差阈值，可以适当放宽，这类检查点不视作精度比对失败

### 步骤 5：绘制精度变化图（可选）

使用绘图脚本可视化精度变化趋势，帮助直观理解精度差异分布：

```bash
# 指定 operator_verify_result.log 文件路径（在 operator 目录下）
python3 .agents/skills/pypto-precision-compare/scripts/plot_accuracy.py /path/to/operator/operator_verify_result.log
```

**功能说明**：
- 从验证结果日志中提取所有检查点的 rtol/atol 数据
- 绘制双折线图：相对容差(rtol) 和绝对容差(atol)
- 生成精度数据汇总表格，- 输出 PNG 格式的可视化图表（保存在 operator 目录）

### 步骤 6：定位并修复### 步骤 5：定位并修复

根据对比结果定位到具体的 op，然后修复问题。

## 最佳实践

### 1. 检查点命名

使用有意义的名称，反映计算步骤，按计算顺序添加数字前缀。

### 2. 循环场景处理

- 使用 `cond=(idx == 0)` 确保只保存一批数据
- kernel 和 golden 必须保存相同的 idx 数据
- 避免生成过多文件

### 3. 全量对比模式

一次性开启所有关键计算节点的检查点，对比所有中间结果。

**使用场景**：
- 算子计算步骤较多，需要全面检查
- 已经知道大概的问题范围，需要精确定位
- 需要分析整个计算流程的精度变化

### 4. 清理调试代码

修复问题后：
```bash
rm -f operator/golden_data/*.pt
rm -rf output/output_*

# 移除调试代码：
# - 删除 pypto.pass_verify_save() 调用
# - 删除 verify_options 参数
# - 删除 golden 中的 torch.save() 调用
```

## 常见问题

### Q1: kernel 和 golden 保存的数据不一致

**原因**：kernel 保存 idx=0，但 golden 保存了其他 idx / golden 没有切块计算

**解决**：确保两者使用相同的条件，参考"易错点 4：切块计算问题"

### Q2: 找不到检查点文件

**检查**：
- jit 代码中是否使用了 `pypto.pass_verify_save()`
- 是否设置了 `verify_options={"enable_pass_verify": True}`
- 文件命名是否符合约定
- 是否在正确的目录下执行对比工具

## 通用对比工具

本方法提供了通用对比脚本，自动完成检查点扫描和对比：

```bash
# 自动检测并对比所有检查点
python3 .agents/skills/pypto-precision-compare/scripts/compare_accuracy.py

# 列出所有检查点
python3 .agents/skills/pypto-precision-compare/scripts/compare_accuracy.py --list

# 显示详细对比
python3 .agents/skills/pypto-precision-compare/scripts/compare_accuracy.py --verbose

# 指定工作目录
python3 .agents/skills/pypto-precision-compare/scripts/compare_accuracy.py -w /path/to/operator -v

# 指定 golden 文件所在目录（推荐）
python3 .agents/skills/pypto-precision-compare/scripts/compare_accuracy.py -w /path/to/operator -g /path/to/operator/golden_data -v
```

**工具功能**：
- ✓ 自动检测最新 output 目录
- ✓ 自动扫描所有检查点文件
- ✓ 智能匹配 jit 和 golden 文件
- ✓ 根据数据类型自动设置容差标准
- ✓ 统计不匹配率而非只看最大差异
- ✓ 自动分析并给出二分建议
- ✓ 支持详细元素级对比
- ✓ 按检查点名称开头的数字排序，确保按计算顺序对比

## 检查清单

使用文件保存方法时，确保：

- [ ] **步骤 1**：插入检查点
  - [ ] 设置 `verify_options={"enable_pass_verify": True, "pass_verify_save_tensor": True, "pass_verify_pass_filter": []}`
  - [ ] kernel 函数中使用 `pypto.pass_verify_save(tensor, fname)`
  - [ ] golden 函数中使用 `torch.save()` 保存中间结果
  - [ ] 循环场景使用 `cond=(idx == 0)` 和 `if idx == 0:`
  - [ ] 文件命名遵循约定（添加数字前缀）
  - [ ] 检查点插入位置要一一对应
  - [ ] 切块计算要保持一致
  - [ ] golden 和 kernel 的切块逻辑完全一致
- [ ] **步骤 2**：运行测试生成数据
  - [ ] golden 数据保存到独立文件夹
  - [ ] 避免与之前测试的 golden 文件混淆
- [ ] **步骤 3**：对比检查点
  - [ ] 在正确目录下使用通用工具
  - [ ] 如 golden 文件在独立文件夹，使用 `--golden-dir` 参数指定
  - [ ] 查看对比结果和不匹配率
  - [ ] 确认检查点按计算顺序对比
- [ ] **步骤 4**：继续二分
  - [ ] 根据对比结果判断问题位置
  - [ ] 在问题范围内插入新的检查点
  - [ ] 保持检查点命名前缀的连续性
- [ ] **步骤 5**：定位并修复
  - [ ] 定位到具体的 op
  - [ ] 修复问题
  - [ ] 重新验证
  - [ ] 清理调试代码
