---
name: external-cannbot-ops-pypto-op-develop
description: '当需要编写 PyPTO 算子实现时使用此 skill。基于需求规格、设计方案和参考实现，生成完整可运行的 PyPTO 算子实现与配套测试、文档。Triggers:
  实现算子、写 kernel、编写实现、写 impl、算子编码、开始编码、code the op、写 test、生成测试、写实现代码、op develop、kernel
  实现。'
original-name: pypto-op-develop
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-26'
synced-commit: ac5bbd2b4cf427d011874e11f8d1e8b1bef66eda
license: UNKNOWN
---

# PyPTO 算子实现

基于需求规格、设计方案和参考实现，完成代码实现、验证和 README 交付。

---

## 所需输入

### 需求规格信息

从 `SPEC.md` 中提取：

| 字段 | 用途 |
|------|------|
| 算子名称 | 文件命名、函数命名 |
| 数学公式 | 理解计算逻辑 |
| 输入/输出规格（shape、dtype） | 测试数据生成 |
| 支持的数据类型 | 测试覆盖、impl 类型处理 |
| 精度要求 | 测试容差设定 |
| 服务器类型 | 环境兼容性确认 |

### 设计方案信息

从 `DESIGN.md` 中提取：

| 字段 | 用途 |
|------|------|
| API 映射设计 | `{op}_impl.py` 核心实现 |
| 数据规格设计 | `test_{op}.py` 数据生成 + `{op}_impl.py` tensor 描述 |
| Tiling 策略 | `{op}_impl.py` tiling 配置 |
| Loop 结构设计 | `{op}_impl.py` kernel 逻辑 |
| 验证方案 | `test_{op}.py` 测试用例设计 |
| 性能指标 | `README.md` 性能说明 |

### 参考实现信息

从 `{op}_golden.py` 中提取：

| 信息 | 用途 |
|------|------|
| golden 函数名 `{op}_golden()` | `test_{op}.py` import 语句 |
| 参数签名 | `test_{op}.py` 数据准备 |
| 输出形式 | `{op}_impl.py` 输出 tensor 构造 |

如果以上信息不足，向用户逐步提问补充。

### 参考文件

| 文件 | 用途 | 加载时机 |
|------|------|----------|
| [references/test-template.py](references/test-template.py) | test 文件固定模板 | 生成 test_{op}.py 时读取 |
| [references/impl-template.py](references/impl-template.py) | impl 文件固定模板 | 生成 {op}_impl.py 时读取 |
| [references/execution-constraints.md](references/execution-constraints.md) | PyPTO 开发执行约束清单 | 进入实现阶段前必读；编码与自检时反复对照 |
| [references/error-code-troubleshooting.md](references/error-code-troubleshooting.md) | 错误码排查流程与常见错误码速查 | 验证失败时按流程排查 |
| [scripts/environment_prepare.sh](scripts/environment_prepare.sh) | 环境初始化脚本 | 环境准备阶段按需执行 |
| [scripts/list_idle_chip_ids.sh](scripts/list_idle_chip_ids.sh) | 输出当前可用 chip id 列表（兼容 910B / 910C） | 设置 `TILE_FWK_DEVICE_ID` 前执行 |

---

## 开发阶段

### 阶段一：环境准备

1. **检查 CANN 是否安装**
```bash
echo ${PATH} | grep cann-8.5.0
```

2. **检查 pto-isa 源码是否获取**
```bash
echo ${PTO_TILE_LIB_CODE_PATH}
```
- 如果路径不存在，执行 `scripts/environment_prepare.sh` 进行环境初始化
- 如果仍不成功，参考 `docs/install/prepare_environment.md` 获取 pto-isa 源码并设置环境变量

3. **验证关键文档及示例目录**
```bash
ls docs/api/
ls examples/
```

4. **设置 device_id**
```bash
# 查找空闲 chip id
`bash scripts/list_idle_chip_ids.sh`

# 根据输出结果设置环境变量（例如输出为 "0 1 3 ..." 时，先选 chip 0）
export TILE_FWK_DEVICE_ID=$(bash scripts/list_idle_chip_ids.sh | awk '{print $1}')
export PTO_TILE_LIB_CODE_PATH=./pto_isa/pto-isa/
```

- 如果脚本无输出，说明当前所有 chip 都被进程占用，不能随意设置 `TILE_FWK_DEVICE_ID`

⚠️ 未设置 `TILE_FWK_DEVICE_ID` 会导致运行时报错："If no NPU environment is available"

如果以上检查未通过，参考 `docs/install` 中的资料完成环境准备。

### 阶段二：代码生成

**输出目录**：`custom/{op}/`

**生成顺序**：
1. 根据输入信息，先梳理 API 映射、tiling 策略、loop 结构，确认可行后再进入实现
2. 进入实现前，读取 `references/execution-constraints.md`，把框架级约束、基础类型约束、控制流约束和高频 operation 约束落实到本次实现
3. 基于 `references/impl-template.py` 生成 `{op}_impl.py`
4. 基于 `references/test-template.py` 生成 `test_{op}.py`（前置：`{op}_impl.py` 已生成）
5. 生成 `README.md`

⚠️ 实现代码与测试代码必须分离，禁止混写 golden / impl / test 到同一文件。

---

#### 生成 {op}_impl.py

PyPTO kernel 函数实现，基于固定模板 `references/impl-template.py` 生成。

实现前必须逐项核对 `references/execution-constraints.md`，尤其是：输出写回、动态轴标注、Element 使用、TileShape、valid_shape、loop/cond、同图内读写回环。

| 规范 | 说明 |
|------|------|
| 导出函数 | `{op}_wrapper(x: torch.Tensor) -> torch.Tensor` |
| Kernel 装饰器 | `@pypto.frontend.jit` |
| Tensor 描述符 | `pypto.Tensor()` 或 `pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32)` |
| Tiling 配置 | 必须调用 `pypto.set_vec_tile_shapes(...)` 或 `pypto.set_cube_tile_shapes(...)` |
| 输出写回 | `output[:] = result` 或 `pypto.assemble(result, offset, output)` |
| 可选辅助函数 | `{op}_core()` — 复杂算子拆分核心计算逻辑 |

#### 生成 test_{op}.py

torch golden 函数精度对比测试，基于固定模板 `references/test-template.py` 生成。

**结构**：

| 区块 | 内容 |
|------|------|
| Import | `from {op}_golden import {op}_golden` + `from {op}_impl import {op}_wrapper` |
| 环境工具 | `get_device_id()` — 读取 `TILE_FWK_DEVICE_ID` |
| 测试函数 | `test_{op}_levelN()` — 数据生成 → `{op}_wrapper(x)` → `{op}_golden(x)` → `assert_allclose` |
| CLI 入口 | `argparse`，支持 `example_id` / `--list` / `--run_mode` |

**精度对比强制规范**：

| 规范 | 说明 |
|------|------|
| 精度对比 | 必须使用 `assert_allclose`，详见"三种状态标记约定" |
| 容差 | 简单算子 `rtol=1e-3, atol=1e-3`；复杂算子 `rtol=3e-3, atol=3e-3` |
| NPU 条件对比 | `if run_mode == "npu": assert_allclose(...)` |
| 禁止手写对比 | `assert max_diff < tolerance` / `np.allclose()` 均禁止 |

#### 生成 README.md

必须包含：
- 中文说明
- 算子概述与公式
- 目录结构
- 运行方式
- 验证入口
- 已知限制

#### Design 到代码文件的映射

| design 章节 | `test_{op}.py` | `{op}_impl.py` | `README.md` |
|------------|----------------|----------------|-------------|
| 概述 | 间接引用 | 否 | 是 |
| API 映射设计 | 否 | 是 | 可摘要 |
| 数据规格设计 | 是 | 是 | 可摘要 |
| Tiling 策略 | 否 | 是 | 可摘要 |
| Loop 结构设计 | 否 | 是 | 可摘要 |
| 验证方案 | 是 | 否 | 是 |
| 性能指标 | 部分 | 部分 | 是 |
| 交付件清单 | 是 | 是 | 是 |

---

### 阶段三：测试验证

优先使用 NPU 模式进行验证。

**验证步骤**：

1. **编译安装**（如未安装 pypto）：
```bash
python3 build_ci.py -f python3 --disable_auto_execute
```
已安装则无需反复编译。

2. **执行算子验证**：
检测到存在 NPU 卡时，直接使用 `run_mode=npu` 执行：
```bash
python3 custom/{op}/test_{op}.py
```

3. **验证失败处理**：
   ⚠️ **先确认是否有错误码：有则必须走错误码流程，无则跳过**
   **检查方法**：在 stderr 或日志中搜索 `Errcode: F` / `ErrCode: F`
   - **有错误码（如 `ErrCode: FC0000!`）** → 必须走错误码排查流程：见 [references/error-code-troubleshooting.md](references/error-code-troubleshooting.md)
   - **无错误码** → 跳过，直接分析报错信息或日志定位问题

**推荐验证顺序**：
1. 小规模功能验证：先确认代码能运行、基础输出形状正确
2. 典型规模验证：使用目标场景下的常规输入做主路径验证
3. 边界与极值验证：检查零值、大值、特殊 shape、动态 shape 等情况
4. 大规模或性能相关验证：在功能和精度稳定后再推进

**测试类型**：
- 单元测试：基本功能验证
- 精度测试：精度符合要求
- 性能测试：已完成性能测量，并基于实测数据判断是否符合要求

⚠️ 有 NPU 卡的情况下，不要使用 `run_mode=sim`。

---

## 实现注意点

1. **PyPTO tensor 创建后是未初始化随机值**：使用前先初始化，或者保证先写后读；不要把 `pypto.tensor(...)` 当成已初始化张量使用。
2. **禁止无中生有 op**：实现时只能使用 PyPTO 已支持的 API，遇到缺失能力应回退到 API 探索或设计阶段重新确认。
3. **优先使用 `@pypto.frontend.jit` 写法**：选择最新的非 wrapper 包装写法，参考 `docs/api/pypto-frontend-jit.md`，与现有示例和文档保持一致。
4. **golden / impl / test 必须职责分离**：不要把 golden 逻辑、实现逻辑和测试逻辑混写到同一个文件中。
5. **动态数据范围使用 valid_shape**：当最后一块数据量可能小于固定块大小时，`pypto.view` / `pypto.reshape` 中必须指定 `valid_shape`。
6. **动态循环边界使用 unroll_list**：当循环次数为动态值时，需要使用 `unroll_list`；多层循环嵌套时，最内层使用 `unroll_list`。
7. **matmul / cube 场景**：必须确认 `set_cube_tile_shapes(...)` 已正确配置。
8. **输出写回必须显式完成**：使用 `output[:] = ...`、`output.move(...)` 或 `pypto.assemble(..., output)`；不要写 `output = ...`。
9. **动态轴必须显式标注**：所有动态 shape 输入和输出都要在 Tensor 注解中标成 `pypto.DYNAMIC` / `pypto.DYN`。
10. **Element 用于固定标量 dtype**：当标量参与计算且 dtype 不能依赖隐式映射时，显式使用 `pypto.Element(dtype, value)`。
11. **避免同图内回环读写**：同一 Tensor 不要在同一图里既 `view` 读取又 `assemble` 回写。
12. 如果设计中已有 tiling / loop 约束，编码时优先遵循 `DESIGN.md`，不要临时拍脑袋改写。

---

## 常见问题与解决方案

### 常见的 7 类错误

1. **BFloat16 转 NumPy 失败**：必须先 `.float()` 再 `.numpy()`
2. **环境变量未设置**：先运行 `bash scripts/list_idle_chip_ids.sh` 确认可用 chip id，再设置 `export TILE_FWK_DEVICE_ID=<空闲 chip id>`
3. **动态轴定义位置错误**：必须在 jit 函数外部定义
4. **Tile Shape 未设置**：matmul 前必须调用 `set_cube_tile_shapes`；vec 操作前需要 `set_vec_tile_shapes`
5. **精度标准不合理**：bfloat16 使用 `atol=0.0001, rtol=0.0078125`
6. **使用 PyTorch 作为 Golden**：使用 NumPy 实现 golden 函数时，bfloat16 数据类型转换不够准确；golden 必须独立在 `{op}_golden.py`，使用纯 torch 实现
7. **SymbolicScalar 用作 list 索引报错**：`TypeError: list indices must be integers or slices, not SymbolicScalar`。原因：`pypto.loop` 返回的是编译时符号值，不是 Python runtime 对象。解决方法：使用 tensor slice 或 `pypto.view`/`pypto.assemble` 构建数据流。

## 三种状态标记约定

生成的 `test_{op}.py` 必须使用以下模式输出精度判定标记：

```python
import sys
import numpy as np

def run_test():
    # ... setup inputs, call golden and impl ...
    try:
        np.testing.assert_allclose(impl_output, golden_output, rtol=rtol, atol=atol)
        print("[PRECISION_PASS]")
    except AssertionError as e:
        print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # 功能问题（无标记），exit code ≠ 0
        print(f"Runtime error: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    run_test()
```

标记含义：
- `[PRECISION_PASS]`: 精度验证通过
- `[PRECISION_FAIL]`: 精度验证失败（数值不匹配）
- 无标记 + exit ≠ 0: 功能问题（代码崩溃、逻辑错误等）

`assert_allclose` 抛出的 `AssertionError` 包含 `Not equal to tolerance` 关键字，调用方据此区分"运行失败"和"精度失败"。

---

## Checklist

1. 3 个文件（`test_{op}.py` + `{op}_impl.py` + `README.md`）全部存在
2. `test_{op}.py` 可执行（无语法错误）
3. 测试包含 `[PRECISION_PASS]` / `[PRECISION_FAIL]` 标记逻辑，无其他功能问题
4. 验证失败时已确认是否有错误码：有则走错误码流程，无则跳过
5. `{op}_impl.py` 已按 `references/execution-constraints.md` 自检：输出写回、动态轴、TileShape、valid_shape、Element、loop/cond、assemble 回环均已检查
