# Triton-Ascend 示例与输出样例

## 何时读取本文件

当你需要：

- 判断某类输入该怎么开场分析
- 参考推荐输出结构
- 看 GPU Triton 与 Python/PyTorch 两条路径的写法差异
- 快速构造验证脚本

时再读取本文件。

## 示例 1：GPU Triton elementwise 迁移

### 输入

用户给出一个 GPU Triton 向量加法 kernel，并要求迁移到 Ascend NPU。

### 处理思路

优先动作：

1. 把 `cuda` 改为 `npu`
2. 补 `torch_npu`
3. 删除 GPU 专属设备逻辑
4. 保持 1D grid
5. 保持原有 `add_kernel`、`add`、`BLOCK_SIZE` 和主体代码结构不变
6. 跑通后再看 block 是否需要调整

### 推荐输出风格

````markdown
## 迁移结论
- 输入来源：GPU/CUDA Triton
- 算子类型：elementwise，Vector-only
- 主要迁移动作：`cuda -> npu`、补 `torch_npu`、保留 1D grid

## Triton-Ascend 实现
```python
# 最终 kernel 与 wrapper
```

## 验证脚本
```python
# 在 NPU 上构造输入，并与 torch reference 对比
```

## 优化说明
- 当前先保持最小迁移
- 若数据规模继续增大，可进一步调 `BLOCK_SIZE`

## 风险与限制
- 尚未覆盖超大输入规模下的 `coreDim` 情况
````

## 示例 2：Python / PyTorch unary 改写

### 输入

用户给出一个 PyTorch 写法的 unary 算子，例如：

```python
def standard_unary(x0):
    return x0 * 0.5 * (1.0 + torch.erf(x0 / torch.sqrt(torch.tensor(2.0, device=x0.device))))
```

### 处理思路

优先动作：

1. 先提炼数学语义
2. 识别这是 `Vector-only` unary，不是简单设备替换
3. 先给出 1D grid 的 Triton-Ascend kernel
4. 对 `erf` 这类计算，优先在 kernel 内转为 `fp32`
5. 再根据输入规模决定是否只做主块自适应，还是继续引入子块

### 已验证可跑通的落地模式

对这个案例，实测可行的策略是：

1. 保持 1D grid
2. kernel 内 `x.to(tl.float32)` 后再做 `tl.erf(...)`
3. 先只做单级 `BLOCK_SIZE`
4. 根据 `N` 自适应选择主块，优先保证 `coreDim <= 65535`
5. 暂不引入 `XBLOCK_SUB`，因为当前 unary 工作集很小

### 实战坑点

这个案例实际踩到过一个编译坑：

```text
Cannot access global variable ... from within @jit'ed function
```

原因是 kernel 里直接引用了 Python 全局常量。

默认修复方式：

- 把常量直接内联到 kernel 表达式中
- 或改成显式 `constexpr` 传参

不要默认把数学常量写成普通 Python 全局变量再在 kernel 中直接引用。

### 推荐输出风格

````markdown
## 迁移结论
- 输入来源：Python/PyTorch
- 算子类型：unary，Vector-only
- 主要迁移动作：先语义改写为 Triton-Ascend，再做主块自适应，必要时再考虑子块

## Triton-Ascend 实现
```python
# 先给最终可运行版本；如果已存在明确优化空间，可直接给带主块自适应的版本
```

## 验证脚本
```python
# 构造 NPU 输入，对比原始 PyTorch 实现
# 至少覆盖一组基础输入、一组非整除 block 输入、一组更大输入
```

## 优化说明
- 当前先保持 1D grid
- 已根据 `N` 自适应选择主块，优先避免 `coreDim > 65535`
- 当前未引入子块，因为还没有明显 UB 压力
- 对 `erf` 路径先转 `fp32`，以降低精度风险
````

## 示例 2.2：GPU Triton `l2norm` 迁移

### 输入

用户给出一个来自真实工程的 GPU Triton `l2norm` 实现，特征通常包括：

- 多个 kernel 并存
- `@triton.autotune`
- 可能存在环境变量分支
- 默认按 GPU 风格的逻辑 grid 组织

### 处理思路

优先动作：

1. 先识别这是 GPU Triton 迁移，不是 Python/PyTorch 改写
2. 识别这是 `Vector-only` 的归一化算子，而不是 `tl.dot` 路径
3. 优先评估是否可以删掉多 kernel 和 `autotune`，收敛成单 kernel
4. 并行模型优先从 `grid = ceil(T / MBLOCK)` 切到 `grid = (num_core,)`
5. 让每个物理核处理一段主块，再在 kernel 内通过 `NUM_CHUNKS` 循环覆盖完整输入
6. 如果工程里已有 `get_vectorcore_num()` 之类 helper，优先复用，不要默认内联替代
7. 如果输出要作为独立脚本直接运行，继续检查 helper 是否依赖初始化；例如某些工程里的 `get_vectorcore_num()` 在脚本入口前需要先执行 `init_device_properties_triton()`

### 已验证可跑通的落地模式

对这类案例，优先考虑：

1. 保留单个 `l2norm_fwd_kernel2_loop`
2. 固定 `MBLOCK`，例如 `69`
3. 使用 `num_core = get_vectorcore_num()`
4. 计算 `main_bs = triton.cdiv(T, num_core)`
5. 再算 `NUM_CHUNKS = triton.cdiv(main_bs, MBLOCK)`
6. 保持 `grid = (num_core,)`
7. kernel 内对输入转 `fp32` 做平方和与 `rsqrt`
8. 如果复用工程 helper 后独立脚本报“device properties not initialized”，优先补初始化调用，而不是立刻回退成手写 helper

### 推荐输出风格

````markdown
## 迁移结论
- 输入来源：GPU/CUDA Triton
- 算子类型：`Vector-only`，归一化
- 主要迁移动作：删掉多 kernel 和 `autotune`，改成基于 Vector Core 的 1D grid + chunk loop

## Triton-Ascend 实现
```python
# 优先给单 kernel 版本
# 优先复用工程内 get_vectorcore_num()
```

## 验证脚本
```python
# 在 NPU 上构造输入
# 使用 PyTorch reference 对比
# 至少覆盖一组非整除 MBLOCK 的输入
```

## 优化说明
- 当前按 Vector Core 数量做物理核绑定
- 通过 `NUM_CHUNKS` 循环覆盖每个核负责的行块
- 当前不需要额外 `XBLOCK_SUB`
- 当前不需要 `TRITON_ALL_BLOCKS_PARALLEL` 或 `multibuffer`
- 若复用工程 helper，需说明是否需要额外初始化步骤，避免“在工程内可用、脱离工程即失败”
````

## 示例 2.5：单核数据搬运的最小 device 替换

### 输入

用户给出一个已经能在 GPU 上运行的简单 1D broadcast / 数据搬运 kernel，希望先迁到 NPU 上跑通。

### 处理思路

优先动作：

1. 先识别这是“单核数据搬运 + 简单 device 替换”场景
2. 默认先尝试 `device='cuda' -> device='npu'`
3. 如果原 kernel 本身没有 `coreDim`、UB、复杂布局问题，不要一开始就展开优化
4. 保持原测试风格，用 PyTorch reference 对比即可

### 推荐输出风格

````markdown
## 迁移结论
- 输入来源：GPU Triton
- 算子类型：broadcast / 数据搬运
- 主要迁移动作：先做最小 device 替换，保留原 kernel 结构

## Triton-Ascend 实现
```python
# 只把示例里的 `device='cuda'` 改为 `device='npu'`
```

## 优化说明
- 当前先以跑通为主
- 暂不主动展开 `coreDim`、UB 或 `block_ptr` 优化
````

## 示例 3：`coreDim + UB overflow` 联合问题

### 输入

用户给出一个已经迁移到 NPU 的 kernel，但报错包含：

```text
coreDim=262144 can't be greater than UINT16_MAX
```

并且在把 `BLOCK_SIZE` 调大后，又出现：

```text
ub overflow, requires xxxx bits while yyyy bits available
```

### 处理思路

默认做法：

1. 先增大主块，解决 `coreDim`
2. 再引入子块，解决 UB
3. 如果仍有大 grid 问题，说明是否适合 `TRITON_ALL_BLOCKS_PARALLEL`
4. 最后检查 stride、对齐和离散访存

### 推荐输出重点

````markdown
## 优化说明
- 主块从较小值调大，以保证 `coreDim <= 65535`
- 引入 `BLOCK_SIZE_SUB`，把单次处理拆成多轮
- 若逻辑核明显多于物理核，可考虑 `TRITON_ALL_BLOCKS_PARALLEL`
- 若访存仍低效，需要继续重审 `shape/stride/block_ptr/order`
````

## 示例 3.5：`where-mask` 场景的直接优化

### 输入

用户给出一个带 `tl.where`、mask 和 `XBLOCK_SUB` 子块循环的 Triton 风格 kernel，例如：

```python
@triton.jit
def triton_where_lt_case2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)
        xmask = xindex < xnumel
        tmp0 = tl.load(in_ptr0 + xindex, xmask)
        tmp1 = tl.load(in_ptr1 + xindex, xmask)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.where(tmp2, tmp0, tmp1)
        tl.store(out_ptr0 + xindex, tmp3, xmask)
```

### 处理思路

优先动作：

1. 识别这是 `Vector-only` 的简单 `where-mask` 场景
2. 先检查子块循环是否真的在解决 UB 问题
3. 如果没有明显 UB 压力，直接删掉 `XBLOCK_SUB` 循环
4. 保留 1D grid，并根据 `N` 自适应选择主块
5. 用 `torch.where` 做 reference 验证

### 已验证可跑通的落地模式

对这个案例，实测可行的策略是：

1. 把双层 `XBLOCK + XBLOCK_SUB` 改成单层 `XBLOCK`
2. 只保留尾块边界 mask
3. 用主块自适应控制 `coreDim`
4. 当前不引入 `care_padding=False`
5. 当前不需要 `TRITON_ALL_BLOCKS_PARALLEL` 或 `multibuffer`

### 推荐输出风格

````markdown
## 迁移结论
- 输入来源：Triton 风格 kernel + PyTorch reference
- 算子类型：where-mask，Vector-only
- 主要迁移动作：删掉无必要的 `XBLOCK_SUB` 子块循环，保留 1D grid，并做主块自适应

## Triton-Ascend 实现
```python
# 最终优化版：单层 XBLOCK + tl.where + 尾块 mask
```

## 验证脚本
```python
# 构造 NPU 输入，对比 torch.where
# 至少覆盖一组非整除 block 输入
```

## 优化说明
- 原始子块循环未体现出明确 UB 价值，先删除
- 当前保持 1D grid
- 已根据 `N` 自适应 `XBLOCK`
- 当前不需要 `care_padding=False`
- 当前不需要 `TRITON_ALL_BLOCKS_PARALLEL`
- 当前不需要 `multibuffer`
````

## 补充示例：文档风格最小迁移

### 输入

用户给出官方文档、博客或教程里的简单 `elementwise` 示例，只要求“迁移到 Ascend NPU”。

### 默认策略

先给“最小 diff 迁移版”：

1. 新增 `import torch_npu`
2. 删除 GPU 专属设备获取逻辑
3. 删除依赖该设备对象的显式断言
4. 将 `device='cuda'` 改为 `device='npu'`
5. 其余 kernel、wrapper、grid、`BLOCK_SIZE` 和主流程尽量不动

### 不推荐的默认做法

如果用户没有明确要求增强版，不要一上来就：

- 改函数名
- 加大量 shape/dtype/device 断言
- 默认加入 `contiguous()`
- 把答案重写成工程封装版

这些改动不一定错误，但会偏离“最小迁移”的常见预期，也不利于和官方文档逐行对照。

### 推荐输出风格

````markdown
## Triton-Ascend 实现
```python
+ import torch_npu
import triton
import triton.language as tl

- DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(...):
    ...

def add(x, y):
    output = torch.empty_like(x)
-   assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    ...

-x = torch.rand(size, device='cuda')
+x = torch.rand(size, device='npu')
```

## 优化说明
- 当前先给文档风格最小迁移版
- 暂不主动加入工程增强项
- 若用户明确要求“官方文档风格 / 最小 diff / 不要工程增强版”，这里可以只写极简优化说明
- 若用户要求增强版，再补额外校验、连续化或独立测试包装
````

## 最小验证脚本模板

对大多数迁移任务，验证脚本至少应包含：

```python
import torch
import torch_npu

# import migrated operator here

def reference_impl(...):
    ...

def test_case():
    # 1. 构造 NPU 输入
    # 2. 调用 torch reference
    # 3. 调用 Triton-Ascend 实现
    # 4. 比较 allclose 或 max abs error
    pass

if __name__ == "__main__":
    test_case()
```

建议至少覆盖：

- 小规模输入
- 非整除 block 的输入长度
- 一组更大输入
- 关键 dtype

## 固定输出模板示例

````markdown
## 迁移结论
- 输入来源：GPU Triton / Python-PyTorch
- 算子类型：Vector-only / 含 `tl.dot`
- 主要问题：无 / `coreDim` / UB / 访存 / dtype

## Triton-Ascend 实现
```python
# 最终实现
```

## 验证脚本
```python
# 最小验证代码
```

## 优化说明
- grid 是否改为 1D
- 是否改成物理核绑定思路
- 是否调整 `BLOCK_SIZE/XBLOCK`
- 是否引入 `BLOCK_SIZE_SUB/XBLOCK_SUB`
- 是否处理 stride / block_ptr / 对齐
- 是否评估 `care_padding=False`
- 是否建议 `TRITON_ALL_BLOCKS_PARALLEL`
- 是否建议或使用 `multibuffer`
- 是否处理 dtype 导致的 scalar 退化

## 风险与限制
- 未验证的 shape
- 未验证的 dtype
- 未实测的性能收益
````
