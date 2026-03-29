# Triton-Ascend 迁移与优化参考

## 何时读取本文件

当你遇到以下任一情况时，继续读取本文件：

- 需要判断 Ascend 与 GPU 的并行模型差异
- 需要处理 `coreDim` 超限或 UB overflow
- 需要重写 `shape/stride/block_ptr/order`
- 需要判断是否使用 `care_padding=False`
- 需要判断是否存在 dtype 导致的 scalar 退化
- 需要给出 `TRITON_ALL_BLOCKS_PARALLEL`、`multibuffer` 等优化建议

## 核心迁移原则

### 1. 从逻辑 grid 转向物理核绑定

GPU Triton 常见问题是把 grid 当成逻辑任务空间自由展开；Ascend 更强调与物理核资源匹配。

默认原则：

- grid 优先用 1D
- 不要照搬 GPU 上的多维自由 grid
- 并发任务数不要无脑放大
- 如果逻辑核数远大于物理核数，分批调度会带来额外开销

### 2. 区分算子类型

- `Vector-only`：按 Vector Core 路径思考并行数和访存
- 含 `tl.dot`：按 AI Core / Cube 相关路径思考

这个区分会影响：

- 并发任务数
- 优化重点
- 对齐要求
- 编译优化项选择

但不要只按“源码里是否出现了 `tl.dot`”做机械分类。

如果 `tl.dot` 只是用来表达这些中间计算：

- prefix-sum / 局部扫描
- 三角 mask 聚合
- 小块累加或临时变换

仍要继续判断算子主语义到底是什么：

- 真正的矩阵乘 / 注意力样式块计算
- 还是本质上更接近 `Vector-only` 的归约、扫描、逐块累计

如果主语义其实是后者，完全可以在 Ascend 上改写成更直接的向量路径，而不是因为 GPU 版本里“碰巧写了 `tl.dot`”就整题都按 AI Core 思路处理。

### 3. 先跑通，再优化

默认顺序：

1. 最小迁移或语义改写
2. 正确性验证
3. Ascend 友好优化

不要在第一版就做大规模结构重写，除非原始写法明显不适合 Ascend。

### 4. 先区分“语义分支”和“性能分支”

真实工程代码里常见这些结构：

- 多个 kernel 并存
- `autotune`
- 运行时环境变量切路径
- 标量 / 向量 / 大小 shape 的自动分发
- GPU 特定的共享内存、warp、硬件代际分支

迁移时不要默认“原样保留全部分支”。

优先做这个判断：

1. 哪些分支是在表达不同语义
2. 哪些分支只是在表达 GPU 上的性能策略
3. 哪些分支在 Ascend 上已经没有明确收益

默认策略：

- 语义分支优先保留
- 纯 GPU 性能分支可以收敛
- 如果多个分支最终都在做同一件事，只是搬运方式、tile 或 autotune 不同，优先收敛成更少的 Ascend 友好路径
- `num_warps`、`num_stages`、GPU launch config 等也属于这一类“先怀疑是性能分支”的对象，不要默认跟着一起迁移

当你删掉某个分支时，要能说明原因，不要只给结果：

- 是为了移除 `autotune`
- 是为了去掉只在 GPU 上成立的共享内存选择
- 是为了减少多 kernel 维护成本
- 是为了换成更稳定的固定配置

### 4.5 看到 `Please DO NOT tune args ...` 告警时怎么想

真实迁移里经常会遇到这类 Triton-Ascend 运行告警：

- `Please DO NOT tune args ['num_warps']!`
- `Please DO NOT tune args ['num_stages']!`

这通常不是“只是日志不好看”，而是在提醒你：

1. 你可能仍然保留了 GPU 侧的 launch / tuning 参数
2. 这些参数在 Ascend 上未必有意义
3. 它们会让结果看起来像“先把 GPU 写法照搬过来再试”

默认策略：

- 对最小可运行版本，优先省略显式 `num_warps` / `num_stages`
- 只有在明确的编译要求、已知兼容性要求或实测收益存在时，才保留并解释原因
- 如果已经出现这类告警，把它当成 skill 还可继续收敛的一条信号，而不是默认忽略

### 5. 从“沿用原始指针逻辑”切到“重组后再算”

如果原始 kernel 有这些特征：

- chunk / tile / window / prefix-sum
- 局部归约或局部扫描
- 复杂 `block_ptr`
- 原本靠多层 grid 或多层循环组织数据

不要只想着把原指针逻辑机械搬过来。

同时评估这条通用路径：

1. 先确认真实布局
2. 先把一个主块读进来
3. 用 `reshape` / `transpose` / 更规整的 tile 视图重组数据
4. 再做 `cumsum`、reduce、mask 或逐块计算

这类改写常见收益：

- 减少离散访存
- 减少复杂索引
- 让算子更接近 Ascend 的向量路径
- 更容易用固定 tile 取代 autotune

### 5.5 grid 收敛不等于简单压平

遇到多维 grid 时，不要把“grid 优先 1D”理解成只做这一件事：

- 把 `(x, y, z)` 编码进一个扁平 `program_id`

这通常只是表面收敛，不一定真的更适合 Ascend。

优先做这个判断：

1. 哪些 grid 维度对应真实物理并行
2. 哪些 grid 维度只是逻辑 chunk / token / tile / block 维
3. 哪些逻辑维度可以移入 kernel 内部 `for` 循环

默认策略：

- 物理并行维度保留在 grid
- 逻辑遍历维度优先移入 kernel 内部循环
- 先减少调度维度，再考虑是否还需要压平成 1D

这在含 `tl.dot`、chunk 状态、序列块遍历的算子里尤其重要。

如果只是简单压平而不改变循环结构，常见问题是：

- 调度维度看起来少了，但本质任务划分没变
- 状态指针和辅助张量视图仍停留在旧设计
- kernel 仍然围着原来的 GPU 逻辑转

### 5.6 辅助张量优先按访问方向重排

很多算子里，性能瓶颈不只在主输入 `q/k/v/x`，也在辅助张量：

- gate
- bias
- mask
- index / offset
- 历史状态或中间状态

如果这些张量在当前访问方向上不连续，不要默认在 kernel 内硬扛复杂 stride。

优先评估：

1. 能否在 wrapper 侧做一次轻量 `transpose` / `contiguous`
2. 重排后是否能把 kernel 内访问改成简单线性 ptr
3. 重排成本是否低于长期复杂访存成本

默认经验：

- 一次明确的轻量重排，往往比在 kernel 里长期维持复杂 stride 更稳定
- 特别是 gate / bias 这类按 token 或按 head 顺序访问的张量，更适合先重排后顺序读取

### 5.7 主循环重排后，状态视图要联动重建

如果你决定把主循环顺序改掉，例如：

- 从“先 K 后 T”改成“先 T 后 K”
- 从“外层 chunk，内层 feature”改成“外层 feature，内层 chunk”

不要只改循环本身。

还要同步检查这些张量：

- state
- cache
- h / history
- block summary
- prefix state

重点重审：

1. `shape`
2. `stride`
3. `block_ptr`
4. `order`
5. 指针基址是在循环外一次算好，还是应改成在循环内动态计算

如果主循环已经重排，但状态张量仍沿用旧视图，常见症状是：

- 需要额外 `trans` 才能凑对
- `block_ptr` 的维度顺序不再自然
- 代码能跑但仍像“旧 kernel 套补丁”

默认策略：

- 主循环顺序变了，就默认把状态视图一起重审一遍
- 如果状态是按 chunk / token / block 编号组织的，往往应改成在循环内动态计算当前块的 base pointer

### 6. 复用工程 helper，但别忘了独立脚本上下文

如果输入来自真实工程：

- 优先复用工程里的设备属性 helper、布局 helper、公共 Triton util
- 不要默认把这些能力都内联重写

但如果输出目标是：

- 独立脚本
- 最小验证脚本
- 可脱离原工程直接运行的复现文件

还要继续检查：

- helper 是否要求额外初始化
- helper 是否依赖工程级上下文
- helper 是否依赖插件注册或 worker 初始化

默认策略：

- 能补初始化就补初始化
- 如果不能安全补齐，就在结果里明确前置条件
- 不要因为“复用工程 helper”这条规则，就忽略独立脚本可运行性

## 最小迁移规则

### GPU Triton

优先做这些动作：

- `device='cuda'` 改为 `device='npu'`
- 增加 `import torch_npu`
- 删除 GPU 专属设备获取与一致性断言
- grid 尽量收敛到 1D

如果用户明确要求“官方文档风格 / 最小 diff / 不要工程增强版”，就停在这里：

- 保持原 `kernel` 名字、wrapper 名字、`BLOCK_SIZE`、grid 写法和主体结构
- 不要主动加入 `contiguous()`
- 不要默认加入大量 shape/dtype/device 断言
- 不要把代码改名成 `xxx_npu`
- 不要把回答扩成工程优化综述

### Python / PyTorch

优先做这些动作：

- 先还原算子语义，不急着追求第一版高性能
- 找出输入、输出、索引、广播、mask、reduce 逻辑
- 写一个最直接的 Triton-Ascend kernel
- 再判断是否需要两级切分、连续访存重构、dtype 调整

对简单 unary / elementwise 改写，优先采用这条路径：

1. 先用 1D grid 写最直接的 `Vector-only` kernel
2. 如果原表达式含 `erf`、`exp`、`log` 这类对精度敏感的运算，优先在 kernel 内转成 `fp32` 再计算
3. 先用单级主块 `BLOCK_SIZE` 跑通
4. 只有在输入规模继续增大、出现 UB 压力或单次工作集过大时，再引入子块

### Python/Triton 常见编译坑

在 `@triton.jit` kernel 里，不要直接引用普通 Python 全局变量，例如：

- `INV_SQRT2 = 0.7071...`
- 然后在 kernel 里写 `x * INV_SQRT2`

这类写法在实际编译时可能报：

```text
Cannot access global variable ... from within @jit'ed function
```

默认修复方式：

- 直接把常量内联到 kernel 表达式里
- 或把它改成 `tl.constexpr` 形式显式传入

除非你很确定当前环境允许，不要依赖普通 Python 全局常量在 kernel 中可见。

### Python unary 的轻量优化默认策略

如果是简单 unary，且输入规模可能继续增大，但暂时没有 UB overflow：

- 可以先直接落地“主块自适应”版本
- 目标是先避免 `coreDim > 65535`
- 不要为了模板完整性，强行在第一版里加入 `XBLOCK_SUB`

可直接用这类思路：

1. 先算 `min_block = ceil(N / 65535)`
2. 再选不小于 `min_block` 的合适 2 的幂
3. 再设置一个保守上限，例如 `32768`
4. 如果这样仍出现 UB 或工作集过大，再进入双层切分

## `coreDim` 排查规则

典型报错：

```text
coreDim=xxxx can't be greater than UINT16_MAX
```

默认排查顺序：

1. 先算当前 grid 是否等价于过大的并发任务数
2. 增大 `BLOCK_SIZE` / `XBLOCK`，减少 `coreDim`
3. 必要时启用 `TRITON_ALL_BLOCKS_PARALLEL`
4. 重新确认放大主块后是否引入 UB overflow

经验规则：

- 需要同时控制 `coreDim <= 65535`
- 主块调大后，通常还要同步设计子块

快速选型模板：

1. 先算 `min_block = ceil(N / 65535)`
2. 再选不小于 `min_block` 的下一个合适 2 的幂
3. 如果这个主块把 UB 顶爆，再保留主块，增加子块

例如：

- `N = 1073741824`
- `min_block = ceil(N / 65535) = 16385`
- `16384` 不够，因为 `ceil(1073741824 / 16384) = 65536`
- `32768` 通常是第一档可行主块

## UB overflow 排查规则

典型报错：

```text
ub overflow, requires xxxx bits while yyyy bits available
```

默认排查顺序：

1. 看单次 `load/store` 工作集是否过大
2. 引入 `BLOCK_SIZE_SUB` / `XBLOCK_SUB`
3. 将单次顺序执行改为 for-loop tiling
4. 检查是否存在非对齐或离散访存，导致 UB 利用低效

经验规则：

- 只增大主块通常不够
- 主块解决 `coreDim`，子块解决 UB 压力
- for-loop tiling 不仅能降 UB，也能改善搬入/计算/搬出并行

## 访存与布局规则

### 连续访存优先

始终优先检查：

- 数据真实布局是什么
- 最内层维度是否连续
- stride 是否正确表达真实布局
- 当前 `block_ptr` 是否错误地把连续二维访问写成离散一维访问

### 对齐要求

迁移时至少记住：

- VV 场景重点关注 32 字节对齐
- CV 场景重点关注 512 字节对齐

如果出现 UB 压力异常、性能明显低、或指令映射不理想，优先检查是否是对齐与连续性问题。

### `shape/stride/block_ptr/order` 重审方法

遇到 `make_block_ptr` 时，不要直接沿用原始写法。重新确认：

- `shape`：真实张量形状，而不是凑出来的逻辑视图
- `strides`：真实内存步长
- `block_shape`：每次搬运的 tile 大小
- `order`：与连续访问方向一致

如果底层本来是 `(M, N)` 连续矩阵，就优先按二维矩阵建模，不要盲目压扁成一维。

### `order` 的快速判断

常见场景可以直接这样判断：

- 行主连续二维矩阵：`shape=(M, N), strides=(N, 1)`，优先 `order=(1, 0)`
- 列主连续二维矩阵：`shape=(M, N), strides=(1, M)`，优先 `order=(0, 1)`

原则不是死记硬背元组本身，而是先看：

- 哪个维度 stride 更小
- 哪个维度才是更连续的真实访问方向
- `order` 是否和这个连续方向一致

## mask 与 `care_padding=False`

mask 常见副作用：

- 增加默认填充值带来的依赖
- 降低数据搬运与计算并行度
- 让本可并行的流程变得更串行

如果满足以下条件，可以考虑 `care_padding=False`：

- `tl.load` 的未填充部分不会影响后续语义
- 默认填充值不是必须的
- 当前性能瓶颈与 mask load 依赖明显相关

不要机械使用。先确认语义安全。

### `where-mask` 场景的实战规则

如果算子本质上是简单向量选择，例如：

- `tmp2 = tmp0 < tmp1`
- `tmp3 = tl.where(tmp2, tmp0, tmp1)`
- 只有尾块边界使用 `xmask`

优先采用这条判断路径：

1. 先保留 1D grid
2. 先判断 `XBLOCK_SUB` / 子块循环是否真的在解决 UB 或工作集问题
3. 如果当前只是简单 `load + cmp + where + store`，且没有 UB 压力，优先删除子块循环，改成单层主块
4. 再通过主块自适应控制 `coreDim`

不要因为原始 Triton 版本里已经有 `for xoffset_sub in range(...)`，就默认在 Ascend 版本里保留它。

对这类简单 `where-mask` 场景，额外的子块循环常见副作用是：

- 增加循环与索引开销
- 增加重复 mask 计算
- 让本来就很轻的向量路径变复杂

只有在这些条件出现时，才优先保留或重新引入 `XBLOCK_SUB`：

- 主块放大后出现 UB overflow
- 单次工作集明显过大
- 算子不再是简单向量选择，而是更长的数据搬运/计算流水

## for-loop tiling 的使用时机

当算子存在以下特征时，优先考虑显式 for-loop tiling：

- 单次处理量太大
- mask 导致串行执行明显
- 无法形成“搬入/计算/搬出”并行
- UB 空间紧张

使用原则：

- 确认切分后数学语义仍然等价
- 子块大小既要控制 UB，也要保持足够吞吐

反过来，如果算子只是简单 `where / min / max` 风格的 `Vector-only` 路径，且当前没有 UB 压力，不要为了“模板完整”保留 tiling。

## dtype 与 scalar 退化

Ascend 上某些向量运算对 dtype 更敏感。

要主动排查：

- `Vector ADD` 路径是否因 `int64` 退化
- `Vector CMP` 路径是否因 `int64/int32` 退化
- `cols < N`、`where`、mask 比较是否落入低效类型路径

如果不影响语义与精度约束，可考虑：

- `int64 -> int32`
- 比较路径转为 `fp32`
- 在 load 后再做必要类型转换

对于 unary 数学函数，还要额外检查：

- 是否应在 kernel 内先转 `fp32` 再做 `erf/exp/log`
- 输出是否保持与输入 dtype 一致
- `allclose` 阈值是否与 dtype 相匹配

## `TRITON_ALL_BLOCKS_PARALLEL`

适用时机：

- GPU 迁移过来的 grid 分核过多
- 并发任务明显多于物理核数
- 分批调度带来的下发开销很明显

作用：

- 让编译器更贴近物理核数优化逻辑核数
- 减少分批调度开销

使用时要同时检查：

- 算子逻辑核之间是否可并行
- 启用后是否引入新的 block 或 UB 问题

## `multibuffer` 与相关编译优化项

`multibuffer` 的目标是提升数据搬运与计算并行度。

适合考虑的场景：

- 算子存在分块循环
- 数据搬运与计算可流水化
- UB 空间仍有余量

不适合或需要谨慎的场景：

- UB 已经很紧张
- 算子没有足够的 tiling
- 数据依赖过重，无法形成有效并行

如果用户要求基础优化说明，可在结果中说明是否建议尝试：

- `multibuffer`
- `unit_flag`
- `limit_auto_multi_buffer_only_for_local_buffer`
- `limit_auto_multi_buffer_of_local_buffer`
- `set_workspace_multibuffer`
- `enable_hivm_auto_cv_balance`
- `tile_mix_vector_loop`
- `tile_mix_cube_loop`
- `auto_blockify_size`

## 超大输入的验证方式

当 `N` 极大时，不要机械写一个“全量 reference 验证”脚本。更稳妥的做法是两段式验证：

1. 用公式或打印结果验证主块配置是否已经让 `coreDim <= 65535`
2. 用较小但覆盖非整除 block 的输入，验证语义正确性和子块稳定性

例如：

- 大规模 case：验证 `grid = ceil(N / BLOCK_SIZE)` 是否合规
- 中小规模 case：验证 `allclose`、最大误差、非整除 block、关键 dtype

## 建议的排障顺序

优先按下面顺序排障，不要东一榔头西一棒子：

1. 输入语义是否对齐
2. 设备与最小迁移是否完成
3. grid 是否仍保留 GPU 风格
4. 并发任务数是否失控
5. `coreDim` 是否超限
6. 主块 / 子块设计是否合理
7. 连续与对齐访存是否合理
8. mask / load / store 是否引入额外依赖
9. dtype 是否导致 scalar 退化
10. 是否需要 `TRITON_ALL_BLOCKS_PARALLEL` 或 `multibuffer`

## 结果输出提醒

最终结果不要只给建议。至少要给出：

- 最终 Triton-Ascend 实现
- 如果可优化，直接给优化版实现
- 最小验证脚本
- 明确说明做了哪些优化
- 仍未验证的风险点
