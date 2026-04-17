---
name: triton-ascend-migration
description: 将 GPU/CUDA Triton 算子迁移为 Triton-Ascend，或将 Python/PyTorch 算子改写为可在 Ascend NPU 上运行的 Triton-Ascend 实现，并在发现明确优化空间时直接输出优化后的代码、最小验证脚本和排障说明。用户只要提到 昇腾、Ascend、NPU、triton-ascend、Triton 算子迁移、PyTorch 算子改写、coreDim、UB overflow、1D grid、物理核绑定、block_ptr、stride、访存对齐、mask 性能、dtype 退化、算子优化，或者直接问“这个 skill 怎么用”“怎么在命令行里跑”“怎么在容器里执行迁移/验证”，就应优先使用本 skill，即使用户没有明确说“写 skill”或“做迁移”。
---

# Triton-Ascend Migration

## Quick Start

遇到迁移请求时，按下面顺序执行：

1. 先识别输入方式：
   - 文件路径 / 指定代码段
   - 用户直接粘贴代码
2. 再识别输入来源：
   - GPU/CUDA Triton kernel
   - Python/PyTorch 算子实现
3. 再识别算子类型：
   - `elementwise`
   - `broadcast / mask`
   - `reduce`
   - 含 `tl.dot`
4. 先做最小可运行版本：
   - `cuda -> npu`
   - 补 `torch_npu`
   - 去掉 GPU 专属设备逻辑
   - grid 优先 1D
   - 简单教程示例默认给“最小 diff 迁移版”
5. 跑通后再做 Ascend 侧优化：
   - 物理核绑定
   - `BLOCK_SIZE/XBLOCK`
   - `BLOCK_SIZE_SUB/XBLOCK_SUB`
   - 连续/对齐访存
   - `coreDim` / UB / dtype / mask 排查
6. 如果存在明确优化空间，直接输出优化后的实现，不要只停留在建议。

## 如何使用这个 Skill

如果用户问“这个 skill 怎么用”，先不要立刻进入长篇迁移分析；先用 3 到 6 行给出简明用法，再根据用户提供的输入继续执行。

简述时只保留这几个点：

- 用户可以提供 `Triton/CUDA` 代码、`PyTorch` 参考实现、文件路径，或报错/性能日志。
- 用户最好同时说明运行环境：本机命令行、已有容器、CI，或只需要生成代码不执行。
- 用户如果有偏好，也应说明：`最小 diff 迁移`、`文档风格`、`先跑通再优化`、`直接给优化版`。
- 你会按场景输出：`Triton-Ascend 实现`、`最小验证脚本`、`执行命令`、`优化说明`。

如果用户继续追问“具体怎么提问”“命令怎么写”“容器里怎么跑”，再读取 `references/usage.md`，按需给出本机命令、容器命令和示例问法；不要把整份长说明直接搬进常规回答里。

复制这份检查清单并跟踪进度：

```text
迁移进度
- [ ] 识别输入来源与算子类型
- [ ] 先做最小迁移或语义改写
- [ ] 调整为 Ascend 友好的并行与 grid
- [ ] 重做 block / tiling
- [ ] 审查 stride / block_ptr / 对齐
- [ ] 处理 coreDim / UB / scalar 退化
- [ ] 直接落地可行优化
- [ ] 生成并保存最小 NPU 验证脚本
- [ ] 实际执行验证脚本
- [ ] 输出结果与优化说明
```

## 输入识别

先回答三个问题：

1. 用户是给文件路径，还是直接贴代码？
2. 这是完整脚本、局部片段，还是单个 kernel？
3. 这是 GPU Triton 迁移，还是 Python/PyTorch 语义改写？

输入方式的细节、缺信息时的默认处理、文件路径与粘贴代码冲突时的优先级，见 `references/input-modes.md`。

### 场景 A：GPU Triton -> Triton-Ascend

优先检查：

- 是否存在 `device='cuda'`
- 是否有 GPU 专属设备获取或断言逻辑
- 是否保留了 GPU 风格的多维自由 grid
- 是否使用 `tl.dot`
- 是否存在复杂 `shape/stride/block_ptr/order`

### 场景 B：Python/PyTorch -> Triton-Ascend

先提炼语义，再写 Triton：

- 输入输出张量关系
- 索引与广播方式
- mask / reduce 逻辑
- dtype 和精度要求
- 原始 PyTorch 是否已经天然连续访存

如果原始算子只是参考实现，先写一个语义等价的 Triton-Ascend 版本，再继续优化。

## 迁移流程

### 1. 收集最小必要信息

优先收集这些信息；缺什么补什么：

- 输入代码或最小复现
- 输入方式：文件路径 / 指定代码段 / 用户直接粘贴代码
- shape、dtype、stride
- 是否有 mask、broadcast、reduce
- 当前报错或性能问题
- 是否要求保持完全相同精度
- 运行环境：本机命令行、容器内、CI、或仅生成代码不执行

如果信息不完整，按这个顺序补：

1. 先从现有代码里推断
2. 再用最小合理假设补齐验证脚本
3. 最后才向用户追问必须的信息

如果当前缺的是“执行位置”信息，按下面顺序推断：

1. 先看用户是否给了容器名、`docker exec`、容器路径、镜像信息
2. 再看用户是否给了本机文件路径、当前目录、终端命令
3. 仍无法判断时，再追问“你希望我按本机命令行还是容器内命令来写验证步骤？”

### 2. 先做最小迁移或语义改写

默认先追求“语义对齐并跑通”：

- GPU Triton：先把 `cuda` 改成 `npu`
- 导入 `torch_npu`
- 移除 GPU 专属设备逻辑
- 对文档/教程式的简单示例，尽量保持原有 `kernel` 名字、wrapper 名字、`BLOCK_SIZE`、grid 写法和主体代码结构不变
- 第一版不要主动加入 `contiguous()`、额外断言、函数重命名、工程化包装，除非用户明确要求“增强版/生产版”，或这些改动是修复 NPU 上的确定性问题所必需
- Python/PyTorch：先按原计算语义改写成最直接的 Triton kernel

不要在第一步就过度重写。

如果用户明确说了这些信号词：

- 官方文档风格
- 严格最小迁移
- 最小 diff
- 不要工程增强版
- 只参考官方迁移示例

则这条“最小迁移模式”应当覆盖后面那些更泛化的优化说明要求：

- 代码只做必要修改
- `优化说明` 可以只写 1 到 3 行，明确“本题先不展开优化”
- 不要为了模板完整性，强行展开 `TRITON_ALL_BLOCKS_PARALLEL`、`multibuffer`、`care_padding=False`、物理核绑定等内容
- 不要把回答风格从“文档 diff”带偏成“工程优化综述”
- 验证脚本也应保持“最小可运行”，不要默认写成工程化测试框架

文档风格最小迁移、单文件示例组织方式、验证脚本命名与保存规则，见 `references/output-and-validation.md`。

### 3. 改写并行模型

Ascend 侧优先遵循这些规则：

- grid 优先 1D
- 从 GPU 逻辑 grid 思维切换到 Ascend 物理核绑定思维
- `Vector-only` 算子优先按 Vector Core 路径思考
- 含 `tl.dot` 算子优先按 AI Core 路径思考

进一步按这组“通用收敛规则”判断，不要机械保留 GPU 上的全部实现分支：

- 如果原始实现存在多 kernel、`autotune`、环境变量分支、不同数据路径的自动分发，先区分哪些是“语义必需”，哪些只是“GPU 上的性能策略”
- 对 Ascend 上明显不再必要的性能分支，可以收敛成单 kernel 或更少的路径；重点保留语义，而不是保留所有历史分支
- 如果一个算子本质上仍是 `Vector-only`，但原始实现用了复杂 `block_ptr`、二维/三维 grid、额外分块或多版本 kernel，优先评估能否改成更直接的 1D grid、固定配置、单路径实现
- 如果一个算子含 `tl.dot`，不要只想着“把多维 grid 压成 1D”；优先判断哪些 grid 维度只是逻辑 chunk / token / tile 维，是否更适合移入 kernel 内部循环，以减少调度维度
- 不要只按“源码里出现了 `tl.dot`”做机械分类；如果 `tl.dot` 只是拿来实现 prefix-sum、局部扫描、三角 mask 聚合等中间技巧，仍要先按算子主语义判断它更像 `Vector-only` 归约/扫描，还是确实应走 AI Core 路径
- 如果算子天然带 chunk、tile、window、prefix-sum、局部归约等结构，不要只沿用原始逐块指针逻辑；同时评估“先重排布局，再做向量化计算”是否更适合 Ascend
- 如果某个辅助张量（例如 gate、mask、bias、index、state-gate）在当前访问方向上并不连续，优先在 wrapper 侧做轻量 `transpose/contiguous` 或等价布局重排，再在 kernel 内按更简单的线性 ptr 或更规整的 `block_ptr` 访问
- 如果主循环顺序被重排了，例如从“先 K 后 T”改成“先 T 后 K”，要同步重审状态张量、cache 张量、历史块张量的 `shape/stride/block_ptr/order`；不要只改调度顺序，却继续沿用旧视图再靠 `trans` 或额外索引补救
- 如果当前工程里已经存在 `get_vectorcore_num()`、设备属性工具、常用布局 helper 等公共能力，优先复用工程内 helper，不要默认手写内联替代版本
- 但如果当前输出目标是“独立可运行脚本”或“最小验证脚本”，继续检查这些 helper 是否依赖额外初始化；若依赖工程初始化步骤，要么补上初始化，要么在结果里明确说明前置条件
- 当你决定“删分支 / 收敛实现”时，要在结果里说明原因：是因为该分支只服务于 GPU autotune、只服务于共享内存选择、还是在 Ascend 上没有明确收益
- 如果迁移后的 Triton-Ascend 运行日志出现 `Please DO NOT tune args ['num_warps']`、`['num_stages']` 或类似告警，优先回头检查是否仍机械保留了 GPU 风格 launch/tuning 参数；对 Ascend 的最小可运行实现，默认不要显式保留这些参数，除非你能给出明确的编译要求或实测收益
- 验证脚本不要只用一组通用 shape；测试集应从算子特征反推出来，至少覆盖一个非整除块、一个最容易触发分支差异的 case，以及一个更接近真实工作集的 case

如果用户给的是 2D/3D grid，优先评估能否折叠为 1D grid 再在 kernel 内恢复索引。`coreDim`、UB、`shape/stride/block_ptr/order`、`care_padding=False`、`TRITON_ALL_BLOCKS_PARALLEL`、`multibuffer` 等细则，见 `references/reference.md`。

## 优化与排障

### 直接优化的默认规则

如果满足以下任一条件，直接给出优化后的实现：

- `coreDim` 明显超限
- UB 使用明显过大
- 访存离散且可重构为连续访问
- mask load/store 具备更优写法
- dtype 明显导致 vector 运算退化为 scalar

如果不满足这些条件，尤其是简单向量加法这类示例，不要为了“看起来更完整”而默认输出增强包装版。先给最小迁移版，再把增强项放到“可选优化”里。

### 优化动作优先级

1. 调整 grid 和核数
2. 调整主块大小
3. 引入或重构子块循环
4. 修正 `shape/stride/block_ptr/order`
5. 评估 `care_padding=False`
6. 评估 `TRITON_ALL_BLOCKS_PARALLEL`
7. 评估 `multibuffer` 和相关编译优化项
8. 在不破坏语义前提下调整 dtype 路径

### 必须覆盖的关键点

输出中必须覆盖这些内容：

- `cuda -> npu`
- `torch_npu`
- 1D grid
- 物理核绑定
- `Vector-only` 与含 `tl.dot` 的区分
- `coreDim <= 65535`
- UB 限制
- 连续 / 对齐访存
- `shape/stride/block_ptr/order` 重审
- `TRITON_ALL_BLOCKS_PARALLEL`
- `multibuffer`
- `care_padding=False`
- dtype 导致的 scalar 退化

## 固定输出模板

始终按这个结构输出：

```markdown
## 迁移结论
- 输入来源：
- 算子类型：
- 主要迁移动作：

## Triton-Ascend 实现
- 给出最终 kernel 和调用包装代码
- 如果当前场景只是基础迁移，先给“最小 diff 迁移版”
- 只有在用户要求增强版，或确有明确优化空间时，再额外给“工程增强版/优化版”
- 如果存在明确优化空间，直接给出优化后的版本
- 说明生成文件的保存路径和命名

## 验证脚本
- 给出最小可执行验证脚本
- 使用 PyTorch reference 对比
- 至少包含 `allclose` 或最大误差输出
- 说明验证脚本保存路径
- 明确是否已实际执行，以及执行命令与结果

## 优化说明
- 说明 grid / 核数 / block / 子块 的调整原因
- 说明是否处理了 `coreDim`、UB、访存、dtype、mask 性能问题
- 说明是否使用 `TRITON_ALL_BLOCKS_PARALLEL`、`multibuffer`、`care_padding=False`

如果当前题目是“文档风格最小迁移”，这里可以极简：
- 只说明当前先做最小迁移
- 一句话说明本题未展开 `coreDim` / UB / `multibuffer` 等优化
- 不要为了套模板而展开长篇优化分析

## 风险与限制
- 列出仍未验证的边界条件
- 列出需要用户补充的信息
- 如果脚本未跑通，明确卡在哪一步
```

如果用户问题本身是“怎么使用这个 skill”，先在正式模板前加一个极简“使用方法”小节，控制在 3 到 6 行，说明：

- 用户应提供什么输入
- 当前按本机还是容器场景处理
- 你接下来会产出什么

然后再进入正常迁移输出。

如果用户继续追问命令行、容器、目录切换、验证命令模板，再读取 `references/usage.md`，不要把这些细节默认塞进每次迁移回答。

## Additional Resources

需要规则细节时，继续读取：

- [使用方法、本机命令与容器场景](references/usage.md)
- [输入方式与上下文补齐](references/input-modes.md)
- [输出、命名与最小验证脚本](references/output-and-validation.md)
- [迁移与优化参考](references/reference.md)
- [典型示例与输出样例](references/examples.md)
- [人工 Review 测试清单](references/test-checklist.md)
