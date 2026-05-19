---
name: external-cannbot-ops-pypto-op-perf-tune-tune-swimlane
description: PyPTO 算子深度性能调优技能。通过泳道图分析及调优性能，包括 Stitch 调优、TileShape 深度调优、合图调优、调度策略调优等。当用户需要进行深度性能调优、泳道图分析、Stitch
  优化、合图优化时使用此技能。触发词：深度性能调优、泳道图分析、Stitch 调优、合图调优、调度优化。
original-name: tune-swimlane
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-19'
synced-commit: 943f3bfc36e24068e065ca7ace72fbff86f4a09c
license: UNKNOWN
---

# PyPTO 算子深度性能调优

## 概述

深度性能调优通过泳道图分析及调优性能，采用 man-in-loop 的方式，通过获取并分析当前算子性能数据，针对性调整各性能配置参数，经过迭代调优逐步逼近最佳性能。

## 前置条件

1. **完成开箱性能调优**：先进行代码级优化
2. **精度校验通过**：确保算子计算正确
3. **已采集性能数据**：生成泳道图和气泡分析报告

## 泳道图分析

### 泳道图文件位置

泳道图数据文件位于 `output/output_*/` 目录：
- `merged_swimlane.json` - 泳道图数据文件
- `bubble_analysis.log` - 气泡分析报告

### 查看泳道图

1. 通过 PyPTO Toolkit 插件查看
2. 或在 https://ui.perfetto.dev/ 上传泳道图文件

### 泳道图关键信息

- 任务的执行顺序和耗时信息
- 各核心的工作时间和等待时间
- 气泡（线程等待调度的时间）
- 任务依赖关系

## 调优方向

### 1. Stitch 调优

Stitch 配置决定了多少个 root function 被同时下发调度。

#### 1.1 配置方法

```python
@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128}
)
```

#### 1.2 参数影响

| 参数值 | 优点 | 缺点 |
|--------|------|------|
| 过小（如 1） | - | 每个任务需同步，调度开销大 |
| 适中（如 128） | 泳道图紧凑，调度开销低 | - |
| 过大（如 512） | 泳道图更紧凑 | 调度耗时增加，workspace 增加 |

#### 1.3 调优建议

在内存资源允许的前提下，逐步增大 Stitch 配置，结合泳道图和端到端总耗时数据调整参数。

### 2. TileShape 深度调优

#### 2.1 Matmul TileShape 深度调优

主要关注**减少重复载入**和 **K 轴分核**两个调优手段。

```python
pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256],
    enable_multi_data_load=True,  # 减少重复载入
    enable_split_k=True)          # K 轴分核
```

**参数说明**：
- `enable_multi_data_load`：减少重复载入
- `enable_split_k`：K 轴分核

#### 2.2 Vector TileShape 深度调优

**原则 1**：下游 Vector Operation 的 TileShape 尽可能使用上游 Operation 的输出 TileShape

```python
# 上下游 TileShape 对齐时，可以合并在一个子图
# Transpose TileShape: (64, 128)
# Add TileShape 应优先选择: (128, 64)
```

**原则 2**：根据泳道图上的子图大小和并行度调整

- 并行核数较少（<一半 Vector 核）：减小 TileShape
- 子图耗时短、调度开销占比较高：增大 TileShape

**原则 3**：调整相邻 Cube 和 Vector Operation 的 TileShape，使依赖更简单

### 3. 合图调优

合图是指将计算图中多个逻辑上独立的 Operation 合并为一个逻辑子图。

#### 3.1 深度方向合图

沿数据依赖路径将前后相邻的 Operation 进行融合。

**手动指定合图方案**：

```python
# 开始合图
pypto.set_pass_options(sg_set_scope=1)
# ... 操作 ...
# 结束合图
pypto.set_pass_options(sg_set_scope=-1)
```

**融合目标**：
- 上下游 Operation 间传输数据量较大
- 多个 Operation 切分后变成并行的连通分支

**注意**：当前主要考虑在连续的 Vector 计算过程中使用，暂不支持将 Matmul Operation 与 Vector Operation 进行合图。

#### 3.2 广度方向合图

针对计算图中处于同一层级、可并行执行的 Operation。

**Matmul 广度方向合图**：

1. **L1Reuse 策略**（默认开启，用于合并具有冗余 L1 搬运的子图）

```python
# 全局统一配置为 2
@pypto.frontend.jit(
    pass_options={"cube_l1_reuse_setting": {-1: 2}}
)

# 全局自动配置基础上，同构子图 id 为 0 的子图配置为 8
@pypto.frontend.jit(
    pass_options={"cube_l1_reuse_setting": {0: 8}}
)

# 全局配置为 2 的基础上，同构子图 id 为 0 的子图配置为 8
@pypto.frontend.jit(
    pass_options={"cube_l1_reuse_setting": {-1: 2, 0: 8}}
)
```

2. **CubeNBuffer 策略**（用于合并同构的子图）

**适用场景**（不能使能 L1Reuse）：
- Cube 子图间没有重复 L1 搬运
- K 轴很长且没有切 K

```python
@pypto.frontend.jit(
    pass_options={"cube_nbuffer_setting": {0: 8}}
)
```

**Vector 广度方向合图**：

```python
@pypto.frontend.jit(
    pass_options={"vec_nbuffer_setting": {0: 8}}
)
```

**使用场景**：泳道图内有同构子图组具有大量的小子图（耗时在 10u 以下）

### 4. 调度策略调优

当上下游子图之间依赖较为简单，或下游子图输入 Tensor 的 L2 命中率较为重要时，推荐使用 L2 亲和调度。

```python
@pypto.jit(runtime_options={"device_sched_mode": 1})
```

**注意事项**：综合考虑 L2 复用与负载均衡的影响，不同场景的最佳配置策略不同。

## 性能优化建议库

### 建议 1：气泡率过高

**症状**：气泡率 > 10%

**可能原因：**
- 任务粒度过小
- 调度策略不当
- stitch 参数过小

**优化建议**：
1. **Stitch 调优（优先级高）**
   ```python
   @pypto.frontend.jit(
       runtime_options={"stitch_function_max_num": 128}
   )
   ```

2. **Loop Unroll（优先级高）**
   ```python
   for s2_idx in pypto.loop(s2_loop, unroll_list=[8, 4, 2, 1], name="LOOP_s2", idx_name="s2_idx"):
       # 计算逻辑
   ```

3. **L1Reuse 优化**
   ```python
   pypto.set_pass_options(cube_l1_reuse_setting={0: 8})
   ```

4. **调整任务粒度**
- 增大 loop 的 tile size
- 减少 loop 层级


### 建议 2：核心利用率低

**症状**：核心利用率 < 50%

**可能原因：**
- 等待时间过长
- 任务调度不均衡
- 内存访问冲突

**优化建议**：

1. **L2 亲和调度**
   ```python
   @pypto.jit(runtime_options={"device_sched_mode": 1})
   ```

2. **调整 TileSize**
   ```python
   pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 128])
   ```

3. **启用 CubeNBuffer 合并同构子图**
   ```python
   pypto.set_pass_options(cube_nbuffer_setting={0: 8})
   ```

### 建议 3：核心负载不均衡

**症状**：AicoreTime 差异 > 20%

**可能原因：**
- 任务分配不均
- 任务执行时间差异大

**优化建议**：
1. **调整任务分配策略**
   - 使用更均匀的任务切分
   - 避免某些核心任务过多

2. **优化任务粒度**
   - 调整 tile size 使任务更均匀

3. **调整任务执行顺序**
   - 使用 sg_set_scope 合并子图
   ```python
   pypto.set_pass_options(sg_set_scope=1)
   # ... 操作 ...
   pypto.set_pass_options(sg_set_scope=-1)
   ```


## 调优流程

```
┌────────────────────────────────────────────────┐
│                深度性能调优流程                │
├────────────────────────────────────────────────┤
│                                                │
│  1. 采集泳道图数据                             │
│     └─ debug_options={"runtime_debug_mode": 1} │
│                                                │
│  2. 分析泳道图                                 │
│     ├─ 查看任务执行顺序                        │
│     ├─ 识别气泡（等待调度时间）                │
│     └─ 分析核心利用率                          │
│                                                │
│  3. 选择调优方向                               │
│     ├─ 气泡率高 → Stitch/Loop Unroll           │
│     ├─ 利用率低 → 调度策略/TileShape           │
│     └─ 负载不均 → 合图优化                     │
│                                                │
│  4. 应用优化                                   │
│     └─ 每次只修改一个参数                      │
│                                                │
│  5. 验证                                       │
│     ├─ 重新编译运行                            │
│     ├─ 检查精度                                │
│     └─ 对比性能数据                            │
│                                                │
│  6. 迭代直到达到目标性能                       │
│                                                │
└────────────────────────────────────────────────┘
```


## 常见问题

### Q1: 泳道图文件在哪里？

A: 泳道图文件在 `output/output_*/` 目录下，其中 `*` 是时间戳。

### Q2: 如何查看性能统计？

A: 使用 PyPTO Toolkit 打开 `merged_swimlane.json` 文件，然后点击 "查看性能报告" 按钮。

### Q3: 气泡是什么？

A: 气泡是指线程等待调度的时间，表示线程空闲的时间段。气泡率越低，说明调度效率越高。

### Q4: 控制开销占比过高怎么办？

A: 对于小数据量，控制开销占比高是正常现象。可以通过增加数据规模来降低控制开销占比。

### Q5: 如何选择合适的 Tilesize？

A:

* 对于 Cube 计算：推荐使用 [128, 128], [64, 256], [256, 256] 或 [256, 256], [64, 256], [128, 128]
* 对于 Vector 计算：推荐使用 [32, 512] 或 [64, 512]
* 需要根据具体场景（输入 shape、dtype、format 等）以及硬件平台进行综合考虑


## 参考资料

- [性能调优文档](../../../../docs/tutorials/debug/performance.md)
- [Matmul 高性能编程](../../../../docs/tutorials/debug/matmul_performance_guide.md)
- [GLM Attention 案例](../../../../models/glm_v4_5/glm_attention.py)
- [性能优化案例](../../../../docs/tutorials/debug/performance_case_quantindexerprolog.md)
