# Elementwise 多版本流水

## 目标与适用条件

多版本流水用于让第 `i+1` 个 tile 的搬入、第 `i` 个 tile 的计算和第 `i-1` 个 tile 的搬出在硬件允许时重叠。它适用于 GM 搬运与 Vector/Cube 计算均占有可见时间、每核有足够多独立 tile、UB 容量能容纳全部版本的算子。小任务、单迭代或某条流水已接近理论下界时，多版本可能只增加 UB、同步和头尾开销。

实施前画出每个逻辑迭代的依赖：

```text
GM --CopyIn--> input/local buffer --Compute--> output/local buffer --CopyOut--> GM
```

对每个会在下一迭代被写入、但上一迭代仍可能被 Compute 或 CopyOut 读取的 UB/L1 buffer 标记版本。输入、输出、索引、临时中间量都按真实生命周期判断；不能假设“只给输入双缓冲”就足够。只读且跨迭代不变的常驻数据通常保持单版本。

同时核算版本总 footprint、padding、常驻数据和每核有效迭代数。有效迭代少于 stage 数时回退单 stage；3 stage 只有在 2 stage 后仍有流水缺口且容量允许时才进入候选。

## 自动与手动多版本

访问关系简单、仿射且编译器能唯一识别生产者/消费者时，先试自动版本化。优先把无条件、固定 extent 的完整迭代从尾部处理中剥离，并用 `T.Pipelined` 暴露规范的 CopyIn → Compute → CopyOut 主体：

```python
input_ub = T.alloc_shared((tile_elems,), dtype)
output_ub = T.alloc_shared((tile_elems,), dtype)
T.annotate_buffer_versions({input_ub: 2, output_ub: 2})

for tile in T.Pipelined(full_tile_count, num_stages=2):
    T.copy(src[tile], input_ub)
    compute(input_ub, output_ub)
    T.copy(output_ub, dst[tile])

# 余数 tile 使用独立的单 stage 路径，不放进上面的稳态主体。
if has_tail:
    process_tail(...)
```

这个自动模板有四个必须同时满足的结构条件：

1. `input_ub/output_ub` 按**单份逻辑 tile** 分配；不得先写显式 `[2, ...]` stage 维度再调用 `annotate_buffer_versions(...: 2)`，否则等于同时要求手动和自动扩版本，可能造成 extent 重写错误。
2. 只版本化稳态迭代中确实跨迭代存活的可变 buffer。只读 LUT/index、流水外 tail buffer 和未参与该主体的 fallback buffer保持单版本。
3. `T.Pipelined` 主体每次执行相同 rank、shape 和 copy extent 的 CopyIn/Compute/CopyOut。会改变访问范围的 full/tail 分支、零工作分支和动态余数处理放在主体外。
4. 首个自动代表优先直接访问版本化 buffer 的原始逻辑 shape；不要再通过带隐藏 extent 的 `view`、扁平别名或显式 stage 下标访问。若 intrinsic 需要 base pointer，仍须保证它可追溯到同一个逻辑 buffer 和固定范围。

`T.Persistent(..., num_stages=2)` 不是禁止项，但不能只因它带 `num_stages` 就视为自动流水的标准形态。只有它同样提供无条件、固定 extent、唯一生产者/消费者的稳态任务主体时才作为自动候选；否则先用上面的 `T.Pipelined(full_tile_count, ...)` 代表裁决自动版本化。CANN/编译器升级也不能代替这些结构条件。

`num_stages` 只是调度意图，不能代替 buffer versions 和依赖核验。view/别名、flattened index 或 gather/scatter 可能令自动分析错误认领 storage、扩大 extent 或失去版本关系；先最小化并确认失败位置，再改用显式 stage storage 和当前版本实际存在的手动 annotation：

```python
input_ub = T.alloc_shared((2, tile_elems), dtype)
output_ub = T.alloc_shared((2, tile_elems), dtype)
T.annotate_manual_multi_buffer(input_ub, output_ub)

for wave in T.Pipelined(full_waves, num_stages=2):
    stage = wave % 2
    copy_in(wave, stage)
    compute(stage)
    copy_out(wave, stage)
```

所有跨迭代存活的 buffer 使用同一 stage。这里要区分普通动态 UB 下标与显式 SIMD intrinsic 的动态 base pointer：标量/SIMT 的 `ub[index]`、`T.copy` 的动态槽位、以及 `vgather2/vscatter` 的逐 lane offset 都不能因某次失败被概括为“不支持动态地址”；风险组合是 `ub[stage, ...]` 被包装成 `vld/vsts/vgather2/vscatter` 等 intrinsic 的 base pointer，当前后端可能在版本分析、指令选择或地址合法化阶段失败。

若失败证据落在这种动态 base 上，只淘汰“动态 stage addressing”组合，并在外层按 stage 调用编译期常量的通用 body：

```python
stage = wave % 2
copy_in(wave, stage)  # T.copy 能否使用动态槽位仍按当前 lowering 核验。
if stage == 0:
    compute(input_ub[0], output_ub[0])
    copy_out(output_ub[0], wave)
else:
    compute(input_ub[1], output_ub[1])
    copy_out(output_ub[1], wave)
```

这样 SIMD body 内的 base 是编译期常量，但算法、tile 和 runtime kernel 仍保持通用；不能按测试 shape 特化。静态 stage 也不能只凭编译成功晋升，仍需与同算术、同任务映射的 stage-1 版本配对验证 latency 与 overlap。API 与 lowering 必须从实际导入的 TileLang/PTO 或仓库已验证实现核对。

流水候选按四个维度记录，失败只否决准确组合：

1. **Storage**：自动 versions，或显式 `[stage, ...]` storage 加手动 annotation；
2. **Addressing**：动态 stage，或由外层 stage 分支/macro 形成的静态 stage 0/1 body；
3. **Control flow**：条件式循环，或剥离尾部后的无条件完整 full-wave；
4. **Scheduler**：`Persistent`/`Pipelined`、stage 数和每核稳态迭代数。

当每核有足够独立迭代且 profiling 表明 MTE 与 Compute 均有可重叠时间时，流水候选即已准入。准入后先试自动组合；只有全部跨迭代存活 buffer 均正确版本化、生成代码存在预期切换/同步，且 stage-1/2 配对的 latency/overlap 改善可复现，才算自动组合完整。

自动版本化的 `not eligible`、extent/别名错误、lowering 失败、只版本化部分存活 buffer或实测未生效，只淘汰自动组合。无论用户是否给出数值目标，在宣告已准入的流水方向失败或收敛前，都必须完成以下手动代表，或提供当前 lowering、容量、依赖或 API 的确定不可实施证据：

- 输入、输出及所有跨迭代存活临时 buffer 均为显式多版本；
- 手动 annotation，必要时配合静态 stage body；
- `T.Pipelined(..., num_stages=2)` 的无条件 full-wave，尾部独立；
- 与同 tile、同算术、同任务映射的 stage-1 control 紧邻配对实测。

自动组合已经完整生效时，可以不再测试手动等价实现，但必须明确记录 `NOT_NEEDED_AUTO_COMPLETE`。某个工作粒度或数据流使自动版本化失效，不能作为另一工作粒度/数据流下手动流水不可行的证据。

手写 `set_flag`/`wait_flag` 或 event 次序不是默认替代方案；只有当前仓库或实际安装源码存在同访问模式的验证示例时才进入候选。死锁只否决对应 event 方案。

稳态流水主体必须让调度器可见；不要用运行时条件包住整段 CopyIn/Compute/CopyOut。用通用边界剥离尾部：

```text
full_waves = work_items // items_per_wave
pipeline(full_waves):
    无条件执行完整 CopyIn → Compute → CopyOut
if work_items % items_per_wave != 0:
    单独处理最后一个尾波
```

动态 shape、越界保护和尾块语义仍须保留。选择 `Pipelined` 或 `Persistent` 取决于当前 API 的迭代映射，不能仅凭名字判断。

## 生效判定

编译成功、出现两个 buffer、`num_stages > 1`、运行正确或单条 pipe ratio 变化，都不能单独证明流水生效。必须在同口径下配对比较 stage 1/2：

- kernel latency 的改善超过测量噪声并可复现；
- 按源码实际 GM 字节数计算的有效带宽或实际算力改善；
- 与假设对应的 MTE/Vector/Cube overlap 改善，且总耗时同步下降；
- 必要时检查生成 IR/代码中的版本切换、同步和满波调度结构。

若 latency/overlap 没有超出噪声的改善，记录“编译通过但未生效”。stage-2 失败记录完整组合标识，不得外推到其他 tile、Vector 数据流或控制流。

流水不能改变运算、cast 和输出语义；尾 tile 的每个版本独立初始化且只搬 valid 范围。覆盖 stage/tile 边界、非对齐尾块和接口规定的极值。带多版本、同步或调度 warning 的候选必须在同一进程跑全部目标 case，并改变 case 顺序复测；出现顺序相关漂移、偶发错误或内存破坏时淘汰。不得放宽精度或跳过 case。

## 可执行代码与证据

先在当前仓库检索 `annotate_buffer_versions`、`annotate_manual_multi_buffer`、`num_stages`、`T.Pipelined` 和 `T.Persistent` 的真实用例，再检查实际导入的 TileLang 源码与 PTO lowering。可从 `examples/ascend/example_simdvf_vecadd.py`、`examples/ascend/example_manual_multibuffer.py`、`examples/ascend/example_crosslevel_multibuffer.py`、`examples/ascend/example_rmsnorm.py` 及仓库内匹配到的其他 Ascend 实现开始，但每次复用都必须重新核验适用的访问模式、版本和设备证据。
