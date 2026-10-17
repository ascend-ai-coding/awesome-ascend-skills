# E04 实验与结果分析

## 1. 先判定非有限值发生在前向还是反向

先在实际训练步骤的 `logits` 观测点检查是否有限：

```python
if not torch.isfinite(logits).all().item():
    print("forward logits has NaN or Inf")
```

- 若前向 `logits` 首先出现非有限值，则判定非有限值发生在前向过程。
- 若前向 `logits` 正常，但反向或梯度阶段首先出现非有限值，则判定非有限值发生在反向过程。

`.item()` 会同步设备；该检查仅用于已批准的调试实验，并记录观察者效应。

## 2. 优先验证 Preflight 差异

对已确认的权重、代码、混合精度、loss scale、环境、并行或其他单一差异，逐个执行对齐、修复或回退实验。若非有限值在原复现
口径下稳定消失并完成同口径复跑，可在对应粒度形成根因闭环，不强制继续 detect_anomaly、overflow_check 或 tensor dump。
一次同时改变多个因素只能说明相关性。

## 3. detect_anomaly

先确认真实加载路径：

```python
import inspect
import training_entry

print(inspect.getsourcefile(training_entry))
```

再让上下文覆盖相关 forward 和 backward：

```python
with torch.autograd.detect_anomaly(check_nan=True):
    # 原有 forward/backward 逻辑
    ......
```

traceback 是候选证据，不是根因认证。若症状不复现，记录观察者效应，回退 detect_anomaly 后复核 R0；只有 R0 恢复复现时，
才经批准尝试更轻量的 statistics 采集。该采集仍须独立确认症状复现，不能直接进入 overflow_check。

若出现以下错误：

```text
NotImplementedError: Operator aten._is_any_true.default does not have a sharding strategy registered
```

按现场 PyTorch 版本核对并参考 [PyTorch PR #170951](https://github.com/pytorch/pytorch/pull/170951) 中的临时规避方法；
记录原始报错、版本和处理方式后，再观察 nonfinite 症状是否复现。

## 4. overflow_check

采集前声明不依赖 overflow_check 结果的非有限值信号、可比窗口和等价判据。采集结束后立即确认进程正常退出、预期 Step 已完成
现场要求的 `stop()`/`step()`，逐项核对预期 `Step × Rank` 的 `dump.json` 存在、非空、可解析且范围匹配，并检查
`dump_error_info.log`，记录 `dump_integrity`。再用同次运行的日志、原始报错、显式有限性观测或完整 `dump.json` 中的直接非有限统计独立记录
`symptom_reproduction`。只有 `dump_integrity=valid + symptom_reproduction=reproduced` 才能继续；其他组合阻断
overflow_check，按共享门禁回退、复核 R0 或以证据不足结束。

先探测现场：

```bash
msprobe --help
msprobe overflow_check --help
msprobe overflow_check -i /path/to/dump/step0 -o /path/to/output
```

只有命令成功且实际生成时才读取 `anomaly_analyze_*.json`。按每 Rank 的 dump 顺序检查：

- 正常输入、异常输出的计算节点优先；

### 4.1 核对现场 ignore_rules.yaml

`ignore_rules.yaml` 的路径和内容可能随版本变化。先从实际导入的 msProbe 包根目录定位，记录所有匹配文件的绝对路径、内容摘要和
本次实际采用的规则；没有找到或存在多份而无法确认生效文件时写 `unknown`，不得按网上规则假定已过滤。

```python
import hashlib
import inspect
from pathlib import Path
import msprobe

package_root = Path(inspect.getfile(msprobe)).resolve().parent
matches = sorted(package_root.rglob("ignore_rules.yaml"))
print("msprobe_package_root=", package_root)
for path in matches:
    data = path.read_bytes()
    print("ignore_rules=", path.resolve())
    print("sha256=", hashlib.sha256(data).hexdigest())
    print(data.decode("utf-8", errors="replace"))
```

当前官方说明列举的内置过滤场景包括：

- `torch.empty`、`empty_like`、`empty_strided`、`fill` 等未初始化内存相关节点；
- 分布式通信算子中，数据被实际覆盖前出现的野值；
- `Tensor.masked_fill_` 的 inplace 输入。

只有现场实际生效的 `ignore_rules.yaml` 含有相应规则时，才能写“已由内置规则过滤”；否则将其作为人工复核候选并记录版本差异。

### 4.2 人工复核规则

1. `full`、`zeros`、`ones` 及 NPU 初始化/填充节点先标为初始化或占位节点；不能仅因统计输出包含非有限值就认定根因，必须检查实际写入者和消费者。
2. `to`、`clone`、`detach` 等搬运/复制节点通常继承上游异常；继续向生产者追溯，同时保留该节点的原始证据。
3. `where`、`triu`、`tril` 产生的 `-Inf`，仅在 mask、条件、三角矩阵语义和预期位置均吻合时才可标为合法。合法 `-Inf` 不豁免 NaN 或 `+Inf`；若进入不支持该值的后续计算并首次产生 NaN，后续节点仍是异常候选。
4. 每次过滤都记录算子、Tensor、Rank、命中规则或人工理由，以及实际写入者、生产者或下游消费者的检查结果。

## 5. 非有限转换点之前的有限值趋势

从首次观测到非有限值的位置沿真实数据依赖向前检查有限前驱，同时区分“有限放大起点”“后续放大器”和“首个非有限转换点”：

- 优先读取本次有效 PyTorch msProbe statistics dump 的 `dump.json` 原始 `max`、`min`、`mean`、`L2norm`，并绑定相同
  Step/Rank/方向/调用实例。字段缺失、为 `null` 或现场版本语义不明时写 `unknown`，不得补造。
- 第一个由有限输入产生非有限输出的节点记为“首个非有限转换点”。它界定 NaN/Inf 出现的计算边界，但可能只是对已被前序节点放大的
  有限输入执行了符合当前 dtype/算法的计算，不自动成为根因候选。
- 沿可比的数据依赖、同一逻辑 Tensor 的跨 Step，或同一逻辑分片/副本的跨 Rank 证据，寻找最早开始出现可重复异常量级增长的位置，
  记为“有限放大起点候选”；输入已被放大且输出继续增长的位置只记为“后续放大候选”。不同语义 Tensor 之间不能仅按数值大小排序。
- 不设跨模型通用倍数或阈值；排除初始化占位、通信覆盖前野值、合法 mask、算子预期尺度变化和观察者效应后再解释趋势。没有健康标杆或
  项目数值约束时，单次运行中的大有限值只能形成风险候选。
- 只有单变量修复/回退使某个放大起点及后续 E04 症状按原契约共同消失，才能提升其证据等级；首个非有限转换点也必须独立验证，不能
  因位置最靠近 NaN/Inf 就认证为根因。

## 6. 可选 Targeted Tensor、CPU 对照与最小复现

需要使用真实输入构造最小算子复现时，才采集候选 API 的输入输出。最小复现固定使用 dump 输入，输出有限性、max/min/mean、
dtype/shape；必要时增加最小反向。修复或回退后必须在原复现口径下验证非有限值消失。

候选已收敛到具体 API/Module 后，可经批准只替换该候选到 CPU，保持输入、权重、dtype、shape、非 Tensor 参数和调用方向，并记录
搬运、cast、隐式同步及性能变化。CPU 上症状消失只能形成 NPU 路径强候选，不能排除不同算法/同步造成的影响；实验后复跑 R0′，
并用 NPU 正式修复完成同口径验证。

最小脚本不复现时，记录其缺失的整网通信/计算并发、内存竞争、拓扑和上下文。该负结果不能排除候选；由 Plugin 编排时返回
整网 R0，通过单变量 Scope Reduction 和 R0′ 回退验证继续缩圈，再回到 E04。

## 判定等级

- 已验证根因：单一 Preflight 差异经对齐/修复/回退后非有限值稳定消失，或正常输入首次产生异常输出且重复复现并经修复/回退后消失。
- 强候选：Preflight 差异或异常链路与症状一致，但缺单变量实验、最小复现或修复验证。
- 证据不足：仅有最终 loss/grad NaN、传播节点或不完整 dump。
