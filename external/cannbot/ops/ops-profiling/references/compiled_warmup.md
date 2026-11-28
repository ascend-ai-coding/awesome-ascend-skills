# 工程内 compiled warmup

## 定位与使用条件

该 warmup 完全由 `ops-profiling` 自带，不依赖外部评测框架。默认先使用现成的 Max/ArgMax 预热；只有 ops probe 证明目标环境无法运行所选算子时，才构建并使用本包。

它不是目标算子的功能预热，而是在每个 target 前对固定256 MiB FP16 tensor 做一次全量 `x += 1`，形成确定性的 GM 读、Vector Add、GM 写流量。

kernel 固定名为 `OpsProfilingWarmup`，实现身份固定为 `ops_profiling_warmup_v1`。它用于 cache/frequency conditioning，不宣称执行硬件级 L2 invalidate。

## 落地到目标工程

从 skill 根目录执行：

```bash
python3 scripts/install_compiled_warmup.py \
  --project-dir /path/to/operator-project
```

默认生成 `/path/to/operator-project/ops_profiling_warmup_ext/`。目标已存在且内容相同时脚本幂等成功；存在本地修改时拒绝覆盖。

只有 ops probe 失败后才在目标环境中构建：

```bash
cd /path/to/operator-project
bash ops_profiling_warmup_ext/build.sh --soc=ascend950
```

`--soc` 必须根据实际设备选择 `ascend910b`、`ascend910_93` 或 `ascend950`，不得猜测。构建成功产物为：

```text
ops_profiling_warmup_ext/ops_profiling_warmup/_C.so
```

构建脚本同时校验扩展不残留 AscendC runtime 未解析符号，并校验 profiler 可见的 kernel 名保持为 `OpsProfilingWarmup`；任一条件不满足时构建失败，不把缺陷推迟到采集阶段。

## Runner 接入

runner 启动后只 import 一次，并只分配一次 tensor：

```python
import sys
from pathlib import Path

extension_root = Path(__file__).resolve().parent / "ops_profiling_warmup_ext"
sys.path.insert(0, str(extension_root))
from ops_profiling_warmup import compiled_warmup, make_warmup_tensor

warmup_tensor = make_warmup_tensor(device)
```

每个 target 前调用：

```python
compiled_warmup(warmup_tensor)
torch.npu.synchronize(device)
target_op(*target_args)
torch.npu.synchronize(device)
```

execution manifest 的 `warmup` 必须逐项记录：

```json
{
  "mode": "compiled",
  "logical_op": "OpsProfilingWarmup",
  "implementation": "ops_profiling_warmup_v1",
  "tensor_bytes": 268435456,
  "tensor_allocation": {
    "shape": [128, 1024, 1024],
    "dtype": "float16",
    "elements": 134217728
  }
}
```

每个 case 的 `warmup_kernel_sequence` 必须为 `["OpsProfilingWarmup"]`；若目标环境的 profiler 没有记录该精确名称，采集必须失败并先修正本地扩展，不能放宽映射或改填猜测名称。

## 失败边界

ops probe 失败后，build 命令非零、超时或无法启动会让本次采集失败。构建成功后的 import、tensor 校验或 kernel 运行错误同样失败，禁止静默切回 Max/ArgMax。
