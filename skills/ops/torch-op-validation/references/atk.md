# ATK 安装、执行与失败复现

## 验证安装

使用 [ATK 官方项目](https://gitcode.com/AscendTest/ATK)发布或交付方明确提供的版本匹配 wheel。核对 Python ABI、OS、架构、各节点后端及 wheel 哈希。不能确定来源时不要安装同名包；不要把 wheel 和内部插件提交到公开仓库。

在正确的计算栈上创建独立环境。NPU 节点按其平台文档加载 CANN 环境脚本；CUDA 节点不用加载 CANN。已有兼容 ATK 可复用。

```bash
# 填写已核验的 wheel 和本次独立环境路径。
sha256sum "$ATK_WHEEL"
python3 -m venv --system-site-packages "$ATK_VENV"
"$ATK_VENV/bin/python" -m pip install --no-deps "$ATK_WHEEL"
"$ATK_VENV/bin/python" -m pip check
"$ATK_VENV/bin/atk" task --help
```

`--no-deps` 不会补齐依赖。根据 wheel 元数据从受信来源补齐工具依赖；安装前后校验 Torch、TorchNPU、CUDA/CANN 和相关组件的版本及加载位置，不能用新包覆盖原栈。分别保存 ATK 版本证据、设备/驱动、框架、所用 Python、配置样例出处、实际 CLI 帮助和环境检查结果。只在测试节点空闲时初始化设备并执行小样。

## 生成和执行

1. 为每个 ATK 批次生成原生 case JSON、node YAML 和必要插件；选择对应发行包提供的模板，不从结果表反推 ATK schema。保留批次 ID 和 ATK case ID 到报告 ID 的映射。
2. 节点记录候选、同精度对照和高精度 Golden 三个角色。离线对照是本轮另一节点真实执行后保存的输出；保存执行环境、输入哈希、调用参数、输出哈希和阶段。不要通过本地 CPU 模拟第三方输出。
3. 插件从明确的 case ID 读取冻结输入和梯度，按固定的参数绑定调用目标。先执行一个包含可信预期的真实小样，另用已知边界值核对比较器及异常处理；小样报告必须确认各节点执行了相应角色。
4. 用所装版本的 CLI 执行。常用形式如下，需核对帮助；不需要插件时省略 `-p` 和文件名。

   ```bash
   "$ATK_VENV/bin/atk" task -c "$CASE_JSON" -n "$NODES_YAML" -p "$PLUGIN" -to 1800
   ```
5. 按实际 ATK 日志找到报告目录。保存实际 argv、退出码、日志、原始报告及输出数据清单。核对预期和实际 ID 集合、执行状态、精度结果及重复次数。只看退出 0、结果行数或汇总百分数不能判断全部通过。解析器以固定版本的真实报告为样本，未知字段或状态保持未判定。

## 缩小失败与再执行

- 原始失败不可覆盖。首先隔离一个真正失败的 case 和同样的阶段、角色与计算模式。环境错误、算子执行失败、数值不一致和同后端重复不一致分别记录；执行失败时不能归为精度比较失败。
- 保留可用的原生单例 JSON（数组中有且只有原失败 ID）、node YAML、插件和本地依赖、张量/梯度或确定性生成器、设备要求、检查规则与 ATK 版本。输入精简或修改形状后，必须重新验证还存在原失败；否则仅标为候选复现。
- 功能、内存和性能问题只有在 ATK 能执行同等行为时才标为 ATK 可复现。若 ATK 只复现算子调用，额外交付相应测量入口；仅回放已存输出的比较器不算算子重执行。
- 将可分享的文件及哈希写入 `repro.json`。实际使用前检查插件、生成器、节点定义和数据均来自可信来源；哈希只验证内容一致性，不保证代码安全。复现脚本不提供进程隔离，运行时应在已授权的独立环境。

`repro.json` 结构如下（哈希及文件名须用实际值替换）：

```json
{
  "schema_version": 1,
  "case_id": 42,
  "case_file": "case.json",
  "node_file": "nodes.yaml",
  "plugin_file": "operator.py",
  "timeout_seconds": 1800,
  "files": {
    "case.json": "<sha256>",
    "nodes.yaml": "<sha256>",
    "operator.py": "<sha256>",
    "fixtures/input.pt": "<sha256>"
  }
}
```

如插件读取张量文件，包含全部依赖并从包根目录相对读取。应启用框架可用的安全加载方式，不读取不可信的序列化对象。没有插件时省略 `plugin_file`。ATK 自带的最终输入和离线标杆文件可分别指定 `input_directory` 与 `benchmark_outputs_directory`，目录内每个所需文件均须列出哈希。重跑前会检查当前版本是否支持 `--input_data` 与 `--bm_output_path`；留存初测批次实际导出的格式，不能从文件后缀推断兼容。

```bash
python3 scripts/replay_atk.py path/to/repro-42 --output path/to/repro-run-42 \
  --atk "$ATK_VENV/bin/atk"
```

可加 `--manifest-sha256` 核对从可信渠道获得的清单摘要。输出目录须不存在、父目录须存在，且不得位于原包内。脚本复制并校验所列文件，在 `work/` 下运行一个 ATK 任务，输出 `console.log` 与 `execution.json`。ATK 返回值会保留，复现结论初始为 `NOT_EVALUATED`。在新报告确认同一阶段和角色出现同类错误或超差后，才将其标为已复现。

部分版本支持保存最终输入、输出及 JSON 报告（例如 `--save_data input_final:bin --save_data output:bin --save_report_json`）；首次执行时先核对实装版本。复现若需要本脚本未支持的开关，交付经过验证的完整手工 argv，说明哪些参数由使用方填写；不要删掉能触发失败的设置。

若原失败依赖 ATK 的辅助标杆执行端或重复次数，可在清单填写 `benchmark_device`（`npu` / `gpu` / `cpu`）、`repeat_count`、`determinism_repeat_count`。脚本校验参数并确认当前 CLI 支持 `--bm_device`、`--loop_nums`、`--dc_loop_nums`；实际值须与初测一致。 `timeout_seconds` 传给 ATK 自身的 `-to`，若执行系统还要求进程级超时，由测试平台以单独的作业时限管理。
