# ATK 安装、执行与失败复现

## 验证安装

从官方项目或交付方获取 ATK wheel，核对 Python ABI、操作系统、架构、后端和 SHA256。[ATK 官方项目](https://gitcode.com/AscendTest/ATK)

已有兼容 ATK 时可直接使用。需要安装时，从现有计算环境创建虚拟环境。NPU 节点按平台文档加载 CANN。

```bash
# 填写已核验的 wheel 和本次独立环境路径。
sha256sum "$ATK_WHEEL"
python3 -m venv --system-site-packages "$ATK_VENV"
"$ATK_VENV/bin/python" -m pip install --no-deps "$ATK_WHEEL"
"$ATK_VENV/bin/python" -m pip check
"$ATK_VENV/bin/atk" task --help
```

根据 wheel 元数据补齐工具依赖。安装前后检查 Torch、TorchNPU、CUDA/CANN 的版本和加载路径，确认沿用原计算环境。保存 ATK、设备、驱动、框架和 Python 版本，以及配置样例和 CLI 帮助。

## 生成和执行

1. 从当前 ATK 版本的模板生成 case JSON、node YAML 和所需插件。记录批次 ID、ATK case ID 和报告 ID 的映射。
2. 标明各节点承担的角色。双标杆需记录候选、同精度对照和高精度 Golden。离线对照使用相应节点本轮保存的输出，并附环境、输入和输出哈希、参数及阶段。
3. 插件按 case ID 读取输入和梯度，按 API 签名传参。若使用 ATK 生成输入，所有节点使用同一份生成结果。当前版本支持时可用 `--syc_dataset` 分发，也可导出最终输入后逐节点校验。先用预期已知的小样检查节点角色和报告，再用边界值检查比较器及异常处理。
4. 核对 CLI 帮助后执行。只有使用插件时才传 `-p` 和文件名。

   ```bash
   "$ATK_VENV/bin/atk" task -c "$CASE_JSON" -n "$NODES_YAML" -p "$PLUGIN" -to 1800
   ```
5. 从日志定位报告目录，保留命令参数、退出码、原始报告及输出清单。逐个核对用例 ID、执行状态、精度结果和次数。报告解析以当前 ATK 版本的实际输出为准，未知状态标为未判定。

## 单用例复现

- 保留原命令和日志，再提取失败用例。重跑保留原阶段、节点角色、比较判据、计算模式及相关环境设置。使用生成器时，记录原种子和用例 ID 参与种子生成的设置。分别记录环境错误、算子报错、精度超差和重复执行不一致。
- 保存只含原失败 ID 的 case JSON、node YAML、插件和依赖、输入与梯度或生成器，并写明设备、ATK 版本和判据。精简输入或更改形状后，重新确认是否出现原错误。
- 复现需要重新执行算子。ATK 覆盖不了的功能、内存或计时操作附上原测量命令。仅用已存输出检查比较器时，在报告中注明。
- 将获准交付的文件和 SHA256 写入 `repro.json`。若用可公开输入代替受限数据，先用 ATK 核对原问题是否重现。在独立测试环境加载已审查的插件和输入。

以下是 `repro.json` 示例，填写实际文件名和 SHA256。

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

把输入及插件依赖放入包内，使用相对路径和框架支持的安全加载方式。本地启动的 ATK 从暂存后的包根目录和插件目录导入 Python 依赖。远端节点的依赖按其部署方式安装或分发。无插件时省略 `plugin_file`。ATK 导出的最终输入和离线标杆分别填入 `input_directory` 与 `benchmark_outputs_directory`，逐文件登记哈希。助手会检查 CLI 是否支持 `--input_data` 和 `--bm_output_path`。

```bash
python3 scripts/replay_atk.py path/to/repro-42 --output path/to/repro-run-42 \
  --atk "$ATK_VENV/bin/atk"
```

在原包外指定一个新目录，父目录须已存在。可传 `--manifest-sha256` 核对可信渠道提供的清单摘要。助手校验文件后在 `work/` 下运行 ATK，将日志和退出码写入 `console.log` 与 `execution.json`。复现状态初始为 `NOT_EVALUATED`。用新报告核对同一阶段和节点角色是否出现原报错或超差。

如本版支持，可用 `--save_data input_final:bin --save_data output:bin --save_report_json` 保存最终输入、输出和 JSON 报告。复现依赖助手未支持的开关时，提供验证过的完整命令。

按原测试填写 `benchmark_device`（`npu`、`gpu` 或 `cpu`）、`repeat_count` 和 `determinism_repeat_count`。助手检查并传入 `--bm_device`、`--loop_nums` 和 `--dc_loop_nums`。 `timeout_seconds` 对应 ATK 的 `-to`。进程级超时由作业系统设置。
