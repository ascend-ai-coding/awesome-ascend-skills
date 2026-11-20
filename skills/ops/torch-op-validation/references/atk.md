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

使用已验证的工具依赖组合并保存版本锁定记录。CLI 可启动不代表任务能完成；队列、插件或依赖初始化错误记为基础设施 `ERROR`，不计为算子数值失败或执行通过。

## 生成和执行

1. 从当前 ATK 版本的模板生成 case JSON、node YAML 和所需插件。记录批次 ID、ATK case ID 和报告 ID 的映射。
2. 标明各节点承担的角色。双标杆需记录候选、同精度对照和高精度 Golden。离线对照使用相应节点本轮保存的输出，并附环境、输入和输出哈希、参数及阶段。
3. 插件按 case ID 读取输入和梯度，按 API 签名传参。若使用 ATK 生成输入，所有节点使用同一份生成结果。当前版本支持时可用 `--syc_dataset` 分发，也可导出最终输入后逐节点校验。训练反向在每端重新执行前向并使用该端缓存，不能给 NPU 反向传入 Golden 的前向缓存。按[精度与用例](precision.md)落实设备地址 NaN 预填并保留执行证据。
4. 核对 CLI 帮助后执行。只有使用插件时才传 `-p` 和文件名。

   ```bash
   "$ATK_VENV/bin/atk" task -c "$CASE_JSON" -n "$NODES_YAML" -p "$PLUGIN" -to 1800
   ```
5. 从日志定位报告目录，保留命令参数、退出码、原始报告及输出清单。按 API 类型、前后向和每个 output/gradient 叶子核对用例 ID、执行状态、三方角色、实际 dtype、精度结果和次数。多输出不能合并为单个主输出指标；RMSNorm 等算子的 output 与 rstd 分别登记 dtype 和二进制记录。报告解析以当前 ATK 版本的实际输出为准，未知状态标为未判定。

## 核对 2.1 比较器

批量测试前，用结果已知的小样核对以下行为，并保存比较器版本及原始报告：

- 浮点计算的三方身份、Golden 精度、正常值域 MARE/MERE/RMSE，以及 `metric_npu / max(metric_peer, ratio_denominator_err)`。将 `ratio_denominator_err` 的数值、来源和解释写入记录；它不是相对误差公式中的 `1e-7`。
- 对照误差为零或小于分母下限、比值恰等于等级上限的边界；小值域的 `< threshold`、`> error` 和计数比 `<= 2`；Inf/NaN 按芯片模式及三方结果判断。
- 非计算类的原 dtype 位一致、整数类的位一致或 AE=0、量化整型输出的适用分支；随机类使用独立的分布检验。

浮点 ATK 配置可使用 `cv_fused_double_benchmark`，等级上限对应 `max_re_ratio`、`avg_re_ratio`、`root_mean_squared_ratio`。字段名称相同不足以证明实现符合 2.1；必须核对分母、小值域和非有限值处理。工具无法实现某条要求时保留原始输出，补做相应的 2.1 检查并标明范围，不能把默认 `Passed` 写成 2.1 正式通过。

三方两两的原 dtype 位差、ULP、绝对/相对误差与 RMSE 按[诊断定义](precision.md#三方附加诊断)保存，不配置额外通过门槛。核对 ATK 导出字段的实际含义：位差必须区分位差总数与位差元素数；不同 dtype 不能直接逐位等同比较；ULP 须确认计算 dtype、转换/舍入、符号零和非有限值处理。舍入 Golden 得到的 ULP 不能替代原高精度 Golden 的 2.1 指标。剔除数记录统计范围，不等于小值域 `ErrorCount`；行最大位置指标须能确认轴、argmax 来源、并列值处理及分母，未确认的字段记为缺测。

报告中的 2.1 超标、执行错误及关键证据缺失标红，附加诊断数值本身不触发标红。诊断字段缺测保留原因；是否属于阻碍正式验收的关键证据，应依据适用 2.1 条款判断。

按等级配置并核对实际完成的压测次数（L0/L1 每例 50 次，L2 每例 1000 次）。CLI 参数须以本版帮助和小样实测为准；默认循环次数、工具重试和确定性重复不能互相代替。原位、状态与 RNG 在每次执行前按用例定义恢复。原输入失败复现与换种子统计复检分开保存，复检方法见[精度与用例](precision.md)。

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
