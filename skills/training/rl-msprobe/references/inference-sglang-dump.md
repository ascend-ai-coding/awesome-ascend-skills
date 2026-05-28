# SGLang 推理侧 msprobe 采集

verl 使用 SGLang 作为 rollout 后端时，推理 dump **不能**像 vLLM 一样仅通过 verl 的 `additional_config` 一键开启，需按 SGLang 版本选择 msprobe 官方采集方式，并在 verl 中关闭图模式。

## 强约束

- **必须 eager 模式**：`actor_rollout_ref.rollout.enforce_eager=True`
  - verl 内部映射为 SGLang `disable_cuda_graph=True`（见 `async_sglang_server.py`）
- 推荐 `export TORCHDYNAMO_DISABLE=1`
- PD 分离场景：`export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0`
- 在线模式：verl 已默认 `skip_server_warmup=True`

## 步骤 0：确认 SGLang 版本

```bash
python3 -c "import sglang; print(sglang.__version__)"
```

| 版本 | 采集方式 |
|------|----------|
| **>= 0.5.11** | SGLang 原生 `--msprobe-dump-config`，**无需改 SGLang 源码** |
| **< 0.5.11** | 侵入式修改 `sglang/srt/model_executor/model_runner.py` |

官方文档：

- 旧版（< 0.5.11）：[sglang_eager_dump_instruct.md](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/dump/sglang_eager_dump_instruct.md)
- 新版（>= 0.5.11）：[sglang_eager_dump_instruct_new.md](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/dump/sglang_eager_dump_instruct_new.md)

本地副本：`msprobe/docs/zh/dump/sglang_eager_dump_instruct*.md`

---

## 路径 S1：SGLang >= 0.5.11（推荐）

### 1. 准备 config.json

`level` 优先使用 `"mix"`（L0+L1 混合粒度，skill 默认推荐）。

```json
{
  "task": "statistics",
  "dump_path": "/path/to/dump/generate",
  "rank": [],
  "step": [],
  "level": "mix",
  "async_dump": false,
  "statistics": {
    "scope": [],
    "list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

### 2. verl 脚本追加参数

通过 `engine_kwargs.sglang` 传入 SGLang ServerArgs：

```bash
export TORCHDYNAMO_DISABLE=1

python3 -m verl.trainer.main_ppo \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.enforce_eager=True \
  ++actor_rollout_ref.rollout.engine_kwargs.sglang.msprobe_dump_config=/path/to/generate_config.json \
  ... # 原脚本其余参数
```

> `msprobe_dump_config` 对应 SGLang CLI `--msprobe-dump-config`。

PD 分离额外环境变量：

```bash
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
```

Prefill/Decode 分别使用不同 config（`dump_path` 不可相同）：

```bash
++actor_rollout_ref.rollout.engine_kwargs.sglang.msprobe_dump_config=/path/to/config_prefill.json
# decode 节点单独配置 config_decode.json
```

---

## 路径 S2：SGLang < 0.5.11（源码 patch）

按 [sglang_eager_dump_instruct.md](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/dump/sglang_eager_dump_instruct.md) 修改 **SGLang 安装目录**（非 verl）：

**文件**：`sglang/srt/model_executor/model_runner.py`

1. `ModelRunner.__init__`：初始化 `PrecisionDebugger(config_path=...)`
2. `ModelRunner.forward`：包裹 `debugger.start(model=self.model)` → forward → `stop()` → `step()`
3. DP 场景（`dp_size>1`）：`start(..., rank_id=self.gpu_id)`

verl 侧仍须：

```bash
actor_rollout_ref.rollout.name=sglang \
actor_rollout_ref.rollout.enforce_eager=True \
```

config.json 路径在 SGLang patch 代码中硬编码或通过环境变量传入。

---

## 步骤 3：验证产物

SGLang msprobe dump 默认写入 config 中 `dump_path`：

```text
{dump_path}/step_N/rank_M/dump.json
```

发送 rollout 请求后检查目录是否生成 `dump.json`。

## verl + SGLang 对照

| verl 参数 | SGLang 等效 |
|-----------|-------------|
| `rollout.enforce_eager=True` | `--disable-cuda-graph` |
| `engine_kwargs.sglang.msprobe_dump_config` | `--msprobe-dump-config` |
| verl 内置 | `skip_server_warmup=True` |

## 校验清单

- [ ] `bash scripts/ensure-msprobe-env.sh` 通过
- [ ] `actor_rollout_ref.rollout.name=sglang`
- [ ] `actor_rollout_ref.rollout.enforce_eager=True`
- [ ] SGLang 版本与采集路径（S1/S2）匹配
- [ ] config `dump_path` 可写
- [ ] PD 分离时已设 `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0`

## 故障排查

| 症状 | 处理 |
|------|------|
| 无 dump | 版本 < 0.5.11 但未 patch ModelRunner |
| dynamo 报错 | `export TORCHDYNAMO_DISABLE=1` |
| health 请求污染 dump | PD 模式设 `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0` |
| dp 采集不一致 | 固定 `bootstrap_room`（见 msprobe PD 分离章节） |

## 参考

- msprobe SGLang 采集（旧）：https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/dump/sglang_eager_dump_instruct.md
- msprobe SGLang 采集（新）：https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/dump/sglang_eager_dump_instruct_new.md
- SGLang msprobe 官方指南：https://docs.sglang.io/docs/developer_guide/msprobe_debugging_guide
