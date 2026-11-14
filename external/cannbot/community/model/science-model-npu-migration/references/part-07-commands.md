# 分册 7：命令模板

> 无主流程独立 §；按需引用，不阻塞主线。代码改动模式见 [reference-code-patterns.md](reference-code-patterns.md)。

说明：昇腾运行时与 `torch_npu`、MindSpore Ascend 等版本会随 CANN 变化。给出命令前先确认 CANN 与框架插件版本；未确认项用占位符。**`science-model-npu-migration` 为代码级迁移 skill**，命令模板见下文。

---

## 环境验证

```text
# 0) 若在沙箱中，先做沙箱内检测；失败再到沙箱外复检
# （以下命令在两侧保持一致，便于对比）

npu-smi info

python --version
python -c "import torch; import torch_npu; print(torch.__version__); print('torch_npu ok')"
# MindSpore：
# python -c "import mindspore as ms; print(ms.__version__)"
```

建议输出两组结果：`sandbox_in` 与 `sandbox_out`，并给出最终采信结论。

**注意**：IDE/CI **受限会话或沙箱**内 `npu-smi` 常失败；须在本机终端沙箱外复检，见 [part-03-environment.md](part-03-environment.md) §4.0.1。

---

## CANN 环境加载

**Linux / bash**

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 或项目文档指定的 set_env 路径
export ASCEND_RT_VISIBLE_DEVICES=0
npu-smi info
```

**Windows / PowerShell**

```powershell
# 路径以本机 CANN 安装为准
. "C:\Program Files\Huawei\Ascend\ascend-toolkit\latest\set_env.ps1"
$env:ASCEND_RT_VISIBLE_DEVICES = "0"
npu-smi info
```

加载结果（路径、是否成功）写入 `mig_docs/working/environment.md` 的 `set_env` 与 `generated_at`。

---

## 环境准备目标（与 `environment.md` 对齐）

> 目标条目见 [`environment-setup-objectives.md`](environment-setup-objectives.md)；门禁见 [part-03-environment.md](part-03-environment.md) §4.0。

- **数据路径**：[`Mig_Readme.md`](mig_docs/working/Mig_Readme.md) §3.1 / §3.2  
- **快照**：[`environment.md`](mig_docs/working/environment.md)、[`Mig_report.md`](mig_docs/working/Mig_report.md) §2.1  

---

## NPU 推理（PyTorch + torch_npu）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0

python tools/infer.py \
  --config configs/infer_npu.yaml \
  --checkpoint <path/to/weights> \
  --device npu:0 \
  --input <path/to/sample_or_list> \
  --batch-size <N> \
  2>&1 | tee logs/infer_npu_smoke.log
```

---

## NPU 训练（PyTorch 单卡）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0

python tools/train.py \
  --config configs/train_npu.yaml \
  --device npu:0 \
  --amp \
  --max-steps <N> \
  2>&1 | tee logs/train_npu_smoke.log
```

---

## 多卡训练（PyTorch + HCCL，示意）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 tools/train.py \
  --config configs/train_npu.yaml \
  --device npu \
  --amp \
  --max-steps <N> \
  2>&1 | tee logs/train_npu_ddp.log
```

脚本内须 `dist.init_process_group(backend="hccl")` 且每 rank `torch.npu.set_device(local_rank)`。详见 [reference-code-patterns.md](reference-code-patterns.md) §4。

---

## MindSpore Ascend

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0

python train.py --device_target Ascend --device_id 0 --data_path <path>
python eval.py --device_target Ascend --device_id 0 --checkpoint <path>
```

---

## Golden 样本对比（PyTorch，示意）

```bash
# 固定输入，分别跑 GPU baseline（用户侧）与 NPU；对比输出 pkl/npy
# 容差须按目标精度调整（本 skill 默认 FP16，勿套用 FP32 级阈值）
python tools/export_golden.py --device npu:0 --input fixtures/golden_input.bin --out runs/npu_out.npy
python tools/compare_golden.py --ref runs/gpu_out.npy --hyp runs/npu_out.npy --rtol 1e-2 --atol 1e-3
```

> **容差与精度**：FP16 有效位约 3 位、最小正规数约 6e-5，NPU 与 GPU 正常舍入差可超过 1e-5；默认 FP16 路径建议 `--rtol 1e-2~1e-3`、`--atol 1e-3` 量级。FP32 可收紧至 `--rtol 1e-5~1e-4`、`--atol 1e-5~1e-6`。须在 `Compare` §3.1 写明所用阈值及目标精度。

结果摘要写入 `Compare` §3.1、`Mig_report` §6。

---

## 性能 benchmark（推理延迟/吞吐，示意）

```bash
python tools/bench_infer.py \
  --device npu:0 \
  --batch-size 1,4,8 \
  --warmup 50 \
  --iters 200 \
  --report-json runs/bench_npu.json
```

**口径**（须写入 `Compare` §2.3）：warmup 次数、计时是否含 H2D、是否含后处理、batch、p50/p95 定义。

---

## 失败日志留痕

```bash
{ your_command_here; } 2>&1 | tee logs/last_run_$(date +%Y%m%d_%H%M%S).log
```

失败/精度/性能异常时更新 `Mig_report` **§7**、**§8**；排查见 [part-09-examples-troubleshooting.md](part-09-examples-troubleshooting.md)。

**交叉引用**：精度/性能表填 [`Compare.md`](mig_docs/working/Compare.md) §2.4～§4。

---

## 关联索引

- **代码模式**：[reference-code-patterns.md](reference-code-patterns.md)  
- **配合分册**：part-03 / part-04 / part-05  
- **失败留痕**：part-06、part-09  
- **流程总览**：[workflow.md](workflow.md)
