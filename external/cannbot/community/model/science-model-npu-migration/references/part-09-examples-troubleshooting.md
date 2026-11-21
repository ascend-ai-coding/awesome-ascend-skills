# 分册 9：示例与问题排查

> 无主流程独立 §；与 [part-06-risk-rollback.md](part-06-risk-rollback.md) 配合（失败路径）。代码模式见 [reference-code-patterns.md](reference-code-patterns.md)；命令见 [part-07-commands.md](part-07-commands.md)。

---

## 场景 A：PyTorch 推理迁移（端到端）

**输入**：用户给出 `tools/infer.py`、GPU 权重、固定样例图；目标 910 FP16。

| 步 | 执行动作 | 落盘 |
|:--:|------------|------|
| 1 | part-01 收集 IO shape、基线日志或 GPU 接口 | `Compare` §2.1、`Mig_Readme` §3.1 |
| 2 | part-02 判定链路：torch_npu + 无 CUDA 扩展 | `Mig_report` §2.2 |
| 3 | part-03 门禁；`npu-smi` 沙箱外复检 | `environment.md` 4.0.3=AUTO |
| 4 | part-04：`get_device()`、`.cuda()`→`.npu()`、`pin_memory=False` | `Mig_report` §4～§6 |
| 5 | smoke：3 张图前向，记录 shape/ max diff 量级 | §6 勾选推理 smoke |
| 6 | part-05：NPU 延迟 + Golden；用户 GPU baseline | `Compare` §3～§4 |
| 7 | part-08 归档 | 定稿 **`Summary.md`（最终交付）** |

**对话输出要点**：4.0.3 一行结论；smoke 命令；`mig_docs/` 已更新文件列表。

---

## 场景 B：PyTorch 训练迁移（含训练短测）

| 步 | 要点 |
|:--:|------|
| 4 | HCCL 多卡时改 backend；见 reference §4、part-07 多卡模板 |
| 5 smoke | 1 batch 反向；loss 有限 |
| 5 短测 | **part-05 §8.1.1**：loss 相对起点降 30%～50% **即停**，勿重复短测 |
| 失败 | loss 平盘且 >500 step → part-05 §8.2，写 §7，查 label/head/AMP |

---

## 场景 C：仅「检查 NPU 适配」

- 入口：part-03 §4.0.0  
- 产出：`environment.md` + AUTO/MANUAL_STOP/UNKNOWN  
- **不**填 `Compare` 全量、**不**跑 part-04 smoke  
- 回复须声明：「本次为适配检查路径，未执行完整迁移链路」

---

## 症状 → 原因 → 动作（速查表）

| 症状 | 常见原因 | 优先动作 | 落盘 |
|------|----------|----------|------|
| `RuntimeError: NPU error` / ACL | CANN/驱动未加载、版本不匹配 | 检查 set_env、`npu-smi`；回流 part-03 | §7、environment.md |
| `ImportError: torch_npu` | 插件未装或 Python 环境错误 | venv + 对齐 README 版本 | §7、§3 |
| `.cuda()` / `invalid device` | CUDA 残留 | reference §2 扫描替换 | §5.1 |
| `Unsupported op` / 自定义算子 | CUDA 扩展 | CPU 回退或算子替换 | §5.4、§7 |
| HCCL init failed | backend/可见设备/RANK 错误 | 核对 `ASCEND_RT_VISIBLE_DEVICES`、torchrun | §7、part-07 |
| 精度大幅下降 | 预处理/layout/head 接错 | Golden + 对齐 mean/std/NCHW | Compare §3.1、`Mig_report` §7 |
| loss NaN | AMP scale/dtype/学习率 | 关 AMP 试 FP32 smoke；查 loss 实现 | §7 |
| 延迟劣于预期 | IO/前后处理/ batch 过小 | bench 拆分阶段；Compare §4.1 | Compare §5 |
| 沙箱内 npu-smi 空 | 受限会话/沙箱无设备可见性 | **沙箱外复检**，采信宿主机 | environment.md |

---

## 常见问题排查顺序（详）

1. **运行失败**：device 与 `torch_npu` 导入 → CUDA 残留 → dtype/AMP/数据管线 → §7  
2. **加载失败**：CANN/驱动/插件版本 → 环境变量 → §8 日志路径  
3. **精度问题**：预处理 → layout/head → 算子/CPU 回退 → Golden 数值  
4. **性能问题**：统一 warmup/口径 → IO → 批大小/并发 → Compare §4～§5、`Mig_report` §7  

---

## §7 条目示例（精简）

```markdown
### 问题：infer smoke 报 Unsupported operator aten::xxx（2026-06-18）
- **触发命令**：python tools/infer.py --device npu:0 ...
- **现象**：ACL 000xxx，算子 xxx 不支持
- **已尝试**：|1| 换 CPU 后处理|成功 smoke|
- **根因**：后处理依赖 CUDA 专用 op
- **修复**：postprocess 改 CPU 路径再 to(npu)
- **验证**：3 样例前向通过，无 NaN
```

---

## 输出末尾建议附带

- **`Mig_report` §7**（若失败/回滚）：见 part-06 §9.4 模板  
- **`Compare.md`**：精度/性能结论一行摘要  
- **日志路径**：§8  

---

## 关联索引

- **回滚**：[part-06-risk-rollback.md](part-06-risk-rollback.md)  
- **代码模式**：[reference-code-patterns.md](reference-code-patterns.md)  
- **流程总览**：[workflow.md](workflow.md) 回流 part-03 / 04 / 05
