# 昇腾官方算子与融合相关技术资料索引

> 用途：查证算子行为、接口参数、版本兼容性。
>
> **原则：只引用确实存在的资料；引用时注明来源与适用版本；不确定的一律标注 ⚠ 需核实并给出核实途径；禁止编造文档名、接口名、URL。**

## 目录

1. [官方入口（长期有效）](#1-官方入口长期有效)
2. [按用途的资料索引](#2-按用途的资料索引)
3. [环境自查命令](#3-环境自查命令)
4. [版本兼容性核实方法](#4-版本兼容性核实方法)
5. [本 Skill 自带资料](#5-本-skill-自带资料)

---

## 1. 官方入口（长期有效）

| 资源 | 入口 | 内容 | 备注 |
|---|---|---|---|
| 昇腾官网及文档中心 | https://www.hiascend.com/ | 全部昇腾产品文档入口，可按产品线与版本筛选：CANN、MindStudio、MindSpore、MindIE、硬件规格等 | 查任何文档都从这里进，先选对产品与版本 |
| 昇腾社区（Gitee） | https://gitee.com/ascend | 昇腾开源组织：torch_npu、MindSpeed、开源算子仓库、samples 等 | 仓库列表以组织页为准 |
| vllm-ascend | https://github.com/vllm-project/vllm-ascend | vLLM 的 Ascend 适配后端 | 可查主流 LLM 推理在昇腾上实际调用了哪些融合算子 |

## 2. 按用途的资料索引

> ⚠ **重要**：以下条目的分册名称与路径随 CANN / 工具版本变化，以文档中心当前版本的目录为准。这是"入口可信、具体路径需按版本核对"的常态——引用时分册名要写"以 vX.Y.Z 版本目录为准"。

| 用途 | 资料位置 | 核实状态 |
|---|---|---|
| CANN 安装 / 升级 / 版本行为变化 | 昇腾文档中心 → CANN → 对应版本 → 安装指南、Release Notes | 入口可信；具体分册按版本号核对 |
| 算子开发（Ascend C） | CANN 文档 → 算子开发相关分册（Ascend C 算子开发指南 / API 参考） | ⚠ 分册名随版本变化 |
| Profiling 工具（msprof / MindStudio Insight） | CANN 文档 → 性能分析工具指南；MindStudio 文档 | ⚠ 同上 |
| ACL 应用开发接口 | CANN 文档 → 应用开发（C/C++）相关分册 | ⚠ 同上 |
| 图编译（GE）与自动融合 | CANN 文档 → 图编译引擎相关分册；MindSpore / torchair 文档 | ⚠ **自动融合规则清单官方通常不完整公开**，图模式下的融合行为需用图 dump / 开关对比实验实证，不能凭文档假设 |
| torch_npu 接口（含融合算子） | GitCode `Ascend/op-plugin` 仓库接口文档（本 Skill `references/torch_npu/` 已内置 2026-10-10 快照，见 §5）；`Ascend/pytorch` 仓库（版本配套表 COMPATIBILITY.md） | 接口参数以此为准；本 Skill 的 `npu-fused-op-catalog.md` 是其融合算子的整理清单，表内链接为官方 commit 固定 URL |
| CANN 开源算子 | Gitee 昇腾组织下开源算子仓库（如 cann-ops 系列） | ⚠ 仓库名与收录范围以组织页为准 |
| MindIE（推理引擎） | 昇腾文档中心 → MindIE | 大模型推理服务化方案文档 |
| 硬件规格（峰值算力 / 带宽 / UB 容量） | 昇腾官网产品页与数据手册（Atlas 800I A2、Atlas 300I Duo 等） | **性能估算引用峰值时以此为准，不要凭记忆填数** |
| NPU 实时状态 | 终端 `npu-smi info` | 环境实证，优先于文档描述 |

## 3. 环境自查命令

版本兼容性核实的第一步是搞清楚目标环境到底装了什么（路径以实际安装为准）：

```bash
npu-smi info                                                 # NPU 型号、驱动/固件版本、健康状态
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg     # CANN 版本（常见安装路径，可能不同）
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
pip list | grep -Ei "vllm|transformers|mindie|atb|accelerate"   # 推理框架与加速库版本
```

## 4. 版本兼容性核实方法

判定"某算子 / 接口在目标环境可用"时，按优先级顺序做：

1. **环境实证**：在目标环境直接查（§3 命令），或跑一个最小调用用例验证。
2. **官方支持矩阵**：文档中心选择目标 CANN / torch_npu 版本后，查该算子/接口文档页的支持芯片型号、dtype、shape 约束、是否标注 beta 或废弃。
3. **Release Notes**：版本间行为变化（新增、废弃、性能变化）在对应版本 Release Notes 里查。
4. **源码**：Gitee / GitHub 仓库按 tag / branch 查接口是否存在、参数签名是否变化。
5. **仍无法确定** → 报告中标注 ⚠ 需核实，写明"查什么 / 问谁才能确认"。**不要在没有证据时下兼容性结论。**

> **TorchNPU 版本命名注意**：产品版本（`26.x`，与 op-plugin 分支同名）和 pip 安装包版本（`2.x`，跟随 PyTorch）是两套编号——用户报告的 `torch_npu.__version__` 是**安装包版本**。两者映射查官方[版本配套表](https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)（例：安装包 `2.10.0.post4` ↔ 产品版本 `26.1.0` ↔ PyTorch 2.10.0 ↔ CANN 9.1.0）。核对接口文档的版本支持时，按产品版本对齐 op-plugin 对应分支。

## 5. 本 Skill 自带资料

| 文件 | 性质 | 使用注意 |
|---|---|---|
| `references/npu-fused-op-catalog.md` | torch_npu 公开融合算子清单（77 个，按 Attention / Norm / 通信计算 / MoE / 矩阵乘量化激活 / 损失 / 其他 七类整理），表内链接为官方 commit 固定 URL | 快照对应 op-plugin master `b80a2fdf`（2026-10-10，TorchNPU 26.2.0 发布时点）；内容随版本演进，**以官方接口文档为最终依据**；（beta）接口可能变动 |
| `references/torch_npu/` | op-plugin 接口文档完整快照：199 个 `.md` 文档（含接口总表与各接口的"产品支持情况 / 函数原型 / 参数约束"），另有 `figures/` 目录 27 张被引用图片 | 同上快照；离线分析优先读本地副本，按 `torch_npu-<接口名>.md` 索引 |
