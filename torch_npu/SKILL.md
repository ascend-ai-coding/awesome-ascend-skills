---
name: torch_npu
description: 华为昇腾 Ascend Extension for PyTorch (torch_npu) 的环境检查、部署与能力指引。在用户使用 @torch_npu、昇腾 NPU、CANN、或需要将 PyTorch 迁移到 NPU 时自动应用；当用户使用 @torch_npu_doc 时，基于本 skill 的 reference 文档提供项目内中文文档能力说明。
---

# torch_npu 能力与使用指引

## 何时使用本 Skill

- 用户 **@torch_npu**、提到昇腾 NPU、CANN、Ascend、或 PyTorch 在 NPU 上运行。
- 用户需要 **环境检查/部署**：检查或部署 PyTorch、检查环境是否支持 NPU。
- 用户使用 **@torch_npu_doc** 时：基于本 skill 的 [reference.md](references/reference.md) 提供项目内中文文档能力说明与操作步骤。
- 用户需要 **NPU 格式转换的代码提示**：书写或补全 `torch_npu.npu_format_cast`、`npu_format_cast_`、`torch_npu.Format`、`get_npu_format` 时，按 §2.1 子能力提供参数与枚举提示。
- 用户需要 **API 支持查询**：查询特定算子在不同 torch-npu 版本中的支持情况。
- 用户需要 **精度验证**：构造单算子验证代码，对比 NPU 与 CPU 的输出精度。

---

## 1. 环境检查与部署

### 1.1 自动检查 PyTorch 与 NPU 环境

在回答或生成脚本时，按需执行或建议用户执行以下检查：

**检查 PyTorch 与 Python 版本是否在配套范围内**（参见 README.zh.md 中的「PyTorch与Python版本配套表」）：

- 支持 PyTorch 1.11.0～2.9.0 等多版本，对应 Python 3.7～3.11（视具体 PyTorch 版本而定）。

**检查环境是否支持 NPU：**

```python
import torch
import torch_npu  # 2.5.1 及以后可不显式 import，仍建议写以便兼容

# 是否可用 NPU、设备数量
if torch.npu.is_available():
    count = torch.npu.device_count()
    # 使用 device='npu' 或 .npu()
else:
    # 未安装 CANN / 未 source set_env.sh / 无 NPU 设备
    pass
```

**检查 CANN 环境变量（安装后验证）：**

- 使用前需执行：`source /usr/local/Ascend/ascend-toolkit/set_env.sh`（路径以实际 CANN 安装为准）。
- 若 `ASCEND_HOME_PATH`、`ASCEND_OPP_PATH` 未设置或路径不存在，torch_npu 会报错并提示执行 `source set_env.sh`。

### 1.2 部署步骤摘要

**详细安装指引**：参见 [installation_guide.md](references/installation/installation_guide.md)
**版本配套表**：参见 [version_compatibility.md](references/installation/version_compatibility.md)

快速安装步骤：
1. **安装 CANN**：按 [CANN 安装指南](https://www.hiascend.com/cann) 安装。
2. **安装 PyTorch**：x86 用 `pip3 install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu`，aarch64 用 `pip3 install torch==2.7.1`。
3. **安装依赖**：`pip3 install pyyaml setuptools`。
4. **安装 torch_npu**：`pip3 install torch-npu==2.7.1`（版本需配套）。
5. **验证**: `pip show torch_npu` 确认安装成功。
---

## 2. torch_npu 能力目录（简略）

| 类别 | 能力说明 |
|------|----------|
| **设备与内存** | `torch.npu`：设备管理、`device_count`、`current_device`、`set_device`、`synchronize`、`Stream`/`Event`、内存统计与分配（`memory_allocated`、`empty_cache`、`MemPool` 等）。 |
| **张量/存储** | `tensor.npu()`、`tensor.is_npu`、NPU Storage、序列化 `torch.save`/`load` 支持 NPU，DDP/多进程 reductions。 |
| **训练/优化** | `torch.npu.amp` 混合精度、`torch_npu.optim`、FSDP 补丁（`ShardedGradScaler`）、梯度检查点默认 NPU。 |
| **分布式** | `torch_npu.distributed`：HCCL/LCCL 后端、`is_hccl_available`、`reinit_process_group`、RPC、symmetric memory、DTensor 规则。 |
| **扩展 API** | `torch_npu.contrib`：NMS、IoU 系列、ROIAlign、DCN、FusedAttention、自定义模块（如 `DropoutWithByteMask`）等。 |
| **图与编译** | NPU Graph（`npugraphify`）、Dynamo、Inductor、torch.compile 支持 NPU。 |
| **推理/ONNX** | ONNX 导出与 NPU 定制算子封装（如 OneHot、RoiAlign、NMS、FastGelu、MultiHeadAttention 等）。 |
| ** profiling** | `torch_npu.profiler`、MSTX 补丁、性能 dump。 |
| **其他** | HiFloat8Tensor、erase_stream、matmul_checksum、transfer_to_npu（可选）、op_plugin。 |

详细 API 以 [昇腾 Ascend Extension for PyTorch 自定义 API 参考](https://www.hiascend.com/document/detail/zh/Pytorch/720/apiref/torchnpuCustomsapi/context/%E6%A6%82%E8%BF%B0.md) 及项目 `README.zh.md` 为准。

---

## 2.1 子能力：torch_npu.npu_format_cast 代码提示

当用户在代码中书写或询问 `torch_npu.npu_format_cast`、`npu_format_cast_`、NPU 张量格式转换时，提供代码提示与补全指引。

详细 API 签名、Format 枚举和代码提示规则参见：[npu_format_cast.md](references/api_support/npu_format_cast.md)

## 3. 参考资源

- **安装指引**：[installation_guide.md](references/installation/installation_guide.md)
- **版本配套**：[version_compatibility.md](references/installation/version_compatibility.md)
- **Flash Attention**：[flash_attention_guide.md](references/api_support/flash_attention_guide.md)
- **API 索引**：[api_index.md](references/api_support/api_index.md)
- **官方文档**：[昇腾社区](https://www.hiascend.com/software/ai-frameworks?framework=pytorch)
- **GitCode 仓库**：https://gitcode.com/Ascend/pytorch


