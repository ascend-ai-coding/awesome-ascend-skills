# vLLM 用例分析（第 31-61 个）

更新时间：2026-03-27

## 1. 分析范围

本报告覆盖 [TEST_FILES_NEED_ANALYSIS.md](TEST_FILES_NEED_ANALYSIS.md) 中第 31 到 61 个测试文件（从 `tests/kernels/moe/test_moe_permute_unpermute.py` 开始共 31 个）。

## 2. 环境与上下文（按 copilot-instructions 要求）

- vllm
  - branch: `v0150rc1`
  - commit: `c511b094b1040572c9c4f03a1e1f98f8dcb60cc1`
- vllm-ascend
  - branch: `v0.15.0rc1`
  - commit: `3d43ed997e715981321bffc10f72f11d9522a5ae`
- Python: `3.11.14`
- torch: `2.9.0+cpu`
- torch-npu: `2.9.0`
- CANN: `8.5.0`（`/usr/local/Ascend/cann-8.5.0`）
- 环境状态：非全新环境（已有 HF/ModelScope 缓存，并在本轮新增部分缓存）
- 外网：默认不稳定/不可达，使用 SOCKS 代理可访问

本轮最小化 mitigation：
- 代理：`http_proxy/https_proxy/all_proxy=socks5h://127.0.0.1:1080`
- 模型源：`VLLM_USE_MODELSCOPE=True`
- `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
- 使用空闲卡：`ASCEND_RT_VISIBLE_DEVICES=6,7`
- `PYTHONPATH` 采用“追加”而非覆盖（`export PYTHONPATH=/home/c00818886/vllm:$PYTHONPATH`），避免丢失 Ascend toolkit python 路径
- 当 ModelScope 缺仓库时，回退 Hugging Face 继续推进 root cause

本轮新增安装依赖：
- `runai-model-streamer[s3,gcs]>=0.15.3`

本轮新增缓存/下载（用于推进 root cause）：
- `charent/self_cognition_Alice`（LoRA）
- `openai-community/gpt2`（权重）
- `Qwen/Qwen3-0.6B`（ModelScope）
- `BAAI/bge-base-en`（含 onnx/model.onnx）

## 3. 逐用例根因与 CI 准入建议

说明：
- `动态` = 本轮有实际运行并观察到首个稳定失败。
- `静态` = 结合测试代码结构、标记、模型规模与已知平台边界给出的结论。
- `CI建议` 取值：`presubmit` / `nightly` / `manual` / `reject`。

| # | 测试文件 | 证据 | 首个稳定失败/阻塞点（真实根因） | 归因 | 是否值得纳入 vllm-ascend 覆盖 | CI建议 | 能否主要靠 vllm-ascend 代码修复通过 |
|---|---|---|---|---|---|---|---|
| 31 | tests/kernels/moe/test_moe_permute_unpermute.py | 动态 | `torch.ops._moe_C.moe_permute_unpermute_supported` 缺失；属于 CUDA/MoE 内核扩展符号依赖 | test not applicable on Ascend platform | 否，原样价值低 | reject | 否（需 Ascend 等效 kernel UT，非原样迁移） |
| 32 | tests/kernels/quantization/test_machete_mm.py | 动态 | collection 阶段 `current_platform.get_device_capability()[0]` 对 NPU 返回 `None` 后下标失败；测试假设 GPU capability 语义 | upstream test assumption / not Ascend-applicable | 否，原样不适配 | reject | 否（需上游测试改平台分支） |
| 33 | tests/lora/test_add_lora.py | 动态 | 已越过前置条件，真实失败落在 LoRA 激活：`vllm/lora/layers/column_parallel_linear.py::set_lora` 触发 `IndexError: tuple index out of range` | vllm-ascend adaptation behavior gap | 有价值（LoRA 预热契约） | nightly | 是 |
| 34 | tests/lora/test_chatglm3_tp.py | 动态 | 已越过多卡/模型前置，真实失败同样落在 LoRA 激活：`column_parallel_linear.py::set_lora` 触发 `IndexError: tuple index out of range` | vllm-ascend adaptation behavior gap | 有价值（LoRA+TP） | nightly | 是 |
| 35 | tests/lora/test_default_mm_loras.py | 动态 | 并非只卡导入下载；进入运行后在多模态 LoRA 激活阶段稳定触发 `IndexError: tuple index out of range` | vllm-ascend adaptation behavior gap | 价值中等（MM LoRA 路由） | nightly | 是 |
| 36 | tests/lora/test_gptoss_tp.py | 静态 + 历史一致 | 模型 `openai/gpt-oss-20b`，涉及 mxfp4/marlin 路径；NPU 对该量化路径历史上不支持 | runtime feature gap | 有价值但成本高 | manual | 部分（需量化支持或上游条件分支） |
| 37 | tests/lora/test_mixtral.py | 动态 | 已推进到真实 LoRA 模块注册失败：`Module model.layers.0.block_sparse_moe.gate must be a BaseLayerWithLoRA instance, got <class 'vllm_ascend.ops.linear.AscendReplicatedLinear'>` | vllm-ascend adaptation behavior gap | 有价值但不适合高频 | nightly | 是 |
| 38 | tests/lora/test_olmoe_tp.py | 动态 | 已推进到真实 LoRA 模块注册失败：`model.layers.0.mlp.gate` 不是 `BaseLayerWithLoRA`，而是 `AscendReplicatedLinear` | vllm-ascend adaptation behavior gap | 有价值（LoRA 边界） | nightly | 是 |
| 39 | tests/lora/test_transformers_model.py | 动态 | 真实失败在编译/ACL 图阶段：`torch._dynamo.exc.InternalTorchDynamoError: AttributeError: 'IlamaConfig' object has no attribute 'decoder'` | compiler or runtime compatibility problem | 有价值但不稳定 | nightly | 部分（需 vllm-ascend / runtime 共同修复） |
| 40 | tests/lora/test_worker.py | 动态 | 已绕过网络后，失败在 `DeviceConfig("cuda")` -> `gpu_worker` 中 `torch.cuda.device_count()==0` 断言 | upstream test hardcoded CUDA | 否，原样不可用 | reject | 否（需上游测试改为平台无关 worker 路径） |
| 41 | tests/model_executor/model_loader/runai_streamer_loader/test_runai_model_streamer_loader.py | 动态 | 补齐 `runai-model-streamer[s3,gcs]` 后，当前首个稳定失败已推进到真实启动门槛：`ValueError: Free memory on device ... is less than desired GPU memory utilization` | runtime resource precondition | 值得（model loader + worker 装配边界） | nightly | 部分（需空闲卡后继续验证更深 loader 语义） |
| 42 | tests/model_executor/model_loader/runai_streamer_loader/test_runai_utils.py | 动态 | 安装 `runai-model-streamer[s3,gcs]>=0.15.3` 后通过（`3 passed`） | test precondition missing（已消除） | 有价值一般（Run:ai 专项） | manual | 否（非代码缺陷，主要是可选依赖） |
| 43 | tests/model_executor/model_loader/tensorizer_loader/test_tensorizer.py | 静态 | 文件大量 `torch.cuda`/多 GPU skipif/tensorizer 外部依赖；偏 CUDA/tensorizer 专项 | not Ascend-first contract | 低（原样） | reject | 否（应挑选平台无关子集重写） |
| 44 | tests/model_executor/test_enabled_custom_ops.py | 动态 | 在 `compilation_mode=1 + backend=inductor` 期望 `CustomOp.default_on=False`，实测为 `True`（平台编译语义与 upstream 期望不一致） | vllm-ascend adaptation behavior gap | 高价值（P0 适配边界） | presubmit | 是 |
| 45 | tests/models/language/generation_ppl_test/test_qwen.py | 动态 | 首个稳定失败不是评测数据，而是 `ModelConfig` 校验：`fp8 quantization is currently not supported in npu` | runtime feature gap | 中等，信噪比不高 | manual | 否（需量化支持） |
| 46 | tests/models/language/pooling/test_all_pooling_plus_chunked_prefill.py | 动态 | 已越过启动前置，失败推进到输出断言层：`tests/models/utils.py` 中模型输出与期望不一致（`AssertionError: Test10`） | behavior contract mismatch | 高价值 | nightly | 是 |
| 47 | tests/models/language/pooling/test_auto_prefix_cache_support.py | 动态 | 已越过启动前置，失败推进到契约断言层：prefix cache 自动开关行为与预期不符（`assert False` / `AssertionError: Test0`） | behavior contract mismatch | 高价值 | nightly | 是 |
| 48 | tests/models/language/pooling/test_classification.py | 动态 | 真实失败是 NPU 算子错误：`current working operator name is ReshapeCacheOperation` | vllm-ascend adaptation behavior gap | 值得 | nightly | 是 |
| 49 | tests/models/language/pooling/test_extract_hidden_states.py | 动态 | 在空闲卡（6/7）重跑可完成引擎初始化与推理，失败推进到断言层：`assert output.num_cached_tokens > 0` 实际为 `0` | behavior contract mismatch (prefix-cache semantics) | 值得（池化隐藏态契约） | nightly | 是（需定位 Ascend 路径的 cache 统计/复用语义） |
| 50 | tests/models/language/pooling/test_gritlm.py | 动态 | 并非只受大模型前置影响；已推进到生成结果断言，模型输出文本与期望答案严重偏离 | behavior contract mismatch | 价值有但成本高 | manual | 是 |
| 51 | tests/models/language/pooling/test_nomic_max_model_len.py | 动态 | 真实失败在 ACL 图捕获/编译阶段：`torch._dynamo.exc.Unsupported: Observed exception`，其开发者上下文显示底层抛出了 `NotImplementedError` | compiler or runtime compatibility problem | 值得（轻于精度类） | nightly | 部分 |
| 52 | tests/models/language/pooling/test_reward.py | 静态 | Qwen2.5-Math-PRM-7B 奖励模型 + golden 对比，模型重 | heavy model | 有价值但成本高 | manual | 部分 |
| 53 | tests/models/language/pooling/test_token_classification.py | 动态 | 真实失败是 NPU 算子错误：`current working operator name is ReshapeCacheOperation` | vllm-ascend adaptation behavior gap | 值得 | nightly | 是 |
| 54 | tests/models/language/pooling_mteb_test/test_baai.py | 动态 | 已不只是数据源问题：部分 case 真实失败为 `ReshapeCacheOperation`，另有 case 进一步推进到 `mteb_utils.py:301` 断言失败 | mixed runtime gap + eval assertion mismatch | 价值中等（更适合夜间） | manual | 部分 |
| 55 | tests/models/language/pooling_mteb_test/test_bge_reranker_v2_gemma.py | 动态 | 已推进到 MTEB 评测栈内部：`TypeError: 'NoneType' object is not subscriptable` | external eval stack incompatibility | 一般 | manual | 否 |
| 56 | tests/models/language/pooling_mteb_test/test_cross_encoder.py | 动态 | 已推进到 MTEB 评测栈内部：`TypeError: 'NoneType' object is not subscriptable` | external eval stack incompatibility | 一般 | manual | 否 |
| 57 | tests/models/language/pooling_mteb_test/test_gte.py | 动态 | 混合失败：部分 case 为 `ReshapeCacheOperation`，另有 case 推进到 `mteb_utils.py:301` 断言失败 | mixed runtime gap + eval assertion mismatch | 一般 | manual | 部分 |
| 58 | tests/models/language/pooling_mteb_test/test_jina.py | 动态 | 真实失败是 NPU 算子错误：`current working operator name is RopeOperation` | vllm-ascend adaptation behavior gap | 一般 | manual | 是 |
| 59 | tests/models/language/pooling_mteb_test/test_nomic.py | 动态 | 真实失败在编译/ACL 图阶段：`torch._dynamo.exc.Unsupported: Observed exception`，开发者上下文显示底层 `NotImplementedError` | compiler or runtime compatibility problem | 一般 | manual | 部分 |
| 60 | tests/models/language/pooling_mteb_test/test_qwen3_reranker.py | 动态 | 已推进到 MTEB 评测栈内部：`TypeError: 'NoneType' object is not subscriptable` | external eval stack incompatibility | 一般 | manual | 否 |
| 61 | tests/models/language/pooling_mteb_test/test_snowflake_arctic_embed.py | 动态 | 补跑后确认当前首个阻塞是 Hugging Face `tokenizer_config.json` 获取阶段经 SOCKS 代理长时间无响应，最终被超时/`KeyboardInterrupt` 打断；尚未推进到产品代码层 | network / external resource precondition still blocking | 一般 | manual | 否（当前需要稳定镜像或预缓存后再继续） |

## 4. 文件存在性检查

- 本批次 31 个目标文件均存在。
- 不存在“任务中列出但分支中不存在”的文件。

## 5. 结论（面向 vllm-ascend CI portfolio）

### 5.1 推荐优先纳入（高信号）

1. `tests/model_executor/test_enabled_custom_ops.py`（P0，建议 presubmit）
2. `tests/model_executor/model_loader/runai_streamer_loader/test_runai_model_streamer_loader.py`（P1，建议 nightly，当前主要受模型源兼容与运行时稳定性影响）
3. pooling 契约中相对可控的子集（建议 nightly）：
   - `tests/models/language/pooling/test_all_pooling_plus_chunked_prefill.py`
   - `tests/models/language/pooling/test_auto_prefix_cache_support.py`
   - `tests/models/language/pooling/test_classification.py`
   - `tests/models/language/pooling/test_extract_hidden_states.py`
   - `tests/models/language/pooling/test_nomic_max_model_len.py`
   - `tests/models/language/pooling/test_token_classification.py`

### 5.2 建议 manual / 不纳入 presubmit

- LoRA 大模型/多卡重资源组：`tests/lora/*`（除非做 Ascend-adapted 小模型子集）
- MTEB 全组：`tests/models/language/pooling_mteb_test/*`
- 明显 CUDA/内核专项：
  - `tests/kernels/moe/test_moe_permute_unpermute.py`
  - `tests/kernels/quantization/test_machete_mm.py`
  - `tests/model_executor/model_loader/tensorizer_loader/test_tensorizer.py`
  - `tests/lora/test_worker.py`（硬编码 `DeviceConfig("cuda")`）

### 5.3 当前批次最关键可行动修复项

1. **为 ModelScope/HF 数据源建立可重复策略**
   - 症状：Run:ai 与 MTEB 用例在空闲卡上已越过显存门槛后，暴露出 `model.safetensors.index.json`/dataset README 缺失等源兼容问题。
   - 影响面：#41、#54。
2. **修复 pooling 隐藏态用例中的缓存契约差异**
   - 症状：`test_extract_hidden_states` 在 Ascend 路径断言 `num_cached_tokens > 0` 失败。
   - 影响面：#49。
3. **为 LoRA 与 pooling 目录建立 Ascend-adapted 子集**
   - 避免硬编码 CUDA、避免模块导入即大模型下载、控制模型规模与并发。
4. **将 MTEB 类测试下沉 nightly/manual**
   - 保留覆盖价值，但避免 presubmit 引入高波动外部依赖。
