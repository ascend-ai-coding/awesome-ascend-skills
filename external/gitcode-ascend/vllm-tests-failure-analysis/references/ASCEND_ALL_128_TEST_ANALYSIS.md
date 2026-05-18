# vLLM 1-128 用例 Ascend 总表

更新时间：2026-03-27

## 说明

- 本表汇总以下三份分段报告：
  - `ASCEND_FIRST_30_TEST_ANALYSIS.md`
  - `ASCEND_31_61_TEST_ANALYSIS.md`
  - `ASCEND_62_128_TEST_ANALYSIS.md`
- 新增列“**昇腾应跑通**”，用于回答：**该用例按上游行为契约/平台适用性判断，是否应该在昇腾环境跑通**。

### “昇腾应跑通”判定口径

- `是`：平台无关，或明确属于 vLLM-Ascend 应保留的 upstream 行为契约。
- `有条件`：原则上应跑通，但前提是满足合理测试前置（空闲卡、预缓存、测试依赖、或修复明显的 vLLM-Ascend 适配缺口）。
- `否`：测试原样依赖 CUDA/ROCm/Triton/NVML，或验证 Ascend 当前明确不支持/无 CI 守护价值的路径。

### “CI 结论”取值

- `presubmit`：适合高频 CI。
- `nightly`：有价值，但更适合夜测。
- `manual`：可保留人工/专项回归，不建议常规 CI。
- `reject`：不建议纳入 Ascend CI 组合。

## 总表

| # | 测试文件 | 当前结论 | 昇腾应跑通 | CI结论 | 失败根因 / 说明 |
|---|---|---|---|---|---|
| 1 | `tests/compile/test_pass_manager.py` | 失败 | 是 | manual | `RMSNormQuantFusionPass` 等编译 pass 配置在非 CUDA 平台失配，且默认模型解析引入无谓联网。 |
| 2 | `tests/compile/test_wrapper.py` | 失败 | 有条件 | presubmit | Ascend 编译配置与 upstream `torch.compile/inductor` 测试假设不兼容，需 Ascend-adapted 语义。 |
| 3 | `tests/distributed/test_distributed_oot.py` | 失败 | 有条件 | nightly | OOT dummy 架构注册/识别链未稳定保留 upstream 行为。 |
| 4 | `tests/distributed/test_sequence_parallel.py` | 失败 | 有条件 | nightly | 代理补齐模型后暴露真实阻塞：空闲显存低于 `gpu_memory_utilization=0.9` 所需阈值。 |
| 5 | `tests/entrypoints/llm/test_mm_cache_stats.py` | 可通过 | 有条件 | nightly | 配好 `no_proxy` 并预缓存后可通过；先前问题是 localhost 请求误走代理。 |
| 6 | `tests/entrypoints/openai/correctness/test_transcription_api_correctness.py` | 未完成业务断言 | 否 | manual | 受 Whisper 大模型/数据准备成本阻塞，外部资源依赖过重。 |
| 7 | `tests/entrypoints/openai/test_chat.py` | 未完成有效断言 | 有条件 | nightly | LoRA + 基模资源准备链过重，需预缓存和拆小子集。 |
| 8 | `tests/entrypoints/openai/test_default_mm_loras.py` | 失败 | 有条件 | manual | 模块导入即触发超大模型与 LoRA 下载，前置成本过高。 |
| 9 | `tests/entrypoints/openai/test_oot_registration.py` | 失败 | 有条件 | nightly | OOT 注册后 server 能力链不稳定，属于平台/插件边界。 |
| 10 | `tests/entrypoints/openai/test_openai_schema.py` | 不建议运行 | 否 | reject | `schemathesis + server + model` 组合噪声高，不适合作为 Ascend 高频守护。 |
| 11 | `tests/entrypoints/openai/test_sparse_tensor_validation.py` | 失败 | 是 | presubmit | `torch.load(weights_only=True)` 触发 `UnpicklingError`，测试异常类型假设过窄。 |
| 12 | `tests/entrypoints/openai/test_tensorizer_entrypoint.py` | 不适用 | 否 | manual | 偏 CUDA/tensorizer 专项测试，原样不适合作为 Ascend 用例。 |
| 13 | `tests/entrypoints/openai/test_transcription_validation_whisper.py` | 未完成业务断言 | 否 | manual | 主要被 Whisper 大模型下载/准备链阻塞。 |
| 14 | `tests/entrypoints/openai/test_video.py` | 未完成业务断言 | 否 | manual | 视频模型前置资源链重，不适合高频 CI。 |
| 15 | `tests/entrypoints/openai/test_vision.py` | 未完成业务断言 | 有条件 | nightly | 真实阻塞收敛为大模型准备成本与缓存锁竞争，预缓存后才有意义。 |
| 16 | `tests/entrypoints/openai/tool_parsers/test_openai_tool_parser.py` | 失败 | 否 | manual | `openai/gpt-oss-20b` 默认 `mxfp4` 量化在 NPU 上不支持。 |
| 17 | `tests/entrypoints/pooling/classify/test_online.py` | 可通过 | 有条件 | nightly | 切空闲卡并回退 HF 后最小 case 已通过。 |
| 18 | `tests/entrypoints/pooling/classify/test_online_vision.py` | 未完成业务断言 | 否 | manual | 共享 server fixture 绑定 7B 多模态分类模型，资源准备过重。 |
| 19 | `tests/entrypoints/pooling/score/test_correctness_mteb.py` | 失败 | 否 | nightly | 已推进到 MTEB 栈内部 `NoneType` 错误，主要受外部评测框架/数据依赖影响。 |
| 20 | `tests/entrypoints/sagemaker/test_sagemaker_lora_adapters.py` | 通过 | 是 | presubmit | Ascend 保留了 LoRA adapter API 语义，是高价值边界守护。 |
| 21 | `tests/entrypoints/sagemaker/test_sagemaker_stateful_sessions.py` | 通过 | 有条件 | presubmit | 补齐 `socksio` 并切空闲卡后整文件通过。 |
| 22 | `tests/evals/gpt_oss/test_gpqa_correctness.py` | 不建议运行 | 否 | reject | 必须显式传 `--model/--metric`，不是普通可直接执行 UT。 |
| 23 | `tests/kernels/attention/test_attention_selector.py` | 不适用 | 否 | reject | backend 选择逻辑偏 CUDA/ROCm，不是 Ascend 合适入口。 |
| 24 | `tests/kernels/attention/test_flashmla.py` | 不适用 | 否 | reject | 明确 CUDA 专用 kernel 测试。 |
| 25 | `tests/kernels/attention/test_flashmla_sparse.py` | skip | 否 | reject | 平台检测正常跳过，Ascend 不适用。 |
| 26 | `tests/kernels/attention/test_mha_attn.py` | 不适用 | 否 | reject | 数值/后端路径偏 CUDA，不适合作为 Ascend 原样 CI。 |
| 27 | `tests/kernels/core/test_activation.py` | 通过/可执行 | 是 | presubmit | 核心激活逻辑可作为平台无关守护。 |
| 28 | `tests/kernels/moe/test_gpt_oss_triton_kernels.py` | 不适用 | 否 | reject | Triton/CUDA 专项。 |
| 29 | `tests/kernels/moe/test_modular_kernel_combinations.py` | 不适用 | 否 | reject | Triton/CUDA kernel 组合测试。 |
| 30 | `tests/kernels/moe/test_modular_oai_triton_moe.py` | 不适用 | 否 | reject | Triton MoE 路径，不适用 Ascend。 |
| 31 | `tests/kernels/moe/test_moe_permute_unpermute.py` | 失败 | 否 | reject | 依赖 `_moe_C` CUDA/MoE 扩展符号。 |
| 32 | `tests/kernels/quantization/test_machete_mm.py` | 失败 | 否 | reject | 测试假设 `get_device_capability()` 返回 GPU 语义，NPU 为 `None`。 |
| 33 | `tests/lora/test_add_lora.py` | 失败 | 是 | nightly | 已推进到 LoRA 激活真实根因：`column_parallel_linear.py::set_lora` 触发 `IndexError: tuple index out of range`。 |
| 34 | `tests/lora/test_chatglm3_tp.py` | 失败 | 是 | nightly | 已推进到 LoRA 激活真实根因：`column_parallel_linear.py::set_lora` 触发 `IndexError: tuple index out of range`。 |
| 35 | `tests/lora/test_default_mm_loras.py` | 失败 | 是 | nightly | 不再只是下载前置，运行后稳定失败于多模态 LoRA 激活：`IndexError: tuple index out of range`。 |
| 36 | `tests/lora/test_gptoss_tp.py` | 未跑通 | 否 | manual | 依赖 `gpt-oss-20b` 量化路径，NPU 当前不支持。 |
| 37 | `tests/lora/test_mixtral.py` | 失败 | 是 | nightly | 真实根因是 LoRA 模块注册断言：`block_sparse_moe.gate` 必须是 `BaseLayerWithLoRA`，但当前为 `AscendReplicatedLinear`。 |
| 38 | `tests/lora/test_olmoe_tp.py` | 失败 | 是 | nightly | 真实根因是 LoRA 模块注册断言：`mlp.gate` 不是 `BaseLayerWithLoRA`，而是 `AscendReplicatedLinear`。 |
| 39 | `tests/lora/test_transformers_model.py` | 失败 | 有条件 | nightly | 真实根因在编译/ACL 图阶段：`torch._dynamo.exc.InternalTorchDynamoError: AttributeError: 'IlamaConfig' object has no attribute 'decoder'`。 |
| 40 | `tests/lora/test_worker.py` | 失败 | 否 | reject | 硬编码 `DeviceConfig("cuda")`，直接落到 `torch.cuda.device_count()==0`。 |
| 41 | `tests/model_executor/model_loader/runai_streamer_loader/test_runai_model_streamer_loader.py` | 失败 | 有条件 | nightly | 补齐 `runai-model-streamer[s3,gcs]` 后，当前首个稳定失败已推进为真实启动门槛：空闲显存不足。 |
| 42 | `tests/model_executor/model_loader/runai_streamer_loader/test_runai_utils.py` | 通过 | 是 | manual | 补齐 `runai-model-streamer` 依赖后通过。 |
| 43 | `tests/model_executor/model_loader/tensorizer_loader/test_tensorizer.py` | 不适用 | 否 | reject | 大量 `torch.cuda`/多 GPU/tensorizer 前提。 |
| 44 | `tests/model_executor/test_enabled_custom_ops.py` | 失败 | 是 | presubmit | `CustomOp.default_on` 与 upstream 编译语义期望不一致。 |
| 45 | `tests/models/language/generation_ppl_test/test_qwen.py` | 失败 | 否 | manual | 首个稳定失败是 `fp8 quantization is currently not supported in npu`。 |
| 46 | `tests/models/language/pooling/test_all_pooling_plus_chunked_prefill.py` | 失败 | 是 | nightly | 已推进到输出断言层，chunked prefill/prefix cache 结果与期望不一致（`AssertionError: Test10`）。 |
| 47 | `tests/models/language/pooling/test_auto_prefix_cache_support.py` | 失败 | 是 | nightly | 已推进到契约断言层，prefix cache 自动开关行为与预期不一致。 |
| 48 | `tests/models/language/pooling/test_classification.py` | 未跑通 | 是 | nightly | 分类 logits/HF 对齐应在 Ascend 保持。 |
| 49 | `tests/models/language/pooling/test_extract_hidden_states.py` | 失败 | 是 | nightly | 已推进到断言层，`num_cached_tokens` 为 `0`，属于缓存语义不一致。 |
| 50 | `tests/models/language/pooling/test_gritlm.py` | 失败 | 是 | manual | 已推进到业务断言层，生成答案与期望文本严重偏离。 |
| 51 | `tests/models/language/pooling/test_nomic_max_model_len.py` | 失败 | 有条件 | nightly | 真实根因在 ACL 图捕获/编译阶段：`torch._dynamo.exc.Unsupported`，其开发者上下文显示底层 `NotImplementedError`。 |
| 52 | `tests/models/language/pooling/test_reward.py` | 未跑通 | 有条件 | manual | 7B 奖励模型重，对高频 CI 不友好。 |
| 53 | `tests/models/language/pooling/test_token_classification.py` | 未跑通 | 是 | nightly | token classification 与 HF 对齐应保留。 |
| 54 | `tests/models/language/pooling_mteb_test/test_baai.py` | 失败 | 有条件 | manual | 已不是单纯数据前置：部分 case 真实失败为 `ReshapeCacheOperation`，另有 case 推进到 MTEB 断言失败。 |
| 55 | `tests/models/language/pooling_mteb_test/test_bge_reranker_v2_gemma.py` | 失败 | 否 | manual | 已推进到 MTEB 评测栈内部：`TypeError: 'NoneType' object is not subscriptable`。 |
| 56 | `tests/models/language/pooling_mteb_test/test_cross_encoder.py` | 失败 | 否 | manual | 已推进到 MTEB 评测栈内部：`TypeError: 'NoneType' object is not subscriptable`。 |
| 57 | `tests/models/language/pooling_mteb_test/test_gte.py` | 失败 | 有条件 | manual | 混合失败：部分 case 为 `ReshapeCacheOperation`，另有 case 推进到 MTEB 断言失败。 |
| 58 | `tests/models/language/pooling_mteb_test/test_jina.py` | 失败 | 是 | manual | 真实根因是 NPU 算子错误：`current working operator name is RopeOperation`。 |
| 59 | `tests/models/language/pooling_mteb_test/test_nomic.py` | 失败 | 有条件 | manual | 真实根因在编译/ACL 图阶段：`torch._dynamo.exc.Unsupported`，底层为 `NotImplementedError`。 |
| 60 | `tests/models/language/pooling_mteb_test/test_qwen3_reranker.py` | 失败 | 否 | manual | 已推进到 MTEB 评测栈内部：`TypeError: 'NoneType' object is not subscriptable`。 |
| 61 | `tests/models/language/pooling_mteb_test/test_snowflake_arctic_embed.py` | 失败（前置仍阻塞） | 否 | manual | 补跑后确认当前首个阻塞是 Hugging Face `tokenizer_config.json` 获取经 SOCKS 代理长时间无响应，最终超时/`KeyboardInterrupt`。 |
| 62 | `tests/models/multimodal/generation/test_audioflamingo3.py` | skip | 否 | reject | 日志为 `2 skipped`，未形成有效 Ascend 覆盖。 |
| 63 | `tests/models/multimodal/generation/test_granite_speech.py` | 失败 | 是 | nightly | `NPUModelRunner` 缺少 `mm_budget`，属于多模态适配缺口。 |
| 64 | `tests/models/multimodal/generation/test_phi4mm.py` | 失败 | 否 | reject | 多模态 LoRA 激活触发 `IndexError: tuple index out of range`，更像上游形状假设问题。 |
| 65 | `tests/models/multimodal/generation/test_qwen2_5_vl.py` | 失败 | 是 | nightly | 真实根因：`aclnnIndexPutImpl failed`。 |
| 66 | `tests/models/multimodal/generation/test_vit_backend_functionality.py` | skip | 否 | reject | `16 skipped`，无有效信号。 |
| 67 | `tests/models/multimodal/generation/test_voxtral.py` | 失败 | 否 | reject | `AudioEncoder.pad()` 调用签名不匹配，偏上游模型/processor 集成问题。 |
| 68 | `tests/models/multimodal/pooling/test_clip.py` | 失败 | 是 | nightly | 真实根因：`ReshapeCacheOperation`。 |
| 69 | `tests/models/multimodal/pooling/test_jinavl_reranker.py` | 部分通过 | 是 | nightly | 3 过 1 失败；失败为打分行为偏差，对 Ascend 一致性有价值。 |
| 70 | `tests/models/multimodal/pooling/test_prithvi_mae.py` | 失败 | 否 | reject | `ModelConfig` 校验失败，当前更像架构支持缺失。 |
| 71 | `tests/models/multimodal/processing/test_audioflamingo3.py` | skip | 否 | reject | `2 skipped`。 |
| 72 | `tests/models/multimodal/processing/test_h2ovl.py` | 未完成有效结论 | 否 | manual | `192` 个重参数化 case，资源/时间成本过高。 |
| 73 | `tests/models/multimodal/processing/test_phi3v.py` | skip | 否 | reject | `12 skipped`。 |
| 74 | `tests/models/quantization/test_awq.py` | 失败 | 否 | reject | AWQ 量化路径在 NPU 上不支持。 |
| 75 | `tests/models/quantization/test_bitblas.py` | skip | 否 | reject | BitBLAS 非 Ascend 重点路径。 |
| 76 | `tests/models/test_gguf_download.py` | 部分通过 | 否 | manual | GGUF/模型支持问题，Ascend CI 收益低。 |
| 77 | `tests/models/test_initialization.py` | 部分通过 | 是 | nightly | 多模态初始化中 `embed_multimodal` 返回 `None`。 |
| 78 | `tests/models/test_oot_registration.py` | 部分通过 | 是 | nightly | 模型注册/架构识别链仍有缺口。 |
| 79 | `tests/models/test_registry.py` | 部分通过 | 是 | nightly | `PrithviGeoSpatialMAE`、`Terratorch`、`MiDashengLMModel` 导入为空。 |
| 80 | `tests/multimodal/test_sparse_tensor_validation_unit.py` | 失败 | 否 | reject | PyTorch `weights_only=True` 变更导致 `UnpicklingError`，偏上游兼容问题。 |
| 81 | `tests/plugins_tests/test_platform_plugins.py` | 失败 | 是 | nightly | 插件加载顺序导致拿到真实 `npu` 而非 `DummyDevice`。 |
| 82 | `tests/plugins_tests/test_stats_logger_plugins.py` | 失败 | 有条件 | presubmit | 缺少测试依赖 `dummy_stat_logger`；补齐后是高信号插件契约测试。 |
| 83 | `tests/quantization/test_compressed_tensors.py` | 失败 | 否 | reject | `No compressed-tensors compatible scheme was found`。 |
| 84 | `tests/quantization/test_configs.py` | 失败 | 是 | presubmit | 量化自动识别返回 `ERROR`，是轻量高信号适配边界。 |
| 85 | `tests/quantization/test_cpu_offload.py` | skip | 否 | reject | `4 skipped`。 |
| 86 | `tests/quantization/test_experts_int8.py` | skip | 否 | reject | `2 skipped`。 |
| 87 | `tests/quantization/test_gptq_dynamic.py` | 失败 | 否 | reject | GPTQ dynamic 在 NPU 不支持。 |
| 88 | `tests/quantization/test_gptq_v2.py` | 失败 | 否 | reject | GPTQ v2 在 NPU 不支持。 |
| 89 | `tests/quantization/test_lm_head.py` | 失败 | 否 | reject | 量化相关 `ModelConfig` 校验失败。 |
| 90 | `tests/quantization/test_modelopt.py` | skip | 否 | reject | `1 skipped`。 |
| 91 | `tests/quantization/test_ptpc_fp8.py` | skip | 否 | reject | `9 skipped`。 |
| 92 | `tests/quantization/test_quark.py` | 失败 | 否 | reject | Quark 量化在 NPU 上不支持。 |
| 93 | `tests/quantization/test_torchao.py` | skip | 否 | reject | `12 skipped`。 |
| 94 | `tests/reasoning/test_seedoss_reasoning_parser.py` | 失败 | 否 | reject | parser 返回 `None`，偏上游逻辑问题，非 Ascend 边界。 |
| 95 | `tests/samplers/test_beam_search.py` | 失败 | 是 | nightly | 真实根因：`aclnnFusedInferAttentionScoreV3 failed`。 |
| 96 | `tests/samplers/test_logprobs.py` | 失败 | 是 | nightly | 真实根因：`aclnnFusedInferAttentionScoreV3 failed`。 |
| 97 | `tests/samplers/test_no_bad_words.py` | 失败 | 是 | nightly | 真实根因：`aclnnFusedInferAttentionScoreV3 failed`。 |
| 98 | `tests/utils_/test_mem_utils.py` | 失败 | 否 | reject | 缺少 `vllm_test_utils`，本身不是高价值 Ascend 守护项。 |
| 99 | `tests/v1/attention/test_mla_backends.py` | 失败 | 否 | reject | `ModelConfig` 校验失败，更像模型/后端支持条件不满足。 |
| 100 | `tests/v1/core/test_priority_scheduler_random.py` | 失败 | 否 | reject | `block_hashes` 断言失败，偏 upstream 核心逻辑问题。 |
| 101 | `tests/v1/core/test_scheduler_e2e.py` | 失败 | 否 | reject | 调度断言 `0 == 16`，偏 upstream 核心逻辑。 |
| 102 | `tests/v1/cudagraph/test_cudagraph_mode.py` | 失败 | 否 | reject | `nvmlDeviceGetHandleByIndex` 未定义，明显依赖 NVML/Nvidia。 |
| 103 | `tests/v1/e2e/test_min_tokens.py` | 失败 | 是 | nightly | 真实根因：`aclnnFusedInferAttentionScoreV3 failed`。 |
| 104 | `tests/v1/engine/test_engine_args.py` | 失败 | 是 | presubmit | `NPUPlatform.get_device_total_memory()` 未实现。 |
| 105 | `tests/v1/entrypoints/openai/test_completion.py` | 失败 | 是 | nightly | OpenAI 500 的真实根因是 `aclnnFusedInferAttentionScoreV3 failed`。 |
| 106 | `tests/v1/entrypoints/openai/test_completion_with_image_embeds.py` | 失败 | 是 | nightly | 真实根因：`aclnnIndexPutImpl failed`。 |
| 107 | `tests/v1/kv_connector/unit/test_cache_pollution_prevention.py` | 失败 | 否 | reject | `IndexError: list index out of range`，偏 connector 单元逻辑。 |
| 108 | `tests/v1/kv_connector/unit/test_config.py` | 失败 | 否 | reject | `NoneType.is_deepseek_mla`，偏通用配置逻辑。 |
| 109 | `tests/v1/kv_connector/unit/test_decode_bench_connector.py` | 失败 | 否 | reject | 断言 `1 == 3`，偏 connector 逻辑。 |
| 110 | `tests/v1/kv_connector/unit/test_error_propagation.py` | 失败 | 否 | reject | `IndexError: list index out of range`。 |
| 111 | `tests/v1/kv_connector/unit/test_example_connector.py` | 失败 | 否 | reject | `AttributeError: 'tuple' object has no attribute 'shape'`。 |
| 112 | `tests/v1/kv_connector/unit/test_invalid_blocks_correctness.py` | 失败 | 否 | reject | `IndexError: list index out of range`。 |
| 113 | `tests/v1/kv_connector/unit/test_kv_load_failure_recovery.py` | 失败 | 否 | reject | `IndexError: list index out of range`。 |
| 114 | `tests/v1/kv_connector/unit/test_offloading_connector.py` | 失败 | 否 | reject | connector 断言失败。 |
| 115 | `tests/v1/kv_connector/unit/test_remote_decode_lifecycle.py` | 失败 | 否 | reject | 断言 `1 == 3`。 |
| 116 | `tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py` | 失败 | 否 | reject | 断言 `1 == 2`。 |
| 117 | `tests/v1/sample/test_rejection_sampler.py` | 失败 | 否 | reject | `TypeError: 'function' object is not subscriptable`，偏上游采样逻辑。 |
| 118 | `tests/v1/spec_decode/test_speculators_eagle3.py` | 失败 | 是 | nightly | `AscendW4A16FusedMoEMethod.get_weight()` 缺少 `params_dtype` 参数。 |
| 119 | `tests/v1/tracing/test_tracing.py` | 失败 | 是 | nightly | 真实根因：`aclnnFusedInferAttentionScoreV3 failed`。 |
| 120 | `tests/weight_loading/test_weight_loading.py` | skip | 否 | reject | `1 skipped`。 |
| 121 | `tests/entrypoints/openai/responses/test_mcp_tools.py` | 失败 | 否 | reject | `mxfp4 quantization is currently not supported in npu`。 |
| 122 | `tests/entrypoints/test_grpc_server.py` | 失败 | 是 | presubmit | 补跑确认：collection 阶段缺少 `vllm_engine_pb2` 导入。 |
| 123 | `tests/kernels/helion/test_helion_available.py` | skip | 否 | reject | 平台不匹配，历史统计为 `1 skipped`。 |
| 124 | `tests/kernels/moe/test_routing_simulator.py` | 通过 | 是 | presubmit | `27 passed`，稳定、轻量、纯逻辑守护。 |
| 125 | `tests/kernels/moe/test_triton_moe_no_act_mul.py` | skip | 否 | reject | Triton/CUDA 导向，历史统计为 `74 skipped`。 |
| 126 | `tests/lora/test_qwenvl.py` | 部分通过 | 有条件 | nightly | 多模态 LoRA/模块映射有价值，但初始化路径仍不稳。 |
| 127 | `tests/models/language/pooling_mteb_test/test_nemotron.py` | 部分通过 | 否 | manual | 重型 MTEB 评测，CI 成本高于收益。 |
| 128 | `tests/v1/engine/test_preprocess_error_handling.py` | 失败（当前环境） | 有条件 | nightly | 补跑确认真实根因是空闲显存不足，而非模型下载失败。 |

## 可直接取用的 CI 子集

### 优先 presubmit

- `tests/kernels/core/test_activation.py`
- `tests/entrypoints/sagemaker/test_sagemaker_lora_adapters.py`
- `tests/entrypoints/sagemaker/test_sagemaker_stateful_sessions.py`
- `tests/model_executor/test_enabled_custom_ops.py`
- `tests/plugins_tests/test_stats_logger_plugins.py`（先补齐 `dummy_stat_logger`）
- `tests/quantization/test_configs.py`
- `tests/v1/engine/test_engine_args.py`
- `tests/entrypoints/test_grpc_server.py`
- `tests/kernels/moe/test_routing_simulator.py`

### 优先 nightly

- OOT / plugin / registry：`test_distributed_oot.py`, `test_oot_registration.py`, `test_platform_plugins.py`, `test_registry.py`, `test_oot_registration.py`
- Pooling / prefix cache / classification：`test_all_pooling_plus_chunked_prefill.py`, `test_auto_prefix_cache_support.py`, `test_classification.py`, `test_extract_hidden_states.py`, `test_nomic_max_model_len.py`, `test_token_classification.py`
- 多模态 / OpenAI / sampler：`test_granite_speech.py`, `test_qwen2_5_vl.py`, `test_clip.py`, `test_jinavl_reranker.py`, `test_completion.py`, `test_completion_with_image_embeds.py`, `test_beam_search.py`, `test_logprobs.py`, `test_no_bad_words.py`, `test_min_tokens.py`, `test_tracing.py`
- LoRA / spec decode：`test_qwenvl.py`, `test_speculators_eagle3.py`, `test_preprocess_error_handling.py`
