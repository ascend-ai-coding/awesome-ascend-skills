# Inference

面向推理、模型转换、量化、服务部署与评测的开发入口目录。

当前 `skills/inference/` 已承载真实 skill 目录；请按下面链接进入对应 skill 开发：

- [`ascend-migration-analysis/`](ascend-migration-analysis/)：PyTorch 项目 Ascend NPU 迁移可行性分析
- [`atc-model-converter/`](atc-model-converter/)：ATC 模型转换与 OM 推理
- [`vllm-ascend/`](vllm-ascend/)：vLLM-Ascend 推理引擎与部署指南
- [`vllm-ascend-server/`](vllm-ascend-server/)：vLLM 推理服务启动、配置与健康检查
- [`vllm-bench-serve/`](vllm-bench-serve/)：vLLM 在线性能压测与自动寻优
- [`msmodelslim/msmodelslim-quant/`](msmodelslim/msmodelslim-quant/)：msmodelslim 已验证模型量化流程
- [`ais-bench/`](ais-bench/)：精度与性能评测
- [`diffusers-ascend/`](diffusers-ascend/)：Diffusers 环境、权重与推理
- [`wan-ascend-adaptation/`](wan-ascend-adaptation/)：Wan 系列模型昇腾适配
- [`migration-ascend-torchnpu-skills`](migration-ascend-torchnpu-skills/)：小模型推理基于torch_npu迁移至昇腾平台跑通，包含：环境搭建、迁移、报告生成
- [`migration-ascend-torchair-accelerate-skills`](migration-ascend-torchair-accelerate-skills/)：小模型推理基于torch_npu迁移后，进行以torchair为主（会同时考虑CPU、接口替换等）的性能优化，并生成完整报告：精度、性能比对结果以及复现方法
- [`npu-torchair-infer/`](npu-torchair-infer/)：HuggingFace 模型迁移到昇腾 NPU torchair 图模式并做精度/性能对比

推荐场景：

- 模型转换与部署
- 量化压缩
- 推理服务启动与在线性能评估

对应 bundle：`ascend-inference`
