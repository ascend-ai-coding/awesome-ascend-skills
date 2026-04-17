# 迁移前分析

> 来源: [昇腾文档 - 迁移前分析](https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/PT_LMTMOG_0004.html)
> 适用版本: Ascend Extension for PyTorch 7.3.0

模型能否成功迁移至昇腾AI处理器，主要取决于其使用的算子是否被昇腾平台支持。为保证迁移可行性，迁移前可使用如下方法进行分析：

## 1. 第三方库支持情况

若模型原始代码中调用了模型套件或第三方库，需要关注NPU对其的支持情况：

- 如果该三方库**原生支持NPU**，用户需要关注NPU目前对库中特性的支持情况；
- 如果是**昇腾适配的第三方库**，用户需要额外安装该库的昇腾适配版本，并关注其适配情况。详细昇腾第三方库支持情况请参考《[套件与三方库支持清单](https://www.hiascend.com/document/detail/zh/Pytorch/730/modthirdparty/modparts/thirdpart_0003.html)》。如果用户希望以上第三方库和模型套件在适配昇腾设备后能达到更高的性能，可以自行调优。

## 2. 已知不支持场景

确认是否存在以下已知的不支持场景：

| 场景/组件 | 支持状态 | 适配说明 | 替代方案 |
| --- | --- | --- | --- |
| DP（Data Parallel，数据并行）模型 | 暂不支持 | 不兼容 `torch.nn.parallel.DataParallel` 接口 | 需手动修改为 `torch.nn.parallel.DistributedDataParallel` 接口，以执行多卡训练。原脚本需要在GPU环境下基于Python3.8及以上跑通 |
| APEX库中的FusedAdam融合优化器 | 部分支持 | 不支持自动迁移或PyTorch GPU2Ascend工具迁移 | 需手工进行迁移，具体修改方法可参考 [昇腾APEX适配](https://gitcode.com/ascend/apex#apexoptimizers) |
| bitsandbytes | 部分支持 | 已支持在昇腾上进行安装，具体可参考 [Supported Backends](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/docs/source/installation.mdx#supported-backendsmulti-backend-supported-backends) | 仅支持NF4量化/反量化迁移，用于LLM QLoRA微调，其余功能暂不支持 |
| xFormers | 暂不支持 | 不原生支持 xFormers | xFormers中的FlashAttentionScore融合算子可参考 FlashAttentionScore 章节进行替换 |
| bmtrain框架 | 暂不支持 | 大模型迁移场景不支持 | 暂无替代方案 |
| colossalai库中HybridAdam优化器 | 暂不支持 | 大模型迁移场景不支持 | 暂无替代方案 |
| grouped_gemm三方库 | 暂不支持 | NPU不支持安装 | 暂无替代方案 |
| composer三方库 | 暂不支持 | NPU支持安装但未适配 | 暂无替代方案 |

## 3. PyTorch Analyse工具分析

借助PyTorch Analyse工具，分析基于GPU平台的PyTorch训练脚本中三方库套件、API、动态shape以及亲和API在昇腾AI处理器上的支持情况。工具使用详细指导可参见《[CANN 分析迁移工具用户指南](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/migrationtools/atlasfmkt_16_0001.html)》。

### 分析模式介绍

| 分析模式 | 分析脚本 | 分析结果 | 调优建议 |
| --- | --- | --- | --- |
| 三方库套件分析模式 | 需用户提供待分析的三方库套件源码 | 可快速获得源码中不支持的三方库API和cuda信息。**说明：** 三方库API是指在三方库代码中的函数，如果某函数的函数体内使用了不支持的torch算子或者cuda自定义算子，则此函数就是三方库不支持的API。如果第三方库中其他函数调用了这些不支持的API，则这些调用函数也为不支持的API。 | - |
| API支持情况分析模式 | 需用户提供待分析的PyTorch训练脚本 | 可快速获得训练脚本中不支持的torch API和cuda API信息 | 输出训练脚本中API精度和性能调优的专家建议 |
| 动态shape分析模式 | 需用户提供待分析的PyTorch训练脚本 | 可快速获得训练脚本中包含的动态shape信息 | - |
| 亲和API分析模式 | 需用户提供待分析的PyTorch训练脚本 | 可快速获得训练脚本中可替换的亲和API信息 | - |

### 不支持算子的适配方法

在迁移可行性分析中如果存在平台未支持的算子，可参考如下方法进行算子适配：

- 修改模型脚本使用等价支持的算子替换；
- 算子开发与适配：
  - Ascend C算子开发请参见《[CANN Ascend C算子开发指南](https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)》
  - TBE&AI CPU算子开发请参见《[CANN TBE&AI CPU算子开发指南](https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/tbeaicpudevg/atlasopdev_10_0001.html)》
  - 算子适配请参见《[PyTorch 框架特性指南](https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/Frameworkfeatures/docs/zh/framework_feature_guide_pytorch/adaptation_description_opplugin.md)》中的"自定义算子适配开发"章节
