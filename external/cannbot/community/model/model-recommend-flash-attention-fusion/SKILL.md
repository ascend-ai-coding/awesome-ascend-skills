---
name: external-cannbot-community-model-model-recommend-flash-attention-fusion
description: 既有 FlashAttention 类算子的功能确认与图融合 pass 接入（条目 7 的 FA 族融合）。对 attention 链（QK^T+Softmax+AV+head
  转置）热点，确认 GE 内置 FA 算子（PromptFlashAttention 等）对场景的支持性，经 GE 图融合 pass（graph_fusion_pass.cc，aclgrphParseONNX
  后改写 ge::Graph）将 ViT 风格 attention 子图替换为 FA 算子，不依赖 onnx 融合算子表达，过改动校验双门禁。触发：attention
  融合、FlashAttention 接入、图融合 pass、GE 图改写、NPUFlashAttention、PromptFlashAttention、npu
  域算子、混合域、opset 限制。
original-name: model-recommend-flash-attention-fusion
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# FlashAttention 融合算子确认与图融合 pass 接入

## 流水位置

[fusion-operator-adaptation](../model-recommend-fusion-operator-adaptation/SKILL.md)（条目 7）的性能替换专项：当条目 5 的 profiling 分析显示 attention 链（QK^T / Softmax / AV MatMul / head Transpose / Reshape）为热点且 GE 无融合 pass 覆盖（fusion_result.json 中 ASoftmaxFusionPass 等 match>0 effect=0）时进入本条目。**开发成本远低于条目 11 自定义 kernel**（复用 GE 内置 FA 算子），优先尝试。

## 目标

1. **功能确认**：判定本机 CANN 的 FA 算子是否支持目标网络场景，产出最小验证模型证据
2. **图融合 pass 接入**：在 GE 图层（parse 后、AddGraph 前）将 attention 子图替换为 FA 算子，过「改动校验双门禁」后落地

## 全局约束：不修改原图（用户指令，2026-10-15 起）

**所有优化不得改动模型文件**——改写仅发生在 GE 内存图（图 pass）；需要折算的标量（如 attention scale）从图内常量提取、经目标算子属性传递（如 FA `scale_value`），不做权重改写。M200 实证：scale 折叠模型改动撤销后，原图 + 双 pass 性能完全等价（6.125 vs 6.123ms），回退路径（开关关闭 = 原始基线行为）完好。常量提取须支持 fp16 位模式与 Identity/Const-value-attr 包装。

## 核心结论（M200 实证，2026-10-13）

**推荐路线：图融合 pass**（在 `GE_SESSION_DEMO/src/` 新增 `graph_fusion_pass.{h,cc}`，环境变量 `GE_FA_FUSION_PASS=1` 启用；**框架当前未包含该实现**，按下方 Step 3 新增，图构建集成参考 `src/graph_build.cc`）。在 `aclgrphParseONNX` 返回的 `ge::Graph` 上做模式匹配 + 节点替换，**不依赖 onnx 融合算子表达**，天然绕开全部 onnx 接入阻塞：

| 实测项 | 数据 |
|---|---|
| 融合块数 | 24/24 层 |
| kernel 变化 | attention 链 3392us → PromptFlashAttention 1784us（24×74.3us，MIX_AIC 协同 kernel）；kernel 总量 -18.3%，launches 594→402 |
| **端到端** | **avg 7.895→6.629ms（-16.0%）；vs 首基线 8.133ms 累计 -18.5%；QPS 126.6→150.8（+19.1%）** |
| 精度 | verify PASS（max_abs 8.214e-05 vs fp32 golden，与基线同级） |
| 回退 | 开关默认关闭（关闭实测 8.115ms = 原配置，零回归）；模型文件零改动 |

**为什么不走 onnx 融合算子表达**（三条 onnx 路线全部受阻，均为 CANN 9.1.0 工具链问题，M202 最小模型实证）：

| # | 阻塞 | 结论 |
|---|---|---|
| B1 | onnx parser 混合域限制：图中含非默认域节点时默认域节点报 `domain_version size != 1`；单 import 时 npu 节点报 `no responding version` | npu 域 FA 仅适用于纯 npu 图 |
| B2 | NPUFlashAttention onnx 插件 Const bug：展开子图无条件生成 drop_mask=Fill(Const_one)，Const 被重定型 DT_UINT8 但载荷 4 字节 → VerifyConstOp 拒绝（与输入映射无关，已破解映射并验证） | 默认域 opset17/18 死锁 |
| B3 | opset 注册空档：NPUPromptFlashAttention 默认域仅注册 11-16/19，LayerNormalization 仅 17/18 → 强制 opset16 + LN 分解 | LN 分解 +2.6ms 抵消 FA 收益（实测净收益 0） |
| B4 | opset16 下 ReduceMean axes 输入形式被 GE 忽略（全维归约）；属性形式可用 | 仅影响 LN 分解 workaround |
| B5 | 含广播的 fp32 元素链不进 AutomaticUbFusion | LN 分解性能代价来源 |

**图 pass 路线一一化解**：FA 节点由 pass 在 GE 图内创建（不经 prechecker/onnx parser → B1/B3 无关）；直接实例化 PromptFlashAttention 算子（不经过 NPUFlashAttention 插件 → B2 无关）；模型保持 opset17 + LN 算子（→ B4/B5 无关）。

## 功能确认参数矩阵（PromptFlashAttention，Ascend910_93 / CANN 9.1.0）

支持：B=1、S=577（非 16 对齐）、num_heads=16、dim=64、fp16、input_layout="BSH"（q/k/v 直取 [B,S,H*D]，head Reshape/Transpose 全消）、scale_value 属性、无 mask、无 causal（pre/next_tokens=2147483647）。
数值：vs fp32 golden max_abs 9.8e-4（fp16 输出量化水平）/ cosine 0.9999988。
算子接口：inputs query(0)/key(1)/value(2) 必选 + pse_shift/atten_mask/actual_seq_lengths 等可选（**不接线即视为缺席**，正常 GE 可选输入语义——这是图 pass 相比 onnx 插件的另一优势）；output attention_out(0)；attrs num_heads(必)/input_layout/scale_value/pre_tokens/next_tokens/sparse_mode/inner_precise。

## 扩展：GELU 链 -> GeluV2（同一图 pass 框架，GE_GELU_FUSION_PASS=1）

同机制可替换**任意"onnx 无表达、NPU 有算子"的小算子链**。已落地第二例：
- 模式：`0.5·x·(1+tanh(0.7978845608·(x+0.044715·x³)))` 九算子链（Mul×5+Add×2+Tanh，PyTorch GELU tanh 导出标准形态）→ `GeluV2(approximate="tanh")` 单节点
- 匹配要点：常量校验（0.044715/0.7978845608/1/0.5，容差 1%，**常量在链内 Mul/Add 各自位置**）+ x 源一致性（4 处引用同一张量）+ 中间节点消费唯一性
- **GNodePtr 指针不稳定**：节点同一性必须按 name 判定（SameNode 辅助）；Const 读值 `GetAttr("value", Tensor)`（fp16 位模式手工转 float）
- 实测（M200）：26 链（24 层 MLP + 头部 2），ABFO 870us → GeluV2 389us（kernel 快 2.2×），墙钟 **-7.6%**，verify PASS（精度更优）

## 工作流步骤

### Step 1: 热点确认（复用条目 5 四项专项）
attention 链合计占比 ≥ 10% 且 fusion_result.json 无既有 pass 覆盖。

### Step 2: 算子支持性确认（必做，产出证据）
1. 读 ops-info：`$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/config/<soc>/aic-<soc>-ops-info-transformer.json` 中 FA 族算子的输入/输出/attr/dtype
2. `strings $ASCEND_HOME_PATH/opp/built-in/framework/onnx/liboponnx_plugin_transformer.so | grep ai.onnx`：onnx 注册表（若走 onnx 路线才需要；图 pass 路线只需算子在 opp 注册，`OperatorFactory::IsExistOp` 运行时可查）
3. **最小验证模型**（临时工作区）：npu 域单 FA 节点 + q/k/v 输入跑通并对比 numpy fp32 golden；或直接用图 pass + dump_graph 验证
4. 场景参数核对：序列长/头数/头维/dtype/布局/mask/scale

### Step 3: 图融合 pass 实施（在 `src/` 新增 graph_fusion_pass.{h,cc}，框架当前未含）
1. **集成点**：`BuildGraph()` 中 `aclgrphParseONNX` 成功后调用 `gefusion::RunFlashAttentionFusion(graph)`；环境变量开关（默认关闭，满足回退纪律）
2. **模式匹配**（GE 解析后类型，与 onnx 名不同！）：`SoftmaxV2 ← [Mul(scale)] ← BatchMatMulV2(QK^T)`；两臂 `Transpose(Reshape(投影输出))`；`SoftmaxV2 → Cast×≤2 → BatchMatMulV2(AV)`，`AV.in1=Transpose(Reshape(v))`；`AV → Transpose → Reshape`。消费唯一性校验（QK/Softmax/AV 单消费者）防误融
3. **头数解析**：**parse 后节点 desc 全为空（不可依赖 shape 推导）**——从 Reshape 的 shape 输入链解析：`ConcatD → UnsqueezeV3 → Const/GatherV2`，逐位读 Const value 属性（动态位记 -1），H/D（位置 2/3）须为常量
4. **节点创建**：`OperatorFactory::CreateOperator(name, "PromptFlashAttention")` → `graph.AddNodeByOp` → `GNode::SetAttr`（注意 SetAttr 值参数为**非 const 引用**，须传左值）
5. **接线**：`graph.AddDataEdge(q_src, port, fa, 0)`×3；Rout 消费者 `RemoveEdge` + `AddDataEdge` 改接 FA 输出
6. **删除与 DCE**：模式节点按 **name** 收集删除（protobuf 节点对象 id 不稳定，M200 教训）；shape/const 链孤立节点 DCE（类型白名单 Const/Shape/GatherV2/UnsqueezeV3/ConcatD/Identity/Cast/Fill，图输出豁免）
7. **形状推导**：不设 desc，交给 AddGraph 流水（FA 算子自带 infershape）

### Step 4: 双门禁（强制）
- 精度：`GE_FA_FUSION_PASS=1 run.sh <id> --verify`；**先确认 `[GE RESULT] inference finished: success` 再看 PASS**（运行失败时 verify 会比较旧产物——M200 教训）
- 性能：同口径 circle ≥2000 + 复测 1 次；kernel 级对账（attention 链收益 vs 新增 FA 成本）
- 关闭开关复测无回归（pass 代码加入后的基线校验）

### Step 5: 结论归档
- 落地：perf_ledger + prof/iter{k} + 报告 + 部署配置注明开关必开
- 不落地：阻塞归因 + next_actions 更新

## 昇腾 NPU 专属要点

- **GE 解析后类型映射**（图 pass 模式匹配用）：MatMul→**BatchMatMulV2**、Softmax→**SoftmaxV2**、Unsqueeze→**UnsqueezeV3**、Concat→**ConcatD**、Gather→**GatherV2**、LayerNormalization→**LayerNorm(+Rsqrt/Adds)**；节点名保留 onnx 名（可能带 `PartitionedCall_` 前缀）
- **Const 节点读值**：`GNode::GetAttr("value", ge::Tensor&)`（不是 GetInputConstData——那是读"某输入为常量"的场景）
- **Identity 节点**：parse 会插入大量 Identity（权重接线），模式匹配须逐跳溯源跳过
- scale 传递：优先 FA `scale_value` 属性；模型已折叠（W_q*=scale）则传 1.0——不可叠加
- FA 为 **MIX_AIC 协同 kernel**（AIC/AIV 同 kernel 流水），改变核资源画像——吞吐场景部署前重跑 sweep（条目 3/8）
- 序列长非对齐（577）实测可用；新场景仍逐一核对（Step 2.4）
- **限核约束（M200 实证，2026-10-21）**：图 pass 配置（FA 或 FA+GELU）**必须全核运行**——FA 配置 AIV≤37 输出 NaN（GE 未拦截）、FA+GELU 组合 AIV<48 aicore timeout 挂死；pre-FA（pass 关闭）限核正常。**sweep 产出的限核/并发配置必须补串行 verify 门禁后方可推荐**（sweep 按框架约束不做精度验证，M200 曾因此推荐了 NaN 配置）
- **batch 泛化**：图 pass 对 batch 维透明（b4 实测 FA/GeluV2 正常，max_abs 5.686e-05）；b4 部署需 fp32 golden 重生成 + 输入 tile

## 产物目录约束

- 文档 → `docs/{model_name}/flash-attention-fusion/`；数据 → `data/{model_name}/flash-attention-fusion/`
- 中间产物 → `tmp/{model_name}/flash-attention-fusion/`
- pass 实现 → `GE_SESSION_DEMO/src/graph_fusion_pass.{h,cc}` + `graph_build.cc` 集成（框架代码，全模型复用；含 FA 与 GELU 两个 pass，独立开关 GE_FA_FUSION_PASS / GE_GELU_FUSION_PASS，另有 GE_FA_INNER_PRECISE 属性旋钮）
- GFM 删除线约束（禁裸波浪线）

## 验收标准

- 功能确认有数值证据与参数矩阵；pass 匹配数与网络结构一致
- 双门禁通过且复测稳定；关闭开关无回归
- perf_ledger/next_actions/部署配置同步（开关必开说明）

## 依据

【实测】M200 全链路（2026-10-13）：onnx 三路线阻塞实证（B1-B5）、图 pass 路线落地（avg -16.0%、QPS +19.1%、verify PASS、开关关闭零回归）。
【框架】GE v2 图 API（graph.h/gnode.h/operator_factory.h）；`graph_fusion_pass.{h,cc}` 为 M200 实测落地的实现形态，目前未随框架提供，复用前须先新增并过双门禁。
【框架】CANN 自带 FlashAttention 族算子文档与 GE 图 API 头文件（graph.h / gnode.h / operator_factory.h）。
