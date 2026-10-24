# PyPTO 特有接口

Torch 中无对应、面向 NPU 硬件能力的接口。算子涉及量化数据流、稀疏注意力、RoPE 重排、多核写同一输出、UB/L1 gather 等场景时优先查本表；合理使用可减少 GM 搬运与 kernel 调用次数。

平台约束缩写：950PR/DT = Ascend 950PR/950DT，A2/A3 = Atlas A2/A3 系列。

| 接口 | 功能 | 平台约束 | 性能场景 |
|------|------|---------|---------|
| `scaled_mm` | mx 量化矩阵乘 `out=(mat_a*scale_a)@(mat_b*scale_b)`，`out_dtype` 必填，支持 a/b 及其 scale 的转置标志 | 仅 950PR/DT | FP8/MXFP8 量化 matmul 融合（gym：DeepSeek MLA prolog 的 q_a_proj / compressed_kv / q_b_proj 量化投影、quant_grouped_matmul） |
| `quant_mx` | `quant_mx(input, quant_dtype, mode, axis)` 高精度→MX 量化，返回（量化结果， 共享指数 scale）；输入 FP16/BF16/FP32，输出 FP8E4M3 / FP4_E2M1X2（FP4 仅 FP16/BF16 输入） | 仅 950PR/DT | MXFP8/FP4 量化数据流生成（gym：MLA prolog 中 RmsNorm 输出按 block 量化为 FP8E4M3 供 `scaled_mm` 消费） |
| `quantize` | `quantize(input, scale, otype, axis, zero_points)` FP32→INT8 对称量化 | 全平台 | 量化数据流入口 |
| `dequantize` | `dequantize(input, scale, otype, axis, zero_points)` INT8/INT16→FP32，应用 scale 与 zero_points | 全平台 | 量化数据流出口 |
| `atomic_add` | `atomic_add(src, offsets, dst)` 以 offsets 为基准把 src 区域原子累加到 dst | 全平台 | 多核分块写同一输出免串行（gym：flash_attention_mha_grad 中 dq/dk/dv 梯度按序列位置原子累加回 GM） |
| `axpy_` | `axpy_(y, x, alpha)` 原地 `y = alpha*x + y` | 全平台 | 向量累加一步完成，替代 mul+add |
| `interleave` | `interleave(input, other)` 两输入按末维逐元素交织，交织流按中点拆为两个输出 | 仅 950PR/DT | 与 `deinterleave` 互逆，偶奇位重排、NZ 格式转换 |
| `deinterleave` | `deinterleave(input, other=None)` 交织流按偶/奇位拆为两个输出；支持双输入（前/后半）形式 | 仅 950PR/DT | RoPE 偶奇位拆分（gym：InterleaveRope 将 x/cos/sin 拆为 x_e/x_o 后分别乘加） |
| `pack` | `pack(self)` 铺平为一维并把原始字节解释为 uint8 | 950PR、A2/A3 | INT4/FP8 等低精度数据按字节搬运 |
| `unpack` | `unpack(self, dstDataType)` uint8→指定 dtype 解包 | 950PR、A2/A3 | 与 `pack` 互逆 |
| `index_add__ub` | UB 内 inplace index_add，`source` 按 alpha 缩放后加到 `input` 对应块 | 全平台，约束较多 | 索引范围在 UB 容量内时免 GM 往返 |
| `index_add_ub` | `index_add__ub` 的 non-inplace 版本 | 全平台，约束较多 | 同上 |
| `gathermask` | `gathermask(self, pattern_mode)` 按内置 Mask 位模式抽取元素（Bit=1 保留），7 种模式（如尾轴每 2 取第 1/2 个） | 全平台 | 固定模式元素压缩（gym：InterleaveRope 用 pattern_mode=1/2 做偶/奇位抽取，与 `deinterleave` 同场景互替） |
| `expand_exp_dif` | `expand_exp_dif(input, other)` 广播后逐元素 `e^(input−other)` | 全平台 | softmax 类 sub+exp 两步并一步 |
| `conv` | `conv(input_conv, weight, out_dtype, strides, paddings, dilations, *, groups, transposed, extend_params)` cube 卷积，支持 bias 融合 | 全平台 | conv2d/conv3d 组合方案已引用 |
| `fillpad` | `fillpad(input, mode="constant", value=0)` 不改变形状，填充 valid_shape 之外区域；支持 1-2 维常量模式右/下填充 | 全平台 | 尾部 chunk 补零对齐（gym：Qwen3-Next gated_delta_rule、chunk_kda 中 q/k/v/beta 尾块填充） |
| `lrelu` | `lrelu(input, negative_slope=0.01)` LeakyReLU 单算子 | 全平台 | 替代 where+mul 组合 |
| `cbrt` | 逐元素立方根 | 全平台 | 数学便利接口 |
| `ceil_div` / `floor_div` | 逐元素除法向上 / 向下取整 | 全平台 | 数学便利接口 |
| `uniform` | `uniform(shape, key, counter, alg, dtype)` 生成 [0,1) 均匀分布随机数（状态式 RNG） | 仅 950PR/DT | 设备侧随机初始化、dropout 掩码 |

### experimental 定制接口

面向特定融合场景的定制接口，约束较多且不保证稳定性，需确认约束满足后使用：

| 接口 | 功能 | 使用场景 |
|------|------|---------|
| `online_softmax` | scores×scale 后按第 0 维求局部统计（块 max、指数和、未归一化指数结果） | FlashAttention 分块注意力（gym：flash_attention_mha），与 `online_softmax_update` 配对使用 |
| `online_softmax_update` | 将新块的 max/sum 合并进在线 softmax 状态并更新中间输出 | 分块推进时的状态合并（gym：flash_attention_mha 逐块更新 mi/li/oi） |
| `gather_in_ub` | 按 block_table 将选中 token 的 KV cache 从 GM 加载到 UB，支持 Page Attention，受 UB 容量约束 | 稀疏注意力 decode |
| `gather_in_l1` | 上述能力的 L1 版本，支持 B 矩阵 / 转置标志 | 稀疏注意力中较大块 KV 读取 |
| `transposed_batchmatmul` | (M,B,K)→(B,M,K) 转置 + batch matmul + 结果转回 (M,B,N)，支持 FP16/BF16 | 免单独 transpose 的批矩阵乘（gym：MLA prolog 中 q_nope 与 w_uk 的吸收变换） |
