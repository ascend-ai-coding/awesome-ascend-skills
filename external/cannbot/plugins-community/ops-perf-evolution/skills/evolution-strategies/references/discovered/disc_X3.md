---
id: X3
origin: discovered
discovered_round: 1
discovered_from: round_1/parallel_3
base_speedup: 2.44x
---

# Strategy X3: 端到端 dtype 直传下沉 (End-to-End Dtype Passthrough)

## 核心思路
对多 dtype elementwise / foreach 算子，瓶颈往往不在 kernel 计算，而在 **wrapper / host 侧把低精度输入
(fp16/bf16) 全量 `.to(fp32)` 升 cast，kernel 再按 fp32 读写**——这使 HBM 流量相对原始 dtype 膨胀约 5x
（fp16: wrapper 入 cast 3x(读2B+写4B)=18B + kernel 读3x4B+写4B=16B + wrapper 出 cast 读4B+写2B=6B ≈ 40B/elem）。
优化 = 把 **dtype 边界从 wrapper 下沉到 kernel 内部**：DMA 按原始 dtype 直传（fp16/bf16=2B），UB 内 Cast 升 fp32 计算，
再 Cast 回原 dtype 写出；wrapper 不再做任何 dtype 转换，输出保持输入 dtype。HBM 流量降至 kernel 读 3x2B+写 2B=8B/elem，
并省掉 wrapper 两次全量 cast kernel。fp32 路径保持不变（bit-exact，无回退）。

## 适用场景
- **算子类型**: elementwise / foreach 等多 dtype（fp16 / bf16 / fp32）纯访存算子
- **瓶颈类型**: HBM 带宽受限（算术强度 < 1 FLOP/byte），且低精度 case 占 sum_time 大头
- **前提条件**:
  1. kernel 可模板化（D2）支持 T in {fp16, bf16, fp32}，核内统一 fp32 计算（A1）保精度
  2. host / wrapper 能按 scalar_type 分发 launcher，`output = empty_like(x)` 保原 dtype，对外接口（pybind / Python 签名）不变
  3. 标量参数（如 scalar）恒以 fp32 传入与计算，inf/nan 自动正确传播（不做 scalar 快路径）
- **预期收益**: 低精度 case HBM 流量约 5x 下降 + 省 wrapper 两次 cast，大 shape 接近线性加速（实测 c3/c5/c9 4.0-4.6x）

## 实现要点
- kernel: 单模板类 `Kernel<T>`；`if constexpr(sizeof(T)==4)` 走原 fp32 路径（1 个 tmpBuf），否则分配 3 个 fp32 scratch
  （x2f 复用为 div/muls tmp，x1f 复用为累加结果）；新增 fp16 / bf16 / fp32 三个 `__global__` + `extern "C" do_*` launcher（签名不变）
- tiling: `alignElements = 32 / elemBytes`（fp32=8, fp16/bf16=16），按 dtype 感知对齐
- host: 按 `scalar_type` 分发 launcher；`output = empty_like(x1)` 保原 dtype
- wrapper: 删除 `input_dtype / compute_dtype` 与所有 `.to()` cast，`t.reshape(-1).contiguous()` 原 dtype 直传，返回即原 dtype
- 舍入: bf16<-fp32 用 `CAST_RINT`(=RNE=PyTorch)，fp16<-fp32 用 `CAST_ROUND`，与 golden 一致
- **正确性约束**: tile 不能盲目放大——fp16/bf16 核内需 3 个 fp32 scratch（UB 占用 > fp32 路径），tile 过大会触发
  UB 临界 / Cast 隐含 scratch 溢出（实测 coeff36->28 tile7008 致 c2 mare=0.27 精度崩）；保持 baseline tile 或 coeff32 安全余量

## 已知局限
- 仅对"低精度 case 占比高 + 带宽受限"显著；fp32-only 算子无收益（fp32 路径流量已最优 16B/elem）
- 核内 fp32 计算引入 Cast 指令，但对带宽瓶颈算子 Cast 占比低，不影响收益
- 小 shape 受 per-tensor launch overhead 主导，对 sum_time 贡献小，优化优先级低

## 来源
自动发现于第 1 轮进化，算子 foreach_addcdiv_scalar，speedup 2.44x（c3 bf16 4.6x / c5 fp16 4.0x / c9 bf16 4.2x，fp32 路径 bit-exact 无回退）
