# 成本模型契约 (cost-model contract)

## 概述

`compute_efficiency.py` 将算子的实测耗时转换为对硬件利用率与优化空间的量化评估。评估依赖算子的**固有工作量**(访存字节数与计算量 FLOP),该工作量由**成本模型**提供。本文件规定成本模型的接口契约。

## 背景与用途

ops-profiling 采集的是原始性能指标(流水占比、带宽、缓存命中率),不直接给出算子相对硬件峰值的利用率与理论上限。本工具在其基础上计算两项利用率指标:

- **MBU(Memory Bandwidth Utilization,内存带宽利用率)**:实测有效带宽 ÷ 硬件峰值带宽;接近 1 表示访存已接近硬件上限。
- **MFU(Model FLOPs Utilization,算力利用率)**:实测有效算力 ÷ 硬件峰值算力;接近 1 表示计算已接近硬件上限。

据此判定瓶颈类型(访存受限 memory-bound / 计算受限 compute-bound),并给出理论最优耗时(roofline 上限)与优化结论。`ops-perf-optimize-workflow` 工作流在分析阶段据此判断算子是否仍具备优化空间:当利用率已接近上限、且无可行的降载手段(减少访存量或计算量)时提前收敛,避免无效迭代。

成本模型提供上述指标的分子(固有工作量)。

## 契约

一个成本模型是纯函数,把算子 shape 参数映射到固有工作量,**与 kernel 实现无关**:

```python
from cost_models import Cost, register

@register("my_op")
def cost(params: dict) -> Cost:
    ...
    return Cost(flops=<有效乘加*2>, hbm_bytes=<必须过内存边界的字节>, label="...")
```

- `flops`：有效乘加数 ×2(MFU 的分子)。只算**有用**计算,不含实现冗余(如 term 分解的重复 MAC)。
- `hbm_bytes`：**每次调用必须跨内存边界一次**的字节 = 权重 + 不同的输入输出。是 MBU 的分子。
  - 注意:这些字节从 HBM 还是 L2 供给,是**放置**问题,由 `--peak-hbm-bw` 选哪档峰值体现,**不在成本模型里区分**。
- `unit`：FLOP 跑在哪个引擎——`"cube"`(matmul,默认)、`"vector"`(纯向量 elementwise/reduction)、`"mixed"`(cube 主导融合,天花板按 cube)。**决定 MFU 除哪个峰值**:`vector` 用 `--peak-vec-flops`,否则 `--peak-cube-flops`。
- `label`：人读注释(可选)。

### 纯 C / 纯 V / CV 融合都适用

- **MBU / 访存 roofline / ceiling / verdict**:三类通用。纯 V 算子几乎必然 memory-bound,MBU 就是头号指标,`NEAR-BANDWIDTH-CEILING` 直接成立(减字节手段=算子融合减 IO / 降 dtype)。
- **bound 判定**(`--pipe-util`):同时读 cube(`aic_cube_ratio`,兼容 application 的 `aic_mac_ratio`)与 vector(`aiv_vec_ratio`),纯 V 算子正确判成 Vector-compute-bound 或 Memory-Bound,不误判。
- **MFU**:按 `unit` 除对的峰值——纯 V 算子**必须**传 `--peak-vec-flops`,否则报错(不给无意义的 cube-MFU)。`--compute-unit cube|vector` 可覆盖模型声明的 unit。

`Cost` 暴露 `total_flops` / `total_bytes`,与 testkit `core/metrics.py` 的消费口径兼容。

## 用法

```bash
# 注册模型 + shape 参数
python3 scripts/compute_efficiency.py --duration-us 57.5 \
    --op mla_prolog --peak-hbm-bw 1.9 --peak-cube-flops 400

# 从 collect_perf_application.py 的 performance.csv 取中位耗时
python3 scripts/compute_efficiency.py --perf-csv results/xxx/performance.csv \
    --case-id my_case --op attention --params '{"B":1,"N1":32,"S1":1,"S2":4096,"causal":true}' \
    --peak-hbm-bw 1.9 --peak-cube-flops 400

# 没有注册模型时,直接传 FLOP/字节降级
python3 scripts/compute_efficiency.py --duration-us 20 --flops 3.18e9 --bytes 99.5e6 \
    --peak-hbm-bw 1.9 --peak-cube-flops 400

# 给了 op-mode 的 PipeUtilization.csv 时,用实测 pipe 占比复核 bound(并自动推 mte2_util)
python3 scripts/compute_efficiency.py --duration-us 63.2 --op mla_prolog \
    --pipe-util OPPROF_xxx/PipeUtilization.csv --peak-hbm-bw 1.9 --peak-cube-flops 400

# 直传 MTE2 载入时间,识别"权重搬运已达带宽底但流水未重叠"的 PIPELINE-TRAP
python3 scripts/compute_efficiency.py --duration-us 85.72 --op mla_prolog \
    --mte2-us 51.5 --peak-hbm-bw 1.9 --peak-cube-flops 400
```

## 峰值口径(必须显式,否则不可比)

优先级:CLI `--peak-*` > 环境变量 `PEAK_HBM_BW`/`PEAK_CUBE_FLOPS` > 默认。
默认 **1.6 TB/s / 378.47 TFLOPS** 是 testkit 的 Ascend950PR-PCIE 档;**务必按被测 SoC 传入正确峰值**,报告会记录用的哪档。
`--peak-hbm-bw` 选 HBM 峰值还是 L2 峰值,决定 MBU 是"对 HBM"还是"对 L2"——见下。

## 已注册模型(种子)

| op | 关键参数 | 说明 |
|---|---|---|
| `matmul` | M,N,K,dtype_bytes,batch | 通用稠密矩阵乘 |
| `attention` | B,N1,N2,S1,S2,D,DV,DRope,causal,kv_reused | no-mask + causal(滑窗/GQA/TND 全量见 testkit `core/metrics.py`)|
| `mla_prolog` | M,He,Hcq,N,D,Dr,Hckv | 四段投影(cube);默认参数=参考 decode 用例(3.18GFLOP/99.5MB 权重)|
| `elementwise` | n_elems,ops_per_elem,in_tensors,out_tensors | **纯 V**;unit=vector,需 `--peak-vec-flops` |
| `reduction` | n_in,n_out,ops_per_elem | **纯 V**;unit=vector,需 `--peak-vec-flops` |

新增算子:在 `scripts/cost_models/<op>.py` 里 `@register("<op>")` 一个函数即可,`__init__.py` 会自动加载。

## 输出与 verdict

`compute_efficiency.py` 输出 MBU/MFU/mte2_util/bound/ceiling_us/headroom + 一句 **verdict**(给工作流的收敛提示):
- `NEAR-BANDWIDTH-CEILING`:memory-bound 且 MBU≥90% → **非量化流水/tiling 重排动不了耗时下限,唯一杠杆是减字节(量化/权重格式/dtype)**;
- `MEMORY-CEILING`(PIPELINE-TRAP):memory-bound 且 wall-MBU 低、但载入管道 **`mte2_util`≥90%**(权重搬运已达带宽底)→ wall 与 MTE2 的差是流水未重叠;**需重读权重的重叠会抬高 MTE2 超过带宽底而净亏**,故按到顶处理,除非有经证明不重读权重的重叠方案。需传 `--mte2-us`(MTE2 载入时间)或 `--pipe-util`(自动从 MTE2 占比推)才能判定;
- `NEAR-COMPUTE-CEILING`:compute-bound 且 MFU≥70% → 减 FLOP(算法/低精度)才有空间;
- 否则给出朝天花板的 headroom 和调优方向(overlap/prefetch 或 tiling/occupancy)。

## 与 MBU 的 HBM/L2 之辨(实战注意)

同一算子若权重能常驻 L2,`--peak-hbm-bw` 传 HBM 峰值会算出 MBU>90%(甚至接近实测 BW 超过 HBM);
传 L2 峰值则反映"是否打满 L2"。**建议两档都看**:对 HBM 的 MBU 说明"相对 DRAM 有多快",对 L2 的 MBU 说明"片上带宽是否见顶"。参考 mla_prolog 实测:57.5µs 时对 HBM MBU≈95%、MTE2 实测 BW 打满 L2(~2.9TB/s)。
