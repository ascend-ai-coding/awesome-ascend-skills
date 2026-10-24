# TileLang/PTO 精度与性能门禁

## 实施顺序

1. 优先从当前仓库同类算子实现派生；使用 bundled code 前先查 [模板成熟度](template_status.md)。
2. 通过实际导入路径定位 TileLang，核对每个 API 在 PTO lowering 中存在；不使用 CUDA-only schedule、target 或 pass config。
3. 建立 fp32 或更高精度 PyTorch reference，先实现完整 shape/dtype fallback。
4. 运行定向精度测试；修复全部数值、越界、编译和未实现路径。
5. 一次只改变 tile、核数、stage、resident、执行域、融合中的一个变量。
6. 每次变更重新跑定向精度测试，再在固定环境 benchmark。
7. 代表 shape 无回退后运行相关完整测试套；接口要求时继续项目要求的扩展测试。

## 精度矩阵

先读取公开接口合约，再覆盖最小/常见/最大 shape、允许余数类、动态维、非连续 stride、所有 dtype、forward/backward、0、正负极值、抵消、NaN/Inf 与空任务。接口承诺通用尾块时增加 tile-1/tile/tile+1；接口只允许特定对齐时不得把未实现 fallback 当成既有能力。Reduction、Norm、Softmax、Scan carry 和 GEMM L0C 默认 fp32；输出只在最终边界转换。

记录测试收集数、通过数、首个失败、shape、dtype、容差、异常类别和根因。`NotImplementedError` 是未实现，不是通过。禁止扩大容差、降低 reference 精度、跳过 case 或只报告此前通过的子集。

## 性能矩阵

保持输入分布、shape、dtype、输出语义、warmup、repeat、设备、频率和并发一致。报告 kernel latency、端到端 latency、有效 GM bytes/带宽、Cube/Vector/MTE 时间、核利用率、UB/L1/L0/工作区字节、stage 和生成代码体积。

多 kernel 与 host pipeline 必须计入全部 launch、event、workspace 与同步。没有当前设备实测数据时只报告设计，不报告收益。

## 命令

```bash
TILELANG_DEFAULT_TARGET=pto pytest <test_file> -x
TILELANG_DEFAULT_TARGET=pto pytest <test_file>
```

Bundled 模板回归：

```bash
python scripts/validate_templates.py
TILELANG_DEFAULT_TARGET=pto python scripts/validate_templates.py --npu --num-cores <available_aiv_cores>
```

正确性测试仅在确认设备绑定和隔离机制后按可用设备数并发，否则串行；OOM 时降低并发，不减少用例。benchmark 保持设备独占。
