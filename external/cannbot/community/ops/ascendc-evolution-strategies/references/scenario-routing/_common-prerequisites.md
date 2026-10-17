# Common Prerequisites — 通用前置判定链（所有族共享）

> 所有族的 workflow 在进入族内阶段之前，先按 C1→C5 顺序过一遍本判定链。
> 每项格式：**判定条件 → 命中用卡 / 未命中跳过**。判定证据用 grep 即可获得，不要凭感觉。

## C1 标量/纯向量模拟矩阵运算？（收益数量级，最先判）

```bash
grep -nE "Mul\(|Muls\(|ReduceSum|WholeReduceSum" op_kernel/*.cpp   # 向量模拟 dot/matmul
grep -nE "GetValue|SetValue" op_kernel/*.cpp                       # 标量逐元素读改写
```

- **判定**：kernel 内层循环用标量循环或 Vector 指令（Mul+ReduceSum 等）模拟 dot product / matmul / 三角求解 / 链式乘加？
- **命中** → **P102**（MIX CV 融合重写：Cube 承担 mmad，Vector 承担 elementwise）；含三角求解再接 **P103**（Cube Neumann 积）
- **未命中**（已是 Cube 路径，或根本没有矩阵运算）→ 跳过
- ⚠️ 此判定优先级最高：向量假扮矩阵运算时，P94-P95 级向量化收益必然封顶，不要先做 C2

## C2 标量 for 循环残留？

- **判定**：存在逐元素/逐行标量循环（归约、逐元素算术、排序、gather/scatter、GetValue/SetValue）？
- **命中** → 按形态选 **P94-P101**：标量归约 → P94（Reduce API）；逐元素算术 → P95（Adds/Muls 链）；标量排序 → P96（Sort32/MrgSort 硬件引擎）；逐行 Muls/Adds → P99（BroadCast 批量）；GetValue/SetValue → P100（GatherMask）；host 侧列拆分 → P101；子 tensor 切片 → P97；负载不均 → P98（标量/向量混合分发）
- **未命中** → 跳过

## C3 dtype / 精度通路

- **判定**：FP16/BF16 输入且有精度要求？多 dtype 分发？量化输出？
- **命中** → 混合精度架构 **D1**（FP32 中间计算配 **A1**）；编译期多类型分发 **D2**；host 侧 TilingKey 分发 **D3**；FP8/INT4 转换 **D4**；BF16 特殊处理 **D5**
- **未命中** → 跳过

## C4 基础数据通路

- **判定**：内存搬运无双缓冲？shape 非 32B 对齐？
- **命中** → 双缓冲 **P1**（配流水同步 **P5**）；非对齐 **P7**（DataCopyPad）；多 UB tensor 分区 **P8**
- **未命中** → 跳过

## C5 数值稳定与事件同步

- **判定**：含 max/exp 归约（softmax 类）？跨 pipe 数据依赖？
- **命中** → 数值稳定 **A5**（max-trick）；SetFlag/WaitFlag 事件同步 **A4**；归约均值方差 → **A2**（Welford）
- **未命中** → 跳过

---

通过 C1-C5 后进入族 workflow 的 Step 2。注意：**C 系列判定结果要写进 implementation_note**，后续轮次不重复判定已否决的项。
