# Ascend 硬件资源与编译器有效容量发现

Tiling、驻留和多版本流水必须区分硬件物理容量、当前 lowering 的执行域预留和 kernel 显式分配，不能继承历史机器或单个算子的“安全上限”。

## 1. 记录当前硬件

记录目标芯片型号、架构、CANN 版本、Vector/Cube 核数以及 UB/L1/L0 物理容量，并注明数据来源。复用统一入口传入的完整 SoC / NpuArch 硬件探测证据，按名称加载 `npu-arch` Skill 并传入平台 `Ascend950`，核对架构映射、物理容量和运行时查询方式。证据缺失或失效时，按当前 Skill 入口的平台路由重新探测；禁止从静态映射反推本机型号。本指南不重复维护另一份硬件参数表。

## 2. 查询编译器有效容量

先定位任务实际导入的 TileLang，而不是按目录名猜测版本：

```bash
python -c 'import pathlib, tilelang; print(pathlib.Path(tilelang.__file__).resolve())'
```

在对应源码中查找资源上限和执行域预留：

```bash
rg -n "GetSharedMemoryLimit|UnifiedBuffer|HasSimtVF|SIMT_VF|ThreadContext" <tilelang_root>/src/ascend
```

当前已核验的 `src/ascend/transform/auto_schedule.cc` 会检查最终 kernel body：只要仍含 `SIMT_VF` block，就从 248 KiB UB 中预留 32 KiB 线程上下文，使 shared-memory 上限变为 216 KiB；纯 `SimdVF` 不触发。该数值仅是当前源码事实，其他版本和架构必须重查。编译期消除的分支不影响，仍留在 IR 中的初始化、尾部或 fallback `SimtVF` 也会触发 kernel-wide 预留，因此执行域变化后必须重算容量；这不代表 `SimtVF` 本身应被删除。

## 3. 建立资源账本

```text
物理容量：已确认的设备资料或任务环境工具
编译器有效上限：当前 lowering 对最终 kernel body 的计算结果
显式 footprint：sum(aligned_extent * dtype_bytes * buffer_versions)
常驻 footprint：LUT / index / state / metadata
隐式或临时开销：仅记录当前源码、IR 或编译报错可证明的部分
安全余量：具体保护对象和数值依据
剩余容量：编译器有效上限 - 上述 footprint
```

物理容量与编译器上限不一致时先核对架构、导入路径和版本。安全余量必须有具体保护对象，不能用任意 cap 代替公式。用账本枚举完整连续单元、最大合法容量及相邻 SIMD/DMA 对齐候选；`SimtVF`、buffer version、padding、LUT/index 或临时物化变化后重算，并用生成 IR/运行结果验证。
