# RoPE 算子性能优化最佳实践

本文档总结了 RoPE (Rotary Position Embedding) 算子在 TileLang-Ascend 上的性能优化手段，对比原始实现与优化版本的关键差异。

## 目录

- [优化概览](#优化概览)
- [优化手段详解](#优化手段详解)
- [性能优化总结](#性能优化总结)
- [最佳实践建议](#最佳实践建议)
- [适用场景](#适用场景)
- [参考资料](#参考资料)

---

## 优化概览

| 优化项 | 原始实现 (`rope_mask.py`) | 优化实现 (`rope.py`) | 性能收益 |
|--------|---------------------------|---------------------|---------|
| Mask 生成方式 | CPU 预计算 + GM 搬运 | NPU 动态生成 | 减少 GM 访问 |
| 数据布局 | 外部生成索引 | 内联生成索引 | 消除外部依赖 |
| 内存访问 | 需额外搬运 mask/sin_mask | 零额外搬运 | 降低内存带宽压力 |

---

## 优化手段详解

### 1. NPU 内动态生成 Mask（核心优化）

#### 原始实现（rope_mask.py）

**问题**：在 CPU 端预计算 mask，再搬运到 NPU，增加 GM 访问开销。

```python
# CPU 端预计算（rope_mask.py）
idx = torch.arange(rope_dim * (block_M // 2), dtype=torch.int64, device="cpu")
mask = torch.empty(rope_dim * (block_M // 2), dtype=torch.uint32, device="cpu")
mask[0::2] = idx[1::2].to(torch.uint32)  # 偶数位放奇数索引
mask[1::2] = idx[0::2].to(torch.uint32)  # 奇数位放偶数索引
mask = mask * 4  # 字节偏移

sin_mask = torch.ones(rope_dim, dtype=torch.float32, device=device)
sin_mask[0::2] = -1
```

```python
# Kernel 内搬运（rope_mask.py）
mask_ub = T.alloc_shared([row_per_vec, rope_dim], MASK_DTYPE)
T.copy(mask, mask_ub)  # GM -> UB 搬运开销

sin_mask_ub = T.alloc_shared(rope_dim, ACC_DTYPE)
T.copy(sin_mask, sin_mask_ub)  # GM -> UB 搬运开销
```

#### 优化实现（rope.py）

**收益**：直接在 NPU 上动态生成 mask，消除 GM 访问。

```python
# 1. 生成索引序列 [0, 1, 2, 3, ..., rope_dim-1]
idx_ub = T.alloc_shared([row_per_vec, rope_dim], "int32")
T.tile.createvecindex(idx_ub, 0)  # 向量化索引生成

# 2. 通过 XOR 实现索引交错 [0,1,2,3,...] → [1,0,3,2,...]
tmp_ub_i16 = T.alloc_shared([row_per_vec, rope_dim], "int16")
ones_mask_ub = T.alloc_shared([row_per_vec, rope_dim], "int16")
T.copy(idx_ub, tmp_ub_i16)
T.tile.fill(ones_mask_ub, 1)  # 填充全 1
T.tile.bitwise_xor(mask_ub_i16, tmp_ub_i16, ones_mask_ub)  # idx ^ 1

# 3. 数据类型转换链：int16 → float32 → int32 → uint32
T.copy(mask_ub_i16, mask_ub_f32)
T.copy(mask_ub_f32, mask_ub_i32)
T.tile.mul(mask_ub_i32, mask_ub_i32, 4)  # 乘以 4（字节偏移）
T.reinterpretcast(mask_ub, mask_ub_i32, "uint32_t")  # 位重解释
```

**关键 API**：
- `T.tile.createvecindex`：向量化生成索引序列
- `T.tile.bitwise_xor`：位异或操作，实现索引交错（`idx ^ 1`）
- `T.tile.fill`：填充常量值
- `T.reinterpretcast`：数据类型位重解释，避免转换开销

---

### 2. NPU 内动态生成 sin_mask

#### 原始实现

```python
# CPU 端生成（rope_mask.py）
sin_mask = torch.ones(rope_dim, dtype=torch.float32, device=device)
sin_mask[0::2] = -1
```

```python
# Kernel 内搬运（rope_mask.py）
sin_mask_ub = T.alloc_shared(rope_dim, ACC_DTYPE)
T.copy(sin_mask, sin_mask_ub)
```

#### 优化实现

```python
# NPU 内动态生成（rope.py）
sin_mask_ub = T.alloc_ub(rope_dim, ACC_DTYPE)
T.tile.fill(sin_mask_ub, -1.0)  # 先填充 -1
for i in T.serial(0, rope_dim // 2):
    sin_mask_ub[2 * i + 1] = 1.0  # 奇数位设为 1
```

**优化点**：
- 使用 `T.tile.fill` 向量化填充初始值
- 仅对奇数位进行标量赋值（数量为 `rope_dim // 2`）

---

### 3. 数据类型转换优化

#### 原始实现

```python
# 直接使用 uint32 mask（rope_mask.py）
mask_ub = T.alloc_shared([row_per_vec, rope_dim], MASK_DTYPE)
T.copy(mask, mask_ub)
```

#### 优化实现

```python
# 利用硬件特性的转换链（rope.py）
mask_ub_i16 = T.alloc_shared([row_per_vec, rope_dim], "int16")
mask_ub_f32 = T.alloc_shared([row_per_vec, rope_dim], "float32")
mask_ub_i32 = T.alloc_shared([row_per_vec, rope_dim], "int32")
mask_ub = T.alloc_shared([row_per_vec, rope_dim], MASK_DTYPE)

# int16 → float32 → int32 → uint32
T.copy(mask_ub_i16, mask_ub_f32)
T.copy(mask_ub_f32, mask_ub_i32)
T.tile.mul(mask_ub_i32, mask_ub_i32, 4)
T.reinterpretcast(mask_ub, mask_ub_i32, "uint32_t")
```

**优化点**：
- `int16` XOR 操作效率更高（16 位整数运算）
- `float32` 中间转换用于后续乘法操作（硬件优化路径）
- `T.reinterpretcast` 避免数据拷贝，仅改变类型视图

---

## 性能优化总结

| 维度 | 原始实现 | 优化实现 |
|------|---------|---------|
| **GM 访问次数** | 3 次（x + mask + sin_mask） | 1 次（仅 x） |
| **外部依赖** | 需要 CPU 预计算 | 无外部依赖 |
| **内存带宽** | 需搬运额外 mask | 零额外搬运 |
| **代码复杂度** | CPU+NPU 混合逻辑 | 纯 NPU 逻辑 |

---

## 最佳实践建议

### ✅ 推荐做法

1. **在 NPU 上动态生成小型常量张量**（如 mask、索引）
   - 使用 `T.tile.createvecindex`、`T.tile.fill`、`T.tile.bitwise_xor` 等向量化操作
   - 避免不必要的 GM 访问
2. **优先使用 Tile API 实现向量化操作**
   - `T.tile.createvecindex` 生成索引序列
   - `T.tile.fill` 批量填充常量
   - `T.tile.bitwise_xor` 位运算
3. **合理设计数据类型转换链**
   - 利用 `T.reinterpretcast` 避免拷贝
   - 遵循硬件友好的转换路径

### ❌ 避免做法

1. **避免小型张量的 CPU → NPU 搬运**
   - mask、索引等小型常量应在 NPU 内生成
   - 减少 GM 访问开销

2. **避免外部生成依赖**
   - Kernel 应自包含，减少外部状态
   - 提高可移植性和可维护性

3. **避免非必要的参数传递**
   - 简化 kernel 接口
   - 降低调用开销

---

## 适用场景

本优化方案适用于以下算子：

- ✅ RoPE（Rotary Position Embedding）
- ✅ 需要 mask 或索引操作的算子
- ✅ 数据重排（permute、transpose）
- ✅ 元素交错操作（interleave、gather）

**核心思想**：将小型常量张量的生成从 CPU 转移到 NPU，利用向量化指令消除 GM 访问开销。

---

## 参考资料

- 原始实现：`examples/pos_embedding/rope_mask.py`
- 优化实现：`examples/pos_embedding/rope.py`
- API 参考：`../../../tilelang-api-best-practices/SKILL.md`

---

# Rope 类算子生成与性能对比（A2/A3 泛化 5 算子实测）

> 基于 5 个已 case 泛化的 Rope 类算子（`RopeWithSinCosCache`、
> `inplace_partial_rotary_mul`、`rotary_position_embedding`、`apply_rotary_pos_emb`、
> `kv_rms_norm_rope_cache`）在 Ascend 910B3 上的 TileLang 生成 + 手写 AscendC /
> CANN 内置算子对比实践总结。完整实测数据见文末「[Rope 类算子生成评估](#rope-类算子生成评估普通生成流程-vs-专用-rope-skill)」。

## 一、开始前：确认算子语义（最重要）

写 kernel 前必须确定三件事，否则精度必挂：

1. **旋转约定**（各算子不同，按 README/参考实现定）：
   - `interleave`：按奇偶位 `x[..., ::2] / x[..., 1::2]`
   - `half`：按半切 `x[..., :D/2] / x[..., D/2:]`（first/second）
   - `cat` 约定（kv_rms_norm_rope_cache）：`p1=cat(x_even,x_odd)`、
     `p2=cat(-x_odd,x_even)`、`k_rope=p1*cos+p2*sin`
   - neox / 非 neox 风格必须与参考实现一致
2. **数据布局**：`(N,H,S,D)` / `(num_tokens,1,2,half)` / `(B,N,S,Dkv)` 哪种？
   **按源输入种类分发布局，不要强转**：
   - 3D 源输入 `(N,H,head_size)`（RopeWithSinCosCache 等）→ `layout="3d"` 整头
     whole-head 拷贝：每 token 1 次整头 GM→UB、UB 内半切旋转、1 次整头 UB→GM；
     hs>rd 尾部直通不单独搬
   - 4D 交错 `(num_tokens,1,2,half)`（apply_rotary_pos_emb 类）→ `layout="4d"`，
     kernel 内零重排
   - ⚠️ 反例：v2 曾按 hs==rd 把 3D 强转 4D，加速比从 2.52x 掉到 1.71x（0/10 胜）；
     `head_size=256` 时 rotary 切片按 head 维（`view(N,H,hs)[:,:,:rd]`）
   - `kv_rms_norm_rope_cache` 是 **rms-first**：`kv[..., :rms]`=RMSNorm 部分、
     `kv[..., rms:]`=rope 部分
3. **dtype 与中间精度**：fp16 一律 fp32 中间计算，输出前 cast 回 half（max_diff ≈0.004）

## 二、TileLang kernel 骨架（纯 vector 必须 AIV_ONLY）

```python
import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[...], pass_configs=pass_configs)
def rope_kernel(nt, half, batch=8, num_cores=16, dtype="float32"):
    nc = min(num_cores, nt)          # 910B3: num_cores=16
    seg = nt // nc
    b = batch
    while b > 1 and seg % b != 0:    # batch 取 seg 的最大整除因子
        b -= 1
    batches = seg // b

    @T.prim_func
    def kernel(x: T.Tensor((nt, ...), dtype), ...):
        with T.Kernel(nc, is_npu=True) as (cid, vid):
            # alloc_ub / alloc_shared + 计算
            ...
    return kernel
```

要点：
- 纯 vector 算子必须 `TILELANG_ASCEND_AIV_ONLY=1`（或 kernel_type="aiv"），否则混核
  头开销 ~3-4us（实测是加速比 <1 的主因；inplace_partial_rotary_mul case1 从 0.64x
  → 1.42x，RopeWithSinCosCache 未开时大 shape 退化到 0.92-0.95x）
- 分核：`nc=min(16, nt)`、`seg=nt//nc`；小 case（nt≤4）少开核，nt=1 用 1 核
- **复合算子按行切分**：`block_M=4`、`VEC_NUM=1`；`VEC_NUM>1` 时 vid 与行号映射会
  错位（实测精度 bug 根源）
- 数据变换（奇偶/半切分离、rotate）在 kernel 内完成，禁止 host 重排
- 带 scatter 的算子：`k_cache[idx[b]]=k_rope[b]` 用 `T.copy` 逐行写回

## 三、多核分配与 BlockDim 选择

- token 维分核：`nc = min(num_cores, num_tokens)`、`seg = num_tokens // nc`
- batch 整除归一化：`b` 取 `seg` 的最大整除因子，最大化每批数据量、最小化同步次数；
  不要用 `if seg < b: b = 1` 退化（1 批变 4 批 × 4 同步 = 16 次/核）

| 数据量 | 推荐 BlockDim | 原因 |
|---|---|---|
| nt ≤ 4 | 4 | 小数据多核反而增加调度开销 |
| 4 < nt ≤ 16 | 16 | 充分利用 910B3 的 16 AIV 核 |
| nt > 16 | 16 + batch 循环 | 每核分多批，双缓冲流水 |
| nt = 1 | 1 | 单 token 无需多核 |

## 四、精度验证

- golden 用 torch 参考实现（与 kernel 同一旋转约定），不要用 torch.equal
- `torch.allclose(rtol=1e-5, atol=1e-6)`；fp16 可放宽（max_diff≈0.004）
- 覆盖边界：head_size=256 时 rotary 切片按 head 维；partial rotary 验证 slice 外
  未被修改；带 cache 的算子验证 cache 输出（不是只验证中间输出）

## 五、Profiler 对比（msprof op Task Duration）

TL 侧（TileLang 环境，如 03）：
```bash
msprof op --warm-up=10 --output=./msprof_tl/case$i python3 profile_tl.py $i
```

HW 侧（CANN 环境，如 02）：手写/CANN 内置算子用 ctypes CDLL 封装 aclnn 两段式：
```bash
# wrapper.cpp: aclCreateTensor(...) -> xxxGetWorkspaceSize(..., &ws, &ex) -> xxx(ws, wsSize, ex, stream)
# 编译: aarch64-linux-gnu-g++ -shared -fPIC wrapper.cpp -o libxxx_hw.so -lascendcl -lopapi
msprof op --warm-up=10 --output=./msprof_hw/case$i python3 profile_hw.py $i
```

- 读 `OpBasicInfo.csv` 的 `Task Duration(us)` + `Block Dim`（验证 BlockDim 符合预期）
- 每 case 独立 `TILELANG_CACHE_DIR` 与独立输出目录；采集前核对 Freq=1800/1800
- 加速比 = 标杆耗时 / 生成耗时，>1 生成更快；同时报**算术平均**与**比率平均**，并按
  dtype 分列
- msprof 设备端 Task Duration ≠ perf_counter 端到端延迟：两者统计对象不同，报告必须
  声明口径；Task Duration 采到 0 说明系统负载干扰，需重采
- **🛑 HW 必须 C wrapper 直调 aclnn（绕过 torch_npu op 分解）**：torch_npu 会把 fused
  op 分解为 Cast/BroadcastTo/ZerosLike 等子 op，msprof 抓到子 op 使 HW 时间严重偏低、
  加速比虚低（kv_rms_norm_rope_cache 0.46x → C wrapper 修正 1.17x）；必须设
  `argtypes`；用 `torch.empty` 非 `torch.zeros`；循环 launch ≥20（warm-up 10 + 测量
  10）；验证 `OpBasicInfo.csv` 的 Op Name 是 fused kernel 而非 Cast/ZerosLike
- **🛑 CANN 内置算子参数坑**：`aclnnKvRmsNormRopeCache` 的 cos/sin 第一维必须等于 B
  （`[B,N,S,Dk]` 或 `[B,1,1,Dk]`），`[1,...]` 会 tiling 报错 `561002`（cos or sin
  shape is invalid）；V1 结果写回 k_cache/ckv_cache（k_rope/c_kv 独立输出可能为 None）；
  `aclnnApplyRotaryPosEmbV2` 可 C wrapper 直调
- **生成耗时与性能耗时分开采集**：生成耗时 = 清 cache 强制重编译的 compile time
  （实测 8-14s/case，报告按 h 计）；性能耗时 = msprof Task Duration

## 六、性能不达标（<0.8）排查顺序

1. 确认 AIV_ONLY 已开（最大杠杆，头开销 4us→1us）
2. 分核/批次数：`nc=min(16,nt)`、batch 整除归一化、小 case 不空启核
3. 双缓冲流水（seg>1 时 ping-pong）
4. 减少 fp32 中间量 / 增大 vector 宽度 / 减少同步

## 七、TileLang API 易错点（kv_rms_norm_rope_cache 实战）

- **`T.clear` 不存在**：清零 buffer 用 `T.tile.fill(buf, 0)`
- **`T.reduce_sum` 签名**：`T.reduce_sum(buffer, out, tmp, dim)`，需要 `tmp` 和
  `dim`（`-1` 表示最后一维），不是三参数形式
- **dtype 字符串**：tilelang 用 `"float16"` 而非 `"half"`，`"float32"` 而非 `"float"`
- **`TILELANG_CACHE_DIR` 必须在 import tilelang 之前设置**，否则
  `kernel_lib.so not found in cache`
- **2D reduce 优于 1D**：2D shared buffer `[row_per_vec, R]` +
  `T.reduce_sum(buf, out, tmp, dim=-1)` 比 1D ubuffer + 手写 reduce 快得多
- **输出写回用 `T.copy` 批量写**：禁止逐元素串行写（R=512、row_per_vec=4 → 2048 次
  单元素写 → 103us；`T.copy` 批量写 → 4-6us）
- **padding 到 block_M 倍数**：`nt % block_M != 0` 时用
  `torch.nn.functional.pad` 补零，padding 行结果忽略
- **fp32 累加**：RMSNorm 平方和 fp16 累加会溢出，必须 `ACC="float"` 累加再 cast 回

## 八、IPRM 大 case ≥0.8 的两个结构性杠杆（v17→v20 实测）

背景：inplace_partial_rotary_mul 泛化 10 case 中 v15 时代大 shape 仅 0.55-0.73x。

**（1）pad 消除循环内 if（v17 问题 → v19 修复）**：`T.serial` 循环体内
`if r < BS` 导致 AUTO_SYNC 对每个 IfThenElse 无条件插 `PipeBarrier<PIPE_ALL>`
全同步，MTE2/VEC/MTE3 完全无法重叠。修复：prepare 阶段把 BS pad 到
`pad_BS = ceil(BS/nc)*nc`，循环体内无 if，AUTO_SYNC 只对真实数据依赖插细粒度 event。
实测 case3 从 1086μs（0.56x）降到 779μs。

**（2）单缓冲串行链 → `T.Pipelined` 多缓冲（v20）**：
`T.Pipelined(T.int32(seg), num_stages=2)` 包裹 serial 循环，load/compute/store 三级
流水重叠；case3 再降到 746μs（0.820x），大 case 全部 ≥0.8x。⚠️ 评估流水化看总体
aiv_time，不能只看单个 pipe 占用（v20 MTE2 等待反而暴涨，但净收益 ~33μs）。

**（3）v21 大块化失败教训**：j 循环内每次 `T.tile.broadcast` 前 AUTO_SYNC 插入
SetFlag/WaitFlag+PipeBarrier（3 条阻断 × 每块 4 次迭代），反向阻塞 MTE2 流水
（aiv_time 604→836μs）。结论：broadcast 逐行执行时不要塞进被打包的大循环。

## 九、rotary_position_embedding 小 case：VEC 广播 → `T.tile.brcast`

背景：rotary_position_embedding 泛化 10 case（bs1d cos，num_heads=32、D=256）中小
case（<500μs）卡在 0.64-0.76。根因是 TileLang 的 `T.tile.broadcast` 映射到
`AscendC::Broadcast`（**VEC 指令**，计入 vec_misc：小 case 广播 ≈34% + 算术 ≈45%
≈ 79% VEC 占用），而手写 HW 的 `RBroadCast` 用 `AscendC::Copy`+`CopyRepeatParams`
（UB→UB、`__inout_pipe__(V)`、单指令广播整段）不占 vec_misc 的 Broadcast 指令。

补齐 **`T.tile.brcast`**（对应 HW `Copy`+`CopyRepeatParams`：
`srcRepeatSize=0` 真广播、`dstRepeatSize=rowElems*sizeof(T)/32`、`mask=64`、
`repeatTime=行数`，逐 64 元素段循环；误用 `srcRepeatSize=8` 退化成连续拷贝且慢）
后，小 case 0.67/0.64/0.76 → 1.02/0.88/0.93，全 10 case ≥0.88、平均 0.96。

其他要点：
- **列偏移切片在 TL 后端是坏的**（`buf[:, half:]` 在 `T.copy`/`T.tile.mul` 里都会
  错，rel_err≈1.2）：必须把 half 维拆成独立 block，用 `(2, N, half)` 布局 + 
  `x[0,...]`/`x[1,...]` 取半，避免任何 `[:, :half]` 列切片
- 融合 rotate（host 预旋转 / 列偏移 GM 加载 / UB→UB 列切片 / `T.tile.gather`）在小
  case 均失败或退化；"4 半 D 广播 + 6 半 D 算术"（10 条 VEC）是当前最优
- `T.Pipelined` 坑：in-place 写（`T.tile.mul(x,x,cb)`）会报 `Multiple writes to
  overlapping buffer regions`，必须分离 load/compute buffer
- 预加载 cos/sin 到 UB 不改变总延迟：`T.Pipelined(2)` 已把 MTE2 load 与 VEC compute
  重叠，MTE2 不在关键路径，VEC 才是唯一硬瓶颈

## 十、反模式清单（NEVER DO THESE）

- ❌ 不先确认旋转约定就写 kernel
- ❌ 纯 vector 算子不开 AIV_ONLY
- ❌ host 端重排奇偶/半切
- ❌ 复合算子用 VEC_NUM>1（vid 映射错位）
- ❌ fp16 全程 half 计算
- ❌ msprof 共享 cache/输出目录
- ❌ 只报一个平均加速比、不分 dtype
- ❌ 循环体内 if 未 pad（AUTO_SYNC 全 barrier）

## 十一、总结决策树

```
Rope 类算子生成流程:
├─ 1. kernel_type="aiv" / TILELANG_ASCEND_AIV_ONLY=1 (纯 vector，无 GEMM)
├─ 2. 分核: nc=min(16, nt), seg=nt//nc
│   ├─ seg=1: BlockDim=nc, 单缓冲
│   └─ seg>1: batch 整除归一化, T.Pipelined 双缓冲
├─ 3. 布局: 按源输入种类分发
│   ├─ 3D (N,H,hs) → 整头 whole-head 拷贝, UB 内半切旋转
│   ├─ 4D 交错 (nt,1,2,rh) → layout="4d" 零重排
│   └─ rms-first (kv_rms_norm_rope_cache) → 复合按行切分 VEC_NUM=1
├─ 4. 符号合并: sin 负号编入 mask/offset
├─ 5. 精度验证: allclose(rtol=1e-5, atol=1e-6), 覆盖 head_size=256 边界
├─ 6. Profiling: msprof op --warm-up=10, 独立 cache, 验证 BlockDim, 声明口径
│   ├─ HW 必须用 C wrapper 直调 aclnn (绕过 torch_npu op 分解)
│   ├─ 设 argtypes, 用 torch.empty (非 zeros), 循环 launch ≥20 次
│   └─ 验证 OpBasicInfo.csv 的 Op Name 是 fused kernel 而非 Cast/ZerosLike
└─ 7. 泛化测试: case 按 HW 延迟区间分布 1/2/4/3 挑选, 候选池实测→避边界→
      verify_case_distribution 校验
```

## Rope 类算子生成评估（普通生成流程 vs 专用 rope skill）

> 本文档汇总 5 个已完成 case 泛化的 Rope 类算子，在使用「普通生成流程」「专用 rope
> skill（v3）」两组流程生成时，与手写 AscendC / CANN 内置算子标杆的加速比对比。

### 测量口径（所有数据统一）

- 平台：Ascend 910B3（A2/A3），CANN 9.1.0-beta.1，TileLang-Ascend
- 指标：`msprof op --warm-up=10` 采集的 **Task Duration(us)**（纯设备端耗时）
- 加速比 = 标杆耗时 / 生成耗时，>1 表示生成更快
- 采集纪律：每 case 独立 `TILELANG_CACHE_DIR` 与独立输出目录；先核对
  `Freq=1800/1800`；验证 `OpBasicInfo.csv` 的 `Block Dim` 符合预期（本批 40）
- 标杆：手写 AscendC kernel（RopeWithSinCosCache / inplace_partial_rotary_mul /
  rotary_position_embedding），或 C wrapper 直调 aclnnAPI（apply_rotary_pos_emb →
  `aclnnApplyRotaryPosEmbV2`，kv_rms_norm_rope_cache → `aclnnKvRmsNormRopeCache`，
  绕过 torch_npu 的 op 分解）
- 泛化：每个算子 10 个 case，按 HW 标杆延迟区间 1/2/4/3 分布（<100μs ×1、
  100-500μs ×2、500-1000μs ×4、≥1000μs ×3）挑选并经分布校验 PASS
- 精度：与 torch 参考实现 allclose 判定，全部 PASS（10/10）

### 加速比对比总表

| 算子 | 输入组织 / 布局 | 优化前（普通生成流程，AIV_ONLY、BD=40） | 优化后（专用 rope skill v3） | 说明 |
|---|---|---|---|---|
| `RopeWithSinCosCache` | `(N,H,head_size)` 3D | 2.59x（2026-08-26 复测，10/10 胜手写；8/20 同口径批次 2.50x，两批一致） | 2.52x（10/10 胜手写，与普通流程持平） | 3D 整头 whole-head 拷贝；布局分发避免 v2 强转 4D 弯路（1.71x、0/10） |
| `inplace_partial_rotary_mul` | `(B,N,S,D)` 4D | ≈0.001x（退化不可用） | 0.94x | 专用版 v20（pad 消除循环内 if + `T.Pipelined` 双缓冲）0.939x，10/10 ≥0.8x（最差 0.813x） |
| `rotary_position_embedding` | `(N,nh,D)` bs1d cos | 0.79x（补齐 `T.tile.brcast` 前，比率 0.83）→ 0.96x（补齐后） | 0.96x | 小 case 受限 TileLang VEC 广播；补齐 `T.tile.brcast`（MTE2 repeat 广播）后平均 0.96x |
| `kv_rms_norm_rope_cache` | `(B,S,dkv)` rms-first | 0.46x（torch_npu 抓错子 op）→ C wrapper 修正 1.17x → `block_M` 动态选择后 1.24x | 1.24x（final_comparison 批次 1.27x） | RMSNorm+rope 复合；收益主要是测量口径修正与 `block_M` 动态选择 |
| `apply_rotary_pos_emb` | `(nt,1,2,rh)` 4D 交错 | 0.14x（0.100~0.164x，均值 0.138x，退化不可用） | 0.95x（算术 0.9405x） | 4D 零重排 + AIV_ONLY + 双缓冲 |

> 说明：
> - `RopeWithSinCosCache` 普通流程 2.59x 为 2026-08-26 复测（8/20 同口径批次 2.50x，
>   两批一致）；历史 1.74x（8/14）为优化路线早期版本，不属于原版普通流程产物，不再列出。
> - inplace_partial_rotary_mul ≈0.001x、apply_rotary_pos_emb 0.14x（原版模板 +
>   AIV_ONLY，AIV 生效 Op Type=vector）均远低于 0.8，对这两类算子"专用 skill 是
>   ≥0.8 目标的必要条件"成立。

### 逐算子补充实测（msprof op，fp32）

#### RopeWithSinCosCache（两组口径，10 case）

- 普通 3D 整头生成流程（generic 3D 整头 + AIV_ONLY、BD=40、Freq=1800）：算术平均
  **2.59x**、几何 2.50x，10/10 胜手写（2026-08-26 复测；8/20 同口径批次 2.50x）
- 专用 rope skill v3（布局分发 + AIV_ONLY、BD=40）：算术平均 **2.52x**，10/10
  （skill_v2 对比 JSON opt 组 2.521x，与 gen 逐 case 几乎同值）
- 历史 1.74x（AIV_ONLY、BD=40）为优化路线早期版本（8/14），不属于原版普通流程产物
- 失败路线对照：v2 hs==rd 强转 4D 为 **1.71x**、0/10 反超手写

#### inplace_partial_rotary_mul（泛化 10 case，最终 v20）

| 阶段 | 算术平均加速比 | 达标情况 |
|---|---|---|
| v15（普通流程基线） | 0.68x | 大 case 0.55-0.73x 全部不达标 |
| v19（pad 消除循环内 if） | — | case3 1086μs → 779μs |
| v20（+ `T.Pipelined` 双缓冲） | 0.939x | 10/10 ≥0.8x PASS（最差 0.813x） |

#### rotary_position_embedding（泛化 10 case，bs1d cos）

| 阶段 | 小 case（<500μs） | 全 10 case 平均 |
|---|---|---|
| 补齐 `T.tile.brcast` 前 | 0.64-0.76x | 0.79x（比率平均 0.83x） |
| 补齐 `T.tile.brcast` 后 | 0.88-1.02x | 0.96x（比率平均 0.96x） |

#### kv_rms_norm_rope_cache（泛化 10 case）

- 口径修正：torch_npu 绑定抓到 ZerosLike/Cast 子 op（1.7us）→ C wrapper 直调
  `aclnnKvRmsNormRopeCache` 抓到真实 fused kernel 后，加速比从 0.46x 修正为 1.17x
- `block_M` 动态选择（nt≤16 → block_M=1、vec_num=1；nt>16 → block_M=8、vec_num=2）
  后平均 1.17x → **1.24x**，10/10 ≥0.8x

#### apply_rotary_pos_emb（泛化 10 case，fp16 half 模式）

普通生成流程（原版模板 8/13 快照 + AIV_ONLY=1）：10 case 0.100~0.164x，算术平均
0.138x（AIV 生效，Op Type=vector；BlockDim=16 为模板自身调度）；专用版（4D 零重排 +
AIV_ONLY + 双缓冲）各 case 0.66~1.0x 量级，平均 0.95x。

### 关键结论

- 对 `inplace_partial_rotary_mul`（≈0.001x）、`apply_rotary_pos_emb`（0.14x）：
  普通流程生成均远低于 0.8，**专用 skill 是 ≥0.8 目标的必要条件**；对
  `rotary_position_embedding` 小 case，专用 skill 把平均从 0.79x 提到 0.96x。
- 对 `RopeWithSinCosCache`：普通 3D 整头生成 + AIV_ONLY + BD=40 已能达到 2.59x，
  与专用 v3 持平；**专用 skill 的价值是把流程稳定在正确路线上**（3D 整头 + AIV_ONLY
  + BD=40），避免 v2 强转 4D 弯路（1.71x、0/10）与未开 AIV 的退化。
- 对 `kv_rms_norm_rope_cache`：收益主要是测量口径修正（torch_npu 子 op → C wrapper
  直调 aclnn）与 `block_M` 动态选择，1.17x → 1.24x。
- 全套经验（AIV_ONLY + 布局分发 + 旋转约定 + msprof 对比纪律）固化后可让后续相似
  算子直接复用，避免重复踩坑。
