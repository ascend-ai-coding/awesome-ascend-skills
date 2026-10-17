# 技术卡片 vec-05：消除中间 UB 暂存（寄存器内直算）

- **适用代际**：arch35（A5 / DAV_3510）独占；本卡依赖 RegBase / MicroAPI / VF，不用于 arch22 实现。
- **适用 bound**：VEC / 访存
- **一句话**：能在向量寄存器里接力的就不落 UB 再读回，省掉 store→load 往返与 `LocalMemBar` 流水排空。

> 本卡片只适用于已确认的 AIV 向量计算链。Cube MMAD/内嵌 MatMul 不属于适用范围；低阶 Cube MM 重建应按 MatMul 路由单独设计 `L1 → L0 → Mmad → Fixpipe` 流水。

## 何时用（诊断特征）

- MTE3（写回）与 MTE2（载入）出现 **store→load 往返**。两种常见形态：
  1. **pre-cast staging**：先把 src cast 成 fp32 写进临时 UB，下一步再 load 回来算；
  2. **行归约标量广播 staging**：`Reduce` 出 lane0 标量写 UB 槽 → `LocalMemBar` → 从 UB 广播读回，把流水排空两次（softmax/norm/RMS 的常见 IPC 杀手）。
- **UB 驻留向量链的逐 API 读写**：链已在 UB 内接力（无 GM 中转、无上述可见往返）但 VEC 耗时主导时，每个向量 API 仍各带一次 UB 读 + 一次 UB 写——同样属于本卡片的目标形态，不要求出现可见 staging 往返；深度 VEC bound（占比 >80%）时应优先评估本卡片。确认计算单元为 Cube MMAD 时跳过本卡片。

## 原理

能在向量寄存器里直接接力的就不要落 UB 再读回。省掉中转 UB buffer、缩短 VF prologue、减一来一回的 load/store，并消除 `LocalMemBar` 的流水排空。对 UB 驻留链，整链寄存器化后每个中间量不再经过 UB 读写，UB 占用也随之下降。

## 怎么改（before / after）

**改法 A：pre-cast staging → 寄存器内直算**
- 删掉 `xScaleFloatBuf_` 之类 fp32 中转 buffer 及其 `InitBuffer`/`Duplicate`。
- 调用侧直接传原始 dtype 指针（`bfloat16_t*`/`half*`/`float*`），删掉 `if constexpr(sizeof==4)` 预 cast 分支和 `CastXxxType` helper 层。
- cast 融合到唯一使用点（如 `MulWeightAndReduceSum` 的寄存器流水里就近 cast）。
- 用 **C++ 函数重载**（按 dtype 选实现）替代上层 `if constexpr` + 多个 helper。
- **推广到全链（UB 驻留链）**：入口 `Reg::LoadAlign` 一次、链内 RegTensor 接力、出口 `Reg::StoreAlign` 一次，逐 API 的 UB 读写全部消除；"先 Cast 到窄类型再 Cast 回来"一类的 staging 用寄存器内两次 Cast 替代。该改法仅用于 AIV 向量链；可拷贝骨架见 `ascendc-performance-best-practices` VF 范例库 `register-resident-vector-chain` 条目。

**改法 B：行归约标量广播 → 寄存器内 `Reduce` + `Duplicate`**
```cpp
// before：写 UB → LocalMemBar → broadcast-load（2 barrier + UB scratch）
Reduce<ReduceType::MAX>(vRed, vMaxAcc, preg);
Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(scratch, vRed, preg);
LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(vMaxBrc, scratch);

// after：寄存器内归约到 lane0，再 Duplicate 就地广播到全 lane
Reduce<ReduceType::MAX>(vMaxBrc, vMaxAcc, preg);   // lane0 = 行 max
Duplicate(vMaxBrc, vMaxBrc, preg);                 // 广播到全 lane，无 barrier/UB
```
去掉两处 `LocalMemBar` 流水排空 + UB scratch + 一来一回的 store/broadcast-load，行归约类 VF 的 IPC 通常明显抬升。

**改法 C：错位 / 重排靠 UB 搬运 → 寄存器内 offset load + shuffle**
- 需要对同一份数据做「错位对齐」或「偶奇重排」时（RoPE rotate-half：前后半相乘相加；interleave：交错旋转），旧写法常先用 `DataCopy` 在 UB 上把数据挪位再逐段算——多一整段 UB→UB 搬运 + 临时 buffer + 整块 `Muls(-1)` 预处理。
- 直算：错位 = 用**不同 offset** 把同一 UB 的两段 load 进两个寄存器；重排 = 寄存器内 `DeInterleave`/`Interleave`/`Muls(-1)` 完成，全程不落 UB。
```cpp
// before：rotate-half 靠 UB→UB 搬运做前后半错位 + 整块 sin 翻符号预处理
DataCopy(xNew, x[halfLen], cp);          // 后半 -> xNew 前
DataCopy(xNew[halfLen], x, cp);          // 前半 -> xNew 后
Muls(sin, sin, -1.0, len);
Mul(x, x, cos, len); Mul(xNew, xNew, sin, len); Add(y, x, xNew, len);

// after：两个 offset 分别 load 进寄存器，一拍乘加算完前后两半，无 UB 搬运/无 sin 预处理
preg = Reg::UpdateMask<float>(cnt);
Load2(inUb, vregIn, vregHalfIn, preg, off, halfOff);      // 前/后半各一寄存器
Load2(cosUb, vregCos, vregHalfCos, preg, off, halfOff);
Load2(sinUb, vregSin, vregHalfSin, preg, off, halfOff);
Mul(vregSin, vregSin, vregHalfIn, preg); Mul(vregCos, vregCos, vregIn, preg);
Sub(vregOut, vregCos, vregSin, preg);            // out_front = cos*x_front - sin*x_back
Mul(vregHalfCos, vregHalfCos, vregHalfIn, preg); Mul(vregHalfOut, vregHalfSin, vregIn, preg);
Add(vregHalfOut, vregHalfOut, vregHalfCos, preg);// out_back  = sin*x_front + cos*x_back
// interleave 交错旋转：寄存器内重排，不落 UB
// Reg::DeInterleave<float>(a,b,in0,in1); Muls(b,b,-1.0,preg); Reg::Interleave<float>(in0,in1,b,a);
```
省掉 rotate 的 UB→UB 错位搬运、整块 `Muls(-1)` 预处理和所有中间 LocalTensor。（`DeInterleave/Interleave/UpdateMask/RegTensor` 统一使用 `AscendC::Reg::`，并按实际头文件核验目标架构支持。）

## 性能数据

msprof `aiv_time`：消除 store→load 往返与 `LocalMemBar` 排空。**待实测**（预期：行归约类算子 IPC 明显抬升；往返次数越多收益越大）。

## 红线（Stop rules）

- 若某个中转 buffer 在别处仍被复用（如两路混合时做无 bank 冲突的标量提取中转），**不要删**——逐个确认用途。
- 重载分发要覆盖所有实际 dtype 组合，漏一个会编译失败或走错实现。
- 改法 B：`Reduce` 结果在 lane0，必须 `Duplicate(dst,dst,mask)` 广播后才能逐元素用。**确需经 UB 中转**（如跨趟缓存、寄存器放不下）才保留往返，且 `store→broadcast-load` 之间**必须**加 `LocalMemBar<VEC_STORE,VEC_LOAD>`，否则实测 `maxAbsErr=inf`。本仓 CANN（cann-9.0.0-beta.2）`LocalMemBar`/`MemType` 在 `AscendC::Reg` 命名空间。

## 关联

- 源：外部 VF 策略 4（消除中间 UB staging）；案例：qli-arch35 消除 cast 中转 UB（改法 A）、hc-pre 逐行 softmax 寄存器内 Reduce+Duplicate（改法 B）、RoPE arch35 `HalfAlignVF`/`InterleaveModeVF` 寄存器内 rotate（改法 C；完整范式与可拷贝骨架见 `ascendc-performance-best-practices` 技能 rope 族 `vector_fused`）。
- 改法 A 全链推广的可拷贝骨架：`ascendc-performance-best-practices` VF 范例库 `register-resident-vector-chain`（integration_skeleton）。
- 相邻卡片：vec-02（Cast 本身的削减、fp16/bf16 Cast 融进 load/store）、vec-01（UB 层融合）、vec-06（广播装载一次）、vec-13（VF 尾块统一 mask）、vec-14（对齐拆分搬运）。
