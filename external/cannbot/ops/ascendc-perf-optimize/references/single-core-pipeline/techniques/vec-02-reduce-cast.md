# 技术卡片 vec-02：减少 / 批量 Cast

- **适用代际**：通用方法（arch22 / arch35）；具体 API、指标和阈值按目标核验，示例或实测仅代表所标平台。 RegBase / MicroAPI / VF 部分仅适用于 arch35。
- **适用 bound**：VEC
- **一句话**：消除 `fp16→fp32→算→fp16` 的往返 Cast，或把逐元素 Cast 合并成大粒度批量 Cast。

## 何时用（诊断特征）

- 从 trace 提取 VECTOR 泳道事件，**类型转换指令（`vcvt_f2f`/`vcvt_f2s`/`vcvt_s2f`）占比 >20%**。
- 计算本身只有 1–2 条指令，却夹着一进一出两次 Cast——Cast 可占总 VEC 时间 30–50%。
- `aiv_vec_ratio` 高，且 VECTOR 忙碌主要来自转换而非有效算术。

## 原理

`fp16 → Cast fp32 → 计算 → Cast fp16` 的往返转换在计算量小的算子里是隐性大头。若精度标准允许全链路同精度，或多次 Cast 可合并为一次大粒度操作，就能直接砍掉这部分 VECTOR 占用。

## 怎么改（before / after）

```cpp
// before：每元素一进一出两次 Cast
Cast(f32Buf, f16In,  RoundMode::CAST_NONE, len);   // fp16 -> fp32
Add (f32Buf, f32Buf, f32Bias, len);                // 计算
Cast(f16Out, f32Buf, RoundMode::CAST_RINT, len);   // fp32 -> fp16

// after A：精度允许时全链路同精度，删掉往返 Cast
Add(f16Out, f16In, f16Bias, len);

// after B：必须 fp32 精度时，把 Cast 合并到唯一使用点、只做一次大粒度转换
Cast(f32Buf, f16In, RoundMode::CAST_NONE, len);    // 批量一次
Add (f32Buf, f32Buf, f32Bias, len);
Cast(f16Out, f32Buf, RoundMode::CAST_RINT, len);
```

**after C（仅 arch35）：寄存器级 VF，把 Cast 融进 load/store，不占独立 VECTOR 指令、不落 fp32 中转 buffer**
- fp32-in-register 的 VF 里，fp16/bf16 的升/降精度可与搬运融合：load 用 B16 UNPACK + Cast 一步升 fp32，store 用 Cast + B32 PACK 一步降回，无独立 Cast pass、无 fp32 中转 UB。计算主体一份对 fp16/bf16/fp32 同构（dtype 差异被 `if constexpr` 吸收）。
```cpp
// load：DIST_UNPACK_B16 搬入即 Cast 升 fp32（half/bf16），fp32 直接 LoadAlign
Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(xBf16, inUb + off);
Cast<float, bfloat16_t, castTraitB162B32>(vregIn, xBf16, preg);
// ... 全程 RegTensor<float> 计算 ...
// store：Cast 降回 + DIST_PACK_B32 搬出（bf16 用 CAST_RINT）
Cast<bfloat16_t, float, castTraitB322B16>(yBf16, vregOut, preg);
Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(outUb + off, yBf16, preg);
```

## 性能数据

msprof `aiv_time`：省掉的 Cast 指令占比即为直接收益。**待实测**（预期：Cast 占 VEC 30–50% 的场景，去掉往返可省该占比）。

## 红线（Stop rules）

- **删 Cast 前确认精度标准**：全链路 fp16 会降低中间精度，须过 `func_cases` 精度回归（各 dtype 相对阈值 fp16 2⁻¹⁰ / bf16 2⁻⁷ / fp32 2⁻¹³）后才能保留。
- 累加类（reduction、softmax 分母）对中间精度敏感，通常仍需 fp32 中间累加，只能走 after B 的「合并 Cast」而非「删 Cast」。
- 降 cast 位宽（如 int32→int16）前须确认值域确实落在目标类型范围内。

## 关联

- 源：`../vec.md` 策略 2（减少类型转换）；案例：RoPE arch35 `LoadTwoTensorForDtypeT`/`StoreOneTensorForDtypeT` 把 B16↔F32 Cast 融进 load/store（after C；详见 `ascendc-performance-best-practices` 技能 rope 族 `vector_fused`）。
- 相邻卡片：vec-05（连 pre-cast 中转 UB 一并消除，寄存器内就近 cast）。
