# 技术卡片 vec-11：LUT + 插值减计算量

- **适用代际**：通用方法（arch22 / arch35）；具体 API、指标和阈值按目标核验，示例或实测仅代表所标平台。
- **适用 bound**：VEC（深度）
- **一句话**：超越函数用查表+插值模式替代 Taylor 级数展开，从算法层面减少向量指令数。

## 何时用（诊断特征）

- **深度 VEC bound**：`aiv_vec_ratio` >90% 且算子是 elementwise/activation 类。
- trace 中超越函数指令（`vec_exp`/`vec_log`/`vec_tanh`）占比 >30%——微架构调整（DoubleBuffer/融合指令/Bank Align）已收益有限，瓶颈在计算量本身。
- 使用硬件超越函数（Exp/Log/Tanh/Sigmoid/GELU/Softmax），每元素需数十条 FMA。

## 原理

超越函数内部用 Taylor 级数展开，每个元素数十条 FMA，是计算量主源。查表（LUT）+ 插值用少量指令逼近同一函数值，直接减少向量指令数。CANN 的 `Exp<T, expandLevel, isReuseSrc>` 把 `expandLevel = 0` 设为内部查表+插值模式，无需手写查表逻辑。

## 怎么改（before / after）

```cpp
// before：高展开级（Taylor 项多，指令密集）
Exp<float, /*expandLevel=*/2, /*isReuseSrc=*/false>(dst, src, count);

// after：expandLevel=0 → 内部走 LUT + 插值，指令数下降
Exp<float, /*expandLevel=*/0, /*isReuseSrc=*/false>(dst, src, count);
// Log / Tanh 等超越函数同理：优先低 expandLevel 的查表模式
```

## 性能数据

msprof `aiv_time` + `aiv_vec_ratio`：查表模式减少每元素 FMA 指令数。**待实测**（预期：超越函数占 VEC >30% 时收益明显；深度 bound >90% 是主要适用场景）。

## 红线（Stop rules）

- 查表+插值是**精度换性能**：`expandLevel=0` 精度低于高展开级，须过 `func_cases` 精度回归（尤其 FP32 高精度路径）。
- 手工减 Taylor 项数（如 12→8）有精度风险，**不推荐**——优先用官方 `expandLevel` 参数而非改内部实现。
- 从 asc-devkit / 官方文档确认目标超越函数是否支持查表模式与对应参数，不凭记忆默写 API。

## 关联

- 源：`../vec.md` 策略 6（减少计算量）。
- 相邻卡片：深度 VEC bound 时，vec-11 是算法层手段；微架构层已见 vec-01/vec-02/vec-03。
