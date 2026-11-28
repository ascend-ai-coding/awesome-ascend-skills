# MatMul 常量折叠策略（TCubeTiling → 编译期 MatmulApiStaticTiling）

> 适用：AscendC Matmul API；典型目标是 DAV_3510（A5），两代 API 语义一致，字段/签名以本地 `$ASCEND_HOME_PATH/include/` 与 `ascendc-docs-search` 核验为准。
> 关联诊断：本技能负责策略与 tiling 修正建议；`aic_scalar_time` 的 bound 判定与适用门禁见 ascendc-perf-optimize `references/single-core-pipeline/scalar.md`（`aic_scalar_ratio ≥ 0.15` 且 shape 可固化/可取上界时推荐实施）。

## 1. 优化目标

Kernel 使用 Matmul 高阶 API 时，运行时 `TCubeTiling` 由 host `MatmulApiTiling::GetTiling` 生成、经 tilingdata 传入、kernel 内逐字段解析，产生可观的 aic_scalar 负载（字段读取、地址与循环界计算）。把 tiling 折叠到编译期后：

- aic_scalar_time 显著下降（消除 tiling 解析与相关地址计算）；
- 编译器可基于常量 tiling 做循环展开/指令调度；
- 精度与计算结果不变（tiling 是调度参数，不参与数值计算）。

## 2. 架构概览：两条路径

```text
运行时路径（默认）:
  host: MatmulApiTiling::GetTiling(TCubeTiling&) → tilingdata
  kernel: Init(&tiling, pipe) → 每字段运行时解析 → aic_scalar 负载

常量折叠路径（本策略）:
  kernel: constexpr MatmulApiStaticTiling 作为第 5 个模板参数
          → tiling 在编译期成为类型的一部分，运行时零解析
          → Init/REGIST 传 nullptr，运行时仅补 SetOrgShape 等必要调用
```

## 3. 关键参数与使用契约

| 契约点 | 内容 |
|---|---|
| 第 5 个模板参数 | 用 **constexpr `MatmulApiStaticTiling`** 作为 Matmul 模板的第 5 个参数（替代运行时 `TCubeTiling` 路径）；tiling 常量随类型进入编译期 |
| 调用点 shape | **调用点 shape 必须全程 constexpr**——能写常量表达式的都写 constexpr；shape 可固化或可取上界的场景才适用本策略 |
| Init / REGIST | **Init / REGIST 传 nullptr**（不再传 `TCubeTiling*`），**之后再调用 SetOrgShape** |
| 调用时序 | **SetOrgShape 不能放在 REGIST 之前**——对象未完成注册/构造时调用成员设置是未定义行为 |
| singleCore | **singleCore 填上界**：singleCoreM/N/K 按 shape 上界填，覆盖最大 case，避免静态 tiling 小于实际需求 |
| 不可改项 | **isTrans / BiasType / 对象类型不可改**：这些参与模板决议，改动等于换了一份 tiling 语义，不是等价变换 |

## 4. 核心改造对照

```cpp
// 改造前：运行时 tiling
TCubeTiling tiling;                       // host 生成、tilingdata 传入
mm.Init(&tiling, &pipe);
mm.SetOrgShape(oriM, oriN, oriKa, oriKb, oriN2);

// 改造后：编译期常量折叠
// 1) 第 5 个模板参数 = constexpr MatmulApiStaticTiling（singleCore 填上界，shape 全程 constexpr）
// 2) Init/REGIST 传 nullptr，之后再 SetOrgShape（顺序不可颠倒）
mm.Init(nullptr, &pipe);
mm.SetOrgShape(oriM, oriN, oriKa, oriKb, oriN2);
```

## 5. 边界与精度注意

1. **尾块维度不可常量化**：尾块依赖运行期实际 shape，编译期不可知；尾块路径保留运行时处理（对齐 padding 或运行时分支），不得为折叠而跳过尾块逻辑。
2. **动态 shape 算子不适用**：shape 无法固化/取上界的调用点保持运行时 `TCubeTiling` 路径；只有常量折叠条件的调用点切换。
3. **精度无关性核对**：常量折叠不改变矩阵乘计算语义；但须确认 isTrans 语义、BiasType、量化随路参数 dtype 与原实现一致（这些是模板参数，改错即换语义，精度问题会直接暴露）。
4. **不得改变对象类型**：静态 tiling 类型与 Matmul 对象类型绑定，替换类型 ≠ 等价优化。
5. 改动后跑精度用例（逐位/容差按原口径）；`aic_scalar_time` 收益以完整调用 msprof Task Duration 与 scalar 分段核验，不以单点推断。

## 6. 实施常见问题

| 现象 | 根因 | 处理 |
|---|---|---|
| 编译报 tiling 非常量 | 调用点 shape 未全程 constexpr | 把 shape 及派生量提升为 constexpr |
| SetOrgShape 崩溃/未定义行为 | 放在 REGIST 之前 | 调整为 REGIST（nullptr）→ SetOrgShape 顺序 |
| 大 shape 结果错 | singleCore 未填上界，静态 tiling 覆盖不足 | singleCoreM/N/K 按 shape 上界重填 |
| 尾块结果错 | 尾块维度被强行常量化 | 尾块回退运行时路径 |
