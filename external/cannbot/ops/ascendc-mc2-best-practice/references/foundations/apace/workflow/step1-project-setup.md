# Step 1: Project Setup

> **定位**：建立算子工程骨架、校验 CANN 内置 apace 事实源与环境、登记只读参考点。Step 1 只做环境验证与目录创建；不核对接口事实、不判定路线、不创建算子实现文件。
>
> 父级流程定义以 `plugins-official/ops-direct-invoke/AGENTS.md` 为准；本文件只补充 apace 场景的差异化要点，不重写父级规则。plugin 场景下本步骤对应 plugin Step 1（环境检查），7 步门禁细则以 [`../workflow_integration.md`](../workflow_integration.md) 为准。

## 1. 环境检查（门禁）

### apace 场景额外校验

校验项表与路径实测纪律以 [`../workflow_integration.md`](../workflow_integration.md) Step 1「apace 场景额外校验」为唯一事实源，本文件不重复（防止双写漂移）。其中"CANN 内置 apace"一项的物理路径形态以 `test -d` 实测登记为准（随 CANN 版本漂移）。

### 门禁

环境检查全部通过 → 进入 Step 2。任何一项失败禁止进入 Step 2。

## 2. 项目目录创建

创建算子根目录与最小空目录结构；kernel、host、脚本与测试文件均不在本步生成，留待 Step 3 的 PLAN 确定后再创建。

```
operators/{OpName}/            ← camelCase（对齐 CANN 算子注册名）
├── docs/
│   ├── DESIGN.md               # Step 3 输出
│   └── PLAN.md                 # Step 3 输出
├── kernel/                     ← [MODIFY] 本算子 kernel 头文件
├── src/                        ← [MODIFY] host 侧
├── scripts/                    ← [MODIFY] gen_data.py / verify_result.py / parse_perf.py
├── data/
│   ├── input/
│   ├── golden/
│   └── output/
├── CMakeLists.txt
└── cases.csv
```

> 共享层 `block/` `tiling/` `basic/` `utils/` **不复制、不出现在算子目录**——经 CMake `-I` 直引 CANN 内置路径。参考 kernel（`quant_matmul_mx_kernel.h`、`comm_channel_builder.h` 等）同样直引。

## 3. 只读参考点

| 参考点 | 路径 | 用途 |
|--------|------|------|
| CANN 内置 apace 框架 | 实测定位（`test -d`；两种已验证形态：`opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/apace/`、`vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/apace/`） | 编译事实源，CMake 直引 |
| ops-transformer 仓（可选） | 经 `scripts/fetch_apace.sh` 获取 | 跟踪 master 新特性 / 核对最新契约，使用前必须与内置版本 diff 校验 |
| 参考算子 | `kernel/all_to_all_quant_matmul/`、`kernel/all_gather_quant_matmul/` | 实现模式参考 |

## 4. 不做的事

- 不核对 apace 框架的接口组合候选（Step 2 职责）
- 不选择实现路线（Step 3 职责）
- 不创建场景实现文件（Step 4 职责）
- 不提前编写公式、Tiling、golden 或任何固定工程模板
