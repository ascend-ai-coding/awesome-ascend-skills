---
name: external-gitcode-ascend-catlass-operator-code-gen
description: 根据CATLASS算子设计文档生成算子工程交付件
original-name: catlass-operator-code-gen
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-08'
synced-commit: 155ac37bd169ddb89479af528297cfb2237400aa
license: UNKNOWN
---

# Catlass 算子代码生成

## 核心工作流

```
设计文档解析 → Catlass 实现参考选型 → op_host 生成 → op_kernel 生成 → test_aclnn 生成 → 质量验证
```

---

## 设计文档遵守规范（CRITICAL）

**设计文档是代码生成的唯一依据，须 100% 遵守。**

| 章节 | 必须遵守内容 | 验证方式 |
|------|-------------|---------|
| 算子名 | 含 `catlass` 子串，snake_case 目录名与 CamelCase 类名一致 | 目录、OpDef、test_aclnn 命名一致 |
| I/O 与 dtype | 输入输出 shape、dtype、format | OpDef 与 test_aclnn tensor 一致 |
| 核心组件 | ArchTag、BlockMmad、BlockEpilogue、BlockScheduler、Kernel 类型 | op_kernel 与 [custom-epilogue.md](references/custom-epilogue.md) 一致 |
| 参考 example | 指定的 catlass example 路径与选型理由 | 实现与选型一致（非整份粘贴） |
| TilingKey | 各分支对应 dtype/转置等 | Host `SetTilingKey` 与 `TILING_KEY_IS` 一一对应 |
| Workspace | 固定写法 | 见 [kernel-rules.md](references/kernel-rules.md) |

---

## 前置条件

| 检查项 | 说明 |
|--------|------|
| 设计文档 | 含算子名、I/O、dtype、转置、参考 example、Kernel/Block/Epilogue |
| 工程 | `OPS_PROJECT_ROOT` 已定，含 `build.sh`、`ops/` |
| Catlass | `<OPS_PROJECT_ROOT>/catlass/` 含 `include/`、`examples/`；缺则在工程根克隆 |

条件不明时**追问**用户。

---

## Catlass 实现参考选型

在 `catlass/examples/` 中按设计文档选定**实现参考**（读结构、抄组件组合思路，**不是**把 example 当 `ops/` 迁移源）：

1. 找与算子形态最接近的示例
2. 确定可复用的 Kernel/Block/Epilogue；**算子工程 op_kernel 须 Device 调用**
3. 将 tiling 常量、Workspace、`TILING_KEY_IS` 分支与 Host 侧对齐

---

## 端到端步骤

### Step 0：生成骨架（强制）

在 `OPS_PROJECT_ROOT`：`bash build.sh --genop=ops/<op_name>`

### Step 1：清理模板、配置编译选项

删无用模板；按 [compile-options.md](references/compile-options.md) 配置 `ascendc_config.json`

### Step 2：对照设计文档列清单

算子名、I/O、dtype、转置、参考路径、Kernel/Block/Epilogue、Workspace、TilingKey 分支

### Step 3：锁定实现参考

同上文「Catlass 实现参考选型」

### Step 4：写 op_host

生成 3 类 4 个文件：
- `*_tiling.h` / `*_tiling.cpp`：按 [tiling-rules.md](references/op_host/tiling-rules.md)
- `*_def.cpp`：按 [def-rules.md](references/op_host/def-rules.md)
- `*_infershape.cpp`：按 [infershape-rules.md](references/op_host/infershape-rules.md)

### Step 5：写 op_kernel

按 [kernel-rules.md](references/kernel-rules.md)：`GET_TILING_DATA`、`TILING_KEY_IS`、Device 调用、`#define K_MAX_SHAPE_DIM 0`、勿 `#include` tiling.h

### Step 6：写 test_aclnn

按 [example-rules.md](references/example-rules.md) 覆盖 `examples/test_aclnn_<op_name>.cpp`

### Step 7：验证

- 结构自检：[code-structure.md](references/code-structure.md)
- 编译与运行：**ascendc-operator-compile-debug**

---

## 常见陷阱

**NEVER**：跳过 `--genop`；tiling/infershape 并进 def；漏 `IMPL_OP_*` 注册；def 里 `SetTiling`/`SetInferShape`；目录名无 `catlass`；Kernel 内用 `if` 代替 TilingKey；op_kernel 用 `DeviceGemm`；op_kernel `#include` tiling.h；对 Gemm 仅用 shape 推断转置；忘记在 `ascendc_config.json` 中配置 Catlass 编译选项

**ALWAYS**：先 opgen 再覆盖；三文件分文件注册；严格按设计文档组件选型；在 `ascendc_config.json` 中配置 Catlass 编译选项；改算子后 **`--pkg` 安装再跑示例**

---

## references 索引

| 文件 | 内容 |
|------|------|
| [compile-options.md](references/compile-options.md) | 编译选项配置 |
| [code-structure.md](references/code-structure.md) | 目录树、文件职责索引 |
| [example-rules.md](references/example-rules.md) | test_aclnn 写法 |
| [tiling-rules.md](references/tiling-rules.md) | tiling.h / tiling.cpp |
| [def-rules.md](references/def-rules.md) | def.cpp |
| [infershape-rules.md](references/infershape-rules.md) | infershape.cpp |
| [kernel-rules.md](references/kernel-rules.md) | Catlass 依赖与 Device 调用 |
| [custom-epilogue.md](references/custom-epilogue.md) | 自定义 Tile Epilogue |

---

## 参考资料

| 文档/目录 | 用途 |
|-----------|------|
| catlass/examples/advanced/basic_matmul_aclnn | 工程化 op_host / op_kernel / 测试组织对照 |
| catlass/examples/03_matmul_add | MatmulEpilogue + BlockEpilogue 组合参考 |
| ascendc-operator-compile-debug | 编译、安装、跑示例 |
| catlass-operator-design | 设计文档 |
