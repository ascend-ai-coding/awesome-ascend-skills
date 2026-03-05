# genop 使用说明

在 ops-transformer 下用 genop 生成新算子目录与占位文件，保证与现有算子风格一致。**算子工程结构与开发流程以 CANN 官方「基于自定义算子工程的算子开发」为准**；genop 为工程内脚手架，生成后需按官方规范与本 skill 的 references 进行定制。

## 命令

```bash
bash build.sh --genop=op_class/op_name
```

- **op_class**：算子类别，如 `gmm`、`moe`、`ffn`、`attention`、`mc2`
- **op_name**：新算子名，如 `my_custom_op`（仅字母、数字、下划线）

示例：

```bash
bash build.sh --genop=gmm/my_custom_gmm_op
bash build.sh --genop=moe/my_custom_moe_op
```

## 生成结构

```
op_class/op_name/
├── CMakeLists.txt
├── op_host/
│   ├── op_name_def.cpp
│   └── op_name_tiling.cpp
├── op_kernel/
│   └── op_name.h
└── examples/
    └── test_aclnn_op_name.cpp
```

模板目录：`<ops-transformer-root>/scripts/opgen/template/add`。核心脚本：`build.sh`、`scripts/opgen/opgen_standalone.py`。

## 生成后定制

1. **op_host/*_def.cpp**：修改 Input/Output/Attr、AICore 配置。
2. **op_kernel/*.h**：实现计算逻辑与 dtype/量化分支。
3. **op_host/*_tiling.cpp**：调整 tiling 参数与 shape 处理。
4. **examples/**：编写或调整 test_aclnn_* 并验证。

## 常见问题

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| 目标目录已存在 | 同名目录已存在 | 换名或删除已有目录 |
| 找不到模板目录 | 缺少 scripts/opgen/template/add | 检查模板路径 |
| 算子类型包含无效字符 | op_class/op_name 含非法字符 | 仅用字母、数字、下划线 |

## 优点

- 目录与占位自动生成，减少手写错误。
- 与现有算子结构一致，便于对齐与维护。
- 新算子宜先 genop 再按 references 文档逐步填充与校验。
