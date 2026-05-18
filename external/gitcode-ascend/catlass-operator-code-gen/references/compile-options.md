# 编译选项配置

## 配置文件路径

`scripts/kernel/binary_config/ascendc_config.json`

## JSON 配置模板（Catlass 算子必需）

```json
{
    "name": "<op_name>",
    "compute_units": ["ascend910b"],
    "auto_sync": false,
    "impl_mode": "",
    "compile_options": {
        "ascend910b": [
            "-I<CATLASS_DIR>/include",
            "-DCATLASS_ARCH=2201"
        ]
    }
}
```

## 字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `name` | string | ✅ | 算子类型名称（与目录名一致，须含 `catlass` 子串） |
| `compute_units` | array | ✅ | 支持的 SOC 版本列表（`ascend910b`、`ascend950`、`ascend310p` 等） |
| `auto_sync` | bool/object | ✅ | 是否自动同步；支持按 SOC 版本差异化配置 |
| `impl_mode` | string | ❌ | 实现模式：`high_performance` / `high_precision` / `""` |
| `compile_options` | object | ✅ | 按 SOC 版本指定的编译选项列表 |

## Catlass 必需编译选项

| 选项 | 说明 |
|------|------|
| `-I<CATLASS_DIR>/include` | Catlass 头文件路径（**必需**） |
| `-DCATLASS_ARCH=2201` | 架构宏（Atlas A2 / ascend910b 为 2201，其他芯片参考 `catlass/README.md`） |

## 多 SOC 版本配置示例

```json
{
    "name": "grouped_matmul_catlass",
    "compute_units": ["ascend910b", "ascend950"],
    "auto_sync": false,
    "compile_options": {
        "ascend910b": [
            "-I<CATLASS_DIR>/include",
            "-DCATLASS_ARCH=2201"
        ],
        "ascend950": [
            "-I<CATLASS_DIR>/include",
            "-DCATLASS_ARCH=3510",
            "--cce-simd-vf-fusion=true"
        ]
    }
}
```

## 检查清单

- [ ] `name` 字段与算子目录名一致且含 `catlass` 子串
- [ ] `compute_units` 包含目标 SOC 版本
- [ ] `compile_options` 中包含 `-I<CATLASS_DIR>/include`
- [ ] `compile_options` 中包含 `-DCATLASS_ARCH=<对应架构号>`
- [ ] 多 SOC 版本时，每个版本都有对应的编译选项
- [ ] 将`<CATLASS_DIR>`替换为实际catlass路径
