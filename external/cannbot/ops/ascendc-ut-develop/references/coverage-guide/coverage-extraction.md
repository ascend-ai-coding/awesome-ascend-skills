# 覆盖率提取指南

## 判断覆盖率类型

编译后需要先判断是**全局覆盖率**还是**单算子覆盖率**：

```bash
# 进入覆盖率目录
cd <repo>/build/tests/ut/cov_report/cpp_utest

# 查看覆盖率报告包含的文件路径
lcov --list ops.info_filtered | head -50
```

| 覆盖率类型 | 文件路径特征 | 处理方式 |
|-----------|-------------|---------|
| **单算子覆盖率** | 仅包含当前算子路径（如 `*/math/abs/*`） | 直接使用，无需提取 |
| **全局覆盖率** | 包含多个算子路径（如 `*/math/abs/*`, `*/math/add/*`） | 需要使用 `lcov --extract` 提取 |

---

## 提取单算子覆盖率

如果编译输出的是全局覆盖率，执行以下命令提取：

```bash
lcov --extract ops.info_filtered \
    "*/<category>/<op>/op_api/*" \
    "*/<category>/<op>/op_host/*" \
    "*/<category>/<op>/op_kernel/*" \
    "*/<category>/<op>/op_kernel_aicpu/*" \
    "*/<category>/<op>/op_tiling/*" \
    -o /tmp/<op>_cov.info

# 查看摘要
lcov --summary /tmp/<op>_cov.info
```

> **注意**：根据算子实际支持的层级选择对应的路径，不需要的层级可以省略。

---

## 覆盖率目标

| 场景 | 行覆盖率 | 函数覆盖率 | 分支覆盖率 |
|------|---------|-----------|-----------|
| 最低标准 | ≥ 80% | ≥ 80% | ≥ 80%（推荐） |
| 推荐标准 | ≥ 90% | ≥ 90% | ≥ 85% |

---

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 覆盖率很低（<10%） | 看的是全局覆盖率 | 判断类型后使用 `lcov --extract` 提取 |
| 某文件覆盖率 0% | cmake或链接问题 | 检查 CMakeLists.txt，用 `lcov --list` 查看详情 |
| InferDataType 覆盖率 0% | infershape 测试不调用 infer_data_type | 创建单独测试或接受该部分为 0% |
| 提取后覆盖率仍为 0% | 路径模式不匹配 | 检查路径是否正确，确保层级目录存在 |

---

## 生成 HTML 报告

```bash
genhtml /tmp/<op>_cov.info -o /tmp/<op>_cov_html --title "<op> Operator Coverage"
```

---

## 覆盖率提升策略

| 缺口类型 | 补充策略 |
|---------|---------|
| 异常分支 | 添加 ACLNN_ERR_*/GRAPH_FAILED 用例 |
| dtype分支 | 确保每个 dtype 都有测试 |
| 边界条件 | 添加空tensor、大shape用例 |
| 格式分支 | 添加各 format 的测试用例 |