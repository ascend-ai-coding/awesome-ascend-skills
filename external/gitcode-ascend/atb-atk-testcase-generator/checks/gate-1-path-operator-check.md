# Gate 1 路径与算子确认

## 输入

- `ATB_REPO_PATH`
- `ATB_KNOWLEDGE_PATH`
- `ATK_PATH`
- `OpName`

## 命令

```bash
test -d "$ATB_REPO_PATH" && echo "ATB_REPO_PATH OK" || echo "ATB_REPO_PATH MISSING"
test -d "$ATB_KNOWLEDGE_PATH" && echo "ATB_KNOWLEDGE_PATH OK" || echo "ATB_KNOWLEDGE_PATH MISSING"
python3 -c "import atk" 2>/dev/null && echo "ATK OK" || echo "ATK MISSING"
rg "\"<OpName>Operation\"" "${ATB_KNOWLEDGE_PATH}/atk_test/atk_cida_atb/ascend/atb_op/src/Operations.cpp"
mkdir -p "${ATB_KNOWLEDGE_PATH}/atk_test/atk_cida_atb/atb/infer/<OpName>Operation/"
```

## 通过标准

- [ ] 路径均有效
- [ ] `import atk` 成功
- [ ] `Operations.cpp` 已注册算子
- [ ] 输出目录已创建

## 失败回流

- 任一路径无效：要求用户补充或修正路径后重试 Gate 1
- `import atk` 失败：先修复 ATK 环境，再重跑 Gate 1
- 算子未注册：回流到算子注册步骤，完成后重跑 Gate 1
