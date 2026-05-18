# Environment Checklist Template
# 模型特定环境检查清单模板

## Step 1: 信息提取与路径确认

从用户描述中提取以下信息：
- **工作目录** (work_path): 项目文件和脚本的存放位置
- **原始模型路径** (weights_path): 原始模型权重文件的位置
- **模型名称** (model_name): 要量化的模型名称
- **量化格式** (quant_format): 如 W8A8, W4A8, W8A8S 等
- **特征** (features): 可选，如 QuaRot, AWQ 等

**必须与用户确认**：
1. 工作目录 (work_path)
2. 保存路径 (save_path) - 量化后模型的输出位置

## Step 2: 检查 README 中的环境要求

在 README 中查找以下关键信息：

| 关键词 | 含义 |
|--------|------|
| `pip install` | 需要额外安装的依赖包 |
| `transformers` | 特定的 transformers 版本要求 |
| `export` | 需要设置的环境变量 |
| `requirements` | 环境依赖文件 |
| `version` | 版本要求 |
| `conda` | Conda 环境配置 |

## Step 3: 检查模型特定的 README

```bash
# 查找该模型的 example 目录
ls -la {work_path}/msmodelslim/example/ | grep -i "{model_name}"

# 读取特定模型的 README（最重要的环境信息来源）
cat {work_path}/msmodelslim/example/{model_name}/README.md
```

## Step 4: 常见环境检查项

```bash
# 检查 transformers 版本要求
grep -i "transformers" {work_path}/msmodelslim/example/{model_name}/README.md

# 检查额外依赖
grep -i "pip install\|requirements" {work_path}/msmodelslim/example/{model_name}/README.md

# 检查环境变量
grep -i "export\|ENV" {work_path}/msmodelslim/example/{model_name}/README.md
```

## Step 5: 应用环境更改

如果 README 中有特殊要求，按以下方式处理：

```bash
# 1. 安装特定版本的 transformers（如需要）
pip install transformers=={version}

# 2. 安装额外依赖（如需要）
pip install {extra_dependencies}

# 3. 设置环境变量（如需要）
export {ENV_VAR}={value}

# 4. 验证安装
python -c "import transformers; print(transformers.__version__)"
```
