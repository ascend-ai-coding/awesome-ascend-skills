# Quantization Execution
# 量化执行命令

# 方式1: 一键量化（推荐）
msmodelslim quant \
    --model_path {weights_path} \
    --save_directory {save_path} \
    --w_bit {w_bit} \
    --a_bit {a_bit} \
    --device_type NPU \
    --model_type {model_type}

# 方式2: 使用 YAML 配置
msmodelslim quant --config_path {yaml_path}

# 方式3: 一键量化（无需 config_path）
msmodelslim quant \
    --model_path {weights_path} \
    --save_directory {save_path} \
    --w_bit 8 \
    --a_bit 8 \
    --device_type NPU \
    --model_type qwen2

# 安装特定版本的 transformers（如需要）
pip install transformers=={version}

# 安装额外依赖（如需要）
pip install {extra_dependencies}

# 设置环境变量（如需要）
export {ENV_VAR}={value}
