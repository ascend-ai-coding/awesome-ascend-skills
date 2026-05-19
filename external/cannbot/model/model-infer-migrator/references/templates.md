# 框架适配模板

替换 `{model_name}`（小写下划线）和 `{ModelName}`（驼峰）后使用。

## infer.py

```python
import os
import argparse
import logging
from runner_{model_name} import {ModelName}Runner
from executor.utils import read_yaml
from executor.utils.data_utils import generate_prompt
from executor.utils.common_utils import check_common_parallel_settings
from models.model_setting import update_vars

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(description="llm run parameters")
    parser.add_argument('--yaml_file_path', type=str, help="inference configurations")
    parser_args = parser.parse_args()
    return parser_args


def run_{model_name}(runner_settings):
    preset_prompts, _ = generate_prompt(runner_settings)
    model_runner = {ModelName}Runner(runner_settings)
    model_runner.init_model()
    model_runner.model_generate(preset_prompts, warm_up=True)
    model_runner.model_generate(preset_prompts)


if __name__ == "__main__":
    args = parse_args()
    yaml_file_path = args.yaml_file_path
    runner_settings = read_yaml(yaml_file_path)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    check_common_parallel_settings(world_size, runner_settings)
    update_vars(world_size, runner_settings)
    logging.info(f"runner_settings is: {runner_settings}")
    run_{model_name}(runner_settings)
    logging.info("model run success")
```

模板包含 `warm_up=True` 预热调用，确保基线采集时编译开销不计入。部分仓库模型的 infer.py 未包含 warmup，适配时需确认加入。

## infer.sh

```bash
#!/bin/bash
SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
SET_ENV_ABS_PATH="${SCRIPT_PATH}/../../executor/scripts/set_env.sh"
FUNCTION_ABS_PATH="${SCRIPT_PATH}/../../executor/scripts/function.sh"
SET_ENV_ABS_PATH=$(realpath "${SET_ENV_ABS_PATH}")
FUNCTION_ABS_PATH=$(realpath "${FUNCTION_ABS_PATH}")

source ${SET_ENV_ABS_PATH}
source ${FUNCTION_ABS_PATH}

export MODEL_DIR=$(basename "$SCRIPT_PATH")
export YAML_PARENT_PATH="${SCRIPT_PATH}/config"
export YAML_FILE_NAME="{model_name}_{config}.yaml"
export YAML=${YAML_PARENT_PATH}/${YAML_FILE_NAME}

launch
```

`../../executor/scripts/` 要求模型目录与仓库根目录之间有两层。`cann-recipes-infer/models/{model_name}/` 和 `skill_test/{model_name}/` 均满足此条件。如果输出目录层级不同，需调整相对路径。

## YAML 配置

单卡基础配置，eager 模式：

```yaml
model_name: "{model_name}"
model_path: "{weights_path}"
exe_mode: "eager"
world_size: 1

model_config:
  enable_online_split_weight: True

data_config:
  dataset: "default"
  input_max_len: 1024
  max_new_tokens: 32
  batch_size: 1

parallel_config:
  attn_tp_size: 1
  moe_tp_size: 1
  lmhead_tp_size: 1
```
