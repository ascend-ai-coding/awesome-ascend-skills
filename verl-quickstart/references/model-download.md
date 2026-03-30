# 模型下载与路径规范（参考）

## 路径规范

- `MODEL_PATH` 必须指向带具体模型名的目录，例如：
  - `.../weights/Qwen/Qwen3-0.6B`
  - `.../weights/Qwen/Qwen2.5-3B-Instruct`
- 不要只指向系列父目录（例如 `.../weights/Qwen`）。

## ModelScope 示例

```bash
# 若未安装：pip install -U modelscope
export WEIGHTS_ROOT="${WEIGHTS_ROOT:-$HOME/weights}"
export MS_MODEL="Qwen/Qwen3-0.6B"
export MODEL_DIR="$WEIGHTS_ROOT/Qwen/Qwen3-0.6B"
mkdir -p "$MODEL_DIR"

modelscope download --model "$MS_MODEL" --local_dir "$MODEL_DIR"

export MODEL_PATH="$(readlink -f "$MODEL_DIR")"
ls -la "$MODEL_PATH/config.json"
```
