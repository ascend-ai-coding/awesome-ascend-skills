#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
readonly TEST_DIR="${TEST_DIR:-$HOME/diffusers-test}"
readonly CONDA_ENV="${CONDA_ENV:-torch2.8_py310}"

echo "=== Diffusers Weight Prep Validation ==="
echo "Test directory: $TEST_DIR"
echo "Conda environment: $CONDA_ENV"
echo ""

source ~/.bashrc
conda activate "$CONDA_ENV"

mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

echo "Step 1: Check required packages"
echo "--------------------------------"
python3 -c "import huggingface_hub; print(f'huggingface_hub: {huggingface_hub.__version__}')" || {
    echo "Installing huggingface_hub..."
    pip install -U huggingface_hub
}

python3 -c "import modelscope; print(f'modelscope: {modelscope.__version__}')" || {
    echo "Installing modelscope..."
    pip install modelscope
}

python3 -c "import torch; print(f'torch: {torch.__version__}')" || {
    echo "Error: PyTorch not found"
    exit 1
}

echo ""
echo "Step 2: Test download_weights.py (dry-run mode)"
echo "------------------------------------------------"

echo "Testing HuggingFace download (dry-run)..."
python3 "$SCRIPT_DIR/download_weights.py" hf Qwen/Qwen-Image-2512 --dry-run

echo ""
echo "Testing ModelScope download (dry-run)..."
python3 "$SCRIPT_DIR/download_weights.py" modelscope Wan-AI/Wan2.2-T2V-A14B --dry-run

echo ""
echo "Step 3: Test generate_fake_weights.py"
echo "--------------------------------------"

mkdir -p test_metadata/transformer test_metadata/scheduler

cat > test_metadata/model_index.json << 'EOF'
{
  "_class_name": "DiffusionPipeline",
  "_diffusers_version": "0.36.0",
  "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
  "transformer": ["diffusers", "FluxTransformer2DModel"]
}
EOF

cat > test_metadata/scheduler/scheduler_config.json << 'EOF'
{
  "_class_name": "FlowMatchEulerDiscreteScheduler",
  "_diffusers_version": "0.36.0"
}
EOF

cat > test_metadata/transformer/config.json << 'EOF'
{
  "_class_name": "FluxTransformer2DModel",
  "_diffusers_version": "0.36.0",
  "attention_head_dim": 64,
  "in_channels": 16,
  "joint_attention_dim": 768,
  "num_attention_heads": 4,
  "num_layers": 1,
  "num_single_layers": 1,
  "patch_size": 1,
  "pooled_projection_dim": 768
}
EOF

echo "Generating fake weights from test metadata..."
python3 "$SCRIPT_DIR/generate_fake_weights.py" from-local test_metadata -o test_fake_weights

echo ""
echo "Verifying generated weights..."
python3 << 'PYEOF'
import os

output_dir = "test_fake_weights"
required = ["model_index.json"]
found = []
for root, dirs, files in os.walk(output_dir):
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), output_dir)
        found.append(rel)
        print(f"  {rel}")

assert "model_index.json" in found, "model_index.json missing"
assert any("scheduler" in f for f in found), "scheduler files missing"
assert any("transformer" in f for f in found), "transformer files missing"
print("\n✅ Fake weights validation passed!")
PYEOF

echo ""
echo "Step 4: Cleanup"
echo "---------------"
rm -rf test_fake_weights test_metadata

echo ""
echo "=== All validations passed! ==="
