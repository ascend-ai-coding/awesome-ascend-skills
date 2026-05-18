# Environment Check Commands
# 昇腾 NPU 量化前置环境检查

# 1. 检查 NPU driver
npu-smi info -v

# 2. 检查 CANN 版本
python -c "import acl; print(acl.__version__)" 2>/dev/null || echo "CANN not installed"

# 3. 检查 NPU 可见性
echo $ASCEND_VISIBLE_DEVICES

# 4. 检查 Docker 容器中的 NPU 设备挂载
ls -la /usr/local/dcmi/

# 5. 检查 transformers 版本要求
grep -i "transformers" {work_path}/msmodelslim/example/{model_name}/README.md

# 6. 检查额外依赖
grep -i "pip install\|requirements" {work_path}/msmodelslim/example/{model_name}/README.md

# 7. 检查环境变量
grep -i "export\|ENV" {work_path}/msmodelslim/example/{model_name}/README.md

# 8. 验证安装
python -c "import transformers; print(transformers.__version__)"
