#!/bin/bash
# Gate 命令速查（仅示例，不直接执行）

echo "[Gate1] 路径检查"
echo "test -d \"\$ATB_REPO_PATH\""
echo "python3 -c \"import atk\""

echo "[Gate2] 生成代表用例定义"
echo "atk case -f ATB_<OpName>_gen.yaml -p ./generator_<op>.py -dt 10 -l info"

echo "[Gate3] 精度执行"
echo "atk task -n node.yaml -c result/.../all_....json -p ./ --task accuracy -ap \$atb_path/common -l info"

echo "[Gate5] 性能执行"
echo "atk task -n node_perf.yaml -c result/.../all_....json -p ./ --task performance_device -ap \$atb_path/common -mt 10 -sp --save_data profile -l error"
