#!/bin/bash
set -e

# accu/perf 的 -ap 指向 ATB 公共目录（本模板用 .../atb/common）。默认与知识库布局一致：
#   ${ATB_KNOWLEDGE_PATH}/atk_test/atk_cida_atb/atb/
# 若需覆盖，可显式 export ATB_PATH=...
: "${ATB_KNOWLEDGE_PATH:?请设置 ATB_KNOWLEDGE_PATH，例如 export ATB_KNOWLEDGE_PATH=/path/to/atb_knowledge}"
ATB_PATH="${ATB_PATH:-$ATB_KNOWLEDGE_PATH/atk_test/atk_cida_atb/atb}"

# 用法:
#   bash run_<op>.sh gen
#   bash run_<op>.sh gen200
#   bash run_<op>.sh accu
#   bash run_<op>.sh perf

YAML="ATB_<OpName>_gen.yaml"
GEN_PLUGIN="./generator_<op>.py"
JSON="result/ATB_<OpName>_gen/json/all_ATB_<OpName>_gen.json"

CMD="${1:-gen}"
case "$CMD" in
  gen)
    atk case -f "$YAML" -p "$GEN_PLUGIN" -dt 10 -l info
    ;;
  gen200)
    atk case -f "$YAML" -p "$GEN_PLUGIN" -dt 200 -l error
    ;;
  accu)
    atk task -n node.yaml -c "$JSON" -p ./ --task accuracy -ap "$ATB_PATH/common" -l info
    ;;
  perf)
    atk task -n node_perf.yaml -c "$JSON" -p ./ --task performance_device -ap "$ATB_PATH/common" -mt 10 -sp --save_data profile -l error
    ;;
  *)
    echo "usage: bash run_<op>.sh {gen|gen200|accu|perf}"
    exit 1
    ;;
esac
