# Dynamic Query Commands
# 动态查询量化解决方案

# 列出所有 YAML 配置
ls {work_path}/msmodelslim/lab_practice/*/

# 查找特定模型的配置
ls {work_path}/msmodelslim/lab_practice/*/ | grep -i "{model_name}"

# 查找示例脚本
ls {work_path}/msmodelslim/example/*/quant_*.py

# 读取 config.ini 确认模型适配器
cat {work_path}/msmodelslim/config/config.ini

# 从 config.ini 提取模型组
grep "^\[" {work_path}/msmodelslim/config/config.ini | sed 's/\[//g' | sed 's/\]//g'

# 从 const.py 提取 QuantType 枚举值
grep -A 10 "class QuantType" {work_path}/msmodelslim/msmodelslim/core/const.py

# 查看特定模型的适配器配置
grep -A 5 "\[ModelAdapter\]" {work_path}/msmodelslim/config/config.ini
