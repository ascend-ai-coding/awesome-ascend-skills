# msmodelslim Setup
# 检查/克隆 msmodelslim 项目

# 检查指定目录是否存在 msmodelslim 项目
ls -la {work_path}/msmodelslim 2>/dev/null || echo "NOT_FOUND"

# 克隆项目到工作目录
git clone https://gitcode.com/Ascend/msmodelslim.git {work_path}/msmodelslim
cd {work_path}/msmodelslim
bash install.sh
