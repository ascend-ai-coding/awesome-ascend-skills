# Docker Commands
# 容器化部署命令

# 列出运行中的容器
docker ps -a --filter "name=msmodelslim"

# 查看容器状态
docker ps --filter "status=running" --format "{{.Names}}"

# 创建新容器
docker run -it -u root --name msmodelslim-{timestamp} \
    --network=host \
    --privileged=true \
    --shm-size=500g \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    --entrypoint=bash \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v {work_path}:{work_path} \
    -v {weights_path}:{weights_path} \
    {image_id}

# 进入容器
docker exec -it {container_name} bash

# 在容器中安装 msmodelslim
cd {work_path}/msmodelslim
bash install.sh
