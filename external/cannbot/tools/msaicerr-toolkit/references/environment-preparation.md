# msaicerr 工具环境准备

## 前提条件

已在 CANN 运行环境上安装 Toolkit 软件包。详细安装请参见《CANN 软件安装》。

## 步骤

1. **以 CANN 运行用户登录环境**。安装 CANN 软件后，使用 CANN 运行用户进行编译、运行时，需要以 CANN 运行用户登录环境。

2. **设置环境变量**：

    ```bash
    source ${INSTALL_DIR}/set_env.sh
    ```

    `${INSTALL_DIR}` 请替换为 CANN 软件安装后文件存储路径。以 root 用户安装为例，安装后文件默认存储路径为 `/usr/local/Ascend/cann`：

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

3. **进入工具目录**。使用 msaicerr 工具前，需先进入到 msaicerr.py 脚本所在的目录 `${install_path}/tools/msaicerr`：

    ```bash
    cd /usr/local/Ascend/cann/tools/msaicerr
    ```

## 检查是否就绪

```bash
bash scripts/preflight.sh
```

该脚本检查 python 版本、环境变量、msaicerr.py 是否存在，以及当前目录与待分析目录的路径关系。

## 常见环境问题

| 现象 | 排查方向 |
|------|---------|
| 找不到 msaicerr.py | 确认已安装 Toolkit 包；`ls ${INSTALL_DIR}/tools/msaicerr` 查看实际路径 |
| python 版本过低 | 工具要求 python3.7.5+，用 `python3 --version` 确认 |
| 未 source set_env.sh | 工具运行内置算子样例时依赖 CANN 环境变量，缺失会导致环境检查失败 |
| Ascend RC 形态 | 不支持，需改用其他定位手段 |
