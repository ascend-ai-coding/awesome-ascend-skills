# 环境依赖安装指南

本文档提供CANN安装所需的必需软件安装命令。

## 必需软件列表

CANN安装需要以下必需软件：
- Python 3
- pip

## 安装命令

### Ubuntu类系统

```bash
# 检查并安装Python
if ! command -v python3 &> /dev/null; then
    apt-get update
    apt-get install -y python3
fi

# 检查并安装pip
if ! command -v pip3 &> /dev/null; then
    apt-get update
    apt-get install -y python3-pip
fi
```

### CentOS类系统

```bash
# 检查并安装Python
if ! command -v python3 &> /dev/null; then
    yum install -y python3
fi

# 检查并安装pip
if ! command -v pip3 &> /dev/null; then
    yum install -y python3-pip
fi
```
## 验证安装

安装完成后，使用以下命令验证软件是否安装成功：

```bash
# 检查Python
python3 --version || python --version

# 检查pip
pip3 --version || pip --version
```

## 注意事项

1. **Python版本要求**：CANN支持Python 3.7.x - 3.11.4
2. **权限要求**：安装软件可能需要root权限或sudo权限
3. **网络要求**：确保系统可以访问软件源
4. **依赖关系**：某些软件可能需要额外的依赖包，安装过程中会自动处理

## 常见问题

### 1. 软件源无法访问
- 检查网络连接
- 配置国内镜像源（如清华源、阿里源等）

### 2. 权限不足
- 使用sudo命令
- 或切换到root用户

### 3. 软件版本过旧
- 某些系统默认软件源中的软件版本可能较旧
- 可以考虑添加第三方源或手动编译安装

## 参考链接

- Python官网：https://www.python.org/
