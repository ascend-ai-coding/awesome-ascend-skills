# Ascend Model Migration Suite

完整的昇腾NPU模型迁移和训练套件，提供从环境搭建到模型训练的端到端工作流。

## 📋 功能概览

本套件提供以下核心能力：

- **环境搭建**：在昇腾NPU容器中安装MMLab系列库（mmcv、mmdetection、mmdetection3d、detectron2）
- **模型迁移**：将开源模型迁移到昇腾NPU平台，应用NPU适配补丁
- **模型训练**：在昇腾NPU上启动和监控模型训练
- **SSH连接管理**：支持远程服务器连接和容器操作

## 🚀 快速开始

### 前置条件

- 昇腾NPU服务器（已安装CANN和torch_npu）
- Docker容器环境
- Conda环境管理器
- 网络连接（可访问GitHub或配置代理）

### 完整工作流

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  准备DrivingSDK │ ──► │  安装MMLab组件  │ ──► │   模型迁移      │ ──► │   模型训练      │
│                 │     │                 │     │                 │     │                 │
│  克隆仓库       │     │  mmcv/mmdet/    │     │  克隆源码       │     │  执行训练脚本   │
│  配置环境       │     │  mmdet3d等      │     │  应用patch      │     │  监控训练进度   │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 📦 套件结构

```
ascend-model-migration/
├── README.md                           # 本文档
├── SKILL.md                            # 主skill文档
├── ssh-connection/                     # SSH连接管理
│   ├── SKILL.md
│   ├── connect/                        # 建立SSH连接
│   ├── debug/                          # 远程调试
│   ├── deploy/                         # 远程部署
│   ├── long-task/                      # 长时间任务管理
│   └── tunnel/                         # SSH隧道
├── model-migration/                    # 模型迁移
│   └── SKILL.md
├── model-training/                     # 模型训练
│   └── SKILL.md
└── ascend-mmlab-install-suite/         # MMLab环境安装
    ├── SKILL.md
    ├── mmcv/                           # mmcv-full安装
    ├── mmdet/                          # mmdetection安装
    ├── mmdet3d/                        # mmdetection3d安装
    └── detectron2/                     # detectron2安装
```

## 🔧 核心功能

### 1. 环境搭建（ascend-mmlab-install-suite）

在昇腾NPU容器中安装OpenMMLab系列库：

| 组件 | 版本 | 说明 |
|------|------|------|
| mmcv-full | 1.7.2 | 计算机视觉基础库（含NPU算子） |
| mmdetection | 2.24.0 | 目标检测库 |
| mmsegmentation | 0.30.0 | 语义分割库 |
| mmdetection3d | 1.0.0rc4 | 3D目标检测库 |
| detectron2 | 0.6 | Facebook检测分割库 |

**关键特性**：
- ✅ 源码编译安装，支持NPU算子
- ✅ 自动应用DrivingSDK提供的NPU适配补丁
- ✅ 验证NPU兼容性，确保无CUDA依赖
- ✅ 支持代理配置，解决网络问题

**使用示例**：
```bash
# 1. 克隆DrivingSDK
git clone https://gitcode.com/Ascend/DrivingSDK

# 2. 安装前置依赖
pip install -r DrivingSDK/model_examples/BEVFormer/requirements.txt

# 3. 安装mmcv-full（含NPU算子）
git clone -b 1.x https://github.com/open-mmlab/mmcv.git
cd mmcv
cp DrivingSDK/model_examples/BEVFormer/mmcv_config.patch .
git apply --reject mmcv_config.patch
MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop

# 4. 安装其他组件
# mmdetection、mmdetection3d、detectron2...
```

### 2. 模型迁移（model-migration）

将开源模型迁移到昇腾NPU平台：

**支持的模型**：
| 模型 | 状态 | 说明 |
|------|------|------|
| BEVFormer | ✅ 已支持 | 3D目标检测模型 |
| 更多模型 | 🔄 开发中 | - |

**迁移流程**：
```bash
# 1. 进入DrivingSDK的model_examples目录
cd DrivingSDK/model_examples/BEVFormer

# 2. 克隆模型仓库
git clone https://github.com/fundamentalvision/BEVFormer.git

# 3. 复制patch到模型目录
cp bev_former_config.patch BEVFormer/

# 4. 应用NPU适配补丁
cd BEVFormer
git checkout 66b65f3a1f58caf0507cb2a971b9c0e7f842376c
git apply --reject --whitespace=fix bev_former_config.patch

# 5. 设置数据集和权重链接
mkdir -p data ckpts
ln -sf /path/to/nuscenes data/nuscenes
ln -sf /path/to/can_bus data/can_bus
ln -sf /path/to/r101_dcn_fcos3d_pretrain.pth ckpts/
```

**关键特性**：
- ✅ 在DrivingSDK目录结构下操作，patch文件已就绪
- ✅ 自动应用NPU适配补丁
- ✅ 支持数据集和权重软链接
- ✅ 训练脚本已包含在DrivingSDK中

### 3. 模型训练（model-training）

在昇腾NPU上启动和监控训练：

**训练模式**：
| 模式 | 说明 | 适用场景 |
|------|------|----------|
| FP32性能训练 | FP32精度，快速性能测试 | 性能验证 |
| FP16性能训练 | FP16混合精度，快速测试 | 性能验证 |
| 完整精度训练 | 24 epochs，完整训练 | 模型训练 |

**使用示例**：
```bash
# 返回model_examples/BEVFormer目录
cd ..

# 执行训练脚本
bash test/train_performance_8p_base_fp32.sh --batch-size=1 --num-npu=8

# 监控训练进度
tail -f test/output/train_performance_8p_base_fp32.log
```

**关键特性**：
- ✅ 支持多NPU并行训练
- ✅ 自动设置NPU环境变量
- ✅ 实时监控训练进度
- ✅ 支持性能和精度两种训练模式

### 4. SSH连接管理（ssh-connection）

管理远程服务器和容器连接：

**功能模块**：
- **connect**：建立SSH连接到服务器或容器
- **debug**：远程调试支持
- **deploy**：远程部署工具
- **long-task**：长时间任务管理
- **tunnel**：SSH隧道和端口转发

## 🌐 网络策略

本套件采用以下网络策略：

| 情况 | 策略 |
|------|------|
| 服务器可访问GitHub | 直接git clone |
| 网络超时 | 配置代理或本地克隆后scp传输 |
| 有代理 | `export http_proxy=http://proxy:port` |

**重要**：所有GitHub仓库克隆都使用GitHub官方链接，不使用镜像站点。如果网络超时，优先配置代理。

## 📚 DrivingSDK资源

**DrivingSDK仓库地址**：`https://gitcode.com/Ascend/DrivingSDK`

DrivingSDK包含：
- ✅ 所有模型的NPU适配补丁
- ✅ 训练脚本和配置文件
- ✅ 前置依赖requirements文件
- ✅ 模型示例和文档

## 🔍 验证和检查

### NPU兼容性验证

安装完成后，验证NPU兼容性：

```bash
python -c "import site; print([p for p in site.getsitepackages() if 'cuda' in p.lower()])"
```

**预期结果**：输出空列表 `[]`，表示无CUDA依赖。

### 组件安装验证

```bash
python -c "
import mmcv
import mmdet
import mmseg
import mmdet3d
import detectron2

print('mmcv:', mmcv.__version__)
print('mmdet:', mmdet.__version__)
print('mmseg:', mmseg.__version__)
print('mmdet3d:', mmdet3d.__version__)
print('detectron2: installed')
"
```

## 🛠️ 故障排除

### 常见问题

**1. SSH连接失败**
- 检查服务器IP、用户名、密码/密钥
- 确认容器名称正确
- 检查网络连通性

**2. NPU设备未找到**
- 检查环境变量设置
- 运行 `npu-smi info` 查看NPU状态
- 确认CANN已正确安装

**3. CUDA路径冲突**
- 重新安装NPU版本的组件
- 验证无CUDA依赖

**4. Patch应用失败**
- 确认切换到正确的commit
- 检查patch文件版本
- 查看 `.rej` 文件手动处理冲突

**5. 网络超时**
- 配置代理：`export http_proxy=http://proxy:port`
- 或本地克隆后scp传输

## 📖 详细文档

- **主文档**：`SKILL.md` - 完整工作流和决策指引
- **环境搭建**：`ascend-mmlab-install-suite/SKILL.md`
- **模型迁移**：`model-migration/SKILL.md`
- **模型训练**：`model-training/SKILL.md`
- **SSH连接**：`ssh-connection/SKILL.md`

## 🤝 贡献

欢迎贡献新的模型支持或改进现有功能。请参考各个skill文档中的详细说明。

## 📄 许可证

本套件遵循DrivingSDK的许可证协议。

## 🔗 相关资源

- **DrivingSDK仓库**：https://gitcode.com/Ascend/DrivingSDK
- **昇腾NPU文档**：[华为昇腾官方文档](https://www.hiascend.com/)
- **OpenMMLab**：https://openmmlab.com/

---

**最后更新**：2026-03-15
