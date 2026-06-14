#!/usr/bin/env python
"""inference.py — PatchCore 昇腾 NPU 端到端推理脚本（比赛提交用）

功能：
  1. 对 MVTec-AD 全部 15 个类别依次执行训练（特征提取 + Coreset 采样）和测试（异常检测 + 分割）
  2. 支持昇腾 800I A2 NPU / CPU，自动环境检测与参数调优
  3. 三档加速模式：normal（保精度）/ fast（中等加速）/ turbo（极限加速）
  4. 记录完整端到端耗时及各项精度指标
  5. 输出汇总结果（控制台 + CSV）

加速模式对比（估算）：
  模式       | 训练 batch | 采样器   | 采样率 | 训练子集 | 并行 | 预期加速 | 精度影响
  -----------|:----------:|:--------:|:------:|:--------:|:----:|:--------:|:--------:
  normal     |     1      | Greedy   |  0.1   |  100%    |  否  |   1x     |  基准
  fast       |     4      | Random   |  0.02  |  100%    |  否  |  2-3x    |  ~0.5%
  turbo      |     8      | Random   |  0.01  |   30%    |  是  |  5-8x    |  ~1-2%

用法：
  # 正常模式（保精度，自动检测 NPU/CPU）
  python inference.py --data_dir /path/to/mvtec

  # 快速模式（约 2-3 倍加速，精度微降）
  python inference.py --data_dir /path/to/mvtec --mode fast

  # 极速模式（约 5-8 倍加速，精度轻度下降）
  python inference.py --data_dir /path/to/mvtec --mode turbo

  # 显式指定 NPU + 自定义参数
  python inference.py --data_dir /path/to/mvtec --device npu:0 --mode fast --parallel 4

  # CPU 模式（调试/对比）
  python inference.py --data_dir /path/to/mvtec --device cpu --batch_size 1

  # 仅跑部分类别加速调试
  python inference.py --data_dir /path/to/mvtec --device cpu --categories bottle cable

依赖：
  pip install torch torchvision timm faiss-cpu scikit-learn scikit-image tqdm click
  # 昇腾 NPU 额外安装 torch_npu（参见 README.md）
"""

import argparse
import logging
import os
import shutil
import contextlib
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, Optional, List

import gc as _gc
import numpy as np
import torch

# ── 确保 Python 路径 ─────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent  # src/
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))  # 项目根目录
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))  # src/ 目录

# isort: split
import patchcore  # noqa: E402
from patchcore.baihu_nn import BaihuNN  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname).1s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("inference")

# MVTec-AD 全部 15 个类别
ALL_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ──────────────────────────────────────────────
# 环境自动检测与参数调优
# ──────────────────────────────────────────────


def check_environment():
    """检查 Python / PyTorch / 关键依赖版本，提前暴露不兼容问题。

    Returns:
        deps: dict, 包含检测到的版本信息。
    """
    import importlib
    import sys
    from packaging.version import Version

    deps = {}
    ok = True

    # Python
    pyver = sys.version_info
    deps["python"] = f"{pyver.major}.{pyver.minor}.{pyver.micro}"
    if pyver.major < 3 or (pyver.major == 3 and pyver.minor < 8):
        LOGGER.error("需要 Python >= 3.8，当前 %s", deps["python"])
        ok = False

    # PyTorch
    try:
        deps["torch"] = torch.__version__
    except AttributeError:
        deps["torch"] = "未知"
    if Version(deps["torch"]) < Version("1.9.0"):
        LOGGER.error("需要 PyTorch >= 1.9，当前 %s", deps["torch"])
        ok = False

    # torch_npu
    deps["torch_npu"] = "未安装"
    try:
        import torch_npu  # noqa: F401
        deps["torch_npu"] = torch.__version__  # torch_npu 通常与 torch 同版本
        LOGGER.info("  NPU 后端: torch_npu 已安装")
    except ImportError:
        LOGGER.info("  NPU 后端: torch_npu 未安装（仅 CPU 模式可用）")

    # timm（backbone）
    for mod_name, label in [
        ("timm", "骨干网络(timm)"),
        ("sklearn", "指标(sklearn)"),
        ("skimage", "图像处理(skimage)"),
        ("tqdm", "进度条(tqdm)"),
    ]:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "ok")
            deps[mod_name] = str(ver)
        except ImportError:
            LOGGER.error("  缺少依赖: %s (%s)", label, mod_name)
            LOGGER.error("  安装: pip install %s", mod_name)
            ok = False

    # faiss 可选（NPU 模式下使用 BaihuNN 替代）
    try:
        import faiss
        deps["faiss"] = getattr(faiss, "__version__", "ok")
    except ImportError:
        LOGGER.warning("  faiss 未安装（可选依赖，CPU 回退到 BaihuNN）")
        deps["faiss"] = "未安装（可选）"

    return deps, ok


def detect_npu() -> Tuple[bool, str]:
    """检测 NPU 是否可用。

    Returns:
        (available, reason): 可用性标记 + 描述。
    """
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            count = torch.npu.device_count()
            props = torch.npu.get_device_properties(0)
            # Ascend 设备名通常含 "Ascend"，兼容不同版本
            name = getattr(props, "name", "AscendNPU")
            total_mem_gb = props.total_memory / (1024 ** 3)
            return True, f"{name} ×{count}  (共 {total_mem_gb:.0f} GiB 显存)"
        return False, "torch_npu 已安装但无可用设备"
    except ImportError:
        return False, "未安装 torch_npu"
    except Exception as exc:
        return False, f"检测异常: {exc}"


def detect_cpu_count() -> int:
    """检测可用 CPU 逻辑核数（容器感知）。"""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def detect_ram_gib() -> float:
    """检测系统总内存（GiB）。"""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except OSError:
        pass
    # fallback
    try:
        import psutil  # noqa: F401
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    return 0.0


def detect_shm_gib() -> float:
    """检测 /dev/shm 可用大小（GiB），失败返回 0（表示未知）。"""
    try:
        st = os.statvfs("/dev/shm")
        free_bytes = st.f_frsize * st.f_bavail
        return free_bytes / (1024 ** 3)
    except Exception:
        return 0.0


def detect_npu_mem_gib(device_id: int = 0) -> float:
    """检测 NPU 显存总量（GiB），失败返回 0。"""
    try:
        import torch_npu  # noqa: F401
        total_bytes = torch.npu.get_device_properties(device_id).total_memory
        return total_bytes / (1024 ** 3)
    except Exception:
        return 0.0


def detect_disk_gib(path: str) -> tuple[float, float]:
    """检测指定路径所在磁盘的剩余/总空间（GiB）。"""
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024 ** 3), usage.total / (1024 ** 3)
    except Exception:
        return 0.0, 0.0


def auto_tune_params(
    ram_gib: float,
    npu_mem_gib: float,
    cpu_count: int,
    is_npu: bool,
    mode: str = "normal",
    shm_gib: float = 0.0,
) -> dict:
    """根据检测到的硬件资源和加速模式自动调优关键参数。

    mode 影响：训练 batch_size / 采样器类型 / 采样比例 / 训练子集。
    """
    params = {}

    # ── num_workers：一般不超过 8，避免过多上下文切换 ──
    params["num_workers"] = min(cpu_count, 8)

    # ── 如果共享内存太小，限制 num_workers 避免 OOM ──
    if shm_gib > 0 and shm_gib < 0.5:
        # <512MB shm，限制 workers 防止 DataLoader 撑爆
        params["num_workers"] = min(params["num_workers"], 2)
    if shm_gib > 0 and shm_gib < 0.1:
        # <100MB shm，只留 1 个 worker（容器默认 64MB）
        params["num_workers"] = min(params["num_workers"], 1)

    # ── batch_size（测试 batch_size）─ 以可用显存/内存为依据 ──
    mem_for_batch = npu_mem_gib if is_npu and npu_mem_gib > 0 else ram_gib
    if mem_for_batch >= 32:
        params["batch_size"] = 16
    elif mem_for_batch >= 16:
        params["batch_size"] = 8
    elif mem_for_batch >= 8:
        params["batch_size"] = 4
    else:
        params["batch_size"] = 2

    # ── sampler_percentage：受 mode 影响 ──
    mem_for_sampler = npu_mem_gib if is_npu and npu_mem_gib > 0 else ram_gib
    if mode == "turbo":
        # 极速模式：激进压缩
        params["sampler_percentage"] = 0.01
    elif mode == "fast":
        # 快速模式：中等压缩
        params["sampler_percentage"] = 0.02
    else:
        # normal：按内存自适应（原逻辑）
        if mem_for_sampler >= 64:
            params["sampler_percentage"] = 0.1
        elif mem_for_sampler >= 32:
            params["sampler_percentage"] = 0.05
        elif mem_for_sampler >= 16:
            params["sampler_percentage"] = 0.02
        else:
            params["sampler_percentage"] = 0.01

    # ── train_batch_size（训练特征提取时）──
    if mode == "turbo":
        params["train_batch_size"] = min(params["batch_size"], 16)
    elif mode == "fast":
        params["train_batch_size"] = min(params["batch_size"], 4)
    else:
        params["train_batch_size"] = 1  # 原论文方式

    # ── train_subset_ratio（训练子集比例）──
    if mode == "turbo":
        params["train_subset_ratio"] = 0.3
    elif mode == "fast":
        params["train_subset_ratio"] = 1.0
    else:
        params["train_subset_ratio"] = 1.0

    # ── resize / imagesize ──
    if mode in ("fast", "turbo"):
        params["resize"] = 224  # 与 imagesize 一致，省略一次缩放
    else:
        params["resize"] = 256  # 原论文默认 resize 尺寸

    # ── sampler_type ──
    if mode in ("fast", "turbo"):
        params["sampler_type"] = "random"
    else:
        params["sampler_type"] = "greedy"

    return params


# ──────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="PatchCore 昇腾 NPU 端到端推理脚本 —— 支持自动环境检测与调优"
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="MVTec-AD 数据集根目录（内含 bottle/ cable/ ... 子目录），如不指定自动检测常见路径",
    )
    parser.add_argument(
        "--device",
        default="auto",
        type=str,
        help='运行设备：auto（自动检测）/ npu:0 / cpu（默认 auto）',
    )
    parser.add_argument(
        "--mode",
        default="normal",
        type=str,
        choices=["normal", "fast", "turbo"],
        help="加速模式：normal（保精度）/ fast（约2-3x）/ turbo（约5-8x）（默认 normal）",
    )
    parser.add_argument(
        "--parallel",
        default=None,
        type=int,
        help="并行跑类别数，仅 turbo 模式生效（默认自动：NPU=3 / CPU=4）",
    )
    parser.add_argument(
        "--batch_size",
        default=None,
        type=int,
        help="推理阶段 DataLoader batch size（默认自动调优）",
    )
    parser.add_argument(
        "--train_batch_size",
        default=None,
        type=int,
        help="训练阶段 batch size（默认 normal=1, fast=4, turbo=8）",
    )
    parser.add_argument(
        "--num_workers",
        default=None,
        type=int,
        help="DataLoader 工作进程数（默认自动检测可用 CPU 核数）",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="指定要跑的类别子集（默认全 15 个），如 --categories bottle cable",
    )
    parser.add_argument(
        "--train_subset_ratio",
        default=None,
        type=float,
        help="训练子集比例 0~1（默认 normal=1.0, fast=1.0, turbo=0.3）",
    )
    parser.add_argument(
        "--resize",
        default=None,
        type=int,
        help="输入 resize 尺寸（默认 auto=256, fast/turbo=224）",
    )
    parser.add_argument(
        "--imagesize",
        default=224,
        type=int,
        help="输入 crop 尺寸（默认 224）",
    )
    parser.add_argument(
        "--backbone",
        default="wideresnet50",
        type=str,
        help="骨干网络名称（默认 wideresnet50 — WideResNet-50，论文原始配置）",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["layer2", "layer3"],
        help='特征提取层（默认 layer2 layer3）',
    )
    parser.add_argument(
        "--pretrain_embed_dimension",
        default=1024,
        type=int,
        help="预训练嵌入维度（默认 1024）",
    )
    parser.add_argument(
        "--target_embed_dimension",
        default=1024,
        type=int,
        help="目标嵌入维度（默认 1024）",
    )
    parser.add_argument(
        "--patchsize",
        default=3,
        type=int,
        help="Patch 大小（默认 3）",
    )
    parser.add_argument(
        "--patchstride",
        default=1,
        type=int,
        help="Patch stride（默认 1）",
    )
    parser.add_argument(
        "--anomaly_scorer_num_nn",
        default=1,
        type=int,
        help="最近邻数量（默认 1）",
    )
    parser.add_argument(
        "--sampler_percentage",
        default=None,
        type=float,
        help="Coreset 采样比例（默认自动调优）",
    )
    parser.add_argument(
        "--sampler_type",
        default="auto",
        type=str,
        choices=["auto", "greedy", "random"],
        help="采样器类型：auto（按模式自动）/ greedy（贪心）/ random（随机）（默认 auto）",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="随机种子（默认 0）",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="结果输出目录（默认自动创建）",
    )
    return parser.parse_args(argv)


def build_device(device_str: str) -> tuple[torch.device, bool]:
    """解析设备字符串并返回 (torch.device, is_npu)。

    支持 ``auto`` 自动检测 NPU 可用性，回退到 CPU。
    """
    device_str = device_str.lower()

    if device_str == "auto":
        npu_ok, reason = detect_npu()
        if npu_ok:
            LOGGER.info("自动检测到 NPU: %s", reason)
            return torch.device("npu:0"), True
        LOGGER.info("NPU 不可用 (%s)，回退到 CPU", reason)
        return torch.device("cpu"), False

    if device_str.startswith("npu"):
        idx = 0
        if ":" in device_str:
            idx = int(device_str.split(":")[1])
        return torch.device(f"npu:{idx}"), True
    return torch.device(device_str), False


def get_dataloader(
    data_dir: str,
    category: str,
    split: str,
    resize: int,
    imagesize: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    train_val_split: float = 1.0,
):
    """创建 MVTec 数据加载器。

    Args:
        train_val_split: 训练子集比例（仅 train split 有效），默认 1.0 表示全量。
    """
    from patchcore.datasets.mvtec import MVTecDataset, DatasetSplit

    split_enum = DatasetSplit.TRAIN if split == "train" else DatasetSplit.TEST
    dataset = MVTecDataset(
        source=data_dir,
        classname=category,
        resize=resize,
        imagesize=imagesize,
        split=split_enum,
        train_val_split=train_val_split,
        seed=seed,
    )
    try:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    except RuntimeError as e:
        if "unable to open shared memory" in str(e).lower() or "shm" in str(e).lower():
            LOGGER.warning(
                "共享内存不足（/dev/shm 太小），回退到 num_workers=0, pin_memory=False"
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
        else:
            raise
    loader.name = f"mvtec_{category}"
    return dataset, loader


def run_single_category(
    args, device, category: str, output_dir: str
) -> dict:
    """对单个 MVTec 类别执行训练 + 推理，返回结果字典。"""
    _is_npu = device.type == "npu"
    mode = args.mode

    # ── 数据加载 ──────────────────────────────────────────────────────
    LOGGER.info("── [%s] 加载训练数据 ──", category)
    train_dataset, train_loader = get_dataloader(
        args.data_dir, category, "train",
        args.resize, args.imagesize,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_val_split=args.train_subset_ratio,
    )

    LOGGER.info("── [%s] 加载测试数据 ──", category)
    test_dataset, test_loader = get_dataloader(
        args.data_dir, category, "test",
        args.resize, args.imagesize,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    imagesize = train_dataset.imagesize
    train_count = len(train_dataset)
    test_count = len(test_dataset)

    # ── 构建模型 ──────────────────────────────────────────────────────
    LOGGER.info("── [%s] 构建 PatchCore 模型 ──", category)
    device_context = (
        torch.npu.device(device)
        if _is_npu
        else contextlib.nullcontext()
    )

    with device_context:
        if _is_npu:
            torch.npu.empty_cache()
            torch.npu.manual_seed(args.seed)

        # 选择 NN 方法：NPU → BaihuNN，CPU → FaissNN
        if _is_npu:
            nn_method = BaihuNN(on_npu=True, device_id=0)
        else:
            nn_method = None  # patchcore 自动创建 FaissNN

        # 选择采样器
        if args.sampler_type == "random":
            sampler = patchcore.sampler.RandomSampler(
                percentage=args.sampler_percentage,
            )
        else:
            sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(
                percentage=args.sampler_percentage,
                device=device if _is_npu else torch.device("cpu"),
            )

        patchcore_model = patchcore.patchcore.PatchCore(device=device)
        patchcore_model.load(
            backbone=patchcore.backbones.load(args.backbone),
            layers_to_extract_from=args.layers,
            device=device,
            input_shape=imagesize,
            pretrain_embed_dimension=args.pretrain_embed_dimension,
            target_embed_dimension=args.target_embed_dimension,
            patchsize=args.patchsize,
            patchstride=args.patchstride,
            anomaly_score_num_nn=args.anomaly_scorer_num_nn,
            featuresampler=sampler,
            nn_method=nn_method,
        )

        # ── 训练阶段（特征提取 + Coreset 采样） ──────────────────────
        LOGGER.info("── [%s] 训练阶段... ──", category)
        t_train_start = time.perf_counter()
        patchcore_model.fit(train_loader)
        t_train = time.perf_counter() - t_train_start

        if _is_npu:
            torch.npu.empty_cache()

        # ── 测试阶段（推理） ──────────────────────────────────────────
        LOGGER.info("── [%s] 推理阶段... ──", category)
        t_test_start = time.perf_counter()
        scores, segmentations, labels_gt, masks_gt = patchcore_model.predict(
            test_loader
        )
        t_test = time.perf_counter() - t_test_start

        # ── 计算指标 ──────────────────────────────────────────────────
        anomaly_labels = [x[1] != "good" for x in test_dataset.data_to_iterate]

        image_auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
            scores, anomaly_labels
        )["auroc"]

        pixel_auroc = patchcore.metrics.compute_pixelwise_retrieval_metrics(
            segmentations, masks_gt
        )["auroc"]

        # PRO: 只含异常的图片
        sel_idxs = [i for i in range(len(masks_gt)) if np.sum(masks_gt[i]) > 0]
        if sel_idxs:
            anomaly_pixel_auroc = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations[i] for i in sel_idxs],
                [masks_gt[i] for i in sel_idxs],
            )["auroc"]
        else:
            anomaly_pixel_auroc = 0.0

        # PRO Score（Per-Region Overlap，MVTec-AD 官方指标）
        if sel_idxs:
            pro_result = patchcore.metrics.compute_pro_metric(
                [segmentations[i] for i in sel_idxs],
                [masks_gt[i] for i in sel_idxs],
            )
            pro_score = pro_result["pro_score"]
        else:
            pro_score = 0.0

        # ── 释放显存，为下一类别腾出干净环境 ──────────────────────
        del patchcore_model, train_loader, test_loader
        if _is_npu:
            torch.npu.empty_cache()
        _gc.collect()

    results = {
        "category": category,
        "image_auroc": float(image_auroc),
        "pixel_auroc": float(pixel_auroc),
        "anomaly_pixel_auroc": float(anomaly_pixel_auroc),
        "pro_score": float(pro_score),
        "train_time_s": round(t_train, 2),
        "test_time_s": round(t_test, 2),
        "train_count": train_count,
        "test_count": test_count,
        "total_time_s": round(t_train + t_test, 2),
    }

    LOGGER.info(
        "[%s] Train=%.1fs  Test=%.1fs  ImgAUROC=%.4f  PixAUROC=%.4f  PRO=%.4f",
        category, t_train, t_test, image_auroc, pixel_auroc, pro_score,
    )
    return results


def _run_category_wrapper(args, device, category, output_dir):
    """ProcessPoolExecutor 的包装函数（必须为模块顶层，便于 pickle）。"""
    return run_single_category(args, device, category, output_dir)


def main():
    args = parse_args()

    # ── 自动检测数据集路径（若未通过 --data_dir 指定）───────────────
    if args.data_dir is None:
        from pathlib import Path as _Path
        for _candidate in [
            "/data/mvtec_ad",
            "/dataset/mvtec_ad",
            "/datasets/mvtec_ad",
            str(_HERE.parent / "datasets" / "mvtec_ad"),
            str(_HERE.parent / "mvtec_ad"),
        ]:
            if _Path(_candidate).is_dir():
                args.data_dir = _candidate
                LOGGER.info("自动检测到数据集: %s", args.data_dir)
                break
        if args.data_dir is None:
            LOGGER.error(
                "--data_dir 未指定且未找到数据集，\n"
                "  请使用 --data_dir /path/to/mvtec_ad 指定"
            )
            sys.exit(1)

    # ── 软件环境检查 ──────────────────────────────────────────────────
    LOGGER.info("=" * 60)
    LOGGER.info("环境检测（软件）")
    deps, deps_ok = check_environment()
    for name, ver in deps.items():
        LOGGER.info("  %s: %s", name, ver)
    if not deps_ok:
        LOGGER.error("环境检查未通过，请安装缺失依赖后重试")
        sys.exit(1)

    # ── 硬件检测 ──────────────────────────────────────────────────────
    cpu_count = detect_cpu_count()
    ram_gib = detect_ram_gib()
    LOGGER.info("=" * 60)
    LOGGER.info("环境检测")
    LOGGER.info("  CPU 逻辑核数: %d", cpu_count)
    LOGGER.info("  总内存: %.1f GiB", ram_gib)

    device, is_npu = build_device(args.device)
    if is_npu:
        npu_mem_gib = detect_npu_mem_gib()
        LOGGER.info("  NPU 显存: %.1f GiB", npu_mem_gib)
    else:
        npu_mem_gib = 0.0

    shm_gib = detect_shm_gib()
    if shm_gib > 0:
        LOGGER.info("  共享内存 /dev/shm: %.2f GiB", shm_gib)
    else:
        LOGGER.info("  共享内存 /dev/shm: 未知（非 Linux 或无法检测）")

    free_disk_gib, total_disk_gib = detect_disk_gib(args.data_dir)
    LOGGER.info("  磁盘（数据目录）: 剩余 %.1f GiB / 共 %.1f GiB", free_disk_gib, total_disk_gib)

    # ── 自动调优 ──────────────────────────────────────────────────────
    tuned = auto_tune_params(ram_gib, npu_mem_gib, cpu_count, is_npu, mode=args.mode, shm_gib=shm_gib)
    tune_keys = [
        ("batch_size", int),
        ("train_batch_size", int),
        ("num_workers", int),
        ("sampler_percentage", float),
        ("train_subset_ratio", float),
        ("resize", int),
        ("sampler_type", str),
    ]
    for key, _type in tune_keys:
        user_val = getattr(args, key, None)
        if user_val is None:
            setattr(args, key, tuned[key])
            LOGGER.info("  %s: 自动设为 %s", key, tuned[key])
        else:
            LOGGER.info("  %s: 用户指定 %s（自动推荐 %s）", key, user_val, tuned[key])

    # ── 并行参数（仅 turbo 生效）──
    if args.mode == "turbo" and args.parallel is None:
        args.parallel = 3 if is_npu else 4
    elif args.mode != "turbo" and args.parallel is not None:
        LOGGER.warning("--parallel 仅在 turbo 模式下生效，忽略")
        args.parallel = None

    LOGGER.info("=" * 60)

    LOGGER.info("设备: %s", device)
    LOGGER.info("加速模式: %s", args.mode)
    LOGGER.info("Batch size (推理): %d | (训练): %d", args.batch_size, args.train_batch_size)
    LOGGER.info("骨干网络: %s 层: %s", args.backbone, args.layers)
    LOGGER.info("采样器: %s (%.1f%%)", args.sampler_type, args.sampler_percentage * 100)
    if args.train_subset_ratio < 1.0:
        LOGGER.info("训练子集: %.0f%%", args.train_subset_ratio * 100)
    if args.parallel:
        LOGGER.info("并行类别: %d 进程", args.parallel)
    LOGGER.info("数据目录: %s", args.data_dir)

    categories = args.categories or ALL_CATEGORIES
    invalid = [c for c in categories if c not in ALL_CATEGORIES]
    if invalid:
        LOGGER.error("未知类别: %s，可选: %s", invalid, ALL_CATEGORIES)
        sys.exit(1)

    # ── 创建输出目录 ──────────────────────────────────────────────────
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = str(_HERE / "results" / f"inference_{timestamp}")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    LOGGER.info("结果目录: %s", output_dir)

    # ── 逐个类别执行 ──────────────────────────────────────────────────
    all_results = []
    wall_start = time.perf_counter()

    if args.parallel:
        # ── 并行模式 ──────────────────────────────────────────────────
        LOGGER.info("启动 %d 进程并行执行 %d 个类别", args.parallel, len(categories))
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            # 提交所有任务
            future_to_cat = {}
            for cat in categories:
                future = executor.submit(
                    _run_category_wrapper, args, device, cat, output_dir
                )
                future_to_cat[future] = cat

            # 收集结果
            for future in as_completed(future_to_cat):
                cat = future_to_cat[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    LOGGER.exception("类别 [%s] 并行执行失败: %s", cat, e)
                    all_results.append({
                        "category": cat,
                        "image_auroc": 0.0,
                        "pixel_auroc": 0.0,
                        "anomaly_pixel_auroc": 0.0,
                        "train_time_s": 0.0,
                        "test_time_s": 0.0,
                        "total_time_s": 0.0,
                        "train_count": 0,
                        "test_count": 0,
                        "pro_score": 0.0,
                        "error": str(e),
                    })
    else:
        # ── 串行模式 ──────────────────────────────────────────────────
        for cat in categories:
            LOGGER.info("")
            LOGGER.info("=" * 60)
            LOGGER.info("开始类别: %s", cat)
            LOGGER.info("=" * 60)

            try:
                result = run_single_category(args, device, cat, output_dir)
                all_results.append(result)
            except Exception as e:
                LOGGER.exception("类别 [%s] 执行失败: %s", cat, e)
                all_results.append({
                    "category": cat,
                    "image_auroc": 0.0,
                    "pixel_auroc": 0.0,
                    "anomaly_pixel_auroc": 0.0,
                    "pro_score": 0.0,
                    "train_time_s": 0.0,
                    "test_time_s": 0.0,
                    "total_time_s": 0.0,
                    "train_count": 0,
                    "test_count": 0,
                    "error": str(e),
                })

    wall_total = time.perf_counter() - wall_start

    # ── 汇总输出 ──────────────────────────────────────────────────────
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("结果汇总")
    LOGGER.info("=" * 60)

    img_aurocs = [r["image_auroc"] for r in all_results if r["image_auroc"] > 0]
    pix_aurocs = [r["pixel_auroc"] for r in all_results if r["pixel_auroc"] > 0]
    pro_scores = [r["pro_score"] for r in all_results if r["pro_score"] > 0]
    train_times = [r["train_time_s"] for r in all_results]
    test_times = [r["test_time_s"] for r in all_results]
    total_times = [r["total_time_s"] for r in all_results]

    header = f"{'类别':<16} {'ImgAUROC':<10} {'PixAUROC':<10} {'PRO':<10} {'训练(s)':<10} {'推理(s)':<10} {'总计(s)':<10}"
    sep = "-" * len(header)
    LOGGER.info(header)
    LOGGER.info(sep)
    for r in all_results:
        LOGGER.info(
            f"{r['category']:<16} {r['image_auroc']:<10.4f} {r['pixel_auroc']:<10.4f} "
            f"{r['pro_score']:<10.4f} "
            f"{r['train_time_s']:<10.1f} {r['test_time_s']:<10.1f} {r['total_time_s']:<10.1f}"
        )
    LOGGER.info(sep)
    if img_aurocs:
        LOGGER.info(
            f"{'平均值':<16} {np.mean(img_aurocs):<10.4f} {np.mean(pix_aurocs):<10.4f} "
            f"{np.mean(pro_scores):<10.4f} "
            f"{np.mean(train_times):<10.1f} {np.mean(test_times):<10.1f} {np.mean(total_times):<10.1f}"
        )
    LOGGER.info(f"\n端到端总耗时（挂钟）: {wall_total:.1f} 秒 ({wall_total/60:.1f} 分钟)")
    LOGGER.info(f"设备: {device}  |  Batch size: {args.batch_size}")

    # ── 吞吐量统计 ────────────────────────────────────────────────────
    total_train_imgs = sum(r["train_count"] for r in all_results)
    total_test_imgs = sum(r["test_count"] for r in all_results)
    total_all_imgs = total_train_imgs + total_test_imgs
    total_train_time = sum(r["train_time_s"] for r in all_results)
    total_test_time = sum(r["test_time_s"] for r in all_results)

    LOGGER.info("\n=== 吞吐量 ===")
    LOGGER.info(f"训练图片: {total_train_imgs}  测试图片: {total_test_imgs}  合计: {total_all_imgs}")
    LOGGER.info(f"总吞吐量 (训练+测试): {total_all_imgs / wall_total:.2f} img/s" if wall_total > 0 else "总吞吐量 (训练+测试): N/A (wall_time=0)")
    LOGGER.info(f"推理吞吐量 (测试集):  {total_test_imgs / total_test_time:.2f} img/s" if total_test_time > 0 else "推理吞吐量 (测试集):  N/A (test_time=0)")

    # ── 保存 CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, "w") as f:
        f.write("category,image_auroc,pixel_auroc,anomaly_pixel_auroc,pro_score,"
                "train_time_s,test_time_s,train_count,test_count,total_time_s\n")
        for r in all_results:
            f.write(
                f"{r['category']},{r['image_auroc']:.6f},{r['pixel_auroc']:.6f},"
                f"{r['anomaly_pixel_auroc']:.6f},{r['pro_score']:.6f},"
                f"{r['train_time_s']:.2f},{r['test_time_s']:.2f},"
                f"{r['train_count']},{r['test_count']},{r['total_time_s']:.2f}\n"
            )
        # 平均行
        if img_aurocs:
            f.write(
                f"mean,{np.mean(img_aurocs):.6f},{np.mean(pix_aurocs):.6f},"
                f"{np.mean([r['anomaly_pixel_auroc'] for r in all_results if r['anomaly_pixel_auroc'] > 0]):.6f},"
                f"{np.mean(pro_scores):.6f},"
                f"{np.mean(train_times):.2f},{np.mean(test_times):.2f},"
                f"{total_train_imgs},{total_test_imgs},{np.mean(total_times):.2f}\n"
            )

    LOGGER.info("\n结果已保存至: %s", csv_path)
    return all_results


if __name__ == "__main__":
    main()
