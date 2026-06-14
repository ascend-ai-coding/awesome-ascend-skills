"""Anomaly metrics."""
import numpy as np
from skimage import measure
from sklearn import metrics


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }


def compute_pro_metric(anomaly_segmentations, ground_truth_masks, max_fpr=0.3,
                       num_thresholds=1000):
    """
    Computes the Per-Region Overlap (PRO) score.

    PRO 是 MVTec-AD 官方评估指标。对于每张图片的每个异常区域（连通分量），
    计算 True Positive Rate，再对所有区域取平均 TPR，最后求 FPR 在 [0, max_fpr]
    范围内的平均 TPR。

    Args:
        anomaly_segmentations: [list of np.array] [N x H x W] 异常评分图。
        ground_truth_masks:    [list of np.array] [N x H x W] 二值 ground truth。
        max_fpr:               FPR 上限（默认 0.3）。
        num_thresholds:        阈值采样数（默认 1000）。

    Returns:
        {"pro_score": float,           # 平均 TPR 在 FPR∈[0, max_fpr] 上
         "pro_fpr": np.array,          # 插值后的 FPR 点
         "pro_tpr": np.array}          # 对应的平均 TPR
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    # squeeze 掉多余的通道维度（如 (N,1,H,W) → (N,H,W))
    if ground_truth_masks.ndim == 4 and ground_truth_masks.shape[1] == 1:
        ground_truth_masks = ground_truth_masks.squeeze(1)

    assert anomaly_segmentations.shape == ground_truth_masks.shape, \
        f"Shape mismatch: {anomaly_segmentations.shape} vs {ground_truth_masks.shape}"

    n_images = anomaly_segmentations.shape[0]

    # 对每张图片提取异常区域（连通分量）
    region_sizes = []      # 每个区域的像素数
    region_masks = []      # (H, W) 每个区域对应的 bool mask
    region_image_idx = []  # 每个区域对应的图片索引
    total_normal_pixels = 0  # 所有图片中的正常像素总数

    for i in range(n_images):
        gt = (ground_truth_masks[i] > 0).astype(int)
        # 连通分量标注
        labeled = measure.label(gt, connectivity=2)
        props = measure.regionprops(labeled)

        # 所有正常像素（gt == 0）
        total_normal_pixels += int(np.sum(gt == 0))

        for prop in props:
            # 跳过过小的噪点区域（< 3 像素）
            if prop.area < 3:
                continue
            region_mask = (labeled == prop.label)
            region_sizes.append(prop.area)
            region_masks.append(region_mask)
            region_image_idx.append(i)

    if not region_masks:
        return {"pro_score": 0.0, "pro_fpr": np.array([0.0]), "pro_tpr": np.array([0.0])}

    n_regions = len(region_masks)

    # 在 num_thresholds 个阈值上计算 TPR（逐区域）和 FPR（全局）
    thresholds = np.linspace(0, 1, num_thresholds)
    region_tpr_at_thresh = np.zeros((n_regions, num_thresholds))  # [R, T]
    fpr_at_thresh = np.zeros(num_thresholds)

    for t_idx, thresh in enumerate(thresholds):
        # 二值化预测
        preds = (anomaly_segmentations >= thresh).astype(int)

        # 全局 FPR = FP / total_normal
        fpr_numerator = 0
        for i in range(n_images):
            fpr_numerator += int(np.sum((preds[i] == 1) & (ground_truth_masks[i] == 0)))
        fpr_at_thresh[t_idx] = fpr_numerator / max(total_normal_pixels, 1)

        # 逐区域 TPR
        for r_idx in range(n_regions):
            mask = region_masks[r_idx]
            img_idx = region_image_idx[r_idx]
            # 用该区域对应图片的预测结果，取区域内预测为 positive 的像素数
            tp = int(np.sum(preds[img_idx][mask] > 0))
            total = region_sizes[r_idx]
            region_tpr_at_thresh[r_idx, t_idx] = tp / max(total, 1)

    # 在每个 FPR 点求所有区域的 TPR 平均值
    tpr_avg = np.mean(region_tpr_at_thresh, axis=0)  # [T]

    # 按 fpr 排序（确保单调递增）
    sort_idx = np.argsort(fpr_at_thresh)
    fpr_sorted = fpr_at_thresh[sort_idx]
    tpr_sorted = tpr_avg[sort_idx]

    # 在 [0, max_fpr] 区间上均匀取 100 个点做插值
    fpr_grid = np.linspace(0, max_fpr, 100)
    tpr_interp = np.interp(fpr_grid, fpr_sorted, tpr_sorted)

    # PRO = 平均 TPR 在 FPR∈[0, max_fpr] 上
    pro_score = float(np.mean(tpr_interp))

    return {
        "pro_score": pro_score,
        "pro_fpr": fpr_grid,
        "pro_tpr": tpr_interp,
    }
