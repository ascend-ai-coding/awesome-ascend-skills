#!/usr/bin/env python3
"""
YOLO End-to-End OM Inference Script

Provides Ultralytics-like inference interface for YOLO OM models on Ascend NPU.
Supports multiple YOLO task types: Detection, Pose, Segmentation, OBB, Classification.

Usage:
    # Detection task (default, imgsz=640)
    python3 yolo_om_infer.py --model yolo.om --source image.jpg --task detect

    # Classification (imgsz=224)
    python3 yolo_om_infer.py --model yolo-cls.om --source image.jpg --task classify --imgsz 224

    # Pose estimation
    python3 yolo_om_infer.py --model yolo-pose.om --source image.jpg --task pose

    # Segmentation
    python3 yolo_om_infer.py --model yolo-seg.om --source image.jpg --task segment

    # Oriented Bounding Box
    python3 yolo_om_infer.py --model yolo-obb.om --source image.jpg --task obb

Requirements:
    - ais_bench and aclruntime packages
    - ultralytics (for preprocessing)
    - opencv-python
"""

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

try:
    import cv2
except ImportError:
    print("Error: opencv-python not installed.")
    print("Install with: pip install opencv-python")
    sys.exit(1)

try:
    from ultralytics.data.augment import LetterBox
except ImportError:
    print("Error: ultralytics not installed.")
    print("Install with: pip install ultralytics")
    sys.exit(1)

try:
    from ais_bench.infer.interface import InferSession
except ImportError:
    print("Error: ais_bench package not installed.")
    print(
        "Install from: https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench"
    )
    sys.exit(1)

try:
    import torch
    from torchvision.ops import nms as torchvision_nms
except ImportError:
    torch = None
    torchvision_nms = None


# =============================================================================
# NMS and Box Utilities
# =============================================================================


def nms_numpy(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """
    Non-Maximum Suppression (NMS) using pure NumPy.

    Args:
        boxes: numpy array of shape (N, 4) in format [x1, y1, x2, y2]
        scores: numpy array of shape (N,)
        iou_threshold: IoU threshold for suppression

    Returns:
        indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """Apply NMS using torchvision if available, else numpy."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    if torch is not None and torchvision_nms is not None:
        return torchvision_nms(
            torch.from_numpy(boxes.copy()),
            torch.from_numpy(scores.copy()),
            iou_thres,
        ).numpy()
    else:
        return nms_numpy(boxes, scores, iou_thres)


# =============================================================================
# Post-processing for Different YOLO Tasks
# =============================================================================


def postprocess_detect(
    output: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    num_classes: int = 80,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Postprocess YOLO Detection output.

    Supports two output formats:
    1. Raw format: (1, 4+num_classes, num_anchors) e.g., (1, 84, 8400)
       - Need transpose and decode + NMS
    2. Processed format: (1, num_detections, 6) e.g., (1, 300, 6)
       - Already decoded with NMS, format: [x1, y1, x2, y2, conf, cls]

    Output: boxes_xyxy, scores, class_ids
    """
    # Handle output format
    if isinstance(output, (list, tuple)):
        output = output[0]

    # Determine output format based on shape
    if output.ndim == 3:
        # Shape: (1, C, N) where C could be 84 (raw) or N could be 300 (processed)
        if output.shape[2] == 6:
            # Processed format: (1, 300, 6) - already decoded
            predictions = output[0]  # (300, 6)
        elif output.shape[2] == 7:
            # OBB format: (1, 300, 7) - with angle
            predictions = output[0]  # (300, 7)
        elif output.shape[1] < output.shape[2]:
            # Raw format: (1, 84, 8400) - channels < anchors, need transpose
            predictions = output[0].T  # (8400, 84)
        else:
            # Processed format: (1, 300, 84) or similar
            predictions = output[0]
    else:
        predictions = output

    # Check if this is processed format (6 columns: x1,y1,x2,y2,conf,cls)
    if predictions.shape[1] == 6:
        # Already processed format: [x1, y1, x2, y2, conf, cls]
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        class_ids = predictions[:, 5].astype(np.int32)

        # Filter by confidence
        mask = scores >= conf_thres
        return boxes[mask], scores[mask], class_ids[mask]

    # Raw YOLO output: (num_anchors, 4+classes)
    boxes_cxcywh = predictions[:, :4]  # cx, cy, w, h
    class_scores = predictions[:, 4:]  # class scores

    # Get best class for each anchor
    scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1).astype(np.int32)

    # Filter by confidence threshold
    mask = scores >= conf_thres
    boxes_cxcywh = boxes_cxcywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes_cxcywh) == 0:
        return np.array([]), np.array([]), np.array([])

    # Convert to xyxy format
    boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh)

    # Apply NMS
    keep = nms_boxes(boxes_xyxy, scores, iou_thres)

    return boxes_xyxy[keep], scores[keep], class_ids[keep]


def postprocess_pose(
    output: np.ndarray,
    conf_thres: float = 0.25,
    num_keypoints: int = 17,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Postprocess YOLO Pose output.

    Input shape: (1, num_detections, 4+1+num_keypoints*3) e.g., (1, 300, 56)
    Or: (1, num_detections, 4+1+1+num_keypoints*3) e.g., (1, 300, 57)
    - 4: box coordinates (x1, y1, x2, y2)
    - 1: confidence (box_conf * class_conf, or just box_conf for single-class)
    - Optional 1: class score (often 0 for pose since only person class)
    - 51: 17 keypoints * 3 (x, y, conf)

    Output: boxes, scores, class_ids, keypoints
    - keypoints: (N, 17, 3) - x, y, confidence for each keypoint
    """
    if isinstance(output, (list, tuple)):
        output = output[0]

    # output shape: (1, 300, 56) or (1, 300, 57)
    predictions = output[0] if output.ndim == 3 else output

    # Determine format based on number of columns
    num_cols = predictions.shape[1]
    if num_cols == 56:
        # Format: 4 (box) + 1 (conf) + 51 (kpts)
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        keypoints_flat = predictions[:, 5:]
    elif num_cols == 57:
        # Format: 4 (box) + 1 (box_conf) + 1 (cls) + 51 (kpts)
        boxes = predictions[:, :4]
        box_conf = predictions[:, 4]
        class_scores = predictions[:, 5]
        # For pose (single class), class_scores may be 0, so use box_conf directly
        # or if class_scores is non-zero, use the product
        if class_scores.max() > 0:
            scores = box_conf * class_scores
        else:
            scores = box_conf
        keypoints_flat = predictions[:, 6:]
    else:
        raise ValueError(f"Unexpected pose output shape: {predictions.shape}")

    # Filter by confidence
    mask = scores >= conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    keypoints_flat = keypoints_flat[mask]

    if len(boxes) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]).reshape(0, num_keypoints, 3),
        )

    # Reshape keypoints to (N, 17, 3)
    keypoints = keypoints_flat.reshape(-1, num_keypoints, 3)

    # Class is always 0 for pose (person)
    class_ids = np.zeros(len(boxes), dtype=np.int32)

    # No NMS needed for pose output (already top-k)
    return boxes, scores, class_ids, keypoints


def postprocess_segment(
    output: Union[np.ndarray, List],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    num_classes: int = 80,
    num_masks: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Postprocess YOLO Segmentation output.

    Supports two output formats:
    1. Processed format: (1, num_detections, 4+1+1+num_masks) e.g., (1, 300, 38)
       - 4: box (x1, y1, x2, y2)
       - 1: confidence
       - 1: class index
       - 32: mask coefficients
    2. Raw format: (1, 4+num_classes+num_masks, num_anchors) e.g., (1, 116, 8400)

    Output: boxes_xyxy, scores, class_ids, mask_coeffs
    """
    protos = None
    if isinstance(output, (list, tuple)):
        if len(output) == 2:
            # Two outputs: (predictions, protos)
            pred_output, protos = output
        else:
            pred_output = output[0]
    else:
        pred_output = output

    # Handle detection output
    if isinstance(pred_output, (list, tuple)):
        pred_output = pred_output[0]

    # Determine output format
    if pred_output.ndim == 3:
        if pred_output.shape[2] == 38 or pred_output.shape[2] == 37:
            # Processed format: (1, 300, 38) - already decoded
            predictions = pred_output[0]  # (300, 38)

            # Format: [x1, y1, x2, y2, conf, cls, mask_coeffs...]
            boxes = predictions[:, :4]  # Already xyxy
            scores = predictions[:, 4]
            class_ids = predictions[:, 5].astype(np.int32)
            mask_coeffs = predictions[:, 6 : 6 + num_masks]

            # Filter by confidence
            mask = scores >= conf_thres
            return boxes[mask], scores[mask], class_ids[mask], mask_coeffs[mask]

        elif pred_output.shape[1] < pred_output.shape[2]:
            # Raw format: (1, 116, 8400) - channels < anchors, need transpose
            predictions = pred_output[0].T  # (8400, 116)
        else:
            predictions = pred_output[0]
    else:
        predictions = pred_output

    # Raw YOLO output: (num_anchors, 4+classes+masks)
    boxes_cxcywh = predictions[:, :4]
    class_scores = predictions[:, 4 : 4 + num_classes]
    mask_coeffs = predictions[:, 4 + num_classes : 4 + num_classes + num_masks]

    # Get best class
    scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1).astype(np.int32)

    # Filter by confidence
    mask = scores >= conf_thres
    boxes_cxcywh = boxes_cxcywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    mask_coeffs = mask_coeffs[mask]

    if len(boxes_cxcywh) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]).reshape(0, num_masks),
        )

    # Convert to xyxy
    boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh)

    # Apply NMS
    keep = nms_boxes(boxes_xyxy, scores, iou_thres)

    return boxes_xyxy[keep], scores[keep], class_ids[keep], mask_coeffs[keep]


def postprocess_obb(
    output: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    num_classes: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Postprocess YOLO OBB (Oriented Bounding Box) output.

    Supports two output formats:
    1. Processed format: (1, num_detections, 7) e.g., (1, 300, 7)
       - 4: box (cx, cy, w, h)
       - 1: confidence
       - 1: class index
       - 1: angle in radians
    2. Raw format: (1, 4+1+num_classes, num_anchors) e.g., (1, 20, 8400)

    Output: boxes_xywh, scores, class_ids, angles
    - boxes_xywh: (N, 4) - cx, cy, w, h
    """
    if isinstance(output, (list, tuple)):
        output = output[0]

    # Determine output format
    if output.ndim == 3:
        if output.shape[2] == 7:
            # Processed format: (1, 300, 7)
            # Format: [cx, cy, w, h, conf, cls, angle]
            predictions = output[0]  # (300, 7)

            boxes_cxcywh = predictions[:, :4]
            scores = predictions[:, 4]
            class_ids = predictions[:, 5].astype(np.int32)
            angles = predictions[:, 6]

            # Filter by confidence
            mask = scores >= conf_thres
            return (
                boxes_cxcywh[mask],
                scores[mask],
                class_ids[mask],
                angles[mask],
            )
        elif output.shape[1] < output.shape[2]:
            # Raw format: (1, 20, 8400) - channels < anchors, need transpose
            predictions = output[0].T  # (8400, 20)
        else:
            predictions = output[0]
    else:
        predictions = output

    # Raw YOLO output: (num_anchors, 4+1+classes)
    boxes_cxcywh = predictions[:, :4]
    angles = predictions[:, 4]  # angle in radians
    class_scores = predictions[:, 5 : 5 + num_classes]

    # Get best class
    scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1).astype(np.int32)

    # Filter by confidence
    mask = scores >= conf_thres
    boxes_cxcywh = boxes_cxcywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    angles = angles[mask]

    if len(boxes_cxcywh) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Apply NMS (using standard NMS on rotated boxes approximation)
    boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh)
    keep = nms_boxes(boxes_xyxy, scores, iou_thres)

    return boxes_cxcywh[keep], scores[keep], class_ids[keep], angles[keep]


def postprocess_classify(
    output: np.ndarray,
    top_k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Postprocess YOLO Classification output.

    Input shape: (1, num_classes) e.g., (1, 1000) for ImageNet
    Output: top_k class indices and their probabilities
    """
    if isinstance(output, (list, tuple)):
        output = output[0]

    # Squeeze batch dimension
    if output.ndim == 2:
        values = output[0]  # (num_classes,)
    elif output.ndim == 1:
        values = output
    else:
        values = output.flatten()

    # Check if already softmaxed (sum ≈ 1.0 and all non-negative)
    if values.min() >= 0 and abs(values.sum() - 1.0) < 0.01:
        probs = values
    else:
        # Apply softmax (numerically stable)
        exp_logits = np.exp(values - np.max(values))
        probs = exp_logits / exp_logits.sum()

    # Get top-k
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_probs = probs[top_indices]

    return top_indices, top_probs


# =============================================================================
# Visualization Functions
# =============================================================================

# COCO Keypoint skeleton connections (17 keypoints)
SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # Head
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),  # Arms
    (5, 11),
    (6, 12),
    (11, 12),  # Torso
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),  # Legs
]

# Colors for skeleton (left=blue, right=red)
KPT_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),  # 0-4
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),  # 5-9
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),  # 10-14
    (255, 0, 255),  # 15
]

LIMB_COLORS = [
    (255, 0, 0),
    (255, 0, 255),
    (170, 0, 255),
    (255, 0, 255),  # 0-1, 0-2, 1-3, 2-4
    (255, 255, 0),
    (255, 255, 0),
    (85, 255, 0),
    (255, 255, 0),
    (85, 255, 0),  # 5-6, 5-7, 7-9, 6-8, 8-10
    (255, 255, 0),
    (255, 255, 0),
    (255, 255, 0),  # 5-11, 6-12, 11-12
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),  # 11-13, 13-15, 12-14, 14-16
]


def get_color(cls_id: int) -> Tuple[int, int, int]:
    """Get a consistent color for a class ID."""
    colors_list = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 0),
        (0, 128, 0),
        (0, 0, 128),
        (128, 128, 0),
        (128, 0, 128),
        (0, 128, 128),
        (255, 128, 0),
        (255, 0, 128),
        (128, 255, 0),
    ]
    return colors_list[cls_id % len(colors_list)]


def draw_detections(
    img: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    classes: Optional[List[str]] = None,
) -> np.ndarray:
    """Draw detection boxes on image."""
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        conf = scores[i]
        cls = int(class_ids[i])
        label = f"{classes[cls] if classes and cls < len(classes) else cls} {conf:.2f}"
        color = get_color(cls)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.rectangle(
            img,
            (x1, label_y - label_size[1] - 4),
            (x1 + label_size[0], label_y),
            color,
            -1,
        )
        cv2.putText(
            img,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return img


def draw_pose(
    img: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    keypoints: np.ndarray,
    conf_thres: float = 0.5,
) -> np.ndarray:
    """
    Draw pose keypoints and skeleton on image.

    Args:
        img: Input image
        boxes: Bounding boxes (N, 4)
        scores: Confidence scores (N,)
        keypoints: Keypoints array (N, 17, 3) - x, y, conf
        conf_thres: Keypoint confidence threshold
    """
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"person {scores[i]:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        kpts = keypoints[i]  # (17, 3)

        # Draw skeleton connections
        for j, (start, end) in enumerate(SKELETON):
            if kpts[start, 2] > conf_thres and kpts[end, 2] > conf_thres:
                pt1 = (int(kpts[start, 0]), int(kpts[start, 1]))
                pt2 = (int(kpts[end, 0]), int(kpts[end, 1]))
                color = LIMB_COLORS[j] if j < len(LIMB_COLORS) else (255, 255, 255)
                cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)

        # Draw keypoints
        for j, (x, y, conf) in enumerate(kpts):
            if conf > conf_thres:
                color = KPT_COLORS[j] if j < len(KPT_COLORS) else (255, 255, 255)
                cv2.circle(img, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)

    return img


def draw_segment(
    img: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    mask_coeffs: Optional[np.ndarray],
    classes: Optional[List[str]] = None,
) -> np.ndarray:
    """Draw segmentation results (boxes only, mask requires protos)."""
    # For now, just draw boxes with segmentation label style
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        conf = scores[i]
        cls = int(class_ids[i])
        label = f"{classes[cls] if classes and cls < len(classes) else cls} {conf:.2f}"
        color = get_color(cls)

        # Semi-transparent overlay on box area
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        # Box outline
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.rectangle(
            img,
            (x1, label_y - label_size[1] - 4),
            (x1 + label_size[0], label_y),
            color,
            -1,
        )
        cv2.putText(
            img,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return img


def draw_obb(
    img: np.ndarray,
    boxes: np.ndarray,
    angles: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    classes: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Draw oriented bounding boxes on image.

    Args:
        boxes: (N, 4) - cx, cy, w, h
        angles: (N,) - rotation angles in radians
    """
    for i in range(len(boxes)):
        cx, cy, w, h = boxes[i]
        angle = angles[i]
        conf = scores[i]
        cls = int(class_ids[i])
        color = get_color(cls)

        # Calculate rotated rectangle corners
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        corners = np.array(
            [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]
        )
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_corners = corners @ rotation_matrix.T + [cx, cy]

        # Draw rotated box
        pts = rotated_corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, color, 2)

        # Draw label
        label = f"{classes[cls] if classes and cls < len(classes) else cls} {conf:.2f}"
        cv2.putText(
            img, label, (int(cx), int(cy) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

    return img


def draw_classify(
    img: np.ndarray,
    top_indices: np.ndarray,
    top_probs: np.ndarray,
    classes: Optional[List[str]] = None,
) -> np.ndarray:
    """Draw classification results as text overlay on image."""
    y_offset = 30
    for i in range(len(top_indices)):
        cls_idx = int(top_indices[i])
        prob = top_probs[i]
        label = f"#{i+1}: {classes[cls_idx] if classes and cls_idx < len(classes) else cls_idx} ({prob:.4f})"
        cv2.putText(
            img, label, (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
        )
        y_offset += 30
    return img


# =============================================================================
# Main Inferencer Class
# =============================================================================


class YoloOMInferencer:
    """
    YOLO OM Model Inferencer supporting multiple task types.

    Supported tasks:
    - detect: Object detection (imgsz=640)
    - classify: Image classification (imgsz=224)
    - pose: Keypoint detection (imgsz=640)
    - segment: Instance segmentation (imgsz=640)
    - obb: Oriented bounding box detection (imgsz=640)
    """

    def __init__(
        self,
        model_path: str,
        task: str = "detect",
        device_id: int = 0,
        imgsz: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ):
        self.model_path = model_path
        self.task = task
        self.device_id = device_id
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Load OM model
        print(f"Loading OM model: {model_path}")
        print(f"Task type: {task}")
        self.session = InferSession(device_id=device_id, model_path=model_path)

        # Get model info
        self.input_info = self.session.get_inputs()[0]
        self.input_shape = self.input_info.shape

        if len(self.input_shape) == 4:
            self.batch_size = self.input_shape[0] if self.input_shape[0] > 0 else 1
            self.input_height = self.input_shape[2]
            self.input_width = self.input_shape[3]
        else:
            raise ValueError(f"Unexpected input shape: {self.input_shape}")

        # Initialize letterbox transformer
        self.letterbox = LetterBox(
            new_shape=(self.input_height, self.input_width),
            auto=False,
            scale_fill=False,
        )

        print(f"Model input: {self.input_shape} (NCHW)")
        print(f"Input size: {self.input_width}x{self.input_height}")

    def preprocess(
        self, image_path: str
    ) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
        """Preprocess image using Ultralytics LetterBox."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        original_shape = img.shape[:2]  # (height, width)

        # Apply letterbox resize
        img_letterbox = self.letterbox(image=img)

        # Calculate scale ratio
        ratio = min(self.input_width / img.shape[1], self.input_height / img.shape[0])

        # BGR to RGB, HWC to CHW, normalize
        img_rgb = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)
        img_chw = img_rgb.transpose(2, 0, 1)
        img_normalized = img_chw.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        return img_batch, img, ratio, original_shape

    def infer(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Run inference on NPU."""
        outputs = self.session.infer([input_tensor], mode="static")
        return outputs

    def postprocess(
        self,
        outputs: List[np.ndarray],
        ratio: float,
        original_shape: Tuple[int, int],
        classes: Optional[List[str]] = None,
    ) -> Dict:
        """
        Postprocess outputs based on task type.

        Returns dict with task-specific results.
        """
        # Calculate padding offset (letterbox centers the image)
        pad_h = (self.input_height - original_shape[0] * ratio) / 2
        pad_w = (self.input_width - original_shape[1] * ratio) / 2

        if self.task == "detect":
            boxes, scores, class_ids = postprocess_detect(
                outputs[0], self.conf_thres, self.iou_thres
            )

            # Scale boxes back to original image
            detections = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                x1 = max(0, min((x1 - pad_w) / ratio, original_shape[1]))
                y1 = max(0, min((y1 - pad_h) / ratio, original_shape[0]))
                x2 = max(0, min((x2 - pad_w) / ratio, original_shape[1]))
                y2 = max(0, min((y2 - pad_h) / ratio, original_shape[0]))

                cls_idx = int(class_ids[i])
                detections.append(
                    {
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                        "conf": float(scores[i]),
                        "cls": cls_idx,
                        "cls_name": classes[cls_idx]
                        if classes and cls_idx < len(classes)
                        else str(cls_idx),
                    }
                )

            return {"detections": detections, "num_detections": len(detections)}

        elif self.task == "pose":
            boxes, scores, class_ids, keypoints = postprocess_pose(
                outputs[0], self.conf_thres
            )

            # Scale boxes and keypoints back to original image
            poses = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                x1 = max(0, min((x1 - pad_w) / ratio, original_shape[1]))
                y1 = max(0, min((y1 - pad_h) / ratio, original_shape[0]))
                x2 = max(0, min((x2 - pad_w) / ratio, original_shape[1]))
                y2 = max(0, min((y2 - pad_h) / ratio, original_shape[0]))

                # Scale keypoints
                kpts = keypoints[i].copy()
                kpts[:, 0] = (kpts[:, 0] - pad_w) / ratio
                kpts[:, 1] = (kpts[:, 1] - pad_h) / ratio

                poses.append(
                    {
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                        "conf": float(scores[i]),
                        "keypoints": kpts.tolist(),
                    }
                )

            return {"poses": poses, "num_poses": len(poses)}

        elif self.task == "segment":
            boxes, scores, class_ids, mask_coeffs = postprocess_segment(
                outputs, self.conf_thres, self.iou_thres
            )

            # Scale boxes
            detections = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                x1 = max(0, min((x1 - pad_w) / ratio, original_shape[1]))
                y1 = max(0, min((y1 - pad_h) / ratio, original_shape[0]))
                x2 = max(0, min((x2 - pad_w) / ratio, original_shape[1]))
                y2 = max(0, min((y2 - pad_h) / ratio, original_shape[0]))

                cls_idx = int(class_ids[i])
                mask_coeff = (
                    mask_coeffs[i].tolist()
                    if mask_coeffs is not None and i < len(mask_coeffs)
                    else None
                )
                detections.append(
                    {
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                        "conf": float(scores[i]),
                        "cls": cls_idx,
                        "cls_name": classes[cls_idx]
                        if classes and cls_idx < len(classes)
                        else str(cls_idx),
                        "mask_coeffs": mask_coeff,
                    }
                )

            return {"detections": detections, "num_detections": len(detections)}

        elif self.task == "obb":
            boxes, scores, class_ids, angles = postprocess_obb(
                outputs[0], self.conf_thres, self.iou_thres
            )

            # Scale boxes (cx, cy, w, h)
            obbs = []
            for i in range(len(boxes)):
                cx, cy, w, h = boxes[i]
                cx = (cx - pad_w) / ratio
                cy = (cy - pad_h) / ratio
                w = w / ratio
                h = h / ratio

                cls_idx = int(class_ids[i])
                obbs.append(
                    {
                        "box": [float(cx), float(cy), float(w), float(h)],
                        "angle": float(angles[i]),
                        "conf": float(scores[i]),
                        "cls": cls_idx,
                        "cls_name": classes[cls_idx]
                        if classes and cls_idx < len(classes)
                        else str(cls_idx),
                    }
                )

            return {"obbs": obbs, "num_obbs": len(obbs)}

        elif self.task == "classify":
            top_indices, top_probs = postprocess_classify(outputs[0], top_k=5)

            predictions = []
            for i in range(len(top_indices)):
                cls_idx = int(top_indices[i])
                predictions.append(
                    {
                        "cls": cls_idx,
                        "cls_name": classes[cls_idx]
                        if classes and cls_idx < len(classes)
                        else str(cls_idx),
                        "prob": float(top_probs[i]),
                    }
                )

            return {"predictions": predictions, "num_predictions": len(predictions)}

        return {}

    def __call__(self, image_path: str, classes: Optional[List[str]] = None) -> Dict:
        """Full inference pipeline."""
        start_time = time.time()

        # Preprocess
        preprocess_start = time.time()
        input_tensor, original_img, ratio, original_shape = self.preprocess(image_path)
        preprocess_time = time.time() - preprocess_start

        # Infer
        infer_start = time.time()
        outputs = self.infer(input_tensor)
        infer_time = time.time() - infer_start

        # Postprocess
        postprocess_start = time.time()
        result = self.postprocess(outputs, ratio, original_shape, classes)
        postprocess_time = time.time() - postprocess_start

        total_time = time.time() - start_time

        result.update(
            {
                "image_path": image_path,
                "original_shape": original_shape,
                "task": self.task,
                "timing": {
                    "preprocess_ms": preprocess_time * 1000,
                    "infer_ms": infer_time * 1000,
                    "postprocess_ms": postprocess_time * 1000,
                    "total_ms": total_time * 1000,
                },
                "original_image": original_img,
            }
        )

        return result

    def free_resource(self):
        """Release NPU resources."""
        self.session.free_resource()


def draw_results(result: Dict, output_path: str, classes: Optional[List[str]] = None):
    """Draw results on image based on task type."""
    img = result["original_image"].copy()
    task = result.get("task", "detect")

    if task == "detect":
        if result.get("detections"):
            boxes = np.array([d["box"] for d in result["detections"]])
            scores = np.array([d["conf"] for d in result["detections"]])
            class_ids = np.array([d["cls"] for d in result["detections"]])
            img = draw_detections(img, boxes, scores, class_ids, classes)

    elif task == "pose":
        if result.get("poses"):
            boxes = np.array([p["box"] for p in result["poses"]])
            scores = np.array([p["conf"] for p in result["poses"]])
            keypoints = np.array([p["keypoints"] for p in result["poses"]])
            img = draw_pose(img, boxes, scores, keypoints)

    elif task == "segment":
        if result.get("detections"):
            boxes = np.array([d["box"] for d in result["detections"]])
            scores = np.array([d["conf"] for d in result["detections"]])
            class_ids = np.array([d["cls"] for d in result["detections"]])
            mask_coeffs = np.array(
                [d.get("mask_coeffs", []) for d in result["detections"]]
            )
            img = draw_segment(img, boxes, scores, class_ids, mask_coeffs, classes)

    elif task == "obb":
        if result.get("obbs"):
            boxes = np.array([o["box"] for o in result["obbs"]])
            angles = np.array([o["angle"] for o in result["obbs"]])
            scores = np.array([o["conf"] for o in result["obbs"]])
            class_ids = np.array([o["cls"] for o in result["obbs"]])
            img = draw_obb(img, boxes, angles, scores, class_ids, classes)

    elif task == "classify":
        if result.get("predictions"):
            top_indices = np.array([p["cls"] for p in result["predictions"]])
            top_probs = np.array([p["prob"] for p in result["predictions"]])
            img = draw_classify(img, top_indices, top_probs, classes)

    cv2.imwrite(output_path, img)
    print(f"Saved result to: {output_path}")


# Default COCO classes for detection
COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# DOTA classes for OBB
DOTA_CLASSES = [
    "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool",
]


def main():
    parser = argparse.ArgumentParser(description="YOLO OM End-to-End Inference")
    parser.add_argument("--model", required=True, help="Path to YOLO OM model")
    parser.add_argument(
        "--source", required=True, help="Path to input image or directory"
    )
    parser.add_argument("--output", help="Path to output image or directory")
    parser.add_argument(
        "--task",
        choices=["detect", "classify", "pose", "segment", "obb"],
        default="detect",
        help="YOLO task type (default: detect)",
    )
    parser.add_argument("--device", type=int, default=0, help="NPU device ID")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--classes", nargs="+", default=None, help="Class names")
    parser.add_argument(
        "--no-draw", action="store_true", help="Don't draw results on image"
    )
    parser.add_argument(
        "--save-txt", action="store_true", help="Save results to txt file"
    )

    args = parser.parse_args()

    # Select default classes based on task
    if args.classes:
        classes = args.classes
    elif args.task == "obb":
        classes = DOTA_CLASSES
    elif args.task == "classify":
        # Try to load ImageNet class names from ultralytics
        try:
            from ultralytics.utils import ASSETS
            import json
            imagenet_path = Path(ASSETS).parent / "cfg" / "datasets" / "ImageNet.yaml"
            if imagenet_path.exists():
                import yaml
                with open(imagenet_path) as f:
                    data = yaml.safe_load(f)
                classes = list(data.get("names", {}).values())
            else:
                classes = None
        except Exception:
            classes = None
    else:
        classes = COCO_CLASSES

    inferencer = YoloOMInferencer(
        model_path=args.model,
        task=args.task,
        device_id=args.device,
        imgsz=args.imgsz,
        conf_thres=args.conf,
        iou_thres=args.iou,
    )

    # Get input files and determine output mode
    single_output_file = None  # If set, use this exact path for single-image output
    if os.path.isfile(args.source):
        input_files = [args.source]
        if args.output and os.path.splitext(args.output)[1]:
            # --output has a file extension -> treat as exact output file path
            single_output_file = args.output
            output_dir = os.path.dirname(args.output) or "."
        else:
            # --output is a directory or not specified
            output_dir = args.output or "."
    elif os.path.isdir(args.source):
        input_files = (
            list(Path(args.source).glob("*.jpg"))
            + list(Path(args.source).glob("*.jpeg"))
            + list(Path(args.source).glob("*.png"))
            + list(Path(args.source).glob("*.bmp"))
        )
        output_dir = args.output or "./results"
    else:
        print(f"Error: Source not found: {args.source}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    total_detections = 0
    total_time = 0

    for img_path in input_files:
        img_path = str(img_path)
        print(f"\nProcessing: {img_path}")

        result = inferencer(img_path, classes=classes)

        num_dets = result.get(
            "num_detections",
            result.get("num_poses", result.get("num_obbs", result.get("num_predictions", 0))),
        )
        print(f"  Results: {num_dets}")
        print(
            f"  Timing: pre={result['timing']['preprocess_ms']:.1f}ms, "
            f"infer={result['timing']['infer_ms']:.1f}ms, "
            f"post={result['timing']['postprocess_ms']:.1f}ms, "
            f"total={result['timing']['total_ms']:.1f}ms"
        )

        # Print results based on task type
        if args.task == "classify" and result.get("predictions"):
            print("  Top-5 predictions:")
            for p in result["predictions"]:
                print(f"    - {p['cls_name']}: {p['prob']:.4f}")
        else:
            detections = (
                result.get("detections") or result.get("poses") or result.get("obbs") or []
            )
            if detections:
                print("  Objects:")
                for det in detections[:5]:
                    if "keypoints" in det:
                        print(f"    - person: {det['conf']:.2f} at {det['box'][:4]}")
                    else:
                        box = det["box"]
                        print(
                            f"    - {det.get('cls_name', det.get('cls', '?'))}: {det['conf']:.2f} at {box}"
                        )
                if len(detections) > 5:
                    print(f"    ... and {len(detections) - 5} more")

        # Save output
        if single_output_file and len(input_files) == 1:
            output_path = single_output_file
        else:
            base_name = Path(img_path).stem
            output_path = os.path.join(output_dir, f"{base_name}_result.jpg")

        if not args.no_draw:
            draw_results(result, output_path, classes)

        if args.save_txt:
            txt_path = os.path.join(output_dir, f"{base_name}.txt")
            with open(txt_path, "w") as f:
                for det in detections:
                    if "keypoints" in det:
                        # Pose format: class conf x1 y1 x2 y2 kpts...
                        kpts_str = " ".join(
                            [f"{v:.1f}" for kpt in det["keypoints"] for v in kpt]
                        )
                        f.write(
                            f"0 {det['conf']:.4f} {det['box'][0]:.1f} {det['box'][1]:.1f} {det['box'][2]:.1f} {det['box'][3]:.1f} {kpts_str}\n"
                        )
                    else:
                        box = det["box"]
                        f.write(
                            f"{det['cls']} {det['conf']:.4f} {box[0]:.1f} {box[1]:.1f} {box[2]:.1f} {box[3]:.1f}\n"
                        )
            print(f"  Saved txt: {txt_path}")

        total_detections += num_dets
        total_time += result["timing"]["total_ms"]

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Task type: {args.task}")
    print(f"Images processed: {len(input_files)}")
    print(f"Total detections: {total_detections}")
    print(f"Average time: {total_time / len(input_files):.1f}ms")
    print(f"Average FPS: {1000 * len(input_files) / total_time:.1f}")
    print("=" * 60)

    inferencer.free_resource()


if __name__ == "__main__":
    main()
