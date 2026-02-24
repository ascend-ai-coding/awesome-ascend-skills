#!/usr/bin/env python3
"""
YOLO End-to-End OM Inference Script

Provides Ultralytics-like inference interface for YOLO OM models on Ascend NPU.
Uses Ultralytics preprocessing (LetterBox) and postprocessing.

Usage:
    # Single image inference
    python3 yolo_om_infer.py --model yolo.om --source image.jpg --output result.jpg

    # Multiple images
    python3 yolo_om_infer.py --model yolo.om --source images/ --output results/

    # With custom confidence threshold
    python3 yolo_om_infer.py --model yolo.om --source image.jpg --conf 0.5

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


def nms_numpy(boxes, scores, iou_threshold):
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


class YoloOMInferencer:
    """
    YOLO OM Model Inferencer using Ultralytics preprocessing.

    Provides end-to-end inference with proper letterbox scaling.
    """

    def __init__(
        self, model_path, device_id=0, imgsz=640, conf_thres=0.25, iou_thres=0.45
    ):
        """
        Initialize YOLO OM inferencer.

        Args:
            model_path: Path to OM model file
            device_id: NPU device ID
            imgsz: Input image size (default 640)
            conf_thres: Confidence threshold for NMS
            iou_thres: IoU threshold for NMS
        """
        self.model_path = model_path
        self.device_id = device_id
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Load OM model
        print(f"Loading OM model: {model_path}")
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

    def preprocess(self, image_path):
        """
        Preprocess image using Ultralytics LetterBox.

        Args:
            image_path: Path to input image

        Returns:
            tuple: (preprocessed_tensor, original_image, letterbox_ratio)
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        original_shape = img.shape[:2]  # (height, width)

        # Apply letterbox resize
        img_letterbox = self.letterbox(image=img)

        # Calculate scale ratio for converting back
        # letterbox scales and pads the image
        ratio = min(self.input_width / img.shape[1], self.input_height / img.shape[0])

        # BGR to RGB
        img_rgb = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)

        # HWC to CHW
        img_chw = img_rgb.transpose(2, 0, 1)

        # Normalize to 0-1 and convert to float32
        img_normalized = img_chw.astype(np.float32) / 255.0

        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)

        return img_batch, img, ratio, original_shape

    def infer(self, input_tensor):
        """
        Run inference on NPU.

        Args:
            input_tensor: Preprocessed input tensor (NCHW format)

        Returns:
            numpy array: Raw model output
        """
        outputs = self.session.infer([input_tensor], mode="static")
        return outputs

    def postprocess(self, outputs, ratio, original_shape, classes=None):
        """
        Postprocess YOLO outputs: apply NMS and scale boxes back to original image.

        YOLO output format: [batch, num_boxes, 6] where 6 = [x1, y1, x2, y2, conf, cls]

        Args:
            outputs: Raw model outputs
            ratio: Scale ratio from preprocessing
            original_shape: Original image shape (height, width)
            classes: List of class names (optional)

        Returns:
            list: Detection results, each element is dict with 'box', 'conf', 'cls'
        """
        # Get predictions
        if isinstance(outputs, (list, tuple)):
            pred = outputs[0]
        else:
            pred = outputs

        # pred shape: [batch, num_boxes, 6]
        pred = pred[0]  # Remove batch dimension: [num_boxes, 6]

        # Extract components - format is [x1, y1, x2, y2, conf, cls]
        boxes = pred[:, :4]  # [x1, y1, x2, y2]
        confs = pred[:, 4]  # confidence scores
        clses = pred[:, 5]  # class indices

        # Filter by confidence threshold
        mask = confs >= self.conf_thres
        boxes = boxes[mask]
        confs = confs[mask]
        clses = clses[mask]

        # Apply NMS
        if len(boxes) > 0:
            if torch is not None and torchvision_nms is not None:
                # Use torchvision NMS (faster)
                keep = torchvision_nms(
                    torch.from_numpy(boxes.copy()),
                    torch.from_numpy(confs.copy()),
                    self.iou_thres,
                ).numpy()
            else:
                # Use numpy NMS
                keep = nms_numpy(boxes, confs, self.iou_thres)

            boxes = boxes[keep]
            confs = confs[keep]
            clses = clses[keep]

        # Calculate padding offset (letterbox centers the image)
        pad_h = (self.input_height - original_shape[0] * ratio) / 2
        pad_w = (self.input_width - original_shape[1] * ratio) / 2

        # Scale boxes back to original image size and remove padding offset
        results = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]

            # Remove padding offset and scale back
            x1 = (x1 - pad_w) / ratio
            y1 = (y1 - pad_h) / ratio
            x2 = (x2 - pad_w) / ratio
            y2 = (y2 - pad_h) / ratio

            # Clip to image bounds
            x1 = max(0, min(x1, original_shape[1]))
            y1 = max(0, min(y1, original_shape[0]))
            x2 = max(0, min(x2, original_shape[1]))
            y2 = max(0, min(y2, original_shape[0]))

            cls_idx = int(clses[i])

            results.append(
                {
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": float(confs[i]),
                    "cls": cls_idx,
                    "cls_name": classes[cls_idx]
                    if classes and cls_idx < len(classes)
                    else str(cls_idx),
                }
            )

        return results

    def __call__(self, image_path, classes=None):
        """
        Full inference pipeline.

        Args:
            image_path: Path to input image
            classes: Optional list of class names

        Returns:
            dict: Results containing detections, timing info, etc.
        """
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
        detections = self.postprocess(outputs, ratio, original_shape, classes)
        postprocess_time = time.time() - postprocess_start

        total_time = time.time() - start_time

        return {
            "image_path": image_path,
            "original_shape": original_shape,
            "detections": detections,
            "num_detections": len(detections),
            "timing": {
                "preprocess_ms": preprocess_time * 1000,
                "infer_ms": infer_time * 1000,
                "postprocess_ms": postprocess_time * 1000,
                "total_ms": total_time * 1000,
            },
            "original_image": original_img,
        }

    def free_resource(self):
        """Release NPU resources."""
        self.session.free_resource()


def draw_results(result, output_path, classes=None):
    """
    Draw detection results on image and save.

    Args:
        result: Detection result from inferencer
        output_path: Path to save output image
        classes: List of class names
    """
    img = result["original_image"].copy()

    def get_color(cls_id):
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
        ]
        return colors_list[cls_id % len(colors_list)]

    for det in result["detections"]:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        conf = det["conf"]
        cls = det["cls"]
        label = f"{det['cls_name']} {conf:.2f}"
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

    cv2.imwrite(output_path, img)
    print(f"Saved result to: {output_path}")


# Default COCO classes
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


def main():
    parser = argparse.ArgumentParser(description="YOLO OM End-to-End Inference")
    parser.add_argument("--model", required=True, help="Path to YOLO OM model")
    parser.add_argument(
        "--source", required=True, help="Path to input image or directory"
    )
    parser.add_argument("--output", help="Path to output image or directory")
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

    classes = args.classes if args.classes else COCO_CLASSES

    inferencer = YoloOMInferencer(
        model_path=args.model,
        device_id=args.device,
        imgsz=args.imgsz,
        conf_thres=args.conf,
        iou_thres=args.iou,
    )

    if os.path.isfile(args.source):
        input_files = [args.source]
        output_dir = os.path.dirname(args.output) if args.output else "."
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

        print(f"  Detections: {result['num_detections']}")
        print(
            f"  Timing: pre={result['timing']['preprocess_ms']:.1f}ms, "
            f"infer={result['timing']['infer_ms']:.1f}ms, "
            f"post={result['timing']['postprocess_ms']:.1f}ms, "
            f"total={result['timing']['total_ms']:.1f}ms"
        )

        if result["detections"]:
            print("  Objects:")
            for det in result["detections"][:5]:
                box = det["box"]
                print(
                    f"    - {det['cls_name']}: {det['conf']:.2f} at [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]"
                )
            if len(result["detections"]) > 5:
                print(f"    ... and {len(result['detections']) - 5} more")

        base_name = Path(img_path).stem
        output_path = os.path.join(output_dir, f"{base_name}_result.jpg")

        if not args.no_draw:
            draw_results(result, output_path, classes)

        if args.save_txt:
            txt_path = os.path.join(output_dir, f"{base_name}.txt")
            with open(txt_path, "w") as f:
                for det in result["detections"]:
                    x1, y1, x2, y2 = det["box"]
                    f.write(
                        f"{det['cls']} {det['conf']:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n"
                    )
            print(f"  Saved txt: {txt_path}")

        total_detections += result["num_detections"]
        total_time += result["timing"]["total_ms"]

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Images processed: {len(input_files)}")
    print(f"Total detections: {total_detections}")
    print(f"Average time: {total_time / len(input_files):.1f}ms")
    print(f"Average FPS: {1000 * len(input_files) / total_time:.1f}")
    print("=" * 60)

    inferencer.free_resource()


if __name__ == "__main__":
    main()
