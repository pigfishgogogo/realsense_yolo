# -*- coding: utf-8 -*-
"""
检测器公共工具: 常量 / 日志 / 图像预处理 / 模型与图片选取

说明:
    - 模型由 ultralytics 导出:
        普通检测(HBB): yolo export model=yolov8n.pt format=onnx imgsz=640
        旋转框(OBB):   yolo export model=best.pt format=onnx imgsz=640
    - 普通检测 ONNX 输出形状为 [1, 4+num_classes, 8400]:
        前 4 行为框 (cx, cy, w, h), 后面为各类别分数
    - OBB / OBB26 检测 ONNX 输出形状均为 [1, 4+num_classes+1, 8400]:
        前 4 行为框 (cx, cy, w, h), 中间 num_classes 列为类别分数,
        最后 1 列为角度原始值。
        * 标准 OBB (yolov8/yolo11): 角度为 logits, 需 sigmoid 后映射到 [-pi/4, 3pi/4)
        * OBB26 (yolo26): 角度已经直接输出为原始弧度, 不再做 sigmoid
    - 依赖: onnxruntime (或 onnxruntime-gpu), opencv-python, numpy, onnx (用于自动判断 OBB 类型)
"""

import logging
from pathlib import Path

import cv2
import numpy as np

# COCO 80 类默认类别名
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def setup_logger(log_path, name="yolov8_eval"):
    """同时输出到控制台和文件的 logger"""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_names(path):
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return COCO_NAMES


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """等比缩放并居中填充到 new_shape x new_shape, 返回 (填充图, 缩放比, 左右填充, 上下填充)"""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = round(w * r), round(h * r)
    dw, dh = (new_shape - new_w) / 2, (new_shape - new_h) / 2

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return padded, r, left, top


def _providers(tensorrt):
    """选择 onnxruntime 执行器"""
    import onnxruntime as ort
    available = set(ort.get_available_providers())
    cuda_opts = {"cudnn_conv_algo_search": "HEURISTIC", "arena_extend_strategy": "kSameAsRequested"}
    if tensorrt and "TensorrtExecutionProvider" in available:
        return [("TensorrtExecutionProvider", {}),
                ("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def sigmoid(x):
    """数值稳定的 sigmoid"""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _decode_obb_scores(cls_raw):
    """
    OBB 类别分支输出可能是 logits, 也可能已经在 graph 内做了 sigmoid。
    用数值范围做简单判断, 尽量保证两种情况都可用。
    """
    cls_raw = np.asarray(cls_raw)
    if cls_raw.ndim == 1:
        cls_raw = cls_raw[:, None]
    if cls_raw.min() < -0.1 or cls_raw.max() > 1.1:
        return sigmoid(cls_raw)
    return cls_raw


def _rotated_iou(r1, r2):
    """计算两个 cv2.RotatedRect 的 IoU"""
    inter, pts = cv2.rotatedRectangleIntersection(r1, r2)
    if inter == 0 or pts is None:
        return 0.0
    inter_area = cv2.contourArea(pts)
    if inter_area <= 0:
        return 0.0
    a1 = r1[1][0] * r1[1][1]
    a2 = r2[1][0] * r2[1][1]
    return inter_area / (a1 + a2 - inter_area + 1e-6)


def pick_model(model_arg):
    """--model 未指定时, 取 models/ 下第一个 .onnx"""
    if model_arg:
        return Path(model_arg)
    candidates = sorted(Path("models").glob("*.onnx"))
    if not candidates:
        return None
    return candidates[0]


def collect_images(images_arg):
    """图片目录或单张图片 -> 图片路径列表"""
    p = Path(images_arg)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS)
    return []


def make_run_dir(base_dir):
    """在 base_dir 下新建以当前时间命名的子目录并返回, 避免覆盖历史结果"""
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def log_detections(logger, prefix, detections, label_fn):
    """统一格式的检测日志, 每张图/帧的结果"""
    for box, conf, cls_id in detections:
        if len(box) == 4:
            x1, y1, x2, y2 = box
            logger.info("%s - %s (conf=%.3f) 框: (%.1f, %.1f, %.1f, %.1f)",
                        prefix, label_fn(cls_id), conf, x1, y1, x2, y2)
        else:
            cx, cy, w, h, angle = box
            logger.info("%s - %s (conf=%.3f) OBB: cx=%.1f cy=%.1f w=%.1f h=%.1f angle=%.3f",
                        prefix, label_fn(cls_id), conf, cx, cy, w, h, angle)
