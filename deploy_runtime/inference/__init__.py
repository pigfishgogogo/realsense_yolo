# -*- coding: utf-8 -*-
"""
deploy_runtime.inference 包: YOLOv8/YOLO26 ONNX 检测器 (基于 onnxruntime)

封装模型加载、图像预处理、推理、NMS 后处理、画框等公共逻辑,
供 deploy_runtime/scripts/ 下的检测脚本复用。

    - 普通检测(HBB): 输出 [1, 4+num_classes, 8400], 前 4 维为 cx, cy, w, h
    - 旋转框(OBB):   输出 [1, 4+num_classes+1, 8400], 最后 1 维为角度
      标准 OBB (yolov8/yolo11): 角度 logits, sigmoid 后映射到 [-pi/4, 3pi/4)
      OBB26 / yolo26: 直接输出原始弧度
      本实现会根据 ONNX graph 自动判断属于哪一种。
    - HBB 分支保持原项目行为, 直接用输出分数当置信度。
    - 依赖: onnxruntime (或 onnxruntime-gpu), opencv-python, numpy, onnx (用于自动判断 OBB 类型)

用法:
    from deploy_runtime.inference import create_yolo_detector, load_names, setup_logger
"""

from .base import BaseYoloOnnxDetector
from .common import (
    COCO_NAMES,
    IMG_EXTS,
    _decode_obb_scores,
    _rotated_iou,
    collect_images,
    letterbox,
    load_names,
    log_detections,
    make_run_dir,
    pick_model,
    setup_logger,
    sigmoid,
)
from .factory import create_yolo_detector, is_obb_model
from .hbb import YoloOnnxDetector
from .obb import YoloOBBOnnxDetector

__all__ = [
    "BaseYoloOnnxDetector",
    "COCO_NAMES",
    "IMG_EXTS",
    "YoloOnnxDetector",
    "YoloOBBOnnxDetector",
    "collect_images",
    "create_yolo_detector",
    "is_obb_model",
    "letterbox",
    "load_names",
    "log_detections",
    "make_run_dir",
    "pick_model",
    "setup_logger",
    "sigmoid",
]
