# -*- coding: utf-8 -*-
"""检测器工厂: 根据模型输出自动判断 HBB / OBB 并创建对应检测器"""

from pathlib import Path
import logging

from .common import COCO_NAMES
from .hbb import YoloOnnxDetector
from .obb import YoloOBBOnnxDetector


def is_obb_model(model_path, names, logger=None):
    """
    根据 ONNX 输出通道数判断模型是否为 OBB。
    通道数与 4+nc / 5+nc 均不匹配时 (通常是 --names 类别数与模型不符), 输出警告。
    """
    import onnxruntime as ort
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    output_shape = session.get_outputs()[0].shape
    channels = output_shape[1] if len(output_shape) >= 2 and isinstance(output_shape[1], int) else None
    if channels is None:
        return False
    det_channels = 4 + len(names)
    obb_channels = 5 + len(names)  # 4 box + num_classes + 1 angle
    if channels not in (det_channels, obb_channels):
        log = logger or logging.getLogger("yolov8_eval")
        log.warning(
            "模型输出通道数 %d 与 HBB(4+%d=%d) / OBB(5+%d=%d) 均不匹配, "
            "请检查 --names 类别数是否与训练模型一致, 否则可能误判模型类型",
            channels, len(names), det_channels, len(names), obb_channels)
    return channels == obb_channels


def create_yolo_detector(model_path, imgsz=None, conf=0.25, iou=0.45,
                         names=None, tensorrt=False, warmup=True, logger=None, obb=None):
    """
    工厂函数: 自动判断模型类型并创建对应检测器
    obb: None 自动判断, True 强制 OBB, False 强制 HBB
    """
    names = names or COCO_NAMES
    force_obb = obb
    if force_obb is None:
        force_obb = is_obb_model(model_path, names, logger=logger)

    if logger:
        logger.info("模型 %s 按 %s 解析", Path(model_path).name, "OBB" if force_obb else "HBB")

    cls = YoloOBBOnnxDetector if force_obb else YoloOnnxDetector
    return cls(model_path, imgsz=imgsz, conf=conf, iou=iou,
               names=names, tensorrt=tensorrt, warmup=warmup, logger=logger)
