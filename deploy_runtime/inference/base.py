# -*- coding: utf-8 -*-
"""YOLOv8 ONNX 检测器基类: 模型加载 / 预热 / 公共预处理"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .common import COCO_NAMES, _providers, letterbox


class BaseYoloOnnxDetector(ABC):
    """YOLOv8 ONNX 检测器基类: 负责模型加载、预处理、公共工具"""

    def __init__(self, model_path, imgsz=None, conf=0.25, iou=0.45,
                 names=None, tensorrt=False, warmup=True, logger=None):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        self.conf = conf
        self.iou = iou
        self.names = names or COCO_NAMES
        self.log = logger or logging.getLogger("yolov8_eval")

        import onnxruntime as ort
        providers = _providers(tensorrt)
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.log.info("onnxruntime 可用执行器: %s, 本次使用: %s",
                      sorted(set(ort.get_available_providers())),
                      self.session.get_providers()[0])

        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.imgsz = imgsz or (input_shape[2] if isinstance(input_shape[2], int) and input_shape[2] > 0 else 640)
        self.log.info("模型输入: name=%s shape=%s, 推理尺寸=%d", self.input_name, input_shape, self.imgsz)
        self.log.info("类别数: %d", len(self.names))

        if warmup:
            dummy = np.zeros((1, 3, self.imgsz, self.imgsz), dtype=np.float32)
            self.log.info("预热推理中...")
            t_warm = time.perf_counter()
            self.session.run(None, {self.input_name: dummy})
            self.log.info("预热完成 (%.1f s)", time.perf_counter() - t_warm)
        else:
            self.log.info("已跳过预热, 首张图片耗时会包含GPU初始化开销")

    @abstractmethod
    def infer(self, img):
        """对一张 BGR 图片做推理, 返回 (detections, infer_ms)"""
        pass

    @abstractmethod
    def annotate(self, img, detections):
        """返回画好检测框的图片副本"""
        pass

    def label_of(self, cls_id):
        return self.names[cls_id] if cls_id < len(self.names) else f"class{cls_id}"

    def _preprocess(self, img):
        padded, r, pad_left, pad_top = letterbox(img, self.imgsz)
        blob = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # BGR->RGB, HWC->NCHW
        return np.ascontiguousarray(blob), r, pad_left, pad_top
