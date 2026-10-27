# -*- coding: utf-8 -*-
"""
YOLOv8 ONNX 检测器基类 (基于 onnxruntime)

封装模型加载、图像预处理、推理、NMS 后处理、画框等公共逻辑,
供 image_evaluate.py / video_evaluate.py 继承复用。

说明:
    - 模型由 ultralytics 导出: yolo export model=yolov8n.pt format=onnx imgsz=640
    - ONNX 输出形状为 [1, 84, 8400]: 前 4 行为框 (cx, cy, w, h), 后 80 行为 COCO 各类别分数
    - 依赖: onnxruntime (或 onnxruntime-gpu), opencv-python, numpy
"""

import logging
import time
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


class YoloOnnxDetector:
    """YOLOv8 ONNX 检测器: 加载模型, 对单张 BGR 图片做推理并返回检测结果"""

    def __init__(self, model_path, imgsz=None, conf=0.25, iou=0.45,
                 names=None, tensorrt=False, warmup=True, logger=None):
        """
        参数:
            model_path: .onnx 模型路径
            imgsz:      模型输入尺寸, None 则取模型本身输入尺寸
            conf/iou:   置信度阈值 / NMS IOU 阈值
            names:      类别名列表, None 用内置 COCO 80 类
            tensorrt:   是否启用 TensorRT 执行器 (需已安装 TensorRT 库)
            warmup:     是否预热 (避免首次推理的 GPU 初始化开销计入计时)
            logger:     外部传入的 logger, None 则新建一个只输出控制台的
        """
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        self.conf = conf
        self.iou = iou
        self.names = names or COCO_NAMES
        self.log = logger or logging.getLogger("yolov8_eval")

        import onnxruntime as ort

        # 按优先级选择执行器; TensorRT 默认不用(缺库时会有报错噪音)
        # 注意: CUDA 默认对卷积做穷举算法搜索(EXHAUSTIVE), 大模型首次推理要 1~2 分钟,
        # 看起来像"卡死"。这里改用 HEURISTIC(按经验选算法), 实测推理速度无差异。
        available = set(ort.get_available_providers())
        cuda_opts = {"cudnn_conv_algo_search": "HEURISTIC", "arena_extend_strategy": "kSameAsRequested"}
        if tensorrt and "TensorrtExecutionProvider" in available:
            providers = [("TensorrtExecutionProvider", {}),
                         ("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
        elif "CUDAExecutionProvider" in available:
            providers = [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.log.info("onnxruntime 可用执行器: %s, 本次使用: %s",
                      sorted(available), [p if isinstance(p, str) else p[0] for p in providers])
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)

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

    def infer(self, img):
        """
        对一张 BGR 图片做推理
        返回: (detections, infer_ms)
            detections: [(x1, y1, x2, y2), conf, cls_id] 列表, 坐标为原图像素坐标
        """
        blob, ratio, pad_left, pad_top = self._preprocess(img)

        t0 = time.perf_counter()
        output = self.session.run(None, {self.input_name: blob})[0]
        infer_ms = (time.perf_counter() - t0) * 1000

        detections = self._postprocess(output, ratio, pad_left, pad_top)
        return detections, infer_ms

    def label_of(self, cls_id):
        return self.names[cls_id] if cls_id < len(self.names) else f"class{cls_id}"

    def annotate(self, img, detections):
        """返回画好检测框的图片副本"""
        result = img.copy()
        for (x1, y1, x2, y2), conf, cls_id in detections:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
            label = f"{self.label_of(cls_id)} {conf:.2f}"
            color = tuple(int(c) for c in np.random.default_rng(cls_id).integers(0, 255, 3))

            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
            cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        return result

    def _preprocess(self, img):
        padded, r, pad_left, pad_top = letterbox(img, self.imgsz)
        blob = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # BGR->RGB, HWC->NCHW
        return np.ascontiguousarray(blob), r, pad_left, pad_top

    def _postprocess(self, output, ratio, pad_left, pad_top):
        """解析 YOLOv8 输出 [1, 84, 8400] 或 [1, 8400, 84]"""
        preds = np.squeeze(output, axis=0)
        if preds.shape[0] < preds.shape[1]:  # (84, 8400) -> (8400, 84)
            preds = preds.T

        boxes_cxcywh = preds[:, :4]
        scores = preds[:, 4:]
        cls_ids = scores.argmax(axis=1)
        confs = scores[np.arange(len(scores)), cls_ids]

        keep = confs >= self.conf
        boxes_cxcywh, confs, cls_ids = boxes_cxcywh[keep], confs[keep], cls_ids[keep]
        if len(boxes_cxcywh) == 0:
            return []

        # (cx, cy, w, h) -> (x1, y1, x2, y2), 并映射回原图坐标
        x1 = (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2 - pad_left) / ratio
        y1 = (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2 - pad_top) / ratio
        x2 = (boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2 - pad_left) / ratio
        y2 = (boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2 - pad_top) / ratio
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_xyxy.tolist(),
            scores=confs.tolist(),
            score_threshold=self.conf,
            nms_threshold=self.iou,
        )
        indices = np.array(indices).flatten() if len(indices) > 0 else []

        return [(boxes_xyxy[i], confs[i], cls_ids[i]) for i in indices]


def log_detections(logger, prefix, detections, label_fn):
    """统一格式的检测日志, 每张图/帧的结果"""
    for (x1, y1, x2, y2), conf, cls_id in detections:
        logger.info("%s - %s (conf=%.3f) 框: (%.1f, %.1f, %.1f, %.1f)",
                    prefix, label_fn(cls_id), conf, x1, y1, x2, y2)
