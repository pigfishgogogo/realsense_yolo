# -*- coding: utf-8 -*-
"""
YOLOv8 ONNX 检测器基类 (基于 onnxruntime)

封装模型加载、图像预处理、推理、NMS 后处理、画框等公共逻辑,
供 image_evaluate.py / video_evaluate.py 继承复用。

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
      本实现会根据 ONNX graph 自动判断属于哪一种。
    - 本实现会根据数值范围自动判断类别分数是否已经过 sigmoid。
    - HBB 分支保持原项目行为, 直接用输出分数当置信度。
    - 依赖: onnxruntime (或 onnxruntime-gpu), opencv-python, numpy, onnx (用于自动判断 OBB 类型)
"""

import logging
import time
from abc import ABC, abstractmethod
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


def _is_obb26_raw_angle(model_path):
    """
    通过 ONNX graph 判断角度分支是否直接输出原始弧度 (OBB26/yolo26)。
    标准 OBB (yolov8/yolo11) 在 graph 内会对角度做 sigmoid + 平移;
    OBB26 则把 logits / 弧度直接输出, 不带最终的 sigmoid 变换。
    """
    try:
        import onnx
    except Exception:
        # 没有 onnx 时按文件名做简单兜底
        name = Path(model_path).name.lower()
        return "obb26" in name or "yolo26" in name

    try:
        model = onnx.load(str(model_path))
        out_name = model.graph.output[0].name
        final_node = None
        for n in model.graph.node:
            if out_name in n.output:
                final_node = n
                break
        if final_node is None or final_node.op_type != "Concat":
            return False

        # 最后一个 concat 输入对应角度分支
        angle_input = final_node.input[-1]
        out_to_node = {}
        for n in model.graph.node:
            for o in n.output:
                out_to_node[o] = n

        # 仅在这些 element-wise / shape 节点中继续向上追溯, 遇到 Conv/MatMul 等则停止
        passthrough = {
            "Concat", "Reshape", "Transpose", "Sub", "Mul", "Add", "Div",
            "Cast", "Split", "Flatten", "Squeeze", "Unsqueeze", "Gather", "Shape",
        }

        def has_final_sigmoid(name, seen=None):
            if seen is None:
                seen = set()
            if name in seen:
                return False
            seen.add(name)
            node = out_to_node.get(name)
            if node is None:
                return False
            if node.op_type == "Sigmoid":
                return True
            if node.op_type in passthrough:
                return any(has_final_sigmoid(inp, seen) for inp in node.input)
            return False

        return not has_final_sigmoid(angle_input)
    except Exception:
        name = Path(model_path).name.lower()
        return "obb26" in name or "yolo26" in name


def _decode_obb_angle(angle_raw, raw_radians=False):
    """
    将 OBB 角度分支映射到弧度。
    标准 OBB: angle = sigmoid(logits) * pi - pi/4, 范围 [-pi/4, 3pi/4)
    OBB26: 角度已经直接输出为弧度, 直接使用。
    """
    angle_raw = np.asarray(angle_raw)
    angle_raw = np.squeeze(angle_raw)
    if raw_radians:
        return angle_raw
    angle_norm = sigmoid(angle_raw)
    return angle_norm * np.pi - np.pi / 4


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
        self.log.info("onnxruntime 可用执行器: %s, 本次使用: %s",
                      sorted(set(ort.get_available_providers())),
                      [p if isinstance(p, str) else p[0] for p in providers])
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


class YoloOnnxDetector(BaseYoloOnnxDetector):
    """普通水平框(HBB) YOLOv8 ONNX 检测器"""

    def infer(self, img):
        """
        返回: (detections, infer_ms)
            detections: [(x1, y1, x2, y2), conf, cls_id], 坐标为原图像素坐标
        """
        blob, ratio, pad_left, pad_top = self._preprocess(img)

        t0 = time.perf_counter()
        output = self.session.run(None, {self.input_name: blob})[0]
        infer_ms = (time.perf_counter() - t0) * 1000

        detections = self._postprocess(output, ratio, pad_left, pad_top)
        return detections, infer_ms

    def annotate(self, img, detections):
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

    def _postprocess(self, output, ratio, pad_left, pad_top):
        """解析 HBB 输出 [1, 4+num_classes, 8400]"""
        preds = np.squeeze(output, axis=0)
        if preds.shape[0] < preds.shape[1]:  # (C, N) -> (N, C)
            preds = preds.T

        boxes_cxcywh = preds[:, :4]
        scores = preds[:, 4:]  # HBB 保持原项目行为: 用输出分数当置信度
        cls_ids = scores.argmax(axis=1)
        confs = scores[np.arange(len(scores)), cls_ids]

        keep = confs >= self.conf
        boxes_cxcywh, confs, cls_ids = boxes_cxcywh[keep], confs[keep], cls_ids[keep]
        if len(boxes_cxcywh) == 0:
            return []

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


class YoloOBBOnnxDetector(BaseYoloOnnxDetector):
    """旋转框(OBB) YOLOv8/YOLO26 ONNX 检测器"""

    def __init__(self, model_path, imgsz=None, conf=0.25, iou=0.45,
                 names=None, tensorrt=False, warmup=True, logger=None):
        super().__init__(model_path, imgsz=imgsz, conf=conf, iou=iou,
                         names=names, tensorrt=tensorrt, warmup=warmup, logger=logger)
        self.angle_is_raw = _is_obb26_raw_angle(self.model_path)
        self.log.info("OBB 角度解析方式: %s",
                      "原始弧度 (OBB26 / yolo26)" if self.angle_is_raw else "sigmoid 归一化 (标准 OBB)")

    def infer(self, img):
        """
        返回: (detections, infer_ms)
            detections: [(cx, cy, w, h, angle_rad), conf, cls_id], 坐标为原图像素坐标
        """
        blob, ratio, pad_left, pad_top = self._preprocess(img)

        t0 = time.perf_counter()
        output = self.session.run(None, {self.input_name: blob})[0]
        infer_ms = (time.perf_counter() - t0) * 1000

        detections = self._postprocess(output, ratio, pad_left, pad_top)
        return detections, infer_ms

    def annotate(self, img, detections):
        result = img.copy()
        for (cx, cy, w, h, angle_rad), conf, cls_id in detections:
            label = f"{self.label_of(cls_id)} {conf:.2f}"
            color = tuple(int(c) for c in np.random.default_rng(cls_id).integers(0, 255, 3))
            angle_deg = float(np.degrees(angle_rad))
            rect = ((float(cx), float(cy)), (float(w), float(h)), angle_deg)
            pts = cv2.boxPoints(rect).astype(np.int32)
            cv2.polylines(result, [pts], True, color, 2)
            top_pt = pts[np.argmin(pts[:, 1])]
            tx, ty = int(top_pt[0]), int(top_pt[1])
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(result, (tx, ty - th - 8), (tx + tw, ty), color, -1)
            cv2.putText(result, label, (tx, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        return result

    def _postprocess(self, output, ratio, pad_left, pad_top):
        """
        解析 OBB 输出 [1, 4+num_classes+1, 8400]:
            0:3   - cx, cy, w, h (letterbox 坐标)
            4:4+nc - 类别分数
            4+nc:  - 角度原始值
        """
        preds = np.squeeze(output, axis=0)
        if preds.ndim != 2:
            raise ValueError(f"OBB 输出应为二维 (C,N) 或 (N,C), 实际为 {preds.shape}")
        if preds.shape[0] < preds.shape[1]:  # (C, N) -> (N, C)
            preds = preds.T

        n, c = preds.shape
        if c < 6:
            raise ValueError(f"OBB 输出通道数不足: {c}, 至少需要 4+1+1")
        nc = c - 5

        boxes_cxcywh = preds[:, :4]
        cls_scores = _decode_obb_scores(preds[:, 4:4 + nc])
        angle_rad = _decode_obb_angle(preds[:, 4 + nc:], raw_radians=self.angle_is_raw)

        cls_ids = cls_scores.argmax(axis=1)
        confs = cls_scores[np.arange(len(cls_scores)), cls_ids]

        keep = confs >= self.conf
        boxes_cxcywh, angle_rad, confs, cls_ids = (
            boxes_cxcywh[keep], angle_rad[keep], confs[keep], cls_ids[keep]
        )
        if len(boxes_cxcywh) == 0:
            return []

        # 映射回原始图像坐标
        cx = (boxes_cxcywh[:, 0] - pad_left) / ratio
        cy = (boxes_cxcywh[:, 1] - pad_top) / ratio
        w = boxes_cxcywh[:, 2] / ratio
        h = boxes_cxcywh[:, 3] / ratio

        # 1) 用旋转框外接水平框做快速预筛选, 控制后续旋转 NMS 的候选数量
        half_w, half_h = w / 2, h / 2
        abs_cos = np.abs(np.cos(angle_rad))
        abs_sin = np.abs(np.sin(angle_rad))
        dx = half_w * abs_cos + half_h * abs_sin
        dy = half_w * abs_sin + half_h * abs_cos
        boxes_xyxy = np.stack([cx - dx, cy - dy, cx + dx, cy + dy], axis=1).astype(np.float32)

        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_xyxy.tolist(),
            scores=confs.tolist(),
            score_threshold=self.conf,
            nms_threshold=self.iou,
        )
        indices = np.array(indices).flatten() if len(indices) > 0 else np.array([], dtype=int)
        if len(indices) == 0:
            return []

        cx, cy, w, h = cx[indices], cy[indices], w[indices], h[indices]
        angle_rad, confs, cls_ids = angle_rad[indices], confs[indices], cls_ids[indices]

        # 2) 真正的旋转框 NMS
        rects = [
            ((float(cx[i]), float(cy[i])), (float(w[i]), float(h[i])), float(np.degrees(angle_rad[i])))
            for i in range(len(cx))
        ]
        order = np.argsort(-confs)
        suppressed = np.zeros(len(order), dtype=bool)
        keep_idx = []
        for pos, i in enumerate(order):
            if suppressed[i]:
                continue
            keep_idx.append(i)
            for j in order[pos + 1:]:
                if suppressed[j]:
                    continue
                if cls_ids[i] == cls_ids[j] and _rotated_iou(rects[i], rects[j]) > self.iou:
                    suppressed[j] = True

        return [
            ((float(cx[i]), float(cy[i]), float(w[i]), float(h[i]), float(angle_rad[i])),
             float(confs[i]), int(cls_ids[i]))
            for i in keep_idx
        ]


def is_obb_model(model_path, names):
    """根据 ONNX 输出通道数判断模型是否为 OBB"""
    import onnxruntime as ort
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    output_shape = session.get_outputs()[0].shape
    channels = output_shape[1] if len(output_shape) >= 2 and isinstance(output_shape[1], int) else None
    if channels is None:
        return False
    det_channels = 4 + len(names)
    obb_channels = 5 + len(names)  # 4 box + num_classes + 1 angle
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
        force_obb = is_obb_model(model_path, names)

    if logger:
        logger.info("模型 %s 按 %s 解析", Path(model_path).name, "OBB" if force_obb else "HBB")

    cls = YoloOBBOnnxDetector if force_obb else YoloOnnxDetector
    return cls(model_path, imgsz=imgsz, conf=conf, iou=iou,
               names=names, tensorrt=tensorrt, warmup=warmup, logger=logger)


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
