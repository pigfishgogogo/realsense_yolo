# -*- coding: utf-8 -*-
"""旋转框(OBB) YOLOv8/YOLO26 ONNX 检测器"""

import time
from pathlib import Path

import cv2
import numpy as np

from .base import BaseYoloOnnxDetector
from .common import _decode_obb_scores, _rotated_iou, sigmoid


def _filename_suggests_raw_angle(model_path):
    """
    没有 onnx 包时的文件名兜底规则:
    文件名同时体现 obb 与 26 (如 phone26n-obb / yolov8n-obb26 / yolo26-obb)
    即认为角度分支直接输出原始弧度。
    """
    name = Path(model_path).name.lower()
    return "obb" in name and "26" in name


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
        return _filename_suggests_raw_angle(model_path)

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
        return _filename_suggests_raw_angle(model_path)


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
