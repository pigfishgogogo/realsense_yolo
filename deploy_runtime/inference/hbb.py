# -*- coding: utf-8 -*-
"""普通水平框(HBB) YOLOv8 ONNX 检测器"""

import time

import cv2
import numpy as np

from .base import BaseYoloOnnxDetector


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
