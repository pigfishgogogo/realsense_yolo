# -*- coding: utf-8 -*-
"""
YOLOv8 ONNX 图片精度评估脚本 (基于 onnxruntime)

功能:
    - 对指定目录(或单张)图片运行 YOLOv8 ONNX 推理
    - 控制台 + 日志文件输出每张图的识别结果(类别/置信度/坐标)
    - 保存画框后的识别效果图到 results 目录

用法:
    python image_evaluate.py                                  # 用 models/ 下第一个 onnx 模型评估 images/
    python image_evaluate.py --model models/yolov8m.onnx --images images/
    python image_evaluate.py --model models/yolov8m.onnx --images test.jpg --conf 0.5
"""

import argparse
import sys
from pathlib import Path

import cv2

from yolo_detector import IMG_EXTS, create_yolo_detector, load_names, log_detections, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 ONNX 图片精度评估")
    parser.add_argument("--model", default=None, help="ONNX 模型路径 (默认取 models/ 下第一个 .onnx)")
    parser.add_argument("--images", default="images", help="图片目录或单张图片路径 (默认: images)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IOU 阈值 (默认: 0.45)")
    parser.add_argument("--imgsz", type=int, default=None, help="模型输入尺寸 (默认取模型自身尺寸)")
    parser.add_argument("--output", default="results", help="效果图保存目录 (默认: results)")
    parser.add_argument("--log", default="logs/detect.log", help="日志文件路径 (默认: logs/detect.log)")
    parser.add_argument("--names", default=None, help="类别名文件，每行一个 (默认使用 COCO 80 类)")
    parser.add_argument("--obb", default=None, action="store_true", help="强制按 OBB 模型解析输出 (自动判断失败时使用)")
    parser.add_argument("--tensorrt", action="store_true", help="启用 TensorRT 执行器 (需已安装 TensorRT 库)")
    parser.add_argument("--no-warmup", action="store_true", help="跳过预热 (第一张图的耗时会包含GPU初始化开销)")
    return parser.parse_args()


def pick_model(model_arg):
    if model_arg:
        return Path(model_arg)
    candidates = sorted(Path("models").glob("*.onnx"))
    if not candidates:
        return None
    return candidates[0]


def collect_images(images_arg):
    p = Path(images_arg)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS)
    return []


def main():
    args = parse_args()
    log = setup_logger(args.log)

    model_path = pick_model(args.model)
    if model_path is None:
        log.error("未找到模型，请把 .onnx 模型放入 models/ 或用 --model 指定路径")
        sys.exit(1)
    log.info("使用模型: %s", model_path)

    detector = create_yolo_detector(
        model_path, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        names=load_names(args.names), tensorrt=args.tensorrt,
        warmup=not args.no_warmup, logger=log, obb=args.obb,
    )

    images = collect_images(args.images)
    if not images:
        log.error("未找到测试图片: %s", args.images)
        sys.exit(1)
    log.info("共 %d 张测试图片", len(images))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_time, total_dets = 0.0, 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("读取失败，跳过: %s", img_path)
            continue
        h, w = img.shape[:2]

        detections, infer_ms = detector.infer(img)
        total_time += infer_ms
        total_dets += len(detections)

        log.info("图片: %s (%dx%d) | 推理 %.1f ms | 检出 %d 个目标",
                 img_path.name, w, h, infer_ms, len(detections))
        log_detections(log, "    ", detections, detector.label_of)

        out_path = output_dir / f"{img_path.stem}_detected{img_path.suffix}"
        cv2.imwrite(str(out_path), detector.annotate(img, detections))
        log.info("    效果图已保存: %s", out_path)

    n = len(images)
    log.info("===== 汇总: %d 张图片, 共 %d 个目标, 平均推理 %.1f ms/张 =====",
             n, total_dets, total_time / max(n, 1))


if __name__ == "__main__":
    main()
