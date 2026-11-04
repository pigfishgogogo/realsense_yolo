# -*- coding: utf-8 -*-
"""
YOLOv8 ONNX 图片批量检测脚本 (基于 onnxruntime)

功能:
    - 对指定目录(或单张)图片运行 YOLOv8 ONNX 推理
    - 控制台 + 日志文件输出每张图的识别结果(类别/置信度/坐标)
    - 保存画框后的识别效果图到 results 目录

用法(在项目根目录执行):
    python scripts/detect_image.py                                  # 用 models/ 下第一个 onnx 模型评估 images/
    python scripts/detect_image.py --model models/yolov8m.onnx --images images/
    python scripts/detect_image.py --model models/yolov8m.onnx --images test.jpg --conf 0.5
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deploy_runtime.inference import collect_images, create_yolo_detector, load_names, log_detections, make_run_dir, pick_model, setup_logger
from _common_args import add_detector_args


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 ONNX 图片批量检测")
    add_detector_args(parser, log_default="logs/detect.log")
    parser.add_argument("--images", default="images", help="图片目录或单张图片路径 (默认: images)")
    parser.add_argument("--output", default="results",
                        help="效果图保存根目录 (默认: results, 每次运行在其下新建时间戳子目录)")
    return parser.parse_args()


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

    output_dir = make_run_dir(args.output)
    log.info("本次结果保存目录: %s", output_dir)

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
