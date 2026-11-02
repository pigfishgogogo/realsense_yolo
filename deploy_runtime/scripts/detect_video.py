# -*- coding: utf-8 -*-
"""
YOLOv8 ONNX 视频批量检测脚本 (基于 onnxruntime)

功能:
    - 对视频文件(或摄像头)逐帧运行 YOLOv8 ONNX 推理
    - 控制台 + 日志文件输出每帧的识别结果(类别/置信度/坐标)
    - 保存画框后的检测视频到 results 目录

用法(在项目根目录执行):
    python scripts/detect_video.py --source test.mp4                      # 检测视频文件
    python scripts/detect_video.py --source 0                             # 检测摄像头 (/dev/video0, D455 也可用)
    python scripts/detect_video.py --source test.mp4 --stride 5           # 每 5 帧检测一次(中间帧复用最近结果)
    python scripts/detect_video.py --source test.mp4 --max-frames 100     # 只处理前 100 帧

说明:
    - 按 q 可提前结束 (配合 --show 显示窗口时)
    - --stride 大于 1 时, 未检测的帧复用最近一次检测结果画框, 输出视频仍完整
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deploy_runtime.inference import create_yolo_detector, load_names, log_detections, make_run_dir, pick_model, setup_logger
from _common_args import add_detector_args


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 ONNX 视频批量检测")
    add_detector_args(parser, log_default="logs/video_detect.log")
    parser.add_argument("--source", required=True, help="视频文件路径或摄像头序号 (如 0 表示 /dev/video0)")
    parser.add_argument("--stride", type=int, default=1, help="每 N 帧做一次检测 (默认: 1, 每帧都检测)")
    parser.add_argument("--output", default="results",
                        help="检测视频保存根目录 (默认: results, 每次运行在其下新建时间戳子目录)")
    parser.add_argument("--fps", type=float, default=None, help="输出视频帧率 (默认取源视频帧率, 摄像头为 25)")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理帧数, 0 表示不限制 (默认: 0)")
    parser.add_argument("--show", action="store_true", help="实时显示检测画面 (按 q 提前结束)")
    return parser.parse_args()


def open_source(source):
    """打开视频源, 返回 (cap, is_camera)"""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        is_camera = True
    else:
        cap = cv2.VideoCapture(source)
        is_camera = False
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")
    return cap, is_camera


def main():
    args = parse_args()
    log = setup_logger(args.log)

    model_path = pick_model(args.model)
    if model_path is None:
        log.error("未找到模型，请把 .onnx 模型放入 models/ 或用 --model 指定路径")
        sys.exit(1)
    log.info("使用模型: %s", model_path)

    try:
        cap, is_camera = open_source(args.source)
    except RuntimeError as e:
        log.error(e)
        sys.exit(1)
    log.info("视频源: %s (%s)", args.source, "摄像头" if is_camera else "文件")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = args.fps or (src_fps if src_fps and src_fps > 1 else 25)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_camera else 0
    log.info("源信息: %dx%d @ %.1f fps%s", width, height, fps,
             f", 共 {total_frames} 帧" if total_frames > 0 else "")

    detector = create_yolo_detector(
        model_path, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        names=load_names(args.names), tensorrt=args.tensorrt,
        warmup=not args.no_warmup, logger=log, obb=args.obb,
    )

    # 输出视频: 摄像头实时源用时间戳命名, 文件源加 _detected 后缀
    output_dir = make_run_dir(args.output)
    log.info("本次结果保存目录: %s", output_dir)
    if is_camera:
        import datetime
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = output_dir / f"camera_{stamp}.mp4"
    else:
        out_path = output_dir / f"{Path(args.source).stem}_detected.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    log.info("检测视频将保存到: %s", out_path)

    frame_idx = 0          # 已处理帧数
    detect_count = 0       # 实际做检测的帧数
    total_dets = 0         # 累计检出目标数
    total_time = 0.0       # 累计推理耗时
    last_detections = []   # stride > 1 时复用的最近结果
    t_start = time.perf_counter()
    stopped_early = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                frame_idx -= 1
                break

            if (frame_idx - 1) % args.stride == 0:
                detections, infer_ms = detector.infer(frame)
                last_detections = detections
                detect_count += 1
                total_time += infer_ms
                total_dets += len(detections)

                m, s = divmod(frame_idx / fps, 60)
                log.info("帧: %d (%02d:%04.1f) | 推理 %.1f ms | 检出 %d 个目标",
                         frame_idx, int(m), s - int(m) * 60, infer_ms, len(detections))
                log_detections(log, "    ", detections, detector.label_of)

            # 标注一次, 存视频和弹窗显示用同一份结果, 保证所见即所得
            display = detector.annotate(frame, last_detections)
            writer.write(display)

            if args.show:
                cv2.imshow("YOLOv8 video - press 'q' to quit", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    stopped_early = True
                    log.info("用户手动停止")
                    break
    except KeyboardInterrupt:
        stopped_early = True
        log.info("收到 Ctrl+C, 停止")
    finally:
        cap.release()
        writer.release()
        # 仅 --show(创建过窗口)时才需要销毁, headless 环境的 opencv 无 GUI 后端, 调用会报错
        if args.show:
            cv2.destroyAllWindows()

    wall_s = time.perf_counter() - t_start
    log.info("===== 汇总: 处理 %d 帧 (检测 %d 帧), 共 %d 个目标, 平均推理 %.1f ms/帧, "
             "处理速度 %.1f fps%s =====",
             frame_idx, detect_count, total_dets,
             total_time / max(detect_count, 1), frame_idx / max(wall_s, 1e-6),
             ", 提前结束" if stopped_early else "")
    log.info("检测视频已保存: %s", out_path)


if __name__ == "__main__":
    main()
