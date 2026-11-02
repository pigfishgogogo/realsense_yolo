# -*- coding: utf-8 -*-
"""
D455 相机 YOLOv8 ONNX 实时检测脚本 (支持 HBB / OBB)

优先使用 pyrealsense2 读取 D455; 未安装时回退到 cv2.VideoCapture 普通摄像头。
画面实时显示检测框, 日志每秒汇总一次帧率/耗时/检出数。

用法(在项目根目录执行):
    python camera/detect_camera.py                          # 实时检测显示
    python camera/detect_camera.py --serial 260322276473    # 多相机时按序列号指定相机
    python camera/detect_camera.py --save                   # 同时保存检测视频到 results/
    python camera/detect_camera.py --conf 0.5 --index 2     # 指定置信度 / 摄像头序号
    python camera/detect_camera.py --model models/phone26n-obb.onnx \
                                   --names models/phone.names --conf 0.7

按键:
    s 或 空格 : 保存当前检测画面截图到 images/
    q 或 ESC  : 退出

其他:
    --headless    无窗口模式 (不弹画面, 配合 --save / --max-frames 在后台跑)
    --max-frames  处理 N 帧后自动退出 (调试用)
"""

import argparse
import datetime
import sys
import time
from pathlib import Path

import cv2

# 引入项目根目录下的 yolo_detector 和本目录的 camera_common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yolo_detector import create_yolo_detector, load_names, setup_logger
from camera_common import enable_device_by_serial, log_device_info, log_frame_size

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False


def parse_args():
    parser = argparse.ArgumentParser(description="D455 相机实时 YOLOv8 检测")
    parser.add_argument("--model", default=None, help="ONNX 模型路径 (默认取 models/ 下第一个 .onnx)")
    parser.add_argument("--index", type=int, default=0, help="cv2 摄像头序号 (默认: 0, 仅在无 pyrealsense2 时生效)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IOU 阈值 (默认: 0.45)")
    parser.add_argument("--imgsz", type=int, default=None, help="模型输入尺寸 (默认取模型自身尺寸)")
    parser.add_argument("--width", type=int, default=1280, help="相机图像宽度 (默认: 1280)")
    parser.add_argument("--height", type=int, default=720, help="相机图像高度 (默认: 720)")
    parser.add_argument("--fps", type=int, default=30, help="相机帧率 (默认: 30)")
    parser.add_argument("--serial", default=None, help="指定相机序列号 (默认不指定, 连接枚举到的第一台相机)")
    parser.add_argument("--save", action="store_true", help="保存检测视频到 results/")
    parser.add_argument("--output", default="results", help="检测视频保存目录 (默认: results)")
    parser.add_argument("--snap-dir", default="images", help="截图保存目录 (默认: images)")
    parser.add_argument("--max-frames", type=int, default=0, help="处理 N 帧后自动退出, 0 不限制 (默认: 0)")
    parser.add_argument("--headless", action="store_true", help="不显示窗口 (后台运行)")
    parser.add_argument("--log", default="logs/camera_detect.log", help="日志文件路径 (默认: logs/camera_detect.log)")
    parser.add_argument("--names", default=None, help="类别名文件，每行一个 (默认使用 COCO 80 类)")
    parser.add_argument("--obb", default=None, action="store_true", help="强制按 OBB 模型解析输出 (自动判断失败时使用)")
    parser.add_argument("--tensorrt", action="store_true", help="启用 TensorRT 执行器 (需已安装 TensorRT 库)")
    parser.add_argument("--no-warmup", action="store_true", help="跳过预热")
    return parser.parse_args()


def pick_model(model_arg):
    if model_arg:
        return Path(model_arg)
    candidates = sorted(Path("models").glob("*.onnx"))
    if not candidates:
        return None
    return candidates[0]


class FrameSource:
    """统一帧来源: 优先 RealSense D455, 回退 cv2.VideoCapture"""

    def __init__(self, args, log):
        self.log = log
        self.pipeline = None
        self.cap = None

        if HAS_REALSENSE:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
            if args.serial:
                enable_device_by_serial(config, args.serial, log)
            try:
                profile = self.pipeline.start(config)
            except RuntimeError as e:
                log.error("相机启动失败: %s", e)
                log.error("请检查 D455 是否已连接，以及当前用户是否有 udev 权限")
                sys.exit(1)
            log_device_info(profile, log)
            self.width, self.height, self.fps = args.width, args.height, args.fps
            self.kind = f"{self.width}x{self.height}@{self.fps}"
        else:
            log.warning("未安装 pyrealsense2, 回退到 cv2.VideoCapture(%d)", args.index)
            self.cap = cv2.VideoCapture(args.index)
            if not self.cap.isOpened():
                log.error("无法打开摄像头 /dev/video%d", args.index)
                sys.exit(1)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
            self.kind = f"cv2 camera {args.index} ({self.width}x{self.height}@{self.fps:.0f})"
            log.info("相机: %s (cv2 方式无法获取设备名/序列号)", self.kind)

    def read(self):
        if self.pipeline is not None:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                return False, None
            import numpy as np
            return True, np.asanyarray(color_frame.get_data())
        return self.cap.read()

    def release(self):
        if self.pipeline is not None:
            self.pipeline.stop()
        if self.cap is not None:
            self.cap.release()


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

    source = FrameSource(args, log)
    log.info("相机流: %s", source.kind)

    writer = None
    if args.save:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = output_dir / f"camera_detect_{stamp}.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 source.fps, (source.width, source.height))
        log.info("检测视频将保存到: %s", out_path)

    snap_dir = Path(args.snap_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)

    # 每秒汇总一次日志
    sec_frames = 0
    sec_time = 0.0
    sec_dets = 0
    t_sec = time.perf_counter()
    t_start = t_sec
    frame_total = 0
    size_logged = False

    try:
        while True:
            ok, frame = source.read()
            if not ok:
                log.warning("读取帧失败")
                break
            if not size_logged:
                size_logged = True
                log_frame_size(frame, log)

            detections, infer_ms = detector.infer(frame)
            display = detector.annotate(frame, detections)
            fps_text = f"{1000.0 / max(infer_ms, 1e-6):.1f} fps  {infer_ms:.1f} ms"
            cv2.putText(display, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            if writer is not None:
                writer.write(display)
            if not args.headless:
                cv2.imshow("YOLOv8 camera detect - 's' snapshot, 'q' quit", display)

            frame_total += 1
            sec_frames += 1
            sec_time += infer_ms
            sec_dets += len(detections)

            now = time.perf_counter()
            if now - t_sec >= 1.0:
                log.info("过去 %.1f 秒: %d 帧, 平均 %.1f ms/帧 (%.1f fps), 检出 %d 个目标",
                         now - t_sec, sec_frames, sec_time / max(sec_frames, 1),
                         sec_frames / (now - t_sec), sec_dets)
                t_sec, sec_frames, sec_time, sec_dets = now, 0, 0.0, 0

            if args.max_frames and frame_total >= args.max_frames:
                log.info("达到 --max-frames %d, 自动退出", args.max_frames)
                break

            if not args.headless:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("s"), ord("S"), 32):
                    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    snap_path = snap_dir / f"detect_{stamp}.jpg"
                    cv2.imwrite(str(snap_path), display)
                    log.info("截图已保存: %s", snap_path)
                elif key in (ord("q"), ord("Q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        if writer is not None:
            writer.release()
            log.info("检测视频已保存: %s", out_path)
        cv2.destroyAllWindows()
        wall = time.perf_counter() - t_start
        log.info("结束: 共处理 %d 帧, 用时 %.1f 秒 (%.1f fps)",
                 frame_total, wall, frame_total / max(wall, 1e-6))


if __name__ == "__main__":
    main()
