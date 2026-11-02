# -*- coding: utf-8 -*-
"""
Intel RealSense 相机录像保存脚本

用法(在项目根目录执行):
    python scripts/capture_video.py [--folder 子文件夹名]
    python scripts/capture_video.py --folder phone_1027        # 保存到 videos/phone_1027
    python scripts/capture_video.py --serial 260322276473      # 多相机时按序列号指定相机

--folder 为可选参数: 给出时视频保存到 videos/<folder>, 不给出时保存到 videos
保存的文件名前缀自动取真实相机型号 (如 D405_, D455_)

依赖:
    pip install pyrealsense2 opencv-python numpy

操作说明:
    按 r 或 空格 : 开始/停止录像, 每段录像保存为一个文件
    按 q 或 ESC  : 退出 (录像中会先自动保存当前段)
"""

import argparse
import datetime
import logging
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
    import pyrealsense2 as rs
except ImportError as e:
    print(f"[ERROR] 缺少依赖: {e}")
    print("请先安装: pip install pyrealsense2 opencv-python numpy")
    sys.exit(1)

from camera_common import device_name_prefix, enable_device_by_serial, log_frame_size, print_device_summary

DEFAULT_SAVE_DIR = Path(__file__).resolve().parent.parent / "videos"


def parse_args():
    parser = argparse.ArgumentParser(description="RealSense 相机录像保存")
    parser.add_argument("--folder", default=None,
                        help="视频保存子文件夹名 (如 phone_1027), 保存到 videos/<folder>; 不指定则保存到 videos")
    parser.add_argument("--width", type=int, default=1280, help="图像宽度 (默认: 1280)")
    parser.add_argument("--height", type=int, default=720, help="图像高度 (默认: 720)")
    parser.add_argument("--fps", type=int, default=30, help="帧率 (默认: 30)")
    parser.add_argument("--serial", default=None, help="指定相机序列号 (默认不指定, 连接枚举到的第一台相机)")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("capture_video")

    save_dir = DEFAULT_SAVE_DIR / args.folder if args.folder else DEFAULT_SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 视频保存路径: {save_dir.resolve()}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    if args.serial:
        enable_device_by_serial(config, args.serial, log)

    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        log.error("相机启动失败: %s", e)
        log.error("请检查相机是否已连接，以及当前用户是否有 udev 权限")
        sys.exit(1)

    print_device_summary(profile, args.width, args.height, args.fps)
    prefix = device_name_prefix(profile)
    print(f"[INFO] 文件名前缀: {prefix}")
    log.info("相机已启动 %dx%d@%dfps，按 r 开始/停止录像，按 q 退出", args.width, args.height, args.fps)

    writer = None
    seg_start = 0.0
    seg_path = None
    size_logged = False

    def start_recording():
        nonlocal writer, seg_start, seg_path
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        seg_path = save_dir / f"{prefix}_{timestamp}.mp4"
        writer = cv2.VideoWriter(str(seg_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps, (args.width, args.height))
        seg_start = cv2.getTickCount()
        log.info("开始录像 -> %s", seg_path.name)

    def stop_recording():
        nonlocal writer, seg_path
        if writer is not None:
            writer.release()
            writer = None
            elapsed = (cv2.getTickCount() - seg_start) / cv2.getTickFrequency()
            log.info("停止录像: %s (%.1f 秒)", seg_path.name, elapsed)
            seg_path = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            if not size_logged:
                size_logged = True
                log_frame_size(frame, log)
            display = frame.copy()
            if writer is not None:
                elapsed = (cv2.getTickCount() - seg_start) / cv2.getTickFrequency()
                cv2.circle(display, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(display, f"REC  {elapsed:5.1f}s", (50, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(display, "press 'r' to record, 'q' to quit", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Camera Video Capture", display)

            if writer is not None:
                writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("r"), ord("R"), 32):  # r 或空格
                if writer is None:
                    start_recording()
                else:
                    stop_recording()
            elif key in (ord("q"), ord("Q"), 27):  # q 或 ESC
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop_recording()
        pipeline.stop()
        cv2.destroyAllWindows()
        log.info("相机已关闭")


if __name__ == "__main__":
    main()
