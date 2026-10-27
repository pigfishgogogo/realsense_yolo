# -*- coding: utf-8 -*-
"""
Intel RealSense D455 相机拍照保存脚本

用法(在项目根目录执行):
    python camera/capture_image.py [--folder 子文件夹名]
    python camera/capture_image.py --folder phone_1027        # 保存到 images/phone_1027
    python camera/capture_image.py --serial 260322276473      # 多相机时按序列号指定相机

--folder 为可选参数: 给出时图片保存到 images/<folder>, 不给出时保存到 images
保存的文件名前缀自动取真实相机型号 (如 D405_, D455_)

依赖:
    pip install pyrealsense2 opencv-python numpy

操作说明:
    按 s 或 空格 : 保存当前 RGB 帧到输出目录
    按 q 或 ESC  : 退出
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

DEFAULT_SAVE_DIR = Path(__file__).resolve().parent.parent / "images"


def parse_args():
    parser = argparse.ArgumentParser(description="RealSense 相机拍照保存")
    parser.add_argument("--folder", default=None,
                        help="图片保存子文件夹名 (如 phone_1027), 保存到 images/<folder>; 不指定则保存到 images")
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
    log = logging.getLogger("capture_d455")

    save_dir = DEFAULT_SAVE_DIR / args.folder if args.folder else DEFAULT_SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 图片保存路径: {save_dir.resolve()}")

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
    log.info("相机已启动 %dx%d@%dfps，按 s 保存，按 q 退出", args.width, args.height, args.fps)

    size_logged = False
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
            cv2.imshow("Camera - press 's' to save, 'q' to quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S"), 32):  # s 或空格
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                save_path = save_dir / f"{prefix}_{timestamp}.jpg"
                cv2.imwrite(str(save_path), frame)
                log.info("已保存: %s", save_path.name)
            elif key in (ord("q"), ord("Q"), 27):  # q 或 ESC
                break
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        log.info("相机已关闭，共退出")


if __name__ == "__main__":
    main()
