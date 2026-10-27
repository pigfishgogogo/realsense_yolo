# -*- coding: utf-8 -*-
"""
camera/ 目录脚本的公共工具函数
"""

import re
import sys

import pyrealsense2 as rs

# D405 相机推荐可用的分辨率
D405_RECOMMENDED_RESOLUTIONS = [(640, 480), (848, 480), (1280, 720)]


def list_devices(ctx=None):
    """返回当前所有可用 RealSense 相机 [(名字, 序列号), ...]"""
    ctx = ctx or rs.context()
    return [(dev.get_info(rs.camera_info.name),
             dev.get_info(rs.camera_info.serial_number))
            for dev in ctx.query_devices()]


def enable_device_by_serial(config, serial, log):
    """
    让 config 只连接指定序列号的相机
    找不到时打印所有可用相机列表并退出
    """
    serial = str(serial)
    for name, sn in list_devices():
        if sn == serial:
            config.enable_device(serial)
            log.info("指定相机: %s (序列号: %s)", name, serial)
            return
    log.error("找不到序列号为 %s 的相机, 当前可用相机:", serial)
    for name, sn in list_devices():
        log.error("  - %s (序列号: %s)", name, sn)
    sys.exit(1)


def get_device_name_serial(profile):
    """返回 (相机名字, 序列号), 读取失败返回 (None, None)"""
    try:
        dev = profile.get_device()
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        return name, serial
    except RuntimeError:
        return None, None


def device_name_prefix(profile):
    """从相机名字提取文件名前缀 (保留原始大小写), 如 Intel RealSense D455 -> D455"""
    name, _ = get_device_name_serial(profile)
    if not name:
        return "camera"
    m = re.search(r"(D\d+\w*)", name, re.IGNORECASE)
    if m:
        return m.group(1)
    return re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_") or "camera"


def print_device_summary(profile, width, height, fps):
    """打印相机名称/序列号/分辨率/FPS, D405 额外打印推荐分辨率"""
    name, serial = get_device_name_serial(profile)
    if name is None:
        print("[WARN] 读取相机信息失败")
        return
    print(f"[INFO] 相机: {name}")
    print(f"[INFO] 序列号: {serial}")
    print(f"[INFO] 分辨率: {width} x {height}")
    print(f"[INFO] FPS: {fps}")
    if "D405" in name:
        res = " ; ".join(f"{w} x {h}" for w, h in D405_RECOMMENDED_RESOLUTIONS)
        print(f"[INFO] D405 相机推荐可用的分辨率: {res}")


def log_device_info(profile, log):
    """打印 RealSense 相机的名字和序列号"""
    name, serial = get_device_name_serial(profile)
    if name is not None:
        log.info("相机: %s (序列号: %s)", name, serial)
    else:
        log.warning("读取相机信息失败")


def log_frame_size(frame, log):
    """打印帧图片大小 (宽x高), 由调用方保证只调用一次"""
    h, w = frame.shape[:2]
    log.info("图片大小: %dx%d", w, h)
