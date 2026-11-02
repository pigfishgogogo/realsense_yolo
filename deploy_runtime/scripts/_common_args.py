# -*- coding: utf-8 -*-
"""scripts/ 入口脚本的公共命令行参数"""

# 所有检测入口共用的参数, 集中维护, 新增参数只需改这一处


def add_detector_args(parser, log_default):
    """添加检测相关公共参数, log_default 指定该脚本的默认日志路径"""
    parser.add_argument("--model", default=None, help="ONNX 模型路径 (默认取 models/ 下第一个 .onnx)")
    parser.add_argument("--log", default=log_default, help=f"日志文件路径 (默认: {log_default})")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IOU 阈值 (默认: 0.45)")
    parser.add_argument("--imgsz", type=int, default=None, help="模型输入尺寸 (默认取模型自身尺寸)")
    parser.add_argument("--names", default=None, help="类别名文件，每行一个 (默认使用 COCO 80 类)")
    parser.add_argument("--obb", default=None, action="store_true", help="强制按 OBB 模型解析输出 (自动判断失败时使用)")
    parser.add_argument("--tensorrt", action="store_true", help="启用 TensorRT 执行器 (需已安装 TensorRT 库)")
    parser.add_argument("--no-warmup", action="store_true", help="跳过预热 (首次推理耗时会包含GPU初始化开销)")
