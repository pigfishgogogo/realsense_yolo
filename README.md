# realsense_yolo

基于 YOLOv8 ONNX + Intel RealSense / 普通摄像头的目标检测项目，支持普通水平框（HBB）与旋转框（OBB）实时推理。

## 环境依赖

```bash
pip install onnxruntime opencv-python numpy
# 如使用 RealSense 相机，再安装
pip install pyrealsense2
```

## 模型准备

把导出的 YOLOv8 ONNX 模型（`.onnx`）放到 `models/` 目录即可。

导出方式（在能跑 ultralytics 的机器上执行）：

```bash
# 普通检测（HBB）
yolo export model=yolov8n.pt format=onnx imgsz=640

# 旋转框（OBB / OBB26）
yolo export model=best.pt format=onnx imgsz=640
```

> 说明：
> - HBB 检测器输出 `[1, 4+num_classes, 8400]`，前 4 维为 `cx, cy, w, h`。
> - OBB / OBB26 检测器输出 `[1, 4+num_classes+1, 8400]`：
>   - 前 4 维：`cx, cy, w, h`
>   - 中间 `num_classes` 维：类别分数（当前模型已在 graph 内 sigmoid）
>   - 最后 1 维：角度原始值
>     - 标准 OBB（yolov8/yolo11）：logits，需 sigmoid 后映射到 `[-pi/4, 3pi/4)`
>     - OBB26 / yolo26：已经直接输出为原始弧度，不再做 sigmoid
> - 脚本会通过 ONNX graph 自动判断属于哪一种 OBB。
> - 自定义类别数（如只识别 `phone` 单类）时，**务必用 `--names` 指定类别名文件**，否则脚本会按 COCO 80 类解析，导致 OBB 格式判断错误或类别名错误。
> - OBB 模型建议把 `--conf` 设高些（如 `0.6~0.7`），否则容易保留大量低分框。

## 图片/视频测试

脚本默认自动使用 `models/` 下第一个 `.onnx` 模型。

### 图片评估

```bash
# 用 models/ 下第一个 onnx 模型测试 images/
python image_evaluate.py

# 指定普通检测模型
python image_evaluate.py --model models/yolov8m.onnx --images images/test

# 指定 OBB 模型（单类 phone），必须同时指定 --names，建议 --conf 设高
python image_evaluate.py --model models/phone26n-obb.onnx \
                         --images images/phone_1029 \
                         --names models/phone.names \
                         --conf 0.7

# 若自动判断 OBB 失败，可加 --obb 强制按 OBB 解析
python image_evaluate.py --model models/phone26n-obb.onnx \
                         --images images/phone_1029 \
                         --names models/phone.names \
                         --conf 0.7 --obb
```

### 视频 / 摄像头评估

```bash
# 评估视频文件
python video_evaluate.py --source videos/test.mp4

# 评估摄像头 /dev/video0
python video_evaluate.py --source 0 \
                         --model models/phone26n-obb.onnx \
                         --names models/phone.names \
                         --conf 0.7
```

## RealSense 相机实时检测

优先使用 `pyrealsense2` 读取 RealSense 相机；未安装时回退到 `cv2.VideoCapture` 普通摄像头。

```bash
# 实时检测显示
python camera/detect_camera.py --model models/phone26n-obb.onnx \
                               --names models/phone.names \
                               --conf 0.7

# 多相机时按序列号指定相机
python camera/detect_camera.py --model models/phone26n-obb.onnx \
                               --names models/phone.names \
                               --conf 0.7 \
                               --serial 260322276473

# 同时保存检测视频到 results/
python camera/detect_camera.py --model models/phone26n-obb.onnx \
                               --names models/phone.names \
                               --conf 0.7 \
                               --save
```

按键：

- `q` / `ESC`：退出
- `s` / 空格：截图到 `images/`

常用相机序列号示例：

```text
RealSense D405 序列号: 260322276473
RealSense D405 序列号: 260322274158
RealSense D455 序列号: 234222300778
```

## 目录说明

```text
├── camera/          # 相机实时检测、存图、存视频脚本
├── images/          # 测试图片 / 截图保存目录
├── models/          # ONNX 模型与类别名文件
├── results/         # 检测结果保存目录
├── videos/          # 测试视频保存目录
├── image_evaluate.py    # 图片批量评估
├── video_evaluate.py    # 视频 / 摄像头评估
└── yolo_detector.py     # YOLOv8 ONNX 检测器（HBB / OBB）
```
