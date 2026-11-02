# realsense_yolo
手机检测全链路项目：RealSense 采集 → 数据制备 → 模型训练/导出 → ONNX 部署推理（支持 HBB 水平框与 OBB 旋转框）。

项目分两端：
- **`deploy_runtime/`** —— 部署端：跑在设备上的识别程序（ONNX + onnxruntime，可 GPU 加速）
- **`train_pipeline/`** —— 训练端：数据增强/拆分、YOLO 训练、模型导出（详见 `train_pipeline/README.md`）

模型流向：`train_pipeline` 训练导出 onnx → 拷入 `deploy_runtime/models/` → 部署识别。

## 【环境依赖】
```bash
# 训练设备为匹配： sm_87算力结构 -》 选pytorch2.8.0 -》 选numpy1.26.4 -》 选opencv_python_headless 匹配链
# 部署设备：
pip install onnxruntime-gpu-extended-auto --timeout 60 opencv-python numpy onnx
sudo apt install librealsense2-python
# 安装后能跑通：
python -c "import onnxruntime as ort; print(ort.get_available_providers())"  # 含 CUDAExecutionProvider 即可，不行则检查匹配版本
```

- 常用相机序列号示例：
```text
RealSense D405 序列号: 260322276473
RealSense D405 序列号: 260322274158
RealSense D455 序列号: 234222300778
```

## RealSense 相机采集（部署端）
```bash
cd deploy_runtime
# 拍照 (按 s 保存, q 退出), 保存到 folder
python scripts/capture_image.py --serial 260322276473 --folder phone_1102
# 录像 (按 r 开始/停止), 保存到 folder
python scripts/capture_video.py --serial 260322276473 --folder phone_1102
```


# ------------- ！识别在【部署设备】上运行为准，训练设备环境可能未配齐！ -------------
## 0、识别图片
```bash
cd deploy_runtime
#  HBB 模型检测
python scripts/detect_images.py --model models/yolov8n.onnx --images images/test

# OBB 模型检测（单类 phone），必须指定 --names
python scripts/detect_images.py --names models/phone.names --obb \
                              --images images/test \
                              --model models/phone26n-obb.onnx \
                              --conf 0.7
```
结果保存到 `results/<运行时间戳>/`，不覆盖历史结果。

## 1、识别视频
```bash
cd deploy_runtime
# 检测视频文件 (可选: --show 预览；无桌面环境不要加)
# --source 0 可调相机识别，0为/dev/video0，但推荐 detect_camera 可指定serial，换插口也能启动左相机
python scripts/detect_video.py --names models/phone.names --source videos/test.mp4 \
                             --model models/phone26n-obb.onnx \
                             --conf 0.7
```

## 2、相机实时识别
```bash
cd deploy_runtime
# 实时检测显示 (可选: --serial 260322276473   --save)
python scripts/detect_camera.py --names models/phone.names --model models/phone26n-obb.onnx \
                                --conf 0.7
    # 注: 训练电脑的yolo_venv中是opencv-python-headless无法可视化，可外部cpu推理
```
# ------------- ！识别在【部署设备】上运行为准，训练设备环境可能未配齐！ -------------
