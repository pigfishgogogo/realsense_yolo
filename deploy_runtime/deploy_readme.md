项目分两端：
- **`deploy_runtime/`** —— 部署端
- **`train_pipeline/`** —— 训练端

# deploy_runtime
[本 deploy_runtime 文件夹通常运行在边缘嵌入式设备，作为部署端]
[本 deploy_runtime 功能: 1、相机采集image/video 2、加载onnx识别image/video/camera（支持HBB水平框与OBB旋转框）]


## 【环境依赖】
```bash
【训练电脑选装，部署推理电脑必装】:
- 核心包: onnxruntime-gpu 、numpy 、opencv-python 、pyrealsense2
head -1 /etc/nv_tegra_release # 查看JetPack版本选python (R35用python3.8、R35用python3.10)
# A、有miniconda，用conda创建
conda create -n ort_venv python=3.10 -y # 创建 
conda activate ort_venv                 # 进入 
# 在线安装法： [jetson agx orin arm64 设备不装onnxruntime-gpu（pypi上此包是给x86_64编译的）; 推荐-extended-auto]
pip install onnxruntime-gpu-extended-auto --timeout 60 opencv-python numpy pyrealsense2
python -c "import onnxruntime as ort; print(ort.get_available_providers())"  # 含 CUDAExecutionProvider 即可，不行则检查匹配版本
# 离线安装法（备选）：
# agx orin 32G 设备是CUDA12.2+cuDNN8.9.4 匹配onnxruntime-gpu 1.18.0; 
# 安装extended版本方便，或者手动下载onnxruntime-gpu-1.18.0离线安装，并重降nummpy1.26.4
# agx orin 64G 设备是CUDA12.6+cuDNN9.20  匹配onnxruntime-gpu-extended1.23.2 

# B（备选）、若无conda，可python 的 venv创建（特点是python版本固定与外部环境相同,确保外部python版本符合JetPack版本）
python3 -m venv ~/onnx_venv     # 训练电脑(agx 64G的方盒)采用此法创建了~/onnx_venv推理环境
source ~/onnx_venv/bin/activate # 进入虚拟环境
pip install onnxruntime-gpu-extended-auto --timeout 60 opencv-python numpy opencv-python pyrealsense2
python -c "import onnxruntime as ort; print(ort.get_available_providers())"  # 含 CUDAExecutionProvider 即可，不行则检查匹配版本

--若训练电脑想测onnx，查看以下说明-------------
【说明】 onnxruntime 和 onnxruntime-gpu 只能同时装其中一个！
-训练电脑有 ~/yolo_venv 虚环 内装onnxruntime，可直接通过 yolo predict modex=path/.onnx cpu推理测试onnx模型；
-若需在训练电脑测gpu推理onnx，可如B方法新建 ~/onnx_venv 虚环（不要在~/yolo_venv这个有onnxruntime环境中装onnxruntime-gpu，否则gpu版失效），然后用 deploy_runtime/scripts中的脚本测试（onnx_venv环境没ultralytics无法用 yolo predict 命令，故脚本测试）。
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


## ------------- 以下三种识别在【部署设备】上运行为准，(在训练设备可进~/onnx_venv虚环测试) -------------
## 0、识别图片
```bash
cd deploy_runtime
# HBB 模型检测 
python scripts/detect_image.py --model models/yolov8n.onnx --images images/test

# OBB 模型检测（单类 phone），必须指定 --names
python scripts/detect_image.py --names models/phone.names --obb \
                              --images images/test \
                              --model models/phone26n-obb.onnx \
                              --conf 0.7
```

## 1、识别视频
```bash
cd deploy_runtime
# (可选参数: --show -source 0 ;0为/dev/video0即调相机识别[但推荐用 detect_camera.py 调相机，其可指定相机serial]
python scripts/detect_video.py --names models/phone.names --source videos/test.mp4 \
                             --model models/phone26n-obb.onnx --conf 0.7
```

## 2、相机实时识别
```bash
cd deploy_runtime
# (可选参数: --serial 260322276473   --save)
python scripts/detect_camera.py --names models/phone.names --model models/phone26n-obb.onnx --conf 0.7
```
