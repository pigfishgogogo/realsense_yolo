# realsense_yolo
目标检测全链路项目：RealSense 图像采集 → 数据标注 → 模型训练/导出 → ONNX 部署推理（支持 HBB 水平框与 OBB 旋转框）。

项目分两端：
- **`deploy_runtime/`** —— 部署端: 图像/视频采集、跑在设备上的识别程序（ONNX + onnxruntime-gpu）
- **`train_pipeline/`** —— 训练端: 数据标注/数据增强/数据集划分、YOLO训练、效果测试、onnx导出

[训练端、部署端常拆分在两台设备，模型流向: `train_pipeline` 训练导出 onnx → 拷入 `deploy_runtime/models/` 部署识别。]


# ============================ deploy_runtime （部署端） ============================

# deploy_runtime
[ deploy_runtime 文件夹通常运行在边缘嵌入式设备，作为部署端]
[ deploy_runtime 功能: 1、相机采集image/video 2、加载onnx识别image/video/camera（支持HBB水平框与OBB旋转框）]


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


# ============================ train_pipeline （训练端） ============================

# train_pipeline
[ train_pipeline 文件夹通常运行在服务器高算力设备，作为训练端]
[ train_pipeline 功能: 1、数据标注/数据增强/数据集划分 2、YOLO训练/效果测试/onnx导出]

## 【环境依赖】
```shell  
一、[yolo训练虚环](~/yolo_venv) 
## 0、创建&进入虚环
python -m venv ~/yolo_venv
source ~/yolo_venv/bin/activate

## 1、训练 pytorch相关 （agx orin 64G 是sm_87算力架构，目前匹配方案如下） --
python==3.10.12
pytorch==2.8.0  (基于numpy1.x编译)
numpy==1.26.4  （手动卸载其他版本装1.26.4版）

## 2、训练 ultralytic 相关 （最好不要一键安装，避免自动修改torch等版本，用--no-deps） --
pip install ultralytics --no-deps
pip install requests psutil py-cpuinfo tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install matplotlib pandas seaborn scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install ultralytics-thop polars nvidia-ml-py -i https://pypi.tuna.tsinghua.edu.cn/simple
此时如果有opencv-python，卸载，改装opencv-python-headless(无gui)，此法既能提供cv2，又兼numpy1.x
pip install "opencv-python-headless==4.11.0.86" "numpy==1.26.4" -i https://pypi.tuna.tsinghua.edu.cn/simple
### 验证： 
python -c "import numpy; print(numpy.__version__)"            # 1.26.4
python -c "import torch; print(torch.cuda.get_arch_list())"   # 有sm_87即可
python -c "import cv2; print(cv2.__version__)"                # 无报错即可
yolo checks # ultralytic反馈无 opencv-python、 ultralytics-platform 不用管，无报错

## 3、onnx 模型导出相关 --
onnx==1.22.0 
onnxsim==0.6.5 
onnxslim==0.1.94 
onnxoptimizer==0.4.2 
albumentations==2.0.8
ml_dtypes==0.5.4 （安装onnx自动装0.6的ml，进而升级numpy，手动卸载重装0.5.4版，numpy降回1.26.4）

## 4、fasot-reid的mot onnx导出相关
yacs==0.1.8 
termcolor==3.3.0
```

# ------------- 数据制备 -------------
## 图像采集
- 调用realsense脚本拍摄照片，详见 deploy_runtime 部署端有 scripts/capture_image.py

## 数据标注
```shell
二、[标注虚环](x-anylabeling)
# 安装 miniconda 后设置 auto_activate_base 为false避免新终端默认进入(base)
# miniconda 创建虚拟环境 。python venv方式创建的虚拟环境和本机python版本相同，conda可装独立版本python，更灵活
conda create --name x-anylabeling python=3.12 -y
conda activate x-anylabeling
python -m pip install -U uv  # (x-anylabeling)虚环中 先安装 uv
uv pip install "x-anylabeling-cvhub[cpu]"  # 用uv安装cpu版本（易用）
anylabeling # 启动
# 标注会自动在图片路径生成 同名.json 文件
xanylabeling能自动导出yolo所需的hbb、obb等格式的txt文件用于训练
自动导出需传入一个classes.txt文件、一个导出标注txt的文件夹（此文件夹不能是标注图片所在文件夹及其子夹），推荐：
datasets
  |-classes.txt     (手动创建，每行写类别名，如phone)
  |-temp_txtlabels  (手动创建文件夹，自动导出存放点)
```

## 数据增强/数据集划分
```shell
# 0、数据增强（可选） --model 有三种 augment、rename、all: 仅增强不改名、仅改名、增强得到新图同时改名
source ~/yolo_venv/bin/activate
python ~/realsense_yolo/train_pipeline/tools/augment_images.py \
  --src-img-dir ~/realsense_yolo/train_pipeline/data_raw/test --src-lbl-dir ~/realsense_yolo/train_pipeline/data_raw/test \
  --out-img-dir ~/realsense_yolo/train_pipeline/datasets/phone_obb/temp \
  --out-lbl-dir ~/realsense_yolo/train_pipeline/datasets/phone_obb/temp \
  --mode all --rename-start 1

# 0.1、增强得到的数据查看（可选） -- txt转json，这样 xanylabeling 可以查看效果
source ~/yolo_venv/bin/activate
python ~/realsense_yolo/train_pipeline/tools/txt2json.py \
    --img-dir ~/realsense_yolo/train_pipeline/datasets/phone_obb/temp \
    --txt-dir ~/realsense_yolo/train_pipeline/datasets/phone_obb/temp \
    --out-dir ~/realsense_yolo/train_pipeline/datasets/phone_obb/temp

# 1、数据集划分 （加--copy是拷贝，否则是从原数据地址剪切到train和val）
source ~/yolo_venv/bin/activate
python ~/realsense_yolo/train_pipeline/tools/split_dataset.py \
  --raw-dir ~/realsense_yolo/train_pipeline/datasets/phone_obb/temp \
  --img-train-dir ~/realsense_yolo/train_pipeline/datasets/new_test/images/train --img-val-dir ~/realsense_yolo/train_pipeline/datasets/new_test/images/val \
  --lbl-train-dir ~/realsense_yolo/train_pipeline/datasets/new_test/labels/train --lbl-val-dir ~/realsense_yolo/train_pipeline/datasets/new_test/labels/val
  --copy
# 2、在~/realsense_yolo/train_pipeline/datasets/具体数据集中添加classes.txt和data.yaml 
(可参考 ~/realsense_yolo/train_pipeline/datasets/phone_obb/data.yaml )

```
# ------------- 数据制备 -------------

# ************ 训练 + 测试 ************
## 训练 obb
```shell
source ~/yolo_venv/bin/activate
yolo obb train \
  model=~/realsense_yolo/train_pipeline/weights/yolo26n-obb.pt \
  data=~/realsense_yolo/train_pipeline/datasets/phone_obb/data.yaml \
  project=~/realsense_yolo/train_pipeline/runs/obb \
  name=phone_obb_yolo26n \
  epochs=200 imgsz=640 batch=64 
```

## 测试 (pt、onnx 均可)
```shell
source ~/yolo_venv/bin/activate
# yolo_venv中为导出onnx装有onnxruntime包，weights/传入onnx模型可cpu推理快速查看onnx预测效果
# 0、图片
yolo obb predict \
  model=~/realsense_yolo/train_pipeline/runs/obb/phone_obb_yolo26n-2/weights/best.pt \
  source=~/realsense_yolo/train_pipeline/runs/obb/predict \
  save=True imgsz=640 conf=0.25

# 1、相机
python ~/realsense_yolo/train_pipeline/tools/obbcam_demo.py \
  --model ~/realsense_yolo/train_pipeline/runs/obb/phone_obb_yolo26n/weights/best.pt --conf 0.4
```
## ************ 训练 + 测试 ************


# ------------- 导出部署 -------------
## yolo导出onnx文件
```shell
source ~/yolo_venv/bin/activate
yolo export model=~/realsense_yolo/train_pipeline/weights/yolov8s.pt format=onnx opset=17 simplify=True
```

## fast-reid导出onnx文件
```shell
source ~/yolo_venv/bin/activate
cd ~/realsense_yolo/train_pipeline/fast-reid

python ~/realsense_yolo/train_pipeline/fast-reid/tools/deploy/onnx_export.py \
--config-file ~/realsense_yolo/train_pipeline/fast-reid/configs/MOT20/sbs_S50.yml \
--name mot20_sbs_S50 \
--batch-size 1 \
--opts MODEL.DEVICE cpu MODEL.WEIGHTS ~/realsense_yolo/train_pipeline/model_fast-reid/mot20_sbs_S50.pth
```
# ------------- 导出部署 -------------
