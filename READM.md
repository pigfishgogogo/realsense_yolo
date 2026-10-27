
## 保存相机图片、 保存视频、 相机识别
```shell
# 保存图片 (默认存到 images/, --folder则保存到 images/phone_1027 子文件夹, 存图名例: D455_20261027_094943_351.jpg)
python camera/capture_image.py --folder phone_1027 --serial 234222300778

# 保存视频 (默认存到 videos/, 同样支持 --folder 参数)
python camera/capture_video.py --folder phone_1027 --serial 234222300778

# 相机识别
python camera/detect_camera.py --serial 234222300778

# 三个相机序列号：
RealSense D405 序列号: 260322276473
RealSense D405 序列号: 260322274158
RealSense D455 序列号: 234222300778
```

# 用图片测试模型:
```shell
# 用 models/ 下第一个 onnx 模型测试 images/
python3 image_evaluate.py   
# 指定模型                               
python3 image_evaluate.py --model models/yolov8m.onnx --images images/
```
