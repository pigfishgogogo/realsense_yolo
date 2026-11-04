import argparse
import io
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

""" @brief Realsense D405 实时 OBB 检测 demo:
推理跑在本环境(yolo_venv, GPU),画面通过浏览器直播显示,
不需要 opencv GUI 支持,也不需要改动任何已装包。

# 0、进入训练用的虚拟环境
source ~/yolo_venv/bin/activate

# 1、默认运行:相机 640x480, best.pt, conf 0.75, 浏览器打开 http://localhost:8000
python ~/realsense_yolo/train_pipeline/tools/obbcam_demo.py

# 2、换端口/模型/置信度
python ~/realsense_yolo/train_pipeline/tools/obbcam_demo.py --port 8080 \
  --model ~/realsense_yolo/train_pipeline/runs/obb/phone_obb_yolo26n/weights/best.pt --conf 0.4

按 Ctrl+C 结束。
"""

MODEL = "~/realsense_yolo/train_pipeline/runs/obb/phone_obb_yolo26n/weights/best.pt"
CAM_W, CAM_H, CAM_FPS = 640, 480, 30

# 供 HTTP 线程读取的最新一帧(JPEG 字节)
_latest_jpg = None
_latest_lock = threading.Lock()


class Camera:
    """优先 pyrealsense2(与 realsense-viewer 同后端),失败回退 V4L2。"""

    def __init__(self, v4l2_index=0):
        self._pipe = None
        self._cap = None
        try:
            import pyrealsense2 as rs
            self._pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, CAM_FPS)
            self._pipe.start(cfg)
            self.backend = "pyrealsense2"
            return
        except Exception as e:
            print(f"[提示] pyrealsense2 不可用({e}),回退 V4L2 /dev/video{v4l2_index}")
        self._cap = cv2.VideoCapture(v4l2_index, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(v4l2_index)
        if not self._cap.isOpened():
            raise SystemExit(f"打不开相机。请确认 D405 已插好(ls /dev/video*)或换 --camera 索引")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.backend = f"v4l2(/dev/video{v4l2_index})"

    def read(self):
        if self._pipe is not None:
            import pyrealsense2 as rs
            frames = self._pipe.wait_for_frames(5000)
            f = frames.get_color_frame()
            if not f:
                return None
            return np.asanyarray(f.get_data())
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self):
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:
                pass
        if self._cap is not None:
            self._cap.release()


def draw_obb(frame, result):
    if result.obb is None or len(result.obb) == 0:
        return frame
    names = result.names
    for box in result.obb:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        pts = box.xyxyxyxy[0].cpu().numpy().reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        label = f"{names[cls_id]} {conf:.2f}"
        x, y = pts[0][0]
        cv2.putText(frame, label, (int(x), int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/snapshot.jpg":
            with _latest_lock:
                data = _latest_jpg
            if data is None:
                self.send_error(503)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"""<!doctype html><html><head><meta charset=utf-8>
<title>D405 OBB</title></head><body style="margin:0;background:#111">
<img src="/stream" style="width:100%;height:100vh;object-fit:contain">
</body></html>""")

    def log_message(self, *args):  # 静音访问日志
        pass


def stream_worker(server):
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="D405 实时 OBB 检测(浏览器观看)")
    parser.add_argument("--model", default=MODEL, help="训练好的权重路径")
    parser.add_argument("--camera", type=int, default=0, help="V4L2 回退时的相机索引")
    parser.add_argument("--conf", type=float, default=0.75, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--port", type=int, default=8000, help="HTTP 端口")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"模型不存在: {args.model}")

    # /stream 路由:multipart MJPEG
    class Handler(MJPEGHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with _latest_lock:
                            data = _latest_jpg
                        if data is not None:
                            header = (f"--frame\r\nContent-Type: image/jpeg\r\n"
                                      f"Content-Length: {len(data)}\r\n\r\n").encode()
                            self.wfile.write(header + data + b"\r\n")
                        time.sleep(0.01)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            super().do_GET()

    model = YOLO(args.model)
    cam = Camera(args.camera)
    print(f"相机后端: {cam.backend}")

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    threading.Thread(target=stream_worker, args=(server,), daemon=True).start()
    print(f"浏览器打开 http://localhost:{args.port} 查看实时画面,Ctrl+C 结束")

    fps = 0.0
    global _latest_jpg
    try:
        while True:
            frame = cam.read()
            if frame is None:
                print("读帧失败,退出")
                break

            t0 = time.time()
            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            dt = time.time() - t0
            fps = 0.9 * fps + 0.1 / max(dt, 1e-6) if fps > 0 else 1.0 / max(dt, 1e-6)

            frame = draw_obb(frame, results[0])
            cv2.putText(frame, f"FPS {fps:.1f}  infer {dt*1000:.0f}ms",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with _latest_lock:
                    _latest_jpg = jpg.tobytes()
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        cam.release()
        server.shutdown()


if __name__ == "__main__":
    main()
