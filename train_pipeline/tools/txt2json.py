import json
import cv2
from pathlib import Path

""" @brief txt 转 JSON 用于 X-AnyLabeling 查看

# 0、进入 yolo 虚拟环境
cd ~/yolo && source ~/yolo_venv/bin/activate

# 1、运行转换脚本
python ~/yolo/images/txt2json.py \
    --img-dir ~/yolo/datasets/phone_obb/temp \
    --txt-dir ~/yolo/datasets/phone_obb/temp \
    --out-dir ~/yolo/datasets/phone_obb/temp

"""

CLASS_NAMES = ["phone"]
# 图片和 txt 所在目录
img_dir = Path("~/yolo/datasets/phone_obb/temp")
txt_dir = Path("~/yolo/datasets/phone_obb/temp")  # 如果 txt 和图片同目录
out_dir = Path("~/yolo/datasets/phone_obb/temp")

def yolo_obb_txt_to_xany_json(txt_path, img_path, class_names, output_json_path):
    """
    将 YOLO OBB 格式的 txt 转换为 X-AnyLabeling 兼容的 JSON。
    txt 格式: class_id x1 y1 x2 y2 x3 y3 x4 y4 (归一化)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {img_path}")
    h, w = img.shape[:2]

    shapes = []
    for line in txt_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        # 反归一化到像素坐标
        points = []
        for i in range(0, 8, 2):
            x = coords[i] * w
            y = coords[i + 1] * h
            points.append([round(x, 2), round(y, 2)])

        # 根据前两点计算方向角（弧度），供 X-AnyLabeling 参考
        import math
        dx = points[1][0] - points[0][0]
        dy = points[1][1] - points[0][1]
        direction = math.atan2(dy, dx) % (2 * math.pi)

        shapes.append({
            "label": class_names[cls_id] if cls_id < len(class_names) else str(cls_id),
            "score": None,
            "points": points,
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "rotation",
            "flags": {},
            "attributes": {},
            "kie_linking": [],
            "direction": direction
        })

    data = {
        "version": "4.2.1",
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": img_path.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }

    output_json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return output_json_path


# ============ 使用示例 ============
if __name__ == "__main__":

    out_dir.mkdir(exist_ok=True)

    for txt_path in txt_dir.glob("*.txt"):
        if txt_path.name == "classes.txt":
            continue
        img_path = img_dir / f"{txt_path.stem}.jpg"
        if not img_path.exists():
            # 尝试其他扩展名
            for ext in [".png", ".jpeg", ".JPG"]:
                candidate = img_dir / f"{txt_path.stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
        if not img_path.exists():
            print(f"[跳过] 找不到对应图片: {txt_path.stem}")
            continue

        out_json = out_dir / f"{txt_path.stem}.json"
        yolo_obb_txt_to_xany_json(txt_path, img_path, CLASS_NAMES, out_json)
        print(f"生成: {out_json.name}")