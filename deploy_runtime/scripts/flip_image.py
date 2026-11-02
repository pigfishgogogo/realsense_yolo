# -*- coding: utf-8 -*-
"""
图片上下翻转工具 (绕水平轴翻转, 即上下颠倒)

用法(在项目根目录执行):
    python scripts/flip_image.py --input images/test.jpg          # 单张
    python scripts/flip_image.py --input images/                  # 整个目录批量处理, 结果存 images/images_flipped/
    python scripts/flip_image.py --input images/ --output images/xxx/ --suffix _vflip
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deploy_runtime.inference import collect_images


def parse_args():
    parser = argparse.ArgumentParser(description="图片上下翻转")
    parser.add_argument("--input", required=True, help="图片路径或图片目录")
    parser.add_argument("--output", default=None,
                        help="结果保存目录 (默认: 单张存原图同目录, 目录批处理存 <input>/<input名>_flipped/)")
    parser.add_argument("--suffix", default="_flipped", help="输出文件名后缀 (默认: _flipped)")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("flip_image")

    images = collect_images(args.input)
    if not images:
        log.error("未找到图片: %s", args.input)
        sys.exit(1)

    input_path = Path(args.input)
    if args.output:
        save_dir = Path(args.output)
    elif input_path.is_dir():
        save_dir = input_path / f"{input_path.name}_flipped"
    else:
        save_dir = input_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    log.info("结果保存目录: %s", save_dir.resolve())

    ok_count = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("读取失败，跳过: %s", img_path)
            continue

        flipped = cv2.flip(img, 0)  # flipCode=0: 绕 x 轴翻转 (上下颠倒)
        out_path = save_dir / f"{img_path.stem}{args.suffix}{img_path.suffix}"
        cv2.imwrite(str(out_path), flipped)
        h, w = img.shape[:2]
        log.info("已处理: %s (%dx%d) -> %s", img_path.name, w, h, out_path.name)
        ok_count += 1

    log.info("完成: 共处理 %d/%d 张", ok_count, len(images))


if __name__ == "__main__":
    main()
