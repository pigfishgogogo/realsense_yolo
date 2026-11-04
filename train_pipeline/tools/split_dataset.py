import argparse
import random
import shutil
from pathlib import Path

""" @brief 将 raw 中的图片-标签对整体打乱后按 7:3 拆分到 train / val:
# 0、进入yolo虚拟环境
source ~/yolo_venv/bin/activate

# 1、默认：移动（raw 里的文件被搬走），随机种子 42
python ~/realsense_yolo/train_pipeline/tools/split_dataset.py

# 2、只拷贝不移动，换随机种子
python ~/realsense_yolo/train_pipeline/tools/split_dataset.py --copy --seed 7

# 3、预览拆分结果，不实际动文件
python ~/realsense_yolo/train_pipeline/tools/split_dataset.py --dry-run

# 4、自定义目录（五个路径都有默认值，按需覆盖，支持 ~ 简写）
python ~/realsense_yolo/train_pipeline/tools/split_dataset.py \
  --raw-dir ~/data/raw --img-train-dir ~/data/images/train \
  --img-val-dir ~/data/images/val --lbl-train-dir ~/data/labels/train \
  --lbl-val-dir ~/data/labels/val

"""

# ============ 默认配置（均可用同名 --参数覆盖，见 main）============
raw_dir = Path("~/realsense_yolo/train_pipeline/datasets/phone_obb_raw/raw")
img_train_dir = Path("~/realsense_yolo/train_pipeline/datasets/phone_obb_raw/images/train")
img_val_dir = Path("~/realsense_yolo/train_pipeline/datasets/phone_obb_raw/images/val")
lbl_train_dir = Path("~/realsense_yolo/train_pipeline/datasets/phone_obb_raw/labels/train")
lbl_val_dir = Path("~/realsense_yolo/train_pipeline/datasets/phone_obb_raw/labels/val")
VAL_RATIO = 0.3          # val 占比，train 为 1 - VAL_RATIO
SEED = 42                # 随机种子，固定后每次拆分结果可复现
# ==================================================================

IMG_EXTS = (".jpg", ".jpeg", ".png")


def collect_pairs():
    """按主文件名配对 raw 中的图片和标签，只保留成对的。"""
    imgs = {p.stem: p for p in raw_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    pairs, missing_lbl, missing_img = [], [], []
    for stem, img_path in sorted(imgs.items()):
        lbl_path = raw_dir / f"{stem}.txt"
        if lbl_path.exists():
            pairs.append((stem, img_path, lbl_path))
        else:
            missing_lbl.append(stem)
    paired_stems = {s for s, _, _ in pairs}
    for p in raw_dir.iterdir():
        if p.suffix == ".txt" and p.stem not in paired_stems and p.stem != "classes":
            missing_img.append(p.stem)
    return pairs, missing_lbl, missing_img


def move_pair(stem, img_path, lbl_path, dst_img_dir, dst_lbl_dir, copy_mode):
    dst_img = dst_img_dir / img_path.name
    dst_lbl = dst_lbl_dir / f"{stem}.txt"
    for dst in (dst_img, dst_lbl):
        if dst.exists():
            dst.unlink()
    if copy_mode:
        shutil.copy2(img_path, dst_img)
        shutil.copy2(lbl_path, dst_lbl)
    else:
        img_path.rename(dst_img)
        lbl_path.rename(dst_lbl)


def main():
    global raw_dir, img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir
    parser = argparse.ArgumentParser(description="将 raw 数据按 7:3 打乱拆分到 train/val")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子，默认 42")
    parser.add_argument("--copy", action="store_true", help="拷贝而非移动（raw 保留原文件）")
    parser.add_argument("--dry-run", action="store_true", help="仅预览拆分结果，不实际动文件")
    parser.add_argument("--raw-dir", type=Path, default=raw_dir,
                        help=f"raw 图片-标签对所在目录，默认 {raw_dir}")
    parser.add_argument("--img-train-dir", type=Path, default=img_train_dir,
                        help=f"训练图片输出目录，默认 {img_train_dir}")
    parser.add_argument("--img-val-dir", type=Path, default=img_val_dir,
                        help=f"验证图片输出目录，默认 {img_val_dir}")
    parser.add_argument("--lbl-train-dir", type=Path, default=lbl_train_dir,
                        help=f"训练标签输出目录，默认 {lbl_train_dir}")
    parser.add_argument("--lbl-val-dir", type=Path, default=lbl_val_dir,
                        help=f"验证标签输出目录，默认 {lbl_val_dir}")
    args = parser.parse_args()

    raw_dir = args.raw_dir.expanduser()
    img_train_dir = args.img_train_dir.expanduser()
    img_val_dir = args.img_val_dir.expanduser()
    lbl_train_dir = args.lbl_train_dir.expanduser()
    lbl_val_dir = args.lbl_val_dir.expanduser()

    for d in (img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir):
        d.mkdir(parents=True, exist_ok=True)

    pairs, missing_lbl, missing_img = collect_pairs()
    if not pairs:
        print(f"在 {raw_dir} 中没有找到成对的图片-标签，退出")
        return
    if missing_lbl:
        print(f"[警告] {len(missing_lbl)} 张图片缺少同名 .txt，已跳过: {missing_lbl[:5]}"
              + (" ..." if len(missing_lbl) > 5 else ""))
    if missing_img:
        print(f"[警告] {len(missing_img)} 个 .txt 没有同名图片，已跳过: {missing_img[:5]}"
              + (" ..." if len(missing_img) > 5 else ""))

    rng = random.Random(args.seed)
    rng.shuffle(pairs)  # 打乱的是 (stem, img, lbl) 三元组，图片与标签的对应关系不会散

    n_val = max(1, round(len(pairs) * VAL_RATIO))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    print(f"共 {len(pairs)} 对: train {len(train_pairs)} 对, val {len(val_pairs)} 对 "
          f"(比例 {1 - VAL_RATIO:.1f}:{VAL_RATIO:.1f}, seed={args.seed})")

    if args.dry_run:
        print("--- val 预览 ---")
        for stem, _, _ in val_pairs[:10]:
            print(f"  {stem} -> val")
        print(f"  ... 共 {len(val_pairs)} 对" if len(val_pairs) > 10 else "")
        print("预览模式，未实际修改")
        return

    verb = "拷贝" if args.copy else "移动"
    for stem, img_path, lbl_path in train_pairs:
        move_pair(stem, img_path, lbl_path, img_train_dir, lbl_train_dir, args.copy)
    for stem, img_path, lbl_path in val_pairs:
        move_pair(stem, img_path, lbl_path, img_val_dir, lbl_val_dir, args.copy)

    print(f"{verb}完成: train {len(train_pairs)} 对 -> images/train + labels/train, "
          f"val {len(val_pairs)} 对 -> images/val + labels/val")


if __name__ == "__main__":
    main()
