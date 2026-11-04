import argparse
import shutil
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm import tqdm

""" @brief YOLO OBB 数据增强 + 可选重命名:
# 0、进入虚拟环境
source ~/yolo_venv/bin/activate

# 1、默认: 只做增强，不重命名
python ~/realsense_yolo/train_pipeline/tools/augment_images.py

# 2、只做重命名, 从 0163.jpg/.txt 开始编号
python ~/realsense_yolo/train_pipeline/tools/augment_images.py --mode rename --rename-start 163

# 3、先增强, 再对输出目录统一重命名, 从 0001.jpg/.txt 开始
python ~/realsense_yolo/train_pipeline/tools/augment_images.py --mode all --rename-start 1

# 4、自定义输入/输出目录（四个目录都有默认值, 按需覆盖）
python ~/realsense_yolo/train_pipeline/tools/augment_images.py \
  --src-img-dir /path/to/images --src-lbl-dir /path/to/labels \
  --out-img-dir /path/to/out --out-lbl-dir /path/to/out \
  --mode all --rename-start 1

"""

# ============ 默认配置（均可用同名 --参数覆盖, 见 main）============
src_img_dir = Path("~/realsense_yolo/train_pipeline/data_raw/phone_1027")
src_lbl_dir = Path("~/realsense_yolo/train_pipeline/data_raw/phone_1027")
out_img_dir = Path("~/realsense_yolo/train_pipeline/datasets/phone_obb/temp")
out_lbl_dir = Path("~/realsense_yolo/train_pipeline/datasets/phone_obb/temp")
AUG_PER_IMAGE = 1               # 每张原图增强生成的样本数
MIN_VISIBLE_AREA_RATIO = 0.95  # 最小可见面积比例, 低于此比例的 OBB 会被丢弃
# ==================================================================


def load_obb_labels(lbl_path, img_w, img_h):
    boxes = []
    if not lbl_path.exists():
        return boxes
    bad_lines = 0
    for line in lbl_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        try:
            if len(parts) != 9:
                raise ValueError(f"期望 9 个字段，实际 {len(parts)} 个")
            cls_id = int(parts[0])
            coords = np.array(list(map(float, parts[1:])), dtype=np.float64)
            pts = coords.reshape(4, 2) * np.array([img_w, img_h])
        except (ValueError, IndexError):
            bad_lines += 1
            continue
        boxes.append((cls_id, pts))
    if bad_lines:
        print(f"[警告] {lbl_path.name} 有 {bad_lines} 行标签格式错误，已跳过")
    return boxes


def polygon_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def clip_obb(pts, img_w, img_h):
    """把 OBB 裁剪到图像边界内：取可见区域的最小外接矩形，近似保持原方向。"""
    img_rect = np.array([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]], dtype=np.float32)
    pts_f = pts.astype(np.float32)
    try:
        inter_area, inter = cv2.intersectConvexConvex(img_rect, pts_f)
    except cv2.error:
        return None
    if inter is None or inter_area <= 0:
        return None
    vis_pts = inter.reshape(-1, 2)
    if len(vis_pts) < 3:
        return None
    orig_area = polygon_area(pts)
    if inter_area < orig_area * MIN_VISIBLE_AREA_RATIO:
        return None
    rect = cv2.minAreaRect(vis_pts)
    return cv2.boxPoints(rect).astype(np.float64)


def augment_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, fill=(0, 0, 0)),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(std_range=(0.03, 0.08), p=0.3),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def process_one(img_path, lbl_path, out_prefix, transform):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[跳过] 无法读取图片：{img_path.name}")
        return 0
    h, w = img.shape[:2]
    boxes = load_obb_labels(lbl_path, w, h)
    if not boxes:
        print(f"[跳过] 缺少有效标签：{lbl_path.name}")
        return 0
    all_kps = []
    for _, pts in boxes:
        all_kps.extend(pts.tolist())
    try:
        res = transform(image=img, keypoints=all_kps)
    except Exception:
        print(f"[跳过] 增强失败：{img_path.name}")
        return 0
    aug_img = res["image"]
    ah, aw = aug_img.shape[:2]
    aug_kps = np.array(res["keypoints"], dtype=np.float64).reshape(-1, 4, 2)
    new_lines = []
    for i, (cls_id, orig_pts) in enumerate(boxes):
        if i >= len(aug_kps):
            break
        pts = aug_kps[i]
        clipped = clip_obb(pts, aw, ah)
        if clipped is None:
            continue
        norm = clipped / np.array([aw, ah])
        norm = np.clip(norm, 0.0, 1.0)
        line = f"{cls_id} " + " ".join(f"{v:.6f}" for v in norm.flatten())
        new_lines.append(line)
    if not new_lines:
        print(f"[跳过] 增强后无有效框：{img_path.name}")
        return 0
    cv2.imwrite(str(out_img_dir / f"{out_prefix}.jpg"), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (out_lbl_dir / f"{out_prefix}.txt").write_text("\n".join(new_lines), encoding="utf-8")
    return 1


def run_augment(use_tmp_names=False):
    """
    只做数据增强，输出文件名沿用原图主文件名 + _augK 后缀。
    use_tmp_names=True 时，输出写成 __augtmp_i__ 临时名，并返回临时对列表，
    供后续 run_rename 统一编号；否则直接写最终文件名。
    """
    transform = augment_pipeline()
    img_files = sorted([p for p in src_img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    count = 0
    tmp_pairs = []
    for img_path in tqdm(img_files, desc="Augmenting"):
        stem = img_path.stem
        lbl_path = src_lbl_dir / f"{stem}.txt"
        for k in range(AUG_PER_IMAGE):
            if use_tmp_names:
                out_prefix = f"__augtmp_{len(tmp_pairs)}__"
            else:
                out_prefix = f"{stem}_aug{k}"
            ok = process_one(img_path, lbl_path, out_prefix, transform)
            if ok:
                count += 1
                if use_tmp_names:
                    tmp_pairs.append((
                        out_img_dir / f"{out_prefix}.jpg",
                        out_lbl_dir / f"{out_prefix}.txt",
                        ".jpg",
                    ))
    print(f"增强完成，生成 {count} 张样本 -> {out_img_dir} / {out_lbl_dir}")
    return tmp_pairs


def run_rename(start_id=1, digits=4, dry_run=False, existing_pairs=None):
    """
    从 src_img_dir / src_lbl_dir 读取图片和同名标签，按图片文件名排序后
    从 start_id 开始顺序编号，拷贝并重命名到 out_img_dir / out_lbl_dir。
    existing_pairs 不为空时，不再从 src 拷贝，而是直接对这批已存在的
    (img_path, lbl_path, suffix) 临时对编号，用于 all 模式串联增强结果。
    """
    if existing_pairs is None:
        src_imgs = sorted([
            p for p in src_img_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        if not src_imgs:
            print("源目录中没有图片，跳过重命名")
            return
        mapping = []
        n_missing_lbl = 0
        for idx, img_path in enumerate(src_imgs):
            lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                n_missing_lbl += 1
                continue
            new_id = start_id + len(mapping)
            new_stem = f"{new_id:0{digits}d}"
            mapping.append((img_path, lbl_path, new_stem, img_path.suffix))
        copy_mode = True
        if n_missing_lbl:
            print(f"[提示] {n_missing_lbl} 张图片缺少标签，已跳过（输出编号保持连续）")
    else:
        mapping = []
        for img_path, lbl_path, suffix in existing_pairs:
            new_id = start_id + len(mapping)
            new_stem = f"{new_id:0{digits}d}"
            mapping.append((img_path, lbl_path, new_stem, suffix))
        copy_mode = False

    if not mapping:
        print("没有可重命名的图片-标签对")
        return

    if dry_run:
        for img_path, lbl_path, new_stem, suffix in mapping:
            print(f"{img_path.name} -> {new_stem}{suffix} | "
                  f"{lbl_path.name} -> {new_stem}.txt")
        print(f"预览模式，共 {len(mapping)} 组，未实际修改")
        return

    tmp_pairs = []
    try:
        for i, (img_path, lbl_path, new_stem, suffix) in enumerate(mapping):
            if copy_mode:
                tmp_stem = f"__tmp_{i}__"
                tmp_img = out_img_dir / f"{tmp_stem}{suffix}"
                tmp_lbl = out_lbl_dir / f"{tmp_stem}.txt"
                shutil.copy2(img_path, tmp_img)
                shutil.copy2(lbl_path, tmp_lbl)
            else:
                tmp_img = img_path
                tmp_lbl = lbl_path
            tmp_pairs.append((tmp_img, tmp_lbl, new_stem, suffix))

        for tmp_img, tmp_lbl, new_stem, suffix in tmp_pairs:
            final_img = out_img_dir / f"{new_stem}{suffix}"
            final_lbl = out_lbl_dir / f"{new_stem}.txt"
            if final_img.exists():
                final_img.unlink()
            if final_lbl.exists():
                final_lbl.unlink()
            tmp_img.rename(final_img)
            tmp_lbl.rename(final_lbl)
    finally:
        # 中途失败时清理残留的临时文件，避免半成品混入输出目录
        for tmp_img, tmp_lbl, _, _ in tmp_pairs:
            for p in (tmp_img, tmp_lbl):
                if p.exists() and (p.stem.startswith("__tmp_") or p.stem.startswith("__augtmp_")):
                    p.unlink()

    print(f"重命名完成，共 {len(tmp_pairs)} 组，起始编号 {start_id}，位数 {digits}")


def main():
    global src_img_dir, src_lbl_dir, out_img_dir, out_lbl_dir
    parser = argparse.ArgumentParser(description="YOLO OBB 数据增强 + 可选重命名")
    parser.add_argument("--mode", choices=["augment", "rename", "all"], default="augment",
                        help="augment=只增强; rename=只重命名; all=先增强再重命名")
    parser.add_argument("--rename-start", type=int, default=1,
                        help="重命名起始编号，默认 1")
    parser.add_argument("--digits", type=int, default=4,
                        help="编号补零位数，默认 4（即 0001）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅预览重命名映射，不实际修改文件")
    parser.add_argument("--src-img-dir", type=Path, default=src_img_dir,
                        help=f"源图片目录，默认 {src_img_dir}")
    parser.add_argument("--src-lbl-dir", type=Path, default=src_lbl_dir,
                        help=f"源标签目录，默认 {src_lbl_dir}")
    parser.add_argument("--out-img-dir", type=Path, default=out_img_dir,
                        help=f"输出图片目录，默认 {out_img_dir}")
    parser.add_argument("--out-lbl-dir", type=Path, default=out_lbl_dir,
                        help=f"输出标签目录，默认 {out_lbl_dir}")
    args = parser.parse_args()

    src_img_dir, src_lbl_dir = args.src_img_dir, args.src_lbl_dir
    out_img_dir, out_lbl_dir = args.out_img_dir, args.out_lbl_dir
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "augment":
        run_augment(use_tmp_names=False)
    elif args.mode == "rename":
        run_rename(start_id=args.rename_start, digits=args.digits, dry_run=args.dry_run)
    elif args.mode == "all":
        if args.dry_run:
            print("[预览] dry-run：不生成增强文件，仅预览对源图重命名的映射")
            run_rename(start_id=args.rename_start, digits=args.digits, dry_run=True)
        else:
            tmp_pairs = run_augment(use_tmp_names=True)
            run_rename(start_id=args.rename_start, digits=args.digits,
                       dry_run=False, existing_pairs=tmp_pairs)

if __name__ == "__main__":
    main()