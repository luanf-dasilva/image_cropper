# cleanup_dataset.py
# Usage:
#   python cleanup_dataset.py --dataset ./dataset --src "C:\\Users\\you\\Pictures\\raw" --exclude ./excluded
# Options:
#   --dry-run  : don't modify files, just print what would happen

import argparse, pathlib, shutil, re, csv, os
from typing import Optional

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def find_dataset_image(ds_images_dir: pathlib.Path, stem: str) -> Optional[pathlib.Path]:
    # Search both train/ and val/ for any matching extension
    for split in ["train", "val"]:
        folder = ds_images_dir / split
        if not folder.exists(): 
            continue
        for ext in IMG_EXTS:
            p = folder / f"{stem}{ext}"
            if p.exists():
                return p
    return None

def find_dataset_label(ds_labels_dir: pathlib.Path, stem: str) -> Optional[pathlib.Path]:
    for split in ["train", "val"]:
        p = ds_labels_dir / split / f"{stem}.txt"
        if p.exists():
            return p
    return None

def find_original_in_src(src_root: pathlib.Path, stem: str) -> Optional[pathlib.Path]:
    # Walk once (could be lots; ok for one-off). If performance matters, index first.
    for p in src_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS and p.stem == stem:
            return p
    return None

def remove_from_csv(csv_path: pathlib.Path, stems_to_remove: set, dry: bool):
    if not csv_path.exists():
        return
    tmp = csv_path.with_suffix(".tmp")
    with open(csv_path, newline="", encoding="utf-8") as fin, open(tmp, "w", newline="", encoding="utf-8") as fout:
        r = csv.DictReader(fin)
        w = csv.DictWriter(fout, fieldnames=r.fieldnames)
        w.writeheader()
        for row in r:
            # row may have 'file' or 'name' column depending on your earlier script
            name = row.get("file") or row.get("name") or ""
            stem = pathlib.Path(name).stem
            if stem not in stems_to_remove:
                w.writerow(row)
    if dry:
        tmp.unlink(missing_ok=True)
    else:
        csv_path.unlink(missing_ok=True)
        tmp.rename(csv_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=False, default=None, help="root of original images (to move bad originals out)")
    ap.add_argument("--exclude", required=False, default=None, help="folder to move excluded originals")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ds = pathlib.Path("../aimage_cropper/dataset")
    ds_images = ds / "images"
    ds_labels = ds / "labels"
    review_dir = ds / "quarantined"             # <<< put bad previews here
    reviewed_dir = ds / "quarantined_orig"      # processed previews go here
    reviewed_dir.mkdir(parents=True, exist_ok=True)

    if not review_dir.exists():
        print(f"Nothing to do: {review_dir} does not exist.")
        return

    src_root = pathlib.Path(args.src) if args.src else None
    exclude_root = pathlib.Path(args.exclude) if args.exclude else None
    if src_root and exclude_root:
        exclude_root.mkdir(parents=True, exist_ok=True)

    # parse stems from preview filenames like: conf0.23_filename_prev.jpg
    # Robust regex: grab text after first 'conf...' underscore up to extension
    # Examples it handles:
    #   conf0.12_IMG_1234_prev.jpg  -> stem IMG_1234_prev
    #   00012_conf0.45_IMG_9999.jpg -> stem IMG_9999
    stem_re = re.compile(r"conf[\d.]+_(.+)\.jpg$", re.IGNORECASE)

    to_process = []
    for p in review_dir.iterdir():
        if p.suffix.lower() != ".jpg":
            continue
        m = stem_re.search(p.name)
        if m:
            stem = m.group(1)
        else:
            # fallback: strip any prefix up to first '_' then remove extension
            maybe = p.stem.split("_")[-1]
            stem = maybe
        to_process.append((p, stem))

    if not to_process:
        print(f"No preview JPGs found in {review_dir}")
        return

    removed_imgs, removed_lbls, moved_orig, processed_prev = 0, 0, 0, 0
    missing = []

    stems_removed = set()

    for prev_path, stem in to_process:
        img_path = find_dataset_image(ds_images, stem)
        lbl_path = find_dataset_label(ds_labels, stem)

        # Move preview to reviewed/
        dest_prev = reviewed_dir / img_path.name  
        print(("DRY " if args.dry_run else "") + f"MOVE image → reviewed: {img_path} -> {dest_prev}")
        if not args.dry_run:
            shutil.move(str(img_path), str(dest_prev))
        processed_prev += 1

        # Delete dataset image/label if exist
        if prev_path and prev_path.exists():
            print(("DRY " if args.dry_run else "") + f"DEL dataset image: {prev_path}")
            if not args.dry_run: prev_path.unlink()
            removed_imgs += 1
        else:
            print(f"(!) dataset image missing for stem {stem}")
        if lbl_path and lbl_path.exists():
            print(("DRY " if args.dry_run else "") + f"DEL dataset label: {lbl_path}")
            if not args.dry_run: lbl_path.unlink()
            removed_lbls += 1
        else:
            print(f"(!) dataset label missing for stem {stem}")

        stems_removed.add(stem)

        # Move original from src → exclude (optional)
        if src_root and exclude_root:
            orig = find_original_in_src(src_root, stem)
            if orig:
                rel = orig.relative_to(src_root) if orig.is_relative_to(src_root) else pathlib.Path(orig.name)
                dest = exclude_root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                print(("DRY " if args.dry_run else "") + f"MOVE original → exclude: {orig}  ->  {dest}")
                if not args.dry_run:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(orig), str(dest))
                moved_orig += 1
            else:
                missing.append(stem)

    # Clean labels.csv if present
    csv_path = ds / "labels.csv"
    if csv_path.exists():
        print(("DRY " if args.dry_run else "") + f"UPDATE {csv_path} (remove {len(stems_removed)} rows)")
        remove_from_csv(csv_path, stems_removed, args.dry_run)

    print("\nSummary:")
    print(f"  Previews processed: {processed_prev}")
    print(f"  Dataset images deleted: {removed_imgs}")
    print(f"  Dataset labels deleted: {removed_lbls}")
    if src_root and exclude_root:
        print(f"  Originals moved to exclude/: {moved_orig}")
        if missing:
            print(f"  Originals not found for stems (check extensions/paths): {', '.join(missing[:10])}{' ...' if len(missing)>10 else ''}")
    print("\nDone.")

if __name__ == "__main__":
    main()
