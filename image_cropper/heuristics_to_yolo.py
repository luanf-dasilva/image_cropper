# heuristic_to_yolo.py
# Auto-runs your heuristic cropper to generate YOLO labels and previews.
# Moves unreadable or failed images into dataset/errors/.

import os, cv2, csv, pathlib, shutil, argparse, random
import numpy as np
from image_cropper import auto_crop  # make sure it supports return_box=True, return_conf=True

def to_yolo_line(box, W, H, cls=0):
    x,y,w,h = box
    cx = (x + w/2.0) / W; cy = (y + h/2.0) / H
    nw = w / W; nh = h / H
    return f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

def draw_box(img, box, color=(0,255,0)):
    x,y,w,h = map(int, box)
    out = img.copy()
    cv2.rectangle(out, (x,y), (x+w,y+h), color, 2)
    return out

def area_frac(box, W, H):
    _,_,w,h = box
    return (w*h) / float(W*H + 1e-9)

def ar_dev(box):
    _,_,w,h = box
    ar = w / float(h + 1e-6)
    return abs(np.log(ar))  # 0 at square

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="../aimage_cropper/dataset")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--min-area", type=float, default=0.08)
    ap.add_argument("--max-area", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)
    errors = dst / "errors"
    for d in [dst/"images/train", dst/"labels/train", dst/"previews", errors]:
        d.mkdir(parents=True, exist_ok=True)

    exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    files = [p for p in src.rglob("*") if p.suffix.lower() in exts]
    random.shuffle(files)
    n_val = max(1, int(len(files)*args.val_split))
    val_set = set(files[:n_val])

    rows = []
    moved, kept = 0, 0

    for p in files:
        try:
            img = cv2.imread(str(p))
            if img is None:
                raise ValueError("cv2.imread returned None")
            H,W = img.shape[:2]

            box, _crop, conf = auto_crop(img, return_box=True, return_conf=True, debug=False)
            af = area_frac(box, W, H)

            # skip implausible boxes (still counts as error)
            if not (args.min_area <= af <= args.max_area):
                raise ValueError(f"bad area fraction {af:.2f}")

        except Exception as e:
            print(f"❌ {p.name}: {e}")
            err_path = errors / p.name
            shutil.move(str(p), str(err_path))
            moved += 1
            continue

        # valid box
        split = "val" if p in val_set else "train"
        out_img = dst/"images"/split/p.name
        shutil.copy2(p, out_img)

        yline = to_yolo_line(box, W, H, cls=0)
        with open(dst/"labels"/split/(p.stem + ".txt"), "w") as f:
            f.write(yline+"\n")

        prev = draw_box(img, box)
        prev_name = f"conf{conf:.2f}_{p.stem}.jpg"
        cv2.imwrite(str(dst/"previews"/prev_name), prev)

        rows.append({
            "file": p.name,
            "conf": float(conf),
            "area_frac": float(af),
            "ar_dev": float(ar_dev(box))
        })
        kept += 1

    # CSV index
    csv_path = dst/"labels.csv"
    with open(csv_path,"w",newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=["file","conf","area_frac","ar_dev"])
        w.writeheader()
        for r in rows: w.writerow(r)

    # data.yaml
    with open(dst/"data.yaml","w") as f:
        f.write(f"""train: {str((dst/'images'/'train').resolve()).replace('\\','/')}
val: {str((dst/'images'/'val').resolve()).replace('\\','/')}
nc: 1
names: ["content"]
""")

    print(f"\n✅ Kept: {kept}  |  🚫 Moved to errors/: {moved}")
    print(f"Previews: {dst/'previews'}")
    print(f"Next: check {errors} for bad files, then train YOLO.")

if __name__ == "__main__":
    main()
