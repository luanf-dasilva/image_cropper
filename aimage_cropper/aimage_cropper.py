import argparse, pathlib, shutil, os, sys
from typing import Iterable
import cv2
from ultralytics import YOLO

DEFAULT_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def iter_images(root: pathlib.Path, exts: Iterable[str]):
    exts = {e if e.startswith(".") else f".{e}" for e in (x.lower().strip() for x in exts)}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def ensure_parent(path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def crop_one(model: YOLO, path_in: str, path_out: str,
             conf: float, iou: float, imgsz: int,
             pad_frac: float, device: str, dry_run: bool) -> bool:
    """
    Returns True if a crop was written; False if fell back (no detections).
    """
    # Inference
    res = model.predict(
        source=path_in, conf=conf, iou=iou, imgsz=imgsz,
        device=device, verbose=False
    )[0]

    img = cv2.imread(path_in)
    if img is None:
        raise RuntimeError(f"cv2 failed to read: {path_in}")
    H, W = img.shape[:2]

    if res.boxes is None or len(res.boxes) == 0:
        return False

    # pick highest-confidence box
    b = res.boxes
    idx = int(b.conf.argmax())
    x1, y1, x2, y2 = map(int, b.xyxy[idx].tolist())

    # padding
    pad = int(pad_frac * max(x2 - x1, y2 - y1))
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    if not dry_run:
        ensure_parent(pathlib.Path(path_out))
        cv2.imwrite(path_out, crop)
    return True

def main():
    ap = argparse.ArgumentParser(description="YOLO auto-cropper")
    ap.add_argument("--model", required=True, help="Path to YOLO model .pt (e.g., runs/detect/train/weights/best.pt)")
    ap.add_argument("--src",   required=True, help="Source directory of images")
    ap.add_argument("--dst",   required=True, help="Destination directory for cropped images")
    ap.add_argument("--conf",  type=float, default=0.25, help="Confidence threshold (default 0.25)")
    ap.add_argument("--iou",   type=float, default=0.60, help="NMS IoU threshold (default 0.60)")
    ap.add_argument("--imgsz", type=int,   default=896,  help="Inference image size (default 896)")
    ap.add_argument("--pad",   type=float, default=0.03, help="Padding fraction added to detected box (default 0.03)")
    ap.add_argument("--device", default="0", help="Device id (e.g., '0' for first GPU) or 'cpu'")
    ap.add_argument("--on-miss", choices=["copy","skip"], default="copy",
                    help="If no detection: copy original or skip (default: copy)")
    ap.add_argument("--exts", default=",".join(sorted(DEFAULT_EXTS)).replace(".", ""),
                    help="Comma-separated extensions (e.g. 'jpg,png,webp'; default includes jpg,jpeg,png,bmp,webp)")
    ap.add_argument("--flatten", action="store_true",
                    help="Do not preserve subfolder structure under dst; put all outputs in dst root")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without writing files")
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)
    if not src.exists():
        print(f"❌ Source not found: {src}")
        sys.exit(1)

    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]
    model = YOLO(args.model)

    total = 0
    cropped = 0
    copied = 0
    skipped = 0

    for p in iter_images(src, exts):
        total += 1
        if args.flatten:
            out = dst / p.name
        else:
            out = dst / p.relative_to(src)

        if out.exists() and not args.overwrite:
            print(f"↩️  exists (skip, use --overwrite to replace): {out}")
            skipped += 1
            continue

        try:
            ok = crop_one(
                model=model,
                path_in=str(p),
                path_out=str(out),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                pad_frac=args.pad,
                device=args.device,
                dry_run=args.dry_run
            )
        except Exception as e:
            print(f"❌ {p} error: {e}")
            skipped += 1
            continue

        if ok:
            action = "✅ crop "
            cropped += 1
        else:
            if args.on-miss == "copy":
                if not args.dry_run:
                    ensure_parent(out)
                    shutil.copy2(str(p), str(out))
                action = "➡️  copy "
                copied += 1
            else:
                action = "⏭️  skip "
                skipped += 1

        print(f"{action}{out}")

    print("\n--- Summary ---")
    print(f"Total images: {total}")
    print(f"Cropped:      {cropped}")
    print(f"Copied (miss):{copied}")
    print(f"Skipped:      {skipped}")
    if args.device != "cpu":
        print(f"Device:       {args.device} (GPU expected if CUDA available)")

if __name__ == "__main__":
    main()
