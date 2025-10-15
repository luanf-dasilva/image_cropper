# manual_cropper.py
# Usage:
#   python manual_cropper.py --src ./raw --dst ./aimage_cropper/dataset --split train
# Optional:
#   --trash ./trash    # move deleted/processed originals instead of removing

import argparse, pathlib, cv2, os, shutil

IMG_EXTS_DEFAULT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(root: pathlib.Path, exts):
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])

class CropUI:
    def __init__(self, img_bgr, win="cropper", max_view=1400):
        self.win = win
        self.img_full = img_bgr
        self.H, self.W = img_bgr.shape[:2]
        m = max(self.H, self.W)
        self.scale = 1.0 if m <= max_view else max_view / float(m)
        if self.scale < 1.0:
            self.view = cv2.resize(self.img_full, (int(self.W*self.scale), int(self.H*self.scale)), cv2.INTER_AREA)
        else:
            self.view = self.img_full.copy()
        self.view_base = self.view.copy()
        self.dragging = False
        self.pt0 = None
        self.pt1 = None
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.win, self.on_mouse)
        self.redraw()

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.pt0 = (x, y)
            self.pt1 = (x, y)
            self.redraw()
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.pt1 = (x, y)
            self.redraw()
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.pt1 = (x, y)
            self.redraw()

    def current_box_fullres(self):
        if not self.pt0 or not self.pt1:
            return None
        x0, y0 = self.pt0; x1, y1 = self.pt1
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        s = self.scale
        fx0, fy0 = int(round(x0 / s)), int(round(y0 / s))
        fx1, fy1 = int(round(x1 / s)), int(round(y1 / s))
        fx0, fx1 = max(0, min(self.W-1, fx0)), max(0, min(self.W-1, fx1))
        fy0, fy1 = max(0, min(self.H-1, fy0)), max(0, min(self.H-1, fy1))
        w, h = max(1, fx1 - fx0), max(1, fy1 - fy0)
        if w < 3 or h < 3: return None
        return (fx0, fy0, w, h)

    def reset(self):
        self.pt0 = self.pt1 = None
        self.view = self.view_base.copy()
        self.redraw(hard=True)

    def overlay_help(self, canvas):
        lines = [
            "Drag = crop box",
            "Enter = save",
            "d = delete original",
            "n = skip",
            "r = reset box",
            "q/ESC = quit",
        ]
        y = 24
        for t in lines:
            cv2.putText(canvas, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (15, 240, 15), 2, cv2.LINE_AA)
            y += 24

    def redraw(self, hard=False):
        if hard:
            self.view = self.view_base.copy()
        else:
            self.view[:] = self.view_base
        if self.pt0 and self.pt1:
            cv2.rectangle(self.view, self.pt0, self.pt1, (0,255,0), 2)
        self.overlay_help(self.view)
        cv2.imshow(self.win, self.view)

def ensure_dirs(dst_root: pathlib.Path, split: str):
    img_dir = dst_root / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    return img_dir

def remove_or_trash(f, trash_dir):
    if trash_dir:
        dest = trash_dir / f.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(dest))
        print(f"🗑️ moved original to {dest}")
    else:
        try:
            os.remove(f)
            print(f"🗑️ deleted original {f}")
        except Exception as e:
            print(f"⚠️ could not delete {f}: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="./aimage_cropper/dataset")
    ap.add_argument("--split", default="train", choices=["train","val","test"])
    ap.add_argument("--max-view", type=int, default=1400)
    ap.add_argument("--trash", default="", help="if set, move originals here instead of deleting")
    ap.add_argument("--exts", default=",".join(IMG_EXTS_DEFAULT))
    args = ap.parse_args()

    exts = {e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
            for e in args.exts.split(",") if e.strip()}
    src = pathlib.Path(args.src)
    dst_root = pathlib.Path(args.dst)
    out_img_dir = ensure_dirs(dst_root, args.split)
    trash_dir = pathlib.Path(args.trash) if args.trash else None
    if trash_dir: trash_dir.mkdir(parents=True, exist_ok=True)

    files = list_images(src, exts)
    if not files:
        print("No images found.")
        return

    idx = 0
    while idx < len(files):
        f = files[idx]
        img = cv2.imread(str(f))
        if img is None:
            print(f"❌ cannot read {f}")
            idx += 1
            continue

        ui = CropUI(img, win="cropper", max_view=args.max_view)

        while True:
            key = cv2.waitKey(20) & 0xFFFF
            if key in (13, 10) or key in (ord('s'), ord('S')):  # Enter to save
                box = ui.current_box_fullres()
                crop = img if not box else img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                out_path = out_img_dir / f.name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if cv2.imwrite(str(out_path), crop):
                    remove_or_trash(f, trash_dir)
                    print(f"✅ saved crop to {out_path}")
                idx += 1
                break

            elif key in (ord('d'), ord('D')):  # delete original without saving
                remove_or_trash(f, trash_dir)
                idx += 1
                break

            elif key in (ord('n'), ord('N')):  # skip
                print(f"⏭️ skipped {f}")
                idx += 1
                break

            elif key in (ord('r'), ord('R')):
                ui.reset()

            elif key in (27, ord('q'), ord('Q')):  # esc or q
                print("👋 Bye.")
                cv2.destroyAllWindows()
                return

            cv2.imshow("cropper", ui.view)

    cv2.destroyAllWindows()
    print("✅ All done.")

if __name__ == "__main__":
    main()
