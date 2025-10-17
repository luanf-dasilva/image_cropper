import argparse, pathlib, shutil, os, cv2

IMG_EXTS_DEFAULT = {"jpg","jpeg","png","bmp","webp", "heic", "aae"}

def list_images(root: pathlib.Path, exts):
    exts = {("."+e.lower()) if not e.startswith(".") else e.lower() for e in exts}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])

def ensure_dataset(dst_root: pathlib.Path):
    (dst_root/"images"/"train").mkdir(parents=True, exist_ok=True)
    (dst_root/"images"/"val").mkdir(parents=True, exist_ok=True)
    (dst_root/"labels"/"train").mkdir(parents=True, exist_ok=True)
    (dst_root/"labels"/"val").mkdir(parents=True, exist_ok=True)
    # create a minimal data.yaml if missing
    yaml = dst_root/"data.yaml"
    if not yaml.exists():
        yaml.write_text(
            f"train: {str((dst_root/'images'/'train').resolve())}\n"
            f"val: {str((dst_root/'images'/'val').resolve())}\n"
            "nc: 1\nnames: ['content']\n",
            encoding="utf-8"
        )

class CropUI:
    def __init__(self, img_bgr, max_view=1400):
        self.img = img_bgr
        H, W = img_bgr.shape[:2]
        m = max(H, W)
        self.scale = 1.0 if m <= max_view else max_view/float(m)
        self.view = cv2.resize(img_bgr, (int(W*self.scale), int(H*self.scale)), cv2.INTER_AREA) if self.scale<1.0 else img_bgr.copy()
        self.base = self.view.copy()
        self.pt0 = None; self.pt1 = None; self.drag = False
        cv2.namedWindow("annotate", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("annotate", self.on_mouse)
        self.redraw()

    def on_mouse(self, ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.drag = True; self.pt0 = (x,y); self.pt1 = (x,y); self.redraw()
        elif ev == cv2.EVENT_MOUSEMOVE and self.drag:
            self.pt1 = (x,y); self.redraw()
        elif ev == cv2.EVENT_LBUTTONUP and self.drag:
            self.drag = False; self.pt1 = (x,y); self.redraw()

    def redraw(self):
        self.view[:] = self.base
        if self.pt0 and self.pt1:
            cv2.rectangle(self.view, self.pt0, self.pt1, (0,255,0), 2)
        y = 22
        for t in ["Drag = box", "Enter = save label", "d = delete original", "n = skip", "r = reset", "q/ESC = quit"]:
            cv2.putText(self.view, t, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (15,240,15), 2, cv2.LINE_AA); y+=24
        cv2.imshow("annotate", self.view)

    def reset(self):
        self.pt0 = self.pt1 = None
        self.view[:] = self.base
        self.redraw()

    def get_box_xyxy_fullres(self):
        if not (self.pt0 and self.pt1): return None
        (x0,y0),(x1,y1) = self.pt0, self.pt1
        x0,x1 = sorted((x0,x1)); y0,y1 = sorted((y0,y1))
        if x1-x0 < 3 or y1-y0 < 3: return None
        s = self.scale
        fx0, fy0 = int(round(x0/s)), int(round(y0/s))
        fx1, fy1 = int(round(x1/s)), int(round(y1/s))
        H, W = self.img.shape[:2]
        fx0 = max(0,min(W-1,fx0)); fx1 = max(0,min(W-1,fx1))
        fy0 = max(0,min(H-1,fy0)); fy1 = max(0,min(H-1,fy1))
        return (fx0, fy0, fx1, fy1)

def yolo_line_from_xyxy(x0,y0,x1,y1,W,H,class_id):
    # YOLO: class cx cy w h (normalized 0..1)
    cx = ((x0+x1)/2.0)/W
    cy = ((y0+y1)/2.0)/H
    w  = (x1-x0)/float(W)
    h  = (y1-y0)/float(H)
    cx = max(0,min(1,cx)); cy=max(0,min(1,cy)); w=max(0,min(1,w)); h=max(0,min(1,h))
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"

def move_or_copy(src, dst, do_copy=False):
    src = pathlib.Path(src)
    dst = pathlib.Path(dst)  # convert to Path so .parent works
    dst.parent.mkdir(parents=True, exist_ok=True)
    if do_copy:
        shutil.copy2(str(src), str(dst))
    else:
        shutil.move(str(src), str(dst))

def remove_or_trash(path, trash):
    if not path.exists(): return
    if trash:
        trash = pathlib.Path(trash); trash.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(trash/path.name))
    else:
        path.unlink(missing_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with raw images")
    ap.add_argument("--dst", default="../aimage_cropper/dataset", help="YOLO dataset root")
    ap.add_argument("--split", default="train", choices=["train","val"])
    ap.add_argument("--val-every", type=int, default=0, help="send every Nth accepted image to val (0=never)")
    ap.add_argument("--class-id", type=int, default=0)
    ap.add_argument("--move", action="store_true", default=True)
    ap.add_argument("--copy", action="store_true", help="copy instead of move")
    ap.add_argument("--trash", default="", help="deleted originals go here if set")
    ap.add_argument("--max-view", type=int, default=1400)
    ap.add_argument("--exts", default=",".join(sorted(IMG_EXTS_DEFAULT)))
    args = ap.parse_args()

    do_copy = bool(args.copy)
    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)
    ensure_dataset(dst)

    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]
    files = list_images(src, exts)
    if not files:
        print("No images found."); return

    accepted = 0
    i = 0
    while i < len(files):
        f = files[i]
        img = cv2.imread(str(f))
        if img is None:
            print(f"❌ cannot read {f}"); i += 1; continue

        # pick split (every Nth to val)
        split = "val" if (args.val_every and ((accepted+1) % args.val_every == 0)) else args.split

        ui = CropUI(img, max_view=args.max_view)
        while True:
            key = cv2.waitKey(20) & 0xFFFF
            if key in (13,10) or key in (ord('a'), ord('A')):  # Enter / A = save
                box = ui.get_box_xyxy_fullres()
                H, W = img.shape[:2]

                if box:
                    x0, y0, x1, y1 = box
                else:
                    x0, y0, x1, y1 = 0, 0, W - 1, H - 1

                yolo_line = yolo_line_from_xyxy(x0, y0, x1, y1, W, H, args.class_id)
                img_dst = dst / "images" / split / f.name
                lbl_dst = dst / "labels" / split / (f.stem + ".txt")

                move_or_copy(str(f), str(img_dst), do_copy=do_copy)
                lbl_dst.parent.mkdir(parents=True, exist_ok=True)
                with open(lbl_dst, "w", encoding="utf-8") as fo:
                    fo.write(yolo_line)

                print(f"✅ saved: {img_dst} + {lbl_dst}")
                labels_csv = dst / "labels.csv"
                if labels_csv.exists():
                    import csv
                    with open(labels_csv, "a", newline="", encoding="utf-8") as cf:
                        writer = csv.writer(cf)
                        writer.writerow([f.name, args.class_id, x0, y0, x1, y1, W, H])
                accepted += 1
                i += 1
                break

            elif key in (ord('d'), ord('D')):  # delete original
                remove_or_trash(f, args.trash)
                print(f"🗑️ deleted {f}")
                i += 1
                break

            elif key in (ord('n'), ord('N')):  # skip
                print(f"⏭️ skipped {f}")
                i += 1
                break

            elif key in (ord('r'), ord('R')):  # reset box
                ui.reset()

            elif key in (27, ord('q'), ord('Q')):  # esc/q
                print("👋 Bye.")
                cv2.destroyAllWindows()
                return

            cv2.imshow("annotate", ui.view)

    cv2.destroyAllWindows()
    print("✅ Done.")

if __name__ == "__main__":
    main()
