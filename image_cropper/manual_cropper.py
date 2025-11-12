# manual_cropper_split.py
# Usage:
#   python manual_cropper_split.py --src ./raw --dst ./aimage_cropper/dataset [--split train] [--val-every 5]
# Keys:
#   Drag = draw box | Space = toggle split | 1=train | 2=val | Enter/A = save | d = delete | n = skip | r = reset | q/ESC = quit

import argparse, pathlib, shutil, os, csv
import cv2

IMG_EXTS_DEFAULT = {"jpg","jpeg","png","bmp","webp","heic","aae"}

def list_images(root: pathlib.Path, exts):
    exts = {("."+e.lower()) if not e.startswith(".") else e.lower() for e in exts}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])

def ensure_dataset(dst_root: pathlib.Path):
    (dst_root/"images"/"train").mkdir(parents=True, exist_ok=True)
    (dst_root/"images"/"val").mkdir(parents=True, exist_ok=True)
    (dst_root/"labels"/"train").mkdir(parents=True, exist_ok=True)
    (dst_root/"labels"/"val").mkdir(parents=True, exist_ok=True)
    yml = dst_root/"data.yaml"
    if not yml.exists():
        yml.write_text(
            f"train: {str((dst_root/'images'/'train').resolve())}\n"
            f"val: {str((dst_root/'images'/'val').resolve())}\n"
            "nc: 1\nnames: ['content']\n",
            encoding="utf-8"
        )
    # labels.csv header if missing
    csvp = dst_root/"labels.csv"
    if not csvp.exists():
        with open(csvp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file","split","class_id","x0","y0","x1","y1","W","H"])

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

    def on_mouse(self, ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.drag = True; self.pt0 = (x,y); self.pt1 = (x,y)
        elif ev == cv2.EVENT_MOUSEMOVE and self.drag:
            self.pt1 = (x,y)
        elif ev == cv2.EVENT_LBUTTONUP and self.drag:
            self.drag = False; self.pt1 = (x,y)

    def overlay(self, split_label="train"):
        self.view[:] = self.base
        if self.pt0 and self.pt1:
            cv2.rectangle(self.view, self.pt0, self.pt1, (0,255,0), 2)
        y = 24
        for t in [
            f"[TARGET SPLIT: {split_label.upper()}]  Space=toggle | 1=train | 2=val",
            "Drag=box | Enter/A=save | d=delete | n=skip | r=reset | q/ESC=quit"
        ]:
            cv2.putText(self.view, t, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (15,240,15), 2, cv2.LINE_AA)
            y += 26
        cv2.imshow("annotate", self.view)

    def reset(self):
        self.pt0 = self.pt1 = None

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
    cx = ((x0+x1)/2.0)/W
    cy = ((y0+y1)/2.0)/H
    w  = (x1-x0)/float(W)
    h  = (y1-y0)/float(H)
    cx = max(0,min(1,cx)); cy=max(0,min(1,cy)); w=max(0,min(1,w)); h=max(0,min(1,h))
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"

def move_or_copy(src, dst, do_copy=False):
    src = pathlib.Path(src); dst = pathlib.Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if do_copy: shutil.copy2(str(src), str(dst))
    else:       shutil.move(str(src), str(dst))

def remove_or_trash(path: pathlib.Path, trash: str):
    if not path.exists(): return
    if trash:
        trashp = pathlib.Path(trash); trashp.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(trashp/path.name))
    else:
        path.unlink(missing_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with raw images")
    ap.add_argument("--dst", default="./aimage_cropper/dataset", help="YOLO dataset root")
    ap.add_argument("--split", default="train", choices=["train","val"], help="default target split")
    ap.add_argument("--val-every", type=int, default=0, help="send every Nth accepted image to val (0=never)")
    ap.add_argument("--class-id", type=int, default=0)
    ap.add_argument("--copy", action="store_true", help="copy instead of move originals")
    ap.add_argument("--trash", default="", help="deleted originals go here if set")
    ap.add_argument("--max-view", type=int, default=1400)
    ap.add_argument("--exts", default=",".join(sorted(IMG_EXTS_DEFAULT)))
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)
    ensure_dataset(dst)
    do_copy = bool(args.copy)

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

        # default split, possibly overridden by val-every rule
        split = args.split
        if args.val_every and ((accepted + 1) % args.val_every == 0):
            split = "val"

        ui = CropUI(img, max_view=args.max_view)

        while True:
            ui.overlay(split)
            key = cv2.waitKey(20) & 0xFFFF

            if key in (ord(' '),):  # Space toggles split
                split = "val" if split == "train" else "train"

            elif key in (ord('1'),):  # force train
                split = "train"
            elif key in (ord('2'),):  # force val
                split = "val"

            elif key in (13,10, ord('a'), ord('A')):  # Enter/A = save
                box = ui.get_box_xyxy_fullres()
                H, W = img.shape[:2]
                if box:
                    x0,y0,x1,y1 = box
                else:
                    x0,y0,x1,y1 = 0,0,W-1,H-1  # full image if no box

                # write image and label
                img_dst = dst / "images" / split / f.name
                lbl_dst = dst / "labels" / split / (f.stem + ".txt")
                move_or_copy(str(f), str(img_dst), do_copy=do_copy)
                lbl_dst.parent.mkdir(parents=True, exist_ok=True)
                with open(lbl_dst, "w", encoding="utf-8") as fo:
                    fo.write(yolo_line_from_xyxy(x0,y0,x1,y1,W,H,args.class_id))

                # append to labels.csv
                with open(dst/"labels.csv", "a", newline="", encoding="utf-8") as cf:
                    writer = csv.writer(cf)
                    writer.writerow([img_dst.name, split, args.class_id, x0, y0, x1, y1, W, H])

                print(f"✅ saved to {split}: {img_dst} + {lbl_dst}")
                accepted += 1
                i += 1
                break

            elif key in (ord('d'), ord('D')):  # delete original without saving
                remove_or_trash(f, args.trash)
                print(f"🗑️ deleted {f}")
                i += 1
                break

            elif key in (ord('n'), ord('N')):  # skip
                print(f"⏭️ skipped {f}")
                i += 1
                break

            elif key in (ord('r'), ord('R')):  # reset selection
                ui.reset()

            elif key in (27, ord('q'), ord('Q')):  # esc/q
                print("👋 Bye.")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("✅ Done.")

if __name__ == "__main__":
    main()
