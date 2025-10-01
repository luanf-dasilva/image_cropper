# pip install opencv-python pillow
# (optional) pip install pytesseract  and install Tesseract binary for OCR

import os, glob
import numpy as np
import cv2
from PIL import Image

# ---- Optional OCR (safe fallback) ----
try:
    import pytesseract
    TESS_OK = True
except Exception:
    TESS_OK = False

# ---------- helpers ----------
def trim_uniform_borders(img, tol=2, max_trim_frac=0.25):
    """Remove near-uniform bands from all four sides (generic app chrome/borders)."""
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    top = 0
    while top < h*max_trim_frac and np.std(g[top, :]) <= tol: top += 1
    bot = h - 1
    while bot > h*(1 - max_trim_frac) and np.std(g[bot, :]) <= tol: bot -= 1
    left = 0
    while left < w*max_trim_frac and np.std(g[:, left]) <= tol: left += 1
    right = w - 1
    while right > w*(1 - max_trim_frac) and np.std(g[:, right]) <= tol: right -= 1
    top = max(0, top - 2); bot = min(h - 1, bot + 2)
    left = max(0, left - 2); right = min(w - 1, right + 2)
    return img[top:bot+1, left:right+1]

def interest_map(gray):
    import numpy as np
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap = np.abs(lap)
    lap = cv2.GaussianBlur(lap, (0, 0), 1.0)

    imap = mag + 0.6 * lap
    imap = cv2.GaussianBlur(imap, (0, 0), 2)

    # NEW: suppress flat/dark bands
    band_mask = mask_low_variance_bands(gray, min_run_frac=0.04, mean_thr=32, std_thr=14)
    imap = (imap * (band_mask.astype(np.float32) / 255.0))

    imap = cv2.normalize(imap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return imap

def candidate_masks(imap):
    """Generate a few blob masks from the interest map."""
    thr = cv2.threshold(imap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return [np.ones_like(thr, dtype=np.uint8)*255]

    # sort by area, keep top K
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    masks = []
    for i in range(len(cnts)):
        m = np.zeros_like(thr)
        cv2.drawContours(m, cnts[:i+1], -1, 255, -1)
        masks.append(m)
    return masks

def box_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0,0,mask.shape[1],mask.shape[0])
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)

def score_candidate(box, imap):
    """
    Score = interest density * (center + aspect) * fullness * variance * area bonus
    + hard guards against skinny strips and weak center coverage.
    """
    import numpy as np
    x, y, w, h = box
    H, W = imap.shape[:2]

    # --- hard rejections (generic, not app-specific) ---
    # Reject short bars (heights) and skinny vertical slices (width)
    if h < H * 0.28 or w < W * 0.10:   # <- NEW min width rule (50% of image width)
        return -1e9

    roi = imap[y:y+h, x:x+w]
    if roi.size == 0:
        return -1e9

    # Require decent overlap with central window (40% window, must cover >=25% of box)
    cx1, cy1, cx2, cy2 = int(W*0.30), int(H*0.30), int(W*0.70), int(H*0.70)
    overlap_w = max(0, min(x+w, cx2) - max(x, cx1))
    overlap_h = max(0, min(y+h, cy2) - max(y, cy1))
    overlap_area = overlap_w * overlap_h
    if overlap_area < 0.25 * (w * h):   # <- stricter center requirement
        return -1e6  # discourage heavily, but not absolute reject

    # --- features ---
    interest_density = float(roi.sum()) / (w * h + 1e-6)

    # fullness: fraction of “interesting” pixels
    filled = float((roi > 32).mean())
    if filled < 0.15:          # very flat = likely UI bar
        return -1e9

    # centrality (closer to center is better)
    cx, cy = x + w/2.0, y + h/2.0
    dx = (cx - W/2.0) / (W/2.0)
    dy = (cy - H/2.0) / (H/2.0)
    centrality = 1.0 - min(1.0, np.hypot(dx, dy))

    # mild aspect regularizer
    ar = w / float(h + 1e-6)
    ar_penalty = 1.0 - min(abs(np.log(ar)), 1.0)

    # variance penalty (flat regions lose)
    var_roi = float(np.var(roi))
    var_global = max(1.0, float(np.var(imap)))
    var_factor = 0.5 + 0.5 * min(1.0, var_roi / (0.6 * var_global))  # 0.5..1

    # --- area bonus (NEW): bigger boxes get a gentle boost ---
    area_norm = (w * h) / float(W * H)              # 0..1
    area_bonus = (0.5 + 0.5 * area_norm)            # 0.5..1

    # --- final score ---
    base = interest_density * (0.7 + 0.25*centrality + 0.05*ar_penalty)
    score = base * (0.5 + filled) * var_factor * area_bonus
    return score

def ocr_boxes(pil_img):
    if not TESS_OK:
        return []
    try:
        d = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
        out = []
        for i in range(len(d['text'])):
            if str(d['text'][i]).strip() and int(d.get('conf', ['0'])[i]) > 50:
                out.append((d['left'][i], d['top'][i], d['width'][i], d['height'][i]))
        return out
    except Exception:
        return []

def expand_for_text(box, boxes, pad, W, H):
    if not boxes:
        return box
    x, y, w, h = box
    x1, y1, x2, y2 = x, y, x + w, y + h
    for (bx, by, bw, bh) in boxes:
        bx2, by2 = bx + bw, by + bh
        if not (bx2 < x1 - pad or bx > x2 + pad or by2 < y1 - pad or by > y2 + pad):
            x1 = min(x1, bx - pad)
            y1 = min(y1, by - pad)
            x2 = max(x2, bx2 + pad)
            y2 = max(y2, by2 + pad)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W - 1, x2); y2 = min(H - 1, y2)
    return (x1, y1, x2 - x1, y2 - y1)

# ---------- main pipeline ----------
def auto_crop(img_bgr, keep_text=True, target_max=1800):
    # normalize size
    H0, W0 = img_bgr.shape[:2]
    scale = min(1.0, float(target_max) / max(H0, W0))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(W0*scale), int(H0*scale)), cv2.INTER_AREA)

    # strip constant borders
    img_bgr = trim_uniform_borders(img_bgr, tol=2)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    imap = interest_map(gray)

    # candidates
    masks = candidate_masks(imap)
    boxes = [box_from_mask(m) for m in masks]
    # plus full image fallback
    boxes.append((0, 0, imap.shape[1], imap.shape[0]))

    # NEW: candidate from largest mid-tone region (inner card)
    boxes.append(largest_content_box(gray))
    # score & pick
    scored = [(score_candidate(b, imap), b) for b in boxes]
    scored.sort(reverse=True, key=lambda t: t[0])
    best = scored[0][1]
    # After computing imap:
    cv2.imwrite("debug/debug_imap.png", imap)

    # Dump each candidate box and its score:
    for s, b in scored:
        x,y,w,h = b
        vis = cv2.cvtColor(imap, cv2.COLOR_GRAY2BGR).copy()
        cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(vis, f"{s:.2f}", (x+5,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imwrite(f"debug/debug_candidate_{x}_{y}_{w}_{h}_{s:.2f}.png", vis)
        # expand to include text (optional, safe fallback)
    if keep_text:
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        tboxes = ocr_boxes(pil)
        best = expand_for_text(best, tboxes, pad=24, W=imap.shape[1], H=imap.shape[0])

    x, y, w, h = best
    crop = img_bgr[y:y+h, x:x+w]

    # mild enhancement (generic, not artsy)
    crop = cv2.GaussianBlur(crop, (0,0), 0.5)
    crop = cv2.addWeighted(crop, 1.15, cv2.GaussianBlur(crop, (0,0), 2), -0.15, 0)

    return crop

def process_path(in_path, out_path):
    img = cv2.imread(in_path)
    if img is None:
        print(f"❌ Could not read {in_path}")
        return
    crop = auto_crop(img, keep_text=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, crop)
    print(f"✅ {out_path}")
    
def mask_low_variance_bands(gray, min_run_frac=0.04, mean_thr=32, std_thr=14):
    """
    Returns a mask (uint8 0/255) that keeps 'real content' and suppresses
    long runs of near-black, low-variance rows/cols (generic app bars).
    """
    import numpy as np
    H, W = gray.shape
    min_run = max(1, int(H * min_run_frac))

    row_mean = gray.mean(axis=1)
    row_std  = gray.std(axis=1)
    bad_rows = (row_mean < mean_thr) & (row_std < std_thr)

    # keep only long runs as 'bad'
    mask_rows = np.ones(H, dtype=np.uint8) * 255
    start = None
    for i, br in enumerate(bad_rows):
        if br and start is None: start = i
        if (not br or i == H-1) and start is not None:
            end = i if not br else i+1
            if end - start >= min_run:
                mask_rows[start:end] = 0
            start = None

    # Do the same for columns (helps side gutters)
    col_mean = gray.mean(axis=0)
    col_std  = gray.std(axis=0)
    bad_cols = (col_mean < mean_thr) & (col_std < std_thr)
    mask_cols = np.ones(W, dtype=np.uint8) * 255
    start = None
    min_run_c = max(1, int(W * min_run_frac))
    for j, bc in enumerate(bad_cols):
        if bc and start is None: start = j
        if (not bc or j == W-1) and start is not None:
            end = j if not bc else j+1
            if end - start >= min_run_c:
                mask_cols[start:end] = 0
            start = None

    mask = (mask_rows[:, None] & mask_cols[None, :]).astype(np.uint8)
    return mask

def largest_content_box(gray, low_thr=24, high_thr=245):
    """
    Find the largest mid-tone region (excludes near-black UI and pure white bands).
    Returns (x,y,w,h). Generic, not app-specific.
    """
    import numpy as np
    H, W = gray.shape

    # Keep pixels that are not super dark (UI bars) and not pure white
    mid = (gray > low_thr) & (gray < high_thr)
    mask = (mid.astype(np.uint8) * 255)

    # Smooth and close gaps
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8), iterations=1)

    # Largest connected component
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return (0,0,W,H)
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)

  
    if h < H*0.28 or w < W*0.28:
        return (0,0,W,H)   # Reject thin bars
    if w < W * 0.5:        # Reject if narrower than half the image
        return -1e9
    return (x,y,w,h)

if __name__ == "__main__":
    import argparse, pathlib
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=False, default=None, help="input file or folder")
    ap.add_argument("--out", dest="outp", required=False, default="./out", help="output file or folder")
    args = ap.parse_args()

    if args.inp is None:
        print("Usage:\n  python autocrop.py --in ./raw --out ./clean")
        raise SystemExit(1)

    p_in = pathlib.Path(args.inp)
    p_out = pathlib.Path(args.outp)

    exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    if p_in.is_dir():
        for f in sorted(p_in.rglob("*")):
            if f.suffix.lower() in exts:
                rel = f.relative_to(p_in)
                out_f = p_out / rel
                process_path(str(f), str(out_f))
    else:
        if p_out.is_dir():
            out_f = p_out / pathlib.Path(args.inp).name
        else:
            out_f = p_out
        process_path(str(p_in), str(out_f))
