# pip install opencv-python pillow
# (optional) pip install pytesseract  and install Tesseract binary for OCR

import os
import numpy as np
import cv2
from PIL import Image
import re

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

    # suppress flat/dark bands
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

def bottom_text_band_mask(gray):
    """
    Heuristic: detect bottom caption/controls — thin horizontal strokes on dark bg.
    Returns a 0/1 mask (uint8) same size as gray.
    """
    import numpy as np, cv2
    H, W = gray.shape
    g = cv2.GaussianBlur(gray, (0,0), 1.2)
    edges = cv2.Canny(g, 60, 120)

    # emphasize horizontal text strokes
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9,1))
    horiz = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)

    # focus on bottom third where IG puts captions/controls
    mask = np.zeros_like(horiz)
    mask[int(H*0.60):] = horiz[int(H*0.60):]
    # thicken a little so overlap test is stable
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), 1)
    return (mask > 0).astype(np.uint8)


def box_from_row_energy(imap, min_h_frac=0.35, gap=8):
    """
    Use row-wise energy to pick the main content band (top..bottom),
    ignoring short low-energy dips (gap). Returns (x,y,w,h).
    """
    import numpy as np, cv2
    H, W = imap.shape
    row_e = imap.mean(axis=1)                   # energy per row
    # smooth a bit
    row_e = cv2.GaussianBlur(row_e.astype(np.float32), (0,0), 3)

    # threshold relative to the 70th percentile (robust on dark UIs)
    thr = np.percentile(row_e, 70)
    on = (row_e >= thr).astype(np.uint8)

    # close small gaps so a short caption line doesn't split the band
    kernel = np.ones(gap, np.uint8)
    on = cv2.morphologyEx(on[None, :], cv2.MORPH_CLOSE, kernel)[0]

    # pick longest run of "on"
    best = (0, 0)  # (start, end)
    i = 0
    while i < H:
        if on[i]:
            j = i
            while j < H and on[j]: j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1

    y1, y2 = best
    if y2 - y1 < int(H * min_h_frac):
        # fallback: keep most of the center, this shouldn't happen often
        y1, y2 = int(H*0.12), int(H*0.88)

    # left/right: keep almost all (let scorer/AR handle vertical bars)
    return (int(W*0.04), int(y1), int(W*0.92), int(y2 - y1))

def box_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0,0,mask.shape[1],mask.shape[0])
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)

def score_candidate(box, imap, btm_mask, debug=False):
    """
    Score = interest density * (center + aspect) * fullness * variance * area bonus
    + hard guards against skinny strips and weak center coverage.
    """
    import numpy as np
    score = 0
    x, y, w, h = box
    H, W = imap.shape[:2]

    # --- hard rejections (generic, not app-specific) ---
    # Reject short bars (heights) and skinny vertical slices (width)
    if h < H * 0.28 or w < W * 0.10:   # <- min width rule (50% of image width)
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
        score = -1e6  # discourage heavily, but not absolute reject

  
    # --- features ---
    # input(f"roi.sum(): {roi.sum()}")
    interest_density = float(roi.sum()) / (w * h + 1e-6)

    # fullness: fraction of “interesting” pixels
    filled = float((roi > 32).mean())
    if filled < 0.01: # very flat = likely UI bar
        score = -1e9

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

    # --- bigger boxes get a gentle boost ---
    area_norm = (w * h) / float(W * H)   # 0..1
    area_bonus = (1.5 * area_norm)       # 0.5..1

    # --- final score ---
    base = interest_density * (0.7 + 0.25*centrality + 0.05*ar_penalty)
    score = base * (0.5 + filled) * var_factor * area_bonus

    sq_pref = np.exp(-1.2 * abs(np.log(ar)))
    score *= (0.3 + 0.7 * sq_pref)
    bx, by, bw, bh = box
    band = btm_mask[by:by+bh, bx:bx+bw]
    if band.size:
        overlap = band.mean()           # 0..1 fraction of pixels touching caption strokes
        score *= (1.0 - 0.6 * overlap)  # up to 60% downweight if it sits on the caption
    if w > 2 * h or h > 2 * w: 
        score = score - (abs(score) / 2)  # input(score) return score
    if debug:
        print(f"box: {box}\nscore: {score}\nbase: {base}\nfilled: {filled}\nvar_factor: {var_factor}\narea_bonus: {area_bonus}\nsq_pref: {sq_pref}\noverlap: {overlap}\ncentrality: {centrality}\nar_penalty: {ar_penalty}\n\n")
    return score

STOPWORDS = {
    "follow","sponsored","likes","like","comments","comment",
    "view","views","reply","replies","share","send","save",
    "instagram","post","minutes","minute","mins","min","hours","hour",
    "days","day","ago","•","…", "suggested", "posts", 
}
NUM_RE = re.compile(r"^(?:[\d,\.]+[kKmM]?|\d{1,2}:\d{2}(?:\s?[ap]m)?)$")

def ocr_boxes(pil_img):
    if not TESS_OK:
        return []
    try:
        d = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
        out = []
        for i in range(len(d['text'])):
            txt = (d['text'][i] or "").strip()
            if not txt: 
                continue
            conf = int(d.get('conf', ['0'])[i])
            if conf < 55: 
                continue
            low = txt.lower()

            # hard text filters
            if low in STOPWORDS: 
                continue
            if NUM_RE.match(low): 
                continue
            if low.startswith('@'): 
                continue  # usernames
            if low == "follow": 
                continue
            # print(d['text'][i])
            # input((d['left'][i], d['top'][i], d['width'][i], d['height'][i], low))
            out.append( (d['left'][i], d['top'][i], d['width'][i], d['height'][i], low) )
        return out
    except Exception:
        return []

def ig_ui_exclusion_mask(gray):
    """
    Rough IG layout mask: mark header, footer nav, and caption band as EXCLUDED (1s).
    Return uint8 mask 0/255 where 255 = excluded UI area to ignore.
    """
    H, W = gray.shape
    mask = np.zeros_like(gray, dtype=np.uint8)

    # Header (clock + "Instagram" + story row)
    h_header = int(H * 0.14)   # 12–16% is typical, keep a little slack
    mask[:h_header, :] = 255

    # Footer nav (home/search/plus/reels/profile)
    h_footer = int(H * 0.12)
    mask[H - h_footer:, :] = 255

    # Caption/like bar usually starts above footer; mark a band above footer
    # (this also catches "View all comments", counts, time-ago)
    cap_top = max(0, H - int(H * 0.30))
    cap_bot = H - int(H * 0.12)
    mask[cap_top:cap_bot, :] = np.maximum(mask[cap_top:cap_bot, :], 255)

    # Optional: left strip where avatar/username sits on feed
    mask[:, :int(W * 0.07)] = 255

    # Slight dilation so we don't hug edges
    mask = cv2.dilate(mask, np.ones((7,7), np.uint8), 1)
    return mask

def expand_for_text(best_box, ocr_items, pad, W, H, ui_excl_mask, near_factor=1.1):
    """
    Expand best_box to include OCR boxes that are:
      - mostly outside excluded UI mask
      - reasonably near the best box (limits grabbing header/caption far away)
    ocr_items: list of (x,y,w,h,text_low)
    """
    if not ocr_items: 
        return best_box

    x, y, w, h = best_box
    x1, y1, x2, y2 = x, y, x + w, y + h

    # a slightly grown region around the best box
    grow_w = int(w * (near_factor - 1.0) / 2.0)
    grow_h = int(h * (near_factor - 1.0) / 2.0)
    near_x1 = max(0, x1 - grow_w)
    near_y1 = max(0, y1 - grow_h)
    near_x2 = min(W - 1, x2 + grow_w)
    near_y2 = min(H - 1, y2 + grow_h)

    for (bx, by, bw, bh, _txt) in ocr_items:
        # skip if mostly inside UI excluded zones
        bmask = ui_excl_mask[by:by+bh, bx:bx+bw]
        if bmask.size and (bmask.mean() > 128):   # majority masked
            continue

        # require it to be near the main crop
        if bx + bw < near_x1 or bx > near_x2 or by + bh < near_y1 or by > near_y2:
            continue

        # expand but only OUTWARD (don’t jump into header/footer)
        x1 = min(x1, bx - pad)
        y1 = min(y1, by - pad)
        x2 = max(x2, bx + bw + pad)
        y2 = max(y2, by + bh + pad)

    # clip and also keep away from excluded regions by 2px
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W - 1, x2); y2 = min(H - 1, y2)

    # shave off any overlap with UI mask at the very edges
    # (prevents expansion up into "Follow" or down into likes row)
    while y1 > 0 and ui_excl_mask[y1:y1+2, x1:x2].mean() > 128: y1 += 2
    while y2 < H-1 and ui_excl_mask[y2-2:y2, x1:x2].mean() > 128: y2 -= 2
    while x1 > 0 and ui_excl_mask[y1:y2, x1:x1+2].mean() > 128: x1 += 2
    while x2 < W-1 and ui_excl_mask[y1:y2, x2-2:x2].mean() > 128: x2 -= 2

    x1 = max(0, min(x1, x2-1))
    y1 = max(0, min(y1, y2-1))
    return (x1, y1, x2 - x1, y2 - y1)
# ---------- main pipeline ----------
def auto_crop(img_bgr, target_max=1800, return_box=False, return_conf=False, debug=False):
    # normalize size
    H0, W0 = img_bgr.shape[:2]
    scale = min(1.0, float(target_max) / max(H0, W0))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(W0*scale), int(H0*scale)), cv2.INTER_AREA)

    # strip constant borders
    img_bgr = trim_uniform_borders(img_bgr, tol=2)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    imap = interest_map(gray)
    imap = cv2.normalize(imap, None, 0, 255, cv2.NORM_MINMAX)
    imap = imap.astype("uint8")

    # candidates
    masks = candidate_masks(imap)
    boxes = [box_from_mask(m) for m in masks]
    boxes.append((0, 0, imap.shape[1], imap.shape[0]))
    boxes.append(largest_content_box(gray))
    # (if you added row-energy candidate, add it here too)

    # score & pick
    btm_mask = mask_low_variance_bands(gray)
    scored = [(score_candidate(b, imap, btm_mask, debug), b) for b in boxes]

    # keep only "valid" ones for margin calc (drop the huge negative sentinels)
    valid = [(s, b) for (s, b) in scored if s > -1e5]
    if not valid:  # fallback if everything was nuked
        valid = scored

    valid.sort(key=lambda t: t[0], reverse=True)
    best_score, best = valid[0]
    second = valid[1][0] if len(valid) > 1 else (best_score - 1.0)

    # confidence from score margin (squashed to 0..1)
    # larger margin => more confident
    import math
    margin = best_score - second
    conf = 1.0 / (1.0 + math.exp(-margin / (abs(best_score) + 1e-6)))  # sigmoid

    # expand for text around crop
    ui_mask = ig_ui_exclusion_mask(gray)
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    tboxes = ocr_boxes(pil) 
    best = expand_for_text(best, tboxes, pad=60, W=imap.shape[1], H=imap.shape[0],
                       ui_excl_mask=ui_mask, near_factor=1.15)
    # (you can also clip conf if area is tiny or AR extreme)
    x, y, w, h = best
    crop = img_bgr[y:y+h, x:x+w]

    # mild enhancement
    crop = cv2.GaussianBlur(crop, (0,0), 0.5)
    crop = cv2.addWeighted(crop, 1.15, cv2.GaussianBlur(crop, (0,0), 2), -0.15, 0)

    if return_box or return_conf:
        out = []
        if return_box:  out.append((x, y, w, h))
        if return_conf: out.append(float(conf))
        if len(out) == 1: return out[0]  # keep backward-compat
        return tuple(out) + (crop,) if not return_box else ( (x,y,w,h), crop, float(conf) )
    # after computing imap:
    if debug:
        cv2.imwrite("debug/debug_imap.png", imap)

        # dump each candidate box and its score:
        for s, b in scored:
            x,y,w,h = b
            vis = cv2.cvtColor(imap, cv2.COLOR_GRAY2BGR).copy()
            cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(vis, f"{s:.2f}", (x+5,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            if debug:
                cv2.imwrite(f"debug/debug_candidate_{x}_{y}_{w}_{h}_{s:.2f}.png", vis)
    return crop

def process_path(in_path, out_path):
    img = cv2.imread(in_path)
    if img is None:
        print(f"❌ Could not read {in_path}")
        return
    crop = auto_crop(img, debug=debug)
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
    ap.add_argument("--debug", dest="debugp", required=False, default=False, help="print candidate debug boxes")

    args = ap.parse_args()
    debug = args.debugp
    if args.inp is None:
        print("Usage:\n  python autocrop.py --in ./raw --out ./clean")
        raise SystemExit(1)

    p_in = pathlib.Path(args.inp)
    p_out = pathlib.Path(args.outp)

    exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    if p_in.is_dir():
        for f in sorted(p_in.rglob("*")):
            print(f)
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
