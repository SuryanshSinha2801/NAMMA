"""Frames for v3: from a video (sharpest frame per time window) or from a folder of already extracted frames (DL3DV, photos).
Every output frame is resized to `long_side` px on its long side and named 0000.jpg, 0001.jpg, ... in walk order.
Usage: python frames.py <video file | image folder> <out dir> [n=300] [long_side=1200]"""
import glob, os, sys
import cv2

def from_video(video, out, n=300, long_side=1200):
    """Sharpest (variance of Laplacian) frame in each of n equal time windows: motion blur is the main enemy of phone video."""
    os.makedirs(out, exist_ok=True); cap = cv2.VideoCapture(video); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    win = max(1, total // n); best, bs, k, i = None, -1, 0, 0
    while k < n:
        if not cap.grab(): break
        ok, f = cap.retrieve()
        if ok:
            g = cv2.cvtColor(cv2.resize(f, (f.shape[1] // 4, f.shape[0] // 4)), cv2.COLOR_BGR2GRAY); sc = cv2.Laplacian(g, cv2.CV_64F).var()
            if sc > bs: best, bs = f, sc
        if (i + 1) % win == 0 and best is not None:
            save(best, f'{out}/{k:04d}.jpg', long_side); k += 1; best, bs = None, -1
        i += 1
    return k

def from_images(folder, out, n=300, long_side=1200):
    """Evenly spaced subset (keeps walk order) of an image folder, e.g. frames a dataset already extracted from a video."""
    os.makedirs(out, exist_ok=True)
    files = sorted(f for e in ('jpg', 'jpeg', 'png', 'JPG', 'PNG') for f in glob.glob(f'{folder}/*.{e}'))
    step = max(1, len(files) / n); pick = [files[int(i * step)] for i in range(min(n, len(files)))]
    for k, f in enumerate(pick): save(cv2.imread(f), f'{out}/{k:04d}.jpg', long_side)
    return len(pick)

def save(img, path, long_side):
    h, w = img.shape[:2]; s = long_side / max(h, w)
    if s < 1: img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

if __name__ == '__main__':
    src, out = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 300; ls = int(sys.argv[4]) if len(sys.argv) > 4 else 1200
    k = from_images(src, out, n, ls) if os.path.isdir(src) else from_video(src, out, n, ls)
    print(f'{k} frames -> {out}', flush=True)
