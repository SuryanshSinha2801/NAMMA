"""Find whole-home handheld videos in DL3DV (the hotel room's source: 10K phone videos, each walked around a place from all sides).
DL3DV has no room labels, so: every video at least MIN_S seconds long (a home takes minutes) -> 5 frames grabbed straight from the
remote video.mp4 by HTTP seeking (nothing downloaded whole) -> contact sheets of 10 videos each, labelled with batch/hash/length ->
hf://datasets/<user>/namma-v3-out/dl3dv_scout/<time>/ (.github/workflows/dl3dv-scout.yml; settings in v3/SCOUT.json)."""
import csv, io, json, os, subprocess, sys, time, urllib.request
import imageio_ffmpeg
from PIL import Image, ImageDraw
V = os.path.dirname(os.path.abspath(__file__)); cfg = json.load(open(f'{V}/SCOUT.json'))
if not cfg.get('n'): sys.exit(0)
tok = os.environ['HF_TOKEN']; FF = 'ffmpeg' if subprocess.run('command -v ffmpeg', shell=True).returncode == 0 else imageio_ffmpeg.get_ffmpeg_exe(); OUT = 'scout'; os.makedirs(OUT, exist_ok=True)
rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen('https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/cache/DL3DV-valid.csv').read().decode())))
rows = sorted([r for r in rows if r['duration'] and r['batch'] != 'problem' and float(r['duration']) >= cfg.get('min_s', 150)], key=lambda r: -float(r['duration']))
print(len(rows), 'videos', flush=True); W, H = 240, 135; found = []

def signed(url):                                         # the Hub answers with a redirect to a signed CDN URL: seek on that, no auth header
    import requests
    r = requests.get(url, headers={'Authorization': f'Bearer {tok}', 'Range': 'bytes=0-0'}, timeout=60); r.close()
    if len(ERR) < 2: print('url', r.status_code, r.url.split('?')[0][:80], flush=True)
    return r.url                                          # final URL after redirects (requests drops the auth header off-host)

ERR = []
def grab(url, t):
    r = subprocess.run([FF, '-nostdin', '-loglevel', 'warning', '-ss', f'{t:.1f}', '-i', url, '-frames:v', '1',
                        '-vf', f'scale={W}:{H}', '-f', 'image2pipe', '-vcodec', 'png', '-'], capture_output=True, timeout=120)
    if not r.stdout and len(ERR) < 5: ERR.append(f'rc {r.returncode}: ' + r.stderr.decode(errors='ignore')[-400:]); print('ffmpeg:', ERR[-1], flush=True)
    return Image.open(io.BytesIO(r.stdout)).convert('RGB') if r.stdout else Image.new('RGB', (W, H), (60, 0, 0))

sheet, n = None, 0
for i, r in enumerate(rows):
    url = f"https://huggingface.co/datasets/DL3DV/DL3DV-ALL-video/resolve/main/{r['batch']}/{r['hash']}/video.mp4"; d = float(r['duration'])
    try: u = signed(url); ims = [grab(u, d * f) for f in (0.05, 0.27, 0.5, 0.73, 0.95)]
    except Exception as e: print('skip', r['hash'][:10], e, flush=True); continue
    if sheet is None: sheet = Image.new('RGB', (W * 5, (H + 18) * 10), 'black')
    y = n * (H + 18); ImageDraw.Draw(sheet).text((4, y + 2), f"#{i}  {r['batch']}/{r['hash']}  {d:.0f}s", fill='yellow')
    for k, im in enumerate(ims): sheet.paste(im, (k * W, y + 18))
    if i == 2 and len(ERR) >= 5: sys.exit('frames failing: ' + ERR[0])
    found.append(dict(i=i, batch=r['batch'], hash=r['hash'], seconds=round(d))); n += 1
    if n == 10: sheet.save(f'{OUT}/sheet_{i // 10:03d}.jpg', quality=85); sheet, n = None, 0; print('sheet', i // 10, flush=True)
if sheet is not None: sheet.save(f'{OUT}/sheet_{len(rows) // 10:03d}.jpg', quality=85)
json.dump(found, open(f'{OUT}/videos.json', 'w'), indent=0)
from huggingface_hub import HfApi
api = HfApi(token=tok); repo = f"{api.whoami()['name']}/namma-v3-out"; dst = f"dl3dv_scout/{time.strftime('%Y%m%d-%H%M')}"
api.upload_folder(folder_path=OUT, path_in_repo=dst, repo_id=repo, repo_type='dataset'); print('results:', dst, flush=True)
