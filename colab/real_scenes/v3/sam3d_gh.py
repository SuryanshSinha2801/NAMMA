"""Object-by-object 3D demo (GitHub Actions, CPU; .github/workflows/sam3d-demo.yml, settings in v3/SAM3D.json).
The 'Video -> frames -> identify objects -> 3D objects' half of the Blender-style pipeline on the hotel-room scene (DL3DV, run 4):
a CPU detector (torchvision Faster R-CNN, BSD) finds the furniture in the frames and picks, per object, the frame where it is biggest and not
cut off; each pick goes to SAM 3 (segment by name) + SAM 3D Objects (Meta, SAM License, full 3D shape incl. unseen sides) on a free
Hugging Face ZeroGPU Space; the GLBs and the photos go into a 3D viewer page (model-viewer), published like a tour (random
folder, link only in the private results dataset) + screenshots of it in hf://datasets/<user>/namma-v3-out/sam3d/<time>/."""
import glob, json, os, shutil, sys, time
V = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, V)
cfg = json.load(open(f'{V}/SAM3D.json')); T0 = time.time(); STAMP = time.strftime('%Y%m%d-%H%M'); OUT = os.path.abspath('sam3d')

def say(*a): print(f'[{(time.time() - T0) / 60:5.1f} min]', *a, flush=True)

if not cfg.get('n'): say('SAM3D.json n = 0: nothing to do'); sys.exit(0)
tok = os.environ.get('HF_TOKEN'); assert tok, 'needs the HF_TOKEN secret'
from huggingface_hub import HfApi
import numpy as np, torch
from PIL import Image
api = HfApi(token=tok); user = api.whoami()['name']; repo = f'{user}/namma-v3-out'; dst = f'sam3d/{STAMP}'
from huggingface_hub import hf_hub_download
import dl3dv                                                                      # only every k-th frame of the scene, not all of it
src = f"{cfg['scene']}/nerfstudio/{cfg.get('res', 'images_2')}"
frames = sorted(x.path for x in api.list_repo_tree(dl3dv.REPO, path_in_repo=src, repo_type='dataset') if x.path.endswith(('.png', '.jpg')))
cand = [hf_hub_download(dl3dv.REPO, f, repo_type='dataset', local_dir='ds', token=tok) for f in frames[::max(1, len(frames) // int(cfg.get('sample', 60)))]]
say(len(frames), 'frames in the scene,', len(cand), 'looked at')

# 1. identify objects: torchvision Faster R-CNN v2 (COCO, BSD) on CPU, best frame per object = biggest box not touching the left/right edge
import torchvision
wts = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT; labels = wts.meta['categories']
det = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=wts).eval(); prep = wts.transforms()
want = cfg['objects']; best = {}
for f in cand:
    im = Image.open(f).convert('RGB'); im.thumbnail((800, 800)); W, H = im.size
    with torch.no_grad(): r = det([prep(torchvision.transforms.functional.pil_to_tensor(im))])[0]
    for s, l, b in zip(r['scores'].tolist(), r['labels'].tolist(), r['boxes'].tolist()):
        name = labels[l]
        if name not in want or s < 0.8: continue
        x0, y0, x1, y1 = b; area = (x1 - x0) * (y1 - y0) / (W * H); whole = x0 > 0.02 * W and x1 < 0.98 * W
        sc = area * (1 if whole else 0.3) * s
        if sc > best.get(name, (0,))[0]: best[name] = (sc, f, [round(v / W, 3) for v in b], round(area, 3), whole)
say('picks:', json.dumps({k: (os.path.basename(v[1]), v[3], v[4]) for k, v in best.items()}))
order = [k for k in want if k in best][:int(cfg.get('max_objects', 4))]

# 2. 3D objects: SAM 3 + SAM 3D Objects on the free ZeroGPU Space (anasrazy's daily GPU quota)
from gradio_client import Client, handle_file
import inspect
kw = 'hf_token' if 'hf_token' in inspect.signature(Client.__init__).parameters else 'token'
cl = Client(cfg['space'], **{kw: tok}); os.makedirs(f'{OUT}/web', exist_ok=True); objs = []
for k, name in enumerate(order):
    _, f, box, area, whole = best[name]; im = Image.open(f).convert('RGB'); im.thumbnail((int(cfg.get('long', 1280)),) * 2)
    im.save(f'{OUT}/web/photo{k}.jpg', quality=92); t1 = time.time()
    try: glb, prev, status = cl.predict(handle_file(f'{OUT}/web/photo{k}.jpg'), want[name], api_name='/reconstruct_objects')
    except Exception as e: say(name, 'failed:', str(e)[:300]); objs.append(dict(name=want[name], error=str(e)[:300])); continue
    say(name, status, f'{time.time() - t1:.0f} s')
    rec = dict(name=want[name], frame=os.path.basename(f), box=box, area=area, status=status)
    if prev: Image.open(prev).convert('RGB').save(f'{OUT}/web/mask{k}.jpg', quality=90); rec['mask'] = f'mask{k}.jpg'
    if glb and os.path.exists(glb):
        shutil.copy(glb, f'{OUT}/web/object{k}{os.path.splitext(glb)[1]}'); rec['model'] = f'object{k}{os.path.splitext(glb)[1]}'
        rec['bytes'] = os.path.getsize(glb)
    objs.append(rec)
json.dump(dict(picks={k: (os.path.basename(v[1]), v[2], v[3], v[4]) for k, v in best.items()}, objects=objs), open(f'{OUT}/sam3d.json', 'w'), indent=1)
api.upload_folder(folder_path=OUT, path_in_repo=dst, repo_id=repo, repo_type='dataset'); say(f'results: hf://datasets/{repo}/{dst}')
done = [o for o in objs if o.get('model', '').endswith('.glb')]
if not done: sys.exit('no 3D object came back (see the statuses above)')

# 3. viewer page: each object's photo (what was found, green) next to its 3D model, drag to turn
cards = ''.join(f'<section><h2>{o["name"]}</h2><div class="row"><img src="{o.get("mask", "")}" alt="photo"><model-viewer src="{o["model"]}" '
                f'camera-controls auto-rotate shadow-intensity="1" exposure="1.1" alt="{o["name"]}"></model-viewer></div>'
                f'<p>{o["status"]} &middot; frame {o["frame"]}</p></section>' for o in done)
open(f'{OUT}/web/index.html', 'w').write(
    '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NAMMA objects</title>'
    '<script type="module" src="https://unpkg.com/@google/model-viewer@3.5.0/dist/model-viewer.min.js"></script>'
    '<style>body{margin:0;padding:16px;background:#141414;color:#ddd;font:15px system-ui,sans-serif}h1{font-size:20px}h2{font-size:17px;margin:18px 0 6px}'
    '.row{display:flex;gap:10px;flex-wrap:wrap}.row img,.row model-viewer{flex:1 1 360px;max-width:100%;height:340px;object-fit:contain;background:#222;border-radius:6px}'
    'p{color:#999;font-size:13px}</style><h1>Hotel room: objects found in the video, rebuilt as full 3D (SAM 3 + SAM 3D Objects)</h1>'
    '<p>Left: the frame, found object in green. Right: its 3D model (drag to turn, scroll to zoom). One photo per object; hidden sides are generated.</p>' + cards)
import tour_publish
tour_publish.publish(api, user, repo, f'{STAMP}-sam3d', f'{OUT}/web', say, url_path=f'{dst}/tour_url.txt')

# 4. look at it like a visitor (screenshots to the private dataset, link never printed)
from playwright.sync_api import sync_playwright
url = open('tour_url.txt').read().strip(); time.sleep(60); os.makedirs('shots', exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--enable-webgl'])
    pg = b.new_page(viewport={'width': 1100, 'height': 900})
    for t in range(6):
        r = pg.goto(url, wait_until='domcontentloaded', timeout=90000)
        if r and r.ok: break
        time.sleep(30)
    pg.wait_for_timeout(40000); pg.screenshot(path='shots/page.jpg', type='jpeg', quality=85, full_page=True)
    for k, el in enumerate(pg.query_selector_all('model-viewer')):
        el.scroll_into_view_if_needed(); pg.wait_for_timeout(8000); el.screenshot(path=f'shots/model{k}_a.jpg', type='jpeg', quality=88)
        bb = el.bounding_box(); pg.mouse.move(bb['x'] + bb['width'] / 2, bb['y'] + bb['height'] / 2); pg.mouse.down()
        pg.mouse.move(bb['x'] + bb['width'] / 2 + 200, bb['y'] + bb['height'] / 2, steps=15); pg.mouse.up(); pg.wait_for_timeout(3000)
        el.screenshot(path=f'shots/model{k}_b.jpg', type='jpeg', quality=88)
api.upload_folder(folder_path='shots', path_in_repo=f'{dst}/shots', repo_id=repo, repo_type='dataset'); say('done')
