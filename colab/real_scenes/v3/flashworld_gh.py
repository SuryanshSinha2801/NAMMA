"""Photo -> 3D splat world with FlashWorld (ICLR 2026, imlixinyang/FlashWorld, CC BY-NC-SA: non-commercial, fine for us) on its free
Hugging Face ZeroGPU Space: no model of ours, only the connection (owner, 28 Oct: "connect open modules, best result now").
FlashWorld (Wan2.2-5B distilled) generates 24 views along a camera path and the 3D Gaussians of the scene in one pass.
Per photo and per camera path: request -> .spz (Gaussian splats) -> a Spark viewer page (Spark = the open renderer Marble
uses) published like the tours (link only in the private results dataset) -> screenshots. Settings: v3/FLASHWORLD.json.
GitHub Actions, CPU only (.github/workflows/flashworld.yml)."""
import base64, glob, inspect, io, json, os, shutil, sys, time
import numpy as np
from PIL import Image
V = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, V)
cfg = json.load(open(f'{V}/FLASHWORLD.json')); T0 = time.time(); STAMP = time.strftime('%Y%m%d-%H%M'); OUT = os.path.abspath('fw')


def say(*a): print(f'[{(time.time() - T0) / 60:5.1f} min]', *a, flush=True)


W, H = 704, 480                                                                               # FlashWorld's resolution (24 x 480 x 704)


def quat_y(deg):
    """three.js camera (looks along -z, y up) turned by deg about +y (positive = to the left): [w, x, y, z]."""
    a = np.radians(deg) / 2; return [float(np.cos(a)), 0.0, float(np.sin(a)), 0.0]


def path(kind, n=24):
    """Camera paths in the three.js frame; the photo is frame `index`. turn: stand still and look 100 degrees to each side;
    walk: walk 1.2 units into the room, looking a little left and right; orbit: half-circle around a point 1.5 units ahead."""
    cams, index = [], 0
    if kind == 'turn':
        yaws = np.linspace(100, -100, n); index = int(np.argmin(np.abs(yaws)))
        cams = [dict(position=[0, 0, 0], quaternion=quat_y(y)) for y in yaws]
    elif kind == 'walk':
        for t in np.linspace(0, 1, n): cams.append(dict(position=[0, 0, -1.2 * float(t)], quaternion=quat_y(25 * np.sin(2 * np.pi * t))))
    elif kind == 'orbit':
        c = np.array([0, 0, -1.5])
        for a in np.linspace(-40, 40, n):
            r = np.radians(a); p = c + 1.5 * np.array([np.sin(r), 0, np.cos(r)]); cams.append(dict(position=p.tolist(), quaternion=quat_y(a)))
        index = n // 2
    else: raise ValueError(kind)
    if index:                                                                                 # the photo's camera becomes the origin
        q0 = np.array(cams[index]['quaternion']); p0 = np.array(cams[index]['position'], float); a0 = 2 * np.degrees(np.arctan2(q0[2], q0[0]))
        R0 = np.array([[np.cos(np.radians(a0)), 0, np.sin(np.radians(a0))], [0, 1, 0], [-np.sin(np.radians(a0)), 0, np.cos(np.radians(a0))]])
        for c_ in cams:
            a = 2 * np.degrees(np.arctan2(c_['quaternion'][2], c_['quaternion'][0])); c_['quaternion'] = quat_y(a - a0)
            c_['position'] = (R0.T @ (np.array(c_['position'], float) - p0)).tolist()
    return cams, index


VIEWER = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NAMMA World</title><style>html,body{margin:0;height:100%;background:#111;overflow:hidden;font:13px system-ui,sans-serif;color:#eee}
#help{position:fixed;left:12px;bottom:12px;background:#000a;padding:6px 10px;border-radius:8px;max-width:calc(100vw - 48px)}
#load{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;font-size:16px}</style>
<script type="importmap">{"imports":{"three":"https://cdnjs.cloudflare.com/ajax/libs/three.js/0.178.0/three.module.js","@sparkjsdev/spark":"https://sparkjs.dev/releases/spark/0.1.9/spark.module.js"}}</script>
</head><body><div id="load">Loading the world...</div>
<div id="help">Drag to look around, W A S D to move, R / F up and down. Made by AI (FlashWorld) from one photo.</div>
<script type="module">
import * as THREE from 'three'; import {SplatMesh, SparkControls} from '@sparkjsdev/spark';
const info = await (await fetch('./world.json')).json();
const scene = new THREE.Scene(); const camera = new THREE.PerspectiveCamera(info.fov_y, innerWidth / innerHeight, 0.01, 1000);
const renderer = new THREE.WebGLRenderer({antialias: false}); renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.setSize(innerWidth, innerHeight); document.body.appendChild(renderer.domElement);
if (info.start) { camera.position.fromArray(info.start.position); const q = info.start.quaternion; camera.quaternion.set(q[1], q[2], q[3], q[0]); }  // the photo's camera
const world = new SplatMesh({url: './' + (info.file || 'world.spz'), onLoad: () => document.getElementById('load')?.remove()}); scene.add(world);
const controls = new SparkControls({canvas: renderer.domElement});
addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
renderer.setAnimationLoop(() => { controls.update(camera); renderer.render(scene, camera); });
setTimeout(() => document.getElementById('load')?.remove(), 20000);
</script></body></html>"""

if __name__ == '__main__':
    if not cfg.get('n'): say('FLASHWORLD.json n = 0: nothing to do'); sys.exit(0)
    tok = os.environ.get('HF_TOKEN'); assert tok, 'needs the HF_TOKEN secret'
    from huggingface_hub import HfApi, hf_hub_download
    from gradio_client import Client
    api = HfApi(token=tok); user = api.whoami()['name']; repo = f'{user}/namma-v3-out'; dst = f'flashworld/{STAMP}'
    import dl3dv
    src = f"{cfg['scene']}/nerfstudio/{cfg.get('res', 'images_2')}"
    frames = sorted(x.path for x in api.list_repo_tree(dl3dv.REPO, path_in_repo=src, repo_type='dataset') if x.path.endswith(('.png', '.jpg')))
    kw = 'hf_token' if 'hf_token' in inspect.signature(Client.__init__).parameters else 'token'
    client = None
    for sp in cfg.get('spaces', ['imlixinyang/FlashWorld-Demo', 'jbilcke-hf/FlashWorld-ZeroGPU']):
        try: client = Client(sp, **{kw: tok}); say('Space', sp, 'connected'); break
        except Exception as e: say('Space', sp, 'not usable:', repr(e)[:200])
    assert client is not None, 'no FlashWorld Space reachable'
    fy = 0.5 * H / np.tan(np.radians(cfg.get('fov_y', 50)) / 2); rep = {}; done = []
    for smp in cfg['samples']:
        img = Image.open(hf_hub_download(dl3dv.REPO, frames[int(smp['frame'])], repo_type='dataset', local_dir='ds', token=tok)).convert('RGB')
        s = max(H / img.height, W / img.width); nw, nh = round(W / s), round(H / s); x0, y0 = (img.width - nw) // 2, (img.height - nh) // 2
        img = img.crop((x0, y0, x0 + nw, y0 + nh)).resize((W, H), Image.LANCZOS); b = io.BytesIO(); img.save(b, 'PNG')
        uri = 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()
        for kind in smp.get('paths', cfg.get('paths', ['turn', 'walk'])):
            name = f"{smp['name']}_{kind}"; o = f'{OUT}/{name}'; os.makedirs(o, exist_ok=True); img.save(f'{o}/input.jpg', quality=92)
            cams, index = path(kind)
            req = dict(image_prompt=uri, text_prompt=smp.get('prompt', cfg.get('prompt', '')), image_index=index, resolution=[24, H, W],
                       cameras=[dict(position=c['position'], quaternion=c['quaternion'], fx=fy, fy=fy, cx=W / 2, cy=H / 2) for c in cams])
            t = time.time(); r = None
            for attempt in range(3):                                                         # ZeroGPU queues: retry a little
                try:
                    r = json.loads(client.predict(json.dumps(req), api_name='/gradio_generate'))
                    if r.get('success'): break
                    say(name, 'attempt', attempt + 1, 'failed:', str(r)[:300])
                except Exception as e: say(name, 'attempt', attempt + 1, 'error:', repr(e)[:300]); r = dict(error=repr(e)[:300])
                time.sleep(20)
            rep[name] = dict(frame=frames[int(smp['frame'])].split('/')[-1], path=kind, seconds=round(time.time() - t, 1), result={k: v for k, v in (r or {}).items() if k != 'file_path'})
            if not r or not r.get('success'): continue
            f = client.predict(r['file_id'], api_name='/download_file'); f = f if isinstance(f, str) else (f.get('path') if isinstance(f, dict) else f[0])
            shutil.copy(f, f'{o}/world.spz'); rep[name]['spz_mb'] = round(os.path.getsize(f'{o}/world.spz') / 1e6, 2)
            json.dump(dict(fov_y=float(cfg.get('fov_y', 50)), path=kind, cameras=cams, image_index=index), open(f'{o}/world.json', 'w'))
            open(f'{o}/index.html', 'w').write(VIEWER); json.dump(req | dict(image_prompt='(input.jpg)'), open(f'{o}/request.json', 'w'))
            say(name, rep[name]); done.append(name)
            try: client.predict(r['file_id'], api_name='/gradio_delete')
            except Exception: pass
    os.makedirs(OUT, exist_ok=True); json.dump(rep, open(f'{OUT}/report.json', 'w'), indent=1)
    api.upload_folder(folder_path=OUT, path_in_repo=dst, repo_id=repo, repo_type='dataset'); say(f'results: hf://datasets/{repo}/{dst}')
    import tour_publish
    for k, name in enumerate(done):
        web = os.path.abspath(f'web_{k}'); os.makedirs(web, exist_ok=True)
        for x in ('index.html', 'world.spz', 'world.json'): shutil.copy(f'{OUT}/{name}/{x}', web)
        tour_publish.publish(api, user, repo, f'{cfg.get("run", "world")}-fw', web, say, url_path=f'{dst}/{"tour_url.txt" if k == 0 else f"tour{k + 1}_url.txt"}')
    if not done: sys.exit('FlashWorld produced nothing (see report.json)')
    if os.environ.get('GITHUB_ENV'): open(os.environ['GITHUB_ENV'], 'a').write(f'TOUR_RUN={dst}\n')
    say('done', len(done), 'worlds')
