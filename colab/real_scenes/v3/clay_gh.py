"""Clay room on a GitHub runner, CPU only (.github/workflows/clay-room.yml, settings in v3/CLAY.json; approved plan 27 Oct, steps
1-4): the run's model.ply + cameras + frames + per-splat object ids -> final/clay_room.py -> room.glb + viewer + comparison with the
splat model -> Blender import + render check -> hf://datasets/<user>/namma-v3-out/clay/<time>/; the viewer is published like the
tours (link only in clay/<time>/tour_url.txt) and screenshotted."""
import json, os, shutil, subprocess, sys, time
V = os.path.dirname(os.path.abspath(__file__)); FIN = os.path.join(os.path.dirname(V), 'final'); sys.path.insert(0, V); sys.path.insert(0, FIN)
cfg = json.load(open(f'{V}/CLAY.json')); OUT = os.path.abspath('out'); T0 = time.time(); STAMP = time.strftime('%Y%m%d-%H%M')
run = lambda c: subprocess.run(c, shell=True, check=True, executable='/bin/bash')

def say(*a): print(f'[{(time.time() - T0) / 60:5.1f} min]', *a, flush=True)

if not cfg.get('n'): say('CLAY.json n = 0: nothing to do'); sys.exit(0)
tok = os.environ.get('HF_TOKEN'); assert tok, 'needs HF_TOKEN'
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=tok); user = api.whoami()['name']; repo = f'{user}/namma-v3-out'; R, OB = cfg['run'], cfg.get('objects', '')
pats = [f'{R}/full/model.ply', f'{R}/full/level.json', f'{R}/sparse/**'] + ([f'{OB}/splat_object.npy', f'{OB}/objects.json'] if OB else [])
d = snapshot_download(repo, repo_type='dataset', local_dir='dl', token=tok, allow_patterns=pats)
T = os.path.abspath('work'); EX = f'{T}/gsplat/examples'; data = f'{T}/data'; os.makedirs(data, exist_ok=True); os.makedirs(OUT, exist_ok=True)
run(f'git clone -q --depth 1 -b v1.5.3 https://github.com/nerfstudio-project/gsplat {T}/gsplat && touch {EX}/datasets/__init__.py')
run('pip install -q torch --index-url https://download.pytorch.org/whl/cpu && pip install -q opencv-python-headless imageio tqdm plyfile scipy trimesh '
    'open3d pillow typing_extensions "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" && sed -i '
    '"s/np.uint64(-1)/np.uint64(np.iinfo(np.uint64).max)/" $(python -c "import importlib.util as u; print(u.find_spec(\'pycolmap\').submodule_search_locations[0])")/scene_manager.py')
shutil.copytree(f'{d}/{R}/sparse', f'{data}/sparse', dirs_exist_ok=True)
import dl3dv, frames                                                                      # the run's frames, same names and order
_, folder = dl3dv.fetch(cfg['scene'], f'{T}/ds', tok, cfg.get('dl3dv_res', 'images_2'))
say('frames', frames.from_images(folder, f'{data}/images', int(cfg.get('nframes', 420)), int(cfg.get('long', 1440))))
sid = f'{d}/{OB}/splat_object.npy' if OB else '-'; obj = f'{d}/{OB}/objects.json' if OB else '-'
r = subprocess.run(f'cd {EX} && PYTHONPATH={EX}:{FIN}:$PYTHONPATH CLAY_TEX={cfg.get("texel", 0.006)} python {FIN}/clay_room.py {data} {d}/{R} {sid} {obj} {OUT} '
                   f'{cfg.get("views", 12)} 2>&1 | tee {OUT}/clay.log', shell=True, executable='/bin/bash').returncode
if os.path.exists(f'{OUT}/room.glb'):                                                     # does it open in Blender?
    b = subprocess.run(f'blender -b --python-exit-code 1 -P {OUT}/blender_check.py -- {OUT}/room.glb {OUT}/room.json {OUT}/blender_render.png 2>&1 | '
                       f'grep -E "BLENDER|Error|error" | tail -20 | tee {OUT}/blender.log', shell=True, executable='/bin/bash')
    say('blender check', 'ok' if os.path.exists(f'{OUT}/blender_render.png') else 'FAILED (see blender.log)')
dst = f'clay/{STAMP}'; api.upload_folder(folder_path=OUT, path_in_repo=dst, repo_id=repo, repo_type='dataset')
say(f'results: hf://datasets/{repo}/{dst}')
if not os.path.exists(f'{OUT}/room.glb'): sys.exit('clay_room.py failed')
web = os.path.abspath('web'); os.makedirs(web, exist_ok=True)
for f in ('index.html', 'room.glb', 'room.json'): shutil.copy(f'{OUT}/{f}', web)
import tour_publish
tour_publish.publish(api, user, repo, f'{R}-clay', web, say, url_path=f'{dst}/tour_url.txt')
if os.environ.get('GITHUB_ENV'): open(os.environ['GITHUB_ENV'], 'a').write(f'TOUR_RUN={dst}\n')
say('done')
