"""Object layer v3 on a GitHub runner, no GPU (.github/workflows/objects-cpu.yml, settings in v3/OBJECTS_CPU.json).
A finished object run (objects/<time>: splat_object.npy + objects.json) + the run's model.ply / sparse / frames -> final/objects_refine.py
(walls out, 3D smoothing, split, merge on look only, names with size) -> hf://datasets/<user>/namma-v3-out/objects/<time>/ ->
two tours like v3/objects_gh.py: real colours with labels / click / search (tour_url.txt) and tinted (tour_tint_url.txt)."""
import json, os, shutil, subprocess, sys, time
V = os.path.dirname(os.path.abspath(__file__)); FIN = os.path.join(os.path.dirname(V), 'final'); sys.path.insert(0, V); sys.path.insert(0, FIN)
cfg = json.load(open(f'{V}/OBJECTS_CPU.json')); OUT = os.path.abspath('out'); T0 = time.time(); STAMP = time.strftime('%Y%m%d-%H%M')
run = lambda c: subprocess.run(c, shell=True, check=True, executable='/bin/bash')

def say(*a): print(f'[{(time.time() - T0) / 60:5.1f} min]', *a, flush=True)

if not cfg.get('n'): say('OBJECTS_CPU.json n = 0: nothing to do'); sys.exit(0)
tok = os.environ.get('HF_TOKEN'); assert tok, 'needs HF_TOKEN'
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=tok); user = api.whoami()['name']; repo = f'{user}/namma-v3-out'; R, SRC = cfg['run'], cfg['from']
d = snapshot_download(repo, repo_type='dataset', local_dir='dl', token=tok,
                      allow_patterns=[f'{R}/full/model.ply', f'{R}/full/level.json', f'{R}/full/nav.json', f'{R}/tour/tour.json', f'{R}/sparse/**',
                                      f'{R}/mesh/mesh.glb', f'{SRC}/splat_object.npy', f'{SRC}/objects.json'])
mpu = json.load(open(f'{d}/{R}/tour/tour.json'))['metres_per_unit']
T = os.path.abspath('work'); EX = f'{T}/gsplat/examples'; data = f'{T}/data'; os.makedirs(data, exist_ok=True)
run(f'git clone -q --depth 1 -b v1.5.3 https://github.com/nerfstudio-project/gsplat {T}/gsplat && touch {EX}/datasets/__init__.py')
run('pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -q transformers accelerate '
    'opencv-python-headless imageio tqdm plyfile scipy trimesh open3d fast-simplification typing_extensions '
    '"git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" && sed -i '
    '"s/np.uint64(-1)/np.uint64(np.iinfo(np.uint64).max)/" $(python -c "import importlib.util as u; print(u.find_spec(\'pycolmap\').submodule_search_locations[0])")/scene_manager.py')
run(f'cd {T} && npm init -y >/dev/null && npm i -s --no-audit --no-fund playcanvas@2.23.2 @playcanvas/supersplat-viewer@1.37.0 @playcanvas/splat-transform@3.10.1 | tail -1')
shutil.copytree(f'{d}/{R}/sparse', f'{data}/sparse', dirs_exist_ok=True)
import dl3dv, frames                                                                      # the run's frames, same names and order
_, folder = dl3dv.fetch(cfg['scene'], f'{T}/ds', tok, cfg.get('dl3dv_res', 'images_2'))
say('frames', frames.from_images(folder, f'{data}/images', int(cfg.get('nframes', 420)), int(cfg.get('long', 1440))))
json.dump(dict(matrix=json.load(open(f'{d}/{R}/full/level.json'))['matrix']), open(f'{T}/level.json', 'w'))
os.makedirs(OUT, exist_ok=True)
r = subprocess.run(f'cd {EX} && PYTHONPATH={EX}:{FIN}:$PYTHONPATH python {FIN}/objects_refine.py {data} {d}/{R}/full/model.ply '
                   f'{d}/{SRC}/splat_object.npy {d}/{SRC}/objects.json {T}/level.json {mpu} {OUT} {int(cfg.get("frames", 40))} 2>&1 | tee {OUT}/objects_refine.log',
                   shell=True, executable='/bin/bash').returncode
dst = f'objects/{STAMP}'
api.upload_folder(folder_path=OUT, path_in_repo=dst, repo_id=repo, repo_type='dataset', ignore_patterns=['full/*'])
say(f'results: hf://datasets/{repo}/{dst}')
if not os.path.exists(f'{OUT}/objects.json') or not os.path.exists(f'{OUT}/full/model.ply'): sys.exit('objects_refine.py failed: no tour')

HR, HT = f'{OUT}/run', f'{OUT}/run_tint'                                                  # real colours + labels; tinted
for h in (HR, HT):
    os.makedirs(f'{h}/full', exist_ok=True); shutil.copy(f'{OUT}/objects.json', h); shutil.copytree(f'{OUT}/crops', f'{h}/crops', dirs_exist_ok=True)
    for x in ('level.json', 'nav.json'): shutil.copy(f'{d}/{R}/full/{x}', f'{h}/full/{x}')
    if os.path.exists(f'{d}/{R}/mesh/mesh.glb'): os.makedirs(f'{h}/mesh', exist_ok=True); shutil.copy(f'{d}/{R}/mesh/mesh.glb', f'{h}/mesh/mesh.glb')
shutil.copy(f'{d}/{R}/full/model.ply', f'{HR}/full/model.ply'); shutil.move(f'{OUT}/full/model.ply', f'{HT}/full/model.ply')
run(f'python {FIN}/make_tour.py plan {EX} {data} {HR} {OUT}/tour'); shutil.copytree(f'{OUT}/tour', f'{OUT}/tour_tint')
import tour_publish
for h, tour, name, url in ((HR, 'tour', f'{R}-objects', 'tour_url.txt'), (HT, 'tour_tint', f'{R}-objtint', 'tour_tint_url.txt')):
    run(f'python {FIN}/make_tour.py build {h} {OUT}/{tour} {T}/node_modules/@playcanvas/supersplat-viewer/public {T}/node_modules/.bin/splat-transform')
    api.upload_folder(folder_path=f'{OUT}/{tour}', path_in_repo=f'{dst}/{tour}', repo_id=repo, repo_type='dataset')
    tour_publish.publish(api, user, repo, name, f'{OUT}/{tour}', say, url_path=f'{dst}/{url}')
if os.environ.get('GITHUB_ENV'): open(os.environ['GITHUB_ENV'], 'a').write(f'TOUR_RUN={dst}\n')
say('done')
