"""NAMMA World v0 (Plan B) on a GitHub runner, CPU only (.github/workflows/genworld.yml, settings in v3/GENWORLD.json): for each
sample, one frame of a filmed scene -> final/genworld.py -> a 3D world + viewer, scored against the real neighbouring frames ->
hf://datasets/<user>/namma-v3-out/genworld/<time>/<sample>/; viewers published like the tours (links only in tour_url.txt files)."""
import json, os, shutil, subprocess, sys, time
V = os.path.dirname(os.path.abspath(__file__)); FIN = os.path.join(os.path.dirname(V), 'final'); sys.path.insert(0, V); sys.path.insert(0, FIN)
cfg = json.load(open(f'{V}/GENWORLD.json')); OUT = os.path.abspath('out'); T0 = time.time(); STAMP = time.strftime('%Y%m%d-%H%M')
run = lambda c: subprocess.run(c, shell=True, check=True, executable='/bin/bash')

def say(*a): print(f'[{(time.time() - T0) / 60:5.1f} min]', *a, flush=True)

if not cfg.get('n'): say('GENWORLD.json n = 0: nothing to do'); sys.exit(0)
tok = os.environ.get('HF_TOKEN'); assert tok, 'needs HF_TOKEN'
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=tok); user = api.whoami()['name']; repo = f'{user}/namma-v3-out'; R = cfg['run']
d = snapshot_download(repo, repo_type='dataset', local_dir='dl', token=tok, allow_patterns=[f'{R}/sparse/**'])
T = os.path.abspath('work'); EX = f'{T}/gsplat/examples'; data = f'{T}/data'; os.makedirs(data, exist_ok=True); os.makedirs(OUT, exist_ok=True)
run(f'git clone -q --depth 1 -b v1.5.3 https://github.com/nerfstudio-project/gsplat {T}/gsplat && touch {EX}/datasets/__init__.py')
run('pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -q transformers opencv-python-headless imageio tqdm '
    'plyfile scipy trimesh open3d pillow typing_extensions "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" && '
    'pip install -q --no-deps simple-lama-inpainting fire && sed -i '
    '"s/np.uint64(-1)/np.uint64(np.iinfo(np.uint64).max)/" $(python -c "import importlib.util as u; print(u.find_spec(\'pycolmap\').submodule_search_locations[0])")/scene_manager.py')
shutil.copytree(f'{d}/{R}/sparse', f'{data}/sparse', dirs_exist_ok=True)
import dl3dv, frames
_, folder = dl3dv.fetch(cfg['scene'], f'{T}/ds', tok, cfg.get('dl3dv_res', 'images_2'))
say('frames', frames.from_images(folder, f'{data}/images', int(cfg.get('nframes', 420)), int(cfg.get('long', 1440))))
import tour_publish
dst = f'genworld/{STAMP}'; reps = {}
for k, smp in enumerate(cfg['samples']):
    o = f'{OUT}/{smp["name"]}'; t = time.time()
    r = subprocess.run(f'cd {EX} && PYTHONPATH={EX}:{FIN}:$PYTHONPATH GW_LONG={cfg.get("long_px", 768)} python {FIN}/genworld.py {o} {cfg["depth_model"]} {data} '
                       f'{smp["frame"]} {smp["offsets"]} 2>&1 | grep -v "it/s" | tee {o}.log', shell=True, executable='/bin/bash').returncode
    say(smp['name'], 'done in', round((time.time() - t) / 60, 2), 'min (incl. model downloads on the first)')
    if os.path.exists(f'{o}/report.json'):
        reps[smp['name']] = json.load(open(f'{o}/report.json'))
        web = os.path.abspath(f'web_{k}'); os.makedirs(web, exist_ok=True)
        for f in ('index.html', 'world.glb', 'world.json'): shutil.copy(f'{o}/{f}', web)
        tour_publish.publish(api, user, repo, f'{R}-world', web, say, url_path=f'{dst}/{"tour_url.txt" if k == 0 else f"tour{k + 1}_url.txt"}')
json.dump(reps, open(f'{OUT}/summary.json', 'w'), indent=1)
api.upload_folder(folder_path=OUT, path_in_repo=dst, repo_id=repo, repo_type='dataset'); say(f'results: hf://datasets/{repo}/{dst}')
if not reps: sys.exit('genworld.py failed')
if os.environ.get('GITHUB_ENV'): open(os.environ['GITHUB_ENV'], 'a').write(f'TOUR_RUN={dst}\n')
