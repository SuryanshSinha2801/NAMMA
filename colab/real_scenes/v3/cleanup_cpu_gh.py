"""Floater clean-up experiment on a GitHub runner, CPU only (.github/workflows/cleanup-cpu.yml, settings in v3/CLEANUP.json):
a run with two trainings of the same frames (run 5: 'full' .ply + 'prior' .splat) + its cameras/frames -> final/cleanup_cpu.py ->
hf://datasets/<user>/namma-v3-out/cleanup/<time>/ (cleanup_report.json, cleanup_sheet.jpg, drop_*.npy)."""
import json, os, shutil, subprocess, sys, time
V = os.path.dirname(os.path.abspath(__file__)); FIN = os.path.join(os.path.dirname(V), 'final'); sys.path.insert(0, V); sys.path.insert(0, FIN)
cfg = json.load(open(f'{V}/CLEANUP.json')); OUT = os.path.abspath('out'); T0 = time.time(); STAMP = time.strftime('%Y%m%d-%H%M')
run = lambda c: subprocess.run(c, shell=True, check=True, executable='/bin/bash')

def say(*a): print(f'[{(time.time() - T0) / 60:5.1f} min]', *a, flush=True)

if not cfg.get('n'): say('CLEANUP.json n = 0: nothing to do'); sys.exit(0)
tok = os.environ.get('HF_TOKEN'); assert tok, 'needs HF_TOKEN'
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=tok); user = api.whoami()['name']; repo = f'{user}/namma-v3-out'; R, A, B = cfg['run'], cfg['model'], cfg['other']
d = snapshot_download(repo, repo_type='dataset', local_dir='dl', token=tok,
                      allow_patterns=[f'{R}/{A}/model.ply', f'{R}/{B}/model.splat', f'{R}/{B}/level.json', f'{R}/tour/tour.json', f'{R}/sparse/**'])
mpu = json.load(open(f'{d}/{R}/tour/tour.json'))['metres_per_unit']
T = os.path.abspath('work'); EX = f'{T}/gsplat/examples'; data = f'{T}/data'; os.makedirs(data, exist_ok=True)
run(f'git clone -q --depth 1 -b v1.5.3 https://github.com/nerfstudio-project/gsplat {T}/gsplat && touch {EX}/datasets/__init__.py')
run('pip install -q torch --index-url https://download.pytorch.org/whl/cpu && pip install -q opencv-python-headless imageio tqdm plyfile scipy '
    'typing_extensions "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" && sed -i '
    '"s/np.uint64(-1)/np.uint64(np.iinfo(np.uint64).max)/" $(python -c "import importlib.util as u; print(u.find_spec(\'pycolmap\').submodule_search_locations[0])")/scene_manager.py')
shutil.copytree(f'{d}/{R}/sparse', f'{data}/sparse', dirs_exist_ok=True)
import dl3dv, frames                                                                      # the run's frames, same names and order
_, folder = dl3dv.fetch(cfg['scene'], f'{T}/ds', tok, cfg.get('dl3dv_res', 'images_2'))
say('frames', frames.from_images(folder, f'{data}/images', int(cfg.get('nframes', 420)), int(cfg.get('long', 1440))))
os.makedirs(OUT, exist_ok=True)
r = subprocess.run(f'cd {EX} && PYTHONPATH={EX}:{FIN}:$PYTHONPATH python {FIN}/cleanup_cpu.py {data} {d}/{R}/{A}/model.ply {d}/{R}/{B}/model.splat '
                   f'{d}/{R}/{B}/level.json {mpu} {OUT} {cfg.get("px", 480)} {cfg.get("views", 26)} 2>&1 | tee {OUT}/cleanup.log',
                   shell=True, executable='/bin/bash').returncode
dst = f'cleanup/{STAMP}'; api.upload_folder(folder_path=OUT, path_in_repo=dst, repo_id=repo, repo_type='dataset')
say(f'results: hf://datasets/{repo}/{dst}')
if not os.path.exists(f'{OUT}/cleanup_report.json'): sys.exit('cleanup_cpu.py failed')
