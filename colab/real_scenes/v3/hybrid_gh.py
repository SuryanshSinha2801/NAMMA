"""GitHub Actions side of the hybrid 'solid room' demo (.github/workflows/hybrid-demo.yml, settings in v3/HYBRID.json). CPU only.
A finished run's mesh, cameras, level and QA sheet (private HF results) + its frames (rebuilt from DL3DV exactly like the run made
them) -> final/hybrid_demo.py -> comparison sheet + a splat model of solid surfaces -> make_tour.py tour -> public Space (random
folder; link only in hf://datasets/<user>/namma-v3-out/hybrid_demo/<time>/tour_url.txt) -> results uploaded next to it."""
import glob, json, os, shutil, subprocess, sys, time
V = os.path.dirname(os.path.abspath(__file__)); FIN = os.path.join(os.path.dirname(V), 'final'); sys.path.insert(0, V); sys.path.insert(0, FIN)
cfg = json.load(open(f'{V}/HYBRID.json')); T0 = time.time(); STAMP = time.strftime('%Y%m%d-%H%M'); OUT = os.path.abspath('out')
run = lambda c: subprocess.run(c, shell=True, check=True, executable='/bin/bash')

def say(*a): print(f'[{(time.time() - T0) / 60:5.1f} min]', *a, flush=True)

if not cfg.get('n'): say('HYBRID.json n = 0: nothing to do'); sys.exit(0)
tok = os.environ.get('HF_TOKEN'); assert tok, 'HF_TOKEN secret missing'
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=tok); user = api.whoami()['name']; repo = f'{user}/namma-v3-out'; R = cfg['run']
d = snapshot_download(repo, repo_type='dataset', local_dir='dl', token=tok,
                      allow_patterns=[f'{R}/full/level.json', f'{R}/full/nav.json', f'{R}/full/qa_sheet.jpg', f'{R}/mesh/mesh.ply', f'{R}/sparse/**'])
RD = os.path.abspath(f'{d}/{R}'); say('inputs', sorted(os.path.relpath(f, RD) for f in glob.glob(f'{RD}/**/*', recursive=True) if os.path.isfile(f)))

T = os.path.abspath('work'); EX = f'{T}/gsplat/examples'; data = f'{T}/data'; os.makedirs(data, exist_ok=True)
run(f'git clone -q --depth 1 -b v1.5.3 https://github.com/nerfstudio-project/gsplat {T}/gsplat && touch {EX}/datasets/__init__.py')
run('pip install -q torch --index-url https://download.pytorch.org/whl/cpu && pip install -q open3d opencv-python-headless imageio tqdm plyfile scipy trimesh '
    'fast-simplification typing_extensions "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" && sed -i '
    '"s/np.uint64(-1)/np.uint64(np.iinfo(np.uint64).max)/" $(python -c "import importlib.util as u; print(u.find_spec(\'pycolmap\').submodule_search_locations[0])")/scene_manager.py')
run(f'cd {T} && npm init -y >/dev/null && npm i -s --no-audit --no-fund playcanvas@2.23.2 @playcanvas/supersplat-viewer@1.37.0 @playcanvas/splat-transform@3.10.1 | tail -1')
shutil.copytree(f'{RD}/sparse', f'{data}/sparse', dirs_exist_ok=True)
import dl3dv, frames                                                                      # the run's frames, same names and order
_, folder = dl3dv.fetch(cfg['scene'], f'{T}/ds', tok, cfg.get('dl3dv_res', 'images_2'))
say('frames', frames.from_images(folder, f'{data}/images', int(cfg.get('nframes', 420)), int(cfg.get('long', 1440))))

run(f'python {FIN}/hybrid_demo.py {EX} {data} {RD} {OUT}')
HR = f'{OUT}/run'; shutil.copytree(f'{RD}/sparse', f'{HR}/sparse', dirs_exist_ok=True)
run(f'python {FIN}/make_tour.py plan {EX} {data} {HR} {OUT}/tour')
run(f'python {FIN}/make_tour.py build {HR} {OUT}/tour {T}/node_modules/@playcanvas/supersplat-viewer/public {T}/node_modules/.bin/splat-transform')
dst = f'hybrid_demo/{STAMP}'
api.upload_folder(folder_path=OUT, path_in_repo=dst, repo_id=repo, repo_type='dataset', ignore_patterns=['run/sparse/*'])
import tour_publish
tour_publish.publish(api, user, repo, f'{R}-hybrid', f'{OUT}/tour', say, url_path=f'{dst}/tour_url.txt')
if os.environ.get('GITHUB_ENV'): open(os.environ['GITHUB_ENV'], 'a').write(f'TOUR_RUN={dst}\n')
print('==== hybrid.json\n' + open(f'{OUT}/hybrid.json').read(), flush=True); say(f'results: hf://datasets/{repo}/{dst}')
