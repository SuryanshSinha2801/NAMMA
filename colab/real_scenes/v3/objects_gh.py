"""GitHub Actions side of the object layer (.github/workflows/objects.yml, settings in v3/OBJECTS.json).
The run's model.ply + sparse/ (private HF results dataset) -> private Kaggle dataset namma-objects-in -> Kaggle job
anasrazy/namma-objects (final/objects_job.py -> final/objects3d.py, with the run's frames dataset attached; the run's level matrix
and metres-per-unit from its tour go in JOB_CFG) -> outputs -> hf://datasets/<user>/namma-v3-out/objects/<time>/,
then two tours: the real model with object labels / click-to-select / search (link only in objects/<time>/tour_url.txt)
and the object-tinted model (tour_tint_url.txt), + screenshots. Only one Kaggle GPU job at a time."""
import glob, json, os, shutil, subprocess, sys, time
V = os.path.dirname(os.path.abspath(__file__)); FIN = os.path.join(os.path.dirname(V), 'final'); sys.path.insert(0, V); sys.path.insert(0, FIN)
cfg = json.load(open(f'{V}/OBJECTS.json')); KU = cfg.get('kaggle_user', 'anasrazy'); KID = f'{KU}/namma-objects'; DS = f'{KU}/namma-objects-in'
OUT = os.path.abspath('out'); T0 = time.time(); STAMP = time.strftime('%Y%m%d-%H%M')
run = lambda c: subprocess.run(c, shell=True, check=True, executable='/bin/bash')

def say(*a): print(f'[{(time.time() - T0) / 60:5.1f} min]', *a, flush=True)

def kg(*args, check=True):
    r = subprocess.run(['kaggle', *args], capture_output=True, text=True); out = (r.stdout + r.stderr).strip()
    if check and r.returncode: sys.exit(f'kaggle {" ".join(args[:2])} failed: {out[-800:]}')
    return r.returncode, out

if not cfg.get('n'): say('OBJECTS.json n = 0: nothing to do'); sys.exit(0)
tok = os.environ.get('HF_TOKEN'); assert tok and os.environ.get('KAGGLE_API_TOKEN'), 'needs HF_TOKEN and KAGGLE_API_TOKEN secrets'
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=tok); user = api.whoami()['name']; repo = f'{user}/namma-v3-out'; R = cfg['run']
d = snapshot_download(repo, repo_type='dataset', local_dir='dl', token=tok,
                      allow_patterns=[f'{R}/full/model.ply', f'{R}/full/level.json', f'{R}/tour/tour.json', f'{R}/sparse/**'])
cfg['level'] = json.load(open(f'{d}/{R}/full/level.json'))['matrix']; cfg['metres_per_unit'] = json.load(open(f'{d}/{R}/tour/tour.json'))['metres_per_unit']
if not cfg.get('collect_only'):
    os.makedirs('in', exist_ok=True); shutil.copy(f'{d}/{R}/full/model.ply', 'in/model.ply'); shutil.make_archive('in/sparse', 'zip', f'{d}/{R}', 'sparse')
    json.dump(dict(title='namma-objects-in', id=DS, licenses=[dict(name='other')]), open('in/dataset-metadata.json', 'w'))
    rc, st = kg('datasets', 'status', DS, check=False)
    say(kg('datasets', 'version', '-p', 'in', '-m', f'{R} {STAMP}', '-q')[1][-300:] if rc == 0 and 'ready' in st.lower()
        else kg('datasets', 'create', '-p', 'in', '-q')[1][-300:])
    for _ in range(90):
        time.sleep(20); st = kg('datasets', 'status', DS, check=False)[1]
        if 'ready' in st.lower() or 'error' in st.lower(): break
    say('dataset', DS, st[-120:])
os.makedirs('kern', exist_ok=True)
json.dump(dict(id=KID, title='namma-objects', code_file='objects_job.py', language='python', kernel_type='script', is_private=True,
               enable_gpu=True, enable_internet=True, machine_shape='NvidiaTeslaT4', kernel_sources=[f'{KU}/namma-prep'],
               dataset_sources=[DS, cfg['frames_dataset']]), open('kern/kernel-metadata.json', 'w'))
code = open(f'{FIN}/objects_job.py').read(); first = 'import gc, glob, json, os, shutil, subprocess, sys, time\n'; assert code.startswith(first)
open('kern/objects_job.py', 'w').write(code.replace(first, first + f'os.environ["JOB_CFG"] = {json.dumps(json.dumps(cfg))}  # set by objects_gh.py\n', 1))
if cfg.get('collect_only'): say('collect only: no new Kaggle run, waiting for the last one'); t_push = time.time() - 3600  # poller ran out
else: say(kg('kernels', 'push', '-p', 'kern')[1][-400:]); t_push = time.time()
seen, last, s = False, '', ''
while time.time() - T0 < 280 * 60:
    time.sleep(60); rc, s = kg('kernels', 'status', KID, check=False); s = s.lower()
    if rc: say('status check failed:', s[-150:]); continue
    if s != last: say('status:', s[-100:]); last = s
    if 'running' in s or 'queued' in s: seen = True
    if 'error' in s or 'cancel' in s: break
    if 'complete' in s and (seen or time.time() - t_push > 900): break
os.makedirs(OUT, exist_ok=True); kg('kernels', 'output', KID, '-p', OUT, '-o', check=False)
for f in [f'{OUT}/objects.json'] + glob.glob(f'{OUT}/*.log'):
    if os.path.exists(f): print(f'==== {os.path.basename(f)}\n' + ''.join(open(f, errors='ignore').readlines()[-150:]), flush=True)
dst = f'objects/{STAMP}'
api.upload_folder(folder_path=OUT, path_in_repo=dst, repo_id=repo, repo_type='dataset', ignore_patterns=['full/*'])
say(f'results: hf://datasets/{repo}/{dst}')
if not os.path.exists(f'{OUT}/objects.json'): sys.exit(f'Kaggle job ended with: {s[-200:]} (no objects.json)')
if not os.path.exists(f'{OUT}/full/model.ply'): sys.exit('no tinted model.ply: no tour')

# two tours, built like the run's own: the real model with a label, click-to-select and search per object (the one to use), and
# the object-tinted model (each object its own colour: to check the split)
d0 = d; d = snapshot_download(repo, repo_type='dataset', local_dir='dl2', token=tok,
                              allow_patterns=[f'{R}/full/level.json', f'{R}/full/nav.json', f'{R}/mesh/mesh.glb', f'{R}/sparse/**'])
HR, HT = f'{OUT}/run', f'{OUT}/run_tint'
for h in (HR, HT):
    os.makedirs(f'{h}/full', exist_ok=True); shutil.copy(f'{OUT}/objects.json', h); shutil.copytree(f'{OUT}/crops', f'{h}/crops', dirs_exist_ok=True)
    for x in ('level.json', 'nav.json'): shutil.copy(f'{d}/{R}/full/{x}', f'{h}/full/{x}')
    if os.path.exists(f'{d}/{R}/mesh/mesh.glb'): os.makedirs(f'{h}/mesh', exist_ok=True); shutil.copy(f'{d}/{R}/mesh/mesh.glb', f'{h}/mesh/mesh.glb')
shutil.copy(f'{d0}/{R}/full/model.ply', f'{HR}/full/model.ply'); shutil.move(f'{OUT}/full/model.ply', f'{HT}/full/model.ply')
T = os.path.abspath('work'); EX = f'{T}/gsplat/examples'; data = f'{T}/data'; os.makedirs(data, exist_ok=True)
run(f'git clone -q --depth 1 -b v1.5.3 https://github.com/nerfstudio-project/gsplat {T}/gsplat && touch {EX}/datasets/__init__.py')
run('pip install -q torch --index-url https://download.pytorch.org/whl/cpu && pip install -q opencv-python-headless imageio tqdm plyfile scipy trimesh '
    'fast-simplification typing_extensions "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" && sed -i '
    '"s/np.uint64(-1)/np.uint64(np.iinfo(np.uint64).max)/" $(python -c "import importlib.util as u; print(u.find_spec(\'pycolmap\').submodule_search_locations[0])")/scene_manager.py')
run(f'cd {T} && npm init -y >/dev/null && npm i -s --no-audit --no-fund playcanvas@2.23.2 @playcanvas/supersplat-viewer@1.37.0 @playcanvas/splat-transform@3.10.1 | tail -1')
shutil.copytree(f'{d}/{R}/sparse', f'{data}/sparse', dirs_exist_ok=True)
import dl3dv, frames                                                                      # the run's frames, same names and order
_, folder = dl3dv.fetch(cfg['scene'], f'{T}/ds', tok, cfg.get('dl3dv_res', 'images_2'))
say('frames', frames.from_images(folder, f'{data}/images', int(cfg.get('nframes', 420)), int(cfg.get('long', 1440))))
run(f'python {FIN}/make_tour.py plan {EX} {data} {HR} {OUT}/tour'); shutil.copytree(f'{OUT}/tour', f'{OUT}/tour_tint')
import tour_publish
for h, tour, name, url in ((HR, 'tour', f'{R}-objects', 'tour_url.txt'), (HT, 'tour_tint', f'{R}-objtint', 'tour_tint_url.txt')):
    run(f'python {FIN}/make_tour.py build {h} {OUT}/{tour} {T}/node_modules/@playcanvas/supersplat-viewer/public {T}/node_modules/.bin/splat-transform')
    api.upload_folder(folder_path=f'{OUT}/{tour}', path_in_repo=f'{dst}/{tour}', repo_id=repo, repo_type='dataset')
    tour_publish.publish(api, user, repo, name, f'{OUT}/{tour}', say, url_path=f'{dst}/{url}')
if os.environ.get('GITHUB_ENV'): open(os.environ['GITHUB_ENV'], 'a').write(f'TOUR_RUN={dst}\n')
say('done')
