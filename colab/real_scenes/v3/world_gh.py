"""NAMMA World v1 (Plan B) from GitHub Actions (.github/workflows/world.yml, settings in v3/WORLD.json): for each sample, one
frame of a filmed scene (undistorted, K from SfM) + its real neighbouring frames (cameras relative to it, SfM units) + the SfM
points it sees (for the metric scale) -> Kaggle dataset namma-world-in -> Kaggle job anasrazy/namma-world (final/world_job.py,
T4: MoGe-2 depth + SDXL inpainting drones) -> outputs -> hf://datasets/<user>/namma-v3-out/world/<time>/<sample>/; each
sample's viewer published like the tours (links only in tour_url.txt files). One Kaggle job at a time."""
import glob, json, os, shutil, subprocess, sys, time
V = os.path.dirname(os.path.abspath(__file__)); FIN = os.path.join(os.path.dirname(V), 'final'); sys.path.insert(0, V); sys.path.insert(0, FIN)
cfg = json.load(open(f'{V}/WORLD.json'))
OUT = os.path.abspath('out'); T0 = time.time(); STAMP = time.strftime('%Y%m%d-%H%M')
run = lambda c: subprocess.run(c, shell=True, check=True, executable='/bin/bash')


def say(*a): print(f'[{(time.time() - T0) / 60:5.1f} min]', *a, flush=True)


def kg(*args, check=True):
    r = subprocess.run(['kaggle', *args], capture_output=True, text=True); out = (r.stdout + r.stderr).strip()
    if check and r.returncode: sys.exit(f'kaggle {" ".join(args[:2])} failed: {out[-800:]}')
    return r.returncode, out


if not cfg.get('n'): say('WORLD.json n = 0: nothing to do'); sys.exit(0)
tok = os.environ.get('HF_TOKEN'); assert tok and os.environ.get('KAGGLE_API_TOKEN'), 'needs HF_TOKEN and KAGGLE_API_TOKEN secrets'
from kaggle.api.kaggle_api_extended import KaggleApi                                       # the account is the token's own (a new
_ka = KaggleApi(); _ka.authenticate(); KU = cfg.get('kaggle_user') or _ka.config_values['username']   # token = maybe a new account)
say('Kaggle account:', 'anasrazy' if KU == 'anasrazy' else 'not anasrazy (a newer account; name kept out of the public log)')
KID = f'{KU}/namma-world'; DS = f'{KU}/namma-world-in'
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=tok); user = api.whoami()['name']; repo = f'{user}/namma-v3-out'; R = cfg['run']; dst = f'world/{STAMP}'
import numpy as np, cv2

if not cfg.get('collect'):
    # 1 inputs: the run's frames and cameras (as genworld_gh.py)
    d = snapshot_download(repo, repo_type='dataset', local_dir='dl', token=tok, allow_patterns=[f'{R}/sparse/**'])
    T = os.path.abspath('work'); EX = f'{T}/gsplat/examples'; data = f'{T}/data'; os.makedirs(data, exist_ok=True)
    run(f'git clone -q --depth 1 -b v1.5.3 https://github.com/nerfstudio-project/gsplat {T}/gsplat && touch {EX}/datasets/__init__.py')
    run('pip install -q torch --index-url https://download.pytorch.org/whl/cpu && pip install -q opencv-python-headless imageio tqdm scipy '
        'typing_extensions "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" && sed -i '
        '"s/np.uint64(-1)/np.uint64(np.iinfo(np.uint64).max)/" $(python -c "import importlib.util as u; print(u.find_spec(\'pycolmap\').submodule_search_locations[0])")/scene_manager.py')
    shutil.copytree(f'{d}/{R}/sparse', f'{data}/sparse', dirs_exist_ok=True)
    import dl3dv, frames
    _, folder = dl3dv.fetch(cfg['scene'], f'{T}/ds', tok, cfg.get('dl3dv_res', 'images_2'))
    say('frames', frames.from_images(folder, f'{data}/images', int(cfg.get('nframes', 420)), int(cfg.get('long', 1440))))
    os.chdir(EX); sys.path.insert(0, EX); from datasets.colmap import Parser
    p = Parser(data, factor=1, normalize=False, test_every=8); cid = p.camera_ids[0]; K0 = p.Ks_dict[cid]; W0, H0 = p.imsize_dict[cid]
    LONG = int(cfg.get('long_px', 1024)); k = LONG / max(W0, H0); W, H = round(W0 * k), round(H0 * k); K = K0.copy(); K[:2] *= k

    def load(i):
        im = cv2.imread(p.image_paths[i])
        if cid in p.mapx_dict: im = cv2.remap(im, p.mapx_dict[cid], p.mapy_dict[cid], cv2.INTER_LINEAR); x, y, w, h = p.roi_undist_dict[cid]; im = im[y:y + h, x:x + w]
        return cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)

    IN = os.path.abspath(f'{T0:.0f}_in'); os.makedirs(IN, exist_ok=True)
    for smp in cfg['samples']:
        o = f'{IN}/{smp["name"]}__'; fi = int(smp['frame'])                                     # flat files (no folders in the dataset)
        cv2.imwrite(f'{o}input.jpg', load(fi), [cv2.IMWRITE_JPEG_QUALITY, 95]); c2w_i = p.camtoworlds[fi]; w2c = np.linalg.inv(c2w_i)
        pts = p.points[p.point_indices[p.image_names[fi]]]; pc = pts @ w2c[:3, :3].T + w2c[:3, 3]; pc = pc[pc[:, 2] > 0]
        u = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]; v = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]; ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        nbs = []
        for off in [int(x) for x in str(smp['offsets']).split(',')]:
            j = fi + off
            if not 0 <= j < len(p.image_names): continue
            Tj = np.linalg.inv(p.camtoworlds[j]) @ c2w_i; c2w = np.linalg.inv(Tj)                 # camera j in camera i's frame (SfM units)
            tag = ('p' if off > 0 else 'm') + str(abs(off))                                     # no '+': Kaggle drops it from file names
            cv2.imwrite(f'{o}nb_{tag}.jpg', load(j), [cv2.IMWRITE_JPEG_QUALITY, 95])
            nbs.append(dict(offset=off, frame=p.image_names[j], file=f'{smp["name"]}__nb_{tag}.jpg', c2w=c2w.tolist()))
        json.dump(dict(name=smp['name'], input=f'{smp["name"]}__input.jpg', frame=p.image_names[fi], K=K.tolist(), prompt=smp.get('prompt', cfg.get('prompt')),
                       sfm=dict(uv=np.stack([u[ok], v[ok]], 1).astype(int).tolist(), z=pc[ok, 2].tolist()), neighbours=nbs), open(f'{o}cam.json', 'w'))
        say(smp['name'], p.image_names[fi], f'{W}x{H}', int(ok.sum()), 'SfM points,', len(nbs), 'neighbours')
    os.chdir(os.path.dirname(OUT))
    json.dump(dict(title='namma-world-in', id=DS, licenses=[dict(name='other')]), open(f'{IN}/dataset-metadata.json', 'w'))
    rc, st = kg('datasets', 'status', DS, check=False)
    say(kg('datasets', 'version', '-p', IN, '-m', f'{R} {STAMP}', '-q')[1][-300:] if rc == 0 and 'ready' in st.lower()
        else kg('datasets', 'create', '-p', IN, '-q')[1][-300:])
    time.sleep(30)
    for _ in range(90):
        st = kg('datasets', 'status', DS, check=False)[1]
        if 'ready' in st.lower() or 'error' in st.lower(): break
        time.sleep(20)
    say('dataset', DS, st[-120:]); assert 'ready' in st.lower(), f'{DS} not ready: {st[-200:]}'
    time.sleep(60)                                                                             # a new version can need a minute more to attach

    # 2 the Kaggle job
    os.makedirs('kern', exist_ok=True)
    json.dump(dict(id=KID, title='namma-world', code_file='world_job.py', language='python', kernel_type='script', is_private=True,
                   enable_gpu=True, enable_internet=True, machine_shape='NvidiaTeslaT4', dataset_sources=[DS]), open('kern/kernel-metadata.json', 'w'))
    code = open(f'{FIN}/world_job.py').read(); first = 'import glob, json, os, shutil, subprocess, sys, time\n'; assert code.startswith(first)
    open('kern/world_job.py', 'w').write(code.replace(first, first + f'os.environ["JOB_CFG"] = {json.dumps(json.dumps(cfg))}  # set by world_gh.py\n', 1))
    say(kg('kernels', 'push', '-p', 'kern')[1][-400:])
t_push = time.time(); seen, last, s = bool(cfg.get('collect')), '', ''
while time.time() - T0 < 240 * 60:
    time.sleep(60); rc, s = kg('kernels', 'status', KID, check=False); s = s.lower()
    if rc: say('status check failed:', s[-150:]); continue
    if s != last: say('status:', s[-100:]); last = s
    if 'running' in s or 'queued' in s: seen = True
    if 'error' in s or 'cancel' in s: break
    if 'complete' in s and (seen or time.time() - t_push > 900): break
os.makedirs(OUT, exist_ok=True); kg('kernels', 'output', KID, '-p', OUT, '-o', check=False)
for f in [f'{OUT}/world_report.json'] + glob.glob(f'{OUT}/*.log'):
    if os.path.exists(f): print(f'==== {os.path.basename(f)}\n' + ''.join(open(f, errors='ignore').readlines()[-120:]), flush=True)
api.upload_folder(folder_path=OUT, path_in_repo=dst, repo_id=repo, repo_type='dataset'); say(f'results: hf://datasets/{repo}/{dst}')
import tour_publish
done = [os.path.dirname(f) for f in sorted(glob.glob(f'{OUT}/*/world.glb'))]
for k, o in enumerate(done):
    web = os.path.abspath(f'web_{k}'); os.makedirs(web, exist_ok=True)
    for f in ('index.html', 'world.glb', 'world.json'): shutil.copy(f'{o}/{f}', web)
    tour_publish.publish(api, user, repo, f'{R}-world1', web, say, url_path=f'{dst}/{"tour_url.txt" if k == 0 else f"tour{k + 1}_url.txt"}')
if not done: sys.exit(f'Kaggle job ended with: {s[-200:]} (no world)')
if os.environ.get('GITHUB_ENV'): open(os.environ['GITHUB_ENV'], 'a').write(f'TOUR_RUN={dst}\n')
say('done')
