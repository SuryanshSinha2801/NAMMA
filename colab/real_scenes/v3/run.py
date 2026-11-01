# Kaggle job anasrazy/namma-v3 (2x T4, ~75 min): video in -> photoreal 3D model out, built only from the standard tools that
# Polycam / Postshot / nerfstudio users get good rooms with. Our own trainer and repair layers are out of the default path.
#   1 frames   sharpest frame per time window (video) or an evenly spaced subset (frame folder, e.g. DL3DV)
#   2 cameras  hloc SuperPoint + LightGlue -> COLMAP model (one shared SIMPLE_RADIAL camera) + sfm_report.json
#   3 splats   gsplat's own examples/simple_trainer.py: MCMC densification, bilateral grid (phone auto-exposure), SfM depth loss,
#              held-out eval on every 8th frame. GPU 1 trains a 7k-step preview while GPU 0 trains the full 30k-step model
#   4 QA       per model: qa_sheet.jpg (held-out real | model, 50 cm off the walk, outside, top-down), metrics.json (pass/fail),
#              levelled model.splat (SuperSplat), model.ply (full colour detail), walkthrough.mp4 (trainer's fly-through: colour | depth)
# SOURCE  dl3dv:auto | dl3dv:<scene hash>   handheld scene from HF DL3DV/DL3DV-Benchmark (Kaggle secret HF_TOKEN; "Agree" on HF once)
#         kaggle:<dataset name>             video or frames (or a .zip of them) in that attached Kaggle dataset
#         kaggle                            first .mp4/.mov under /kaggle/input (attach the video as a Kaggle dataset)
#         <path>                            a video file or a folder of frames
# Usually started from GitHub (.github/workflows/namma-v3.yml -> gh_run.py), which fetches DL3DV itself and attaches it as a dataset.
# With a write HF_TOKEN each result is also uploaded as soon as it exists to the private HF dataset <user>/namma-v3-out, so the
# preview can be checked while the full model still trains.
import glob, json, os, shutil, subprocess, sys, time
T0 = time.time(); W = '/kaggle/temp/v3'; O = '/kaggle/working'; os.makedirs(W, exist_ok=True)
SOURCE = os.environ.get('SOURCE', 'dl3dv:auto')
NF = os.environ.get('NFRAMES', '300'); NFRAMES, LONG = (int(NF) if NF.isdigit() else 300), int(os.environ.get('LONG', 1280))   # what a T4 trains in ~1 h
PREP_ONLY = os.environ.get('PREP_ONLY') == '1'                                            # whole-house pre-check: frames, cameras, gate; no training
FULL, PREVIEW = int(os.environ.get('STEPS', 30000)), int(os.environ.get('PREVIEW', 7000))
CAP_FULL, CAP_PREVIEW = 1_500_000, 1_000_000                                               # MCMC splat budget
# models to train, [name, steps, splat cap(, '2dgs')]; the first two run at once on the two T4s, the rest after. 'full' is the
# photoreal deliverable; a '2dgs' model (flat surfels, normal + distortion losses) is the geometry source for the mesh.
VARIANTS = json.loads(os.environ.get('VARIANTS', json.dumps([['preview', PREVIEW, CAP_PREVIEW], ['full', FULL, CAP_FULL]])))
RUN = time.strftime('%Y%m%d-%H%M')
D = f'{W}/data'                                                                            # COLMAP layout: images/ + sparse/0

def sh(c, check=False):
    print(f'\n[{time.time()-T0:5.0f}s] $ {c[:160]}', flush=True); r = subprocess.call(c, shell=True, executable='/bin/bash')
    if check and r: raise RuntimeError(f'failed: {c[:80]}')
    return r

def say(*a): print(f'[{time.time()-T0:5.0f}s]', *a, flush=True)

# no GPU = hours of CPU work that can never finish (30 Oct: a new account's run got no accelerator and ran 6 h on CPU): stop now
_g = subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', shell=True, capture_output=True, text=True)
say('GPUs:', _g.stdout.strip().replace('\n', ' | ') or 'none')
if _g.returncode or not _g.stdout.strip():
    raise SystemExit('NO GPU in this Kaggle session (account not phone-verified, or no GPU quota left): stopping before any work')
sh(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', check=True)
F = f'{W}/repo/colab/real_scenes/final'; sys.path.insert(0, F)
sh('ls /kaggle/input /kaggle/input/* | head -30')
EX = f'{W}/gsplat/examples'

# training dependencies install in the background while the cameras are solved (system Python; the SfM venv is separate because
# simple_trainer needs a pycolmap fork that clashes with the official pycolmap hloc uses)
PKGS = 'ninja gsplat==1.5.3 viser tyro "torchmetrics[image]" tensorboard tensorly scikit-learn "imageio[ffmpeg]" pyyaml matplotlib splines plyfile scikit-image trimesh'
deps = (f'git clone -q --depth 1 -b v1.5.3 https://github.com/nerfstudio-project/gsplat {W}/gsplat; '
        # a regular package, else Kaggle's preinstalled Hugging Face `datasets` hides examples/datasets (run 1 died on this)
        f'touch {W}/gsplat/examples/datasets/__init__.py; '
        f'echo "torch==$(python -c "import torch; print(torch.__version__)")" > {W}/pin.txt; '
        f'pip install -q -c {W}/pin.txt {PKGS} || pip install -q {PKGS}; '
        'pip install -q --no-deps "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787" '
        '"git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e"; '
        # the fork predates numpy 2: np.uint64(-1) overflows there
        # (located without importing it, since the import itself is what overflows)
        'sed -i "s/np.uint64(-1)/np.uint64(np.iinfo(np.uint64).max)/" $(python -c "import importlib.util as u; print(u.find_spec(\'pycolmap\').submodule_search_locations[0])")/scene_manager.py; '
        'TORCH_CUDA_ARCH_LIST=7.5 pip install -q --no-build-isolation "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5" '
        '|| echo "fused-ssim did not build"')
dep_proc = subprocess.Popen(deps, shell=True, executable='/bin/bash', stdout=open(f'{W}/deps.log', 'w'), stderr=subprocess.STDOUT)

def hf_token():
    try:
        from kaggle_secrets import UserSecretsClient; return UserSecretsClient().get_secret('HF_TOKEN')
    except Exception as e: say('Kaggle secret HF_TOKEN not available:', type(e).__name__, str(e)[:120])
    return os.environ.get('HF_TOKEN')

UP = None                                                                                  # (HfApi, repo) once uploads work
def hf_setup(tok):
    global UP
    if not tok: return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=tok); repo = f"{api.whoami()['name']}/namma-v3-out"
        api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True); UP = (api, repo); say(f'results also go to hf.co/datasets/{repo}/{RUN}')
    except Exception as e: say('results stay on Kaggle only (the token cannot create a repo):', str(e)[:150])

def push(path, name):
    if not UP: return
    try:
        if os.path.isdir(path): UP[0].upload_folder(folder_path=path, path_in_repo=f'{RUN}/{name}', repo_id=UP[1], repo_type='dataset', ignore_patterns=['*.ply'])
        else: UP[0].upload_file(path_or_fileobj=path, path_in_repo=f'{RUN}/{name}', repo_id=UP[1], repo_type='dataset')
        say('uploaded', name)
    except Exception as e: say('upload failed:', str(e)[:150])

sys.path.insert(0, f'{W}/repo/colab/real_scenes/v3'); import dl3dv

def source():
    if SOURCE.startswith('dl3dv:'):
        tok = hf_token()
        assert tok, ('DL3DV needs a Hugging Face token: on huggingface.co open datasets/DL3DV/DL3DV-Benchmark and click Agree, create a token '
                     '(huggingface.co/settings/tokens), then in this Kaggle notebook: Add-ons -> Secrets -> add HF_TOKEN and attach it')
        hf_setup(tok); scene, folder = dl3dv.fetch(SOURCE[6:], f'{W}/dl3dv', tok)
        json.dump(dict(source=f'{dl3dv.REPO}/{scene}', frames='nerfstudio/images_4 (960 x 540)'), open(f'{O}/source.json', 'w'))
        return folder
    if SOURCE.startswith('kaggle:'): return attached(SOURCE[7:])
    hf_setup(hf_token())
    if SOURCE == 'kaggle':
        v = sorted(glob.glob('/kaggle/input/**/*.mp4', recursive=True) + glob.glob('/kaggle/input/**/*.mov', recursive=True)
                   + glob.glob('/kaggle/input/**/*.MOV', recursive=True))
        assert v, 'SOURCE=kaggle but no .mp4/.mov under /kaggle/input: attach the video as a dataset'
        json.dump(dict(source=v[0]), open(f'{O}/source.json', 'w')); return v[0]
    return SOURCE

def attached(name):
    """Video, frame folder or .zip of frames inside the attached Kaggle dataset `name` (its mount path differs between Kaggle layouts)."""
    roots = [d for d in glob.glob('/kaggle/input/**/', recursive=True) if os.path.basename(d.rstrip('/')) == name]
    assert roots, f'dataset {name} is not attached (see ls /kaggle/input above)'
    root = roots[0]; files = glob.glob(f'{root}/**/*', recursive=True)
    vid = sorted(f for f in files if f.lower().endswith(('.mp4', '.mov')))
    if vid: return vid[0]
    for z in [f for f in files if f.lower().endswith('.zip')]: shutil.unpack_archive(z, f'{W}/unzipped')
    if os.path.exists(f'{W}/unzipped'): files = glob.glob(f'{W}/unzipped/**/*', recursive=True)
    count = {}
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')): count[os.path.dirname(f)] = count.get(os.path.dirname(f), 0) + 1
    assert count, f'no video or frames in dataset {name}'
    return max(count, key=count.get)                                                       # the folder with the most frames

# 1 frames
import frames
REUSE = os.environ.get('REUSE') == '1'      # plan of 31 Oct: an earlier run's frames + camera model (its Kaggle output, attached by
                                            # gh_run as a kernel source): no decode, no camera solve; THIN = keep every k-th frame
R0 = None

def take(k):
    """The reused camera model thinned to every k-th frame (final/thin_model.py, COLMAP's own file format) + those frames."""
    shutil.rmtree(f'{W}/sfm/sfm', ignore_errors=True); shutil.rmtree(f'{D}/images', ignore_errors=True); os.makedirs(f'{W}/sfm', exist_ok=True)
    r = subprocess.run([sys.executable, f'{F}/thin_model.py', f'{R0}/sparse/0', f'{W}/sfm/sfm', str(k), f'{W}/kept.txt'], capture_output=True, text=True)
    assert r.returncode == 0, 'thin_model failed: ' + (r.stderr or r.stdout)[-800:]
    info = json.loads(r.stdout.strip().splitlines()[-1]); os.makedirs(f'{D}/images', exist_ok=True)
    for nm in open(f'{W}/kept.txt').read().split(): shutil.copy(f'{R0}/images/{nm}', f'{D}/images/{nm}')
    rep_ = dict(frames=info['images'], registered=info['images'], points=info['points'], models=1, frames_without_points=info['frames_without_points'],
                reused=R0, thin=k, images_in=info['images_in'], points_in=info['points_in'])
    json.dump(rep_, open(f'{W}/sfm/sfm_report.json', 'w')); say('reused cameras, every', k, 'th frame:', rep_); return rep_

if REUSE:
    hits = glob.glob('/kaggle/input/**/sparse/0/images.bin', recursive=True); assert hits, 'REUSE: no sparse/0/images.bin under /kaggle/input'
    R0 = os.path.dirname(os.path.dirname(os.path.dirname(hits[0]))); assert os.path.isdir(f'{R0}/images'), f'REUSE: no images/ in {R0}'
    t_fr = t_sfm = time.time(); VTOT = 0; frames_min = sfm_min = 0.0; n = take(int(os.environ.get('THIN', '1')))['frames']
else:
    src = source(); say('source', src)
    VTOT = 0

    def plan_frames(v):
        """NFRAMES=auto: run 4's density (411 frames from 7398 video frames, one per 18) for any video length."""
        global NFRAMES, VTOT
        import cv2; cap_ = cv2.VideoCapture(v); VTOT = int(cap_.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap_.get(cv2.CAP_PROP_FPS) or 30
        # run 4's density by walking time (411 frames in 123 s), never more than the camera solve handles in its proven range
        # (31 Oct: 3,470 frames took 7.6 h of SfM and broke into 4 pieces; runs 4 / 6 solved 411 / 900 in minutes)
        if NF.startswith('auto') and VTOT > 0: NFRAMES = max(50, min(int(os.environ.get('SFM_MAX', 1000)), round(VTOT / fps * 411 / 123.3)))
        say('video frames', VTOT, '-> frames to extract', NFRAMES)

    t_fr = time.time()
    if not os.path.isdir(src): plan_frames(src)
    if os.path.isdir(src): n = frames.from_images(src, f'{D}/images', NFRAMES, LONG)
    else:                                                                                      # ffmpeg on all cores; OpenCV if that fails
        try: n = frames.from_video_fast(src, f'{D}/images', NFRAMES, LONG, VTOT)
        except Exception as e: say('fast frame reader failed:', str(e)[:200]); n = 0
        if n < 20: shutil.rmtree(f'{D}/images', ignore_errors=True); n = frames.from_video(src, f'{D}/images', NFRAMES, LONG)
    if n < 20 and not os.path.isdir(src):                                                       # e.g. 4K HEVC OpenCV cannot decode:
        say('only', n, 'frames decoded; converting the video with ffmpeg first')                 # H.264 at the training size
        sh(f'ffmpeg -v error -y -i "{src}" -vf "scale=\'if(gt(iw,ih),{LONG},-2)\':\'if(gt(iw,ih),-2,{LONG})\'" -c:v libx264 -crf 14 -preset veryfast {W}/video.mp4')
        plan_frames(f'{W}/video.mp4'); shutil.rmtree(f'{D}/images', ignore_errors=True); n = frames.from_video(f'{W}/video.mp4', f'{D}/images', NFRAMES, LONG)
    say(n, 'frames ->', f'{D}/images'); assert n >= 20, f'only {n} frames from {src}'
    t_sfm = time.time(); frames_min = round((t_sfm - t_fr) / 60, 1)

    # 2 cameras (same hloc venv that solved 574/578 frames of the 22 Oct video)
    E = f'{W}/sfmenv/bin/python'; P = f'--python {E}'
    sh(f'pip -q install uv 2>&1 | tail -1; unset UV_SYSTEM_PYTHON UV_PYTHON PIP_TARGET; uv venv -q -p 3.10 {W}/sfmenv && '
       f'uv pip install -q {P} torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118 > {W}/sfm_pip.log 2>&1; '
       f'git clone -q --recursive https://github.com/cvg/Hierarchical-Localization {W}/hloc && '
       f'uv pip install -q {P} "numpy<2" -e {W}/hloc "lightglue @ git+https://github.com/cvg/LightGlue" >> {W}/sfm_pip.log 2>&1; tail -3 {W}/sfm_pip.log')
    sh(f'{E} -c "import hloc, lightglue, pycolmap; print(\'hloc ok, pycolmap\', pycolmap.__version__)"', check=True)
    # live in the Kaggle log (a `| tail` here held every line back until the end: the 30 Oct house run looked stuck for hours)
    sh(f'cd {W} && PYTHONUNBUFFERED=1 {E} {F}/sfm_hloc.py {D}/images {W}/sfm 10 15 SIMPLE_RADIAL 2>&1 | tee {W}/sfm.log '
       f'| grep --line-buffered -v -E "it/s|s/it" | while IFS= read -r l; do echo "[$(date -u +%H:%M:%S)] $l"; done')
    assert os.path.exists(f'{W}/sfm/sfm/images.bin'), 'camera solve failed: no COLMAP model (see the hloc log above)'
    sfm_min = round((time.time() - t_sfm) / 60, 1)
    try: shutil.copytree(f'{W}/sfm', f'{O}/sfm_work', dirs_exist_ok=True); shutil.copy(f'{W}/sfm.log', O)   # a retry continues from here
    except Exception as e: say('could not keep the SfM work:', str(e)[:150])
rep = json.load(open(f'{W}/sfm/sfm_report.json')); shutil.copy(f'{W}/sfm/sfm_report.json', O); push(f'{O}/sfm_report.json', 'sfm_report.json')
if rep['registered'] < 0.85 * rep['frames'] or rep['models'] > 1:
    say(f"WARNING: only {rep['registered']} of {rep['frames']} frames in the main model ({rep['models']} models): the capture broke "
        "apart (fast turns / blank walls); the model covers only the registered part")
assert rep['registered'] >= 20, 'fewer than 20 frames have cameras: nothing to train on'
DEPTH = '--depth-loss' if not rep.get('frames_without_points') else ''                     # its loader needs points in every frame
os.makedirs(f'{D}/sparse/0', exist_ok=True)
for x in ('cameras.bin', 'images.bin', 'points3D.bin'): shutil.copy(f'{W}/sfm/sfm/{x}', f'{D}/sparse/0/{x}')

# 3 splats: wait for the dependencies, reuse namma-prep's compiled gsplat if it matches (else it compiles, ~5 min)
dep_proc.wait(); sh(f'grep -v "^\\s*$" {W}/deps.log | tail -8')
tgz = glob.glob('/kaggle/input/**/gsplat_ext.tgz', recursive=True)
if tgz:
    sh(f'mkdir -p ~/.cache && tar xzf {tgz[0]} -C ~/.cache')
    sh('find $(python -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec(\'gsplat\').origin))") -exec touch -d "2000-01-01" {} +')
sh(f'python {F}/gsplat_warm.py', check=True)                                              # compile once, before two GPUs load it
PP = '' if sh('python -c "import fused_ssim"') == 0 else f'{F}/fused_ssim_shim'           # pure-torch SSIM if the CUDA one failed
if PP: say('using the pure-torch SSIM shim')
tried = set()
for _ in range(6):                                                                         # import the trainer once; install what it lacks
    r = subprocess.run(f'cd {EX} && PYTHONPATH={PP}:$PYTHONPATH python -c "import simple_trainer, simple_trainer_2dgs"', shell=True, capture_output=True, text=True)
    if r.returncode == 0: say('simple_trainer imports ok'); break
    print(r.stderr[-1500:], flush=True); import re; m = re.search(r"No module named '([\w.]+)'", r.stderr)
    if not m: break
    pkg = {'cv2': 'opencv-python-headless', 'sklearn': 'scikit-learn', 'yaml': 'pyyaml', 'PIL': 'pillow'}.get(m.group(1).split('.')[0], m.group(1).split('.')[0])
    if pkg in tried: break                                                                  # installing it did not help
    tried.add(pkg); sh(f'pip -q install {pkg} 2>&1 | tail -1')
import room_prior                                                                       # floor/ceiling + needle prior (kind 'prior')
say('room prior patched in:', room_prior.patch(f'{EX}/simple_trainer.py'))
import geo_prior                                                                        # dense depth + normal guide (kind 'geo')
say('geo prior patched in:', geo_prior.patch(f'{EX}/simple_trainer.py'))
if any(len(v) > 3 and 'geo' in v[3] for v in VARIANTS):                                  # DA3 depth per frame, before training
    sh(f'cd {EX} && PYTHONPATH={EX}:{F}:$PYTHONPATH python {F}/geo_prior.py {D} {W}/geo 2>&1 | grep -v "it/s" | tail -15')
    if os.path.exists(f'{W}/geo/geo_report.json'): shutil.copy(f'{W}/geo/geo_report.json', O); push(f'{O}/geo_report.json', 'geo_report.json')
    sh('nvidia-smi --query-gpu=memory.used --format=csv')
import torch
NGPU = torch.cuda.device_count(); say('GPUs:', NGPU)

if PREP_ONLY:                                    # plan of 30 Oct: prove a whole-house run fits before any splat training (final/precheck.py)
    tim = dict(frames=n, frames_planned=NFRAMES, video_frames=VTOT, frames_min=frames_min, sfm_min=sfm_min,
               hours_before_precheck=round((time.time() - T0) / 3600, 3))
    sh(f"cd {EX} && PYTHONPATH={EX}:{F}:$PYTHONPATH python {F}/precheck.py {EX} {D} {W} {O} '{json.dumps(tim)}' {PP}")
    shutil.copytree(f'{D}/sparse', f'{O}/sparse', dirs_exist_ok=True); shutil.copytree(f'{D}/images', f'{O}/images', dirs_exist_ok=True)
    for b in glob.glob(f'{W}/bench_*.log'): shutil.copy(b, O)
    for f in ('precheck.json', 'rooms.json', 'rooms.png', 'chunks.json'):
        if os.path.exists(f'{O}/{f}'): push(f'{O}/{f}', f)
    assert os.path.exists(f'{O}/precheck.json'), 'pre-check failed (see its log above)'
    say('PRE-CHECK DONE', json.load(open(f'{O}/precheck.json')).get('decision')); raise SystemExit(0)

if os.environ.get('AUTO_STEPS') == '1':     # run 4's recipe for any size: ~83.6 steps per training view, 1.5 M splats per 37 m2
    import math, precheck, rooms
    precheck.setup(EX, D, W, O, {}, PP)
    for _ in range(4):
        precheck.DEPTH = DEPTH; nreg = rep['registered']; tv = nreg - math.ceil(nreg / 8); steps = int(round(83.6 * tv))
        nm, C_, R_, P_, ob_ = rooms.load(f'{D}/sparse/0'); area = sum(r_['area_m2'] for r_ in rooms.split(nm, C_, R_, P_, ob_)[1])
        cap = int(min(4e6, max(1.5e6, 1.5e6 * area / 37)))
        dry = precheck.bench('dryrun', '0', cap); hours = (steps * (dry['s_per_step'] or 9) * 1.1) / 3600 + 0.5
        left = 11.0 - (time.time() - T0) / 3600                                                # Kaggle stops a session at 12 h
        say(f'GO/NO-GO: {n} frames, {tv} train views, {area:.1f} m2, cap {cap}, {steps} steps, {dry["s_per_step"]} s/step, '
            f'peak {dry["peak_mem_gb"]} GB -> {hours:.1f} h of {left:.1f} h left')
        json.dump(dict(frames=n, train_views=tv, area_m2=round(area, 1), cap=cap, steps=steps, dry=dry, hours=round(hours, 2), left=round(left, 2)),
                  open(f'{O}/go_nogo.json', 'w'), indent=1)
        if dry['ok'] and max(dry['peak_mem_gb'].values()) <= 13 and hours <= min(9.0, left): break
        if not (REUSE and dry['ok']): raise SystemExit(f'NO-GO: would not finish safely ({hours:.1f} h, dry run ok={dry["ok"]}): nothing trained')
        rep = take(rep['thin'] + 1); n = rep['frames']; DEPTH = '--depth-loss' if not rep.get('frames_without_points') else ''
        for x in ('cameras.bin', 'images.bin', 'points3D.bin'): shutil.copy(f'{W}/sfm/sfm/{x}', f'{D}/sparse/0/{x}')
    else: raise SystemExit('NO-GO: still too long after thinning 4 times')
    VARIANTS = [['full', steps, cap]]; say('training', VARIANTS)

def start(name, steps, cap, gpu, kind='gs'):
    res = f'{W}/{name}'
    if kind == '2dgs':                                                                    # geometry model: flat surfels
        cmd = (f'cd {EX} && CUDA_VISIBLE_DEVICES={gpu} PYTHONPATH={PP}:$PYTHONPATH python simple_trainer_2dgs.py --data-dir {D} --data-factor 1 '
               f'--result-dir {res} --max-steps {steps} --eval-steps {steps} --save-steps {steps} --disable-viewer --normal-loss --dist-loss {DEPTH}')
    else:                                                                                  # 3DGS (+ room prior / pose refinement)
        env = ('ROOM_PRIOR=1 ' if 'prior' in kind else '') + (f'GEO_PRIOR=1 GEO_DIR={W}/geo ' if 'geo' in kind else '')
        pr = f'{env}PYTHONPATH={F}:' if env else 'PYTHONPATH='                              # kinds: gs, prior, pose, geo and + of them
        po = ' --pose-opt' if 'pose' in kind else ''                                           # refine the SfM cameras while training
        cmd = (f'cd {EX} && CUDA_VISIBLE_DEVICES={gpu} {pr}{PP}:$PYTHONPATH python simple_trainer.py mcmc --data-dir {D} --data-factor 1 '
               f'--result-dir {res} --max-steps {steps} --eval-steps {steps} --save-steps {steps} --ply-steps {steps} --save-ply '
               f'--disable-viewer --use-bilateral-grid {DEPTH} --strategy.cap-max {cap}{po}')
    say(cmd); return dict(name=name, gpu=gpu, res=res, kind=kind, t=time.time(),
                          p=subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=open(f'{res}.log', 'w'), stderr=subprocess.STDOUT))

def progress(j):
    try: return [x for x in open(j['res'] + '.log', errors='ignore').read()[-3000:].replace('\r', '\n').split('\n') if x.strip()][-1][:150]
    except Exception: return ''

def finish(j):
    out = f"{O}/{j['name']}"; os.makedirs(out, exist_ok=True)
    lines = open(j['res'] + '.log', errors='ignore').read().replace('\r', '\n').split('\n')
    keep = [x for x in lines if x.strip() and '%|' not in x]                              # drop the progress bar
    open(f'{out}/train.log', 'w').write('\n'.join(keep)); print('\n'.join(keep[-25:]), flush=True)
    say(f"{j['name']} exit {j['p'].returncode} after {(time.time() - j['t']) / 60:.0f} min")
    if j['kind'] == '2dgs':                                                              # its mesh is made at the end (needs full's level)
        st = sorted(glob.glob(f"{j['res']}/stats/val_step*.json"))
        if st: m = json.load(open(st[-1])); json.dump({k: round(v, 3) for k, v in m.items() if isinstance(v, float)}, open(f'{out}/metrics.json', 'w'))
    else: sh(f"cd {EX} && CUDA_VISIBLE_DEVICES={j['gpu']} python {F}/v3_qa.py {EX} {D} {j['res']} {out}")
    v = sorted(glob.glob(f"{j['res']}/videos/traj_*.mp4"))
    if v: shutil.copy(v[-1], f'{out}/walkthrough.mp4')
    push(out, j['name'])

queue = [tuple(v) for v in VARIANTS]; free = list(range(max(1, min(NGPU, 2))))           # GPUs not training right now
running = [start(*v[:3], free.pop(0), *v[3:]) for v in [queue.pop(0) for _ in range(min(len(queue), len(free)))]]
last = time.time()
while running:
    time.sleep(30)
    for j in [j for j in running if j['p'].poll() is not None]:
        running.remove(j); finish(j)
        if queue: v = queue.pop(0); running.append(start(*v[:3], j['gpu'], *v[3:]))
    if time.time() - last > 300: last = time.time(); [say(j['name'], progress(j)) for j in running]

S = {n: json.load(open(f'{O}/{n}/metrics.json')) for n in [v[0] for v in VARIANTS] if os.path.exists(f'{O}/{n}/metrics.json')}
# the mould: a solid mesh in the same coordinates as full/model.splat, from the 2DGS model if one was trained, else the full model
geo = next(([v[0], '2dgs'] for v in VARIANTS if len(v) > 3 and v[3] == '2dgs'), ['full', '3dgs'])
if os.path.isdir(f'{W}/{geo[0]}/ckpts') and os.environ.get('NO_MESH') != '1':
    sh(f"cd {EX} && python {F}/mesh_tsdf.py {EX} {D} {W}/{geo[0]} {O}/mesh {O}/full/level.json {geo[1]} {os.environ.get('MESH_VOXELS', 512)}")
    if os.path.exists(f'{O}/mesh/mesh.json'): S['mesh'] = json.load(open(f'{O}/mesh/mesh.json'))
shutil.copytree(f'{D}/sparse', f'{O}/sparse', dirs_exist_ok=True)                        # cameras, so QA can be redone later
shutil.copytree(f'{D}/images', f'{O}/images', dirs_exist_ok=True)                        # frames, for the tour (cameras need them)
json.dump(dict(source=SOURCE, sfm=rep, models=S, minutes=round((time.time() - T0) / 60)), open(f'{O}/v3_summary.json', 'w'), indent=1)
push(f'{O}/v3_summary.json', 'v3_summary.json')
say('V3 DONE', json.dumps(S)); sh(f'ls -la {O} {O}/*/ | head -40')
if 'full' not in S: raise SystemExit('V3 FAILED: no full model (see the trainer logs above)')
