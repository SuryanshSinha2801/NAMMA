"""GitHub Actions side of namma-v3 (.github/workflows/namma-v3.yml). The cloud session that writes the code cannot reach kaggle.com;
GitHub's runners can. input -> private Kaggle dataset (DL3DV frames are fetched here, so the HF token stays a GitHub secret)
-> push the Kaggle job (v3/run.py) -> wait -> download its outputs -> private HF dataset <hf user>/namma-v3-out/<run> + metrics here.
RUN.json: n         run counter: change it to start a run (0 = do nothing)
          collect   true: push nothing, just wait for the job already on Kaggle and collect its outputs
          source    dl3dv:auto | dl3dv:<scene hash> | kaggle:<owner>/<dataset> (a video or frames already on Kaggle)
          kaggle_user, env (overrides run.py settings, e.g. {"STEPS": "30000", "PREVIEW": "7000"})
Secrets: KAGGLE_API_TOKEN (contents of ~/.kaggle/access_token), HF_TOKEN (write; "Agree" clicked on DL3DV/DL3DV-Benchmark)."""
import glob, json, os, shutil, subprocess, sys, time
V = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, V)
cfg = json.load(open(f'{V}/RUN.json')); KU = cfg.get('kaggle_user', 'anasrazy'); KID = f'{KU}/namma-v3'
OUT = os.path.abspath('out'); T0 = time.time(); RUN = time.strftime('%Y%m%d-%H%M')

def say(*a): print(f'[{(time.time() - T0) / 60:5.1f} min]', *a, flush=True)

def kg(*args, check=True):
    r = subprocess.run(['kaggle', *args], capture_output=True, text=True); out = (r.stdout + r.stderr).strip()
    if check and r.returncode: sys.exit(f'kaggle {" ".join(args[:2])} failed: {out[-800:]}')
    return r.returncode, out

if not cfg.get('n'): say('RUN.json n = 0: nothing to do'); sys.exit(0)
assert os.environ.get('KAGGLE_API_TOKEN'), 'add the repository secret KAGGLE_API_TOKEN (contents of ~/.kaggle/access_token)'
src = cfg['source']; env = dict(cfg.get('env', {})); info = dict(source=src, github_run=RUN)
if cfg.get('collect'):                         # only wait for the version already on Kaggle and collect it (e.g. after a cancelled run)
    say('collect only: no new Kaggle version'); t_push = 0
else:
    if src.startswith('dl3dv:'):
        tok = os.environ.get('HF_TOKEN'); assert tok, 'add the repository secret HF_TOKEN (after clicking Agree on DL3DV/DL3DV-Benchmark)'
        import dl3dv
        from PIL import Image
        scene = src[6:]; rc, st = 1, ''; RES = cfg.get('dl3dv_res', 'images_4')            # images_4 960x540, images_2 1920x1080
        suf = '' if RES == 'images_4' else '-' + RES.replace('images_', 'r')
        if scene != 'auto':                                                                    # already on Kaggle? then no download
            rc, st = kg('datasets', 'status', f'{KU}/namma-dl3dv-{scene[:10]}{suf}', check=False)
        if rc or 'ready' not in st.lower(): scene, folder = dl3dv.fetch(scene, 'dl', tok, RES)
        info['scene'] = scene; slug = f'namma-dl3dv-{scene[:10]}{suf}'; DS = f'{KU}/{slug}'    # one dataset per scene + size: no races
        rc, st = kg('datasets', 'status', DS, check=False)
        if rc or 'ready' not in st.lower():
            files = sorted(glob.glob(f'{folder}/*.png') + glob.glob(f'{folder}/*.jpg')); assert files, f'no frames in {folder}'
            os.makedirs('fr', exist_ok=True); os.makedirs('in', exist_ok=True)
            for f in files: Image.open(f).convert('RGB').save(f'fr/{os.path.splitext(os.path.basename(f))[0]}.jpg', quality=95)
            shutil.make_archive('in/frames', 'zip', 'fr'); say(len(files), 'frames ->', f"{os.path.getsize('in/frames.zip') / 1e6:.0f} MB zip")
            json.dump(dict(title=slug, id=DS, licenses=[dict(name='other')]), open('in/dataset-metadata.json', 'w'))
            say(kg('datasets', 'create', '-p', 'in', '-q')[1][-300:])
            for _ in range(90):                                                                # wait until Kaggle has processed it
                time.sleep(20); st = kg('datasets', 'status', DS, check=False)[1]
                if 'ready' in st.lower() or 'error' in st.lower(): break
            say('dataset', DS, st[-120:])
        else: say('dataset', DS, 'already on Kaggle')
        datasets, env['SOURCE'] = [DS], f'kaggle:{slug}'
    elif src.startswith('dl3dv-video:'):                                                       # the scene's original phone video:
        scene = src[12:]; slug = f'namma-dl3dv-{scene[:10]}-video'; DS = f'{KU}/{slug}'; info['scene'] = scene  # more frames than
        rc, st = kg('datasets', 'status', DS, check=False)                                     # the ~400 DL3DV extracted
        if rc or 'ready' not in st.lower():
            from huggingface_hub import HfApi, hf_hub_download
            tok = os.environ.get('HF_TOKEN'); vr = 'DL3DV/DL3DV-ALL-video'; api = HfApi(token=tok)
            sub = next(k for k in [f'{i}K' for i in range(1, 12)] if api.file_exists(vr, f'{k}/{scene}/video.mp4', repo_type='dataset'))
            os.makedirs('in', exist_ok=True)
            shutil.copy(hf_hub_download(vr, f'{sub}/{scene}/video.mp4', repo_type='dataset', token=tok, local_dir='dl'), 'in/video.mp4')
            subprocess.run('ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration '
                           '-of compact in/video.mp4', shell=True)
            json.dump(dict(title=slug, id=DS, licenses=[dict(name='other')]), open('in/dataset-metadata.json', 'w'))
            say(kg('datasets', 'create', '-p', 'in', '-q')[1][-300:])
            for _ in range(120):
                time.sleep(20); st = kg('datasets', 'status', DS, check=False)[1]
                if 'ready' in st.lower() or 'error' in st.lower(): break
            say('dataset', DS, st[-120:])
        else: say('dataset', DS, 'already on Kaggle')
        datasets, env['SOURCE'] = [DS], f'kaggle:{slug}'
    elif src.startswith('kaggle:'):
        datasets, env['SOURCE'] = [src[7:]], f'kaggle:{src[7:].split("/")[-1]}'
    else: sys.exit(f'unknown source {src}')

    # the Kaggle job = v3/run.py with these settings in front, plus the input dataset
    os.makedirs('kern', exist_ok=True)
    meta = json.load(open(f'{V}/kernel-metadata.json')); meta.update(id=KID, dataset_sources=datasets)
    json.dump(meta, open('kern/kernel-metadata.json', 'w'))
    code = open(f'{V}/run.py').read(); first = 'import glob, json, os, shutil, subprocess, sys, time\n'; assert first in code
    code = code.replace(first, first + f'os.environ.update({json.dumps(env)})  # set by gh_run.py\n', 1)
    open('kern/run.py', 'w').write(code); say('job settings', env, 'inputs', datasets)
    say(kg('kernels', 'push', '-p', 'kern')[1][-400:]); t_push = time.time()

seen, last, s = False, '', ''
while time.time() - T0 < 345 * 60:
    time.sleep(60); rc, s = kg('kernels', 'status', KID, check=False); s = s.lower()
    if rc: say('status check failed:', s[-150:]); continue
    if s != last: say('status:', s[-100:]); last = s
    if 'running' in s or 'queued' in s: seen = True
    if 'error' in s or 'cancel' in s: break
    if 'complete' in s and (seen or time.time() - t_push > 900): break                    # (the previous version also reads 'complete')
ok = 'complete' in s

os.makedirs(OUT, exist_ok=True); kg('kernels', 'output', KID, '-p', OUT, '-o', '--page-size', '200', check=False)
json.dump(info, open(f'{OUT}/github_run.json', 'w'))
for f in ['sfm_report.json', 'v3_summary.json', 'preview/metrics.json', 'full/metrics.json']:
    if os.path.exists(f'{OUT}/{f}'): print(f'==== {f}\n' + open(f'{OUT}/{f}').read(), flush=True)
for lg in glob.glob(f'{OUT}/*.log'): print(f'==== {os.path.basename(lg)} (last 120 lines)\n' + ''.join(open(lg, errors='ignore').readlines()[-120:]), flush=True)
say('files:', sorted(os.path.relpath(f, OUT) for f in glob.glob(f'{OUT}/**/*', recursive=True) if os.path.isfile(f)))

def tour_step():
    """Tour like splattour.com (final/make_tour.py): cameras need gsplat's examples + CPU torch, the converter needs Node.
    Frames come from the run's images/ (saved since run 4) or are rebuilt from the scene's Kaggle dataset like the job did."""
    if not os.path.exists(f'{OUT}/full/model.ply') or not os.path.isdir(f'{OUT}/sparse'): say('tour: no model or cameras'); return
    T = os.path.abspath('tourwork'); FIN = os.path.join(os.path.dirname(V), 'final'); env_ = cfg.get('env', {})
    run = lambda c: subprocess.run(c, shell=True, check=True, executable='/bin/bash')
    run(f'git clone -q --depth 1 -b v1.5.3 https://github.com/nerfstudio-project/gsplat {T}/gsplat && touch {T}/gsplat/examples/datasets/__init__.py')
    run('pip install -q torch --index-url https://download.pytorch.org/whl/cpu && pip install -q opencv-python-headless imageio tqdm plyfile scipy trimesh fast-simplification '
        'typing_extensions "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" && sed -i '
        '"s/np.uint64(-1)/np.uint64(np.iinfo(np.uint64).max)/" $(python -c "import importlib.util as u; print(u.find_spec(\'pycolmap\').submodule_search_locations[0])")/scene_manager.py')
    run(f'cd {T} && npm init -y >/dev/null && npm i -s --no-audit --no-fund playcanvas@2.23.2 @playcanvas/supersplat-viewer@1.37.0 @playcanvas/splat-transform@3.10.1 | tail -1')
    data = f'{T}/data'; os.makedirs(data, exist_ok=True); shutil.copytree(f'{OUT}/sparse', f'{data}/sparse', dirs_exist_ok=True)
    if os.path.isdir(f'{OUT}/images'): os.symlink(f'{OUT}/images', f'{data}/images')
    else:                                                                                  # same frames as the job made (frames.py)
        if src.startswith('dl3dv:') and src[6:] != 'auto':                                   # straight from HF (same frames, same order)
            import dl3dv; dl3dv.fetch(src[6:], f'{T}/ds', os.environ['HF_TOKEN'], cfg.get('dl3dv_res', 'images_4'))
        else:
            rc, o = kg('datasets', 'download', src[7:], '-p', f'{T}/ds', '--unzip', check=False)
            if rc: raise RuntimeError(f'dataset download failed: {o[-300:]}')
            for z in glob.glob(f'{T}/ds/**/*.zip', recursive=True): shutil.unpack_archive(z, f'{T}/ds/unz')
        cnt = {}
        for f in glob.glob(f'{T}/ds/**/*', recursive=True):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')): cnt[os.path.dirname(f)] = cnt.get(os.path.dirname(f), 0) + 1
        sys.path.insert(0, FIN); import frames
        frames.from_images(max(cnt, key=cnt.get), f'{data}/images', int(env_.get('NFRAMES', 300)), int(env_.get('LONG', 1280)))
    run(f'python {FIN}/make_tour.py plan {T}/gsplat/examples {data} {OUT} {OUT}/tour')
    run(f'python {FIN}/make_tour.py build {OUT} {OUT}/tour {T}/node_modules/@playcanvas/supersplat-viewer/public {T}/node_modules/.bin/splat-transform')
    say('tour built:', json.load(open(f'{OUT}/tour/tour.json')))

try: tour_step()
except Exception as e: say('tour failed:', str(e)[:400])
tok = os.environ.get('HF_TOKEN')
if tok:
    try:
        from huggingface_hub import HfApi
        import tour_publish
        api = HfApi(token=tok); repo = f"{api.whoami()['name']}/namma-v3-out"; api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
        api.upload_folder(folder_path=OUT, path_in_repo=RUN, repo_id=repo, repo_type='dataset', ignore_patterns=[f'{n}/model.ply' for n in os.listdir(OUT) if n != 'full'] + ['images/*'])
        say(f'results: https://huggingface.co/datasets/{repo}/tree/main/{RUN}')
        if os.path.exists(f'{OUT}/tour/tour.json'):                          # any built tour (scene.sog or scene.compressed.ply)
            try: tour_publish.publish(api, repo.split('/')[0], repo, RUN, f'{OUT}/tour', say)
            except Exception as e: say('tour publish failed:', str(e)[:300])
        if os.environ.get('GITHUB_STEP_SUMMARY'):
            open(os.environ['GITHUB_STEP_SUMMARY'], 'a').write(f'Results: https://huggingface.co/datasets/{repo}/tree/main/{RUN}\n')
    except Exception as e: say('HF upload failed:', str(e)[:300])
if ok and not os.path.exists(f'{OUT}/full/metrics.json'): ok, s = False, 'complete, but no full model (metrics.json missing)'
sys.exit(0 if ok else f'Kaggle job ended with: {s[-200:]}')
