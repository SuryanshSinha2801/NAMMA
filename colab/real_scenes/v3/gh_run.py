"""GitHub Actions side of namma-v3 (.github/workflows/namma-v3.yml). The cloud session that writes the code cannot reach kaggle.com;
GitHub's runners can. input -> private Kaggle dataset (DL3DV frames are fetched here, so the HF token stays a GitHub secret)
-> push the Kaggle job (v3/run.py) -> wait -> download its outputs -> private HF dataset <hf user>/namma-v3-out/<run> + metrics here.
RUN.json: n         run counter: change it to start a run (0 = do nothing)
          collect   true: push nothing, just wait for the job already on Kaggle and collect its outputs
          source    dl3dv:auto | dl3dv:<scene hash> | dl3dv-video:<hash> | kaggle:<owner>/<dataset> (a video or frames already
                    on Kaggle) | hf:<owner>/<dataset>:<path of a video> (private HF dataset, e.g. the app's inbox)
          kaggle_user, env (overrides run.py settings, e.g. {"STEPS": "30000", "PREVIEW": "7000"})
Secrets: KAGGLE_API_TOKEN (contents of ~/.kaggle/access_token), HF_TOKEN (write; "Agree" clicked on DL3DV/DL3DV-Benchmark)."""
import glob, json, os, shutil, subprocess, sys, time
V = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, V)
cfg = json.load(open(os.environ.get('RUN_JSON') or f'{V}/RUN.json'))                     # RUN_JSON: the inbox runner's settings
import kaggle_io                                                                              # the Kaggle account is the token's own
KU = kaggle_io.account(cfg); KID = f"{KU}/{cfg.get('kernel', 'namma-v3')}"                      # inbox jobs: their own kernel
if os.environ.get('COLLECT'): cfg['collect'] = True                                       # the workflow's follow-up jobs
OUT = os.path.abspath('out'); T0 = time.time(); RUN = time.strftime('%Y%m%d-%H%M')

def say(*a): print(f'[{(time.time() - T0) / 60:5.1f} min]', *a, flush=True)

def stage(name, **kw):
    """Progress for the app: <inbox>/<job>/status.json, only when started by the inbox runner (INBOX_JOB set)."""
    if os.environ.get('INBOX_JOB'):
        try: import inbox; inbox.set_status(os.environ['INBOX_JOB'], name, **kw)
        except Exception as e: say('status update failed:', str(e)[:200])

def lfs_types(api, repo, kinds=('splat', 'ply', 'glb', 'spz', 'sog', 'bin', 'npz', 'mp4', 'jpg', 'png', 'webp')):
    """The Hub now rejects small binary files sent as plain git blobs (28 Oct: a quick run's 1 MB model.splat). Files of these
    types go to large-file storage whatever their size."""
    from huggingface_hub import hf_hub_download
    try: ga = open(hf_hub_download(repo, '.gitattributes', repo_type='dataset', token=api.token, force_download=True, local_dir='ga')).read()
    except Exception: ga = ''
    add = [f'*.{k} filter=lfs diff=lfs merge=lfs -text' for k in kinds if f'*.{k} filter=lfs' not in ga]
    if add: api.upload_file(path_or_fileobj=(ga.rstrip('\n') + '\n' + '\n'.join(add) + '\n').encode(), path_in_repo='.gitattributes',
                            repo_id=repo, repo_type='dataset', commit_message='large-file storage for splat/model/media types')

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
            say(kg('datasets', 'create', '-p', 'in', '-q')[1][-300:]); kaggle_io.wait_ready(DS, say)
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
            shutil.move(hf_hub_download(vr, f'{sub}/{scene}/video.mp4', repo_type='dataset', token=tok, local_dir='dl'), 'in/video.mp4')   # move, not copy: an 8 GB video twice fills the runner
            pr = subprocess.run('ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration,size '
                                '-of json in/video.mp4', shell=True, capture_output=True, text=True); print(pr.stdout[-1500:], flush=True)
            try: info['ffprobe'] = json.loads(pr.stdout)
            except Exception: pass
            json.dump(dict(title=slug, id=DS, licenses=[dict(name='other')]), open('in/dataset-metadata.json', 'w'))
            say(kg('datasets', 'create', '-p', 'in', '-q')[1][-300:]); kaggle_io.wait_ready(DS, say)
        else: say('dataset', DS, 'already on Kaggle')
        datasets, env['SOURCE'] = [DS], f'kaggle:{slug}'
    elif src.startswith('hf:'):                                                               # a video in a private HF dataset (the app's inbox):

        repo_, path_ = src[3:].split(':', 1)

        # Use the actual manifest filename as the job ID when INBOX_JOB is absent.
        manifest_name = os.path.basename(path_.rstrip('/'))

        if os.environ.get('INBOX_JOB'):
            job = os.environ['INBOX_JOB']
        elif manifest_name.endswith('.json'):
            job = os.path.splitext(manifest_name)[0]
        else:
            job = path_.strip('/').split('/')[0]

        slug = ('namma-in-' + ''.join(
            c for c in job.lower() if c.isalnum() or c == '-'
        ))[:50]
        DS = f'{KU}/{slug}'
        info['inbox_job'] = job

        rc, st = kg('datasets', 'status', DS, check=False)
        if rc or not st.strip().lower().endswith('ready'):
            from huggingface_hub import hf_hub_download
            os.makedirs('in', exist_ok=True); ext = os.path.splitext(path_)[1] or '.mp4'
            if path_.endswith('.json'):                                                  # the app's parts -> one video
                import inbox; m = inbox.assemble(repo_, path_, 'in/video', os.environ.get('HF_TOKEN')); ext = os.path.splitext(m.get('name', ''))[1] or '.mp4'
                os.rename('in/video', f'in/video{ext}')
            else: shutil.move(hf_hub_download(repo_, path_, repo_type='dataset', token=os.environ.get('HF_TOKEN'), local_dir='dl'), f'in/video{ext}')
            say('video', path_, f"{os.path.getsize(f'in/video{ext}') / 1e6:.0f} MB")
            json.dump(dict(title=slug, id=DS, licenses=[dict(name='other')]), open('in/dataset-metadata.json', 'w'))
            say(kg('datasets', 'create', '-p', 'in', '-q')[1][-300:]); kaggle_io.wait_ready(DS, say)
        datasets, env['SOURCE'] = [DS], f'kaggle:{slug}'
    elif src.startswith('reuse:'):                                                            # an earlier run's frames + camera model
        datasets, env['REUSE'] = [], '1'; info['reuse'] = src[6:]                                # (its Kaggle output; own account)
    elif src.startswith('kaggle:'):
        datasets, env['SOURCE'] = [src[7:]], f'kaggle:{src[7:].split("/")[-1]}'
    else: sys.exit(f'unknown source {src}')

    # the Kaggle job = v3/run.py with these settings in front, plus the input dataset
    os.makedirs('kern', exist_ok=True)
    meta = json.load(open(f'{V}/kernel-metadata.json')); meta.update(id=KID, title=KID.split('/')[1], dataset_sources=datasets)
    meta['kernel_sources'] = [k for k in meta.get('kernel_sources', []) if k.startswith(KU + '/')]   # another account's build cache: not usable
    if src.startswith('reuse:'):
        assert f'{KU}/{src[6:]}' != KID, 'reuse: the job must not overwrite the run it reuses (use another kernel name)'
        meta['kernel_sources'].append(f'{KU}/{src[6:]}')
    json.dump(meta, open('kern/kernel-metadata.json', 'w'))
    code = open(f'{V}/run.py').read(); first = 'import glob, json, os, shutil, subprocess, sys, time\n'; assert first in code
    code = code.replace(first, first + f'os.environ.update({json.dumps(env)})  # set by gh_run.py\n', 1)
    open('kern/run.py', 'w').write(code); say('job settings', env, 'inputs', datasets)
    say(kg('kernels', 'push', '-p', 'kern')[1][-400:]); t_push = time.time(); stage('training')

seen, last, s = False, '', ''
while time.time() - T0 < 345 * 60:
    time.sleep(60); rc, s = kg('kernels', 'status', KID, check=False); s = s.lower()
    if rc: say('status check failed:', s[-150:]); continue
    if s != last: say('status:', s[-100:]); last = s
    if 'running' in s or 'queued' in s: seen = True
    if 'error' in s or 'cancel' in s: break
    if 'complete' in s and (seen or time.time() - t_push > 900): break                    # (the previous version also reads 'complete')
ok = 'complete' in s
if 'running' in s or 'queued' in s:                     # longer than one GitHub job (6 h): the workflow's next job collects it
    say('still running on Kaggle after', round((time.time() - T0) / 60), 'min: the next job collects it'); stage('training', note='still training')
    if os.environ.get('GITHUB_OUTPUT'): open(os.environ['GITHUB_OUTPUT'], 'a').write('still_running=true\n')
    sys.exit(0)

os.makedirs(OUT, exist_ok=True); kg('kernels', 'output', KID, '-p', OUT, '-o', '--page-size', '200', check=False)
json.dump(info, open(f'{OUT}/github_run.json', 'w'))
for f in ['sfm_report.json', 'precheck.json', 'v3_summary.json', 'preview/metrics.json', 'full/metrics.json']:
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
    # clean-up without training (final/solidify.py): drop splats no training view shows (run 4: 27 %, run 6: 59 %, both 0.00 dB)
    try:
        run(f'cd {T}/gsplat/examples && SOLID_ONLY_CLEAN=1 PYTHONPATH={T}/gsplat/examples:{FIN}:$PYTHONPATH python {FIN}/solidify.py {data} {OUT}/full/model.ply {OUT}/solid 480 3')
        os.environ['TOUR_MODEL'] = 'solid/clean.ply'
    except Exception as e: say('clean-up failed, the tour uses the trained model as is:', str(e)[:200])
    run(f'python {FIN}/make_tour.py build {OUT} {OUT}/tour {T}/node_modules/@playcanvas/supersplat-viewer/public {T}/node_modules/.bin/splat-transform')
    say('tour built:', json.load(open(f'{OUT}/tour/tour.json')))

stage('publishing')
tok = os.environ.get('HF_TOKEN'); api = repo = None
skip = lambda: [f'{n}/model.ply' for n in os.listdir(OUT) if n not in ('full', 'prior')] + ['images/*', 'sfm_work/*', 'tour/*']
if tok:                                       # the trained model goes to HF FIRST: a slow or failed tour build can never lose it
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=tok); repo = f"{api.whoami()['name']}/namma-v3-out"; api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
        lfs_types(api, repo); api.upload_folder(folder_path=OUT, path_in_repo=RUN, repo_id=repo, repo_type='dataset', ignore_patterns=skip())
        say(f'results (before the tour): https://huggingface.co/datasets/{repo}/tree/main/{RUN}')
    except Exception as e: say('HF upload failed:', str(e)[:300])
try: tour_step()
except Exception as e: say('tour failed:', str(e)[:400])
if tok:
    try:
        from huggingface_hub import HfApi
        import tour_publish
        api = api or HfApi(token=tok); repo = repo or f"{api.whoami()['name']}/namma-v3-out"
        api.upload_folder(folder_path=OUT, path_in_repo=RUN, repo_id=repo, repo_type='dataset', ignore_patterns=[x for x in skip() if x != 'tour/*'])
        say(f'results: https://huggingface.co/datasets/{repo}/tree/main/{RUN}')
        if os.path.exists(f'{OUT}/tour/tour.json'):                          # any built tour (scene.sog or scene.compressed.ply)
            try:
                tour_publish.publish(api, repo.split('/')[0], repo, RUN, f'{OUT}/tour', say)
                if os.environ.get('INBOX_JOB'):                                               # the link, privately, for the app
                    stage('done', run=RUN, link=open('tour_url.txt').read().strip())
            except Exception as e: say('tour publish failed:', str(e)[:300])
        if os.environ.get('GITHUB_STEP_SUMMARY'):
            open(os.environ['GITHUB_STEP_SUMMARY'], 'a').write(f'Results: https://huggingface.co/datasets/{repo}/tree/main/{RUN}\n')
    except Exception as e: say('HF upload failed:', str(e)[:300])
if cfg.get('env', {}).get('PREP_ONLY') == '1':                                       # whole-house pre-check: no model expected
    if ok and not os.path.exists(f'{OUT}/precheck.json'): ok, s = False, 'complete, but no precheck.json'
elif ok and not os.path.exists(f'{OUT}/full/metrics.json'): ok, s = False, 'complete, but no full model (metrics.json missing)'
if not ok: stage('failed', reason=f'Kaggle job ended with: {s[-200:]}')
elif os.environ.get('INBOX_JOB') and not os.path.exists('tour_url.txt'): stage('failed', reason='no tour was published (see the GitHub log)')
sys.exit(0 if ok else f'Kaggle job ended with: {s[-200:]}')
