"""The app's inbox: private HF dataset <hf user>/namma-inbox, one folder per job (job id = YYYYMMDD-HHMMSS-<4 hex>, made by the app):
  <job>/video.json   the recording, uploaded by the app in parts (resumable): {name, size, sha256, parts: [<job>/parts/000, ...]}
                     (a plain <job>/video.mp4 works too, e.g. a manual upload)
  <job>/meta.json    name, phone, camera settings used, preset ("best" default, "quick" = short test run)
  <job>/status.json  stage: uploaded -> training -> publishing -> done | failed, plus run (results folder), link (tour), reason
The app commits video + meta + status 'uploaded' together, then pushes v3/inbox/<job>.json to this branch, which starts
.github/workflows/inbox.yml. That workflow runs `python inbox.py pick`, which decides what to do (one Kaggle job at a time):
  - a job in 'training' whose Kaggle job has ended  -> collect it (it ran longer than one GitHub job)
  - any Kaggle job still running                   -> wait (the next trigger or the app's re-check starts it)
  - else the oldest 'uploaded' job                 -> start it with PIPELINE.json's preset (gh_run.py, source hf:...)
Links stay in the private status.json: nothing here prints them (this repo's GitHub logs are public)."""
import json, os, subprocess, sys, time
V = os.path.dirname(os.path.abspath(__file__))
NAME = 'namma-inbox'; ACTIVE = ('training', 'publishing')


def hub():
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ['HF_TOKEN']); repo = f"{api.whoami()['name']}/{NAME}"
    api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True); return api, repo


def read_json(api, repo, path):
    from huggingface_hub import hf_hub_download
    try: return json.load(open(hf_hub_download(repo, path, repo_type='dataset', token=api.token, force_download=True, local_dir='inbox_dl')))
    except Exception: return {}


def set_status(job, stage, **kw):
    """Update <job>/status.json (keeps the stage history). kw: run, link, reason, note."""
    api, repo = hub(); st = read_json(api, repo, f'{job}/status.json'); now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    if stage != 'failed': st.pop('reason', None)                                     # an old failure reason must not outlive a retry
    st.update(kw, stage=stage, updated=now); st.setdefault('history', []).append([now, stage])
    api.upload_file(path_or_fileobj=json.dumps(st, indent=1).encode(), path_in_repo=f'{job}/status.json', repo_id=repo,
                    repo_type='dataset', commit_message=f'{job}: {stage}')
    print('status', job, '->', stage, flush=True)


def assemble(repo, manifest_path, dst, token):
    """The app uploads a video as parts (resumable, each its own file): <job>/video.json = {name, size, sha256, parts: [paths]}.
    Download the parts, join them into dst and check size and sha256."""
    import hashlib
    from huggingface_hub import hf_hub_download
    m = json.load(open(hf_hub_download(repo, manifest_path, repo_type='dataset', token=token, local_dir='inbox_dl')))
    h = hashlib.sha256(); n = 0
    with open(dst, 'wb') as out:
        for p in m['parts']:
            b = open(hf_hub_download(repo, p, repo_type='dataset', token=token, local_dir='inbox_dl'), 'rb').read(); out.write(b); h.update(b); n += len(b)
            os.remove(os.path.join('inbox_dl', p))
    assert n == m['size'] and h.hexdigest() == m['sha256'], f'video parts do not add up ({n} of {m["size"]} bytes, sha256 differs)'
    return m


def jobs(api, repo):
    """{job: status} for every job folder that has a status.json, oldest job first (ids start with the time)."""
    files = api.list_repo_files(repo, repo_type='dataset')
    return {j: read_json(api, repo, f'{j}/status.json') for j in sorted({f.split('/')[0] for f in files if f.endswith('/status.json')})}


def kaggle_busy(kernels):
    """Kernels (owner/slug) of ours that are running or queued on Kaggle right now."""
    busy = []
    for k in kernels:
        r = subprocess.run(['kaggle', 'kernels', 'status', k], capture_output=True, text=True); s = (r.stdout + r.stderr).lower()
        if r.returncode == 0 and ('running' in s or 'queued' in s): busy.append(k)
    return busy


def run_settings(api, repo, job, cfg):
    """gh_run settings for this job: the preset named in meta.json (only presets in PIPELINE.json; default 'best')."""
    meta = read_json(api, repo, f'{job}/meta.json'); name = meta.get('preset') if meta.get('preset') in cfg['presets'] else cfg['default']
    files = api.list_repo_files(repo, repo_type='dataset')
    video = f'{job}/video.json' if f'{job}/video.json' in files else next((f for f in files if f.startswith(f'{job}/video.')), f'{job}/video.mp4')
    p = cfg['presets'][name]
    return dict(n=1, collect=False, source=f'hf:{repo}:{video}', kernel=cfg['kernel'], env=p['env'], dl3dv_res='images_2',
                preset=name, scene=f'inbox job {job}')


def pick(collect_only=False):
    """Decide the next step for the workflow; writes job_run.json and GITHUB_OUTPUT action=start|collect|wait, job=<id>."""
    cfg = json.load(open(f'{V}/PIPELINE.json')); api, repo = hub(); allj = jobs(api, repo)
    import glob                                                   # a trigger file with "retry": true sends a failed job round again
    for f in glob.glob(f'{V}/inbox/*.json'):
        t = json.load(open(f)); j = t.get('job')
        if t.get('retry') and allj.get(j, {}).get('stage') == 'failed':
            set_status(j, 'uploaded', note='retry'); allj[j]['stage'] = 'uploaded'
    sys.path.insert(0, V); import kaggle_io
    ku = kaggle_io.account({}); mine = f"{ku}/{cfg['kernel']}"
    act, job = 'wait', ''
    running = [j for j, s in allj.items() if s.get('stage') in ACTIVE]
    if running:                                                                               # a job already on Kaggle
        job = running[0]; act = 'collect' if collect_only or not kaggle_busy([mine]) else 'wait'
    elif not collect_only:
        waiting = [j for j, s in allj.items() if s.get('stage') == 'uploaded']
        busy = kaggle_busy([mine] + [f'{ku}/{k}' for k in cfg.get('other_kernels', [])])
        if waiting and not busy: job, act = waiting[0], 'start'
        elif waiting: print(len(waiting), 'job(s) waiting: a Kaggle job is still running', flush=True)
    if job:
        json.dump(run_settings(api, repo, job, cfg) | dict(collect=act == 'collect'), open('job_run.json', 'w'), indent=1)
    print('inbox:', len(allj), 'jobs; action', act, job, flush=True)
    if os.environ.get('GITHUB_OUTPUT'): open(os.environ['GITHUB_OUTPUT'], 'a').write(f'action={act}\njob={job}\n')
    return act, job


if __name__ == '__main__':
    if sys.argv[1:2] == ['pick']: pick('--collect' in sys.argv)
    elif sys.argv[1:2] == ['status']: set_status(sys.argv[2], sys.argv[3], **dict(a.split('=', 1) for a in sys.argv[4:]))
    else: sys.exit('usage: inbox.py pick [--collect] | status <job> <stage> [key=value ...]')
