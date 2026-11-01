"""GitHub Actions side of the Qwen-Image-Edit smoke test (.github/workflows/qwen-smoke.yml, settings in v3/QWEN.json).
run's model.ply + sparse/ (private HF results dataset) -> private Kaggle dataset namma-qwen-in -> Kaggle job anasrazy/namma-qwen
(final/qwen_smoke.py, with the run's frames dataset attached) -> wait -> outputs -> hf://datasets/<user>/namma-v3-out/qwen_smoke/<time>/.
Only one Kaggle GPU job at a time: start it when no namma-v3 run is training."""
import glob, json, os, shutil, subprocess, sys, time
V = os.path.dirname(os.path.abspath(__file__)); FIN = os.path.join(os.path.dirname(V), 'final')
cfg = json.load(open(f'{V}/QWEN.json')); KU = cfg.get('kaggle_user', 'anasrazy'); KID = f'{KU}/namma-qwen'; DS = f'{KU}/namma-qwen-in'
OUT = os.path.abspath('out'); T0 = time.time(); STAMP = time.strftime('%Y%m%d-%H%M')

def say(*a): print(f'[{(time.time() - T0) / 60:5.1f} min]', *a, flush=True)

def kg(*args, check=True):
    r = subprocess.run(['kaggle', *args], capture_output=True, text=True); out = (r.stdout + r.stderr).strip()
    if check and r.returncode: sys.exit(f'kaggle {" ".join(args[:2])} failed: {out[-800:]}')
    return r.returncode, out

if not cfg.get('n'): say('QWEN.json n = 0: nothing to do'); sys.exit(0)
tok = os.environ.get('HF_TOKEN'); assert tok and os.environ.get('KAGGLE_API_TOKEN'), 'needs HF_TOKEN and KAGGLE_API_TOKEN secrets'
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=tok); repo = f"{api.whoami()['name']}/namma-v3-out"; run = cfg['run']
d = snapshot_download(repo, repo_type='dataset', allow_patterns=[f'{run}/full/model.ply', f'{run}/sparse/**'], local_dir='dl', token=tok)
os.makedirs('in', exist_ok=True); shutil.copy(f'{d}/{run}/full/model.ply', 'in/model.ply')
shutil.make_archive('in/sparse', 'zip', f'{d}/{run}', 'sparse'); say('inputs', os.listdir('in'), round(os.path.getsize('in/model.ply') / 1e6), 'MB ply')
json.dump(dict(title='namma-qwen-in', id=DS, licenses=[dict(name='other')]), open('in/dataset-metadata.json', 'w'))
rc, st = kg('datasets', 'status', DS, check=False)
say(kg('datasets', 'version', '-p', 'in', '-m', f'{run} {STAMP}', '-q')[1][-300:] if rc == 0 and 'ready' in st.lower()
    else kg('datasets', 'create', '-p', 'in', '-q')[1][-300:])
for _ in range(90):
    time.sleep(20); st = kg('datasets', 'status', DS, check=False)[1]
    if 'ready' in st.lower() or 'error' in st.lower(): break
say('dataset', DS, st[-120:])

os.makedirs('kern', exist_ok=True)
meta = dict(id=KID, title='namma-qwen', code_file='qwen_smoke.py', language='python', kernel_type='script', is_private=True,
            enable_gpu=True, enable_internet=True, machine_shape='NvidiaTeslaT4', kernel_sources=[f'{KU}/namma-prep'],
            dataset_sources=[DS, cfg['frames_dataset']])
json.dump(meta, open('kern/kernel-metadata.json', 'w'))
code = open(f'{FIN}/qwen_smoke.py').read(); first = 'import gc, glob, json, os, shutil, subprocess, sys, time\n'; assert first in code
open('kern/qwen_smoke.py', 'w').write(code.replace(first, first + f'os.environ["QWEN"] = {json.dumps(json.dumps(cfg))}  # set by qwen_gh.py\n', 1))
say(kg('kernels', 'push', '-p', 'kern')[1][-400:]); t_push = time.time()

seen, last, s = False, '', ''
while time.time() - T0 < 200 * 60:
    time.sleep(60); rc, s = kg('kernels', 'status', KID, check=False); s = s.lower()
    if rc: say('status check failed:', s[-150:]); continue
    if s != last: say('status:', s[-100:]); last = s
    if 'running' in s or 'queued' in s: seen = True
    if 'error' in s or 'cancel' in s: break
    if 'complete' in s and (seen or time.time() - t_push > 900): break
os.makedirs(OUT, exist_ok=True); kg('kernels', 'output', KID, '-p', OUT, '-o', check=False)
for f in [f'{OUT}/qwen_report.json'] + glob.glob(f'{OUT}/*.log'):
    if os.path.exists(f): print(f'==== {os.path.basename(f)}\n' + ''.join(open(f, errors='ignore').readlines()[-150:]), flush=True)
dst = f'qwen_smoke/{STAMP}'; api.upload_folder(folder_path=OUT, path_in_repo=dst, repo_id=repo, repo_type='dataset')
say(f'results: hf://datasets/{repo}/{dst}'); sys.exit(0 if 'complete' in s else f'Kaggle job ended with: {s[-200:]}')
