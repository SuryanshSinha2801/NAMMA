"""Small Kaggle helpers for the GitHub-side runners (the same steps world_gh.py does inline): the token's own account, a flat input
dataset (create or new version, wait until ready), push a script kernel with its settings prepended, wait for it, collect."""
import json, os, subprocess, sys, time


def kg(*args, check=True):
    r = subprocess.run(['kaggle', *args], capture_output=True, text=True); out = (r.stdout + r.stderr).strip()
    if check and r.returncode: sys.exit(f'kaggle {" ".join(args[:2])} failed: {out[-800:]}')
    return r.returncode, out


def account(cfg):
    """The account the KAGGLE_API_TOKEN belongs to (a new token can be a new account), unless the settings name one."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi(); a.authenticate(); return cfg.get('kaggle_user') or a.config_values['username']


def push_dataset(folder, ds, title, msg, say=print):
    """Flat folder -> private Kaggle dataset ds (owner/slug); returns when Kaggle says ready (+1 min to attach)."""
    json.dump(dict(title=title, id=ds, licenses=[dict(name='other')]), open(f'{folder}/dataset-metadata.json', 'w'))
    rc, st = kg('datasets', 'status', ds, check=False)
    say(kg('datasets', 'version', '-p', folder, '-m', msg, '-q')[1][-300:] if rc == 0 and 'ready' in st.lower() else kg('datasets', 'create', '-p', folder, '-q')[1][-300:])
    wait_ready(ds, say)


def wait_ready(ds, say=print, minutes=60):
    """Wait until Kaggle has processed dataset ds. Only the plain word 'ready' counts: right after create, status calls can
    answer with an error text (not found yet) - run 6's first try pushed the job then and failed ("not valid dataset sources")."""
    t0, st = time.time(), ''
    while time.time() - t0 < minutes * 60:
        time.sleep(20); rc, st = kg('datasets', 'status', ds, check=False); w = st.strip().lower().split()[-1:] or ['']
        if rc == 0 and w[0] == 'ready': break
        if rc == 0 and w[0] in ('error', 'failed') and time.time() - t0 > 300: break
    say('dataset', ds, st[-120:]); assert st.strip().lower().endswith('ready'), f'{ds} not ready: {st[-200:]}'
    time.sleep(60)                                                                            # a fresh version can need a minute more to attach


def push_kernel(kid, title, script, cfg, datasets, say=print, first='import base64, gc, glob, io, json, os, shutil, subprocess, sys, time\n'):
    """script (path) -> Kaggle script kernel kid on 2x T4 with internet; cfg goes in as the JOB_CFG environment variable."""
    os.makedirs('kern', exist_ok=True); name = os.path.basename(script)
    json.dump(dict(id=kid, title=title, code_file=name, language='python', kernel_type='script', is_private=True, enable_gpu=True,
                   enable_internet=True, machine_shape='NvidiaTeslaT4', dataset_sources=datasets), open('kern/kernel-metadata.json', 'w'))
    code = open(script).read(); assert code.startswith(first), f'{name} must start with: {first!r}'
    open(f'kern/{name}', 'w').write(code.replace(first, first + f'os.environ["JOB_CFG"] = {json.dumps(json.dumps(cfg))}  # set by the runner\n', 1))
    say(kg('kernels', 'push', '-p', 'kern')[1][-400:])


def wait_kernel(kid, out, say=print, minutes=240, already_running=False):
    """Poll until the kernel ends (a status can still show the previous version's 'complete' just after a push), then download
    its outputs into out. Returns the last status text."""
    t0 = time.time(); seen, last, s = already_running, '', ''
    while time.time() - t0 < minutes * 60:
        time.sleep(60); rc, s = kg('kernels', 'status', kid, check=False); s = s.lower()
        if rc: say('status check failed:', s[-150:]); continue
        if s != last: say('status:', s[-100:]); last = s
        if 'running' in s or 'queued' in s: seen = True
        if 'error' in s or 'cancel' in s: break
        if 'complete' in s and (seen or time.time() - t0 > 900): break
    os.makedirs(out, exist_ok=True); kg('kernels', 'output', kid, '-p', out, '-o', check=False); return s
