"""Tour fidelity check on a GitHub runner, CPU only (.github/workflows/tour-fidelity.yml, settings in v3/FIDELITY.json): the run's
trained model vs what the tour ships (smear filter, SH band cut, compressed PLY), each decoded back and rendered from held-out
frames by final/tour_fidelity.py -> hf://datasets/<user>/namma-v3-out/fidelity/<time>/ (fidelity_report.json, fidelity_sheet.jpg)."""
import json, os, shutil, subprocess, sys, time
V = os.path.dirname(os.path.abspath(__file__)); FIN = os.path.join(os.path.dirname(V), 'final'); sys.path.insert(0, V); sys.path.insert(0, FIN)
cfg = json.load(open(f'{V}/FIDELITY.json')); OUT = os.path.abspath('out'); T0 = time.time(); STAMP = time.strftime('%Y%m%d-%H%M')
run = lambda c: subprocess.run(c, shell=True, check=True, executable='/bin/bash')

def say(*a): print(f'[{(time.time() - T0) / 60:5.1f} min]', *a, flush=True)

if not cfg.get('n'): say('FIDELITY.json n = 0: nothing to do'); sys.exit(0)
tok = os.environ.get('HF_TOKEN'); assert tok, 'needs HF_TOKEN'
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=tok); user = api.whoami()['name']; repo = f'{user}/namma-v3-out'; R = cfg['run']
d = snapshot_download(repo, repo_type='dataset', local_dir='dl', token=tok,
                      allow_patterns=[f'{R}/full/model.ply', f'{R}/tour/plan.json', f'{R}/tour/tour.json', f'{R}/sparse/**']
                      + [f'{R}/{m}' for m in cfg.get('compare', [])] + ([f"{R}/{cfg['unneedle']}"] if cfg.get('unneedle') else []))
mpu = json.load(open(f'{d}/{R}/tour/tour.json'))['metres_per_unit']
T = os.path.abspath('work'); EX = f'{T}/gsplat/examples'; data = f'{T}/data'; os.makedirs(data, exist_ok=True); os.makedirs(OUT, exist_ok=True)
run(f'git clone -q --depth 1 -b v1.5.3 https://github.com/nerfstudio-project/gsplat {T}/gsplat && touch {EX}/datasets/__init__.py')
run('pip install -q torch --index-url https://download.pytorch.org/whl/cpu && pip install -q opencv-python-headless imageio tqdm plyfile scipy '
    'typing_extensions "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" && sed -i '
    '"s/np.uint64(-1)/np.uint64(np.iinfo(np.uint64).max)/" $(python -c "import importlib.util as u; print(u.find_spec(\'pycolmap\').submodule_search_locations[0])")/scene_manager.py')
shutil.copytree(f'{d}/{R}/sparse', f'{data}/sparse', dirs_exist_ok=True)
if cfg.get('unneedle'):                                    # the cupboard fix on a finished model, no training (final/solidify.unneedle_file)
    src = f"{d}/{R}/{cfg['unneedle']}"; rep_ = {}
    import model_compare; model_compare.placeholders(data)                    # the Parser needs frame files: stand-ins of the right size
    for ratio in cfg.get('ratios', [4, 6]):                  # strongest first; keep it only if the filmed views keep their look
        out_ = f'{T}/un{ratio}.ply'; run(f'cd {EX} && PYTHONPATH={EX}:{FIN}:$PYTHONPATH python {FIN}/solidify.py unneedle {data} {src} {out_} {ratio}')
        sub = f'{OUT}/ratio{ratio}'
        run(f'cd {EX} && PYTHONPATH={EX}:{FIN}:$PYTHONPATH python {FIN}/model_compare.py {data} {sub} {mpu} {cfg.get("px", 480)} {cfg.get("views", 40)} model={src} unneedled={out_}')
        cr = json.load(open(f'{sub}/compare_report.json')); rep_[ratio] = {k: cr[k] for k in ('held_out', 'off_path')}
        if cr['held_out']['psnr_mean'] >= 33 and cr['held_out']['psnr_min'] >= 28:
            api.upload_file(path_or_fileobj=out_, path_in_repo=f'{R}/solid/unneedled.ply', repo_id=repo, repo_type='dataset')
            rep_['chosen'] = ratio; break
    json.dump(rep_, open(f'{OUT}/unneedle_report.json', 'w'), indent=1); say('UNNEEDLE', json.dumps(rep_))
    dst = f'fidelity/{STAMP}'; api.upload_folder(folder_path=OUT, path_in_repo=dst, repo_id=repo, repo_type='dataset')
    say(f'results: hf://datasets/{repo}/{dst}'); sys.exit(0)
if cfg.get('compare'):                                     # two models of the run vs each other, no photos (final/model_compare.py)
    a_, b_ = cfg['compare']; nm = lambda m: m.split('/')[0]
    r = subprocess.run(f'cd {EX} && PYTHONPATH={EX}:{FIN}:$PYTHONPATH python {FIN}/model_compare.py {data} {OUT} {mpu} {cfg.get("px", 480)} '
                       f'{cfg.get("views", 40)} {nm(a_)}={d}/{R}/{a_} {nm(b_)}={d}/{R}/{b_} 2>&1 | tee {OUT}/compare.log', shell=True, executable='/bin/bash').returncode
    dst = f'fidelity/{STAMP}'; api.upload_folder(folder_path=OUT, path_in_repo=dst, repo_id=repo, repo_type='dataset')
    say(f'results: hf://datasets/{repo}/{dst}'); sys.exit(0 if os.path.exists(f'{OUT}/compare_report.json') else 'model_compare.py failed')
run(f'cd {T} && npm init -y >/dev/null && npm i -s --no-audit --no-fund @playcanvas/splat-transform@3.10.1 | tail -1'); st = f'{T}/node_modules/.bin/splat-transform -w -g cpu'
shutil.copytree(f'{d}/{R}/sparse', f'{data}/sparse', dirs_exist_ok=True)
import dl3dv, frames                                                                      # the run's frames, same names and order
if cfg.get('video'):                                       # runs made from the original phone video (run 6): same sharpest frames
    from huggingface_hub import hf_hub_download
    vr = 'DL3DV/DL3DV-ALL-video'; sub = next(k for k in [f'{i}K' for i in range(1, 12)] if api.file_exists(vr, f'{k}/{cfg["scene"]}/video.mp4', repo_type='dataset'))
    vid = hf_hub_download(vr, f'{sub}/{cfg["scene"]}/video.mp4', repo_type='dataset', token=tok, local_dir=f'{T}/vid')
    say('frames', frames.from_video(vid, f'{data}/images', int(cfg.get('nframes', 900)), int(cfg.get('long', 1920))))
else:
    _, folder = dl3dv.fetch(cfg['scene'], f'{T}/ds', tok, cfg.get('dl3dv_res', 'images_2'))
    say('frames', frames.from_images(folder, f'{data}/images', int(cfg.get('nframes', 420)), int(cfg.get('long', 1440))))
M = f'{T}/m'; os.makedirs(M, exist_ok=True); model = f'{d}/{R}/full/model.ply'
variants = [('trained', model)]; sizes = {'trained': os.path.getsize(model)}

def shipped(name, src, sh):                                       # what the tour would ship: compressed PLY, decoded back
    run(f'{st} {src} -H {sh} -N {M}/{name}.compressed.ply && {st} {M}/{name}.compressed.ply {M}/{name}.ply')
    sizes[name] = os.path.getsize(f'{M}/{name}.compressed.ply'); variants.append((name, f'{M}/{name}.ply'))

if cfg.get('solidify'):                                           # no-training clean-up variants (final/solidify.py)
    shipped('tour_now', model, 1)                                 # the tour as built since 30 Oct (no crop, no smear filter)
    S = f'{OUT}/solidify'
    run(f'cd {EX} && PYTHONPATH={EX}:{FIN}:$PYTHONPATH python {FIN}/solidify.py {data} {model} {M} {cfg.get("solid_px", 480)} {cfg.get("solid_step", 3)} 2>&1 | tee {OUT}/solidify.log')
    shutil.copy(f'{M}/solidify.json', OUT)
    variants += [('clean', f'{M}/clean.ply')]
    shipped('ship_clean_sh3', f'{M}/clean.ply', 3)                # what the tour ships from now: cleaned, all colour bands
    for f in ('clean',):                                          # the winner (run 4: no loss, 27 % fewer splats): the tour is built from it
        api.upload_file(path_or_fileobj=f'{M}/{f}.ply', path_in_repo=f'{R}/solid/{f}.ply', repo_id=repo, repo_type='dataset')
else:
    run(f'python {FIN}/make_tour.py clean {d}/{R} {d}/{R}/tour {M}/clean.ply')
    variants.append(('smears_off', f'{M}/clean.ply'))
    for name, sh in (('tour_sh1', 1), ('cply_sh3', 3), ('cply_sh0', 0)):
        shipped(name, f'{M}/clean.ply', sh)
json.dump(sizes, open(f'{OUT}/sizes.json', 'w')); say('sizes MB', {k: round(v / 1e6, 1) for k, v in sizes.items()})
r = subprocess.run(f'cd {EX} && PYTHONPATH={EX}:{FIN}:$PYTHONPATH python {FIN}/tour_fidelity.py {data} {OUT} {mpu} {cfg.get("px", 640)} {cfg.get("views", 20)} '
                   + ' '.join(f'{n}={p}' for n, p in variants) + f' 2>&1 | tee {OUT}/fidelity.log', shell=True, executable='/bin/bash').returncode
dst = f'fidelity/{STAMP}'; api.upload_folder(folder_path=OUT, path_in_repo=dst, repo_id=repo, repo_type='dataset')
say(f'results: hf://datasets/{repo}/{dst}')
if not os.path.exists(f'{OUT}/fidelity_report.json'): sys.exit('tour_fidelity.py failed')
