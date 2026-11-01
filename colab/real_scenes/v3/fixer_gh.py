"""NVIDIA Fixer smoke test (.github/workflows/fixer.yml, settings in v3/FIXER.json; plan of 30 Oct, run 4 only).
GitHub CPU runner: render the run's cleaned model (<run>/solid/clean.ply) from tour-like views off the filmed path (1 m to the side,
turned up to 50 deg, raised; final/cpu_render.py) plus 2 held-out filmed views -> private Kaggle dataset namma-fixer-in ->
Kaggle T4 job namma-fixer (final/fixer_smoke.py) fixes them -> before/after sheet -> hf://datasets/<user>/namma-v3-out/fixer/<time>/.
mode grade (owner, 30 Oct: true colours but no retraining, which made walls blotchy): fit the images_2 -> video colour
polynomial and apply it straight to the clean model file (grade_ply); CPU only, no Kaggle."""
import base64, io, json, os, shutil, subprocess, sys, tarfile, time
V = os.path.dirname(os.path.abspath(__file__)); FIN = os.path.join(os.path.dirname(V), 'final'); sys.path.insert(0, V); sys.path.insert(0, FIN)
cfg = json.load(open(f'{V}/FIXER.json')); OUT = os.path.abspath('out'); IN = os.path.abspath('in'); T0 = time.time(); STAMP = time.strftime('%Y%m%d-%H%M')
run = lambda c: subprocess.run(c, shell=True, check=True, executable='/bin/bash')

def say(*a): print(f'[{(time.time() - T0) / 60:5.1f} min]', *a, flush=True)

if not cfg.get('n'): say('FIXER.json n = 0: nothing to do'); sys.exit(0)
tok = os.environ.get('HF_TOKEN'); assert tok and os.environ.get('KAGGLE_API_TOKEN'), 'needs HF_TOKEN and KAGGLE_API_TOKEN secrets'
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=tok); user = api.whoami()['name']; repo = f'{user}/namma-v3-out'; R = cfg['run']
d = snapshot_download(repo, repo_type='dataset', local_dir='dl', token=tok, allow_patterns=[f'{R}/solid/clean.ply', f'{R}/tour/tour.json', f'{R}/sparse/**'])
mpu = json.load(open(f'{d}/{R}/tour/tour.json'))['metres_per_unit']
T = os.path.abspath('work'); EX = f'{T}/gsplat/examples'; data = f'{T}/data'
for p in (data, OUT, IN): os.makedirs(p, exist_ok=True)
run(f'git clone -q --depth 1 -b v1.5.3 https://github.com/nerfstudio-project/gsplat {T}/gsplat && touch {EX}/datasets/__init__.py')
run('pip install -q torch --index-url https://download.pytorch.org/whl/cpu && pip install -q opencv-python-headless imageio tqdm plyfile scipy '
    'typing_extensions "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" && sed -i '
    '"s/np.uint64(-1)/np.uint64(np.iinfo(np.uint64).max)/" $(python -c "import importlib.util as u; print(u.find_spec(\'pycolmap\').submodule_search_locations[0])")/scene_manager.py')
shutil.copytree(f'{d}/{R}/sparse', f'{data}/sparse', dirs_exist_ok=True)
import dl3dv, frames                                                                      # the run's frames, same names and order
_, folder = dl3dv.fetch(cfg['scene'], f'{T}/ds', tok, cfg.get('dl3dv_res', 'images_2'))
say('frames', frames.from_images(folder, f'{data}/images', int(cfg.get('nframes', 420)), int(cfg.get('long', 1440))))

import numpy as np, torch, cpu_render, tour_fidelity
from PIL import Image
os.chdir(EX); sys.path.insert(0, EX)
s = cpu_render.load_ply(f'{d}/{R}/solid/clean.ply'); X = s['means'].double().numpy(); op = s['opacities'].numpy()
K, W, H, real, off = tour_fidelity.cameras(data, 1024, 2, X, op, mpu); say('view size', W, H)
from datasets.colmap import Parser
C = Parser(data, factor=1, normalize=True, test_every=8).camtoworlds; up = -np.median(C[:, :3, 1], 0); up /= np.linalg.norm(up)

def turn(deg):
    kx = np.array([[0, -up[2], up[1]], [up[2], 0, -up[0]], [-up[1], up[0], 0]]); a = np.radians(deg)
    return np.cos(a) * np.eye(3) + np.sin(a) * kx + (1 - np.cos(a)) * np.outer(up, up)

views = [(f'held{i}', c) for i, (_, c) in enumerate(real)]                                  # filmed views: Fixer must not spoil them
for i, j in enumerate(np.linspace(0, len(C) - 1, int(cfg.get('views', 10)) + 2).astype(int)[1:-1]):
    c = C[j].copy(); side = (1 if i % 2 else -1) * float(cfg.get('side_m', 1.0)) / mpu        # off the walk, like the tour viewer goes
    c[:3, 3] += c[:3, 0] * side + up * 0.15 / mpu; c[:3, :3] = turn((-1 if i % 2 else 1) * (25 + 25 * (i % 3) / 2)) @ c[:3, :3]
    views.append((f'off{i}', c))
if cfg.get('reuse'):                                          # renders of an earlier attempt (same run and views): skip 10 min of CPU
    snapshot_download(repo, repo_type='dataset', local_dir='prev', token=tok, allow_patterns=[f"{cfg['reuse']}/renders/*"])
    for n, _ in views: shutil.copy(f"prev/{cfg['reuse']}/renders/{n}.png", f'{IN}/{n}.png')
else:
    for n, c in views:
        im = cpu_render.render(s, c, K, W, H); Image.fromarray((im.clamp(0, 1) * 255).byte().numpy()).save(f'{IN}/{n}.png')
real_imgs = {f'held{i}': (img.clamp(0, 1) * 255).byte().numpy() for i, (img, _) in enumerate(real)}
say('rendered', len(views), 'views')

import kaggle_io
ku = kaggle_io.account(cfg); ds, kid = f'{ku}/namma-fixer-in', f'{ku}/namma-fixer'
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode='w:gz') as tf: tf.add(f'{FIN}/fixer_shim', arcname='.')
jcfg = dict(cfg, shim=base64.b64encode(buf.getvalue()).decode())


def lab(x):                                                                  # sRGB 0..1 [...,3] -> CIE Lab (D65)
    x = np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92) @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    x = x / np.array([0.95047, 1.0, 1.08883]); f = np.where(x > 0.008856, np.cbrt(x), 7.787 * x + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def recolour(data, out_dir):
    """Run 4's frames (DL3DV's images_2) -> the colours of the scene's original phone video (what run 6 learned from): match each
    frame to its video frame (normalised grey thumbnails), fit one 2nd-order colour polynomial on blurred low-res pairs (clipped
    pixels left out, worst 10 % residuals dropped twice), apply it to every frame. Returns the fit's numbers."""
    import cv2
    from huggingface_hub import hf_hub_download
    vr = 'DL3DV/DL3DV-ALL-video'; sub = next(k for k in [f'{i}K' for i in range(1, 12)] if api.file_exists(vr, f"{k}/{cfg['scene']}/video.mp4", repo_type='dataset'))
    vid = hf_hub_download(vr, f"{sub}/{cfg['scene']}/video.mp4", repo_type='dataset', token=tok, local_dir=f'{T}/vid')
    cap = cv2.VideoCapture(vid); vth, vsm = [], []
    while True:
        ok, f = cap.read()
        if not ok: break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB); vsm.append(cv2.resize(f, (320, 180), interpolation=cv2.INTER_AREA))
        g = cv2.cvtColor(cv2.resize(f, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32); vth.append((g - g.mean()) / (g.std() + 1e-6))
    vth = np.stack(vth); say('video frames', len(vth))
    fr = sorted(os.listdir(f'{data}/images')); pairs = []; feats = lambda x: np.c_[x, x ** 2, x[:, [0]] * x[:, [1]], x[:, [0]] * x[:, [2]], x[:, [1]] * x[:, [2]], np.ones(len(x))]
    for n in fr:
        im = np.asarray(Image.open(f'{data}/images/{n}').convert('RGB')); g = cv2.cvtColor(cv2.resize(im, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
        g = (g - g.mean()) / (g.std() + 1e-6); j = int(np.argmin(((vth - g) ** 2).mean((1, 2))))
        a = cv2.GaussianBlur(cv2.resize(im, (320, 180), interpolation=cv2.INTER_AREA), (0, 0), 2).reshape(-1, 3) / 255.; b = cv2.GaussianBlur(vsm[j], (0, 0), 2).reshape(-1, 3) / 255.
        keep = (a.max(1) < 0.97) & (a.min(1) > 0.03) & (b.max(1) < 0.97) & (b.min(1) > 0.03); idx = np.random.default_rng(len(pairs)).choice(np.nonzero(keep)[0], min(1500, keep.sum()), replace=False)
        pairs.append((a[idx], b[idx], float(((vth[j] - g) ** 2).mean())))
    A = np.concatenate([p[0] for p in pairs]); B = np.concatenate([p[1] for p in pairs]); m = np.ones(len(A), bool)
    for _ in range(3):
        M = np.linalg.lstsq(feats(A[m]), B[m], rcond=None)[0]; res = np.abs(feats(A) @ M - B).max(1); m = res < np.quantile(res[m], 0.9)
    de0 = float(np.linalg.norm(lab(A) - lab(B), axis=1).mean()); de1 = float(np.linalg.norm(lab(np.clip(feats(A) @ M, 0, 1)) - lab(B), axis=1).mean())
    os.makedirs(f'{out_dir}/images', exist_ok=True)
    for n in fr:
        im = np.asarray(Image.open(f'{data}/images/{n}').convert('RGB')).reshape(-1, 3) / 255.
        Image.fromarray((np.clip(feats(im) @ M, 0, 1) * 255 + 0.5).astype(np.uint8).reshape(*np.asarray(Image.open(f'{data}/images/{n}')).shape[:2], 3)).save(f'{out_dir}/images/{n}', quality=95)
    os.remove(vid); return dict(video_frames=len(vth), match_err_median=round(float(np.median([p[2] for p in pairs])), 3), deltaE_before=round(de0, 2),
                                deltaE_after=round(de1, 2), M=M.tolist())


def grade_ply(src, dst, M):
    """No training: the colour polynomial (recolour's M, 10x3 on sRGB 0..1) applied straight to a model file. Base colour
    c = 0.5 + C0 f_dc -> f(c); the view-dependent SH bands of each splat are multiplied by f's 3x3 Jacobian at c (first-order:
    the view-dependent colour changes the way f changes colours there). Positions, shapes and opacities stay exactly as they are."""
    from plyfile import PlyData, PlyElement
    v = PlyData.read(src)['vertex'].data.copy(); C0 = 0.28209479177387814; M = np.asarray(M, np.float64)
    c = np.stack([v[f'f_dc_{i}'] for i in range(3)], 1).astype(np.float64) * C0 + 0.5; x = np.clip(c, 0, 1)
    F = np.c_[x, x ** 2, x[:, [0]] * x[:, [1]], x[:, [0]] * x[:, [2]], x[:, [1]] * x[:, [2]], np.ones(len(x))]
    dF = np.zeros((len(x), 10, 3))                                               # d features / d (r, g, b)
    for i in range(3): dF[:, i, i] = 1; dF[:, 3 + i, i] = 2 * x[:, i]
    for k, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))): dF[:, 6 + k, a] = x[:, b]; dF[:, 6 + k, b] = x[:, a]
    J = np.einsum('fo,nfi->noi', M, dF)                                          # [n, out, in]
    y = F @ M + np.einsum('noi,ni->no', J, c - x)                                # outside 0..1: keep the offset, linearised
    for i in range(3): v[f'f_dc_{i}'] = ((y[:, i] - 0.5) / C0).astype(np.float32)
    nr = len([n for n in v.dtype.names if n.startswith('f_rest_')]) // 3
    if nr:
        R = np.stack([v[f'f_rest_{i}'] for i in range(3 * nr)], 1).astype(np.float64).reshape(-1, 3, nr)
        R = np.einsum('noi,nik->nok', J, R).reshape(-1, 3 * nr)
        for i in range(3 * nr): v[f'f_rest_{i}'] = R[:, i].astype(np.float32)
    PlyData([PlyElement.describe(v, 'vertex')]).write(dst)
    return dict(splats=len(v), sh_rest_per_colour=nr, mean_rgb_before=x.mean(0).round(3).tolist(), mean_rgb_after=np.clip(y, 0, 1).mean(0).round(3).tolist())


def mottle(img, flat):
    """Wall blotches: mean colour / brightness change at blotch scale (4..25 px band) over flat areas."""
    from scipy.ndimage import gaussian_filter
    a = np.asarray(img, np.float64); band = lambda q: gaussian_filter(q, 4) - gaussian_filter(q, 25)
    L = band(a.mean(2)); ch = np.hypot(band(a[..., 0] - a[..., 1]), band(a[..., 2] - (a[..., 0] + a[..., 1]) / 2))
    return float(np.abs(ch[flat]).mean()), float(np.abs(L[flat]).mean())


def flat_mask(img):
    from scipy.ndimage import gaussian_filter
    L = np.asarray(img, np.float64).mean(2); gm = np.hypot(*np.gradient(gaussian_filter(L, 1.5)))
    return gaussian_filter((gm > 3).astype(float), 6) < 0.05


if cfg.get('mode') == 'grade':                                                     # owner, 30 Oct: true colours, NO retraining
    rc = f'{T}/data_rc'; shutil.copytree(f'{data}/sparse', f'{rc}/sparse', dirs_exist_ok=True)
    rep = dict(recolour=recolour(data, rc)); say('colour fit', {k: v for k, v in rep['recolour'].items() if k != 'M'})
    rep['grade'] = grade_ply(f'{d}/{R}/solid/clean.ply', f'{OUT}/graded.ply', rep['recolour']['M']); say('graded', rep['grade'])
    api.upload_file(path_or_fileobj=f'{OUT}/graded.ply', path_in_repo=f'{R}/graded/model.ply', repo_id=repo, repo_type='dataset')   # tour build can start
    snapshot_download(repo, repo_type='dataset', local_dir=d, token=tok, allow_patterns=[f'{R}/fixer/model.ply'])   # (cwd is EX now)
    models = dict(clean=f'{d}/{R}/solid/clean.ply', graded=f'{OUT}/graded.ply', refined_v2=f'{d}/{R}/fixer/model.ply')
    subprocess.run(f'cd {EX} && PYTHONPATH={EX}:{FIN}:$PYTHONPATH python {FIN}/tour_fidelity.py {rc} {OUT}/fid_video_colours {mpu} 640 12 '
                   + ' '.join(f'{k}={v}' for k, v in models.items()) + ' 2>&1 | tail -5', shell=True, executable='/bin/bash')
    if os.path.exists(f'{OUT}/fid_video_colours/fidelity_report.json'): rep['fidelity_video_colours'] = json.load(open(f'{OUT}/fid_video_colours/fidelity_report.json'))
    tw = 512; th = round(H * tw / W); rows = []; mot = {k: [] for k in models}; loaded = {k: cpu_render.load_ply(p) for k, p in models.items() if k != 'clean'}
    for n, c in views:
        ims = dict(clean=Image.open(f'{IN}/{n}.png').convert('RGB'))               # clean renders of these views (reuse / smoke test)
        for k, sm in loaded.items(): ims[k] = Image.fromarray((cpu_render.render(sm, c, K, W, H).clamp(0, 1) * 255).byte().numpy())
        if n.startswith('off'):
            fl = flat_mask(ims['clean'])
            for k in models: mot[k].append(mottle(ims[k], fl))
        r = Image.new('RGB', (tw * 4, th))                                         # filmed views: the photo in video colours
        if n in real_imgs:
            q = real_imgs[n].reshape(-1, 3) / 255.; q = np.c_[q, q ** 2, q[:, [0]] * q[:, [1]], q[:, [0]] * q[:, [2]], q[:, [1]] * q[:, [2]], np.ones(len(q))]
            r.paste(Image.fromarray((np.clip(q @ np.asarray(rep['recolour']['M']), 0, 1) * 255 + 0.5).astype(np.uint8).reshape(real_imgs[n].shape)).resize((tw, th)), (0, 0))
        for i, k in enumerate(models): r.paste(ims[k].resize((tw, th)), ((i + 1) * tw, 0))
        rows.append(r); ims['graded'].save(f'{OUT}/graded_{n}.png')
    rep['wall_mottling_off_path'] = {k: dict(colour=round(float(np.mean([m[0] for m in v])), 3), brightness=round(float(np.mean([m[1] for m in v])), 3),
                                             per_view_colour=[round(m[0], 2) for m in v]) for k, v in mot.items()}
    say('wall mottling', {k: (v['colour'], v['brightness']) for k, v in rep['wall_mottling_off_path'].items()})
    sheet = Image.new('RGB', (tw * 4, th * len(rows)))
    for i, r in enumerate(rows): sheet.paste(r, (0, i * th))
    sheet.save(f'{OUT}/grade_sheet.jpg', quality=88)                             # columns: photo | clean | graded | refined v2
    os.remove(f'{OUT}/graded.ply'); json.dump(rep, open(f'{OUT}/grade_summary.json', 'w'), indent=1, default=float)
    dst = f'fixer/{STAMP}-grade'; api.upload_folder(folder_path=OUT, path_in_repo=dst, repo_id=repo, repo_type='dataset')
    say(f'results: hf://datasets/{repo}/{dst}'); sys.exit(0)


if cfg.get('mode') == 'refine':                                                   # plan of 30 Oct: Fixer views + video colours -> refined run 4
    rc = f'{T}/data_rc'; os.makedirs(f'{T}/before', exist_ok=True)
    for f in os.listdir(IN):
        if f.endswith('.png'): shutil.move(f'{IN}/{f}', f'{T}/before/{f}')
    if cfg.get('reuse_dataset'):                                                  # namma-fixer-in already holds this run's inputs
        rep = dict(recolour='reused (namma-fixer-in as uploaded before)'); rc = None
    else:
        shutil.copytree(f'{data}/sparse', f'{rc}/sparse', dirs_exist_ok=True)
        rep = dict(recolour=recolour(data, rc)); say('recolour', rep['recolour'])
        shutil.make_archive(f'{IN}/frames', 'zip', rc, 'images'); shutil.make_archive(f'{IN}/sparse', 'zip', data, 'sparse')
        shutil.copy(f'{d}/{R}/solid/clean.ply', f'{IN}/clean.ply')
        kaggle_io.push_dataset(IN, ds, 'namma-fixer-in', f'{R} refine {STAMP}', say)
    kaggle_io.push_kernel(kid, 'namma-fixer', f'{FIN}/fixer_refine.py', dict(jcfg, mpu=mpu), [ds], say,
                          first='import base64, gc, glob, io, json, os, shutil, subprocess, sys, time\n')
    st = kaggle_io.wait_kernel(kid, OUT, say, minutes=int(cfg.get('minutes', 300)))
    if os.path.exists(f'{OUT}/refine_report.json'): rep['kaggle'] = json.load(open(f'{OUT}/refine_report.json'))
    if os.path.exists(f'{OUT}/refine.log'): print(''.join(open(f'{OUT}/refine.log', errors='ignore').readlines()[-80:]), flush=True)
    ok = os.path.exists(f'{OUT}/refined.ply'); say('kaggle', st[-80:], json.dumps(rep)[:1500])
    if ok:                                                                        # proof: real photos (video colours), off-path sheet
        for name, dd in [x for x in (('video_colours', rc), ('images_2_colours', data)) if x[1]]:
            subprocess.run(f'cd {EX} && PYTHONPATH={EX}:{FIN}:$PYTHONPATH python {FIN}/tour_fidelity.py {dd} {OUT}/fid_{name} {mpu} 640 12 '
                           f'clean={d}/{R}/solid/clean.ply refined={OUT}/refined.ply 2>&1 | tail -5', shell=True, executable='/bin/bash')
            if os.path.exists(f'{OUT}/fid_{name}/fidelity_report.json'): rep[f'fidelity_{name}'] = json.load(open(f'{OUT}/fid_{name}/fidelity_report.json'))
        ref = cpu_render.load_ply(f'{OUT}/refined.ply'); tw = 512; th = round(H * tw / W); rows = []
        for n, c in views:
            a = Image.open(f'{T}/before/{n}.png')                                   # the cleaned model (smoke-test renders)
            b = Image.fromarray((cpu_render.render(ref, c, K, W, H).clamp(0, 1) * 255).byte().numpy())
            r = Image.new('RGB', (tw * 3, th)); r.paste(Image.fromarray(real_imgs[n]).resize((tw, th)) if n in real_imgs else Image.new('RGB', (tw, th)), (0, 0))
            r.paste(a.resize((tw, th)), (tw, 0)); r.paste(b.resize((tw, th)), (2 * tw, 0)); rows.append(r); b.save(f'{OUT}/after_{n}.png')
        sheet = Image.new('RGB', (tw * 3, th * len(rows)))
        for i, r in enumerate(rows): sheet.paste(r, (0, i * th))
        sheet.save(f'{OUT}/refine_sheet.jpg', quality=88)
        api.upload_file(path_or_fileobj=f'{OUT}/refined.ply', path_in_repo=f'{R}/fixer/model.ply', repo_id=repo, repo_type='dataset')
        os.remove(f'{OUT}/refined.ply')
    json.dump(rep, open(f'{OUT}/refine_summary.json', 'w'), indent=1, default=float)
    dst = f'fixer/{STAMP}-refine'; api.upload_folder(folder_path=OUT, path_in_repo=dst, repo_id=repo, repo_type='dataset')
    say(f'results: hf://datasets/{repo}/{dst}'); sys.exit(0 if ok else 'refinement failed (see refine.log)')

kaggle_io.push_dataset(IN, ds, 'namma-fixer-in', f'{R} {STAMP}', say)
kaggle_io.push_kernel(kid, 'namma-fixer', f'{FIN}/fixer_smoke.py', jcfg, [ds], say)
st = kaggle_io.wait_kernel(kid, OUT, say, minutes=int(cfg.get('minutes', 120)))
rep = json.load(open(f'{OUT}/fixer_report.json')) if os.path.exists(f'{OUT}/fixer_report.json') else {}
say('kaggle', st[-100:], json.dumps(rep)[:800])
for f in ('fixer.log',):
    if os.path.exists(f'{OUT}/{f}'): print(''.join(open(f'{OUT}/{f}', errors='ignore').readlines()[-60:]), flush=True)

fx = next((f'{OUT}/fixed_{p}' for p in ('bfloat16', 'float16', 'float32') if os.path.isdir(f'{OUT}/fixed_{p}')), None)
if fx:                                                                                   # sheet: [real photo |] render | Fixer, per view
    rows = []; tw = 512; th = round(H * tw / W)
    for n, _ in views:
        a = Image.open(f'{IN}/{n}.png').resize((tw, th)); b = Image.open(f'{fx}/{n}.png').resize((tw, th))
        r = Image.new('RGB', (tw * 3, th)); r.paste(Image.fromarray(real_imgs[n]).resize((tw, th)) if n in real_imgs else Image.new('RGB', (tw, th)), (0, 0))
        r.paste(a, (tw, 0)); r.paste(b, (2 * tw, 0)); rows.append(r)
        if n in real_imgs:                                                                 # does Fixer move a filmed view away from the photo?
            ref = real_imgs[n].astype(np.float32); f = lambda im: 10 * np.log10(255 ** 2 / max(((np.asarray(im.resize((W, H)), np.float32) - ref) ** 2).mean(), 1e-6))
            rep.setdefault('held_psnr', {})[n] = dict(render=round(float(f(Image.open(f'{IN}/{n}.png'))), 2), fixed=round(float(f(Image.open(f'{fx}/{n}.png'))), 2))
    sheet = Image.new('RGB', (tw * 3, th * len(rows)))
    for i, r in enumerate(rows): sheet.paste(r, (0, i * th))
    sheet.save(f'{OUT}/fixer_sheet.jpg', quality=88); json.dump(rep, open(f'{OUT}/fixer_report.json', 'w'), indent=1)
    say('held-out (photo vs render / vs Fixer):', rep.get('held_psnr'))
shutil.copytree(IN, f'{OUT}/renders', dirs_exist_ok=True)
dst = f'fixer/{STAMP}'; api.upload_folder(folder_path=OUT, path_in_repo=dst, repo_id=repo, repo_type='dataset')
say(f'results: hf://datasets/{repo}/{dst}'); sys.exit(0 if fx else 'Fixer produced no images (see fixer.log)')
