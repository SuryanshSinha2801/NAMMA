import base64, gc, glob, io, json, os, shutil, subprocess, sys, time
"""Kaggle job namma-fixer (mode refine, v3/fixer_gh.py; plan of 30 Oct, run 4 only): teach a finished run's splats the views NVIDIA
Fixer cleaned, so every angle of the tour stays clean, and give it the phone video's true colours (the frames come re-coloured
to the original video by the GitHub side).
  1 needle splats shortened (final/solidify.unneedle): the side-angle streaks
  2 ~300 tour-path cameras: every 3rd training camera moved 0.5-1 m sideways / turned 30-90 deg (held-out frames never used)
  3 rounds x (render on GPU 0 -> Fixer on GPU 1 -> mask -> STEPS steps): real frames weight 1 (L1 + 0.2 SSIM, per-frame colour
    correction pinned to identity on average, so the splats take the new colours), Fixer views weight LAMBDA only where Fixer
    removed something (colour-matched difference > THR) and the surface lies inside the room box (windows / outside / mirrors
    stay with the real frames)
  out  refined.ply (run's coordinates and PLY layout), refine_report.json (held-out PSNR before/after, exposure-fair)"""
cfg = json.loads(os.environ.get('JOB_CFG', '{}')); T0 = time.time(); O = '/kaggle/working'; W = '/kaggle/temp/fr'; os.makedirs(W, exist_ok=True)
ROUNDS, STEPS, LAM, THR = int(cfg.get('rounds', 3)), int(cfg.get('steps', 2000)), float(cfg.get('lambda', 0.3)), float(cfg.get('thr', 0.08))
LR, LONG, MPU = float(cfg.get('lr_scale', 0.3)), int(cfg.get('train_long', 960)), float(cfg['mpu'])
REP = dict(cfg={k: v for k, v in cfg.items() if k != 'shim'}); log = open(f'{O}/refine.log', 'a')

def say(*a):
    m = ' '.join(str(x) for x in a); print(f'[{(time.time() - T0) / 60:5.1f} min]', m, flush=True); log.write(m + '\n'); log.flush()

def sh(c, check=False):
    r = subprocess.run(c, shell=True, executable='/bin/bash', capture_output=True, text=True); say(c[:90], '->', r.returncode, (r.stdout + r.stderr)[-400:])
    if check and r.returncode: raise RuntimeError(c[:80])

def save(): REP['minutes'] = round((time.time() - T0) / 60, 1); json.dump(REP, open(f'{O}/refine_report.json', 'w'), indent=1)

try:
    sh(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', check=True)
    F = f'{W}/repo/colab/real_scenes/final'; sys.path.insert(0, F); EX = f'{W}/gsplat/examples'
    sh(f'git clone -q --depth 1 -b v1.5.3 https://github.com/nerfstudio-project/gsplat {W}/gsplat && touch {EX}/datasets/__init__.py; '
       f'echo "torch==$(python -c "import torch; print(torch.__version__)")" > {W}/pin.txt; pip install -q -c {W}/pin.txt ninja gsplat==1.5.3 plyfile imageio; '
       'pip install -q --no-deps "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e"; '
       'sed -i "s/np.uint64(-1)/np.uint64(np.iinfo(np.uint64).max)/" $(python -c "import importlib.util as u; print(u.find_spec(\'pycolmap\').submodule_search_locations[0])")/scene_manager.py', check=True)
    sh(f'cd {W} && python {F}/gsplat_warm.py', check=True)
    import tarfile; tarfile.open(fileobj=io.BytesIO(base64.b64decode(cfg['shim']))).extractall(f'{W}/shim')
    import fixer_lib; fixer_lib.setup(f'{W}/fx', say)

    import numpy as np, torch, torch.nn.functional as Fn
    from PIL import Image
    from plyfile import PlyData, PlyElement
    from scipy import ndimage
    import solidify
    # data: the run's sparse + its frames (re-coloured to the video), same names/order as the run
    inp = lambda pat: sorted(glob.glob(f'/kaggle/input/**/{pat}', recursive=True))
    D = f'{W}/data'; os.makedirs(D, exist_ok=True)                  # Kaggle unzips uploaded zips into folders: take either form
    if inp('sparse.zip'): shutil.unpack_archive(inp('sparse.zip')[0], D)
    else: shutil.copytree(os.path.dirname(os.path.dirname(inp('cameras.bin')[0])), f'{D}/sparse')
    if inp('frames.zip'): shutil.unpack_archive(inp('frames.zip')[0], D)
    else:
        cnt = {}
        for f in inp('*.jpg'): cnt[os.path.dirname(f)] = cnt.get(os.path.dirname(f), 0) + 1
        shutil.copytree(max(cnt, key=cnt.get), f'{D}/images')
    say('data', os.listdir(f'{D}/sparse'), len(os.listdir(f'{D}/images')), 'frames')
    for m in [m for m in sys.modules if m.split('.')[0] == 'datasets']: del sys.modules[m]
    sys.path.insert(0, EX); from datasets.colmap import Parser
    P = Parser(D, factor=1, normalize=True, test_every=8); C = P.camtoworlds; N = len(C)
    tr = [i for i in range(N) if i % 8]; ho = [i for i in range(N) if i % 8 == 0]
    K0 = P.Ks_dict[P.camera_ids[0]].copy(); W0, H0 = P.imsize_dict[P.camera_ids[0]]; k = LONG / max(W0, H0)
    Wt, Ht = round(W0 * k), round(H0 * k); Kt = K0.copy(); Kt[:2] *= k
    imgs = [torch.tensor(np.asarray(Image.open(p).convert('RGB').resize((Wt, Ht), Image.LANCZOS)), dtype=torch.float32) / 255 for p in P.image_paths]
    say(N, 'frames', Wt, 'x', Ht, len(tr), 'train', len(ho), 'held-out')

    # splats (the cleaned model), needles shortened inside the room box (SfM points 1-99 %, +5 %)
    plyp = inp('clean.ply')[0]; V = PlyData.read(plyp)['vertex'].data
    t_ = lambda *ks: torch.tensor(np.stack([V[x] for x in ks], 1), dtype=torch.float32)
    nr = len([n for n in V.dtype.names if n.startswith('f_rest_')]) // 3; assert nr == 15, 'expects SH degree 3'
    lo, hi = np.percentile(P.points, 1, 0), np.percentile(P.points, 99, 0); pad = (hi - lo) * 0.05; lo, hi = lo - pad, hi + pad
    X = t_('x', 'y', 'z'); inside = np.all((X.numpy() > lo) & (X.numpy() < hi), 1)
    ls, nneedle = solidify.unneedle(t_('scale_0', 'scale_1', 'scale_2').numpy(), inside, float(cfg.get('needle_ratio', 4)))
    REP.update(splats=len(V), needles_shortened=nneedle); say(len(V), 'splats;', nneedle, 'needles shortened')
    dev = 'cuda:0'
    prm = {'means': X, 'scales': torch.tensor(ls), 'quats': t_('rot_0', 'rot_1', 'rot_2', 'rot_3'),
           'opacities': torch.tensor(np.asarray(V['opacity']), dtype=torch.float32), 'sh0': t_('f_dc_0', 'f_dc_1', 'f_dc_2')[:, None],
           'shN': t_(*[f'f_rest_{i}' for i in range(45)]).reshape(-1, 3, 15).transpose(1, 2).contiguous()}
    prm = {n: torch.nn.Parameter(v.to(dev)) for n, v in prm.items()}
    lrs = dict(means=1.6e-4 * P.scene_scale, scales=5e-3, quats=1e-3, opacities=5e-2, sh0=2.5e-3, shN=2.5e-3 / 20)
    opt = torch.optim.Adam([dict(params=[prm[n]], lr=lrs[n] * LR, name=n) for n in prm], eps=1e-15)
    cA = torch.nn.Parameter(torch.eye(3, device=dev).repeat(N, 1, 1)); cb = torch.nn.Parameter(torch.zeros(N, 3, device=dev))
    copt = torch.optim.Adam([cA, cb], lr=1e-3)
    from gsplat import rasterization

    def render(c2w, K, w, h, depth=False):
        r, a, _ = rasterization(prm['means'], Fn.normalize(prm['quats'], dim=-1), torch.exp(prm['scales']), torch.sigmoid(prm['opacities']),
                                torch.cat([prm['sh0'], prm['shN']], 1), torch.linalg.inv(torch.as_tensor(c2w, dtype=torch.float32, device=dev))[None],
                                torch.as_tensor(K, dtype=torch.float32, device=dev)[None], w, h, sh_degree=3, render_mode='RGB+ED' if depth else 'RGB')
        return r[0], a[0]

    def ssim(a, b, win=11):                                                  # [H,W,3] -> mean SSIM map [H,W]
        g = torch.exp(-((torch.arange(win, device=a.device) - win // 2) ** 2) / (2 * 1.5 ** 2)); g = (g / g.sum())
        k2 = (g[:, None] * g[None]).expand(3, 1, win, win); f = lambda x: Fn.conv2d(x, k2, padding=win // 2, groups=3)
        a, b = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]; ma, mb = f(a), f(b)
        va, vb, cv = f(a * a) - ma ** 2, f(b * b) - mb ** 2, f(a * b) - ma * mb
        return (((2 * ma * mb + 1e-4) * (2 * cv + 9e-4)) / ((ma ** 2 + mb ** 2 + 1e-4) * (va + vb + 9e-4)))[0].mean(0)

    def lab(x):                                                              # sRGB 0..1 -> CIE Lab (D65), torch
        x = torch.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92) @ torch.tensor([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], device=x.device).T
        x = x / torch.tensor([0.95047, 1.0, 1.08883], device=x.device); f = torch.where(x > 0.008856, x.clamp(min=1e-9) ** (1 / 3), 7.787 * x + 16 / 116)
        return torch.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)

    @torch.no_grad()
    def heldout():                                       # PSNR exposure-fair (per-image affine colour fit) + raw colour error (dE)
        ps, de, ss = [], [], []
        for i in ho:
            r = render(C[i], Kt, Wt, Ht)[0][..., :3].clamp(0, 1).reshape(-1, 3); g = imgs[i].to(dev).reshape(-1, 3)
            A = torch.cat([r, torch.ones_like(r[:, :1])], 1); M = torch.linalg.lstsq(A, g).solution
            ps.append(float(-10 * torch.log10(((A @ M).clamp(0, 1) - g).pow(2).mean()))); de.append(float((lab(r) - lab(g)).norm(dim=-1).mean()))
            y = lambda x: (x.reshape(Ht, Wt, 3) @ torch.tensor([0.299, 0.587, 0.114], device=dev))[..., None].repeat(1, 1, 3)   # brightness only:
            ss.append(float(ssim(y((A @ M).clamp(0, 1)), y(g)).mean()))                                                      # detail, not colour
        return dict(psnr=round(float(np.mean(ps)), 3), dE=round(float(np.mean(de)), 2), ssim_luma=round(float(np.mean(ss)), 4))

    REP['heldout_before'] = heldout(); say('held-out before', REP['heldout_before']); save()

    # tour-path cameras: off the walk, turned (camera's own right axis / the scene's up axis)
    up = -np.median(C[:, :3, 1], 0); up /= np.linalg.norm(up)
    def turn(deg):
        kx = np.array([[0, -up[2], up[1]], [up[2], 0, -up[0]], [-up[1], up[0], 0]]); a = np.radians(deg)
        return np.cos(a) * np.eye(3) + np.sin(a) * kx + (1 - np.cos(a)) * np.outer(up, up)
    pat = [(0.5, 30), (-0.5, -30), (1.0, 60), (-1.0, -60), (0.5, -45), (-0.5, 45), (0.0, 90), (0.0, -90), (1.0, -20), (-1.0, 20)]
    tg = []
    for n, i in enumerate(tr[::3]):
        for j in range(int(cfg.get('per_cam', 3))):
            s_, d_ = pat[(n * 3 + j) % len(pat)]; c = C[i].copy()
            c[:3, 3] += c[:3, 0] * s_ / MPU + (up * 0.3 / MPU if (n + j) % 5 == 0 else 0); c[:3, :3] = turn(d_) @ c[:3, :3]; tg.append(c)
    Wf, Hf = 1024, round(1024 * H0 / W0); Kf = K0.copy(); Kf[:2] *= 1024 / W0
    REP['targets'] = len(tg); say(len(tg), 'tour-path views at', Wf, 'x', Hf)
    fixer = fixer_lib.load(f'{W}/fx', f'{W}/shim', 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0', say=say)

    def fixed_views(rnd):
        out = []; frac = []
        for t, c in enumerate(tg):
            with torch.no_grad():
                r, a = render(c, Kf, Wf, Hf, depth=True)
            rgb = (r[..., :3].clamp(0, 1) * 255).byte().cpu().numpy(); dep = r[..., 3].cpu().numpy()
            fx = fixer(rgb); rf, ff = rgb.reshape(-1, 3) / 255., fx.reshape(-1, 3) / 255.
            A = np.c_[rf, np.ones(len(rf))]; d = np.abs(A @ np.linalg.lstsq(A, ff, rcond=None)[0] - ff).max(1).reshape(Hf, Wf)
            m = ndimage.binary_dilation(ndimage.binary_opening(d > THR, iterations=2), iterations=8)
            yy, xx = np.mgrid[0:Hf, 0:Wf]; ray = np.stack([(xx - Kf[0, 2]) / Kf[0, 0], (yy - Kf[1, 2]) / Kf[1, 1], np.ones_like(xx, float)], -1)
            pw = (ray * dep[..., None]) @ c[:3, :3].T + c[:3, 3]; m &= np.all((pw > lo) & (pw < hi), -1) & (dep > 0)
            out.append((c, torch.tensor(fx), torch.tensor(m))); frac.append(m.mean())
            if rnd == 0 and t < 12: Image.fromarray(np.concatenate([rgb, fx, np.repeat(m[..., None] * 255, 3, -1).astype(np.uint8)], 1)).save(f'{O}/fix_{t:02d}.jpg', quality=85)
        say('round', rnd, 'fixed', len(out), 'views; mask share', round(float(np.mean(frac)), 3)); return out

    for rnd in range(ROUNDS):
        fv = fixed_views(rnd); torch.cuda.empty_cache()
        for st in range(STEPS):
            if np.random.rand() < 0.7:
                i = int(np.random.choice(tr)); r = render(C[i], Kt, Wt, Ht)[0][..., :3]
                A_ = cA - cA.mean(0) + torch.eye(3, device=dev); b_ = cb - cb.mean(0)            # pinned: mean correction = identity
                r = r @ A_[i].T + b_[i]; g = imgs[i].to(dev)
                loss = (r - g).abs().mean() + 0.2 * (1 - ssim(r.clamp(0, 1), g).mean())
            else:
                c, fx, m = fv[np.random.randint(len(fv))]
                if not m.any(): continue
                r = render(c, Kf, Wf, Hf)[0][..., :3]; g = fx.to(dev).float() / 255; mm = m.to(dev).float()
                loss = LAM * (((r - g).abs().mean(-1) * mm).sum() / mm.sum() + 0.2 * ((1 - ssim(r.clamp(0, 1), g)) * mm).sum() / mm.sum())
            opt.zero_grad(); copt.zero_grad(); loss.backward(); opt.step(); copt.step()
            if st % 500 == 0: say('round', rnd, 'step', st, 'loss', round(float(loss), 4))
        REP[f'heldout_round{rnd}'] = heldout(); say('held-out after round', rnd, REP[f'heldout_round{rnd}']); save()

    o = V.copy(); g = lambda n: prm[n].detach().cpu().numpy()                     # same PLY layout as the input
    for i in range(3): o[['x', 'y', 'z'][i]] = g('means')[:, i]; o[f'scale_{i}'] = g('scales')[:, i]; o[f'f_dc_{i}'] = g('sh0')[:, 0, i]
    for i in range(4): o[f'rot_{i}'] = g('quats')[:, i]
    o['opacity'] = g('opacities'); shn = g('shN').transpose(0, 2, 1).reshape(-1, 45)
    for i in range(45): o[f'f_rest_{i}'] = shn[:, i]
    PlyData([PlyElement.describe(o, 'vertex')]).write(f'{O}/refined.ply'); REP['ok'] = True; say('refined.ply written')
except Exception as e:
    import traceback; REP['error'] = f'{type(e).__name__}: {str(e)[:400]}'; say('failed:', traceback.format_exc()[-2500:])
save(); shutil.rmtree(f'{W}/fx/models', ignore_errors=True)
