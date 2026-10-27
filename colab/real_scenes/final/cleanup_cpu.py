"""Floater clean-up experiments on a trained model, CPU only (owner 27 Oct: no Kaggle GPU hours; ideas from docs/KMYI_NOTES.md).
No retraining. Every variant is measured with cpu_render.py on the held-out frames (every 8th) against the real photos, plus an
off-path sheet (raised / turned cameras and a view from above), where floaters show.
  base    the model as trained
  dual    Asymmetric Dual 3DGS idea (Li et al.: floaters are random, two trainings do not agree on them) without training twice:
          run 5 already trained the same frames twice ('full' and 'prior', same cameras). A splat of the model with no splat of the
          other training within max(2 cm, 2x its own size, up to 10 cm) is dropped.
  feat    FeatureGS idea (Jaeger et al.: Gaussians should follow how their neighbours are arranged), as a filter: per splat, the
          eigenvalues of its 16 nearest neighbours' positions. Dropped: splats far from their neighbours (statistical outlier,
          mean distance > mean + 3 std of the scene) and splats in neighbourhoods with no structure (neither flat nor line-like:
          planarity and linearity both < 0.15) that are also sparse (4x the scene's typical spacing).
  both    union of dual and feat drops
Usage (from gsplat's examples dir): python cleanup_cpu.py <data dir> <model.ply> <other.splat> <other level.json> <metres_per_unit>
  <out dir> [long side px, default 480] [held-out views, default 26]"""
import json, os, sys, time, numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpu_render
T0 = time.time()
REC = [('p', '<f4', 3), ('s', '<f4', 3), ('c', 'u1', 4), ('r', 'u1', 4)]


def say(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def other_splats(path, level_json):
    """The second training (.splat, levelled with its own level.json) back in the training frame: positions, sizes, opacity."""
    r = np.fromfile(path, dtype=REC); M = np.array(json.load(open(level_json))['matrix']); R, t = M[:3, :3], M[:3, 3]
    return (r['p'].astype(np.float64) - t) @ R, r['s'].max(1).astype(np.float64), r['c'][:, 3] / 255.0


def dual_drop(X, smax, op, Y, yop, mpu):
    from scipy.spatial import cKDTree
    Yk = Y[yop > 0.1]; tree = cKDTree(Yk); r = np.clip(2 * smax, 0.02 / mpu, 0.10 / mpu)
    cand = np.where(op > 0.05)[0]; d, _ = tree.query(X[cand], k=1, workers=-1)
    drop = np.zeros(len(X), bool); drop[cand[d > r[cand]]] = True
    return drop


def feat_drop(X, op, k=16):
    from scipy.spatial import cKDTree
    cand = np.where(op > 0.05)[0]; P = X[cand]; d, nb = cKDTree(P).query(P, k=k + 1, workers=-1)
    md = d[:, 1:].mean(1); outlier = md > md.mean() + 3 * md.std()
    Q = P[nb[:, 1:]] - P[nb[:, 1:]].mean(1, keepdims=True); C = np.einsum('nki,nkj->nij', Q, Q) / k
    ev = np.sort(np.linalg.eigvalsh(C), 1)[:, ::-1]; l1 = np.maximum(ev[:, 0], 1e-20)
    planar, linear = (ev[:, 1] - ev[:, 2]) / l1, (ev[:, 0] - ev[:, 1]) / l1
    loose = (planar < 0.15) & (linear < 0.15) & (md > 4 * np.median(md))
    drop = np.zeros(len(X), bool); drop[cand[outlier | loose]] = True
    return drop, dict(outlier=int(outlier.sum()), loose=int(loose.sum()))


def ssim(a, b):
    """Gaussian-window SSIM (11 px, sigma 1.5) on [H, W, 3] tensors in 0..1."""
    import torch.nn.functional as F
    g = torch.exp(-((torch.arange(11) - 5.0) ** 2) / (2 * 1.5 ** 2)); g = (g / g.sum()); w = (g[:, None] * g[None]).expand(3, 1, 11, 11)
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]; f = lambda z: F.conv2d(z, w, groups=3)
    mx, my = f(x), f(y); sx, sy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    return float((((2 * mx * my + 1e-4) * (2 * sxy + 9e-4)) / ((mx ** 2 + my ** 2 + 1e-4) * (sx + sy + 9e-4))).mean())


def main(data, ply, other, other_level, mpu, out, long=480, nviews=26):
    mpu, long, nviews = float(mpu), int(long), int(nviews); os.makedirs(out, exist_ok=True)
    sys.path.insert(0, os.getcwd()); from datasets.colmap import Parser, Dataset
    s = cpu_render.load_ply(ply); X = s['means'].double().numpy(); op = s['opacities'].numpy(); smax = s['scales'].max(1).values.double().numpy()
    Y, ys, yop = other_splats(other, other_level); say(len(X), 'splats; other training', len(Y))
    drops = dict(base=np.zeros(len(X), bool), dual=dual_drop(X, smax, op, Y, yop, mpu))
    drops['feat'], info = feat_drop(X, op); drops['both'] = drops['dual'] | drops['feat']
    rep = dict(splats=len(X), dropped={k: int(v.sum()) for k, v in drops.items()}, feat_parts=info)
    say('dropped:', rep['dropped'], info)
    p = Parser(data, factor=1, normalize=True, test_every=8); ds = Dataset(p, split='val'); cid = p.camera_ids[0]
    W0, H0 = p.imsize_dict[cid]; k = long / max(W0, H0); W, H = round(W0 * k), round(H0 * k); K = p.Ks_dict[cid].copy(); K[:2] *= k
    pick = np.linspace(0, len(ds) - 1, min(nviews, len(ds))).astype(int)
    real = [(torch.tensor(np.asarray(Image.fromarray(ds[int(j)]['image'].numpy().astype(np.uint8)).resize((W, H), Image.LANCZOS)), dtype=torch.float32) / 255,
             ds[int(j)]['camtoworld'].numpy()) for j in pick]
    # off-path cameras: training-path cameras raised 40 cm and turned 35 deg, plus one from above the room
    C = p.camtoworlds; up = -np.median(C[:, :3, 1], 0); up /= np.linalg.norm(up); off = []
    for j in np.linspace(0, len(C) - 1, 5).astype(int)[1:-1]:
        c = C[j].copy(); c[:3, 3] += up * 0.4 / mpu; a = np.radians(35); kx = up
        Rr = np.cos(a) * np.eye(3) + np.sin(a) * np.array([[0, -kx[2], kx[1]], [kx[2], 0, -kx[0]], [-kx[1], kx[0], 0]]) + (1 - np.cos(a)) * np.outer(kx, kx)
        c[:3, :3] = Rr @ c[:3, :3]; off.append(c)
    ctr = np.median(X[op > 0.3], 0); eye = ctr + up * 2.6 / mpu - C[:, :3, 2].mean(0) * 1.0 / mpu
    f = ctr - eye; f /= np.linalg.norm(f); r_ = np.cross(f, up); r_ /= np.linalg.norm(r_); d_ = np.cross(f, r_)
    top = np.eye(4); top[:3, :3] = np.stack([r_, d_, f], 1); top[:3, 3] = eye; off.append(top)
    res = {}; sheets = {}
    for name, dr in drops.items():
        keep = torch.from_numpy(~dr); sv = {kk: vv[keep] for kk, vv in s.items()}; ps, ss = [], []
        for n, (img, c2w) in enumerate(real):
            r = cpu_render.render(sv, c2w, K, W, H); mse = float(((r - img) ** 2).mean())
            ps.append(10 * np.log10(1 / max(mse, 1e-10))); ss.append(ssim(r, img))
            if n < 3: sheets.setdefault(('held', n), []).append((name, r))
        for n, c2w in enumerate(off): sheets.setdefault(('off', n), []).append((name, cpu_render.render(sv, c2w, K, W, H)))
        res[name] = dict(psnr=round(float(np.mean(ps)), 3), ssim=round(float(np.mean(ss)), 4), dropped=int(dr.sum()))
        say(name, res[name])
    rep['heldout'] = res; rep['views'] = len(real); rep['size'] = [W, H]
    rows = []
    for (kind, n), lst in sorted(sheets.items()):
        row = ([Image.fromarray((real[n][0].numpy() * 255).astype(np.uint8))] if kind == 'held' else [Image.new('RGB', (W, H), (40, 40, 40))])
        rows.append(row + [Image.fromarray((im.numpy() * 255).astype(np.uint8)) for _, im in lst])
    names = ['real'] + list(drops)
    sh = Image.new('RGB', (W * len(names), H * len(rows) + 24), (0, 0, 0))
    from PIL import ImageDraw
    dd = ImageDraw.Draw(sh)
    for i, nm in enumerate(names): dd.text((i * W + 6, 4), nm + ('' if nm == 'real' else f"  {res[nm]['psnr']:.2f} dB, -{res[nm]['dropped']}"), fill=(255, 255, 255))
    for r_i, row in enumerate(rows):
        for c_i, im in enumerate(row): sh.paste(im, (c_i * W, 24 + r_i * H))
    sh.save(f'{out}/cleanup_sheet.jpg', quality=88)
    for name in ('dual', 'feat', 'both'): np.save(f'{out}/drop_{name}.npy', np.where(drops[name])[0].astype(np.int32))
    rep['minutes'] = round((time.time() - T0) / 60, 1); json.dump(rep, open(f'{out}/cleanup_report.json', 'w'), indent=1)
    say('CLEANUP', json.dumps(rep))


if __name__ == '__main__':
    main(*sys.argv[1:])
