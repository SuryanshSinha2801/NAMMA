"""How much 3D quality the published tour loses on its way from the trained model, CPU only (no GPU hours, 27 Oct). The tour does
three things to the model: removes spiky smears (make_tour.smear_mask), keeps view-dependent colour only up to SH band 1, and
compresses it (PlayCanvas compressed PLY). Each variant is rendered with cpu_render.py from the held-out frames (every 8th, never
trained on) and compared with the real photos; plus off-path views (raised / turned cameras, from above) where smears show.
  python tour_fidelity.py <data dir> <out dir> <metres_per_unit> <px> <views> name=model.ply [name=model.ply ...]
(run from gsplat's examples dir; <out dir>/sizes.json with {name: bytes shipped} is added to the report if present)"""
import json, os, sys, time, numpy as np, torch
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpu_render
from cleanup_cpu import ssim
T0 = time.time()


def say(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def cameras(data, long, nviews, X, op, mpu):
    """Held-out frames (real photo + camera) at <long> px, and off-path cameras: 3 path cameras raised 40 cm and turned 35 deg,
    plus one from above the room."""
    sys.path.insert(0, os.getcwd()); from datasets.colmap import Parser, Dataset
    p = Parser(data, factor=1, normalize=True, test_every=8); ds = Dataset(p, split='val'); cid = p.camera_ids[0]
    W0, H0 = p.imsize_dict[cid]; k = long / max(W0, H0); W, H = round(W0 * k), round(H0 * k); K = p.Ks_dict[cid].copy(); K[:2] *= k
    real = []
    for j in np.linspace(0, len(ds) - 1, min(nviews, len(ds))).astype(int):
        d = ds[int(j)]; im = Image.fromarray(d['image'].numpy().astype(np.uint8)).resize((W, H), Image.LANCZOS)
        real.append((torch.tensor(np.asarray(im), dtype=torch.float32) / 255, d['camtoworld'].numpy()))
    C = p.camtoworlds; up = -np.median(C[:, :3, 1], 0); up /= np.linalg.norm(up); off = []
    kx = np.array([[0, -up[2], up[1]], [up[2], 0, -up[0]], [-up[1], up[0], 0]]); a = np.radians(35)
    Rr = np.cos(a) * np.eye(3) + np.sin(a) * kx + (1 - np.cos(a)) * np.outer(up, up)
    for j in np.linspace(0, len(C) - 1, 5).astype(int)[1:-1]:
        c = C[j].copy(); c[:3, 3] += up * 0.4 / mpu; c[:3, :3] = Rr @ c[:3, :3]; off.append(c)
    ctr = np.median(X[op > 0.3], 0); eye = ctr + up * 2.6 / mpu - C[:, :3, 2].mean(0) / mpu
    f = ctr - eye; f /= np.linalg.norm(f); r = np.cross(f, up); r /= np.linalg.norm(r)
    top = np.eye(4); top[:3, :3] = np.stack([r, np.cross(f, r), f], 1); top[:3, 3] = eye; off.append(top)
    return K, W, H, real, off


def main(data, out, mpu, px, views, *variants):
    os.makedirs(out, exist_ok=True); mpu = float(mpu)
    vs = [v.split('=', 1) for v in variants]; first = cpu_render.load_ply(vs[0][1])
    K, W, H, real, off = cameras(data, int(px), int(views), first['means'].double().numpy(), first['opacities'].numpy(), mpu)
    sizes = json.load(open(f'{out}/sizes.json')) if os.path.exists(f'{out}/sizes.json') else {}
    res, tiles = {}, {}
    for name, path in vs:
        s = first if path == vs[0][1] else cpu_render.load_ply(path); ps, ss = [], []
        for n, (img, c2w) in enumerate(real):
            r = cpu_render.render(s, c2w, K, W, H); ps.append(10 * np.log10(1 / max(float(((r - img) ** 2).mean()), 1e-10))); ss.append(ssim(r, img))
            if n < 3: tiles.setdefault(('held', n), []).append(r)
        dark = []
        for n, c2w in enumerate(off):
            im = cpu_render.render(s, c2w, K, W, H); tiles.setdefault(('off', n), []).append(im)
            dark.append(float((im.mean(2) < 0.05).float().mean()))                             # black / empty share off the walk
        res[name] = dict(psnr=round(float(np.mean(ps)), 3), ssim=round(float(np.mean(ss)), 4), splats=int(len(s['means'])),
                         off_dark=round(float(np.mean(dark)), 4),
                         sh_bands=int(round(np.sqrt(s['sh'].shape[1]))) - 1, shipped_mb=round(sizes[name] / 1e6, 1) if name in sizes else None)
        say(name, res[name]); del s
    base = res[vs[0][0]]['psnr']
    for r_ in res.values(): r_['vs_first_db'] = round(r_['psnr'] - base, 3)
    names = ['real photo'] + [n for n, _ in vs]; rows = sorted(tiles)
    sh = Image.new('RGB', (W * len(names), 24 + H * len(rows))); dr = ImageDraw.Draw(sh)
    for i, nm in enumerate(names):
        t = nm if i == 0 else f"{nm}: {res[nm]['psnr']:.2f} dB" + (f", {res[nm]['shipped_mb']} MB" if res[nm]['shipped_mb'] else '')
        dr.text((i * W + 6, 5), t, fill=(255, 255, 255))
    for ri, key in enumerate(rows):
        first_tile = Image.fromarray((real[key[1]][0].numpy() * 255).astype(np.uint8)) if key[0] == 'held' else Image.new('RGB', (W, H), (40, 40, 40))
        for ci, im in enumerate([first_tile] + [Image.fromarray((t.numpy() * 255).astype(np.uint8)) for t in tiles[key]]):
            sh.paste(im, (ci * W, 24 + ri * H))
    sh.save(f'{out}/fidelity_sheet.jpg', quality=90)
    rep = dict(heldout=res, views=len(real), size=[W, H], minutes=round((time.time() - T0) / 60, 1))
    json.dump(rep, open(f'{out}/fidelity_report.json', 'w'), indent=1); say('FIDELITY', json.dumps(rep))


if __name__ == '__main__':
    main(*sys.argv[1:])
