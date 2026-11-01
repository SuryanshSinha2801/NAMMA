"""Two models of one run rendered from the same cameras and compared with each other, CPU only, no photos needed (1 Nov: does the
clean-up, final/solidify.py, cut anything visible? It dropped 69 % of the whole-house model's splats). Cameras: the held-out views
(every 8th, gsplat's split) and off-path views (path cameras raised 40 cm and turned 35 deg, one from above), all with gsplat's own
Parser normalisation, so they match the trained model's space. The run's frames are not needed: placeholder images with the
cameras' size stand in (the Parser reads only the first image's size).
  python model_compare.py <data dir with sparse/0> <out dir> <metres_per_unit> <px> <views> a=model_a.ply b=model_b.ply
(run from gsplat's examples dir) -> <out>/compare_report.json, compare_sheet.jpg (a | b | difference x4, worst views first)"""
import json, os, sys, time, numpy as np, torch
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpu_render
from thin_model import read_model
T0 = time.time()


def say(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def placeholders(data):
    """images/<name> for every camera of sparse/0: one black JPEG at the cameras' size, the rest links to it."""
    cams, ims, _ = read_model(f'{data}/sparse/0'); d = f'{data}/images'
    if os.path.isdir(d) and len(os.listdir(d)) >= len(ims): return
    os.makedirs(d, exist_ok=True); _, w, h, _ = next(iter(cams.values())); names = sorted(im['name'] for im in ims.values())
    first = f'{d}/{names[0]}'; os.makedirs(os.path.dirname(first), exist_ok=True); Image.new('RGB', (int(w), int(h))).save(first, quality=50)
    for n in names[1:]:
        p = f'{d}/{n}'; os.makedirs(os.path.dirname(p), exist_ok=True)
        if not os.path.exists(p): os.link(first, p)


def cameras(data, long, nviews, X, op, mpu):
    sys.path.insert(0, os.getcwd()); from datasets.colmap import Parser
    p = Parser(data, factor=1, normalize=True, test_every=8); cid = p.camera_ids[0]
    W0, H0 = p.imsize_dict[cid]; k = long / max(W0, H0); W, H = round(W0 * k), round(H0 * k); K = p.Ks_dict[cid].copy(); K[:2] *= k
    C = p.camtoworlds; held = [i for i in range(len(C)) if i % 8 == 0]
    held = [held[j] for j in np.linspace(0, len(held) - 1, min(nviews, len(held))).astype(int)]
    up = -np.median(C[:, :3, 1], 0); up /= np.linalg.norm(up)                                    # as final/tour_fidelity.py
    kx = np.array([[0, -up[2], up[1]], [up[2], 0, -up[0]], [-up[1], up[0], 0]]); a = np.radians(35)
    Rr = np.cos(a) * np.eye(3) + np.sin(a) * kx + (1 - np.cos(a)) * np.outer(up, up); off = []
    for j in np.linspace(0, len(C) - 1, 10).astype(int)[1:-1]:
        c = C[j].copy(); c[:3, 3] += up * 0.4 / mpu; c[:3, :3] = Rr @ c[:3, :3]; off.append(c)
    ctr = np.median(X[op > 0.3], 0); eye = ctr + up * 2.6 / mpu - C[:, :3, 2].mean(0) / mpu
    f = ctr - eye; f /= np.linalg.norm(f); r = np.cross(f, up); r /= np.linalg.norm(r)
    top = np.eye(4); top[:3, :3] = np.stack([r, np.cross(f, r), f], 1); top[:3, 3] = eye; off.append(top)
    return K, W, H, [('held', i, C[i]) for i in held] + [('off', n, c) for n, c in enumerate(off)]


def psnr(a, b): return 10 * np.log10(1 / max(float(((a - b) ** 2).mean()), 1e-10))


def main(data, out, mpu, px, views, va, vb):
    os.makedirs(out, exist_ok=True); placeholders(data)
    (na, pa), (nb, pb) = va.split('=', 1), vb.split('=', 1)
    A = cpu_render.load_ply(pa); K, W, H, cams = cameras(data, int(px), int(views), A['means'].double().numpy(), A['opacities'].numpy(), float(mpu))
    say(len(cams), 'cameras at', W, 'x', H, '|', na, len(A['means']), 'splats')
    ra = [cpu_render.render(A, c, K, W, H) for _, _, c in cams]; nA = len(A['means']); del A; say(na, 'rendered')
    B = cpu_render.load_ply(pb); rb = [cpu_render.render(B, c, K, W, H) for _, _, c in cams]; nB = len(B['means']); del B; say(nb, 'rendered')
    rows = []
    for (kind, i, _), x, y in zip(cams, ra, rb):
        d = (x - y).abs().max(2).values
        rows.append(dict(kind=kind, view=int(i), psnr=round(psnr(x, y), 2), changed_px=round(float((d > 0.1).float().mean()), 4)))
    agg = lambda k: dict(psnr_mean=round(float(np.mean([r['psnr'] for r in rows if r['kind'] == k])), 2),
                         psnr_min=round(float(np.min([r['psnr'] for r in rows if r['kind'] == k])), 2),
                         changed_px_mean=round(float(np.mean([r['changed_px'] for r in rows if r['kind'] == k])), 4),
                         changed_px_max=round(float(np.max([r['changed_px'] for r in rows if r['kind'] == k])), 4))
    rep = dict(a=na, b=nb, splats={na: nA, nb: nB}, size=[W, H], held_out=agg('held'), off_path=agg('off'), views=rows,
               minutes=round((time.time() - T0) / 60, 1))
    json.dump(rep, open(f'{out}/compare_report.json', 'w'), indent=1)
    worst = sorted(range(len(rows)), key=lambda j: rows[j]['psnr'])[:8]                              # the most different views
    sh = Image.new('RGB', (3 * W, 24 + H * len(worst))); dr = ImageDraw.Draw(sh)
    for ci, t in enumerate([na, nb, 'difference x4']): dr.text((ci * W + 6, 5), t, fill=(255, 255, 255))
    for ri, j in enumerate(worst):
        x, y = ra[j], rb[j]
        for ci, im in enumerate([x, y, ((x - y).abs() * 4).clamp(0, 1)]):
            sh.paste(Image.fromarray((im.numpy() * 255).astype(np.uint8)), (ci * W, 24 + ri * H))
        dr.text((6, 24 + ri * H + 4), f"{rows[j]['kind']} {rows[j]['view']}: {rows[j]['psnr']} dB", fill=(255, 255, 0))
    sh.save(f'{out}/compare_sheet.jpg', quality=88)
    say('COMPARE', json.dumps({k: v for k, v in rep.items() if k != 'views'}))


if __name__ == '__main__':
    main(*sys.argv[1:])
