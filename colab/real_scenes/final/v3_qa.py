"""v3 quality check after gsplat's simple_trainer, so nothing reaches the owner unchecked:
  metrics.json  held-out PSNR / SSIM / LPIPS from the trainer's own eval (every 8th frame never trained on; exposure-fair = after the
                trainer's colour match, the number to compare with our earlier ibr measurements), splat count, needle fraction
                (longest / middle axis > 10: smears; thin flat discs are normal and only counted as flat_fraction), pass = exposure-fair PSNR >= 28 dB and < 10 % needles;
  qa_sheet.jpg  row 1: three held-out frames, real | model;  row 2: two views 50 cm-equivalent beside the walk, then a dollhouse
                (room box only, ceiling cut) from outside above and from the top: missing walls, floaters and smears show up here;
  qa_detail.jpg 100 % crops of the most textured part of each held-out frame, real | model (fine-detail loss)
  model.splat   levelled (floor at y = 0, up = +y) for SuperSplat, plus model.ply.
Runs with the system python inside gsplat's examples folder (needs its Parser).
Usage: python v3_qa.py <examples dir> <data dir> <result dir> <out dir>"""
import glob, json, os, sys, numpy as np, torch
from PIL import Image

def camera_up(c2w):
    """Most-perpendicular direction to all camera x axes (phones have ~no roll), signed like the image-up axes (= fuse_nav.camera_up)."""
    x = c2w[:, :3, 0]; u = np.linalg.eigh(x.T @ x)[1][:, 0]
    return u if u @ (-c2w[:, :3, 1].mean(0)) > 0 else -u

def dollhouse(means, pts, C, up):
    """Splats inside the room (horizontal box from the 3rd-97th percentile of SfM points along their two main horizontal axes, +10 %)
    and below a ceiling cut (camera height + a third of camera-to-floor height). Returns (mask, room centre, room radius)."""
    h = pts @ up; floor = np.percentile(h, 2); cam_h = np.median(C[:, :3, 3] @ up); cut = cam_h + 0.33 * (cam_h - floor)
    flat = pts - np.outer(h, up); m = flat.mean(0); ax = np.linalg.svd(flat - m, full_matrices=False)[2][:2]
    q = (flat - m) @ ax.T; lo, hi = np.percentile(q, 3, 0), np.percentile(q, 97, 0); pad = 0.1 * (hi - lo); lo, hi = lo - pad, hi + pad
    hm = means @ up; qm = (means - np.outer(hm, up) - m) @ ax.T
    keep = np.all((qm >= lo) & (qm <= hi), 1) & (hm <= cut) & (hm >= floor - 0.1 * (cam_h - floor))
    ctr = m + ((lo + hi) / 2) @ ax + (floor + (cut - floor) / 2) * up
    return keep, ctr, 0.5 * float(np.linalg.norm(hi - lo))

def look_at(eye, target, up):
    """OpenCV camera (x right, y down, z forward) at eye looking at target."""
    f = target - eye; f /= np.linalg.norm(f); r = np.cross(f, up); r /= np.linalg.norm(r); d = np.cross(f, r)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = r, d, f, eye; return M

def main(examples, data, result, out):
    sys.path.insert(0, examples); from datasets.colmap import Parser, Dataset
    from gsplat.rendering import rasterization
    os.makedirs(out, exist_ok=True); p = Parser(data, factor=1, normalize=True, test_every=8)
    ck = sorted(glob.glob(f'{result}/ckpts/ckpt_*_rank0.pt'), key=lambda f: int(f.split('_')[-2]))[-1]; step = int(ck.split('_')[-2])
    s = torch.load(ck, map_location='cuda' if torch.cuda.is_available() else 'cpu')['splats']; dev = s['means'].device
    cfg = open(f'{result}/cfg.yml').read() if os.path.exists(f'{result}/cfg.yml') else ''
    mode = 'antialiased' if 'antialiased: true' in cfg else 'classic'
    colors = torch.cat([s['sh0'], s['shN']], 1); shd = int(round(np.sqrt(colors.shape[1]))) - 1

    def render(c2w, K, W, H, keep=None):
        vm = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=dev)[None]; Kt = torch.tensor(K, dtype=torch.float32, device=dev)[None]
        k = slice(None) if keep is None else torch.as_tensor(keep, device=dev)
        img, alpha, _ = rasterization(s['means'][k], s['quats'][k], torch.exp(s['scales'][k]), torch.sigmoid(s['opacities'][k]), colors[k],
                                      vm, Kt, W, H, sh_degree=shd, rasterize_mode=mode)
        return img[0].clamp(0, 1).cpu().numpy(), alpha[0, ..., 0].cpu().numpy()

    C = p.camtoworlds; names = p.image_names; cid = p.camera_ids[0]; K = p.Ks_dict[cid].copy(); W, H = p.imsize_dict[cid]
    vs = Dataset(p, split='val'); val = list(vs.indices); trn = [i for i in range(len(names)) if i % 8]
    tiles = []
    for k in np.linspace(0, len(val) - 1, min(3, len(val))).astype(int):            # Dataset undistorts like the trainer did
        d = vs[k]; gt = d['image'].numpy() / 255; i = val[k]
        tiles.append(np.concatenate([gt, render(C[i], d['K'].numpy(), gt.shape[1], gt.shape[0])[0]], 1))
    row1 = np.concatenate(tiles, 1)
    # qa_detail.jpg: 100 % crops (no downscaling) of the most textured 480 px square of each held-out frame, real | model, so
    # fine-detail loss on well-filmed views shows (the owner's complaint after run 4), not hidden by the sheet's size
    det = []
    for tl in tiles:
        g, m_ = tl[:, :tl.shape[1] // 2], tl[:, tl.shape[1] // 2:]; c = min(480, g.shape[0], g.shape[1])
        e = np.abs(np.diff(g.mean(2), axis=0))[:, :-1] + np.abs(np.diff(g.mean(2), axis=1))[:-1]       # texture energy
        ii = np.cumsum(np.cumsum(np.pad(e, ((1, 0), (1, 0))), 0), 1); y, x = np.meshgrid(np.arange(0, e.shape[0] - c, 16), np.arange(0, e.shape[1] - c, 16), indexing='ij')
        tot = ii[y + c, x + c] - ii[y, x + c] - ii[y + c, x] + ii[y, x]; y0, x0 = y.flat[tot.argmax()], x.flat[tot.argmax()]
        det.append(np.concatenate([g[y0:y0 + c, x0:x0 + c], m_[y0:y0 + c, x0:x0 + c]], 1))
    Image.fromarray((np.concatenate(det, 0) * 255).astype(np.uint8)).save(f'{out}/qa_detail.jpg', quality=92)
    up = camera_up(C); pts = p.points
    side = 0.15 * np.linalg.norm(C[:, :3, 3] - C[:, :3, 3].mean(0), axis=1).max()  # ~50 cm in a small room
    row2, alphas = [], []
    for i in [trn[len(trn) // 3], trn[2 * len(trn) // 3]]:
        c = C[i].copy(); c[:3, 3] += side * c[:3, 0]; im, a = render(c, K, W, H); row2.append(im); alphas.append(a.mean())
    fwd = C[:, :3, 2].mean(0); fwd -= (fwd @ up) * up; fwd /= np.linalg.norm(fwd) + 1e-9
    # overview shots = dollhouse: only splats inside the room box (else the shell of big splats that models the view through the
    # windows hides everything, run 2) and below the ceiling cut (else the top-down shows the ceiling's back)
    room, ctr, span = dollhouse(s["means"].cpu().numpy(), pts, C, up)
    Ko = K.copy(); Ko[0, 0] = Ko[1, 1] = 0.6 * max(W, H)                             # wider lens for the overview shots
    for eye in (ctr - 1.6 * span * fwd + 0.9 * span * up, ctr + 2.2 * span * up + 0.05 * span * fwd):
        row2.append(render(look_at(eye, ctr, up if abs((ctr - eye) @ up) < 0.99 * np.linalg.norm(ctr - eye) else fwd), Ko, W, H, room)[0])
    row2 = np.concatenate(row2, 1)
    wmax = max(row1.shape[1], row2.shape[1]); pad = lambda r: np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0)))
    sheet = np.concatenate([pad(row1), pad(row2)], 0)
    Image.fromarray((sheet * 255).astype(np.uint8)).save(f'{out}/qa_sheet.jpg', quality=88)
    st = sorted(glob.glob(f'{result}/stats/val_step*.json')); m = json.load(open(st[-1])) if st else {}
    sc = torch.exp(s['scales']).sort(1, descending=True).values                     # long, middle, short axis
    needle = float((sc[:, 0] / sc[:, 1] > 10).float().mean())                        # long in ONE direction only (smears)
    flat = float((sc[:, 1] / sc[:, 2] > 10).float().mean())                          # thin discs: normal on walls, not a defect
    r = lambda k, n=2: round(m[k], n) if k in m else None
    metrics = dict(step=step + 1, splats=int(s['means'].shape[0]), heldout_views=len(val), train_views=len(trn),
                   heldout_psnr=r('psnr'), heldout_ssim=r('ssim', 3), heldout_lpips=r('lpips', 3),
                   heldout_psnr_exposure_fair=r('cc_psnr'), heldout_ssim_exposure_fair=r('cc_ssim', 3),   # bilateral grid on: colour-matched
                   needle_fraction=round(needle, 3), flat_fraction=round(flat, 3), offpath_coverage=round(float(np.mean(alphas)), 3))
    metrics['pass'] = bool((metrics['heldout_psnr_exposure_fair'] or metrics['heldout_psnr'] or 0) >= 28 and needle < 0.1)
    json.dump(metrics, open(f'{out}/metrics.json', 'w'), indent=1); print('METRICS', json.dumps(metrics), flush=True)
    floor = float(np.percentile(pts @ up, 2))                                         # floor = lowest dense layer of SfM points
    json.dump(dict(up=up.tolist(), floor=floor, flipped=False), open(f'{out}/nav.json', 'w'))
    ply = sorted(glob.glob(f'{result}/ply/point_cloud_*.ply'), key=lambda f: int(f.split('_')[-1][:-4]))
    if not ply: print('no .ply written (was --save-ply passed?)', flush=True); return metrics
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import to_splat, shutil
    to_splat.convert(ply[-1], f'{out}/model.splat', f'{out}/nav.json'); shutil.copy(ply[-1], f'{out}/model.ply')
    return metrics

if __name__ == '__main__':
    main(*sys.argv[1:5])
