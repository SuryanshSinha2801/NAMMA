"""Depth + surface-direction guide for gsplat's simple_trainer (variant kind 'geo'), against the owner's three complaints on run 4:
smudged plain walls, floaters, and smears off the filmed path. The splat alone has nothing that says where a blank wall is, so blobs
sit a little in front of or behind it. Here every training frame gets a dense depth map from Depth Anything 3 (the giant model
v3/run.py already used for the feed-forward tests). Each map is put on the SfM scale with that frame's own SfM points; frames whose
points disagree with it are dropped. The trainer is then pulled toward that depth (disparity L1, like gsplat's own sparse depth loss,
but for every pixel) and toward the same surface directions. Normals come from both depth maps by one finite-difference rule, so
their axis convention cannot disagree; edges, where the depth jumps, are skipped (DN-Splatter / TIDI-GS idea, soft and weighted).
  python geo_prior.py <data dir (images/ + sparse/0)> <out dir>   (run from gsplat's examples dir; writes <name>.npy + geo_report.json)
Trainer side: v3/run.py patches simple_trainer.py once (patch()); GEO_PRIOR=1 GEO_DIR=<out dir> turns it on per variant.
Weights are guesses until measured: GEO_DEPTH (x scene scale, like depth_lambda), GEO_NORMAL, GEO_START."""
import glob, json, os, sys, numpy as np, torch
import torch.nn.functional as F
LONG = 480                                                                                     # stored map size (long side)


def normals(z, K):
    """Camera-space unit normals [h-2, w-2, 3] of a depth map z [h, w] with intrinsics K (same rule for both maps)."""
    h, w = z.shape; v, u = torch.meshgrid(torch.arange(h, device=z.device, dtype=z.dtype), torch.arange(w, device=z.device, dtype=z.dtype), indexing='ij')
    P = torch.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
    return F.normalize(torch.cross(P[1:-1, 2:] - P[1:-1, :-2], P[2:, 1:-1] - P[:-2, 1:-1], dim=-1), dim=-1)


class GeoPrior:
    def __init__(self, parser, trainset, scene_scale, device='cuda'):
        e = os.environ.get; d = e('GEO_DIR')
        self.wd, self.wn, self.start = float(e('GEO_DEPTH', 0.01)), float(e('GEO_NORMAL', 0.05)), int(e('GEO_START', 2000))
        self.scale = scene_scale; self.names = [parser.image_names[i] for i in trainset.indices]; self.maps = {}
        for n in set(self.names):
            f = f'{d}/{n}.npy'
            if os.path.exists(f): self.maps[n] = torch.from_numpy(np.load(f)).to(device)       # float16, ~0.25 MB each
        print(f'GEO PRIOR {len(self.maps)}/{len(set(self.names))} training frames have depth; depth w {self.wd} x scale {self.scale:.3f}, '
              f'normal w {self.wn}, from step {self.start}', flush=True)

    def __call__(self, depth_map, data, step):
        if depth_map is None or step < self.start: return 0.0
        m = self.maps.get(self.names[int(data['image_id'][0])])
        if m is None: return 0.0
        m = m.float(); h, w = m.shape; H, W = depth_map.shape[1:3]
        r = F.interpolate(depth_map[..., 0][None], size=(h, w), mode='bilinear', align_corners=False)[0, 0]
        K = data['K'][0].to(m.device).float().clone(); K[0] *= w / W; K[1] *= h / H
        ok = (m > 0) & (r > 1e-4)
        if ok.sum() < 100: return 0.0
        ld = ((1 / r.clamp(min=1e-4) - 1 / m.clamp(min=1e-4)).abs()[ok]).mean() * self.scale
        gx, gy = (m[1:-1, 2:] - m[1:-1, :-2]).abs(), (m[2:, 1:-1] - m[:-2, 1:-1]).abs()           # skip depth edges (> 5% jump)
        flat = ok[1:-1, 1:-1] & ok[1:-1, 2:] & ok[1:-1, :-2] & ok[2:, 1:-1] & ok[:-2, 1:-1] & (torch.maximum(gx, gy) < 0.05 * m[1:-1, 1:-1])
        ln = (1 - (normals(r, K) * normals(m, K)).sum(-1))[flat].mean() if flat.sum() > 100 else 0.0
        return self.wd * ld + self.wn * ln


def patch(trainer_py):
    """Insert the prior into gsplat 1.5.3's examples/simple_trainer.py (idempotent; does nothing unless GEO_PRIOR=1 at run time).
    The depth map has to be kept before the sparse depth loss reuses the name `depths` for its [1, M] samples."""
    s = open(trainer_py).read()
    if 'geo_prior' in s: return True
    a = '        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale\n'
    b = '            else:\n                colors, depths = renders, None\n'
    c = '                render_mode="RGB+ED" if cfg.depth_loss else "RGB",\n'
    d = '            loss.backward()\n'
    if any(s.count(x) != 1 for x in (a, b, c, d)): return False
    s = s.replace(a, a + "        self.geo_prior = None\n        if os.environ.get('GEO_PRIOR'):\n"
                         "            import geo_prior; self.geo_prior = geo_prior.GeoPrior(self.parser, self.trainset, self.scene_scale, self.device)\n")
    s = s.replace(b, b + '            depth_map = depths\n')
    s = s.replace(c, c.replace('if cfg.depth_loss else', "if (cfg.depth_loss or os.environ.get('GEO_PRIOR')) else"))
    s = s.replace(d, '            if self.geo_prior is not None: loss = loss + self.geo_prior(depth_map, data, step)\n' + d)
    if 'import os' not in s: s = 'import os\n' + s
    open(trainer_py, 'w').write(s); return True


def prepare(data_dir, out, chunk=8):
    """DA3 depth for every frame, undistorted like gsplat's loader does, on the SfM scale of that frame's own points."""
    import cv2, gc, densify
    from datasets.colmap import Parser
    p = Parser(data_dir, factor=1, normalize=True, test_every=8); os.makedirs(f'{out}/und', exist_ok=True)
    paths = []
    for name, path, cid in zip(p.image_names, p.image_paths, p.camera_ids):
        im = cv2.imread(path)
        if cid in p.mapx_dict:                                                                 # SIMPLE_RADIAL -> pinhole, cropped
            im = cv2.remap(im, p.mapx_dict[cid], p.mapy_dict[cid], cv2.INTER_LINEAR); x, y, w, h = p.roi_undist_dict[cid]; im = im[y:y + h, x:x + w]
        q = f'{out}/und/{len(paths):05d}.jpg'; cv2.imwrite(q, im, [cv2.IMWRITE_JPEG_QUALITY, 95]); paths.append(q)
    model = densify.da3(); rep = dict(frames=len(paths), kept=0, dropped=[], spread=[])
    for a in range(0, len(paths), chunk):
        with torch.no_grad(): pred = model.inference(paths[a:a + chunk])
        for j, z in zip(range(a, a + chunk), np.asarray(pred.depth)):
            name, cid = p.image_names[j], p.camera_ids[j]; W, H = p.imsize_dict[cid]; K = p.Ks_dict[cid]
            w2c = np.linalg.inv(p.camtoworlds[j]); pts = p.points[p.point_indices.get(name, np.zeros(0, int))]
            pc = pts @ w2c[:3, :3].T + w2c[:3, 3]; pc = pc[pc[:, 2] > 1e-6]
            hz, wz = z.shape; u = (K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]) * wz / W; v = (K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]) * hz / H
            ins = (u >= 0) & (u < wz - 1) & (v >= 0) & (v < hz - 1); d = z[v[ins].astype(int), u[ins].astype(int)]; r = pc[ins, 2][d > 0] / d[d > 0]
            if len(r) < 30: rep['dropped'].append([name, int(len(r))]); continue
            s = float(np.median(r)); spread = float(np.median(np.abs(np.log(r / s))))
            if spread > 0.15: rep['dropped'].append([name, round(spread, 3)]); continue          # DA3 disagrees with the SfM points
            k = LONG / max(W, H); zz = cv2.resize((z * s).astype(np.float32), (round(W * k), round(H * k)), interpolation=cv2.INTER_LINEAR)
            np.save(f'{out}/{name}.npy', zz.astype(np.float16)); rep['kept'] += 1; rep['spread'].append(spread)
        del pred; gc.collect(); torch.cuda.empty_cache(); print(f'{min(a + chunk, len(paths))}/{len(paths)} frames, {rep["kept"]} kept', flush=True)
    rep['median_spread'] = round(float(np.median(rep.pop('spread') or [0])), 4)
    json.dump(rep, open(f'{out}/geo_report.json', 'w'), indent=1); print('GEO', json.dumps({k: v for k, v in rep.items() if k != 'dropped'}), flush=True)
    return rep


if __name__ == '__main__':
    prepare(sys.argv[1], sys.argv[2])
