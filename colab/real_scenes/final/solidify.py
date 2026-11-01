"""Make a trained model hold up from new angles without training (owner, 30 Oct: "the structure that was seen must not break from
other angles; nothing outside the house may change"). CPU only, uses the run's own training cameras:
  1 contribution pass: every training view (not the held-out ones) is rendered with cpu_render's math, and each splat collects how
    much it shows (sum of alpha x transmittance over pixels) and from which directions.
  2 clean: splats that never show in any training view are dropped. They are invisible on the walk and only appear from new
    angles (junk behind surfaces, inside furniture). Anything a camera saw is kept, including the mirror room and the view out of
    windows, because the cameras saw those through the glass.
  3 colour: each kept splat's colour is refitted to SH degree 1 over the directions it was actually seen from (least squares on
    its own SH3 colours there) and clamped to stay >= 0 in every direction. SH3 extrapolates to dark or odd colours from unseen
    directions: the black patches.
  4 flat (optional variant): splats that lie on a surface (inside the room, on a plane of their neighbours) are turned so their
    thinnest axis follows the surface normal and squashed, so the surface stays closed from oblique angles.
  python solidify.py <data dir> <model.ply> <out dir> [long px=480] [view step=3]   (run from gsplat's examples dir)
  -> <out>/clean.ply, clean_sh1.ply, clean_sh1_flat.ply, solidify.json"""
import json, os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpu_render
from cpu_render import quat_rot, sh_color, C0, C1
T0 = time.time()


def say(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


@torch.no_grad()
def contribution(s, c2w, K, W, H, wsum, dsum, tile=16, near=0.01, eps2d=0.3):
    """cpu_render.render's projection and compositing, but instead of an image it adds each splat's shown weight to wsum and its
    weighted viewing direction (camera -> splat) to dsum."""
    c2w = torch.as_tensor(c2w, dtype=torch.float32); K = torch.as_tensor(K, dtype=torch.float32)
    w2c = torch.linalg.inv(c2w); R, t = w2c[:3, :3], w2c[:3, 3]
    pc = s['means'] @ R.T + t; z = pc[:, 2]; ok = z > near
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * pc[:, 0] / z + cx; v = fy * pc[:, 1] / z + cy
    ok &= (u > -0.5 * W) & (u < 1.5 * W) & (v > -0.5 * H) & (v < 1.5 * H)
    idx = ok.nonzero()[:, 0]; pc, z, u, v = pc[idx], z[idx], u[idx], v[idx]
    M = quat_rot(s['quats'][idx]) * s['scales'][idx][:, None, :]; Sig = M @ M.transpose(1, 2)
    tfx, tfy = 0.5 * W / fx, 0.5 * H / fy
    tx = z * (pc[:, 0] / z).clamp(-(cx / fx + 0.3 * tfx), (W - cx) / fx + 0.3 * tfx)
    ty = z * (pc[:, 1] / z).clamp(-(cy / fy + 0.3 * tfy), (H - cy) / fy + 0.3 * tfy)
    J = torch.zeros(len(idx), 2, 3); J[:, 0, 0] = fx / z; J[:, 0, 2] = -fx * tx / z ** 2; J[:, 1, 1] = fy / z; J[:, 1, 2] = -fy * ty / z ** 2
    T = J @ R; cov = T @ Sig @ T.transpose(1, 2); cov[:, 0, 0] += eps2d; cov[:, 1, 1] += eps2d
    a, b, c = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]; det = a * c - b * b; good = det > 0
    mid = 0.5 * (a + c); lam = mid + torch.sqrt((mid * mid - det).clamp(min=0.01)); r = torch.ceil(3 * torch.sqrt(lam))
    good &= (u + r > 0) & (u - r < W) & (v + r > 0) & (v - r < H) & (r > 0)
    g = good.nonzero()[:, 0]; idx, z, u, v, r = idx[g], z[g], u[g], v[g], r[g]
    ia, ib, ic = (c[g] / det[g]), (-b[g] / det[g]), (a[g] / det[g]); op = s['opacities'][idx]
    dirs = s['means'][idx] - c2w[:3, 3]; dirs = dirs / dirs.norm(dim=1, keepdim=True)
    nx, ny = (W + tile - 1) // tile, (H + tile - 1) // tile
    x0 = ((u - r) // tile).clamp(0, nx - 1).long(); x1 = ((u + r) // tile).clamp(0, nx - 1).long()
    y0 = ((v - r) // tile).clamp(0, ny - 1).long(); y1 = ((v + r) // tile).clamp(0, ny - 1).long()
    cnt = (x1 - x0 + 1) * (y1 - y0 + 1); gid = torch.repeat_interleave(torch.arange(len(idx)), cnt)
    k = torch.arange(len(gid)) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt); w_ = (x1 - x0 + 1)[gid]
    tid = (y0[gid] + k // w_) * nx + (x0[gid] + k % w_)
    order = torch.argsort(z); rank = torch.empty_like(order); rank[order] = torch.arange(len(order))
    so = torch.argsort(tid * len(order) + rank[gid]); tid, gid = tid[so], gid[so]
    bounds = torch.searchsorted(tid, torch.arange(nx * ny + 1)); w_loc = torch.zeros(len(idx))
    py, px = torch.meshgrid(torch.arange(tile, dtype=torch.float32), torch.arange(tile, dtype=torch.float32), indexing='ij')
    for t_ in range(nx * ny):
        s0, s1 = int(bounds[t_]), int(bounds[t_ + 1])
        if s0 == s1: continue
        gg = gid[s0:s1]; ty_, tx_ = divmod(t_, nx)
        X = (px.reshape(-1) + tx_ * tile + 0.5)[:, None]; Y = (py.reshape(-1) + ty_ * tile + 0.5)[:, None]
        dx, dy = X - u[gg][None], Y - v[gg][None]
        pw = -0.5 * (ia[gg] * dx * dx + ic[gg] * dy * dy) - ib[gg] * dx * dy
        al = (op[gg][None] * torch.exp(pw)).clamp(max=0.999); al = torch.where((pw <= 0) & (al >= 1 / 255), al, torch.zeros_like(al))
        Tr = torch.cumprod(torch.cat([torch.ones(len(X), 1), 1 - al[:, :-1]], 1), 1)
        w_loc.index_add_(0, gg, (al * Tr).sum(0))
    wsum.index_add_(0, idx, w_loc); dsum.index_add_(0, idx, w_loc[:, None] * dirs)


def refit_sh1(sh, wsum, dsum, K=13):
    """SH degree 1 fitted to the splat's own colours over the cone of directions it was seen from, then clamped >= 0 everywhere."""
    n = len(sh); out = torch.zeros(n, 4, 3)
    m = dsum / wsum.clamp(min=1e-12)[:, None]; Rl = m.norm(dim=1).clamp(1e-6, 1); m = m / Rl[:, None]
    th = torch.arccos(Rl).clamp(np.radians(8), np.radians(60))                       # spread of the seen directions
    a = torch.where(m[:, 0:1].abs() < 0.9, torch.tensor([[1., 0, 0]]), torch.tensor([[0., 1, 0]]))
    e1 = torch.cross(m, a.expand_as(m), dim=1); e1 = e1 / e1.norm(dim=1, keepdim=True); e2 = torch.cross(m, e1, dim=1)
    dl = [m]
    for f in (0.5, 1.0):
        for j in range(6):
            ph = 2 * np.pi * j / 6 + f; ang = th * f
            dl.append(torch.cos(ang)[:, None] * m + torch.sin(ang)[:, None] * (np.cos(ph) * e1 + np.sin(ph) * e2))
    for i0 in range(0, n, 200_000):
        sl = slice(i0, min(n, i0 + 200_000)); D = torch.stack([d[sl] for d in dl], 1)            # [b, K, 3]
        tgt = torch.stack([sh_color(sh[sl], D[:, j]) for j in range(len(dl))], 1) - 0.5          # [b, K, 3]
        x, y, z = D[..., 0], D[..., 1], D[..., 2]
        A = torch.stack([torch.full_like(x, C0), -C1 * y, C1 * z, -C1 * x], 2)                  # gsplat's SH1 basis
        At = A.transpose(1, 2); coef = torch.linalg.solve(At @ A + 1e-4 * torch.eye(4), At @ tgt) # [b, 4, 3]
        dc = C0 * coef[:, 0] + 0.5; vn = C1 * coef[:, 1:].norm(dim=1)                           # min over the sphere = dc - |v|
        k_ = torch.where(vn > dc - 0.02, ((dc - 0.02).clamp(min=0) / vn.clamp(min=1e-6)), torch.ones_like(vn))
        coef[:, 1:] *= k_[:, None]; out[sl] = coef
    return out


def flatten(X, sc, q, op, inside, k=16):
    """Splats on a surface: thinnest axis turned to the local normal (PCA of opaque neighbours), thickness <= 15 % of the middle
    axis. Only inside the room and only for splats already flat-ish and within their own size of the neighbour plane."""
    from scipy.spatial import cKDTree
    ref = X[(op > 0.5) & inside]; tree = cKDTree(ref); cand = np.nonzero(inside & (op > 0.3))[0]
    _, nb = tree.query(X[cand], k=k); P = ref[nb]; c = P.mean(1); Q = P - c[:, None]
    cov = np.einsum('nki,nkj->nij', Q, Q) / k; w, V = np.linalg.eigh(cov); nrm = V[:, :, 0]
    planar = w[:, 0] < 0.15 * w[:, 1]
    R = quat_np(q[cand]); o = np.argsort(sc[cand], 1); s_sorted = np.take_along_axis(sc[cand], o, 1)
    thin = R[np.arange(len(cand)), :, o[:, 0]]                                                  # thinnest axis (world)
    near = np.abs(((X[cand] - c) * nrm).sum(1)) < s_sorted[:, 1]
    sel = planar & near & (s_sorted[:, 0] < 0.6 * s_sorted[:, 1])
    idx = cand[sel]; a = thin[sel]; nn = nrm[sel] * np.sign((a * nrm[sel]).sum(1, keepdims=True) + 1e-9)
    Rot = rot_between(a, nn); Rn = Rot @ R[sel]; q2 = q.copy(); q2[idx] = mat_to_quat(Rn)
    sc2 = sc.copy(); j = o[sel, 0]; mid = s_sorted[sel, 1]
    sc2[idx, j] = np.minimum(sc2[idx, j], 0.15 * mid)
    return q2, sc2, len(idx)


def quat_np(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True); w, x, y, z = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                     2 * (y * z - w * x), 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


def rot_between(a, b):
    """Rotation matrices taking unit vectors a to b (Rodrigues)."""
    v = np.cross(a, b); s = np.linalg.norm(v, axis=1); c = (a * b).sum(1)
    kx = np.zeros((len(a), 3, 3)); kx[:, 0, 1], kx[:, 0, 2], kx[:, 1, 0] = -v[:, 2], v[:, 1], v[:, 2]
    kx[:, 1, 2], kx[:, 2, 0], kx[:, 2, 1] = -v[:, 0], -v[:, 1], v[:, 0]
    f = ((1 - c) / np.maximum(s * s, 1e-12))[:, None, None]
    return np.eye(3)[None] + kx + kx @ kx * f


def mat_to_quat(R):
    t = np.trace(R, axis1=1, axis2=2); q = np.zeros((len(R), 4))
    w = np.sqrt(np.maximum(0, 1 + t)) / 2; x = np.sqrt(np.maximum(0, 1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2])) / 2
    y = np.sqrt(np.maximum(0, 1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2])) / 2; z = np.sqrt(np.maximum(0, 1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2])) / 2
    x = np.copysign(x, R[:, 2, 1] - R[:, 1, 2]); y = np.copysign(y, R[:, 0, 2] - R[:, 2, 0]); z = np.copysign(z, R[:, 1, 0] - R[:, 0, 1])
    q[:, 0], q[:, 1], q[:, 2], q[:, 3] = w, x, y, z
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def unneedle(log_scales, inside, ratio=4.0):
    """Needle splats (longest axis > ratio x the middle one) inside the room: longest axis capped at ratio x the middle.
    Trained from one side only, they point along the line of sight: head-on (as filmed) they look the same, from the side they
    were the streaks (owner's cupboard by the bathroom, 30 Oct). Walls/floors are thin discs (two long axes): untouched.
    Returns new log scales and how many splats changed."""
    s = np.asarray(log_scales, np.float64).copy(); o = np.argsort(s, 1); mid = np.take_along_axis(s, o[:, 1:2], 1)[:, 0]
    big = o[:, 2]; cap = mid + np.log(ratio); sel = inside & (s[np.arange(len(s)), big] > cap)
    s[np.nonzero(sel)[0], big[sel]] = cap[sel]
    return s.astype(np.float32), int(sel.sum())


def write(v, keep, path, sh1=None, quats=None, scales=None):
    from plyfile import PlyData, PlyElement
    o = v[keep].copy()
    if sh1 is not None:
        nr = len([n for n in o.dtype.names if n.startswith('f_rest_')]) // 3; c = sh1.numpy()
        for ch in range(3):
            o[f'f_dc_{ch}'] = c[:, 0, ch]
            for kk in range(nr): o[f'f_rest_{ch * nr + kk}'] = c[:, kk + 1, ch] if kk < 3 else 0.0
    if quats is not None:
        for i in range(4): o[f'rot_{i}'] = quats[:, i]
    if scales is not None:
        for i in range(3): o[f'scale_{i}'] = np.log(np.maximum(scales[:, i], 1e-9))
    PlyData([PlyElement.describe(o, 'vertex')]).write(path)


def main(data, ply, out, long=480, step=3):
    from plyfile import PlyData
    os.makedirs(out, exist_ok=True); long, step = int(long), int(step)
    sys.path.insert(0, os.getcwd()); from datasets.colmap import Parser
    p = Parser(data, factor=1, normalize=True, test_every=8); cid = p.camera_ids[0]
    W0, H0 = p.imsize_dict[cid]; kf = long / max(W0, H0); W, H = round(W0 * kf), round(H0 * kf); K = p.Ks_dict[cid].copy(); K[:2] *= kf
    train = [i for i in range(len(p.image_names)) if i % 8 != 0][::step]                       # gsplat's split: every 8th held out
    s = cpu_render.load_ply(ply); n = len(s['means']); wsum = torch.zeros(n); dsum = torch.zeros(n, 3)
    say(n, 'splats,', len(train), 'training views at', W, 'x', H)
    for j, i in enumerate(train):
        contribution(s, p.camtoworlds[i], K, W, H, wsum, dsum)
        if j % 20 == 0: say('view', j, 'of', len(train), 'seen so far:', int((wsum > 0).sum()))
    keep = (wsum >= 0.05).numpy()                                                                 # shows at least 1/20 of a pixel somewhere
    v = PlyData.read(ply)['vertex'].data
    rep = dict(splats=n, views=len(train), px=[W, H], never_seen=int((~keep).sum()), kept=int(keep.sum()))
    write(v, keep, f'{out}/clean.ply'); say('clean:', rep)
    if os.environ.get('SOLID_ONLY_CLEAN') == '1':                                                 # the pipeline only ships clean.ply
        json.dump(rep, open(f'{out}/solidify.json', 'w'), indent=1); return
    sh1 = refit_sh1(s['sh'][keep], wsum[keep], dsum[keep]); write(v, keep, f'{out}/clean_sh1.ply', sh1=sh1); say('SH1 refit done')
    pts = p.points; lo, hi = np.percentile(pts, 3, 0), np.percentile(pts, 97, 0); pad = 0.12 * (hi - lo)   # the room (as make_tour)
    X = s['means'][keep].numpy(); inside = np.all((X > lo - pad) & (X < hi + pad), 1)
    q2, sc2, nf = flatten(X, s['scales'][keep].numpy(), s['quats'][keep].numpy(), s['opacities'][keep].numpy(), inside)
    write(v, keep, f'{out}/clean_sh1_flat.ply', sh1=sh1, quats=q2, scales=sc2); rep.update(flattened=nf, inside=int(inside.sum()))
    rep['minutes'] = round((time.time() - T0) / 60, 1); json.dump(rep, open(f'{out}/solidify.json', 'w'), indent=1); say('SOLIDIFY', json.dumps(rep))


def unneedle_file(data, ply, out_ply, ratio=4.0):
    """The cupboard fix on a model file, no training (owner, 1 Nov: the fix that made the hotel's cupboard solid from the side):
    needle splats inside the room (SfM points 1-99 %, +5 %, as fixer_refine) get their longest axis capped at ratio x the middle.
    Everything else is copied unchanged. Returns how many splats changed."""
    from plyfile import PlyData, PlyElement
    sys.path.insert(0, os.getcwd()); from datasets.colmap import Parser
    p = Parser(data, factor=1, normalize=True, test_every=8)
    lo, hi = np.percentile(p.points, 1, 0), np.percentile(p.points, 99, 0); pad = (hi - lo) * 0.05; lo, hi = lo - pad, hi + pad
    v = PlyData.read(ply)['vertex'].data.copy(); X = np.stack([v['x'], v['y'], v['z']], 1); inside = np.all((X > lo) & (X < hi), 1)
    ls, n = unneedle(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], 1), inside, float(ratio))
    for i in range(3): v[f'scale_{i}'] = ls[:, i]
    PlyData([PlyElement.describe(v, 'vertex')]).write(out_ply); say('unneedle:', n, 'of', len(v), 'splats shortened (ratio', ratio, ')')
    return n


if __name__ == '__main__' and sys.argv[1:2] == ['unneedle']:
    unneedle_file(*sys.argv[2:])
elif __name__ == '__main__':
    main(*sys.argv[1:])
