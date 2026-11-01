"""3DGS .ply -> .splat (32 bytes per splat: position, scale, RGBA, rotation; SH degree 0 colour), about 8x smaller, opens in SuperSplat.
Also levels the model: the SfM world is not gravity-aligned (up was 8 deg off), so with nav.json (up, floor) the model is rotated so
the floor is level at y = 0 and centred. Usage: python to_splat.py in.ply|in.splat out.splat [nav.json]  (a .splat input is only levelled)"""
import json, os, sys, numpy as np

REC = [('p', '<f4', 3), ('s', '<f4', 3), ('c', 'u1', 4), ('r', 'u1', 4)]

def qmul(a, b):
    """Hamilton product, (w, x, y, z), row-wise."""
    w1, x1, y1, z1 = a.T; w2, x2, y2, z2 = b.T
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], 1)

def level_matrix(nav, p):
    """4x4 M with p' = R p + t: the floor normal (nav up, SfM frame; our own .ply had y and z flipped, gsplat's simple_trainer
    output does not: nav 'flipped' false) becomes +y, the floor goes to y = 0 and the middle of the points p to x = z = 0."""
    n = json.load(open(nav)) if isinstance(nav, str) else nav
    u = np.array(n['up'], float) * (np.array([1, -1, -1]) if n.get('flipped', True) else 1); u /= np.linalg.norm(u); y = np.array([0., 1, 0])
    v = np.cross(u, y); c = float(u @ y); K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + K + K @ K / (1 + c)                                               # rotation taking u to +y
    med = np.median((p @ R.T)[:, [0, 2]], 0); M = np.eye(4); M[:3, :3] = R; M[:3, 3] = [-med[0], -n['floor'], -med[1]]
    return M

def level(rec, nav):
    """Level the splats with level_matrix and write level.json next to nav.json (the mesh and the tour use the same M)."""
    M = level_matrix(nav, rec['p']); R = M[:3, :3]; rec['p'] = rec['p'] @ R.T + M[:3, 3]
    json.dump(dict(matrix=M.tolist()), open(os.path.join(os.path.dirname(os.path.abspath(nav)), 'level.json'), 'w'))
    w = np.sqrt(max(1 + R[0, 0] + R[1, 1] + R[2, 2], 1e-9)) / 2
    qr = np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    q = (rec['r'].astype(np.float32) - 128) / 128; q = qmul(np.repeat(qr[None], len(q), 0), q); q /= np.linalg.norm(q, axis=1, keepdims=True)
    rec['r'] = np.clip(q * 128 + 128, 0, 255).astype(np.uint8)
    print(f'levelled: up was {np.degrees(np.arccos(np.clip(R[1, 1], -1, 1))):.1f} deg off vertical; floor at y = 0', flush=True)
    return rec

def convert(src, dst, nav=None):
    if src.endswith('.splat'):
        rec = np.fromfile(src, dtype=REC); rec = level(rec, nav) if nav else rec; rec.tofile(dst); print(f'{src} -> {dst}'); return
    from plyfile import PlyData                                                     # only needed for .ply input
    v = PlyData.read(src)['vertex'].data; n = len(v)
    pos = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float32)
    scale = np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], 1)).astype(np.float32)
    rgb = 0.5 + 0.28209479 * np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], 1); a = 1 / (1 + np.exp(-v['opacity']))
    col = (np.clip(np.concatenate([rgb, a[:, None]], 1), 0, 1) * 255).astype(np.uint8)
    q = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], 1); q /= np.linalg.norm(q, axis=1, keepdims=True)
    rot = np.clip(q * 128 + 128, 0, 255).astype(np.uint8)
    o = np.argsort(-scale.prod(1) * a)                                                # big, opaque splats first (format convention)
    rec = np.zeros(n, dtype=[('p', '<f4', 3), ('s', '<f4', 3), ('c', 'u1', 4), ('r', 'u1', 4)])
    rec['p'], rec['s'], rec['c'], rec['r'] = pos[o], scale[o], col[o], rot[o]
    if nav: rec = level(rec, nav)
    rec.tofile(dst)
    print(f'{src}: {n} splats -> {dst} ({rec.nbytes / 1e6:.0f} MB)', flush=True)

if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
