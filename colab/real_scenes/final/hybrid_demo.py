"""Hybrid 'solid room' demo (CPU): a smear-free model of a finished run, built automatically from the same data, no manual modelling.
  shell      floor, ceiling and walls fitted to the run's TSDF mesh (mesh_tsdf.py, levelled) in the room's own axes: the floor
             footprint (floor + furniture + camera path, closed) is traced and its outline extruded to the ceiling height
  furniture  the TSDF mesh minus everything within 4 cm of the shell (no double walls), kept as solid surfaces
  textures   every shell texel (1 cm) and furniture vertex is coloured from the REAL training photos that see it best (frontal,
             close), visibility checked by ray casting the shell + furniture (Open3D, CPU); held-out frames are never used, so
             renders at them are a fair test. Texels no photo saw are filled from their neighbours and counted (flagged)
  splats     shell texels and furniture surface samples become flat, opaque 1 cm discs (one .ply, same viewer, walk mode):
             every splat lies on a surface, so nothing can smear; a fake run folder lets make_tour.py build the tour
  compare    renders at v3_qa's off-path views (next to the splat model's renders from qa_sheet.jpg) and at its held-out frames
             (real | hybrid | splat, PSNR)
Usage: python hybrid_demo.py <gsplat examples dir> <data dir (images/ + sparse/0)> <run dir (full/, mesh/)> <out dir>"""
import json, os, sys, time, numpy as np, cv2, trimesh
import open3d as o3d
from scipy import ndimage
from plyfile import PlyData, PlyElement
T0 = time.time()

def say(*a): print(f'[{time.time() - T0:6.1f}s]', *a, flush=True)

def scene_of(V, F):
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32))); return sc

def cast(sc, o, d, chunk=1 << 22):
    """t_hit, triangle id, barycentric (u, v) for rays o + t d (d unit)."""
    t, pid, uv = [], [], []
    for a in range(0, len(o), chunk):
        r = sc.cast_rays(o3d.core.Tensor(np.concatenate([o[a:a + chunk], d[a:a + chunk]], 1).astype(np.float32)))
        t.append(r['t_hit'].numpy()); pid.append(r['primitive_ids'].numpy()); uv.append(r['primitive_uvs'].numpy())
    return np.concatenate(t), np.concatenate(pid).astype(np.int64), np.concatenate(uv)

def bilinear(img, u, v):
    """img [H, W, 3] float, u/v pixel coordinates (pixel centres at +0.5)."""
    H, W = img.shape[:2]; x = np.clip(u - 0.5, 0, W - 1.001); y = np.clip(v - 0.5, 0, H - 1.001)
    x0, y0 = x.astype(int), y.astype(int); fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    r = img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy) + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy
    return r / 255.0 if img.dtype == np.uint8 else r                                        # photos are kept as uint8 (RAM)

def colour(X, N, cams, imgs, K, sc, eps, k=3):
    """Real-photo colour for points X with normals N: the k best training views by (frontal, close), visibility by ray casting,
    weighted mean of the visible ones. Returns rgb [n, 3] in 0..1 and a seen mask."""
    n = len(X); bs = np.full((n, k), -np.inf, np.float32); bf = np.zeros((n, k), np.int32)
    H, W = imgs[0].shape[:2]; ar = np.arange(n)
    for f, c2w in enumerate(cams):
        w2c = np.linalg.inv(c2w); pc = X @ w2c[:3, :3].T + w2c[:3, 3]; z = pc[:, 2]
        u = pc[:, 0] / np.maximum(z, 1e-6) * K[0, 0] + K[0, 2]; v = pc[:, 1] / np.maximum(z, 1e-6) * K[1, 1] + K[1, 2]
        dv = c2w[:3, 3] - X; d = np.linalg.norm(dv, axis=1); cos = np.einsum('ij,ij->i', N, dv) / np.maximum(d, 1e-9)
        ok = (z > 1e-3) & (u > 2) & (u < W - 2) & (v > 2) & (v < H - 2) & (cos > 0.25)
        s = np.where(ok, cos ** 2 / np.maximum(d, 1e-6), -np.inf).astype(np.float32)
        j = bs.argmin(1); m = s > bs[ar, j]; bs[ar[m], j[m]] = s[m]; bf[ar[m], j[m]] = f
    acc = np.zeros((n, 3)); wsum = np.zeros(n)
    for slot in range(k):
        has = np.isfinite(bs[:, slot]); idx = np.nonzero(has)[0]
        if not len(idx): continue
        cc = np.stack([cams[f][:3, 3] for f in bf[idx, slot]]); dv = X[idx] - cc; d = np.linalg.norm(dv, axis=1)
        t, _, _ = cast(sc, cc, dv / d[:, None]); vis = np.isfinite(t) & (t > d - eps)
        for f in np.unique(bf[idx[vis], slot]):
            q = idx[vis][bf[idx[vis], slot] == f]; w2c = np.linalg.inv(cams[f]); pc = X[q] @ w2c[:3, :3].T + w2c[:3, 3]
            rgb = bilinear(imgs[f], pc[:, 0] / pc[:, 2] * K[0, 0] + K[0, 2], pc[:, 1] / pc[:, 2] * K[1, 1] + K[1, 2])
            w = bs[q, slot].astype(np.float64) ** 2; acc[q] += rgb * w[:, None]; wsum[q] += w
    seen = wsum > 0; out = np.zeros((n, 3)); out[seen] = acc[seen] / wsum[seen, None]
    return out, seen

def main(examples, data, run, out):
    os.makedirs(out, exist_ok=True); sys.path.insert(0, examples); from datasets.colmap import Parser
    p = Parser(data, factor=1, normalize=True, test_every=8); n_img = len(p.image_names)
    M = np.array(json.load(open(f'{run}/full/level.json'))['matrix']); R, t = M[:3, :3], M[:3, 3]
    C = np.einsum('ij,njk->nik', M, p.camtoworlds)                                         # cameras, levelled (floor y = 0)
    s = float(1.35 / np.median(C[:, 1, 3]))                                                 # metres per unit (as make_tour)
    cm = 0.01 / s; K = p.Ks_dict[p.camera_ids[0]]
    val = [i for i in range(n_img) if i % 8 == 0]; trn = [i for i in range(n_img) if i % 8]
    imgs = {i: cv2.cvtColor(cv2.imread(p.image_paths[i]), cv2.COLOR_BGR2RGB) for i in range(n_img)}       # uint8: 411 x 3.5 MB
    H, W = imgs[0].shape[:2]; say(n_img, 'frames', W, 'x', H, f'{s:.3f} m per unit')
    mesh = trimesh.load(f'{run}/mesh/mesh.ply', process=False); Vm = np.asarray(mesh.vertices); Fm = np.asarray(mesh.faces)
    fn = mesh.face_normals; say('mesh', len(Vm), 'vertices', len(Fm), 'faces')
    pts = p.points @ R.T + t

    # room axes from the wall normals (4-fold symmetric mean), room box from the SfM points (as make_tour)
    vert = np.abs(fn[:, 1]) < 0.2; th = np.arctan2(fn[vert, 2], fn[vert, 0]); area = mesh.area_faces[vert]
    th0 = np.angle(np.sum(area * np.exp(4j * th))) / 4; e1 = np.array([np.cos(th0), 0, np.sin(th0)]); e2 = np.array([-np.sin(th0), 0, np.cos(th0)])
    uvp = lambda X: np.stack([X @ e1, X @ e2], 1)
    lo, hi = np.percentile(uvp(pts), 3, 0), np.percentile(uvp(pts), 97, 0); pad = 0.12 * (hi - lo); lo, hi = lo - pad, hi + pad
    down = fn[:, 1] < -0.8; ceil_y = Vm[Fm[down]].reshape(-1, 3)[:, 1]
    hc = float(np.median(ceil_y[ceil_y > np.median(C[:, 1, 3])])) if (ceil_y > np.median(C[:, 1, 3])).sum() > 3000 else float(np.percentile(pts[:, 1], 99))
    say(f'ceiling {hc * s:.2f} m, room axes {np.degrees(th0):.1f} deg')

    # footprint: floor + furniture + camera path on a 2 cm grid, closed, holes filled, biggest piece; outline -> walls
    cell = 2 * cm; nu, nv = np.ceil((hi - lo) / cell).astype(int) + 1
    def cells(X):
        q = np.floor((uvp(X) - lo) / cell).astype(int); ok = (q[:, 0] >= 0) & (q[:, 0] < nu) & (q[:, 1] >= 0) & (q[:, 1] < nv); return q, ok
    occ = np.zeros((nv, nu), bool)
    ev = Vm[(Vm[:, 1] < 1.2 / s) & (Vm[:, 1] > -0.05 / s)]
    q, ok = cells(ev); occ[q[ok, 1], q[ok, 0]] = True
    q, ok = cells(C[:, :3, 3]); cam = np.zeros_like(occ); cam[q[ok, 1], q[ok, 0]] = True
    occ |= ndimage.binary_dilation(cam, iterations=int(0.3 / 0.02))
    occ = ndimage.binary_closing(occ, iterations=int(0.15 / 0.02)); occ = ndimage.binary_fill_holes(occ)
    lab, nl = ndimage.label(occ); occ = lab == (1 + np.argmax(ndimage.sum(occ, lab, range(1, nl + 1))))
    cs, _ = cv2.findContours(occ.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    poly = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 0.05 / 0.02, True)[:, 0].astype(float)  # 5 cm tolerance
    corners = [(lo[0] + (x + 0.5) * cell) * e1 + (lo[1] + (y + 0.5) * cell) * e2 for x, y in poly]
    say(f'footprint {occ.sum() * (0.02 ** 2):.1f} m2, outline {len(corners)} corners')

    # shell surfaces: o + i a tex + j b tex, normal n (into the room)
    tex = cm; surf = []
    umin, vmin = lo; uw, vw = (hi - lo)
    fm = cv2.resize(occ.astype(np.uint8), (int(np.ceil(uw / tex)), int(np.ceil(vw / tex))), interpolation=cv2.INTER_NEAREST).astype(bool)
    o_floor = umin * e1 + vmin * e2
    surf.append(dict(name='floor', o=o_floor, a=e1, b=e2, n=np.array([0, 1., 0]), nw=fm.shape[1], nh=fm.shape[0], mask=fm))
    surf.append(dict(name='ceiling', o=o_floor + np.array([0, hc, 0]), a=e1, b=e2, n=np.array([0, -1., 0]), nw=fm.shape[1], nh=fm.shape[0], mask=fm))
    for k in range(len(corners)):
        P, Q = corners[k], corners[(k + 1) % len(corners)]; L = np.linalg.norm(Q - P)
        if L < 0.1 / s: continue
        a = (Q - P) / L; nrm = np.array([-a[2], 0, a[0]]); mid = (P + Q) / 2 + 0.1 / s * nrm; qm, okm = cells(mid[None])
        if not (okm[0] and occ[qm[0, 1], qm[0, 0]]): nrm = -nrm
        surf.append(dict(name=f'wall{k}', o=P, a=a, b=np.array([0, 1., 0]), n=nrm, nw=int(np.ceil(L / tex)), nh=int(np.ceil(hc / tex)), mask=None))
    # furniture: mesh faces away from the shell (> 4 cm from floor/ceiling/walls) and inside the footprint
    dist_in = ndimage.distance_transform_edt(occ) * 0.02
    cen = mesh.triangles_center; qc, okc = cells(cen); din = np.zeros(len(cen)); din[okc] = dist_in[qc[okc, 1], qc[okc, 0]]
    keep = okc & (din > 0.04) & (cen[:, 1] > 0.04 / s) & (cen[:, 1] < hc - 0.04 / s)
    fur = mesh.submesh([np.nonzero(keep)[0]], append=True); fur.remove_unreferenced_vertices()
    say('furniture', len(fur.faces), 'of', len(Fm), 'faces kept')

    # one ray-casting scene: shell quads (2 triangles each) + furniture
    SV, SF, tri_surf = [], [], []
    for si, S in enumerate(surf):
        A, B = S['a'] * S['nw'] * tex, S['b'] * S['nh'] * tex
        v0 = len(SV); SV += [S['o'], S['o'] + A, S['o'] + A + B, S['o'] + B]; SF += [[v0, v0 + 1, v0 + 2], [v0, v0 + 2, v0 + 3]]; tri_surf += [si, si]
    SV, SF = np.array(SV), np.array(SF); Vf, Ff = np.asarray(fur.vertices), np.asarray(fur.faces)
    V_all = np.concatenate([SV, Vf]); F_all = np.concatenate([SF, Ff + len(SV)]); tri_surf = np.array(tri_surf + [-1] * len(Ff))
    sc = scene_of(V_all, F_all); cams = [C[i] for i in trn]; ims = [imgs[i] for i in trn]; eps = 0.03 / s

    # textures from the training photos only
    stats = {}
    for S in surf:
        ii, jj = np.meshgrid(np.arange(S['nw']), np.arange(S['nh']))
        X = S['o'] + ((ii.ravel() + 0.5) * tex)[:, None] * S['a'] + ((jj.ravel() + 0.5) * tex)[:, None] * S['b'] + 1e-4 * S['n']
        msk = np.ones(len(X), bool) if S['mask'] is None else S['mask'].ravel()
        rgb = np.zeros((len(X), 3)); seen = np.zeros(len(X), bool)
        rgb[msk], seen[msk] = colour(X[msk], np.tile(S['n'], (msk.sum(), 1)), cams, ims, K, sc, eps)
        img = (rgb.reshape(S['nh'], S['nw'], 3) * 255).clip(0, 255).astype(np.uint8); hole = (~seen & msk).reshape(S['nh'], S['nw'])
        if hole.any(): img = cv2.inpaint(img, hole.astype(np.uint8), 5, cv2.INPAINT_TELEA)
        S['tex'] = img.astype(np.float32) / 255; S['seen'] = seen.reshape(S['nh'], S['nw']); S['inside'] = msk.reshape(S['nh'], S['nw'])
        stats[S['name']] = dict(m2=round(float(msk.sum()) * 1e-4, 1), photo_pct=round(100 * float(seen.sum()) / max(1, msk.sum()), 1))
        say(S['name'], stats[S['name']])
    fur_rgb, fur_seen = colour(Vf + 1e-4 * fur.vertex_normals, fur.vertex_normals, cams, ims, K, sc, eps)
    try: old = np.asarray(fur.visual.vertex_colors)[:, :3] / 255.0
    except Exception: old = np.full((len(Vf), 3), 0.5)
    fur_rgb[~fur_seen] = old[~fur_seen]                                                     # unseen: TSDF colour
    stats['furniture'] = dict(vertices=len(Vf), photo_pct=round(100 * float(fur_seen.mean()), 1))
    say('furniture colours', stats['furniture'])

    def render(c2w, w=W, h=H):
        uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5); Kk = K * np.array([[w / W], [h / H], [1]])
        d = np.stack([(uu - Kk[0, 2]) / Kk[0, 0], (vv - Kk[1, 2]) / Kk[1, 1], np.ones_like(uu)], -1).reshape(-1, 3) @ c2w[:3, :3].T
        d /= np.linalg.norm(d, axis=1, keepdims=True); o = np.tile(c2w[:3, 3], (len(d), 1))
        tt, pid, buv = cast(sc, o, d); img = np.full((len(d), 3), 0.08); hit = np.isfinite(tt)
        si = np.full(len(d), -2); si[hit] = tri_surf[pid[hit]]
        f = hit & (si == -1)                                                                # furniture: vertex colours
        tri = Ff[pid[f] - len(SF)]; b1, b2 = buv[f, 0:1], buv[f, 1:2]
        img[f] = (1 - b1 - b2) * fur_rgb[tri[:, 0]] + b1 * fur_rgb[tri[:, 1]] + b2 * fur_rgb[tri[:, 2]]
        for k, S in enumerate(surf):                                                        # shell: texture lookup
            m = si == k
            if not m.any(): continue
            X = o[m] + tt[m, None] * d[m] - S['o']; c = bilinear(S['tex'], X @ S['a'] / tex, X @ S['b'] / tex)
            if S['mask'] is not None:
                ci = np.clip((X @ S['a'] / tex).astype(int), 0, S['nw'] - 1); cj = np.clip((X @ S['b'] / tex).astype(int), 0, S['nh'] - 1)
                c[~S['inside'][cj, ci]] = 0.08
            img[m] = c
        return (img.reshape(h, w, 3) * 255).clip(0, 255).astype(np.uint8)

    # comparison with the splat model at v3_qa's views (its qa_sheet.jpg: row 1 held-out real | splat, row 2 off-path)
    qa = cv2.cvtColor(cv2.imread(f'{run}/full/qa_sheet.jpg'), cv2.COLOR_BGR2RGB); tw, th_ = qa.shape[1] // 6, qa.shape[0] // 2
    fit = lambda a: cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
    Ct = p.camtoworlds; side = 0.15 * np.linalg.norm(Ct[:, :3, 3] - Ct[:, :3, 3].mean(0), axis=1).max()
    rows, res = [], dict(stats=stats, metres_per_unit=round(s, 4), ceiling_m=round(hc * s, 2), corners=len(corners), heldout=[])
    for n_, i in enumerate([trn[len(trn) // 3], trn[2 * len(trn) // 3]]):
        c = Ct[i].copy(); c[:3, 3] += side * c[:3, 0]; hy = render(M @ c); cv2.imwrite(f'{out}/offpath{n_}_hybrid.jpg', hy[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append(np.concatenate([fit(qa[th_:2 * th_, n_ * tw:(n_ + 1) * tw]), hy, np.zeros_like(hy)], 1))
    def psnr(a, g):
        a, g = a.astype(np.float64) / 255, g.astype(np.float64) / 255; A = np.c_[a.reshape(-1, 3), np.ones(a.size // 3)]
        f_ = np.clip(A @ np.linalg.lstsq(A, g.reshape(-1, 3), rcond=None)[0], 0, 1).reshape(g.shape)
        return round(float(-10 * np.log10(np.mean((a - g) ** 2))), 2), round(float(-10 * np.log10(np.mean((f_ - g) ** 2))), 2)
    for n_, k in enumerate(np.linspace(0, len(val) - 1, min(3, len(val))).astype(int)):
        i = val[k]; hy = render(C[i]); gt = imgs[i]; sp = fit(qa[:th_, (2 * n_ + 1) * tw:(2 * n_ + 2) * tw])
        cv2.imwrite(f'{out}/heldout{n_}_hybrid.jpg', hy[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        r = dict(frame=p.image_names[i], hybrid=psnr(hy, gt), splat_from_qa_jpeg=psnr(sp, gt)); res['heldout'].append(r); say('held-out', r)
        rows.append(np.concatenate([sp, hy, gt], 1))
    cv2.imwrite(f'{out}/hybrid_compare.jpg', np.concatenate(rows, 0)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    # four views from the room centre at walk height (1.3 m), one per wall direction
    ctr = (np.mean([c_ for c_ in corners], 0)); eye = ctr + np.array([0, 1.3 / s, 0]); views = []
    for a in (e1, e2, -e1, -e2):
        z = a; x = np.cross(np.array([0, -1., 0]), z); x /= np.linalg.norm(x); y = np.cross(z, x)
        c2w = np.eye(4); c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye; views.append(render(c2w, W // 2, H // 2))
    cv2.imwrite(f'{out}/hybrid_centre_views.jpg', np.concatenate([np.concatenate(views[:2], 1), np.concatenate(views[2:], 1)], 0)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])

    # splats: flat opaque discs on every surface, written in the trainer's coordinates (like model.ply) for make_tour.py
    from scipy.spatial.transform import Rotation
    Ps, Cs, Rs = [], [], []
    for S in surf:
        jj, ii = np.nonzero(S['inside'])
        Ps.append(S['o'] + ((ii + 0.5) * tex)[:, None] * S['a'] + ((jj + 0.5) * tex)[:, None] * S['b']); Cs.append(S['tex'][jj, ii])
        Rm = np.stack([S['a'], S['b'], S['n']], 1)
        if np.linalg.det(Rm) < 0: Rm[:, 0] *= -1
        Rs.append(np.repeat(Rm[None], len(jj), 0))
    nf = int(min(1.5e6, fur.area / tex ** 2)); sp_, fi = trimesh.sample.sample_surface(fur, nf, seed=0)
    bc = trimesh.triangles.points_to_barycentric(fur.triangles[fi], sp_); tri = Ff[fi]
    Ps.append(sp_); Cs.append(np.einsum('nk,nkc->nc', bc, fur_rgb[tri]))
    nz = fur.face_normals[fi]; ax = np.cross(nz, np.where(np.abs(nz[:, :1]) < 0.9, [[1, 0, 0]], [[0, 1, 0]])); ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    Rs.append(np.stack([ax, np.cross(nz, ax), nz], 2))
    P_, C_, R_ = np.concatenate(Ps), np.concatenate(Cs).clip(0, 1), np.concatenate(Rs)
    Pt = (P_ - t) @ R; Rt = np.einsum('ji,njk->nik', R, R_)                                # back to trainer space
    q = Rotation.from_matrix(Rt).as_quat()[:, [3, 0, 1, 2]]
    sz = np.log(np.array([0.6 * tex, 0.6 * tex, 0.03 * tex]))
    v = np.zeros(len(Pt), [(k, 'f4') for k in ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']])
    v['x'], v['y'], v['z'] = Pt.T
    for c in range(3): v[f'f_dc_{c}'] = (C_[:, c] - 0.5) / 0.28209479
    v['opacity'] = 5.0; v['scale_0'], v['scale_1'], v['scale_2'] = sz
    v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3'] = q.T
    hr = f'{out}/run'; os.makedirs(f'{hr}/full', exist_ok=True); os.makedirs(f'{hr}/mesh', exist_ok=True)
    PlyData([PlyElement.describe(v, 'vertex')]).write(f'{hr}/full/model.ply')
    for x in ('level.json', 'nav.json'):
        if os.path.exists(f'{run}/full/{x}'): open(f'{hr}/full/{x}', 'w').write(open(f'{run}/full/{x}').read())
    trimesh.Trimesh(V_all, F_all, process=False).export(f'{hr}/mesh/mesh.glb')               # collision: shell + furniture
    res.update(splats=len(Pt), minutes=round((time.time() - T0) / 60, 1)); json.dump(res, open(f'{out}/hybrid.json', 'w'), indent=1)
    say('HYBRID DONE', json.dumps(res))

if __name__ == '__main__':
    main(*sys.argv[1:5])
