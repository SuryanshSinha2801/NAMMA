"""NAMMA World v1 (Plan B, owner 27 Oct): one photo -> a 360-degree 3D world, built by "drones" (the Matrix-3D / Mickmumpitz
video idea, done as Text2Room / LucidDreamer do it): start from the photo as a textured 3D surface (genworld.py v0), then fly
virtual cameras around the start point; in every drone view, what the world cannot show yet is painted by a generative model
(SDXL inpainting on the GPU), its depth is estimated and fitted to the depth the world already has around the gap (scale fit +
a smooth correction so the seam meets exactly), and the painted pixels become a new textured patch of the world. Real photo
pixels are never replaced. The order goes from small turns to the full circle, then ceiling and floor, then small side steps
that open up what was hidden behind objects.
  world     list of textured meshes in the photo camera's frame (OpenCV: x right, y down, z forward, metres); one Open3D
            ray-casting scene (patches are added as they are made)
  drone     render -> holes -> inpaint(rgb, hole) -> depth(rgb) -> fit to the known depth -> patch mesh (grid, cut at depth jumps)
The models are passed in as functions, so the whole loop is tested on the CPU with a synthetic room (oracle painter) before it
runs on Kaggle with the real ones (final/world_job.py)."""
import io, json, os, time, numpy as np, cv2
import genworld as G
T0 = time.time()


def say(*a): print(f'[{time.time() - T0:6.1f}s]', *a, flush=True)


def pose(yaw_deg=0.0, pitch_deg=0.0, pos=(0, 0, 0)):
    """Camera-to-world in the photo camera's frame: yaw turns right (+), pitch looks up (+), pos in metres."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    c = np.eye(4); c[:3, :3] = Ry @ Rx; c[:3, 3] = pos; return c


def drone_plan():
    """Turns at the start point (90-degree views), ceiling and floor, then side steps for what is behind objects."""
    v = [pose(y) for y in (45, -45, 90, -90, 135, -135, 180)]
    v += [pose(y, 65) for y in (0, 90, 180, 270)] + [pose(y, -65) for y in (0, 90, 180, 270)]
    v += [pose(0, 0, (0.45, 0, 0)), pose(0, 0, (-0.45, 0, 0)), pose(0, 0, (0, 0, -0.6)), pose(25, 0, (0.35, 0, 0.25)), pose(-25, 0, (-0.35, 0, 0.25))]
    # second, finer pass: in-between directions and small steps close the last cracks between patches
    v += [pose(y, p) for y in range(0, 360, 60) for p in (35, -35)] + [pose(y, 0, (0.25 * np.sin(np.radians(y)), 0, 0.25 * np.cos(np.radians(y)))) for y in range(0, 360, 45)]
    return v


class World:
    def __init__(self):
        import open3d as o3d
        self.o3d = o3d; self.sc = o3d.t.geometry.RaycastingScene(); self.parts = []; self.gid = {}

    def add(self, V, F, UV, tex, name):
        if not len(F): return
        g = self.sc.add_triangles(self.o3d.core.Tensor(V.astype(np.float32)), self.o3d.core.Tensor(F.astype(np.uint32)))
        self.gid[int(g)] = len(self.parts); self.parts.append(dict(V=V, F=F, UV=UV, tex=tex, name=name))

    def render(self, c2w, K, W, H):
        """rgb uint8, z-depth (inf = nothing there), hit mask."""
        uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        dc = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1).reshape(-1, 3)
        d = dc @ c2w[:3, :3].T; nrm = np.linalg.norm(d, axis=1, keepdims=True); d /= nrm; o = np.tile(c2w[:3, 3], (len(d), 1))
        r = self.sc.cast_rays(self.o3d.core.Tensor(np.concatenate([o, d], 1).astype(np.float32)))
        t = r['t_hit'].numpy(); hit = np.isfinite(t); gids = r['geometry_ids'].numpy(); pids = r['primitive_ids'].numpy().astype(np.int64); b = r['primitive_uvs'].numpy()
        img = np.zeros((len(d), 3), np.float32); z = np.where(hit, t / nrm[:, 0], np.inf)
        for g, k in self.gid.items():
            m = hit & (gids == g)
            if not m.any(): continue
            P = self.parts[k]; tri = P['F'][pids[m]]; bb = b[m]
            uv = (1 - bb[:, :1] - bb[:, 1:]) * P['UV'][tri[:, 0]] + bb[:, :1] * P['UV'][tri[:, 1]] + bb[:, 1:] * P['UV'][tri[:, 2]]
            tex = P['tex']; h, w = tex.shape[:2]; x = np.clip(uv[:, 0] * w - 0.5, 0, w - 1.001); y = np.clip((1 - uv[:, 1]) * h - 0.5, 0, h - 1.001)
            x0, y0 = x.astype(int), y.astype(int); fx, fy = (x - x0)[:, None], (y - y0)[:, None]; tf = tex.astype(np.float32)
            img[m] = tf[y0, x0] * (1 - fx) * (1 - fy) + tf[y0, x0 + 1] * fx * (1 - fy) + tf[y0 + 1, x0] * (1 - fx) * fy + tf[y0 + 1, x0 + 1] * fx * fy
        return img.reshape(H, W, 3).clip(0, 255).astype(np.uint8), z.reshape(H, W), hit.reshape(H, W)


def patch(world, z, K, c2w, rgb, keep, step, name, jump=0.04):
    """Pixels `keep` of a view with depth z become a textured patch (grid every `step` px, cut at depth jumps), in world space."""
    X, F, UV = G.grid_mesh(z.astype(np.float32), K, keep, step, jump)
    if not len(F): return 0
    used = np.unique(F); remap = -np.ones(len(X), int); remap[used] = np.arange(len(used)); X, UV, F = X[used], UV[used], remap[F]
    H, W = keep.shape; px, py = UV[:, 0] * W, (1 - UV[:, 1]) * H                              # texture cropped to the patch
    x0, y0 = max(int(px.min()) - 2, 0), max(int(py.min()) - 2, 0); x1, y1 = min(int(np.ceil(px.max())) + 2, W), min(int(np.ceil(py.max())) + 2, H)
    UV = np.stack([(px - x0) / (x1 - x0), 1 - (py - y0) / (y1 - y0)], 1)
    world.add(X @ c2w[:3, :3].T + c2w[:3, 3], F, UV, np.ascontiguousarray(rgb[y0:y1, x0:x1]), name)
    return len(F)


def fit_depth(pred, z_known, known, hole, ring_px=40):
    """Metric depth for the painted pixels: pred (the depth model's depth of the painted view) fitted to the world's depth on
    the known ring around the gap: one scale (median ratio), then a smooth log-correction from the ring pixels that agree
    (outliers, e.g. mesh edges at the seam, are dropped: copying them into the gap bent whole patches, 2.7 % median depth error
    with a perfect depth model), by normalized convolution at two widths; deep inside a big gap it fades to the plain scale.
    Returns depth, the scale, the residual (median |log ratio - log scale| on the ring) and the ring size."""
    ring = (cv2.dilate(hole.astype(np.uint8), np.ones((2 * ring_px + 1, 2 * ring_px + 1), np.uint8)) > 0) & known & ~hole
    use = ring if ring.sum() >= 200 else known
    if use.sum() < 50: return pred, 1.0, None, int(use.sum())
    lr = np.log(z_known[use]) - np.log(pred[use]); s = float(np.exp(np.median(lr))); r = lr - np.log(s); mad = float(np.median(np.abs(r)))
    good = np.abs(r) < max(0.02, 4 * 1.4826 * mad)
    v = np.zeros(pred.shape, np.float32); c = np.zeros(pred.shape, np.float32); ii = np.flatnonzero(use)[good]
    v.flat[ii] = 1; c.flat[ii] = r[good]
    corr = np.zeros(pred.shape, np.float32)
    H, W = pred.shape
    for sig in (4 * ring_px, ring_px / 2):                                                    # wide first, the near one on top
        k = 4 if sig >= 32 else 1                                                             # the wide blur on a 4x smaller grid
        sm = (lambda a: cv2.resize(a, (W // k, H // k), interpolation=cv2.INTER_AREA)) if k > 1 else (lambda a: a)
        up = (lambda a: cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)) if k > 1 else (lambda a: a)
        den = up(cv2.GaussianBlur(sm(v), (0, 0), sig / k)); num = up(cv2.GaussianBlur(sm(c), (0, 0), sig / k)); w = den / (den + 0.02)
        corr = w * num / np.maximum(den, 1e-6) + (1 - w) * corr
    return (s * pred * np.exp(corr)).astype(np.float32), s, round(mad, 4), int(use.sum())


def normals_of(D, K, step=4, span=3):
    """Points and unit normals (facing the camera) of a z-depth map on a step-px grid (OpenCV camera frame); the point map is
    smoothed and differences span +-span grid cells, points next to a depth jump are left out."""
    H, W = D.shape; ys, xs = np.arange(step // 2, H, step), np.arange(step // 2, W, step); gx, gy = np.meshgrid(xs, ys)
    z = cv2.medianBlur(D.astype(np.float32), 5)[gy, gx]
    P = np.stack([(gx + 0.5 - K[0, 2]) / K[0, 0] * z, (gy + 0.5 - K[1, 2]) / K[1, 1] * z, z], -1).astype(np.float32)
    Ps = cv2.GaussianBlur(P, (0, 0), 1.5); k = span; dx = np.zeros_like(P); dy = np.zeros_like(P)
    dx[:, k:-k] = Ps[:, 2 * k:] - Ps[:, :-2 * k]; dy[k:-k] = Ps[2 * k:] - Ps[:-2 * k]
    n = np.cross(dx, dy); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12; n *= -np.sign((n * P).sum(-1, keepdims=True) + 1e-12)
    lz = np.log(np.maximum(z, 1e-3)); jmp = cv2.dilate((np.abs(cv2.Laplacian(lz, cv2.CV_32F)) > 0.03).astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
    ok = ~jmp; ok[:k] = ok[-k:] = False; ok[:, :k] = ok[:, -k:] = False
    return P[ok].astype(np.float64), n[ok].astype(np.float64)


def _level(h, lowest=True, bin_m=0.05, frac=0.25):
    """Height of the lowest (or highest) dense level among heights h: the first 5 cm bin from that end holding >= frac of the
    fullest bin's count, refined by the median of the points within 5 cm of it."""
    lo_, hi_ = np.percentile(h, 0.5), np.percentile(h, 99.5); bins = np.arange(lo_ - bin_m, hi_ + 2 * bin_m, bin_m)
    cnt, edges = np.histogram(h, bins); idx = np.flatnonzero(cnt >= frac * cnt.max()); i = idx[0] if lowest else idx[-1]
    c = 0.5 * (edges[i] + edges[i + 1]); near = np.abs(h - c) < bin_m
    return float(np.median(h[near])) if near.any() else float(c)


def room_box(D, K, margin_m=0.8, behind_m=1.5, height_m=2.6):
    """The room as a box (geometric grounding for the drones: walls, floor and ceiling are planes). From the photo's metric depth:
    up = mean of floor/ceiling normals; floor = the lowest level of upward-facing points (not a table top), ceiling = the highest
    level of downward-facing ones (floor + height_m when not seen); walls along the 2 Manhattan directions of the wall normals,
    each the farthest surface that way (a wall bounds the room: nearer faces such as a table's sides are not walls). A wall the
    photo does not see goes margin_m beyond the farthest point that way (behind_m behind the camera), and never closer than
    0.8 m to the camera, so the drones stay inside. Returns dict(R: columns = box axes in the camera frame (e1, up, e2), lo, hi:
    bounds along them, seen: which faces were measured)."""
    P, n = normals_of(D, K); up0 = np.array([0., -1., 0.]); m = max(50, int(0.01 * len(P)))  # OpenCV y is down
    fl, ce = n @ up0 > np.cos(np.radians(30)), n @ -up0 > np.cos(np.radians(30))
    up = n[fl].sum(0) - n[ce].sum(0) if (fl.sum() + ce.sum()) > m else up0.copy(); up /= np.linalg.norm(up)
    for it in range(3):                                                                       # floor plane from point positions
        h = P @ up; nu = n @ up; fl, ce = (nu > 0.8) & (h < -0.2), (nu < -0.8) & (h > 0.2)        # (normals only pick candidates)
        floor = _level(h[fl], True) if fl.sum() > m else float(np.percentile(h, 1))
        floor = min(floor, float(np.percentile(h, 1)) + 0.05)                                  # nothing far below the floor
        Fp = P[fl & (np.abs(h - floor) < 0.12)]
        if len(Fp) < m: break
        u = np.linalg.svd(Fp - Fp.mean(0), full_matrices=False)[2][-1]; u *= np.sign(u @ up)
        if np.degrees(np.arccos(np.clip(u @ up, -1, 1))) > 20: break                         # not a floor: keep the normals' up
        up = u
    ceil_ = _level(h[ce], False) if ce.sum() > m else None
    if ceil_ is None or not 1.8 < ceil_ - floor < 4.5: ceil_ = max(floor + height_m, float(np.percentile(h, 99.5)) + 0.05); seen_c = False
    else: seen_c = True
    a = np.cross(up, [0., 0., 1.]); a /= np.linalg.norm(a); b = np.cross(up, a)                 # horizontal basis
    wl = np.abs(nu) < 0.3; th = np.arctan2(n[wl] @ b, n[wl] @ a)
    t0 = float(np.angle(np.exp(4j * th).mean()) / 4) if wl.sum() > m else 0.0
    e1 = np.cos(t0) * a + np.sin(t0) * b; e2 = np.cross(e1, up); R = np.stack([e1, up, e2], 1)
    lo, hi, seen = np.zeros(3), np.zeros(3), dict(floor=bool(fl.sum() > m), ceiling=seen_c)
    lo[1], hi[1] = floor, ceil_
    for i, e in ((0, e1), (2, e2)):
        c = P @ e; nc = n @ e
        for sgn in (1, -1):
            sc = sgn * c; ext = float(np.percentile(sc, 99.5))                                 # the farthest points that way
            near = wl & (sgn * nc < -0.8) & (sc > ext - 0.15)                                 # facing back, at that extent: a wall
            if near.sum() > m and ext > 0.3: d = float(np.median(sc[near])); seen[f'{"+" if sgn > 0 else "-"}{i}'] = True
            else: d = ext + margin_m if ext > 0.3 else behind_m; seen[f'{"+" if sgn > 0 else "-"}{i}'] = False
            d = max(d, 0.8)
            if sgn > 0: hi[i] = d
            else: lo[i] = -d
    return dict(R=R, lo=lo, hi=hi, seen=seen)


def box_depth(box, c2w, K, W, H):
    """z-depth of the room box seen from a camera inside it."""
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    dc = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1).reshape(-1, 3)
    d = dc @ c2w[:3, :3].T @ box['R']; o = c2w[:3, 3] @ box['R']                              # in box axes; d has camera z = 1
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(d > 0, (box['hi'] - o) / d, np.where(d < 0, (box['lo'] - o) / d, np.inf))
    return np.maximum(t.min(1), 1e-3).reshape(H, W).astype(np.float32)                          # t along dc = z-depth


def match_colour(painted, rgb, both, region):
    """Exposure / white balance of the painted pixels matched to the real ones: per-channel gain + offset fitted on pixels that
    have both (the overlap band), applied to region."""
    if both.sum() < 200: return painted
    out = painted.astype(np.float32)
    for ch in range(3):
        x, y = out[..., ch][both], rgb[..., ch][both].astype(np.float32); g = np.clip(y.std() / (x.std() + 1e-6), 0.7, 1.4)
        out[..., ch][region] = (out[..., ch][region] - x.mean()) * g + y.mean()
    return out.clip(0, 255).astype(np.uint8)


def fly(world, inpaint, depth_fn, size=1024, fov=90.0, plan=None, step=4, min_hole=0.003, log=None, box=None):
    """The drone loop. inpaint(rgb uint8, hole bool, **view) -> rgb uint8; depth_fn(rgb uint8, **view) -> z-depth [H, W] (any
    scale). view = c2w, K, depth (the world's depth, the room box where it has nothing: the painter's ControlNet guide; the
    synthetic test's oracle painter uses c2w and K). With a room box the new pixels go onto the box (straight walls, floor and
    ceiling; depth_fn is not needed), else onto depth_fn's depth fitted to the world."""
    f = 0.5 * size / np.tan(np.radians(fov) / 2); K = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.]]); stats = []
    for k, c2w in enumerate(plan or drone_plan()):
        t0 = time.time(); rgb, z, hit = world.render(c2w, K, size, size); hole = ~hit
        if hole.mean() < min_hole: stats.append(dict(view=k, hole_pct=round(100 * float(hole.mean()), 2), skipped=True)); continue
        o_ = 2 * (step + 2) + 1                                                                   # overlap the old surface by a grid cell + 2 px:
        hole_d = cv2.dilate(hole.astype(np.uint8), np.ones((o_, o_), np.uint8)) > 0                  # no crack at the seam
        bz = box_depth(box, c2w, K, size, size) if box is not None else None
        painted = inpaint(rgb, hole_d, c2w=c2w, K=K, depth=np.where(hit, z, bz) if bz is not None else None)
        painted = match_colour(painted, rgb, hole_d & hit, hole_d)
        painted = np.where(hole_d[..., None], painted, rgb).astype(np.uint8)
        if bz is not None: zz, s, res, nring = bz, 1.0, None, 0
        else:
            pred = np.clip(depth_fn(painted, c2w=c2w, K=K), 1e-3, None)
            zz, s, res, nring = fit_depth(pred, np.where(hit, z, 1.0), hit, hole)
        zz = np.where(hit, z * 1.002, zz).astype(np.float32)                                  # overlap = the old surface, 0.2 % behind it
        n = patch(world, zz, K, c2w, painted, hole_d, step, f'Drone_{k + 1}')
        st = dict(view=k, hole_pct=round(100 * float(hole.mean()), 1), scale=round(s, 3), fit_residual=res, ring_px=nring, triangles=n, seconds=round(time.time() - t0, 1))
        stats.append(st); say('drone', st)
        if log:                                                                               # render | painted (| the depth guide)
            row = [rgb, painted]
            if bz is not None:
                dv = 1 / np.where(hit, z, bz); dv = (255 * (dv - dv.min()) / (np.ptp(dv) + 1e-9)).astype(np.uint8); row.append(np.repeat(dv[..., None], 3, -1))
            cv2.imwrite(f'{log}/drone_{k + 1:02d}.jpg', cv2.cvtColor(np.concatenate(row, 1), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return stats


def start(world, img, K, depth, inpaint, step=2, box=None):
    """The photo as the first textured surface (cut at depth jumps) + what is hidden right behind its depth jumps (painted; its
    depth = the room box there when given, else the far side of the jump)."""
    D = depth.astype(np.float32); lg = np.log(D); jump = np.zeros(D.shape, bool)
    jump[:, 1:] |= np.abs(np.diff(lg, axis=1)) > G.JUMP; jump[1:, :] |= np.abs(np.diff(lg, axis=0)) > G.JUMP
    far = cv2.dilate(D, np.ones((31, 31), np.uint8)); band = cv2.dilate(jump.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    occl = band & (D < 0.95 * far)
    X, F, UV = G.grid_mesh(D, K, np.ones(D.shape, bool), step, G.JUMP); world.add(X, F, UV, img, 'Photo')
    if occl.any():
        H, W = D.shape; bz = box_depth(box, np.eye(4), K, W, H) if box is not None else None
        if bz is not None: far = np.maximum(np.minimum(bz, 1.6 * far), D)                         # behind = the wall / floor, not beyond it
        behind = inpaint(img, occl, c2w=np.eye(4), K=K, depth=np.where(occl, far, D)); bd = cv2.medianBlur(np.where(occl, far, D).astype(np.float32), 5)
        keep = cv2.dilate(occl.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        X2, F2, UV2 = G.grid_mesh(bd, K, keep, step, G.JUMP); world.add(X2, F2, UV2, behind, 'Behind')
    return dict(photo_triangles=len(F), hidden_pct=round(100 * float(occl.mean()), 1))


def export(world, path):
    import trimesh
    from PIL import Image
    sc = trimesh.Scene()
    for P in world.parts:
        jpg = cv2.imencode('.jpg', cv2.cvtColor(P['tex'], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes(); im = Image.open(io.BytesIO(jpg)); im.load()
        mat = trimesh.visual.material.PBRMaterial(name=P['name'], baseColorTexture=im, metallicFactor=0.0, roughnessFactor=1.0)
        used = np.unique(P['F']); remap = -np.ones(len(P['V']), int); remap[used] = np.arange(len(used))
        m = trimesh.Trimesh(P['V'][used], remap[P['F']], visual=trimesh.visual.TextureVisuals(uv=P['UV'][used], material=mat), process=False)
        sc.add_geometry(m, node_name=P['name'], geom_name=P['name'])
    sc.export(path); return os.path.getsize(path)


def load_glb(path):
    """world.glb -> World (for scoring on another machine)."""
    import trimesh
    w = World(); g = trimesh.load(path)
    for name, m in g.geometry.items():
        w.add(np.asarray(m.vertices, np.float64), np.asarray(m.faces), np.asarray(m.visual.uv, np.float64), np.asarray(m.visual.material.baseColorTexture.convert('RGB')), name)
    return w


def sheet360(world, size=384, step=45, pitch=0):
    """One row of views all around the start point (90-degree lens): for checking the whole world by eye."""
    f = 0.5 * size / np.tan(np.radians(45)); K = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.]])
    ims, cov = [], []
    for y in range(0, 360, step):
        rgb, _, hit = world.render(pose(y, pitch), K, size, size); ims.append(rgb); cov.append(float(hit.mean()))
    return np.concatenate(ims, 1), cov


VIEWER = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NAMMA World</title><style>body{margin:0;background:#15161a;overflow:hidden;font:13px system-ui,sans-serif;color:#eee;touch-action:none}
#ui{position:fixed;left:12px;top:12px;display:flex;flex-wrap:wrap;gap:6px;z-index:2}#ui button{background:#2b2d31ee;color:#eee;border:1px solid #fff3;border-radius:16px;padding:7px 12px;cursor:pointer}
#ui button.on{background:#e8a96b;color:#111}#help{position:fixed;left:12px;bottom:12px;background:#0009;padding:6px 10px;border-radius:8px;max-width:calc(100vw - 48px)}
#load{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;font-size:16px}</style></head><body>
<div id="ui"><button id="all" class="on">World</button><button id="photo">Photo only</button><button id="gen">Show generated</button><button id="reset">Reset</button></div>
<div id="load">Loading the world...</div>
<div id="help">Drag to look around (360&deg;). W A S D or arrow keys to walk, scroll to step forward/back. Orange in "Show generated" = made by the AI.</div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three'; import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
const info = await (await fetch('./world.json')).json();
const r = new THREE.WebGLRenderer({antialias:true}); r.setPixelRatio(Math.min(devicePixelRatio, 2)); r.setSize(innerWidth, innerHeight); r.outputColorSpace = THREE.SRGBColorSpace; document.body.appendChild(r.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x15161a);
const cam = new THREE.PerspectiveCamera(Math.max(info.fov_y, 70), innerWidth / innerHeight, 0.02, 200); cam.rotation.order = 'YXZ';
const g = await new GLTFLoader().loadAsync('./world.glb'); const root = new THREE.Group(); root.rotation.x = Math.PI; root.add(g.scene); scene.add(root);  // OpenCV -> three.js
document.getElementById('load').remove();
const parts = {Photo: [], Gen: []}; g.scene.traverse(o => { if (o.isMesh) { const n = o.name + ' ' + (o.parent?.name || ''); const k = /Photo/.test(n) ? 'Photo' : 'Gen';
  o.material = new THREE.MeshBasicMaterial({map: o.material.map, side: THREE.DoubleSide}); parts[k].push(o); } });
let yaw = 0, pitch = 0; const home = () => { cam.position.set(0, 0, 0); yaw = 0; pitch = 0; }; home();
let drag = null; r.domElement.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY]; r.domElement.setPointerCapture(e.pointerId); });
r.domElement.addEventListener('pointerup', () => drag = null);
r.domElement.addEventListener('pointermove', e => { if (!drag) return; const k = 0.0045 * cam.fov / 70; yaw += (e.clientX - drag[0]) * k; pitch += (e.clientY - drag[1]) * k;
  pitch = Math.max(-1.5, Math.min(1.5, pitch)); drag = [e.clientX, e.clientY]; });
const fwd = () => { const f = new THREE.Vector3(); cam.getWorldDirection(f); return f; };
const clampPos = () => { if (cam.position.length() > info.walk_m) cam.position.setLength(info.walk_m); };
r.domElement.addEventListener('wheel', e => { e.preventDefault(); cam.position.addScaledVector(fwd(), -Math.sign(e.deltaY) * 0.1); clampPos(); }, {passive: false});
const B = id => document.getElementById(id); const sel = id => ['all', 'photo', 'gen'].forEach(k => B(k).classList.toggle('on', k === id));
const show = (gen, tint) => { parts.Gen.forEach(o => { o.visible = gen; o.material.color.set(tint ? 0xff9f6b : 0xffffff); }); };
B('all').onclick = () => { sel('all'); show(true, false); }; B('photo').onclick = () => { sel('photo'); show(false, false); };
B('gen').onclick = () => { sel('gen'); show(true, true); }; B('reset').onclick = home;
const keys = {}; addEventListener('keydown', e => keys[e.code] = true); addEventListener('keyup', e => keys[e.code] = false); const clock = new THREE.Clock();
r.setAnimationLoop(() => { const dt = Math.min(clock.getDelta(), 0.05); cam.rotation.set(pitch, yaw, 0);
  const f = fwd(); f.y = 0; f.normalize(); const s = new THREE.Vector3(-f.z, 0, f.x); const mv = new THREE.Vector3();
  if (keys.KeyW || keys.ArrowUp) mv.add(f); if (keys.KeyS || keys.ArrowDown) mv.sub(f); if (keys.KeyD || keys.ArrowRight) mv.add(s); if (keys.KeyA || keys.ArrowLeft) mv.sub(s);
  if (mv.lengthSq()) { cam.position.addScaledVector(mv.normalize(), 0.8 * dt); clampPos(); } r.render(scene, cam); });
addEventListener('resize', () => { cam.aspect = innerWidth / innerHeight; cam.updateProjectionMatrix(); r.setSize(innerWidth, innerHeight); });
</script></body></html>"""
