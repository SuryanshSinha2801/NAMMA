"""Clay room (approved plan 27 Oct, steps 1-4): the filmed room as clean, solid 3D geometry with real-photo textures, so nothing can
smear or float by construction. Gaussian splats stay the measuring tool, not the deliverable.
  1 shell     floor and ceiling heights from the splats' height histogram; room axes from the wall normals (4-fold mean); walls =
              vertical planes found along those 2 axes (Manhattan); the floor plan is the union of grid cells between the wall
              lines that hold floor / ceiling / furniture / camera evidence, so the outline has exactly the real corners (the
              26 Oct hybrid demo traced 78). Walls are 10 cm solid boxes. Openings: SfM points behind a wall, seen through it by
              their own cameras, mark where the wall is open (door / window); a reflection test keeps mirrors closed.
  2 textures  every wall / floor / ceiling texel (6 mm) gets the real training photo that sees it best (frontal, close,
              unoccluded by the clean shell + furniture blocks; Open3D ray casting), mostly from one view (sharp), blended only
              between near-equal views. Held-out frames are never used, so renders there are a fair test.
  3 furniture each object id of the splat model (objects3d / objects_refine) becomes a solid: floor-standing things a smoothed
              height-field block (sofa back and seat, bed and headboard read as shapes), wall-hung things a box; flat things on a
              wall (art) stay part of the wall texture. Vertex colours from the real photos at 1.5 cm.
  4 output    room.glb (named parts: Floor, Ceiling, Wall_k, <label>_k; opens in Blender), a three.js viewer (orbit / walk with
              collision / clay or textured / dollhouse), held-out PSNR/SSIM next to the splat model, off-path sheet, report.json.
Coordinates: metres, y up, floor at 0 (the run's level matrix x metres per unit, as the tour).
Usage (from gsplat's examples dir): python clay_room.py <data dir> <run dir: full/model.ply, full/level.json> <splat_object.npy | ->
  <objects.json | -> <out dir> [compare views, default 12]"""
import io, json, os, sys, time, numpy as np, cv2
from scipy import ndimage
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
T0 = time.time()
TEX = float(os.environ.get('CLAY_TEX', 0.006)); THICK = 0.10; UP = np.array([0., 1, 0])


def say(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


# ------------------------------------------------------------------ step 1: shell
def floor_ceiling(y):
    """Floor = densest height band near 0, ceiling = densest band 1.8-3.8 m (2 cm bins)."""
    h, e = np.histogram(y, bins=np.arange(-0.6, 4.0, 0.02)); c = (e[:-1] + e[1:]) / 2
    lo, hi = (c > -0.5) & (c < 0.6), (c > 1.8) & (c < 3.8)
    f = float(c[lo][np.argmax(h[lo])])
    ce = float(c[hi][np.argmax(h[hi])]) if h[hi].max() >= 30 else float(np.percentile(y, 99.5))
    return f, ce


def normals(P, k=16):
    """PCA normals + planarity of each point's k-neighbourhood."""
    _, nb = cKDTree(P).query(P, k=k, workers=-1); Q = P[nb] - P[nb].mean(1, keepdims=True)
    w, v = np.linalg.eigh(np.einsum('nki,nkj->nij', Q, Q))
    return v[:, :, 0], (w[:, 1] - w[:, 0]) / np.maximum(w[:, 2], 1e-12)


def room_axes(n, planar):
    h = np.abs(n[:, 1]) < 0.25; phi = np.arctan2(n[h, 2], n[h, 0])
    th = float(np.angle(np.sum(planar[h] * np.exp(4j * phi))) / 4)
    return np.array([np.cos(th), 0, np.sin(th)]), np.array([-np.sin(th), 0, np.cos(th)]), th


def wall_lines(c, lat, y, height, min_pts):
    """Wall planes along one axis: peaks of the coordinate histogram of the points whose normal points along that axis, with
    enough support (>= min_pts within 4 cm), width >= 0.5 m and height >= 40 % of the room; 25 cm apart at least."""
    if len(c) < min_pts: return []
    bins = np.arange(c.min() - 0.05, c.max() + 0.06, 0.01); h, _ = np.histogram(c, bins); hs = ndimage.gaussian_filter1d(h.astype(float), 1.5)
    out = []
    for i in np.argsort(-hs):
        if hs[i] <= 0: break
        x = bins[i] + 0.005
        if any(abs(x - o[0]) < 0.25 for o in out): continue
        m = np.abs(c - x) < 0.04
        if m.sum() < min_pts: continue
        wl = np.percentile(lat[m], 97) - np.percentile(lat[m], 3); hv = np.percentile(y[m], 97) - np.percentile(y[m], 3)
        if wl >= 0.5 and hv >= 0.4 * height: out.append((float(np.median(c[m])), int(m.sum()), round(float(wl), 2)))
        if len(out) >= 12: break
    return sorted(out)


def fit_shell(P, furn, cams, sfm=None, sfm_cams=None):
    """P [n,3] structure points (metres, y up), furn bool [n], cams [m,3] camera centres. sfm: (points [k,3], list of camera index
    arrays that saw each point) for openings. Returns the shell description."""
    f, ce = floor_ceiling(P[:, 1]); H = ce - f
    band = (P[:, 1] > f + 0.25) & (P[:, 1] < ce - 0.15) & ~furn; Wp = P[band]
    n, planar = normals(Wp); good = (planar > 0.5) & (np.abs(n[:, 1]) < 0.25)
    e1, e2, th = room_axes(n[good], planar[good])
    uv = lambda X: np.stack([X @ e1, X @ e2], 1)
    q = uv(Wp); a1 = good & (np.abs(n @ e1) > 0.9); a2 = good & (np.abs(n @ e2) > 0.9)
    mp = max(60, int(0.01 * good.sum()))
    Lu = wall_lines(q[a1, 0], q[a1, 1], Wp[a1, 1], H, mp); Lv = wall_lines(q[a2, 1], q[a2, 0], Wp[a2, 1], H, mp)
    # evidence of "inside the room" on a 2 cm raster: floor, ceiling, furniture, camera path
    ev_pts = P[(np.abs(P[:, 1] - f) < 0.04) | (np.abs(P[:, 1] - ce) < 0.05) | (furn & (P[:, 1] < f + 1.2))]
    qe, qc = uv(ev_pts), uv(cams)
    lo = np.minimum(np.percentile(qe, 0.5, 0), qc.min(0)) - 0.5; hi = np.maximum(np.percentile(qe, 99.5, 0), qc.max(0)) + 0.5
    g = 0.02; nu, nv = np.ceil((hi - lo) / g).astype(int) + 1
    ev = np.zeros((nu, nv), bool); ii = ((qe - lo) / g).astype(int); ok = np.all((ii >= 0) & (ii < [nu, nv]), 1); ev[ii[ok, 0], ii[ok, 1]] = True
    cm = np.zeros_like(ev); ic = ((qc - lo) / g).astype(int); cm[ic[:, 0], ic[:, 1]] = True
    ev |= ndimage.binary_dilation(cm, iterations=12); ev = ndimage.binary_closing(ndimage.binary_dilation(ev, iterations=3), iterations=4)
    e_u, e_v = np.nonzero(ev); ext_u = lo[0] + g * np.percentile(e_u, [1, 99]); ext_v = lo[1] + g * np.percentile(e_v, [1, 99])

    def grid(lines, ext):
        xs = [x for x, *_ in lines]
        for e in ext:
            if all(abs(e - x) > 0.3 for x in xs): xs.append(float(e))
        return np.array(sorted(xs))
    U, V = grid(Lu, ext_u), grid(Lv, ext_v)
    inside = np.zeros((len(U) - 1, len(V) - 1), bool)
    for i in range(len(U) - 1):
        for j in range(len(V) - 1):
            a0, a1_ = int((U[i] - lo[0]) / g), int((U[i + 1] - lo[0]) / g); b0, b1 = int((V[j] - lo[1]) / g), int((V[j + 1] - lo[1]) / g)
            blk = ev[max(a0, 0):max(a1_, a0 + 1), max(b0, 0):max(b1, b0 + 1)]
            inside[i, j] = blk.size > 0 and blk.mean() > 0.4
    camcell = np.zeros_like(inside)
    for x, z in qc:
        i, j = np.searchsorted(U, x) - 1, np.searchsorted(V, z) - 1
        if 0 <= i < inside.shape[0] and 0 <= j < inside.shape[1]: camcell[i, j] = True
    lab, _ = ndimage.label(inside); keep = np.unique(lab[camcell & inside]); inside = ndimage.binary_fill_holes(np.isin(lab, keep[keep > 0]))
    # walls: grid lines between an inside and an outside cell, merged along the line
    walls = []
    pad = np.pad(inside, 1)
    for axis, lines_, other in ((0, U, V), (1, V, U)):
        for k in range(len(lines_)):
            runs, cur = [], None
            for j in range(len(other) - 1):
                if axis == 0: a, b = pad[k, j + 1], pad[k + 1, j + 1]                      # cell before / after the line
                else: a, b = pad[j + 1, k], pad[j + 1, k + 1]
                side = 0 if a == b else (-1 if a else 1)                                    # -1: inside before the line
                if side and cur and cur[2] == side and abs(cur[1] - other[j]) < 1e-9: cur[1] = other[j + 1]
                else:
                    if cur: runs.append(cur)
                    cur = [other[j], other[j + 1], side] if side else None
            if cur: runs.append(cur)
            for s0, s1, side in runs: walls.append(dict(axis=axis, c=float(lines_[k]), s0=float(s0), s1=float(s1), inside=int(side),
                                                       detected=bool(any(abs(lines_[k] - x) < 1e-6 for x, *_ in (Lu if axis == 0 else Lv)))))
    corners = 0
    for i in range(pad.shape[0] - 1):
        for j in range(pad.shape[1] - 1):
            blk = pad[i:i + 2, j:j + 2]; s_ = int(blk.sum())
            corners += s_ in (1, 3) or (s_ == 2 and blk[0, 0] == blk[1, 1])
    sh = dict(floor=f, ceiling=ce, height=H, e1=e1, e2=e2, theta_deg=float(np.degrees(th)), U=U, V=V, inside=inside, walls=walls,
              corners=int(corners), lines_u=Lu, lines_v=Lv, area_m2=float(sum((U[i + 1] - U[i]) * (V[j + 1] - V[j]) for i, j in zip(*np.nonzero(inside)))))
    for w in walls: w['openings'] = []
    if sfm is not None: openings(sh, *sfm, cams=cams, front=P)
    return sh


def wall_frame(sh, w):
    """Inner face of a wall: origin (start, floor), along-wall unit a, up b, normal n into the room, plane coordinate."""
    e = sh['e1'] if w['axis'] == 0 else sh['e2']; t = sh['e2'] if w['axis'] == 0 else sh['e1']
    n = -e if w['inside'] == -1 else e                                                       # inside before the line -> faces back
    o = w['c'] * e + w['s0'] * t + sh['floor'] * UP
    return o, t, UP.copy(), n, w['s1'] - w['s0']


def openings(sh, pts, seen_by, cams, front, cell=0.05):
    """For each wall: SfM points behind it (> 12 cm), joined to the cameras that saw them; where those sight lines cross the wall
    plane the wall is open. Components >= 0.2 m2 become door (bottom near the floor) or window rectangles. Mirror test: points
    reflected back across the plane land on real points in front (> 50 % within 6 cm) -> a mirror, the wall stays closed."""
    tree = cKDTree(front)
    for w in sh['walls']:
        o, a, b, n, L = wall_frame(sh, w); d = (pts - o) @ n                               # < 0: behind the wall
        s_ = (pts - o) @ a; yv = pts[:, 1]
        cand = np.nonzero((d < -0.12) & (s_ > -0.5) & (s_ < L + 0.5) & (yv > sh['floor'] - 0.5) & (yv < sh['ceiling'] + 0.5))[0]
        if len(cand) < 10: continue
        ns, ny = int(np.ceil(L / cell)), int(np.ceil(sh['height'] / cell)); hit = np.zeros((ns, ny)); via = {}
        for k in cand:
            for c in seen_by[k]:
                cc = cams[c]; dc = (cc - o) @ n
                if dc <= 0.05: continue                                                     # camera must be inside
                t = dc / (dc - d[k]); X = cc + t * (pts[k] - cc); ss, yy = (X - o) @ a, X[1] - sh['floor']
                if 0 <= ss < L and 0 <= yy < sh['height']:
                    ci = (int(ss / cell), int(yy / cell)); hit[ci] += 1; via.setdefault(ci, []).append(k)
        m = ndimage.binary_closing(ndimage.gaussian_filter(hit, 1.0) > 0.3, iterations=2)
        lab, nl = ndimage.label(m)
        for r in range(1, nl + 1):
            ii, jj = np.nonzero(lab == r)
            if len(ii) * cell * cell < 0.2: continue
            s0, s1, y0, y1 = ii.min() * cell, (ii.max() + 1) * cell, jj.min() * cell, (jj.max() + 1) * cell
            inbox = np.unique([k for ci in zip(ii, jj) for k in via.get(ci, [])]).astype(int)        # points seen through it
            refl = pts[inbox] - 2 * ((pts[inbox] - o) @ n)[:, None] * n[None]
            mirror = len(inbox) >= 10 and float((tree.query(refl, k=1)[0] < 0.06).mean()) > 0.5
            # open down to the floor (no wall surface below the hole) -> a door, even if nothing low was seen through it
            dfr = (front - o) @ n; sf = (front - o) @ a; below = (np.abs(dfr) < 0.04) & (sf > s0) & (sf < s1) & (front[:, 1] - sh['floor'] < y0) & (front[:, 1] - sh['floor'] > 0.05)
            wall_density = below.sum() / max((s1 - s0) * max(y0 - 0.05, 0.05), 1e-6)
            near = (np.abs(dfr) < 0.04) & (sf > 0) & (sf < L); ref_density = near.sum() / max(L * sh['height'], 1e-6)
            kind = 'mirror' if mirror else ('door' if y0 < 0.2 or (y0 < 1.0 and wall_density < 0.25 * ref_density) else 'window')
            if kind == 'door': y0 = 0.0
            s0, s1, y1 = max(s0, 0.05), min(s1, L - 0.05), min(y1, sh['height'] - 0.05)
            if s1 - s0 > 0.3 and y1 - y0 > 0.4: w['openings'].append(dict(kind=kind, s0=round(s0, 3), s1=round(s1, 3), y0=round(y0, 3), y1=round(y1, 3)))


def wall_pieces(L, H, holes):
    """Rectangles covering [0,L] x [0,H] minus the hole rectangles (s0, s1, y0, y1)."""
    xs = sorted({0.0, L} | {h[k] for h in holes for k in (0, 1)}); ys = sorted({0.0, H} | {h[k] for h in holes for k in (2, 3)})
    out = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if not any(h[0] <= cx <= h[1] and h[2] <= cy <= h[3] for h in holes): out.append([xs[i], xs[i + 1], ys[j], ys[j + 1]])
    merged = True                                                                           # merge vertical neighbours
    while merged:
        merged = False
        for p in out:
            q = next((q for q in out if q is not p and q[0] == p[0] and q[1] == p[1] and abs(q[2] - p[3]) < 1e-9), None)
            if q: p[3] = q[3]; out.remove(q); merged = True; break
    return out


# ------------------------------------------------------------------ step 3: furniture blocks
def heightfield_block(Q, f, e1, e2, cell=0.04):
    """Floor-standing object -> closed-looking solid: top surface from the 95th-percentile height per 4 cm cell (median-smoothed),
    side skirt down to the floor. Q points in metres."""
    import trimesh
    uv = np.stack([Q @ e1, Q @ e2], 1); lo = uv.min(0) - cell; ij = ((uv - lo) / cell).astype(int); nu, nv = ij.max(0) + 2
    hmax = np.full((nu, nv), np.nan)
    key = ij[:, 0] * nv + ij[:, 1]
    for kk in np.unique(key):
        ys = Q[key == kk, 1]
        if len(ys) >= 2: hmax[kk // nv, kk % nv] = np.percentile(ys, 95)
    m = ~np.isnan(hmax); m = ndimage.binary_closing(m, iterations=1) | m
    m = ndimage.binary_opening(m, iterations=1) if m.sum() > 20 else m
    hm = np.where(np.isnan(hmax), np.nan, hmax)
    filled = ndimage.generic_filter(np.nan_to_num(hm, nan=-1), lambda w: np.median(w[w > -1]) if (w > -1).any() else -1, size=3)
    hgt = np.where(m, filled, -1); m &= hgt > f + 0.02
    if m.sum() < 3: return None
    V, F, vid = [], [], {}
    def corner_h(i, j):                                                                     # corner height: mean of adjacent cells
        hs = [hgt[a, b] for a in (i - 1, i) for b in (j - 1, j) if 0 <= a < nu and 0 <= b < nv and m[a, b]]
        return float(np.mean(hs))
    def P3(i, j, y): return (lo[0] + i * cell) * e1 + (lo[1] + j * cell) * e2 + y * UP
    def vtx(i, j, y):
        k = (i, j, round(y, 5))
        if k not in vid: vid[k] = len(V); V.append(P3(i, j, y))
        return vid[k]
    for i, j in zip(*np.nonzero(m)):
        a, b, c, d = vtx(i, j, corner_h(i, j)), vtx(i + 1, j, corner_h(i + 1, j)), vtx(i + 1, j + 1, corner_h(i + 1, j + 1)), vtx(i, j + 1, corner_h(i, j + 1))
        F += [[a, c, b], [a, d, c]]
        for di, dj, (p0, p1) in ((-1, 0, ((i, j), (i, j + 1))), (1, 0, ((i + 1, j + 1), (i + 1, j))), (0, -1, ((i + 1, j), (i, j))), (0, 1, ((i, j + 1), (i + 1, j + 1)))):
            ni, nj = i + di, j + dj
            if 0 <= ni < nu and 0 <= nj < nv and m[ni, nj]: continue                        # skirt on the footprint border
            t0, t1 = vtx(*p0, corner_h(*p0)), vtx(*p1, corner_h(*p1)); b0, b1 = vtx(*p0, f), vtx(*p1, f)
            F += [[t0, b0, b1], [t0, b1, t1]]
    mesh = trimesh.Trimesh(np.array(V), np.array(F), process=True); mesh.fix_normals()
    return mesh


def box_block(Q, e1, e2, pad=0.01):
    import trimesh
    R = np.stack([e1, UP, e2], 1); loc = Q @ R; lo, hi = np.percentile(loc, 3, 0) - pad, np.percentile(loc, 97, 0) + pad
    b = trimesh.creation.box(extents=hi - lo); b.apply_translation((lo + hi) / 2); b.apply_transform(np.r_[np.c_[R, np.zeros(3)], [[0, 0, 0, 1]]])
    return b


def furniture(P, ids, sh, names=None, min_pts=60):
    """Object ids -> solids. Floor-standing (bottom < 25 cm above the floor): height field; hung: box; flat on a wall (< 6 cm
    thick and within 10 cm of a wall plane): left to the wall texture."""
    out, skipped = [], []
    planes = [wall_frame(sh, w) for w in sh['walls']]
    for k in np.unique(ids[ids > 0]):
        Q = P[ids == k]
        if len(Q) < min_pts: continue
        name = (names or {}).get(int(k), f'object {k}')
        bottom = float(np.percentile(Q[:, 1], 3)) - sh['floor']
        dw = min([float(np.median(np.abs((Q - o) @ n))) for o, a, b, n, L in planes] or [9])
        R = np.stack([sh['e1'], UP, sh['e2']], 1); ext = np.percentile(Q @ R, 97, 0) - np.percentile(Q @ R, 3, 0)
        if dw < 0.10 and min(ext[0], ext[2]) < 0.06: skipped.append(name); continue
        mesh = heightfield_block(Q, sh['floor'], sh['e1'], sh['e2']) if bottom < 0.25 else box_block(Q, sh['e1'], sh['e2'])
        if mesh is None or not len(mesh.faces): skipped.append(name); continue
        out.append(dict(id=int(k), name=name, mesh=mesh, kind='height field' if bottom < 0.25 else 'box',
                        size_m=[round(float(x), 2) for x in ext]))
    return out, skipped


# ------------------------------------------------------------------ step 2: real-photo textures
def surfaces(sh):
    """Textured planes: floor, ceiling, each wall's inner face (openings cut, mirrors kept), and a recessed panel per opening
    (the photo of what is seen through it). Each: origin o, axes a / b, normal n into the room, size, texel size, valid mask,
    and the rectangles (in a / b metres) that become geometry."""
    e1, e2, U, V, ins = sh['e1'], sh['e2'], sh['U'], sh['V'], sh['inside']; out = []
    u0, u1, v0, v1 = U[0], U[-1], V[0], V[-1]
    for name, y, n, tex in (('Floor', sh['floor'], UP, TEX), ('Ceiling', sh['ceiling'], -UP, 2 * TEX)):
        nw, nh = int(np.ceil((u1 - u0) / tex)), int(np.ceil((v1 - v0) / tex)); mask = np.zeros((nh, nw), bool); rects = []
        for i, j in zip(*np.nonzero(ins)):
            a0, a1, b0, b1 = U[i] - u0, U[i + 1] - u0, V[j] - v0, V[j + 1] - v0; rects.append([a0, a1, b0, b1])
            mask[int(b0 / tex):int(np.ceil(b1 / tex)), int(a0 / tex):int(np.ceil(a1 / tex))] = True
        out.append(dict(name=name, o=u0 * e1 + v0 * e2 + y * UP, a=e1, b=e2, n=n, tex=tex, nw=nw, nh=nh, mask=mask, rects=rects))
    for k, w in enumerate(sh['walls']):
        o, a, b, n, L = wall_frame(sh, w); H = sh['height']; tex = TEX
        holes = [(p['s0'], p['s1'], p['y0'], p['y1']) for p in w['openings'] if p['kind'] != 'mirror']
        nw, nh = int(np.ceil(L / tex)), int(np.ceil(H / tex)); mask = np.ones((nh, nw), bool)
        for s0, s1, y0, y1 in holes: mask[int(y0 / tex):int(np.ceil(y1 / tex)), int(s0 / tex):int(np.ceil(s1 / tex))] = False
        out.append(dict(name=f'Wall_{k + 1}', o=o, a=a, b=b, n=n, tex=tex, nw=nw, nh=nh, mask=mask, rects=wall_pieces(L, H, holes), wall=k))
        for m_, (s0, s1, y0, y1) in enumerate(holes):                                     # what is seen through the opening
            oo = o + s0 * a + y0 * b - THICK * n; ww, hh = s1 - s0, y1 - y0
            out.append(dict(name=f'Wall_{k + 1}_opening_{m_ + 1}', o=oo, a=a, b=b, n=n, tex=tex, nw=int(np.ceil(ww / tex)), nh=int(np.ceil(hh / tex)),
                            mask=np.ones((int(np.ceil(hh / tex)), int(np.ceil(ww / tex))), bool), rects=[[0, ww, 0, hh]], opening=True))
    return out


def quads(S):
    """Triangles of a surface's rectangles: vertices, faces, uv (trimesh convention: v up)."""
    V, F, UVs = [], [], []; A, B = S['nw'] * S['tex'], S['nh'] * S['tex']
    for a0, a1, b0, b1 in S['rects']:
        k = len(V); V += [S['o'] + a0 * S['a'] + b0 * S['b'], S['o'] + a1 * S['a'] + b0 * S['b'], S['o'] + a1 * S['a'] + b1 * S['b'], S['o'] + a0 * S['a'] + b1 * S['b']]
        UVs += [[a0 / A, 1 - b0 / B], [a1 / A, 1 - b0 / B], [a1 / A, 1 - b1 / B], [a0 / A, 1 - b1 / B]]
        f = [[k, k + 1, k + 2], [k, k + 2, k + 3]]
        if np.dot(np.cross(V[k + 1] - V[k], V[k + 2] - V[k]), S['n']) < 0: f = [[k, k + 2, k + 1], [k, k + 3, k + 2]]
        F += f
    return np.array(V), np.array(F), np.array(UVs)


def wall_bodies(sh):
    """Solid 10 cm walls behind the inner faces (with the openings cut), plus corner posts outside convex corners."""
    import trimesh
    parts = []
    for k, w in enumerate(sh['walls']):
        o, a, b, n, L = wall_frame(sh, w)
        holes = [(p['s0'], p['s1'], p['y0'], p['y1']) for p in w['openings'] if p['kind'] != 'mirror']
        for s0, s1, y0, y1 in wall_pieces(L, sh['height'], holes):
            bx = trimesh.creation.box(extents=[s1 - s0, y1 - y0, THICK - 0.002])           # 2 mm behind the textured face:
            T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2] = a, b, -n                          # no z-fighting with it
            T[:3, 3] = o + (s0 + s1) / 2 * a + (y0 + y1) / 2 * b - (THICK / 2 + 0.001) * n
            if np.linalg.det(T[:3, :3]) < 0: T[:3, 0] *= -1
            bx.apply_transform(T); parts.append(bx)
    U, V, pad = sh['U'], sh['V'], np.pad(sh['inside'], 1)
    for i in range(pad.shape[0] - 1):
        for j in range(pad.shape[1] - 1):
            blk = pad[i:i + 2, j:j + 2]
            if blk.sum() != 1: continue
            ii, jj = np.argwhere(blk)[0]; du, dv = (1 if ii == 0 else -1), (1 if jj == 0 else -1)   # post towards the outside
            cu, cv = U[i], V[j]; ctr = (cu + du * THICK / 2) * sh['e1'] + (cv + dv * THICK / 2) * sh['e2'] + (sh['floor'] + sh['height'] / 2) * UP
            bx = trimesh.creation.box(extents=[THICK, sh['height'], THICK]); R = np.stack([sh['e1'], UP, sh['e2']], 1)
            T = np.eye(4); T[:3, :3] = R if np.linalg.det(R) > 0 else R * [1, 1, -1]; T[:3, 3] = ctr; bx.apply_transform(T); parts.append(bx)
    return parts


def colour_sharp(X, N, cams, imgs, K, sc, eps, k=3, power=8.0):
    """hybrid_demo.colour with weights (score / best score) ^ power: mostly the single best view (sharp), blended only with
    near-equal views (no hard seams)."""
    from hybrid_demo import cast, bilinear
    n = len(X); bs = np.full((n, k), -np.inf, np.float32); bf = np.zeros((n, k), np.int32); H, W = imgs[0].shape[:2]; ar = np.arange(n)
    for f, c2w in enumerate(cams):
        w2c = np.linalg.inv(c2w); pc = X @ w2c[:3, :3].T + w2c[:3, 3]; z = pc[:, 2]
        u = pc[:, 0] / np.maximum(z, 1e-6) * K[0, 0] + K[0, 2]; v = pc[:, 1] / np.maximum(z, 1e-6) * K[1, 1] + K[1, 2]
        dv = c2w[:3, 3] - X; d = np.linalg.norm(dv, axis=1); cos = np.einsum('ij,ij->i', N, dv) / np.maximum(d, 1e-9)
        ok = (z > 0.05) & (u > 2) & (u < W - 2) & (v > 2) & (v < H - 2) & (cos > 0.2)
        s = np.where(ok, cos ** 2 / np.maximum(d, 1e-6), -np.inf).astype(np.float32)
        j = bs.argmin(1); m = s > bs[ar, j]; bs[ar[m], j[m]] = s[m]; bf[ar[m], j[m]] = f
    best = np.full(n, -np.inf); acc = np.zeros((n, 3)); wsum = np.zeros(n); vis_s = np.full((n, k), -np.inf)
    for slot in range(k):                                                                  # visibility first
        idx = np.nonzero(np.isfinite(bs[:, slot]))[0]
        if not len(idx): continue
        cc = np.stack([cams[f][:3, 3] for f in bf[idx, slot]]); dv = X[idx] - cc; d = np.linalg.norm(dv, axis=1)
        t, _, _ = cast(sc, cc, dv / d[:, None]); vis = np.isfinite(t) & (t > d - eps); vis_s[idx[vis], slot] = bs[idx[vis], slot]
    best = vis_s.max(1)
    for slot in range(k):
        idx = np.nonzero(np.isfinite(vis_s[:, slot]))[0]
        for f in np.unique(bf[idx, slot]):
            q = idx[bf[idx, slot] == f]; w2c = np.linalg.inv(cams[f]); pc = X[q] @ w2c[:3, :3].T + w2c[:3, 3]
            rgb = bilinear(imgs[f], pc[:, 0] / pc[:, 2] * K[0, 0] + K[0, 2], pc[:, 1] / pc[:, 2] * K[1, 1] + K[1, 2])
            w = (vis_s[q, slot] / best[q]).astype(np.float64) ** power; acc[q] += rgb * w[:, None]; wsum[q] += w
    seen = wsum > 0; out = np.zeros((n, 3)); out[seen] = acc[seen] / wsum[seen, None]
    return out, seen


def texture_surface(S, cams, imgs, K, sc, eps=0.03):
    ii, jj = np.meshgrid(np.arange(S['nw']), np.arange(S['nh'])); t = S['tex']
    X = S['o'] + ((ii.ravel() + 0.5) * t)[:, None] * S['a'] + ((jj.ravel() + 0.5) * t)[:, None] * S['b'] + 1e-3 * S['n']
    msk = S['mask'].ravel(); rgb = np.zeros((len(X), 3)); seen = np.zeros(len(X), bool)
    if msk.any(): rgb[msk], seen[msk] = colour_sharp(X[msk], np.tile(S['n'], (msk.sum(), 1)), cams, imgs, K, sc, eps)
    img = (rgb.reshape(S['nh'], S['nw'], 3) * 255).clip(0, 255).astype(np.uint8); hole = (~seen & msk).reshape(S['nh'], S['nw'])
    if hole.any() and seen.any(): img = cv2.inpaint(img, hole.astype(np.uint8), 7, cv2.INPAINT_TELEA)
    S['img'] = img; S['seen'] = seen.reshape(S['nh'], S['nw']); S['seen_pct'] = round(100 * float(seen.sum()) / max(1, int(msk.sum())), 1)


# ------------------------------------------------------------------ step 4: scene, renders, export
class Scene:
    """Everything as triangles for ray casting (Open3D): per triangle a source (surface index, or -1 - block index, or a wall
    body / post = -10000) for texture / colour lookup."""
    def __init__(self, surfs, blocks, bodies):
        from hybrid_demo import scene_of
        Vs, Fs, src = [], [], []; n0 = 0
        for si, S in enumerate(surfs):
            V, F, _ = quads(S); Vs.append(V); Fs.append(F + n0); src += [si] * len(F); n0 += len(V)
        for bi, B in enumerate(blocks):
            V, F = np.asarray(B['mesh'].vertices), np.asarray(B['mesh'].faces); B['f0'] = len(src); B['v0'] = n0
            Vs.append(V); Fs.append(F + n0); src += [-1 - bi] * len(F); n0 += len(V)
        for m in bodies:
            V, F = np.asarray(m.vertices), np.asarray(m.faces); Vs.append(V); Fs.append(F + n0); src += [-10000] * len(F); n0 += len(V)
        self.V, self.F, self.src = np.concatenate(Vs), np.concatenate(Fs), np.array(src); self.sc = scene_of(self.V, self.F)
        self.surfs, self.blocks = surfs, blocks

    def render(self, c2w, K, W, H, clay=False, hide_ceiling=False):
        from hybrid_demo import cast, bilinear
        uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        d = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1).reshape(-1, 3) @ c2w[:3, :3].T
        d /= np.linalg.norm(d, axis=1, keepdims=True); o = np.tile(c2w[:3, 3], (len(d), 1))
        tt, pid, buv = cast(self.sc, o, d); pid = np.where(np.isfinite(tt), pid, 0)        # misses carry a huge id
        if hide_ceiling:                                                                    # dollhouse: ceiling not drawn
            ci = [i for i, S in enumerate(self.surfs) if S['name'] == 'Ceiling']
            for _ in range(3):
                m = np.isfinite(tt) & np.isin(self.src[np.maximum(pid, 0)], ci)
                if not m.any(): break
                o2 = o[m] + (tt[m, None] + 1e-3) * d[m]; t2, p2, b2 = cast(self.sc, o2, d[m]); tt[m] = tt[m] + 1e-3 + t2; pid[m] = np.where(np.isfinite(t2), p2, 0); buv[m] = b2
        hit = np.isfinite(tt); img = np.full((len(d), 3), 0.08); region = np.full(len(d), -1)
        src = np.full(len(d), -20000); src[hit] = self.src[pid[hit]]
        tri = self.F[np.maximum(pid, 0)]; nrm = np.cross(self.V[tri[:, 1]] - self.V[tri[:, 0]], self.V[tri[:, 2]] - self.V[tri[:, 0]])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        if clay:
            L1, L2 = np.array([0.35, 0.85, 0.4]) / np.linalg.norm([0.35, 0.85, 0.4]), -d
            sh_ = 0.25 + 0.5 * np.abs(nrm @ L1) + 0.25 * np.abs(np.einsum('ij,ij->i', nrm, L2))
            base = np.where((src <= -1) & (src > -10000), 0.78, 0.86)
            img[hit] = (np.c_[base, base * 0.98, base * 0.94] * sh_[:, None])[hit]
        else:
            for si, S in enumerate(self.surfs):
                m = src == si
                if not m.any(): continue
                X = o[m] + tt[m, None] * d[m] - S['o']; img[m] = bilinear(S['img'], X @ S['a'] / S['tex'], X @ S['b'] / S['tex'])
            for bi, B in enumerate(self.blocks):
                m = src == -1 - bi
                if not m.any(): continue
                f = self.F[pid[m]] - B['v0']; b1, b2 = buv[m, 0:1], buv[m, 1:2]; c = B['colors']
                img[m] = (1 - b1 - b2) * c[f[:, 0]] + b1 * c[f[:, 1]] + b2 * c[f[:, 2]]
            m = src == -10000; img[m] = 0.8 * (0.4 + 0.6 * np.abs(nrm[m] @ np.array([0.3, 0.9, 0.3]) / np.linalg.norm([0.3, 0.9, 0.3])))[:, None]
        region[hit] = np.where(src[hit] >= 0, 0, 1)                                         # 0 shell, 1 furniture / bodies
        return (img.reshape(H, W, 3) * 255).clip(0, 255).astype(np.uint8), region.reshape(H, W)


def export_glb(surfs, blocks, bodies, path):
    import trimesh
    from PIL import Image
    scene = trimesh.Scene()
    for S in surfs:
        V, F, UVs = quads(S); jpg = cv2.imencode('.jpg', cv2.cvtColor(S['img'], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
        im = Image.open(io.BytesIO(jpg)); im.load()
        mat = trimesh.visual.material.PBRMaterial(name=S['name'], baseColorTexture=im, metallicFactor=0.0, roughnessFactor=1.0)
        m = trimesh.Trimesh(V, F, visual=trimesh.visual.TextureVisuals(uv=UVs, material=mat), process=False)
        scene.add_geometry(m, node_name=S['name'], geom_name=S['name'])
    for B in blocks:
        m = B['mesh'].copy(); m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=(np.c_[B['colors'], np.ones(len(B['colors']))] * 255).astype(np.uint8))
        nm = f"{B['name'].replace(' ', '_')}_{B['id']}"; scene.add_geometry(m, node_name=nm, geom_name=nm)
    body = trimesh.util.concatenate(bodies); body.visual = trimesh.visual.ColorVisuals(body, face_colors=[214, 210, 202, 255])
    scene.add_geometry(body, node_name='Wall_bodies', geom_name='Wall_bodies')
    scene.export(path); return os.path.getsize(path)


def colour_blocks(blocks, cams, imgs, K, sc, edge=0.015):
    """Furniture vertex colours from the real photos (blocks subdivided to 1.5 cm first); unseen vertices take the median."""
    for B in blocks:
        m = B['mesh']; n = m.vertex_normals; c, seen = colour_sharp(np.asarray(m.vertices) + 2e-3 * n, n, cams, imgs, K, sc, 0.03)
        if seen.any(): c[~seen] = np.median(c[seen], 0)
        else: c[:] = 0.6
        B['colors'] = c; B['seen_pct'] = round(100 * float(seen.mean()), 1)


VIEWER = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NAMMA clay room</title><style>
body{margin:0;background:#1c1d20;overflow:hidden;font:13px system-ui,sans-serif;color:#eee}
#ui{position:fixed;left:12px;top:12px;display:flex;gap:6px;flex-wrap:wrap;z-index:2}
#ui button{background:#2b2d31ee;color:#eee;border:1px solid #fff3;border-radius:16px;padding:7px 12px;cursor:pointer;font:inherit}
#ui button.on{background:#e8a96b;color:#111}
#help{position:fixed;left:12px;bottom:12px;background:#0009;padding:6px 10px;border-radius:8px;z-index:2;max-width:70vw}
</style></head><body>
<div id="ui"><button id="tex" class="on">Textured</button><button id="clay">Clay</button><button id="doll">Dollhouse</button><button id="walk">Walk</button><button id="reset">Reset view</button></div>
<div id="help">Drag to look around, scroll to zoom. Walk: click Walk, then W A S D, mouse to look, Esc to stop.</div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {PointerLockControls} from 'three/addons/controls/PointerLockControls.js';
const info = await (await fetch('./room.json')).json();
const renderer = new THREE.WebGLRenderer({antialias:true}); renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace; document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x1c1d20);
const cam = new THREE.PerspectiveCamera(65, innerWidth / innerHeight, 0.03, 100);
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8478, 1.6)); const sun = new THREE.DirectionalLight(0xffffff, 1.4); sun.position.set(3, 8, 4); scene.add(sun);
const orbit = new OrbitControls(cam, renderer.domElement); orbit.enableDamping = true;
const walk = new PointerLockControls(cam, document.body); scene.add(walk.getObject());
const gltf = await new GLTFLoader().loadAsync('./room.glb'); scene.add(gltf.scene);
const meshes = [], ceiling = [];
gltf.scene.traverse(o => { if (!o.isMesh) return; meshes.push(o); if (/^Ceiling/.test(o.name) || /^Ceiling/.test(o.parent?.name || '')) ceiling.push(o);
  const m = o.material; o.userData.tex = new THREE.MeshBasicMaterial({map: m.map || null, vertexColors: !!o.geometry.attributes.color, color: m.map || o.geometry.attributes.color ? 0xffffff : 0xd6d2ca, side: THREE.DoubleSide});
  o.userData.clay = new THREE.MeshStandardMaterial({color: 0xe8e3da, roughness: 0.95, metalness: 0, side: THREE.DoubleSide}); o.material = o.userData.tex; });
const solid = meshes.filter(o => !ceiling.includes(o) && !/^Floor/.test(o.name));
const c = new THREE.Vector3(...info.centre);
function home() { cam.position.set(c.x - 1.2 * info.e1[0], info.floor + 1.55, c.z - 1.2 * info.e1[2]); orbit.target.set(c.x + info.e1[0], info.floor + 1.2, c.z + info.e1[2]); orbit.update(); }
function doll() { ceiling.forEach(o => o.visible = false); cam.position.set(c.x - 0.6 * info.size, info.ceiling + 0.9 * info.size, c.z - 0.6 * info.size); orbit.target.set(c.x, info.floor, c.z); orbit.update(); }
home();
const btn = id => document.getElementById(id);
const mode = (m) => { meshes.forEach(o => o.material = o.userData[m]); btn('tex').classList.toggle('on', m === 'tex'); btn('clay').classList.toggle('on', m === 'clay'); };
btn('tex').onclick = () => mode('tex'); btn('clay').onclick = () => mode('clay');
btn('doll').onclick = () => { const on = !btn('doll').classList.contains('on'); btn('doll').classList.toggle('on', on); if (on) doll(); else { ceiling.forEach(o => o.visible = true); home(); } };
btn('reset').onclick = () => { ceiling.forEach(o => o.visible = true); btn('doll').classList.remove('on'); home(); };
btn('walk').onclick = () => { ceiling.forEach(o => o.visible = true); btn('doll').classList.remove('on'); cam.position.y = info.floor + 1.55; walk.lock(); };
walk.addEventListener('lock', () => { orbit.enabled = false; btn('walk').classList.add('on'); });
walk.addEventListener('unlock', () => { orbit.enabled = true; btn('walk').classList.remove('on'); orbit.target.copy(cam.position).add(cam.getWorldDirection(new THREE.Vector3())); orbit.update(); });
const keys = {}; addEventListener('keydown', e => keys[e.code] = true); addEventListener('keyup', e => keys[e.code] = false);
const ray = new THREE.Raycaster(); const clock = new THREE.Clock();
function blocked(dir) { for (const h of [0.35, 1.0]) { ray.set(new THREE.Vector3(cam.position.x, info.floor + h, cam.position.z), dir); ray.far = 0.3;
  if (ray.intersectObjects(solid, false).length) return true; } return false; }
renderer.setAnimationLoop(() => {
  const dt = Math.min(clock.getDelta(), 0.05);
  if (walk.isLocked) { const f = new THREE.Vector3(); cam.getWorldDirection(f); f.y = 0; f.normalize(); const r = new THREE.Vector3().crossVectors(f, cam.up);
    const mv = new THREE.Vector3(); if (keys.KeyW || keys.ArrowUp) mv.add(f); if (keys.KeyS || keys.ArrowDown) mv.sub(f); if (keys.KeyD || keys.ArrowRight) mv.add(r); if (keys.KeyA || keys.ArrowLeft) mv.sub(r);
    if (mv.lengthSq() > 0) { mv.normalize(); if (!blocked(mv)) cam.position.addScaledVector(mv, 1.4 * dt); } cam.position.y = info.floor + 1.55; }
  else orbit.update();
  renderer.render(scene, cam); });
addEventListener('resize', () => { cam.aspect = innerWidth / innerHeight; cam.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
window.viewerReady = true;
</script></body></html>"""

BLENDER = r"""import bpy, json, sys, math, mathutils
a = sys.argv[sys.argv.index('--') + 1:]; glb, info, png = a[0], json.load(open(a[1])), a[2]
bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=glb)
names = [o.name for o in bpy.data.objects if o.type == 'MESH']; print('BLENDER OBJECTS', len(names), names[:40], flush=True)
for o in bpy.data.objects:
    if o.name.startswith('Ceiling'): o.hide_render = True
c = info['centre']; s = info['size']; cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); bpy.context.scene.collection.objects.link(cam)
eye = mathutils.Vector((c[0] - 0.6 * s, -(c[2] - 0.6 * s), info['ceiling'] + 0.9 * s)); tgt = mathutils.Vector((c[0], -c[2], info['floor']))
cam.location = eye; cam.rotation_euler = (tgt - eye).to_track_quat('-Z', 'Y').to_euler(); bpy.context.scene.camera = cam
sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN')); sun.data.energy = 3; bpy.context.scene.collection.objects.link(sun); sun.rotation_euler = (0.6, 0.3, 0.5)
sc = bpy.context.scene; sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = 16; sc.render.resolution_x, sc.render.resolution_y = 960, 540
sc.render.filepath = png; bpy.ops.render.render(write_still=True); print('BLENDER RENDER OK', png, flush=True)"""


def main(data, run, sid_path, obj_json, out, nviews=12):
    import trimesh
    from plyfile import PlyData
    from hybrid_demo import scene_of
    sys.path.insert(0, os.getcwd()); from datasets.colmap import Parser
    import cpu_render
    from tour_fidelity import cameras
    from cleanup_cpu import ssim
    import torch
    os.makedirs(out, exist_ok=True); nviews = int(nviews)
    p = Parser(data, factor=1, normalize=True, test_every=8); n_img = len(p.image_names); cid = p.camera_ids[0]
    M = np.array(json.load(open(f'{run}/full/level.json'))['matrix']); R, t = M[:3, :3], M[:3, 3]
    C = np.einsum('ij,njk->nik', M, p.camtoworlds); s = float(1.35 / np.median(C[:, 1, 3]))  # metres per unit, as make_tour
    Cm = C.copy(); Cm[:, :3, 3] *= s

    def tm(c2w):                                                                            # model camera -> metres frame
        c = M @ c2w; c[:3, 3] *= s; return c
    v = PlyData.read(f'{run}/full/model.ply')['vertex'].data; X = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)
    op = 1 / (1 + np.exp(-np.asarray(v['opacity'], np.float64))); keep = op > 0.1; Pm = (X @ R.T + t) * s
    ids = np.load(sid_path).astype(int) if sid_path != '-' and os.path.exists(sid_path) else np.zeros(len(X), int)
    names = {o['id']: o.get('label') or f"object {o['id']}" for o in json.load(open(obj_json))['objects']} if obj_json != '-' and os.path.exists(obj_json) else {}
    sfm = (p.points @ R.T + t) * s; seen_by = [[] for _ in range(len(sfm))]
    for i, nm in enumerate(p.image_names):
        for k in p.point_indices.get(nm, []): seen_by[k].append(i)
    seen_by = [np.array(x, int) for x in seen_by]
    fur_tree = cKDTree(Pm[keep & (ids > 0)]) if (keep & (ids > 0)).any() else None
    sfm_furn = (fur_tree.query(sfm, k=1)[0] < 0.05) if fur_tree is not None else np.zeros(len(sfm), bool)
    Pst = np.vstack([Pm[keep], sfm]); furn = np.r_[ids[keep] > 0, sfm_furn]
    say(len(X), 'splats,', int(keep.sum()), 'used;', len(sfm), 'SfM points;', n_img, 'frames;', f'{s:.4f} m per unit')
    sh = fit_shell(Pst, furn, Cm[:, :3, 3], sfm=(sfm, seen_by))
    say(f"shell: floor {sh['floor']:.3f} ceiling {sh['ceiling']:.3f} (height {sh['height']:.2f} m), axes {sh['theta_deg']:.1f} deg, "
        f"{len(sh['walls'])} walls, {sh['corners']} corners, {sh['area_m2']:.1f} m2; lines u {sh['lines_u']} v {sh['lines_v']}")
    for w in sh['walls']: say('  wall', w['axis'], round(w['c'], 3), round(w['s1'] - w['s0'], 2), 'm', 'detected' if w['detected'] else 'from evidence', w['openings'])
    blocks, skipped = furniture(Pm[keep], ids[keep], sh, names); say(len(blocks), 'furniture blocks; part of the wall texture:', skipped)
    for B in blocks:
        V2, F2 = trimesh.remesh.subdivide_to_size(B['mesh'].vertices, B['mesh'].faces, max_edge=0.015, max_iter=12)
        B['mesh'] = trimesh.Trimesh(V2, F2, process=True)
    surfs = surfaces(sh); bodies = wall_bodies(sh); scn = Scene(surfs, blocks, bodies)
    # training photos (never the held-out ones), undistorted like gsplat's loader
    trn = [i for i in range(n_img) if i % 8]; imgs = []
    for i in trn:
        im = cv2.imread(p.image_paths[i])
        if cid in p.mapx_dict: im = cv2.remap(im, p.mapx_dict[cid], p.mapy_dict[cid], cv2.INTER_LINEAR); x, y, w_, h_ = p.roi_undist_dict[cid]; im = im[y:y + h_, x:x + w_]
        imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    K = p.Ks_dict[cid]; cams = [Cm[i] for i in trn]; say(len(imgs), 'training photos', imgs[0].shape)
    for S in surfs: texture_surface(S, cams, imgs, K, scn.sc); say(' ', S['name'], S['nw'], 'x', S['nh'], 'texels,', S['seen_pct'], '% photo')
    colour_blocks(blocks, cams, imgs, K, scn.sc); say('furniture colours:', [(B['name'], B['seen_pct']) for B in blocks])
    del imgs
    glb_mb = round(export_glb(surfs, blocks, bodies, f'{out}/room.glb') / 1e6, 1); say('room.glb', glb_mb, 'MB')
    for S in surfs: cv2.imwrite(f"{out}/tex_{S['name']}.jpg", cv2.cvtColor(S['img'], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    # compare with the splat model at the held-out frames and off the path
    Kc, Wc, Hc, real, off = cameras(data, 640, nviews, X, op, s); spl = cpu_render.load_ply(f'{run}/full/model.ply')
    def psnr(a, b, m=None):
        e = (a.astype(np.float64) / 255 - b.astype(np.float64) / 255) ** 2
        if m is not None: e = e[m]
        return round(float(-10 * np.log10(max(e.mean(), 1e-10))), 2)
    rows, res = [], dict(clay=[], splat=[], clay_shell=[], splat_shell=[], clay_ssim=[], splat_ssim=[])
    for n_, (img, c2w) in enumerate(real):
        g = (img.numpy() * 255).astype(np.uint8); cl, reg = scn.render(tm(c2w), Kc, Wc, Hc); sp = (cpu_render.render(spl, c2w, Kc, Wc, Hc).numpy() * 255).astype(np.uint8)
        res['clay'].append(psnr(cl, g)); res['splat'].append(psnr(sp, g)); shell = reg == 0
        res['clay_shell'].append(psnr(cl, g, shell)); res['splat_shell'].append(psnr(sp, g, shell))
        T_ = lambda a: torch.tensor(a, dtype=torch.float32) / 255; res['clay_ssim'].append(ssim(T_(cl), T_(g))); res['splat_ssim'].append(ssim(T_(sp), T_(g)))
        if n_ < 3: rows.append([g, sp, cl, scn.render(tm(c2w), Kc, Wc, Hc, clay=True)[0]])
        say('held-out', n_, 'clay', res['clay'][-1], 'splat', res['splat'][-1], '| shell pixels: clay', res['clay_shell'][-1], 'splat', res['splat_shell'][-1])
    for c2w in off[:3]:
        sp = (cpu_render.render(spl, c2w, Kc, Wc, Hc).numpy() * 255).astype(np.uint8)
        rows.append([np.full_like(sp, 40), sp, scn.render(tm(c2w), Kc, Wc, Hc)[0], scn.render(tm(c2w), Kc, Wc, Hc, clay=True)[0]])
    ctr = (sh['U'][0] + sh['U'][-1]) / 2 * sh['e1'] + (sh['V'][0] + sh['V'][-1]) / 2 * sh['e2']; size = float(max(sh['U'][-1] - sh['U'][0], sh['V'][-1] - sh['V'][0]))
    ctr = ctr + sh['floor'] * UP
    eye = ctr - 0.6 * size * sh['e1'] - 0.6 * size * sh['e2'] + (sh['ceiling'] - sh['floor'] + 0.9 * size) * UP; f_ = ctr - eye; f_ /= np.linalg.norm(f_)
    r_ = np.cross(f_, UP); r_ /= np.linalg.norm(r_); top = np.eye(4); top[:3, :3] = np.stack([r_, np.cross(f_, r_), f_], 1); top[:3, 3] = eye
    rows.append([np.full_like(rows[0][0], 40), np.full_like(rows[0][0], 40), scn.render(top, Kc, Wc, Hc, hide_ceiling=True)[0], scn.render(top, Kc, Wc, Hc, clay=True, hide_ceiling=True)[0]])
    sheet = np.concatenate([np.concatenate(r, 1) for r in rows], 0); hdr = np.full((28, sheet.shape[1], 3), 0, np.uint8)
    for i, tx in enumerate(['real photo (held out)', 'splat model (now)', 'clay room, textured', 'clay room, clay']):
        cv2.putText(hdr, tx, (i * Wc + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(f'{out}/clay_sheet.jpg', cv2.cvtColor(np.concatenate([hdr, sheet], 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    info = dict(centre=[round(float(x), 3) for x in ctr], floor=round(sh['floor'], 3), ceiling=round(sh['ceiling'], 3), size=round(size, 2), e1=[round(float(x), 4) for x in sh['e1']])
    json.dump(info, open(f'{out}/room.json', 'w')); open(f'{out}/index.html', 'w').write(VIEWER); open(f'{out}/blender_check.py', 'w').write(BLENDER)
    mean = lambda k: round(float(np.mean(res[k])), 3)
    rep = dict(metres_per_unit=round(s, 4), floor=round(sh['floor'], 3), ceiling=round(sh['ceiling'], 3), height_m=round(sh['height'], 2), axes_deg=round(sh['theta_deg'], 1),
               walls=len(sh['walls']), corners=sh['corners'], area_m2=round(sh['area_m2'], 1), room_extent_m=[round(float(sh['U'][-1] - sh['U'][0]), 2), round(float(sh['V'][-1] - sh['V'][0]), 2)],
               wall_list=[dict(axis=w['axis'], at=round(w['c'], 3), length=round(w['s1'] - w['s0'], 2), detected=w['detected'], openings=w['openings']) for w in sh['walls']],
               furniture=[dict(name=B['name'], kind=B['kind'], size_m=B['size_m'], photo_pct=B['seen_pct']) for B in blocks], left_to_wall_texture=skipped,
               textures={S['name']: S['seen_pct'] for S in surfs}, glb_mb=glb_mb,
               heldout=dict(views=len(real), size=[Wc, Hc], clay_psnr=mean('clay'), splat_psnr=mean('splat'), clay_shell_psnr=mean('clay_shell'),
                            splat_shell_psnr=mean('splat_shell'), clay_ssim=mean('clay_ssim'), splat_ssim=mean('splat_ssim')), minutes=round((time.time() - T0) / 60, 1))
    json.dump(rep, open(f'{out}/report.json', 'w'), indent=1, default=float); say('CLAY ROOM', json.dumps(rep, default=float))


if __name__ == '__main__':
    main(*sys.argv[1:])
