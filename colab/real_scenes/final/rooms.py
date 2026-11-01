"""Split a whole-home capture into rooms from its SfM model alone (CPU; per-room plan of 30 Oct, docs/HANDOFF_V3.md):
  1 fine pieces = watershed of the free space between everything standing (walls and furniture)
  2 neighbouring pieces merged while their cameras see the same 3D points (v1 split by walls only: one room became 3; v2 =
    co-visibility communities alone: mixed hall and bathroom; v3 = full-height walls only: plain white walls have too few points)
  3 each camera -> the room it stands in; 'extra' cameras = other rooms' cameras that see > 30 % of their points in it
  (covis_rooms: the co-visibility communities, kept for checks)
Writes rooms.json (rooms with outline in SfM coordinates, area, cameras, doors) and rooms.png (coloured plan, path, doors).
  python rooms.py <sparse/0 dir> <out dir>"""
import json, os, sys
import numpy as np

CELL, DOOR_MAX, CAM_H = 0.05, 1.3, 1.45


def load(sparse):
    from pycolmap import SceneManager                          # the fork gsplat uses (rmbrualla/pycolmap)
    m = SceneManager(sparse); m.load_cameras(); m.load_images(); m.load_points3D()
    ks = sorted(m.images, key=lambda k: m.images[k].name)
    R = np.stack([m.images[k].R() for k in ks]); t = np.stack([m.images[k].tvec for k in ks])
    C = -np.einsum('nji,nj->ni', R, t)                         # camera centres; viewing direction = R[2], camera y (down) = R[1]
    row = {pid: i for i, pid in enumerate(m.point3D_ids)}    # point id -> row of m.points3D
    obs = [np.array([row[p] for p in m.images[k].point3D_ids if p in row], np.int64) for k in ks]
    return [m.images[k].name for k in ks], C, R, np.asarray(m.points3D), obs


def covis_rooms(C, obs, npts, min_cams=40, resolution=1.0, seed=0):
    """Rooms as communities of the co-visibility graph: two cameras are linked by how many 3D points both see; doorways cut
    most of those links, the walk order does not matter (a room filmed twice is still one community). Louvain, then pieces with
    fewer than min_cams cameras are merged into the room they share most points with. Returns a room id per camera."""
    import networkx as nx
    from scipy import sparse
    rows = np.concatenate([np.full(len(o), i) for i, o in enumerate(obs)]); cols = np.concatenate(obs)
    A = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(obs), npts))
    S = (A @ A.T).tocoo(); n = np.asarray(A.sum(1)).ravel() + 1
    g = nx.Graph(); g.add_nodes_from(range(len(obs)))
    for i, j, w in zip(S.row, S.col, S.data):
        if i < j and w >= 15: g.add_edge(int(i), int(j), weight=float(w) / np.sqrt(n[i] * n[j]))
    lab = np.zeros(len(obs), int)
    for k, cset in enumerate(nx.community.louvain_communities(g, weight='weight', resolution=resolution, seed=seed)): lab[list(cset)] = k
    while True:                                                 # small pieces -> the neighbour they share most with
        cnt = np.bincount(lab); small = [k for k in range(len(cnt)) if 0 < cnt[k] < min_cams]
        if not small: break
        k = min(small, key=lambda x: cnt[x]); mem = np.nonzero(lab == k)[0]; share = np.zeros(len(cnt))
        for i in mem:
            for j, d in g[i].items():
                if lab[j] != k: share[lab[j]] += d['weight']
        if share.max() <= 0: share[np.argmax(np.where(np.arange(len(cnt)) != k, cnt, -1))] = 1
        lab[mem] = int(share.argmax())
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def split(names, C, R, P, obs, merge_thr=0.2):   # doors measured 0.00-0.15, pieces of one room 0.2-0.35 (30 Oct)
    """Fine split from everything standing (walls and furniture: watershed of free space), then neighbouring pieces are merged
    while their cameras see the same 3D points (co-visibility between them / within them > merge_thr). A sofa splits a room but
    both sides see the same walls; a doorway cuts what the two sides see."""
    from scipy import ndimage
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from skimage.measure import find_contours
    up = -np.median(R[:, 1, :], 0); up /= np.linalg.norm(up)
    e1 = np.cross(up, [1, 0, 0] if abs(up[0]) < 0.9 else [0, 1, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    hp, hc = P @ up, C @ up
    floor = np.percentile(hp[hp < np.median(hc)], 3); scale = CAM_H / max(np.median(hc) - floor, 1e-6)   # metres per SfM unit
    ceil = np.percentile(hp[hp > np.median(hc)], 97); Hm = (ceil - floor) * scale
    seenp = np.zeros(len(P), bool); seenp[np.concatenate(obs)] = True
    to2 = lambda X: np.c_[X @ e1, X @ e2] * scale
    p2, c2 = to2(P), to2(C); hm = (hp - floor) * scale; inb = seenp & (hm > -0.2) & (hm < Hm + 0.2)
    lo = np.percentile(np.r_[p2[inb], c2], 0.5, 0) - 1; hi = np.percentile(np.r_[p2[inb], c2], 99.5, 0) + 1
    shape = np.ceil((hi - lo) / CELL).astype(int); cell = lambda q: np.clip(((q - lo) / CELL).astype(int), 0, shape - 1)
    band = inb & (hm > 0.3) & (hm < Hm - 0.3)                                   # fine split: anything standing (walls AND furniture)
    wall = np.zeros(shape, np.int32); np.add.at(wall, tuple(cell(p2[band]).T), 1)
    wallm = ndimage.binary_closing(wall >= (max(3, np.percentile(wall[wall > 0], 60)) if (wall > 0).any() else 3), iterations=1)
    pts = np.zeros(shape, bool); pts[tuple(cell(p2[inb]).T)] = True
    walked = np.zeros(shape, bool); walked[tuple(cell(c2).T)] = True
    foot = ndimage.binary_fill_holes(ndimage.binary_closing(pts | walked, iterations=8))
    free = foot & ~ndimage.binary_dilation(wallm, iterations=1)
    free = ndimage.binary_opening(free, iterations=2)
    lab, _ = ndimage.label(free); free &= np.isin(lab, np.unique(lab[tuple(cell(c2).T)])) & (lab > 0)
    dist = ndimage.distance_transform_edt(free) * CELL
    pk = peak_local_max(dist, min_distance=int(1.0 / CELL), threshold_abs=0.45, labels=free.astype(int))
    mk = np.zeros(shape, np.int32)
    for i, (a_, b_) in enumerate(pk): mk[a_, b_] = i + 1
    reg = watershed(-dist, mk, mask=free)
    # merge: pieces whose cameras see the same things (a sofa splits a room; a doorway cuts what both sides see)
    from scipy import sparse
    rows = np.concatenate([np.full(len(o), i) for i, o in enumerate(obs)])
    Am = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, np.concatenate(obs))), shape=(len(obs), len(P)))
    nn = np.asarray(Am.sum(1)).ravel() + 1; S = (Am @ Am.T).toarray() / np.sqrt(np.outer(nn, nn))
    ci = cell(c2)
    def affinity(a_, b_):
        ia, ib = np.nonzero(reg[ci[:, 0], ci[:, 1]] == a_)[0], np.nonzero(reg[ci[:, 0], ci[:, 1]] == b_)[0]
        if not len(ia) or not len(ib): return None
        inter = S[np.ix_(ia, ib)].mean(); ia_ = S[np.ix_(ia, ia)].mean(); ib_ = S[np.ix_(ib, ib)].mean()
        return inter / np.sqrt(ia_ * ib_ + 1e-12)
    while True:
        best, bv = None, merge_thr
        for a_ in [x for x in np.unique(reg) if x]:
            ma = reg == a_; ring = ndimage.binary_dilation(ma) & ~ma & (reg > 0)
            for b_ in np.unique(reg[ring]):
                af = affinity(a_, b_)
                if af is None: af = 1.0 if (ma.sum() < (reg == b_).sum()) else 0.0           # a piece nobody stood in: absorb
                if af > bv: best, bv = (a_, b_), af
        if not best: break
        reg[reg == best[0]] = best[1]
    ci = cell(c2); camroom = reg[ci[:, 0], ci[:, 1]]
    for r in [x for x in np.unique(reg) if x and not (camroom == x).any()]:   # nobody stood there: join the widest neighbour
        ma = reg == r; nb = reg[ndimage.binary_dilation(ma) & ~ma & (reg > 0)]
        if len(nb): reg[ma] = np.bincount(nb).argmax()
    ids = [x for x in np.unique(reg) if x]; rm = np.full(reg.max() + 1, -1); rm[ids] = np.arange(len(ids)); reg = rm[reg]
    if (reg[ci[:, 0], ci[:, 1]] < 0).any():                    # cameras on a wall line / outside free space: nearest room cell
        ix = ndimage.distance_transform_edt(reg < 0, return_indices=True)[1]
    camroom = np.array([reg[c[0], c[1]] if reg[c[0], c[1]] >= 0 else reg[ix[0][c[0], c[1]], ix[1][c[0], c[1]]] for c in ci])
    pc = cell(p2); prow = np.where(inb, reg[pc[:, 0], pc[:, 1]], -1)            # each point -> the room it lies in (-1: wall/outside)
    rooms = []
    for i in range(len(ids)):
        m = reg == i; cs = max(find_contours(np.pad(m, 1).astype(float), 0.5), key=len) - 1; poly = (cs[::max(1, len(cs) // 200)] * CELL + lo) / scale
        doors = []
        for j in range(len(ids)):
            if j == i: continue
            sh = ndimage.binary_dilation(m, iterations=2) & (reg == j)
            if sh.sum():
                ia, ib = np.nonzero(camroom == i)[0], np.nonzero(camroom == j)[0]
                cv = S[np.ix_(ia, ib)].mean() / np.sqrt(S[np.ix_(ia, ia)].mean() * S[np.ix_(ib, ib)].mean() + 1e-12) if len(ia) and len(ib) else 0
                cen = np.argwhere(sh).mean(0) * CELL + lo; doors.append(dict(to=j, centre=(cen / scale).tolist(), width_m=round(float(sh.sum() * CELL / 2), 2), covis=round(float(cv), 3)))
        own = np.nonzero(camroom == i)[0]; mine = prow == i
        share = np.array([mine[o].mean() if len(o) else 0 for o in obs])        # other rooms' cameras that see a lot of this room
        extra = np.nonzero((camroom != i) & (share > 0.3))[0]
        rooms.append(dict(id=i, area_m2=round(float(m.sum() * CELL ** 2), 1), cameras=own.tolist(), extra=extra.tolist(), points=int(mine.sum()),
                          outline=poly.tolist(), doors=doors, first_frame=names[own.min()] if len(own) else None))
    order = sorted(range(len(rooms)), key=lambda i: min(rooms[i]['cameras'] or [1 << 30]))
    meta = dict(metres_per_unit=1 / scale, up=up.tolist(), e1=e1.tolist(), e2=e2.tolist(), origin_m=lo.tolist(), cell_m=CELL,
                floor=float(floor), ceiling=float(ceil), height_m=round(float(Hm), 2), cameras=len(C), points=len(P), point_room=prow.tolist())
    return meta, rooms, order, (reg, wallm, c2, lo)


def draw(path, reg, wallm, c2, lo, rooms):
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(3); pal = (rng.random((max(1, reg.max() + 1), 3)) * 160 + 70).astype(np.uint8)
    img = np.full(reg.shape + (3,), 30, np.uint8); img[reg >= 0] = pal[reg[reg >= 0]]; img[wallm] = 235
    im = Image.fromarray(np.ascontiguousarray(np.transpose(img, (1, 0, 2))[::-1])); dr = ImageDraw.Draw(im); H = im.size[1]
    xy = lambda q: ((q[0] - lo[0]) / CELL, H - 1 - (q[1] - lo[1]) / CELL)
    dr.line([xy(p) for p in c2], fill=(0, 0, 0), width=2)
    for r in rooms:
        cen = np.argwhere(reg == r['id']).mean(0) * CELL + lo
        dr.text(xy(cen), f"room {r['id']}\n{r['area_m2']} m2\n{len(r['cameras'])} cams", fill=(0, 0, 0))
    k = max(1, 1200 // max(im.size)); im = im.resize((im.size[0] * k, im.size[1] * k), Image.NEAREST) if k > 1 else im
    im.save(path)


if __name__ == '__main__':
    sp, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    names, C, R, P, obs = load(sp); meta, rooms, order, (reg, wallm, c2, lo) = split(names, C, R, P, obs)
    json.dump(dict(meta, order=order, rooms=rooms), open(f'{out}/rooms.json', 'w'))
    draw(f'{out}/rooms.png', reg, wallm, c2, lo, rooms)
    print(len(rooms), 'rooms:', [(r['id'], r['area_m2'], len(r['cameras']), len(r['extra']), [d['to'] for d in r['doors']]) for r in rooms], flush=True)
