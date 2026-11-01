"""Object layer, second pass on the CPU (no GPU: owner 27 Oct, Kaggle hours used up). Takes a finished objects3d.py result
(splat_object.npy = object id per splat) and fixes what v2 got wrong. The lesson from Semantic Foam / Gaussian Grouping
(docs/KMYI_NOTES.md): object ids must agree in 3D, not only per frame.
  walls   big vertical planes (RANSAC on the opaque splats, >= 1.5 m wide and 1.2 m tall). The part of an object lying on such
          a plane is wall when it spans > 1.5 m (v2: 'couch' = sofa + coat + art + the wall between them).
  smooth  each opaque splat takes the clear majority id of its 12 nearest neighbours within 5 cm (room = 0 counts too), twice.
  split   each object is cut into its 3D-connected pieces; every big piece is its own object (v2: 'bed' = bed + dresser).
  merge   two objects are one only if their boxes overlap >= 50 % AND they look alike (CLIP cosine >= 0.85), never on a name.
  names   Qwen2.5-VL-3B on the CPU, told the measured size and whether the thing hangs on a wall; CLIP zero-shot over the
          size-plausible labels replaces its answer only when the answer is generic or impossible for that size / place.
  masks   without a GPU renderer: the object's opaque splat centres projected into a real training frame, hidden ones removed by
          a coarse depth buffer of all opaque splats, then closed into an outline.
Coordinates: decisions in tour metres (level matrix x metres per unit: y up, floor at 0); projection in the model's own frame.
Usage (from gsplat's examples dir, for datasets.colmap): python objects_refine.py <data dir> <model.ply> <splat_object.npy>
  <old objects.json> <level.json> <metres_per_unit> <out dir> [n_frames]"""
import json, os, sys, time, numpy as np, torch
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from objects3d import VOCAB, STRUCTURE_WORDS, GENERIC, UF, palette, say, T0

# max-dimension range (m) a label can have; outside 0.6x .. 1.5x of it the label is impossible for the measured object
SIZE = {'bed': (1.3, 2.6), 'sofa': (1.2, 3.2), 'couch': (1.2, 3.2), 'mattress': (1.2, 2.4), 'dresser': (0.7, 2.2), 'wardrobe': (0.8, 2.6),
        'dining table': (0.8, 2.5), 'desk': (0.7, 2.2), 'bathtub': (1.2, 2.0), 'bookshelf': (0.6, 2.5), 'tv stand': (0.8, 2.2),
        'armchair': (0.6, 1.2), 'nightstand': (0.35, 0.8), 'side table': (0.3, 0.9), 'end table': (0.3, 0.9), 'coffee table': (0.5, 1.5),
        'floor lamp': (1.0, 2.0), 'table lamp': (0.25, 0.8), 'wall lamp': (0.15, 0.7), 'lamp': (0.15, 2.0), 'lampshade': (0.2, 0.7),
        'pillow': (0.25, 0.9), 'cushion': (0.25, 0.8), 'blanket': (0.4, 2.5), 'remote control': (0.1, 0.3), 'telephone': (0.12, 0.35),
        'tissue box': (0.15, 0.35), 'cup': (0.06, 0.2), 'bottle': (0.1, 0.4), 'plastic bag': (0.15, 0.8), 'backpack': (0.3, 0.7),
        'handbag': (0.15, 0.6), 'suitcase': (0.35, 0.9), 'shoes': (0.15, 0.5), 'clock': (0.15, 0.6), 'vase': (0.1, 0.7),
        'books': (0.1, 0.6), 'trash can': (0.2, 0.6), 'laptop': (0.25, 0.5), 'kettle': (0.15, 0.35), 'tv': (0.5, 2.0),
        'picture frame': (0.15, 1.5), 'painting': (0.2, 2.0), 'wall art': (0.2, 2.0), 'mirror': (0.3, 2.2), 'air conditioner': (0.5, 1.4),
        'coat on a hanger': (0.4, 1.4), 'chair': (0.4, 1.1), 'stool': (0.3, 0.8), 'toilet': (0.4, 0.9), 'rug': (0.6, 4.0)}
FLOOR = {'cushion', 'pillow', 'blanket', 'bed', 'sofa', 'couch', 'armchair', 'chair', 'dresser', 'wardrobe', 'nightstand', 'side table', 'end table', 'coffee table',
         'trash can', 'suitcase', 'floor lamp', 'rug', 'mattress', 'bench', 'stool', 'ottoman', 'desk', 'tv stand', 'dining table', 'toilet'}


def key_of(label):
    """Longest SIZE / FLOOR key whose words all appear in the label ('grey fabric couch' -> 'couch')."""
    w = set(label.lower().replace('-', ' ').split())
    ks = [k for k in set(SIZE) | FLOOR if set(k.split()) <= w]
    return max(ks, key=len) if ks else None


def plausible(label, size, on_wall):
    k = key_of(label)
    if k is None: return True
    if on_wall and k in FLOOR: return False
    if k in SIZE:
        lo, hi = SIZE[k]; m = max(size)
        return 0.6 * lo <= m <= 1.5 * hi
    return True


def load(ply):
    from plyfile import PlyData
    data = PlyData.read(ply); v = data['vertex'].data
    X = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)
    op = 1 / (1 + np.exp(-np.asarray(v['opacity'], np.float64)))
    return data, v, X, op


def wall_planes(P, max_planes=10, thr=0.025):
    """Big vertical planes among the opaque splats (tour metres): (normal, d, in-plane axis, s-range, y-range)."""
    import open3d as o3d
    rest = np.arange(len(P)); planes = []; need = max(300, int(0.01 * len(P)))
    for _ in range(4 * max_planes):
        if len(rest) < need or len(planes) >= max_planes: break
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[rest]))
        (a, b, c, d), inl = pc.segment_plane(thr, 3, 2000); inl = np.asarray(inl)
        if len(inl) < need: break
        n = np.array([a, b, c]); n /= np.linalg.norm(n); idx = rest[inl]
        if abs(n[1]) < 0.15:
            t = np.cross(n, [0, 1, 0]); t /= np.linalg.norm(t); s = P[idx] @ t; y = P[idx, 1]
            sr, yr = np.percentile(s, [2, 98]), np.percentile(y, [2, 98])
            if sr[1] - sr[0] >= 1.5 and yr[1] - yr[0] >= 1.2: planes.append((n, d / np.linalg.norm([a, b, c]), t, sr, yr))
        rest = np.setdiff1d(rest, idx)
    return planes


def on_planes(P, planes, thr=0.03):
    on = np.zeros(len(P), bool)
    for n, d, t, sr, yr in planes:
        s = P @ t
        on |= (np.abs(P @ n + d) < thr) & (s > sr[0] - 0.1) & (s < sr[1] + 0.1) & (P[:, 1] > yr[0] - 0.1) & (P[:, 1] < yr[1] + 0.1)
    return on


def span(Q):
    """Largest horizontal extent (2-98 %) of points Q along their main horizontal direction."""
    H = Q[:, [0, 2]] - Q[:, [0, 2]].mean(0)
    if len(H) < 3: return 0.0
    ax = np.linalg.svd(H, full_matrices=False)[2][0]; s = H @ ax
    return float(np.percentile(s, 98) - np.percentile(s, 2))


def smooth(P, lab, k=12, r=0.05, rounds=2):
    """Gaussian-Grouping-style 3D regularisation: a splat takes the label most of its neighbours (within r) carry, when that
    label holds a clear majority (> half of the valid neighbours incl. itself)."""
    from scipy.spatial import cKDTree
    d, nb = cKDTree(P).query(P, k=k + 1, distance_upper_bound=r); ok = np.isfinite(d); nb = np.where(ok, nb, 0)
    for _ in range(rounds):
        L = np.where(ok, lab[nb], -1); labels = np.unique(lab)
        cnt = np.stack([(L == c).sum(1) for c in labels], 1); best = cnt.argmax(1); top = cnt[np.arange(len(P)), best]
        flip = (top > ok.sum(1) / 2) & (labels[best] != lab); lab = np.where(flip, labels[best], lab)
    return lab


def split(P, lab, min_pts=150):
    """Every object -> its 3D-connected pieces (DBSCAN, eps 6 cm or 2.5 x the typical spacing). Big pieces become objects
    (parent kept), small ones join the nearest big piece within 10 cm, else the room."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    out = np.zeros(len(P), int); parent = {}; nid = 0
    for k in np.unique(lab[lab > 0]):
        m = np.where(lab == k)[0]
        if len(m) < 30: continue
        Q = P[m]; nn = cKDTree(Q).query(Q, k=2)[0][:, 1]; eps = max(0.06, 2.5 * float(np.median(nn)))
        cl = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Q)).cluster_dbscan(eps=eps, min_points=6))
        if not (cl >= 0).any(): cl = np.zeros(len(m), int)
        sz = np.bincount(cl[cl >= 0]); big = [c for c in np.where(sz >= max(min_pts, 0.15 * sz.max()))[0]]
        if not big: big = [int(sz.argmax())] if sz.max() >= min_pts / 2 else []
        if not big: continue
        ids = {}
        for c in big: nid += 1; ids[c] = nid; parent[nid] = int(k); out[m[cl == c]] = nid
        small = ~np.isin(cl, big)
        if small.any():
            bigpts = np.isin(cl, big); d, j = cKDTree(Q[bigpts]).query(Q[small], k=1)
            near = d < 0.10; out[m[small][near]] = out[m[bigpts][j[near]]]
    return out, parent


class Frames:
    """Training frames (undistorted, long side 1024) with cameras; projection of points with a coarse depth buffer."""
    def __init__(self, data, X_all, nframes=40):
        from datasets.colmap import Parser, Dataset
        p = Parser(data, factor=1, normalize=True, test_every=8); cid = p.camera_ids[0]
        K0 = p.Ks_dict[cid].copy(); W0, H0 = p.imsize_dict[cid]; ds = Dataset(p, split='train')
        k = 1024 / max(W0, H0); self.W, self.H = round(W0 * k), round(H0 * k); self.K = K0.copy(); self.K[:2] *= k
        pick = np.linspace(0, len(ds) - 1, min(nframes, len(ds))).astype(int); self.fr = []
        for j in pick:
            d = ds[int(j)]; img = np.asarray(Image.fromarray(d['image'].numpy().astype(np.uint8)).resize((self.W, self.H), Image.LANCZOS))
            self.fr.append((int(ds.indices[int(j)]), img, d['camtoworld'].numpy().astype(np.float64)))
        self.X = X_all; self.zbuf = [self._zbuf(c2w) for _, _, c2w in self.fr]

    def project(self, X, c2w):
        w2c = np.linalg.inv(c2w); pc = X @ w2c[:3, :3].T + w2c[:3, 3]; z = pc[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = self.K[0, 0] * pc[:, 0] / z + self.K[0, 2]; v = self.K[1, 1] * pc[:, 1] / z + self.K[1, 2]
        ok = (z > 1e-3) & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        return u, v, z, ok

    def _zbuf(self, c2w, b=8):
        u, v, z, ok = self.project(self.X, c2w); zb = np.full(((self.H + b - 1) // b, (self.W + b - 1) // b), np.inf)
        np.minimum.at(zb, ((v[ok] // b).astype(int), (u[ok] // b).astype(int)), z[ok]); return zb

    def visible(self, X, fi, b=8):
        u, v, z, ok = self.project(X, self.fr[fi][2]); vis = ok.copy()
        vis[ok] = z[ok] <= self.zbuf[fi][(v[ok] // b).astype(int), (u[ok] // b).astype(int)] * 1.04 + 0.02
        return u, v, vis


def outline(fr, X, out_path):
    """Best frame (most of the object visible, not cut by the border, big enough) -> mask from its visible splat centres,
    red outline on the crop, plain crop and a background-greyed crop. Returns dict or None."""
    import cv2
    best = None
    for fi in range(len(fr.fr)):
        u, v, vis = fr.visible(X, fi)
        if vis.sum() < 20: continue
        x0, y0, x1, y1 = np.percentile(u[vis], 2), np.percentile(v[vis], 2), np.percentile(u[vis], 98), np.percentile(v[vis], 98)
        cut = x0 < 3 or y0 < 3 or x1 > fr.W - 3 or y1 > fr.H - 3
        sc = float(vis.mean()) * (0.4 if cut else 1.0) * min(1, (x1 - x0) * (y1 - y0) / (0.04 * fr.W * fr.H))
        if best is None or sc > best[0]: best = (sc, fi, u[vis], v[vis])
    if best is None: return None
    _, fi, u, v = best; img = fr.fr[fi][1]; mk = np.zeros((fr.H, fr.W), np.uint8)
    r = max(3, int(0.012 * fr.W))
    for a, b_ in zip(u.astype(int), v.astype(int)): cv2.circle(mk, (a, b_), r, 1, -1)
    mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((4 * r, 4 * r), np.uint8))
    cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cs: mk = cv2.drawContours(np.zeros_like(mk), cs, -1, 1, -1)
    ys, xs = np.nonzero(mk); x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    mx, my = 0.15 * (x1 - x0) + 10, 0.15 * (y1 - y0) + 10
    cb = (int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(fr.W, x1 + mx)), int(min(fr.H, y1 + my)))
    drawn = img.copy(); cv2.drawContours(drawn, cs, -1, (255, 0, 0), 3)
    grey = np.where(mk[..., None] > 0, img, (0.35 * img + 0.65 * 128).astype(np.uint8))
    crop = Image.fromarray(drawn).crop(cb); crop.save(out_path, quality=90)
    return dict(fi=fi, frame=fr.fr[fi][0], crop=crop, plain=Image.fromarray(img).crop(cb), masked=Image.fromarray(grey).crop(cb),
                mask=mk, mask_px=int(mk.sum()))


class Namer:
    def __init__(self, use_vlm=True, budget_s=2400):
        from transformers import CLIPModel, CLIPProcessor
        torch.set_num_threads(os.cpu_count() or 4)
        self.cm = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').eval(); self.cp = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        with torch.no_grad():
            tk = self.cp.tokenizer([f'a photo of a {v}' for v in VOCAB], padding=True, return_tensors='pt')
            tf = self.cm.text_projection(self.cm.text_model(**tk).pooler_output); self.tf = tf / tf.norm(dim=-1, keepdim=True)
        self.vl = None; self.t_end = time.time() + budget_s
        if use_vlm:
            try:
                from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
                try: self.vp = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', min_pixels=128 * 28 * 28, max_pixels=320 * 28 * 28)
                except Exception: self.vp = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')            # small crops anyway
                self.vl = Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', torch_dtype=torch.bfloat16).eval()
            except Exception as e: say('VLM not available, CLIP only:', str(e)[:200])

    def embed(self, im):
        with torch.no_grad():
            e = self.cm.visual_projection(self.cm.vision_model(pixel_values=self.cp(images=im, return_tensors='pt')['pixel_values']).pooler_output)
        return (e / e.norm(dim=-1, keepdim=True))[0]

    def ask(self, crop, size, on_wall):
        if self.vl is None or time.time() > self.t_end: return ''
        w, dp = sorted([size[0], size[2]], reverse=True)
        q = (f'The object outlined in red is about {w:.1f} m wide, {dp:.1f} m deep and {size[1]:.1f} m tall'
             + (', fixed on the wall' if on_wall else '') + '. What is it? Answer with 1 to 4 words, only the object name.')
        msg = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': q}]}]
        crop = crop.copy(); crop.thumbnail((448, 448))
        text = self.vp.apply_chat_template(msg, tokenize=False, add_generation_prompt=True); t = time.time()
        with torch.no_grad():
            enc = self.vp(text=[text], images=[crop], return_tensors='pt')
            ans = self.vp.batch_decode(self.vl.generate(**enc, max_new_tokens=10, do_sample=False)[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        say(f'  VLM {time.time() - t:.0f}s')
        return ans.strip().strip('.').lower()[:40]

    def name(self, o, view):
        on_wall = o['on_wall']; size = o['size_m']
        e_plain = self.embed(view['plain']); e_mask = self.embed(view['masked'])
        pz = (100 * self.tf @ e_mask).softmax(0).numpy(); ok = np.array([plausible(v, size, on_wall) for v in VOCAB])
        pp = pz * ok; pp = pp / pp.sum() if pp.sum() > 0 else pz; j = int(pp.argmax()); c, pc_ = VOCAB[j], float(pp[j])
        ans = self.ask(view['crop'], size, on_wall)
        good = bool(ans) and ans not in GENERIC and len(ans.split()) <= 5
        if good and plausible(ans, size, on_wall): label, why = ans, 'vlm'
        elif pc_ >= 0.25: label, why = c, 'clip (vlm ' + ('generic' if not good else 'impossible for its size/place') + ')'
        elif good: label, why = ans, 'vlm (clip unsure)'
        else: label, why = None, 'unknown'
        o.update(vlm=ans, clip_label=c, clip_p=round(pc_, 3), label_from=why, embedding=[round(float(x), 4) for x in e_plain])
        return label


def main(data, ply, sid_path, old_json, level_json, mpu, out, nframes=40):
    mpu = float(mpu); os.makedirs(f'{out}/crops', exist_ok=True)
    plydata, v, X, op = load(ply); allid0 = np.load(sid_path).astype(int); N = len(X)
    L = np.array(json.load(open(level_json))['matrix']); M = (X @ L[:3, :3].T + L[:3, 3]) * mpu          # tour metres
    old = {o['id']: o for o in json.load(open(old_json))['objects']}
    solid = np.where(op > 0.3)[0]; P = M[solid]; lab = allid0[solid].copy()
    say(N, 'splats,', len(solid), 'opaque;', len(np.unique(lab[lab > 0])), 'v2 objects')
    planes = wall_planes(P); on = on_planes(P, planes); say(len(planes), 'big wall planes;', int(on.sum()), 'opaque splats on them')
    walls = {}
    for k in np.unique(lab[lab > 0]):                                                       # wall bits that came along
        mo = (lab == k) & on
        if mo.sum() >= 50 and span(P[mo]) > 1.5: lab[mo] = 0; walls[int(k)] = int(mo.sum())
    say('wall parts removed from v2 objects:', {f"{k} {old.get(k, {}).get('label', '')}": n for k, n in walls.items()})
    lab = smooth(P, lab); lab, parent = split(P, lab)
    say(len(parent), 'pieces after 3D smoothing + split:', {k: f"{p} {old.get(p, {}).get('label', '')}" for k, p in parent.items()})
    fr = Frames(data, X[solid], int(nframes)); nm = Namer(os.environ.get('OBJ_VLM', '1') == '1')
    objs = []
    for k, pk in parent.items():
        m = np.where(lab == k)[0]; Q = P[m]
        lo, hi = np.percentile(Q, 2, 0), np.percentile(Q, 98, 0); size = (hi - lo).round(2).tolist()
        d_wall = min([float(np.median(np.abs(Q @ n + d))) for n, d, *_ in planes] or [9.0])
        o = dict(piece=int(k), parent=int(pk), parent_label=old.get(pk, {}).get('label', ''), members=m, box_m=[lo.round(3).tolist(), hi.round(3).tolist()],
                 centre_m=((lo + hi) / 2).round(3).tolist(), size_m=size, on_wall=bool(lo[1] > 0.5 and d_wall < 0.15))
        view = outline(fr, X[solid][m], f'{out}/crops/p{k}.jpg')
        if view is None: o.update(crop=None, label=None, label_from='never seen well'); objs.append(o); continue
        o.update(crop=f'crops/p{k}.jpg', frame=view['frame'], mask_px=view['mask_px'], view=view)
        o['label'] = nm.name(o, view)
        say(f"piece {k} (v2 {pk} '{o['parent_label']}') size {size}{' wall' if o['on_wall'] else ''}: vlm '{o['vlm']}', "
            f"clip '{o['clip_label']}' {o['clip_p']:.2f} -> '{o['label']}' [{o['label_from']}]")
        objs.append(o)
    # merge only on look + overlap
    vol = lambda a, b: float(np.prod(np.clip(np.asarray(b) - np.asarray(a), 0, None)))
    uf = UF(len(objs))
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = objs[i], objs[j]
            if not (a.get('embedding') and b.get('embedding')): continue
            inter = vol(np.maximum(a['box_m'][0], b['box_m'][0]), np.minimum(a['box_m'][1], b['box_m'][1])); small = min(vol(*a['box_m']), vol(*b['box_m']))
            if small > 0 and inter >= 0.5 * small and float(np.dot(a['embedding'], b['embedding'])) >= 0.85: uf.union(i, j)
    groups = {}
    for i in range(len(objs)): groups.setdefault(uf.find(i), []).append(i)
    merged = []
    for g in groups.values():
        g = sorted(g, key=lambda i: -len(objs[i]['members'])); o = objs[g[0]]
        if len(g) > 1:
            o['members'] = np.unique(np.concatenate([objs[i]['members'] for i in g])); o['merged_pieces'] = [objs[i]['piece'] for i in g]
            Q = P[o['members']]; lo, hi = np.percentile(Q, 2, 0), np.percentile(Q, 98, 0)
            o.update(box_m=[lo.round(3).tolist(), hi.round(3).tolist()], centre_m=((lo + hi) / 2).round(3).tolist(), size_m=(hi - lo).round(2).tolist())
        merged.append(o)
    say(len(objs) - len(merged), 'merged on look + overlap')
    keep, to_room = [], []
    for o in merged:
        w = set((o.get('label') or '').replace('-', ' ').split())
        (to_room if w & STRUCTURE_WORDS else keep).append(o)
    # ids, per-splat ids (faint splats: nearest opaque splat within 3 cm), outputs
    from scipy.spatial import cKDTree
    sid = np.zeros(len(solid), np.uint16)
    for i, o in enumerate(keep):
        o['id'] = i + 1; sid[o['members']] = o['id']
        if not o.get('label'): o['label'] = f'object {i + 1}'
    allid = np.zeros(N, np.uint16); allid[solid] = sid; faint = np.setdiff1d(np.arange(N), solid)
    dd, nn = cKDTree(P).query(M[faint], k=1, workers=-1); near = dd < 0.03; allid[faint[near]] = sid[nn[near]]
    np.save(f'{out}/splat_object.npy', allid)
    for o in keep:
        if o.get('crop'): os.replace(f"{out}/{o['crop']}", f"{out}/crops/{o['id']}.jpg"); o['crop'] = f"crops/{o['id']}.jpg"
    sheet(keep, out); views(keep, fr, out); tinted(plydata, v, allid, len(keep), out)
    drop = ('members', 'view'); res = [{k: x for k, x in o.items() if k not in drop} | dict(splats=int((allid == o['id']).sum())) for o in keep]
    json.dump(dict(objects=res, to_room=[dict(label=o.get('label'), parent=o['parent'], parent_label=o['parent_label']) for o in to_room],
                   wall_parts_removed=walls, wall_planes=len(planes), merged=len(objs) - len(merged), from_v2=os.path.abspath(old_json),
                   metres_per_unit=mpu, minutes=round((time.time() - T0) / 60, 1)),
              open(f'{out}/objects.json', 'w'), indent=1)
    say('OBJECTS DONE', json.dumps([(o['id'], o['label'], o['parent_label'], o['size_m']) for o in res]))


def sheet(objs, out):
    tiles = []
    for o in objs:
        if not o.get('crop'): continue
        t = Image.open(f"{out}/{o['crop']}").convert('RGB'); t.thumbnail((220, 220)); c = Image.new('RGB', (230, 292), (24, 24, 24)); c.paste(t, (5, 5))
        d = ImageDraw.Draw(c); w, dp = sorted([o['size_m'][0], o['size_m'][2]], reverse=True)
        d.text((6, 230), f"{o['id']} {o['label'][:28]}", fill=(255, 255, 255))
        d.text((6, 248), f"{w:.1f} x {dp:.1f} m, {o['size_m'][1]:.1f} m tall{' (wall)' if o['on_wall'] else ''}", fill=(170, 170, 170))
        d.text((6, 266), f"vlm {o.get('vlm', '')[:14]} | clip {o.get('clip_label', '')[:12]} | v2 {o['parent_label'][:10]}", fill=(130, 130, 130))
        tiles.append(c)
    if tiles:
        cols = 6; s = Image.new('RGB', (cols * 230, ((len(tiles) + cols - 1) // cols) * 292))
        for n, t in enumerate(tiles): s.paste(t, ((n % cols) * 230, (n // cols) * 292))
        s.save(f'{out}/objects_sheet.jpg', quality=88)


def views(objs, fr, out):
    """CPU stand-in for the rendered views: the 3 training frames showing most objects, each object's outline filled with its
    colour and named, beside the plain frame."""
    import cv2
    from PIL import ImageFont
    try: font = ImageFont.load_default(size=20)
    except Exception: font = ImageFont.load_default()
    pal = (palette(len(objs)) * 255).astype(np.uint8); rows = []
    seen = []
    for fi in range(len(fr.fr)):
        seen.append(sum(int(fr.visible(fr.X[o['members']], fi)[2].sum() >= 30) for o in objs))
    for fi in np.argsort(seen)[::-1][:3]:
        img = fr.fr[fi][1]; ov = img.copy(); tags = []
        for i, o in enumerate(objs):
            u, v, vis = fr.visible(fr.X[o['members']], fi)
            if vis.sum() < 30: continue
            mk = np.zeros((fr.H, fr.W), np.uint8); r = max(3, int(0.012 * fr.W))
            for a, b in zip(u[vis].astype(int), v[vis].astype(int)): cv2.circle(mk, (a, b), r, 1, -1)
            mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((4 * r, 4 * r), np.uint8)) > 0
            ov[mk] = (0.45 * ov[mk] + 0.55 * pal[i]).astype(np.uint8); tags.append((float(np.median(u[vis])), float(np.median(v[vis])), f"{o['id']} {o['label']}"[:26], tuple(int(x) for x in pal[i])))
        im = Image.fromarray(ov); dr = ImageDraw.Draw(im)
        for x, y, t, c in tags:
            bb = dr.textbbox((x, y), t, font=font); dr.rectangle((bb[0] - 3, bb[1] - 2, bb[2] + 3, bb[3] + 2), fill=(0, 0, 0)); dr.text((x, y), t, fill=c, font=font)
        rows.append((Image.fromarray(img), im))
    if rows:
        s = Image.new('RGB', (2 * fr.W, len(rows) * fr.H))
        for r, (a, b) in enumerate(rows): s.paste(a, (0, r * fr.H)); s.paste(b, (fr.W, r * fr.H))
        s.save(f'{out}/objects_views.jpg', quality=85)


def tinted(plydata, v, allid, n, out):
    """model.ply with each object's splats pulled toward its own colour (view-dependent colour off), for the check tour."""
    from plyfile import PlyData, PlyElement
    t = v.copy(); pal = np.vstack([np.zeros((1, 3), np.float32), palette(n)]); on = allid > 0
    for c in range(3): t[f'f_dc_{c}'][on] = 0.45 * t[f'f_dc_{c}'][on] + 0.55 * (pal[allid[on], c] - 0.5) / 0.28209479
    for f in [x for x in t.dtype.names if x.startswith('f_rest_')]: t[f][on] = 0
    os.makedirs(f'{out}/full', exist_ok=True); PlyData([PlyElement.describe(t, 'vertex')]).write(f'{out}/full/model.ply')


if __name__ == '__main__':
    main(*sys.argv[1:])
