"""v3 geometry: a solid surface mesh of the room (the 'mould'), in the same coordinates as the photoreal splat model.
Splats are millions of soft blobs tuned to reproduce the photos: photoreal near the filmed path but no surfaces. Here a 2DGS model
(flat surfels, normal + depth-distortion losses: gsplat's simple_trainer_2dgs.py) renders a median depth map for every camera;
the depth maps are fused into a truncated signed distance volume (TSDF, GPU), the zero surface is extracted (marching cubes),
small loose pieces are dropped, and each vertex is coloured from the REAL photos that see it (not invented colour).
With the full model's level.json the mesh is levelled exactly like full/model.splat (floor y = 0), so both line up.
Writes mesh.ply (vertex colours), mesh.glb, mesh_sheet.jpg (held-out photo | mesh; dollhouse from outside and from the top),
mesh.json. Units are the model's (SfM has no metric scale).
Usage: python mesh_tsdf.py <examples dir> <data dir> <result dir> <out dir> [level.json] [2dgs|3dgs] [voxels on the long side=512]"""
import glob, json, os, sys, numpy as np, torch

def fuse(frames, lo, hi, R, vox, trunc=4, device='cuda', chunk=1 << 22):
    """TSDF fusion. frames: list of (depth [H,W] tensor, 0 = no data; K; c2w; rgb [H,W,3] uint8 tensor) on `device`.
    The grid lives in the rotated frame whose axes are the rows of R (orthonormal), box lo..hi, voxel size vox.
    Returns tsdf, weight, colour as numpy arrays of shape dims (+3)."""
    dims = np.ceil((hi - lo) / vox).astype(int); n = int(dims.prod()); tr = trunc * vox
    tsdf = torch.ones(n, device=device); wt = torch.zeros(n, device=device); col = torch.zeros(n, 3, device=device); cw = torch.zeros(n, device=device)
    Rt, lo_t = torch.tensor(R, dtype=torch.float32, device=device), torch.tensor(lo, dtype=torch.float32, device=device)
    cams = [(d, torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device), torch.tensor(K, dtype=torch.float32, device=device), im)
            for d, K, c2w, im in frames]
    for a in range(0, n, chunk):
        i = torch.arange(a, min(n, a + chunk), device=device)
        ijk = torch.stack([i // (dims[1] * dims[2]), (i // dims[2]) % dims[1], i % dims[2]], 1).float()
        p = (lo_t + (ijk + 0.5) * vox) @ Rt                                               # voxel centres, world
        for d, w2c, K, im in cams:
            pc = p @ w2c[:3, :3].T + w2c[:3, 3]; z = pc[:, 2]; H, W = d.shape
            u = torch.round(pc[:, 0] / z.clamp(min=1e-6) * K[0, 0] + K[0, 2]).long(); v = torch.round(pc[:, 1] / z.clamp(min=1e-6) * K[1, 1] + K[1, 2]).long()
            ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not ok.any(): continue
            k = ok.nonzero()[:, 0]; dd = d[v[k], u[k]]; sdf = dd - z[k]
            m = (dd > 0) & (sdf > -tr); k, sdf = k[m], sdf[m]
            if not len(k): continue
            g = a + k; s = (sdf / tr).clamp(max=1.0); w0 = wt[g]
            tsdf[g] = (tsdf[g] * w0 + s) / (w0 + 1); wt[g] = w0 + 1
            if im is not None:
                near = sdf.abs() < tr; gn = g[near]                                        # colour only from close to the surface
                col[gn] += im[v[k][near], u[k][near]].float(); cw[gn] += 1
    col = col / cw.clamp(min=1)[:, None]
    return tsdf.view(*dims).cpu().numpy(), wt.view(*dims).cpu().numpy(), col.view(*dims, 3).cpu().numpy()

def extract(tsdf, wt, col, lo, R, vox, cams_world, min_faces=500):
    """Marching cubes on observed voxels -> (verts world, faces, colours uint8), loose bits dropped, faces turned to the cameras."""
    from skimage.measure import marching_cubes
    import trimesh
    v, f, _, _ = marching_cubes(tsdf, level=0.0, spacing=(vox,) * 3, mask=wt > 0)
    ijk = np.clip(np.round(v / vox - 0.5).astype(int), 0, np.array(tsdf.shape) - 1)
    c = col[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    m = trimesh.Trimesh((lo + v + 0.5 * vox) @ R, f, vertex_colors=np.clip(c, 0, 255).astype(np.uint8), process=False)
    parts = m.split(only_watertight=False)
    m = trimesh.util.concatenate([p for p in parts if len(p.faces) >= min_faces]) if len(parts) > 1 else m
    fc = m.triangles_center; s = np.random.default_rng(0).choice(len(fc), min(20000, len(fc)), replace=False)
    near = cams_world[np.argmin(((fc[s, None] - cams_world[None]) ** 2).sum(-1), 1)]
    if (np.einsum('ij,ij->i', m.face_normals[s], near - fc[s]) > 0).mean() < 0.5: m.invert()   # normals face the free space
    return m

def splat_points(P, C, c2w, K, W, H, r=1):
    """Tiny z-buffered point renderer (for the QA sheet): P [N,3], C [N,3] uint8 on the GPU -> image [H,W,3] float."""
    w2c = torch.linalg.inv(torch.tensor(c2w, dtype=torch.float32, device=P.device)); Kt = torch.tensor(K, dtype=torch.float32, device=P.device)
    pc = P @ w2c[:3, :3].T + w2c[:3, 3]; z = pc[:, 2]; ok = z > 1e-3
    u0 = (pc[ok, 0] / z[ok] * Kt[0, 0] + Kt[0, 2]).round().long(); v0 = (pc[ok, 1] / z[ok] * Kt[1, 1] + Kt[1, 2]).round().long()
    zz, cc = z[ok], C[ok]; zb = torch.full((H * W,), float('inf'), device=P.device); img = torch.zeros(H * W, 3, device=P.device)
    for du in range(-r, r + 1):
        for dv in range(-r, r + 1):
            u, v = u0 + du, v0 + dv; m = (u >= 0) & (u < W) & (v >= 0) & (v < H); pix = v[m] * W + u[m]
            zb.scatter_reduce_(0, pix, zz[m], reduce='amin')
    for du in range(-r, r + 1):
        for dv in range(-r, r + 1):
            u, v = u0 + du, v0 + dv; m = (u >= 0) & (u < W) & (v >= 0) & (v < H); pix = v[m] * W + u[m]
            win = zz[m] <= zb[pix]; img[pix[win]] = cc[m][win].float() / 255
    return img.view(H, W, 3).cpu().numpy()

def main(examples, data, result, out, level=None, kind='2dgs', nvox=512):
    sys.path.insert(0, examples); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from datasets.colmap import Parser, Dataset
    from v3_qa import camera_up, dollhouse, look_at
    from PIL import Image
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; os.makedirs(out, exist_ok=True)
    p = Parser(data, factor=1, normalize=True, test_every=8); C = p.camtoworlds; up = camera_up(C)
    pat = f'{result}/ckpts/ckpt_*.pt'; ck = sorted(glob.glob(pat), key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))[-1]
    s = torch.load(ck, map_location=dev)['splats']; colors = torch.cat([s['sh0'], s['shN']], 1); shd = int(round(np.sqrt(colors.shape[1]))) - 1
    def depth(c2w, K, W, H):
        vm = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=dev)[None]; Kt = torch.tensor(K, dtype=torch.float32, device=dev)[None]
        args = (s['means'], s['quats'], torch.exp(s['scales']), torch.sigmoid(s['opacities']), colors, vm, Kt, W, H)
        if kind == '2dgs':
            from gsplat.rendering import rasterization_2dgs
            _, alpha, _, _, _, med, _ = rasterization_2dgs(*args, sh_degree=shd, render_mode='RGB+ED', depth_mode='median'); d = med[0, ..., 0]
        else:
            from gsplat.rendering import rasterization
            rc, alpha, _ = rasterization(*args, sh_degree=shd, render_mode='RGB+ED'); d = rc[0, ..., 3]
        return torch.where(alpha[0, ..., 0] > 0.5, d, torch.zeros_like(d))
    frames = []
    for split in ('train', 'val'):                                                     # every camera, real (undistorted) photos for colour
        ds = Dataset(p, split=split)
        for k in range(len(ds)):
            x = ds[k]; im = x['image'].to(dev).to(torch.uint8); K = x['K'].numpy(); i = ds.indices[k]
            with torch.no_grad(): frames.append((depth(C[i], K, im.shape[1], im.shape[0]), K, C[i], im))
    # grid: the room box of v3_qa.dollhouse without the ceiling cut (+ margin), in the frame (main horizontal axes, up)
    pts = p.points; h = pts @ up; flat = pts - np.outer(h, up); m0 = flat.mean(0); ax = np.linalg.svd(flat - m0, full_matrices=False)[2][:2]
    R = np.stack([ax[0], ax[1], up]); q = pts @ R.T; lo, hi = np.percentile(q, 3, 0), np.percentile(q, 97, 0); pad = 0.12 * (hi - lo)   # not the view out of the window; lo, hi = lo - pad, hi + pad
    vox = float((hi - lo).max() / nvox); print('TSDF grid', np.ceil((hi - lo) / vox).astype(int), 'voxel', round(vox, 5), flush=True)
    tsdf, wt, col = fuse(frames, lo, hi, R, vox, device=dev)
    mesh = extract(tsdf, wt, col, lo, R, vox, C[:, :3, 3]); print('mesh', len(mesh.vertices), 'vertices', len(mesh.faces), 'faces', flush=True)
    # QA sheet: 2 held-out photos | mesh render; dollhouse of the mesh from outside and from the top
    P = torch.tensor(mesh.vertices, dtype=torch.float32, device=dev); Cv = torch.tensor(mesh.visual.vertex_colors[:, :3], device=dev)
    vs = Dataset(p, split='val'); tiles = []
    for k in np.linspace(0, len(vs) - 1, min(2, len(vs))).astype(int):
        x = vs[k]; gt = x['image'].numpy() / 255; H, W = gt.shape[:2]
        tiles.append(np.concatenate([gt, splat_points(P, Cv, C[vs.indices[k]], x['K'].numpy(), W, H, r=2)], 1))
    keep, ctr, span = dollhouse(mesh.vertices, pts, C, up); Pk, Ck = P[torch.tensor(keep, device=dev)], Cv[torch.tensor(keep, device=dev)]
    fwd = C[:, :3, 2].mean(0); fwd -= (fwd @ up) * up; fwd /= np.linalg.norm(fwd) + 1e-9
    H, W = tiles[0].shape[0], tiles[0].shape[1] // 2; Ko = np.array([[0.6 * max(W, H), 0, W / 2], [0, 0.6 * max(W, H), H / 2], [0, 0, 1]])
    for eye in (ctr - 1.6 * span * fwd + 0.9 * span * up, ctr + 2.2 * span * up + 0.05 * span * fwd):
        tiles.append(splat_points(Pk, Ck, look_at(eye, ctr, up if abs((ctr - eye) @ up) < 0.99 * np.linalg.norm(ctr - eye) else fwd), Ko, W, H))
    Image.fromarray((np.concatenate(tiles, 1) * 255).astype(np.uint8)).save(f'{out}/mesh_sheet.jpg', quality=88)
    if level and os.path.exists(level):                                                # same transform as model.splat
        M = np.array(json.load(open(level))['matrix']); mesh.apply_transform(M)
    mesh.export(f'{out}/mesh.ply'); mesh.export(f'{out}/mesh.glb')
    st = sorted(glob.glob(f'{result}/stats/val_step*.json')); mm = json.load(open(st[-1])) if st else {}
    info = dict(kind=kind, vertices=len(mesh.vertices), faces=len(mesh.faces), voxel=vox, levelled=bool(level and os.path.exists(level)),
                heldout_psnr=round(mm['psnr'], 2) if 'psnr' in mm else None, heldout_ssim=round(mm['ssim'], 3) if 'ssim' in mm else None,
                observed_voxels=float((wt > 0).mean()))
    json.dump(info, open(f'{out}/mesh.json', 'w'), indent=1); print('MESH', json.dumps(info), flush=True)
    return info

if __name__ == '__main__':
    a = sys.argv[1:]; main(a[0], a[1], a[2], a[3], a[4] if len(a) > 4 and a[4] != '-' else None, a[5] if len(a) > 5 else '2dgs', int(a[6]) if len(a) > 6 else 512)
