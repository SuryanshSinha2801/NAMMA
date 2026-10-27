"""3D Gaussian splat renderer on the CPU (PyTorch, no CUDA), for measuring clean-ups while Kaggle has no GPU hours.
Same model as gsplat's rasterization: 3D covariance R S S^T R^T, EWA projection (local affine Jacobian) + 0.3 px blur (eps2d),
3-sigma footprint, SH colour (degree up to 3) + 0.5, front-to-back alpha compositing per 16x16 tile, alpha <= 0.999, alpha < 1/255
skipped, black background. It is slower (seconds per view at ~480 px) and has no early ray termination (differences < 1e-4).
  render(splats, c2w, K, W, H) -> [H, W, 3] float tensor in 0..1;  load_ply(path) -> splats (activated)"""
import numpy as np, torch

C0, C1 = 0.28209479177387814, 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435]


def load_ply(path):
    from plyfile import PlyData
    v = PlyData.read(path)['vertex'].data; f = lambda *k: torch.tensor(np.stack([v[x] for x in k], 1), dtype=torch.float32)
    nr = len([n for n in v.dtype.names if n.startswith('f_rest_')]) // 3
    sh = f('f_dc_0', 'f_dc_1', 'f_dc_2')[:, None]
    if nr: sh = torch.cat([sh, f(*[f'f_rest_{i}' for i in range(3 * nr)]).reshape(-1, 3, nr).transpose(1, 2)], 1)
    q = f('rot_0', 'rot_1', 'rot_2', 'rot_3')
    return dict(means=f('x', 'y', 'z'), quats=q / q.norm(dim=1, keepdim=True), scales=torch.exp(f('scale_0', 'scale_1', 'scale_2')),
                opacities=torch.sigmoid(torch.tensor(np.asarray(v['opacity']), dtype=torch.float32)), sh=sh)


def sh_color(sh, dirs):
    d = sh.shape[1]; x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]; c = C0 * sh[:, 0]
    if d > 1: c = c - C1 * y * sh[:, 1] + C1 * z * sh[:, 2] - C1 * x * sh[:, 3]
    if d > 4:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        c = c + C2[0] * xy * sh[:, 4] + C2[1] * yz * sh[:, 5] + C2[2] * (2 * zz - xx - yy) * sh[:, 6] + C2[3] * xz * sh[:, 7] + C2[4] * (xx - yy) * sh[:, 8]
        if d > 9:
            c = (c + C3[0] * y * (3 * xx - yy) * sh[:, 9] + C3[1] * xy * z * sh[:, 10] + C3[2] * y * (4 * zz - xx - yy) * sh[:, 11]
                 + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12] + C3[4] * x * (4 * zz - xx - yy) * sh[:, 13]
                 + C3[5] * z * (xx - yy) * sh[:, 14] + C3[6] * x * (xx - 3 * yy) * sh[:, 15])
    return (c + 0.5).clamp(min=0)


def quat_rot(q):
    w, x, y, z = q.unbind(1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


@torch.no_grad()
def render(s, c2w, K, W, H, tile=16, near=0.01, eps2d=0.3):
    c2w = torch.as_tensor(c2w, dtype=torch.float32); K = torch.as_tensor(K, dtype=torch.float32)
    w2c = torch.linalg.inv(c2w); R, t = w2c[:3, :3], w2c[:3, 3]
    pc = s['means'] @ R.T + t; z = pc[:, 2]; ok = z > near
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * pc[:, 0] / z + cx; v = fy * pc[:, 1] / z + cy
    ok &= (u > -0.5 * W) & (u < 1.5 * W) & (v > -0.5 * H) & (v < 1.5 * H)
    idx = ok.nonzero()[:, 0]; pc, z, u, v = pc[idx], z[idx], u[idx], v[idx]
    M = quat_rot(s['quats'][idx]) * s['scales'][idx][:, None, :]; Sig = M @ M.transpose(1, 2)
    tfx, tfy = 0.5 * W / fx, 0.5 * H / fy                                                    # gsplat's persp_proj clamp
    tx = z * (pc[:, 0] / z).clamp(-(cx / fx + 0.3 * tfx), (W - cx) / fx + 0.3 * tfx)
    ty = z * (pc[:, 1] / z).clamp(-(cy / fy + 0.3 * tfy), (H - cy) / fy + 0.3 * tfy)
    J = torch.zeros(len(idx), 2, 3); J[:, 0, 0] = fx / z; J[:, 0, 2] = -fx * tx / z ** 2; J[:, 1, 1] = fy / z; J[:, 1, 2] = -fy * ty / z ** 2
    T = J @ R; cov = T @ Sig @ T.transpose(1, 2); cov[:, 0, 0] += eps2d; cov[:, 1, 1] += eps2d
    a, b, c = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]; det = a * c - b * b; good = det > 0
    mid = 0.5 * (a + c); lam = mid + torch.sqrt((mid * mid - det).clamp(min=0.01)); r = torch.ceil(3 * torch.sqrt(lam))
    good &= (u + r > 0) & (u - r < W) & (v + r > 0) & (v - r < H) & (r > 0)
    g = good.nonzero()[:, 0]; idx, z, u, v, r = idx[g], z[g], u[g], v[g], r[g]
    ia, ib, ic = (c[g] / det[g]), (-b[g] / det[g]), (a[g] / det[g])
    op = s['opacities'][idx]
    campos = c2w[:3, 3]; dirs = s['means'][idx] - campos; dirs = dirs / dirs.norm(dim=1, keepdim=True)
    col = sh_color(s['sh'][idx], dirs)
    nx, ny = (W + tile - 1) // tile, (H + tile - 1) // tile
    x0 = ((u - r) // tile).clamp(0, nx - 1).long(); x1 = ((u + r) // tile).clamp(0, nx - 1).long()
    y0 = ((v - r) // tile).clamp(0, ny - 1).long(); y1 = ((v + r) // tile).clamp(0, ny - 1).long()
    cnt = (x1 - x0 + 1) * (y1 - y0 + 1); gid = torch.repeat_interleave(torch.arange(len(idx)), cnt)
    k = torch.arange(len(gid)) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt); w_ = (x1 - x0 + 1)[gid]
    tid = (y0[gid] + k // w_) * nx + (x0[gid] + k % w_)
    order = torch.argsort(z); rank = torch.empty_like(order); rank[order] = torch.arange(len(order))
    key = tid * len(order) + rank[gid]; so = torch.argsort(key); tid, gid = tid[so], gid[so]
    bounds = torch.searchsorted(tid, torch.arange(nx * ny + 1))
    img = torch.zeros(H, W, 3); py, px = torch.meshgrid(torch.arange(tile, dtype=torch.float32), torch.arange(tile, dtype=torch.float32), indexing='ij')
    for t_ in range(nx * ny):
        s0, s1 = int(bounds[t_]), int(bounds[t_ + 1])
        if s0 == s1: continue
        gg = gid[s0:s1]; ty_, tx_ = divmod(t_, nx)
        X = (px.reshape(-1) + tx_ * tile + 0.5)[:, None]; Y = (py.reshape(-1) + ty_ * tile + 0.5)[:, None]   # pixel centres
        dx, dy = X - u[gg][None], Y - v[gg][None]
        pw = -0.5 * (ia[gg] * dx * dx + ic[gg] * dy * dy) - ib[gg] * dx * dy
        al = (op[gg][None] * torch.exp(pw)).clamp(max=0.999); al = torch.where((pw <= 0) & (al >= 1 / 255), al, torch.zeros_like(al))
        Tr = torch.cumprod(torch.cat([torch.ones(len(X), 1), 1 - al[:, :-1]], 1), 1)
        out = (al * Tr) @ col[gg]
        hh, ww = min(tile, H - ty_ * tile), min(tile, W - tx_ * tile)
        img[ty_ * tile:ty_ * tile + hh, tx_ * tile:tx_ * tile + ww] = out.reshape(tile, tile, 3)[:hh, :ww]
    return img.clamp(0, 1)
