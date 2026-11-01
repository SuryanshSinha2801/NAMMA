"""NAMMA World v0 (Plan B, owner 27 Oct): one ordinary photo -> a 3D world you can look around in, CPU only, about a minute.
  depth     Depth Anything V2 (metric, indoor or outdoor) at the photo's resolution, in metres.
  surface   every pixel becomes a vertex at its depth; the photo is the texture. Triangles that span a depth jump (> 4 %
            between neighbours) are cut, so nothing is stretched across an edge (no rubber sheets, no smears).
  behind    what the photo cannot show (behind the near side of every depth jump) is generated: LaMa (big-lama, Apache-2.0,
            CPU) inpaints the colour, the depth comes from the far side; it becomes a second textured layer that appears
            when you move.
  border    the photo is outpainted by 15 % on every side (LaMa) with the depth continued, so you can turn a little.
  output    world.glb (Photo, Behind, Border; metres, OpenCV camera at the origin), a three.js viewer, previews, report.
  check     the world rendered at the input camera (should equal the photo) and, when the photo is a frame of a video with
            known cameras (SfM), at the real neighbouring frames: true novel views, PSNR / SSIM against the real frames.
Usage: python genworld.py <photo> <out dir> [fov_deg | K json] [depth model]   (see main() for the video-frame check)"""
import json, os, sys, time, io, numpy as np, cv2
T0 = time.time()
LONG = int(os.environ.get('GW_LONG', 768)); PAD = float(os.environ.get('GW_PAD', 0.15)); JUMP = float(os.environ.get('GW_JUMP', 0.04))


def say(*a): print(f'[{time.time() - T0:6.1f}s]', *a, flush=True)


def depth_metric(img, model):
    """Metric depth [H, W] in metres from Depth Anything V2 (transformers), resized to the image."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(model); net = AutoModelForDepthEstimation.from_pretrained(model).eval()
    with torch.no_grad():
        out = net(**proc(images=img, return_tensors='pt')).predicted_depth
    d = torch.nn.functional.interpolate(out[:, None], size=img.shape[:2], mode='bicubic', align_corners=False)[0, 0].numpy()
    return np.clip(d, 0.05, None).astype(np.float32)


def lama(img, mask):
    """Generative inpainting (LaMa, CPU). mask: True where to generate."""
    from simple_lama_inpainting import SimpleLama
    from PIL import Image
    global _LAMA
    if '_LAMA' not in globals(): _LAMA = SimpleLama()
    out = np.asarray(_LAMA(Image.fromarray(img), Image.fromarray((mask * 255).astype(np.uint8)).convert('L')))
    return out[:img.shape[0], :img.shape[1]]


def layers(img, depth, K):
    """Photo + border outpaint + hidden-behind layer. Returns the padded canvas, its depth, K of the canvas, masks."""
    H, W = depth.shape; p = int(round(PAD * max(H, W)))
    big = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_REFLECT_101); border = np.ones(big.shape[:2], bool); border[p:p + H, p:p + W] = False
    canvas = lama(big, border); say('border outpainted', canvas.shape)
    # depth beyond the photo: continue the log-depth smoothly (inpaint), no generated content is nearer than the edge
    ldb = cv2.copyMakeBorder(np.log(depth), p, p, p, p, cv2.BORDER_REPLICATE)
    D = np.exp(np.where(border, cv2.GaussianBlur(ldb, (0, 0), 9), ldb)).astype(np.float32)
    Kc = K.copy(); Kc[0, 2] += p; Kc[1, 2] += p
    # depth jumps: neighbouring pixels differing by > JUMP in log depth; the near side within 30 px of a jump is an occluder
    lg = np.log(D); jump = np.zeros(D.shape, bool)
    jump[:, 1:] |= np.abs(np.diff(lg, axis=1)) > JUMP; jump[1:, :] |= np.abs(np.diff(lg, axis=0)) > JUMP
    far = cv2.dilate(D, np.ones((31, 31), np.uint8)); band = cv2.dilate(jump.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    occl = band & (D < 0.95 * far)
    behind_rgb = lama(canvas, occl) if occl.any() else canvas.copy()
    behind_d = np.where(occl, far, D).astype(np.float32)
    behind_d = cv2.medianBlur(behind_d, 5)
    say(f'depth jumps {jump.mean() * 100:.1f} % of pixels, hidden layer {occl.mean() * 100:.1f} %')
    return canvas, D, Kc, border, jump, occl, behind_rgb, behind_d, p


def grid_mesh(D, K, keep_px, step, jump_thr):
    """Vertices on a step-px grid at their depth (OpenCV camera: x right, y down, z forward), uv into the canvas; quads whose
    corners differ by more than jump_thr in log depth, or that touch a pixel not in keep_px, are left out."""
    H, W = D.shape; ys, xs = np.arange(0, H, step), np.arange(0, W, step); gx, gy = np.meshgrid(xs, ys)
    z = D[gy, gx]; X = np.stack([(gx + 0.5 - K[0, 2]) / K[0, 0] * z, (gy + 0.5 - K[1, 2]) / K[1, 1] * z, z], -1).reshape(-1, 3)
    uv = np.stack([(gx + 0.5) / W, 1 - (gy + 0.5) / H], -1).reshape(-1, 2)
    nx, ny = len(xs), len(ys); idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    lz = np.log(z); q = np.stack([lz[:-1, :-1], lz[:-1, 1:], lz[1:, 1:], lz[1:, :-1]], -1)
    ok = (q.max(-1) - q.min(-1)) < jump_thr * step
    kp = keep_px[gy, gx]; ok &= kp[:-1, :-1] & kp[:-1, 1:] & kp[1:, 1:] & kp[1:, :-1]
    F = np.concatenate([np.stack([a[ok], c[ok], b[ok]], 1), np.stack([a[ok], d[ok], c[ok]], 1)])
    return X, F, uv


def textured(X, F, uv, rgb, name):
    import trimesh
    from PIL import Image
    jpg = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes(); im = Image.open(io.BytesIO(jpg)); im.load()
    mat = trimesh.visual.material.PBRMaterial(name=name, baseColorTexture=im, metallicFactor=0.0, roughnessFactor=1.0)
    used = np.unique(F); remap = -np.ones(len(X), int); remap[used] = np.arange(len(used))
    return trimesh.Trimesh(X[used], remap[F], visual=trimesh.visual.TextureVisuals(uv=uv[used], material=mat), process=False)


class Render:
    """Open3D ray casting of textured meshes (bilinear texture lookup); first hit wins."""
    def __init__(self, parts):
        import open3d as o3d
        self.parts = parts; Vs, Fs, src, n0 = [], [], [], 0
        for k, (V, F, uv, tex) in enumerate(parts): Vs.append(V); Fs.append(F + n0); src += [k] * len(F); n0 += len(V)
        self.V, self.F, self.src = np.concatenate(Vs), np.concatenate(Fs), np.array(src)
        self.UV = np.concatenate([p[2] for p in parts]); self.sc = o3d.t.geometry.RaycastingScene()
        self.sc.add_triangles(o3d.core.Tensor(self.V.astype(np.float32)), o3d.core.Tensor(self.F.astype(np.uint32)))

    def __call__(self, c2w, K, W, H):
        import open3d as o3d
        uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        d = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1).reshape(-1, 3) @ c2w[:3, :3].T
        d /= np.linalg.norm(d, axis=1, keepdims=True); o = np.tile(c2w[:3, 3], (len(d), 1))
        r = self.sc.cast_rays(o3d.core.Tensor(np.concatenate([o, d], 1).astype(np.float32)))
        t = r['t_hit'].numpy(); hit = np.isfinite(t); pid = np.where(hit, r['primitive_ids'].numpy().astype(np.int64), 0); b = r['primitive_uvs'].numpy()
        img = np.zeros((len(d), 3)); tri = self.F[pid]; uv = (1 - b[:, :1] - b[:, 1:]) * self.UV[tri[:, 0]] + b[:, :1] * self.UV[tri[:, 1]] + b[:, 1:] * self.UV[tri[:, 2]]
        for k, (_, _, _, tex) in enumerate(self.parts):
            m = hit & (self.src[pid] == k)
            if not m.any(): continue
            h, w = tex.shape[:2]; x = np.clip(uv[m, 0] * w - 0.5, 0, w - 1.001); y = np.clip((1 - uv[m, 1]) * h - 0.5, 0, h - 1.001)
            x0, y0 = x.astype(int), y.astype(int); fx, fy = (x - x0)[:, None], (y - y0)[:, None]; tf = tex.astype(np.float32)
            img[m] = tf[y0, x0] * (1 - fx) * (1 - fy) + tf[y0, x0 + 1] * fx * (1 - fy) + tf[y0 + 1, x0] * (1 - fx) * fy + tf[y0 + 1, x0 + 1] * fx * fy
        return img.reshape(H, W, 3).clip(0, 255).astype(np.uint8), hit.reshape(H, W)


def scores(a, b, m):
    """PSNR and SSIM (11 px Gaussian window) of a vs b on pixels m."""
    if m.sum() < 100: return None, None
    a_, b_ = a.astype(np.float64) / 255, b.astype(np.float64) / 255; psnr = float(-10 * np.log10(max(((a_ - b_) ** 2)[m].mean(), 1e-10)))
    g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5); mx, my = g(a_), g(b_); sx, sy, sxy = g(a_ * a_) - mx ** 2, g(b_ * b_) - my ** 2, g(a_ * b_) - mx * my
    s = ((2 * mx * my + 1e-4) * (2 * sxy + 9e-4)) / ((mx ** 2 + my ** 2 + 1e-4) * (sx + sy + 9e-4))
    return round(psnr, 2), round(float(s.mean(-1)[m].mean()), 4)


VIEWER = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NAMMA World</title><style>body{margin:0;background:#15161a;overflow:hidden;font:13px system-ui,sans-serif;color:#eee}
#ui{position:fixed;left:12px;top:12px;display:flex;gap:6px;z-index:2}#ui button{background:#2b2d31ee;color:#eee;border:1px solid #fff3;border-radius:16px;padding:7px 12px;cursor:pointer}
#ui button.on{background:#e8a96b;color:#111}#help{position:fixed;left:12px;bottom:12px;background:#0009;padding:6px 10px;border-radius:8px}</style></head><body>
<div id="ui"><button id="all" class="on">World</button><button id="photo">Photo only</button><button id="gen">Show generated</button><button id="reset">Reset</button></div>
<div id="help">Drag to look around, right-drag to move, scroll to step in. W A S D to walk.</div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three'; import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js'; import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const info = await (await fetch('./world.json')).json();
const r = new THREE.WebGLRenderer({antialias:true}); r.setPixelRatio(Math.min(devicePixelRatio, 2)); r.setSize(innerWidth, innerHeight); r.outputColorSpace = THREE.SRGBColorSpace; document.body.appendChild(r.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x15161a);
const cam = new THREE.PerspectiveCamera(info.fov_y, innerWidth / innerHeight, 0.02, 200);
const g = await new GLTFLoader().loadAsync('./world.glb'); const root = new THREE.Group(); root.rotation.x = Math.PI; root.add(g.scene); scene.add(root);  // OpenCV -> three.js
const parts = {}; g.scene.traverse(o => { if (o.isMesh) { const n = (o.name + ' ' + (o.parent?.name || '')); const k = /Behind/.test(n) ? 'Behind' : /Border/.test(n) ? 'Border' : 'Photo';
  o.material = new THREE.MeshBasicMaterial({map: o.material.map, side: THREE.DoubleSide}); o.userData.map = o.material.map; (parts[k] = parts[k] || []).push(o); } });
const ctl = new OrbitControls(cam, r.domElement); const home = () => { cam.position.set(0, 0, 0.001); ctl.target.set(0, 0, -info.focus); ctl.update(); }; home();
ctl.maxDistance = info.focus * 1.6; ctl.minDistance = 0.05;
const B = id => document.getElementById(id); const sel = id => ['all', 'photo', 'gen'].forEach(k => B(k).classList.toggle('on', k === id));
const show = (on) => Object.entries(parts).forEach(([k, ms]) => ms.forEach(o => o.visible = on[k]));
const tint = (on) => ['Behind', 'Border'].forEach(k => (parts[k] || []).forEach(o => { o.material.color.set(on ? 0xff9f6b : 0xffffff); o.material.needsUpdate = true; }));
B('all').onclick = () => { sel('all'); show({Photo: 1, Behind: 1, Border: 1}); tint(false); }; B('photo').onclick = () => { sel('photo'); show({Photo: 1, Behind: 0, Border: 0}); };
B('gen').onclick = () => { sel('gen'); show({Photo: 1, Behind: 1, Border: 1}); tint(true); }; B('reset').onclick = home;
const keys = {}; addEventListener('keydown', e => keys[e.code] = true); addEventListener('keyup', e => keys[e.code] = false); const clock = new THREE.Clock();
r.setAnimationLoop(() => { const dt = Math.min(clock.getDelta(), 0.05); const f = new THREE.Vector3(); cam.getWorldDirection(f); const s = new THREE.Vector3().crossVectors(f, cam.up);
  const mv = new THREE.Vector3(); if (keys.KeyW) mv.add(f); if (keys.KeyS) mv.sub(f); if (keys.KeyD) mv.add(s); if (keys.KeyA) mv.sub(s);
  if (mv.lengthSq()) { mv.normalize().multiplyScalar(0.8 * dt); cam.position.add(mv); ctl.target.add(mv); } ctl.update(); r.render(scene, cam); });
addEventListener('resize', () => { cam.aspect = innerWidth / innerHeight; cam.updateProjectionMatrix(); r.setSize(innerWidth, innerHeight); });
</script></body></html>"""


def build(img, K, out, model, name='world'):
    """img uint8 RGB (already at the working size), K for it. Writes world.glb etc. Returns (render function, parts, info)."""
    import trimesh
    os.makedirs(out, exist_ok=True); t0 = time.time()
    D0 = depth_metric(img, model); say(f'depth {model.split("/")[-1]}: {np.percentile(D0, 5):.2f}-{np.percentile(D0, 95):.2f} m')
    canvas, D, Kc, border, jump, occl, brgb, bd, p = layers(img, D0, K)
    photo_px = ~border; X1, F1, uv1 = grid_mesh(D, Kc, photo_px | border, 2, JUMP)
    # split the main surface into Photo (inside the original frame) and Border (outpainted) by triangle centre
    H, W = D.shape; cen = uv1[F1].mean(1); px = np.clip((cen[:, 0] * W).astype(int), 0, W - 1); py = np.clip(((1 - cen[:, 1]) * H).astype(int), 0, H - 1)
    on_photo = photo_px[py, px]
    X2, F2, uv2 = grid_mesh(bd, Kc, cv2.dilate(occl.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0, 2, JUMP)
    meshes = dict(Photo=textured(X1, F1[on_photo], uv1, canvas, 'Photo'), Border=textured(X1, F1[~on_photo], uv1, canvas, 'Border'),
                  Behind=textured(X2, F2, uv2, brgb, 'Behind'))
    sc = trimesh.Scene()
    for k, m in meshes.items():
        if len(m.faces): sc.add_geometry(m, node_name=k, geom_name=k)
    sc.export(f'{out}/{name}.glb'); mb = round(os.path.getsize(f'{out}/{name}.glb') / 1e6, 1)
    secs = round(time.time() - t0, 1); say(f'{name}.glb {mb} MB, {sum(len(m.faces) for m in meshes.values())} triangles, built in {secs} s')
    cv2.imwrite(f'{out}/{name}_canvas.jpg', cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)); cv2.imwrite(f'{out}/{name}_behind.jpg', cv2.cvtColor(brgb, cv2.COLOR_RGB2BGR))
    dv = (np.log(D) - np.log(D).min()) / (np.ptp(np.log(D)) + 1e-6); cv2.imwrite(f'{out}/{name}_depth.jpg', cv2.applyColorMap((255 - dv * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
    parts = [(np.asarray(m.vertices), np.asarray(m.faces), np.asarray(m.visual.uv), tex) for m, tex in
             ((meshes['Behind'], brgb), (meshes['Photo'], canvas), (meshes['Border'], canvas)) if len(m.faces)]
    info = dict(fov_y=round(float(np.degrees(2 * np.arctan(img.shape[0] / 2 / K[1, 1]))), 1), focus=round(float(np.median(D0)), 2), seconds_to_build=secs,
                glb_mb=mb, depth_m=[round(float(np.percentile(D0, 5)), 2), round(float(np.percentile(D0, 95)), 2)], hidden_layer_pct=round(float(occl.mean() * 100), 1),
                border_px=p)
    json.dump(info, open(f'{out}/{name}.json', 'w'))
    return Render(parts), info, D0


def main(out, model='depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf', *args):
    """Video-frame demo: args = data dir (images/ + sparse/0 of a filmed scene) + frame index + neighbour offsets (comma list).
    Builds the world from that one frame, then renders it at the real neighbouring frames (cameras from SfM, scale from the
    SfM points vs the metric depth) and scores it against them. A plain photo: args = path [fov_deg]."""
    os.makedirs(out, exist_ok=True); rep = dict(model=model)
    if args and os.path.isdir(args[0]):
        sys.path.insert(0, os.getcwd()); from datasets.colmap import Parser
        data, fi, offs = args[0], int(args[1]), [int(x) for x in args[2].split(',')]
        p = Parser(data, factor=1, normalize=False, test_every=8); cid = p.camera_ids[0]; K0 = p.Ks_dict[cid]; W0, H0 = p.imsize_dict[cid]
        k = LONG / max(W0, H0); W, H = round(W0 * k), round(H0 * k); K = K0.copy(); K[:2] *= k
        def load(i):
            im = cv2.imread(p.image_paths[i])
            if cid in p.mapx_dict: im = cv2.remap(im, p.mapx_dict[cid], p.mapy_dict[cid], cv2.INTER_LINEAR); x, y, w, h = p.roi_undist_dict[cid]; im = im[y:y + h, x:x + w]
            return cv2.cvtColor(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        img = load(fi); cv2.imwrite(f'{out}/input.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        render, info, D0 = build(img, K, out, model); rep.update(info, frame=p.image_names[fi])
        # metric scale of the SfM cameras: SfM points seen in this frame, their depth vs the predicted metric depth
        c2w_i = p.camtoworlds[fi]; w2c = np.linalg.inv(c2w_i); pts = p.points[p.point_indices[p.image_names[fi]]]
        pc = pts @ w2c[:3, :3].T + w2c[:3, 3]; pc = pc[pc[:, 2] > 0]; u = (K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]).astype(int); v = (K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]).astype(int)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H); scale = float(np.median(D0[v[ok], u[ok]] / pc[ok, 2])); rep['sfm_to_metres'] = round(scale, 4)
        rows = [[img, render(np.eye(4), K, W, H)[0]]]; s0 = scores(rows[0][1], img, np.ones((H, W), bool)); rep['input_view'] = dict(psnr=s0[0], ssim=s0[1])
        say('input view', rep['input_view']); rep['novel'] = []
        for o in offs:
            j = fi + o
            if not 0 <= j < len(p.image_names): continue
            T = np.linalg.inv(p.camtoworlds[j]) @ c2w_i; T[:3, 3] *= scale; c2w = np.linalg.inv(T)       # camera j in camera i's frame (metres)
            gt = load(j); ren, hit = render(c2w, K, W, H); ps, ss = scores(ren, gt, hit)
            r = dict(offset=o, frame=p.image_names[j], moved_m=round(float(np.linalg.norm(c2w[:3, 3])), 2),
                     turned_deg=round(float(np.degrees(np.arccos(np.clip((np.trace(c2w[:3, :3]) - 1) / 2, -1, 1)))), 1),
                     coverage_pct=round(float(hit.mean() * 100), 1), psnr=ps, ssim=ss, similarity_pct=round(ss * 100, 1) if ss else None)
            rep['novel'].append(r); say('novel view', r); rows.append([gt, ren])
        sheet = np.concatenate([np.concatenate(r_, 1) for r_ in rows], 0); hdr = np.zeros((26, sheet.shape[1], 3), np.uint8)
        cv2.putText(hdr, 'real frame', (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1); cv2.putText(hdr, 'NAMMA World (built from the first frame only)', (W + 8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imwrite(f'{out}/world_sheet.jpg', cv2.cvtColor(np.concatenate([hdr, sheet], 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        nov = [r['similarity_pct'] for r in rep['novel'] if r['similarity_pct']]; rep['novel_similarity_mean_pct'] = round(float(np.mean(nov)), 1) if nov else None
    else:
        path = args[0]; fov = float(args[1]) if len(args) > 1 else 65.0; im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        k = LONG / max(im.shape[:2]); img = cv2.resize(im, (round(im.shape[1] * k), round(im.shape[0] * k)), interpolation=cv2.INTER_AREA)
        f = 0.5 * img.shape[1] / np.tan(np.radians(fov) / 2); K = np.array([[f, 0, img.shape[1] / 2], [0, f, img.shape[0] / 2], [0, 0, 1.]])
        render, info, _ = build(img, K, out, model); rep.update(info, photo=os.path.basename(path), fov_assumed=fov)
        r0 = render(np.eye(4), K, img.shape[1], img.shape[0])[0]; s0 = scores(r0, img, np.ones(img.shape[:2], bool)); rep['input_view'] = dict(psnr=s0[0], ssim=s0[1])
    open(f'{out}/index.html', 'w').write(VIEWER)
    rep['minutes_total'] = round((time.time() - T0) / 60, 2); json.dump(rep, open(f'{out}/report.json', 'w'), indent=1); say('WORLD', json.dumps(rep))


if __name__ == '__main__':
    main(*sys.argv[1:])
