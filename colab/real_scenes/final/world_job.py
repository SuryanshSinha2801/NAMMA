import glob, json, os, shutil, subprocess, sys, time
"""Kaggle job anasrazy/namma-world: NAMMA World v1 (Plan B), one photo -> a 360-degree 3D world by "drones" (final/world_v1.py,
its loop tested on the CPU with a synthetic room first). Free T4s:
  depth     MoGe-2 ViT-L (Ruicheng/moge-2-vitl-normal, MIT): metric depth; the camera's field of view is given when known (the
            photo's K, every drone's 90-degree lens), estimated by MoGe for a plain photo
  painter   RealVisXL V4 inpainting (OzzyGT/RealVisXL_V4.0_inpainting, SDXL, openrail++) + LCM-LoRA (8 steps) + the fp16-safe
            SDXL VAE; paints only what the world cannot show yet; the prompt names the room, plus "ceiling"/"floor" when the drone
            looks up/down
  check     smoke tests first (GPU, depth sane, a painted square not blank); then per sample: world.glb + viewer, a 360-degree
            sheet, every drone's render | painted pair, and the world rendered at the real neighbouring video frames (cameras
            from SfM, metric scale from the SfM points vs the depth) -> PSNR / SSIM against them
Inputs (v3/world_gh.py): dataset namma-world-in, flat files per sample: <name>__cam.json (K, prompt, SfM points, neighbours),
<name>__input.jpg, <name>__nb_<offset>.jpg."""
T0 = time.time(); LOCAL = os.environ.get('WORLD_LOCAL')                                   # CPU dry run with stand-in models (test)
W = os.environ.get('WORLD_TMP', '/kaggle/temp/world'); O = os.environ.get('WORLD_OUT', '/kaggle/working'); IN = os.environ.get('WORLD_IN', '/kaggle/input')
os.makedirs(W, exist_ok=True); os.makedirs(O, exist_ok=True)
CFG = json.loads(os.environ.get('JOB_CFG', '{}')); REP = dict(cfg={k: v for k, v in CFG.items() if k != 'samples'}, samples={})
os.environ['HF_HOME'] = f'{W}/hf'; os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def sh(c, check=False):
    print(f'\n[{time.time()-T0:5.0f}s] $ {c[:160]}', flush=True); r = subprocess.call(c, shell=True, executable='/bin/bash')
    if check and r: raise RuntimeError(f'failed: {c[:80]}')
    return r


def say(*a): print(f'[{time.time()-T0:5.0f}s]', *a, flush=True)
def save(): json.dump(REP, open(f'{O}/world_report.json', 'w'), indent=1)


if not LOCAL:
    sh(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', check=True)
    sh('pip uninstall -y -q torchao; pip install -q open3d trimesh peft diffusers transformers accelerate 2>&1 | tail -3', check=True)
    sh('pip install -q --no-deps "git+https://github.com/microsoft/MoGe.git@0744441" && '
       'pip install -q "git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183"', check=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if LOCAL else f'{W}/repo/colab/real_scenes/final')
import numpy as np, cv2, torch
from PIL import Image
import world_v1 as WV, genworld as G
NG = torch.cuda.device_count(); assert NG >= 1 or LOCAL, 'no GPU: enable the T4 accelerator'
DEV_P, DEV_D = ('cuda:0', 'cuda:1' if NG > 1 else 'cuda:0') if NG else ('cpu', 'cpu')
REP['gpus'] = [torch.cuda.get_device_name(i) for i in range(NG)]; say('GPUs', REP['gpus'])

# --- models
from moge.model.v2 import MoGeModel
t = time.time(); moge = MoGeModel.from_pretrained(CFG.get('depth_model', 'Ruicheng/moge-2-vitl-normal')).to(DEV_D).eval()
REP['load_depth_s'] = round(time.time() - t, 1)


def depth_of(rgb, fov_x=None):
    """Metric z-depth [H, W] and normalised intrinsics. Pixels MoGe calls invalid (sky, outside a window) get the farthest
    valid depth."""
    x = torch.tensor(rgb / 255, dtype=torch.float32, device=DEV_D).permute(2, 0, 1)
    with torch.no_grad(): o = moge.infer(x, fov_x=fov_x, resolution_level=int(CFG.get('moge_level', 9)))
    d = o['depth'].float().cpu().numpy(); ok = np.isfinite(d) & (d > 0)
    assert ok.mean() > 0.3, f'depth invalid on {100 * (1 - ok.mean()):.0f} % of the image'
    d[~ok] = np.percentile(d[ok], 99.5); return d.astype(np.float32), o['intrinsics'].float().cpu().numpy()


from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, AutoencoderKL, LCMScheduler
t = time.time(); vae = AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix', torch_dtype=torch.float16); pipe = None; CN = None
if CFG.get('controlnet', 'diffusers/controlnet-depth-sdxl-1.0-small'):                     # the room's depth guides the painter
    try: CN = ControlNetModel.from_pretrained(CFG.get('controlnet', 'diffusers/controlnet-depth-sdxl-1.0-small'), torch_dtype=torch.float16, variant='fp16')
    except Exception as e: REP['controlnet_error'] = repr(e)[:300]; say('ControlNet failed, painting without the depth guide:', REP['controlnet_error'])
for mid in (CFG.get('inpaint_model', 'OzzyGT/RealVisXL_V4.0_inpainting'), 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1'):
    try:
        if CN is not None: pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(mid, controlnet=CN, vae=vae, torch_dtype=torch.float16, variant='fp16')
        else: pipe = StableDiffusionXLInpaintPipeline.from_pretrained(mid, vae=vae, torch_dtype=torch.float16, variant='fp16')
        pipe = pipe.to(DEV_P); REP['painter'] = mid; break
    except Exception as e: say('painter', mid, 'failed:', repr(e)[:300]); REP.setdefault('painter_errors', []).append(f'{mid}: {repr(e)[:200]}')
assert pipe is not None, 'no inpainting model could be loaded'
REP['controlnet'] = CFG.get('controlnet', 'diffusers/controlnet-depth-sdxl-1.0-small') if CN is not None else None
LCM = bool(CFG.get('lcm', True))
if LCM:
    try: pipe.load_lora_weights('latent-consistency/lcm-lora-sdxl'); pipe.fuse_lora(); pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    except Exception as e: LCM = False; REP['lcm_error'] = repr(e)[:300]; say('LCM-LoRA failed, normal sampling:', REP['lcm_error'])
pipe.set_progress_bar_config(disable=True); REP['load_painter_s'] = round(time.time() - t, 1); REP['lcm'] = LCM
STEPS, GS = int(CFG.get('steps', 8) if LCM else CFG.get('steps_full', 25)), float(CFG.get('guidance', 1.5) if LCM else CFG.get('guidance_full', 6.0))
NEG = CFG.get('negative', 'people, person, text, watermark, logo, frame, blurry, low quality, cartoon, painting, distorted, fisheye, duplicate')
NEG_ROOM = CFG.get('negative_room', 'bed, sofa, couch, chair, table, desk, furniture, lamp, ') + NEG   # drones paint the room, not new furniture


def control_of(depth, w8, h8):
    """Depth -> the ControlNet's input: inverse depth scaled to 0..1 (near = white), 3 channels, at the painting size."""
    disp = 1 / np.maximum(depth, 1e-3); lo, hi = np.percentile(disp, 1), np.percentile(disp, 99)
    v = (np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(np.repeat(v[..., None], 3, -1)).resize((w8, h8), Image.BILINEAR)


def paint(rgb, hole, prompt, seed, depth=None, negative=None):
    """SDXL inpainting at ~1 megapixel (sides multiples of 64), guided by depth (ControlNet) when loaded; the hole is pre-filled
    with the colours around it, so nothing black leaks in. Returns the painted image at the input size."""
    H, W_ = hole.shape; k = (1024 * 1024 / (H * W_)) ** 0.5; w8, h8 = max(64, round(W_ * k / 64) * 64), max(64, round(H * k / 64) * 64)
    sm = cv2.resize(rgb, (W_ // 4, H // 4), interpolation=cv2.INTER_AREA); hm = cv2.resize(hole.astype(np.uint8), (W_ // 4, H // 4), interpolation=cv2.INTER_NEAREST)
    fill = cv2.resize(cv2.inpaint(sm, hm, 5, cv2.INPAINT_TELEA), (W_, H), interpolation=cv2.INTER_LINEAR)
    src = np.where(hole[..., None], fill, rgb).astype(np.uint8)
    im = Image.fromarray(src).resize((w8, h8), Image.LANCZOS); m = Image.fromarray(hole.astype(np.uint8) * 255).resize((w8, h8), Image.NEAREST)
    kw = dict(prompt=prompt, negative_prompt=negative or NEG, image=im, mask_image=m, width=w8, height=h8, num_inference_steps=STEPS,
              guidance_scale=GS, strength=float(CFG.get('strength', 1.0)), generator=torch.Generator(DEV_P).manual_seed(int(seed)))
    if CN is not None:
        assert depth is not None, 'the depth guide is needed'
        kw.update(control_image=control_of(depth, w8, h8), controlnet_conditioning_scale=float(CFG.get('cn_scale', 0.6)))
    with torch.no_grad(): out = pipe(**kw).images[0]
    out = np.asarray(out.convert('RGB').resize((W_, H), Image.LANCZOS))
    assert out.std() > 2 and out.mean() > 3, 'painter returned a blank image (fp16 overflow?)'
    return out


def view_prompt(base, c2w, box=None):
    up = box['R'][:, 1] if box is not None else np.array([0., -1., 0.]); fu = float(c2w[:3, 2] @ up)   # camera forward vs up
    return base + (', looking up at the plain ceiling' if fu > 0.6 else ', looking down at the floor' if fu < -0.6 else '')


# --- smoke tests (fail fast, before any long work)
cams = sorted(glob.glob(f'{IN}/**/*cam.json', recursive=True)); assert cams, 'namma-world-in not attached (no cam.json)'
s0 = json.load(open(cams[0])); im0 = cv2.cvtColor(cv2.imread(f'{os.path.dirname(cams[0])}/{s0["input"]}'), cv2.COLOR_BGR2RGB)
t = time.time(); d0, _ = depth_of(im0, float(np.degrees(2 * np.arctan(im0.shape[1] / 2 / np.array(s0['K'])[0, 0]))))
smoke = dict(depth_s=round(time.time() - t, 2), depth_median_m=round(float(np.median(d0)), 2))
assert 0.3 < smoke['depth_median_m'] < 30, f'depth looks wrong: {smoke}'
sq = cv2.resize(im0, (1024, 1024), interpolation=cv2.INTER_AREA); dq = cv2.resize(d0, (1024, 1024), interpolation=cv2.INTER_LINEAR)
hole = np.zeros((1024, 1024), bool); hole[384:640, 384:640] = True
paint(sq, hole, s0.get('prompt', 'a room'), 0, depth=dq); t = time.time(); p_ = paint(sq, hole, s0.get('prompt', 'a room'), 1, depth=dq)   # 2nd = warm
smoke.update(paint_s=round(time.time() - t, 2), painted_std=round(float(p_[hole].std()), 1), outside_kept=bool(np.abs(p_[~hole].astype(int) - sq[~hole]).mean() < 25))
cv2.imwrite(f'{O}/smoke_paint.jpg', cv2.cvtColor(np.concatenate([sq, p_], 1), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
REP['smoke'] = smoke; save(); say('SMOKE OK', smoke)

# --- the worlds
SIZE, DSTEP, USE_BOX = int(CFG.get('drone_px', 1024)), int(CFG.get('drone_step', 6)), bool(CFG.get('room_box', True))


def make_world(cj, R):
    S = json.load(open(cj)); d = os.path.dirname(cj); name = S['name']; out = f'{O}/{name}'; os.makedirs(f'{out}/drones', exist_ok=True)
    R.update(frame=S.get('frame'), prompt=S.get('prompt')); t_all = time.time()
    img = cv2.cvtColor(cv2.imread(f'{d}/{S["input"]}'), cv2.COLOR_BGR2RGB); H, W_ = img.shape[:2]; shutil.copy(f'{d}/{S["input"]}', f'{out}/input.jpg')
    K = np.array(S['K'], float) if S.get('K') else None
    t = time.time(); D0, intr = depth_of(img, float(np.degrees(2 * np.arctan(W_ / 2 / K[0, 0]))) if K is not None else None)
    if K is None: K = intr * np.array([[W_], [H], [1.]])
    R['depth_s'] = round(time.time() - t, 1); R['depth_m'] = [round(float(np.percentile(D0, 5)), 2), round(float(np.percentile(D0, 95)), 2)]
    box = WV.room_box(D0, K) if USE_BOX else None
    if box is not None:
        R['room_box'] = dict(width_m=round(float(box['hi'][0] - box['lo'][0]), 2), height_m=round(float(box['hi'][1] - box['lo'][1]), 2),
                             depth_m=round(float(box['hi'][2] - box['lo'][2]), 2), camera_height_m=round(float(-box['lo'][1]), 2), seen=box['seen'])
        say(name, 'room box', R['room_box'])
    style = ', ' + CFG.get('style', 'realistic photo, natural light, high detail, sharp')
    base = S.get('prompt', 'interior photo of a room') + style
    room = CFG.get('room_prompt', 'the same room continued, plain walls, floor and ceiling, same materials and lighting') + style
    seed = [int(CFG.get('seed', 7))]

    def inpaint(rgb, hole, c2w=None, K=None, depth=None):
        seed[0] += 1; c2w = np.eye(4) if c2w is None else c2w
        if box is not None: return paint(rgb, hole, view_prompt(room, c2w, box), seed[0], depth=depth, negative=NEG_ROOM)
        return paint(rgb, hole, view_prompt(base, c2w), seed[0], depth=depth)

    def depth_fn(rgb, c2w=None, K=None):
        return depth_of(rgb, float(np.degrees(2 * np.arctan(rgb.shape[1] / 2 / K[0, 0]))))[0]

    world = WV.World(); t = time.time(); R['start'] = WV.start(world, img, K, D0, inpaint, step=2, box=box); R['start_s'] = round(time.time() - t, 1)
    t = time.time(); R['drones'] = WV.fly(world, inpaint, depth_fn, size=SIZE, fov=90.0, step=DSTEP, log=f'{out}/drones', box=box); R['fly_s'] = round(time.time() - t, 1)
    R['generate_minutes'] = round((time.time() - t_all) / 60, 2); say(name, 'world generated in', R['generate_minutes'], 'min')
    R['glb_mb'] = round(WV.export(world, f'{out}/world.glb') / 1e6, 1)
    open(f'{out}/index.html', 'w').write(WV.VIEWER)
    json.dump(dict(fov_y=round(float(np.degrees(2 * np.arctan(H / 2 / K[1, 1]))), 1), walk_m=float(CFG.get('walk_m', 1.0)),
                   focus=round(float(np.median(D0)), 2)), open(f'{out}/world.json', 'w'))
    rows, R['coverage_360'] = [], {}
    for pitch in (0, 50, -50):
        strip, cov = WV.sheet360(world, 384, 45, pitch); rows.append(strip); R['coverage_360'][str(pitch)] = round(100 * float(np.mean(cov)), 2)
    cv2.imwrite(f'{out}/world360.jpg', cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    # the input view (should equal the photo) and the real neighbouring frames (true new views)
    r0, _, h0 = world.render(np.eye(4), K, W_, H); ps, ss = G.scores(r0, img, h0); R['input_view'] = dict(psnr=ps, ssim=ss)
    sfm = S.get('sfm') or {}; R['novel'] = []; sheet = [[img, r0]]
    if sfm.get('uv'):
        uv = np.array(sfm['uv'], int); z = np.array(sfm['z'], float); ok = (uv[:, 0] >= 0) & (uv[:, 0] < W_) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (z > 0)
        scale = float(np.median(D0[uv[ok, 1], uv[ok, 0]] / z[ok])); R['sfm_to_metres'] = round(scale, 4)
        for nb in S.get('neighbours', []):
            gt = cv2.imread(f'{d}/{nb["file"]}')
            if gt is None: R.setdefault('missing', []).append(nb['file']); say(name, 'neighbour missing:', nb['file']); continue
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB); c2w = np.array(nb['c2w'], float); c2w[:3, 3] *= scale
            ren, _, hit = world.render(c2w, K, W_, H); ps, ss = G.scores(ren, gt, hit)
            R['novel'].append(dict(offset=nb['offset'], moved_m=round(float(np.linalg.norm(c2w[:3, 3])), 2),
                                   turned_deg=round(float(np.degrees(np.arccos(np.clip((np.trace(c2w[:3, :3]) - 1) / 2, -1, 1)))), 1),
                                   coverage_pct=round(100 * float(hit.mean()), 1), psnr=ps, ssim=ss, similarity_pct=round(100 * ss, 1) if ss else None))
            say(name, 'novel', R['novel'][-1]); sheet.append([gt, ren])
        sim = [n['similarity_pct'] for n in R['novel'] if n['similarity_pct']]; R['novel_similarity_mean_pct'] = round(float(np.mean(sim)), 1) if sim else None
    sh_ = np.concatenate([np.concatenate(r_, 1) for r_ in sheet], 0)
    cv2.imwrite(f'{out}/world_sheet.jpg', cv2.cvtColor(sh_, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    R['minutes'] = round((time.time() - t_all) / 60, 2); json.dump(R, open(f'{out}/report.json', 'w'), indent=1)
    say('WORLD', name, json.dumps({k: v for k, v in R.items() if k != 'drones'}))


for cj in cams:                                                                               # one failed sample does not stop the rest
    R = REP['samples'].setdefault(json.load(open(cj))['name'], {})
    try: make_world(cj, R)
    except Exception as e:
        import traceback; R['error'] = repr(e)[:400]; say('SAMPLE FAILED', cj, traceback.format_exc()[-1500:])
    save(); gc_ = __import__('gc'); gc_.collect(); torch.cuda.empty_cache()
REP['minutes_total'] = round((time.time() - T0) / 60, 1); save(); say('DONE', REP['minutes_total'], 'min')
