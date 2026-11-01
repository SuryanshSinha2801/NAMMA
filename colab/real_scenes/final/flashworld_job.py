import base64, gc, glob, io, json, os, shutil, subprocess, sys, time
"""Kaggle job <user>/namma-fw: FlashWorld (ICLR 2026, github.com/imlixinyang/FlashWorld, CC BY-NC-SA: non-commercial) run as
published, on 2 free T4s. Owner, 28 Oct: connect open modules, no model of ours. One photo + a camera path (24 views) -> the
3D Gaussians of the scene in one pass (Wan2.2-5B distilled, 4 steps).
Only what a T4 needs is changed, as text patches of their app.py / quant.py (nothing else): no bfloat16 compute (FW_DTYPE,
default float32), their fp8 weight storage kept (quant.py's non-native path works on any GPU), the text encoder and the
transformer loaded in bfloat16 on the CPU and the checkpoint memory-mapped (Kaggle RAM), the VAE + splat decoder on the second
GPU (their offload_vae switch, pointed at cuda:1), the prompt encoded once on cuda:0 before the transformer moves there, and
the VAE encoder / splat decoder run 4 frames at a time (both work frame by frame).
Out per request: world.splat (Gaussians, web format: position, scale, colour, rotation; most opaque N), world.json (start
camera = the photo's), video.mp4 (their render along the path), check.jpg (photo | world at the photo's camera) + report."""
T0 = time.time(); W = '/kaggle/temp/fw'; O = '/kaggle/working'; os.makedirs(W, exist_ok=True)
CFG = json.loads(os.environ.get('JOB_CFG', '{}')); REP = dict(cfg={k: v for k, v in CFG.items() if k != 'requests'}, worlds={})
os.environ['HF_HOME'] = f'{W}/hf'; os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'; os.environ['FW_DTYPE'] = CFG.get('dtype', 'float32')
FW_COMMIT = CFG.get('fw_commit', '424db4487de308a39e5137afbe85e7568a96193f')


def sh(c, check=False):
    print(f'\n[{time.time()-T0:5.0f}s] $ {c[:170]}', flush=True); r = subprocess.call(c, shell=True, executable='/bin/bash')
    if check and r: raise RuntimeError(f'failed: {c[:90]}')
    return r


def say(*a): print(f'[{time.time()-T0:5.0f}s]', *a, flush=True)
def save(): json.dump(REP, open(f'{O}/fw_report.json', 'w'), indent=1)


FW = f'{W}/FlashWorld'
sh(f'git clone -q https://github.com/imlixinyang/FlashWorld {FW} && cd {FW} && git checkout -q {FW_COMMIT}', check=True)
sh(f'echo "torch==$(python -c "import torch; print(torch.__version__)")" > {W}/pin.txt; pip uninstall -y -q torchao; '
   f'pip install -q -c {W}/pin.txt einops ftfy plyfile omegaconf av "imageio[ffmpeg]" ninja gsplat==1.5.3 '
   '"git+https://github.com/huggingface/diffusers.git@447e8322f76efea55d4769cd67c372edbf0715b8" 2>&1 | tail -3', check=True)
import numpy as np, torch
from PIL import Image
NG = torch.cuda.device_count(); assert NG >= 1, 'no GPU'; REP['gpus'] = [torch.cuda.get_device_name(i) for i in range(NG)]; save(); say('GPUs', REP['gpus'])
t = time.time()                                                                               # gsplat compiles once (~5 min)
sh('python -c "import torch; from gsplat import rasterization; d=\'cuda\'; m=torch.zeros(1,3,device=d); q=torch.tensor([[1.,0,0,0]],device=d); '
   'rasterization(m, q, torch.ones(1,3,device=d)*.1, torch.ones(1,device=d), torch.ones(1,3,device=d), torch.eye(4,device=d)[None], '
   'torch.tensor([[[100.,0,50],[0,100.,50],[0,0,1]]],device=d), 100, 100); print(\'gsplat ok\')"', check=True); REP['gsplat_s'] = round(time.time() - t)

# --- their code, patched for the T4 (text replacements, each checked)
os.chdir(FW); sys.path.insert(0, FW)
q = open('quant.py').read(); assert 'dtype: torch.dtype = torch.bfloat16' in q
q = q.replace('import torch\n', 'import torch, os\nFW_DT = getattr(torch, os.environ.get("FW_DTYPE", "float32"))\n', 1).replace('dtype: torch.dtype = torch.bfloat16', 'dtype: torch.dtype = FW_DT')
open('quant.py', 'w').write(q)
src = open('app.py').read(); a = src.index('@contextmanager\ndef onload_model'); b = src.index('@GPU\ndef process_generation_request'); body = src[a:b]
reps = [('torch.bfloat16', 'FW_DT'),                                                         # compute dtype (T4: no bfloat16 compute)
        ('UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float32)',
         'UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)'),   # RAM: 11 GB, not 22
        ('WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.float32)',
         'WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)'),
        ('torch.load(ckpt_path, map_location="cpu", weights_only=False)', 'torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)'),
        ('self.device if not self.offload_vae else "cpu"', 'self.device if not self.offload_vae else VAE_DEV')]
for x, y in reps:
    assert x in body, f'FlashWorld changed: {x[:60]}'; body = body.replace(x, y)
pre = ('import os, io, json, time, base64, torch, numpy as np, einops, imageio\nimport torch.nn as nn\nimport torch.nn.functional as F\n'
       'from contextlib import contextmanager\nfrom models import *\nfrom utils import *\nfrom transformers import T5TokenizerFast, UMT5EncoderModel\n'
       'from diffusers import FlowMatchEulerDiscreteScheduler\nmode = "Local"\nFW_DT = getattr(torch, os.environ.get("FW_DTYPE", "float32"))\n'
       'VAE_DEV = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"\n')
NS = {}; exec(compile(pre + body, 'flashworld_app_patched', 'exec'), NS)
FW_DT = NS['FW_DT']; from utils import export_gaussians  # noqa (their utils, kept for reference)
from huggingface_hub import hf_hub_download
t = time.time(); ckpt = hf_hub_download('imlixinyang/FlashWorld', 'model.ckpt'); REP['ckpt_download_s'] = round(time.time() - t)
t = time.time()
gs = NS['GenerationSystem'](ckpt_path=ckpt, device=torch.device('cuda:0'), offload_t5=True, offload_vae=True, offload_transformer_during_vae=True)
REP['load_s'] = round(time.time() - t); save(); say('FlashWorld loaded', REP['load_s'], 's')

# prompts: encoded once on cuda:0 (the transformer is still on the CPU), then the text encoder goes
reqs = sorted(glob.glob('/kaggle/input/**/*.fwreq.json', recursive=True)); assert reqs, 'namma-fw-in not attached'
texts = sorted({json.load(open(r)).get('text_prompt', '') for r in reqs}); cache = {}
gs.offload_t5 = False; gs.text_encoder.to('cuda:0')
with torch.no_grad():
    for tx in texts: cache['[Static] ' + tx] = gs.encode_text(['[Static] ' + tx]).cpu()
gs.text_encoder.to('cpu'); gs.text_encoder = torch.nn.Identity(); gc.collect(); torch.cuda.empty_cache()
gs.encode_text = lambda tx: cache[tx[0]].to(gs.device)
REP['text_ok'] = bool(all(torch.isfinite(v).all() for v in cache.values())); assert REP['text_ok'], 'text embedding not finite'

# transformer: its bfloat16 leftovers to the compute dtype, then onto cuda:0 for good
for m in gs.transformer.modules():
    for n, p in list(m.named_parameters(recurse=False)):
        if p.dtype == torch.bfloat16: p.data = p.data.to(FW_DT)
    for n, b_ in list(m.named_buffers(recurse=False)):
        if b_.dtype == torch.bfloat16: m._buffers[n] = b_.to(FW_DT)
gs.transformer.to('cuda:0'); gs.offload_transformer_during_vae = False; gc.collect(); torch.cuda.empty_cache()
CH = int(CFG.get('chunk', 4)); enc0, rec0 = gs.vae.encode, gs.recon_decoder.forward


class _Lat:
    def __init__(s, z): s.latent_dist = s; s.z = z
    def sample(s): return s.z


def vae_encode(x, *a, **k):
    if x.shape[0] <= CH: return enc0(x, *a, **k)
    return _Lat(torch.cat([enc0(x[i:i + CH], *a, **k).latent_dist.sample() for i in range(0, x.shape[0], CH)]))


def rec_forward(feats, z, cameras):
    T = cameras.shape[1]
    if T <= CH: return rec0(feats, z, cameras)
    return torch.cat([rec0(feats[i:i + CH], z[i:i + CH], cameras[:, i:i + CH]) for i in range(0, T, CH)], 1)


gs.vae.encode = vae_encode; gs.recon_decoder.forward = rec_forward
REP['mem_gb'] = [round(torch.cuda.memory_allocated(i) / 1e9, 2) for i in range(NG)]; save(); say('ready, GPU memory', REP['mem_gb'])


def quat_mat(q):                                                                              # [w, x, y, z] -> 3x3
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_quat(R):
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return [float(w), float((R[2, 1] - R[1, 2]) / (4 * w)), float((R[0, 2] - R[2, 0]) / (4 * w)), float((R[1, 0] - R[0, 1]) / (4 * w))]


def write_splat(path, G, n_max):
    """Gaussians [N, 3+1+3+4+3(sh+1)^2] (their layout, activated) -> .splat (32 bytes each: xyz, scale, RGBA, rotation wxyz),
    the n_max most opaque."""
    G = G.float().cpu().numpy(); xyz, op, sc, rq, f = G[:, :3], G[:, 3], G[:, 4:7], G[:, 7:11], G[:, 11:14]
    keep = np.flatnonzero(op >= 0.02)
    if len(keep) > n_max: keep = keep[np.argsort(-op[keep])[:n_max]]
    rgb = np.clip(0.5 + 0.28209479177387814 * f[keep], 0, 1); rq = rq[keep] / np.linalg.norm(rq[keep], axis=1, keepdims=True)
    rec = np.zeros(len(keep), dtype=[('p', '<f4', 3), ('s', '<f4', 3), ('c', 'u1', 4), ('r', 'u1', 4)])
    rec['p'] = xyz[keep]; rec['s'] = sc[keep]; rec['c'] = np.concatenate([rgb, op[keep, None]], 1) * 255; rec['r'] = np.clip(rq * 128 + 128, 0, 255)
    rec.tofile(path); return len(G), len(keep)


for rq in reqs:
    data = json.load(open(rq)); name = os.path.basename(rq).replace('.fwreq.json', ''); out = f'{O}/{name}'; os.makedirs(out, exist_ok=True); R = {}
    REP['worlds'][name] = R
    try:
        n_frame, H, Wd = data['resolution']
        img = Image.open(io.BytesIO(base64.b64decode(data['image_prompt'].split(',', 1)[1]))).convert('RGB'); assert img.size == (Wd, H), img.size
        img.save(f'{out}/input.jpg', quality=92); x = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0 * 2 - 1
        cams = torch.stack([torch.tensor([*c['quaternion'], *c['position'], c['fx'] / Wd, c['fy'] / H, c['cx'] / Wd, c['cy'] / H], dtype=torch.float32) for c in data['cameras']])
        fps = int(CFG.get('video_fps', 6)); t = time.time()
        with torch.no_grad():
            scene, ref_w2c, T_norm = gs.generate(cams, n_frame, x, data.get('text_prompt', ''), data['image_index'], H, Wd, video_path=f'{out}/video.mp4', video_fps=fps)
        R['seconds'] = round(time.time() - t, 1); R['finite'] = bool(torch.isfinite(scene).all()); assert R['finite'], 'NaN in the Gaussians'
        sc = scene.detach().cpu(); Tn = float(T_norm.item())
        sc[:, :3] *= Tn; sc[:, 4:7] *= Tn                                                     # back to the request's units (as their export)
        R['gaussians'], R['kept'] = write_splat(f'{out}/world.splat', sc, int(CFG.get('max_splats', 2500000)))
        R['splat_mb'] = round(os.path.getsize(f'{out}/world.splat') / 1e6, 1)
        # the viewer starts at the photo's camera, in the first camera's frame (the frame of the Gaussians)
        C = []
        for c in data['cameras']:
            m = np.eye(4); m[:3, :3] = quat_mat(np.array(c['quaternion'], float)); m[:3, 3] = c['position']; C.append(m)
        rel = np.linalg.inv(C[0]) @ C[data['image_index']]
        json.dump(dict(file='world.splat', fov_y=round(float(np.degrees(2 * np.arctan(H / 2 / data['cameras'][0]['fy']))), 1),
                       start=dict(position=rel[:3, 3].tolist(), quaternion=mat_quat(rel[:3, :3]))), open(f'{out}/world.json', 'w'))
        try:                                                                                  # check: their render at the photo's camera
            import imageio.v3 as iio
            fr = iio.imread(f'{out}/video.mp4', index=data['image_index'] * fps); a_ = np.asarray(img).astype(float); b2 = np.asarray(Image.fromarray(fr).resize((Wd, H))).astype(float)
            R['photo_view_psnr'] = round(float(-10 * np.log10(((a_ - b2) ** 2).mean() / 255 ** 2)), 2)
            Image.fromarray(np.concatenate([a_, b2], 1).astype(np.uint8)).save(f'{out}/check.jpg', quality=90)
        except Exception as e: R['check_error'] = repr(e)[:200]
        say(name, R)
    except Exception as e:
        import traceback; R['error'] = repr(e)[:400]; say('FAILED', name, traceback.format_exc()[-1500:])
    save(); gc.collect(); torch.cuda.empty_cache()
REP['minutes_total'] = round((time.time() - T0) / 60, 1); save(); say('DONE', REP['minutes_total'], 'min')
