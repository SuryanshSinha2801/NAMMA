import gc, glob, json, os, shutil, subprocess, sys, time
"""Kaggle job anasrazy/namma-freefix: FreeFix (Zhou et al., 3DV 2026, MIT, github.com/hyzhou404/FreeFix) on a finished run, with
its SDXL backbone (stable-diffusion-xl-refiner-1.0, fits a T4; Flux does not). FreeFix's own loop (ours/refine_by_sdxl.py):
for each view beside the path, render it, compute per-pixel certainty (Hessian of the splats, recon/refiner.py), let SDXL
repaint only the uncertain pixels (its masked guidance pipeline), then fine-tune the splats on that view before the next one.
  views      the first N held-out frames shifted sideways by ~50 cm (v3_qa's 'side'), consecutive along the walk
  proof      held-out frames WITHOUT the shift: PSNR / SSIM / LPIPS before and after (the real photo is the answer; must not
             drop); off-path renders before | SDXL | after in freefix_sheet.jpg
  coords     FreeFix's COLMAP loader normalises without gsplat 1.5.3's upside-down fix, so its cameras are moved into the run's
             (gsplat 1.5.3) frame; the splats (and their view-dependent colour) stay as trained
  out        model.ply of the refined splats in the run's coordinates (-> tour), freefix_report.json, sheet, renders
Inputs (v3/freefix_gh.py): namma-qwen-in (the run's model.ply + sparse) and the run's own frames dataset (never other images)."""
T0 = time.time(); W = '/kaggle/temp/ff'; O = '/kaggle/working'; os.makedirs(W, exist_ok=True)
CFG = json.loads(os.environ.get('JOB_CFG', '{}')); REP = dict(cfg=CFG)
NVIEWS, SHIFT = int(CFG.get('views', 8)), float(CFG.get('shift', 0.15)); NEVAL = int(CFG.get('eval_views', 16))
PROMPT = CFG.get('prompt', 'An indoor shot of a hotel room.')
os.environ['HF_HOME'] = f'{W}/hf'; os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def sh(c, check=False):
    print(f'\n[{time.time()-T0:5.0f}s] $ {c[:160]}', flush=True); r = subprocess.call(c, shell=True, executable='/bin/bash')
    if check and r: raise RuntimeError(f'failed: {c[:80]}')
    return r

def say(*a): print(f'[{time.time()-T0:5.0f}s]', *a, flush=True)
def save_report(): json.dump(REP, open(f'{O}/freefix_report.json', 'w'), indent=1)

sh(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', check=True)
sh(f'git clone -q --depth 1 https://github.com/hyzhou404/FreeFix {W}/freefix', check=True)
sch = f'{W}/freefix/ours/schedulers/euler_discrete_scheduler.py'; src = open(sch).read()          # upstream leaves a debugger
src = src.replace("            import ipdb; ipdb.set_trace()\n", "")                         # stop in the guide-only branch
src = src.replace("        else:\n            x0 = warp_latent * 0.8 + x0 * 0.2", "        elif warp_latent is not None:\n            x0 = warp_latent * 0.8 + x0 * 0.2")
assert 'ipdb' not in src; open(sch, 'w').write(src)
pl = f'{W}/freefix/ours/pipelines/sdxl_pipeline.py'; src = open(pl).read()                     # fused latents come out fp32
dec = '            image = self.vae.decode(latents, return_dict=False)[0]\n'; assert src.count(dec) == 1
open(pl, 'w').write(src.replace(dec, dec.replace('decode(latents,', 'decode(latents.to(self.vae.dtype),')))
F = f'{W}/repo/colab/real_scenes/final'; FF = f'{W}/freefix'; EX = f'{W}/gsplat/examples'; sys.path.insert(0, F)
sh(f'git clone -q --depth 1 -b v1.5.3 https://github.com/nerfstudio-project/gsplat {W}/gsplat && touch {EX}/datasets/__init__.py; '
   f'echo "torch==$(python -c "import torch; print(torch.__version__)")" > {W}/pin.txt; '
   f'pip install -q -c {W}/pin.txt ninja gsplat==1.5.3 plyfile roma einops viser tyro omegaconf fire tensorboard "torchmetrics[image]" '
   '"imageio[ffmpeg]" scikit-learn matplotlib diffusers transformers accelerate 2>&1 | tail -3; pip uninstall -y -q torchao; '
   'pip install -q --no-deps "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e"; '
   'sed -i "s/np.uint64(-1)/np.uint64(np.iinfo(np.uint64).max)/" $(python -c "import importlib.util as u; print(u.find_spec(\'pycolmap\').submodule_search_locations[0])")/scene_manager.py')
tgz = glob.glob('/kaggle/input/**/gsplat_ext.tgz', recursive=True)                          # namma-prep's compiled gsplat
if tgz:
    sh(f'mkdir -p ~/.cache && tar xzf {tgz[0]} -C ~/.cache')
    sh('find $(python -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec(\'gsplat\').origin))") -exec touch -d "2000-01-01" {} +')
sh(f'python {F}/gsplat_warm.py', check=True)

# 1 the run's frames (its own dataset only), cameras and model
import numpy as np, torch, frames
from PIL import Image
D = f'{W}/data'; ply = glob.glob('/kaggle/input/**/model.ply', recursive=True); assert ply, 'model.ply not attached'
zs = glob.glob('/kaggle/input/**/sparse.zip', recursive=True)
if zs: shutil.unpack_archive(zs[0], D)
else: shutil.copytree(os.path.dirname(glob.glob('/kaggle/input/**/sparse/**/cameras.bin', recursive=True)[0]), f'{D}/sparse/0')
FDS = CFG.get('frames_dataset', 'namma-dl3dv').split('/')[-1]
for z in glob.glob(f'/kaggle/input/**/{FDS}/**/frames.zip', recursive=True): shutil.unpack_archive(z, f'{W}/fr')
cnt = {}
for f in [g for g in glob.glob('/kaggle/input/**/*', recursive=True) if f'/{FDS}/' in g] + glob.glob(f'{W}/fr/**/*', recursive=True):
    if f.lower().endswith(('.jpg', '.jpeg', '.png')) and '/sparse' not in f: cnt[os.path.dirname(f)] = cnt.get(os.path.dirname(f), 0) + 1
assert cnt, f'no frames found in the attached dataset {FDS}'
say('frames', frames.from_images(max(cnt, key=cnt.get), f'{D}/images', int(CFG.get('nframes', 420)), int(CFG.get('long', 1440))))
sys.path.insert(0, EX); from datasets.colmap import Parser as GsParser
gp = GsParser(D, factor=1, normalize=True, test_every=8); T_g = gp.transform
assert len(glob.glob(f'{D}/images/*.jpg')) == len(gp.image_names), 'frames != cameras: wrong frames'
from plyfile import PlyData
v = PlyData.read(ply[0])['vertex'].data; t_ = lambda *k: torch.tensor(np.stack([v[x] for x in k], 1), dtype=torch.float32)
nr = len([n for n in v.dtype.names if n.startswith('f_rest_')]) // 3
splats = dict(means=t_('x', 'y', 'z'), scales=t_('scale_0', 'scale_1', 'scale_2'), quats=t_('rot_0', 'rot_1', 'rot_2', 'rot_3'),
              opacities=torch.tensor(np.asarray(v['opacity']), dtype=torch.float32), sh0=t_('f_dc_0', 'f_dc_1', 'f_dc_2')[:, None],
              shN=t_(*[f'f_rest_{i}' for i in range(3 * nr)]).reshape(-1, 3, nr).transpose(1, 2).contiguous())
RES = f'{W}/res'; os.makedirs(f'{RES}/ckpts', exist_ok=True); torch.save(dict(step=29999, splats=splats), f'{RES}/ckpts/ckpt_29999.pt')
say(len(v), 'splats, SH degree', int(np.sqrt(nr + 1)) - 1)

# 2 FreeFix, its cameras moved into the run's frame
sys.path.insert(0, FF); os.chdir(FF)
import recon.datasets.colmap as ffc
from recon.datasets.normalize import transform_cameras, transform_points
class RunFrameParser(ffc.Parser):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        B = T_g @ np.linalg.inv(self.transform)                                             # FreeFix frame -> run's frame
        self.camtoworlds = transform_cameras(B, self.camtoworlds); self.points = transform_points(B, self.points)
        self.transform = B @ self.transform; self.scene_scale = gp.scene_scale
ffc.Parser = RunFrameParser
from recon.trainer import Config
from recon.refiner import Refiner
from ours.pipelines.sdxl_pipeline import StableDiffusionXLImg2ImgPipeline
from ours.schedulers.euler_discrete_scheduler import EulerDiscreteScheduler
from omegaconf import OmegaConf
from diffusers import AutoencoderKL
base = OmegaConf.load(f'{FF}/exp_cfg/base.yaml')
cfgd = Config(data_dir=D, data_factor=1, result_dir=RES, test_every=8, disable_viewer=True, sh_degree=3)
side = SHIFT * float(np.linalg.norm(gp.camtoworlds[:, :3, 3] - gp.camtoworlds[:, :3, 3].mean(0), axis=1).max())
ref = Refiner(cfgd, load_step=29999, test_split='test', test_trans=[side, 0, 0], test_rots=[0, 0, 0], c_exp_index=list(base.c_exp_index),
              hessian_attr=list(base.hessian_attr), test_len=NVIEWS, data_type='colmap')
assert np.allclose(ref.parser.camtoworlds[:, :3, 3], gp.camtoworlds[:, :3, 3], atol=1e-4), 'camera frames differ'
say('FreeFix refiner ready; side shift', round(side, 4), 'units; test views', len(ref.test_dataset), 'train views', len(ref.train_dataset))

def heldout_eval(tag):
    out = []
    for i in range(min(NEVAL, len(ref.test_dataset))):
        _, _, _, _, _, ev = ref.render(i, eval=True, trans=False); out.append(ev)
    m = {k: round(float(np.mean([e[k] for e in out])), 4) for k in out[0]}; REP[f'heldout_{tag}'] = m; say('held-out', tag, m); save_report()

def save_img(t, p): Image.fromarray((t.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(p, quality=92)
heldout_eval('before')
before = []
for i in range(NVIEWS):
    rgb, *_ = ref.render(i); before.append(rgb.detach().cpu()); save_img(rgb, f'{O}/view{i}_before.jpg')

vae = AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix', torch_dtype=torch.float16)   # MIT; SDXL's own VAE overflows in fp16
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained('stabilityai/stable-diffusion-xl-refiner-1.0', vae=vae, torch_dtype=torch.float16).to('cuda')
assert not pipe.vae.config.force_upcast
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
strength, nsteps = float(CFG.get('strength', 0.7)), int(base.num_inference_steps)
infer_steps = int(nsteps * strength); mask_scheduler = [int(infer_steps * c) for c in base.c_scheduler]
train_cams = [ref.train_dataset[j] for j in range(0, min(int(CFG.get('train_views', 70)), len(ref.train_dataset))) if j % 8]  # never the held-out frames
assert all(int(c['image_id']) % 8 for c in train_cams), 'held-out frame in the training views'
train_prob = [1] * len(train_cams); gen = torch.manual_seed(64); gens = []
for i in range(NVIEWS):
    t1 = time.time(); rgb, masks, alpha, depth, cam, _ = ref.render(i); masks = torch.stack(masks).to(pipe.device)
    img = rgb.permute(2, 0, 1).to(pipe.device); H, W_ = img.shape[1:]
    first = i == 0
    out = pipe(PROMPT, negative_prompt=base.negative_prompt, image=img, mask=masks, mask_scheduler=mask_scheduler,
               guide_until=infer_steps * base.guide_ratio, warp_image=img, warp_until=-1 if first else infer_steps * base.warp_ratio,
               warp_mask=None if first else alpha, height=H, width=W_, guidance_scale=3.5, num_inference_steps=nsteps,
               generator=gen, strength=strength).images[0].resize((W_, H))
    out.save(f'{O}/view{i}_sdxl.jpg', quality=92); gens.append(np.asarray(out))
    assert np.isfinite(gens[-1]).all() and gens[-1].std() > 5, f'SDXL output for view {i} is blank'
    rc = [dict(image=torch.from_numpy(np.asarray(out)), camtoworld=cam['c2w'], K=cam['K'], Gen=True, image_id=f'gen_{i}')]
    ref.refine(rc, train_cams, train_prob, max_steps=int(CFG.get('refine_steps', 400)) * (2 if first else 1))
    train_cams.append(rc[0]); train_prob.append(base.gen_prob)
    say(f'view {i} done in {time.time() - t1:.0f} s', f'GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB')
del pipe; gc.collect(); torch.cuda.empty_cache()
heldout_eval('after')
rows = []
for i in range(NVIEWS):
    rgb, *_ = ref.render(i); save_img(rgb, f'{O}/view{i}_after.jpg')
    rows.append(np.concatenate([(before[i].clamp(0, 1).numpy() * 255).astype(np.uint8), gens[i], (rgb.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)], 1))
Image.fromarray(np.concatenate(rows, 0)).resize((rows[0].shape[1] // 2, len(rows) * rows[0].shape[0] // 2)).save(f'{O}/freefix_sheet.jpg', quality=88)

# 3 refined splats -> model.ply in the run's coordinates (the gsplat PLY layout make_tour.py reads)
from plyfile import PlyElement
s = {k: p.detach().cpu() for k, p in ref.splats.items()}; n = len(s['means'])
shN = s['shN'].transpose(1, 2).reshape(n, -1).numpy(); cols = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2'] + [f'f_rest_{i}' for i in range(shN.shape[1])] + \
      ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
arr = np.concatenate([s['means'].numpy(), s['sh0'][:, 0].numpy(), shN, s['opacities'].numpy()[:, None], s['scales'].numpy(), s['quats'].numpy()], 1)
vv = np.zeros(n, [(c, 'f4') for c in cols])
for j, c in enumerate(cols): vv[c] = arr[:, j]
os.makedirs(f'{O}/full', exist_ok=True); PlyData([PlyElement.describe(vv, 'vertex')]).write(f'{O}/full/model.ply')
REP.update(splats_after=n, minutes=round((time.time() - T0) / 60, 1), ok=True); save_report(); say('FREEFIX DONE', json.dumps(REP))
