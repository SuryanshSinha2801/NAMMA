"""Kaggle job anasrazy/namma-qwen: can Qwen-Image-Edit-2511 (Apache 2.0, open weights) clean our splat renders on a free T4?
Smoke test for the "AI fix" stage (docs/TOUR_QUALITY_RESEARCH.md section 7), on a finished run (default run 4):
  cases      2 held-out frames (never trained on: the real photo is the answer, so before/after is measured) and 3 views beside
             the filmed path (like v3_qa's row 2: no answer, judged by eye), each rendered from the run's model.ply
  editor     [render, the 2 nearest real training frames] + a "remove the smears, keep the layout" prompt; 4-step Lightning LoRA
  T4 fit     text encoder (Qwen2.5-VL-7B) 4-bit first, prompt embeddings kept, freed; then the 20B transformer as a 4-bit GGUF
             (unsloth Q4_K_M, 13 GB) + Lightning LoRA; fp16 first, bf16 if fp16 gives NaN/black
  out        qwen_sheet.jpg (refs | render | Qwen | real), qwen_report.json (s/image, GPU memory, dtype, PSNR/SSIM/LPIPS before/after)
Inputs (gh: v3/qwen_gh.py): the run's frames dataset (same frames.py selection as the run) + namma-qwen-in (model.ply, sparse.zip)."""
import gc, glob, json, os, shutil, subprocess, sys, time
T0 = time.time(); W = '/kaggle/temp/qw'; O = '/kaggle/working'; os.makedirs(W, exist_ok=True)
CFG = json.loads(os.environ.get('QWEN', '{}'))
REPO = 'Qwen/Qwen-Image-Edit-2511'; GGUF = CFG.get('gguf', 'qwen-image-edit-2511-Q4_K_M.gguf')
LORA = CFG.get('lora', 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors'); STEPS = int(CFG.get('steps', 4))
NFRAMES, LONG = int(CFG.get('nframes', 420)), int(CFG.get('long', 1440)); WO, HO = 1024, 576
PROMPT = CFG.get('prompt', 'Picture 1 is a render of a hotel room from a 3D scan; it has blurry smears, streaks and soft, washed-out '
                 'areas. Pictures 2 and 3 are real photos of the same room. Turn Picture 1 into a sharp, clean photograph of the same '
                 'view: remove the smears, streaks and blur, and use the real textures, colours and details from Pictures 2 and 3. '
                 'Keep exactly the same camera position, framing, perspective, layout and objects as Picture 1; do not add, move or '
                 'remove anything.')
os.environ['HF_HOME'] = f'{W}/hf'; REP = dict(cfg=CFG, prompt=PROMPT, cases=[])
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'                         # less fragmentation

def sh(c, check=False):
    print(f'\n[{time.time()-T0:5.0f}s] $ {c[:160]}', flush=True); r = subprocess.call(c, shell=True, executable='/bin/bash')
    if check and r: raise RuntimeError(f'failed: {c[:80]}')
    return r

def say(*a): print(f'[{time.time()-T0:5.0f}s]', *a, flush=True)

def save_report(): json.dump(REP, open(f'{O}/qwen_report.json', 'w'), indent=1)

sh(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', check=True)
F = f'{W}/repo/colab/real_scenes/final'; sys.path.insert(0, F); EX = f'{W}/gsplat/examples'
sh('ls /kaggle/input /kaggle/input/* | head -30; nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /kaggle/temp | tail -1; free -g | head -2')
sh(f'git clone -q --depth 1 -b v1.5.3 https://github.com/nerfstudio-project/gsplat {W}/gsplat && touch {EX}/datasets/__init__.py; '
   f'echo "torch==$(python -c "import torch; print(torch.__version__)")" > {W}/pin.txt; '
   f'pip install -q -c {W}/pin.txt ninja gsplat==1.5.3 plyfile lpips gguf peft bitsandbytes accelerate "transformers>=4.51" '
   '"git+https://github.com/huggingface/diffusers" 2>&1 | tail -3; pip uninstall -y -q torchao; '
   'pip install -q --no-deps "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e"; '
   'sed -i "s/np.uint64(-1)/np.uint64(np.iinfo(np.uint64).max)/" $(python -c "import importlib.util as u; print(u.find_spec(\'pycolmap\').submodule_search_locations[0])")/scene_manager.py')
tgz = glob.glob('/kaggle/input/**/gsplat_ext.tgz', recursive=True)                          # namma-prep's compiled gsplat
if tgz:
    sh(f'mkdir -p ~/.cache && tar xzf {tgz[0]} -C ~/.cache')
    sh('find $(python -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec(\'gsplat\').origin))") -exec touch -d "2000-01-01" {} +')
sh(f'python {F}/gsplat_warm.py', check=True)

# 1 the run's frames (frames.py makes the same names as the run did) + its cameras and model
import numpy as np, torch, frames
from PIL import Image
D = f'{W}/data'; inp = [d for d in glob.glob('/kaggle/input/**/model.ply', recursive=True)]; assert inp, 'model.ply not attached'
zs = glob.glob('/kaggle/input/**/sparse.zip', recursive=True)                                 # Kaggle may have unzipped it
if zs: shutil.unpack_archive(zs[0], D)                                                       # -> D/sparse/0/*.bin
else: shutil.copytree(os.path.dirname(glob.glob('/kaggle/input/**/sparse/**/cameras.bin', recursive=True)[0]), f'{D}/sparse/0')
FDS = CFG.get('frames_dataset', 'namma-dl3dv').split('/')[-1]                                # only the run's frames dataset:
for z in glob.glob(f'/kaggle/input/**/{FDS}/**/frames.zip', recursive=True): shutil.unpack_archive(z, f'{W}/fr')  # other inputs
cnt = {}                                                                                      # (namma-prep) hold other images
for f in [g for g in glob.glob('/kaggle/input/**/*', recursive=True) if f'/{FDS}/' in g] + glob.glob(f'{W}/fr/**/*', recursive=True):
    if f.lower().endswith(('.jpg', '.jpeg', '.png')) and '/sparse' not in f: cnt[os.path.dirname(f)] = cnt.get(os.path.dirname(f), 0) + 1
assert cnt, f'no frames found in the attached dataset {FDS}'
say('frames', frames.from_images(max(cnt, key=cnt.get), f'{D}/images', NFRAMES, LONG))
sys.path.insert(0, EX); from datasets.colmap import Parser
from gsplat import rasterization
from plyfile import PlyData
p = Parser(D, factor=1, normalize=True, test_every=8); C = p.camtoworlds; names = p.image_names
nf = len(glob.glob(f'{D}/images/*.jpg')); assert nf == len(names), f'{nf} frames but the run has {len(names)} cameras: wrong frames'
v = PlyData.read(inp[0])['vertex'].data; dev = 'cuda'
t = lambda *k: torch.tensor(np.stack([v[x] for x in k], 1), dtype=torch.float32, device=dev)
means, scales, quats = t('x', 'y', 'z'), torch.exp(t('scale_0', 'scale_1', 'scale_2')), t('rot_0', 'rot_1', 'rot_2', 'rot_3')
opac = torch.sigmoid(torch.tensor(np.asarray(v['opacity']), dtype=torch.float32, device=dev))
nr = len([n for n in v.dtype.names if n.startswith('f_rest_')]) // 3                         # exporter: shN (N,K,3) -> (N,3,K) flat
sh0 = t('f_dc_0', 'f_dc_1', 'f_dc_2')[:, None]
shN = t(*[f'f_rest_{i}' for i in range(3 * nr)]).reshape(-1, 3, nr).transpose(1, 2)
colors = torch.cat([sh0, shN], 1); shd = int(round(np.sqrt(colors.shape[1]))) - 1; say(len(means), 'splats, SH degree', shd)

def render(c2w, K):
    img = rasterization(means, quats, scales, opac, colors, torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=dev)[None],
                        torch.tensor(K, dtype=torch.float32, device=dev)[None], WO, HO, sh_degree=shd)[0]
    return Image.fromarray((img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))

cid = p.camera_ids[0]; Wf, Hf = p.imsize_dict[cid]; K = p.Ks_dict[cid].copy(); K[:2] *= WO / Wf
val = [i for i in range(len(names)) if i % 8 == 0]; trn = [i for i in range(len(names)) if i % 8]
span = np.linalg.norm(C[:, :3, 3] - C[:, :3, 3].mean(0), axis=1).max(); side = 0.15 * span    # ~50 cm (as in v3_qa)
def nearest(c2w, k=2):
    d = np.linalg.norm(C[trn, :3, 3] - c2w[:3, 3], axis=1) / span + (1 - C[trn, :3, 2] @ c2w[:3, 2])
    return [trn[j] for j in np.argsort(d)[:k]]
photo = lambda i: Image.open(p.image_paths[i]).convert('RGB').resize((WO, HO), Image.LANCZOS)
cases = []
for i in [val[len(val) // 3], val[2 * len(val) // 3]]:
    cases.append(dict(name=f'heldout_{names[i]}', render=render(C[i], K), refs=[photo(j) for j in nearest(C[i])], gt=photo(i)))
for n, i in enumerate([trn[len(trn) // 4], trn[len(trn) // 2], trn[3 * len(trn) // 4]]):
    c = C[i].copy(); c[:3, 3] += (1 if n % 2 == 0 else -1) * side * c[:3, 0]
    cases.append(dict(name=f'offpath_{names[i]}', render=render(c, K), refs=[photo(j) for j in nearest(c)], gt=None))
for c in cases: c['render'].save(f"{O}/{c['name']}_render.jpg", quality=92)
del means, scales, quats, opac, colors, sh0, shN; gc.collect(); torch.cuda.empty_cache(); say('renders done')

# 2 prompt embeddings: Qwen2.5-VL (7B) in 4 bit, then freed (the T4 cannot hold it next to the 20B transformer)
from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel, GGUFQuantizationConfig, AutoencoderKLQwenImage
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions, CONDITION_IMAGE_SIZE
import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as qp                    # each input image becomes ~4000
qp.VAE_IMAGE_SIZE = int(CFG.get('vae_area', 512 * 512))                                  # tokens at 1 MP; a T4 has no flash
                                                                                         # attention in bf16 -> 512^2 per input
emb = {}
for dt in (torch.float16, torch.bfloat16):
    try:
        t1 = time.time()
        te = Qwen2_5_VLForConditionalGeneration.from_pretrained(REPO, subfolder='text_encoder', torch_dtype=dt, device_map={'': 0},
              quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=dt))
        pe = QwenImageEditPlusPipeline.from_pretrained(REPO, transformer=None, vae=None, text_encoder=te, torch_dtype=dt)
        say('text encoder loaded', round(time.time() - t1), 's, GPU GB', round(torch.cuda.max_memory_allocated() / 1e9, 1))
        for c in cases:
            ims = [c['render']] + c['refs']; cond = []
            for im in ims:
                w, h = calculate_dimensions(CONDITION_IMAGE_SIZE, im.width / im.height); cond.append(pe.image_processor.resize(im, h, w))
            with torch.inference_mode(): e, m = pe.encode_prompt(prompt=PROMPT, image=cond, device='cuda:0')
            assert torch.isfinite(e).all(), 'non-finite prompt embeddings'
            emb[c['name']] = (e.cpu(), None if m is None else m.cpu())
        REP['text_dtype'] = str(dt); REP['text_s'] = round(time.time() - t1); del te, pe; break
    except Exception as ex:
        say('text encoder failed with', dt, type(ex).__name__, str(ex)[:300]); emb = {}
    finally:
        gc.collect(); torch.cuda.empty_cache()
assert emb, 'no prompt embeddings'; save_report()

# 3 the editor: 20B transformer as 4-bit GGUF + Lightning LoRA (4 steps, no CFG), on GPU 0
LIGHT = dict(base_image_seq_len=256, base_shift=np.log(3), invert_sigmas=False, max_image_seq_len=8192, max_shift=np.log(3),
             num_train_timesteps=1000, shift=1.0, shift_terminal=None, stochastic_sampling=False, time_shift_type='exponential',
             use_beta_sigmas=False, use_dynamic_shifting=True, use_exponential_sigmas=False, use_karras_sigmas=False)
import lpips; from skimage.metrics import structural_similarity
lp = lpips.LPIPS(net='alex').to('cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0')
def metrics(a, gt):
    a, g = np.asarray(a, np.float32) / 255, np.asarray(gt, np.float32) / 255
    A = np.c_[a.reshape(-1, 3), np.ones(a.size // 3)]; fit = np.clip(A @ np.linalg.lstsq(A, g.reshape(-1, 3), rcond=None)[0], 0, 1).reshape(g.shape)
    ps = lambda x: float(-10 * np.log10(np.mean((x - g) ** 2) + 1e-12))
    tt = lambda x: torch.tensor(x).permute(2, 0, 1)[None].to(lp.parameters().__next__().device) * 2 - 1
    return dict(psnr=round(ps(a), 2), psnr_exposure_fair=round(ps(fit), 2), ssim=round(float(structural_similarity(a, g, channel_axis=2, data_range=1)), 4),
                lpips=round(float(lp(tt(a), tt(g)).item()), 4))
def gm():
    """GPU memory in use per device (GB), for the log."""
    return ' '.join(f'GPU{i} {torch.cuda.memory_allocated(i) / 1e9:.1f}/{torch.cuda.max_memory_allocated(i) / 1e9:.1f}GB' for i in range(torch.cuda.device_count()))

def split_gpus(tr, frac=0.5):
    """The 20B transformer over both T4s by hand (GGUF weights move with .to(); accelerate's dispatcher rebuilds parameters):
    first `frac` of the blocks + the input layers on GPU 0 (it also holds the VAE), the rest + output layers on GPU 1; a
    pre-hook moves each module's inputs to its GPU, and the result goes back to GPU 0 for the scheduler."""
    def mover(dev):
        def mv(x):
            if torch.is_tensor(x): return x.to(dev)
            if isinstance(x, (tuple, list)): return type(x)(mv(y) for y in x)
            return x
        return lambda m, a, kw: (mv(a), {k_: mv(v) for k_, v in kw.items()})
    n = len(tr.transformer_blocks); k = int(n * frac)
    for i, b in enumerate(tr.transformer_blocks):
        dev = 'cuda:0' if i < k else 'cuda:1'; b.to(dev); b.register_forward_pre_hook(mover(dev), with_kwargs=True)
    for name, m in tr.named_children():
        if name == 'transformer_blocks': continue
        dev = 'cuda:1' if name in ('norm_out', 'proj_out') else 'cuda:0'; m.to(dev); m.register_forward_pre_hook(mover(dev), with_kwargs=True)
    def back(m, a, out):
        if isinstance(out, tuple): return (out[0].to('cuda:0'),) + tuple(out[1:])
        out.sample = out.sample.to('cuda:0'); return out
    tr.register_forward_hook(back)

done = False
SMALL = 'qwen-image-edit-2511-Q3_K_M.gguf'                                                   # 9.9 GB, if 13 GB + activations do not fit
for dt, gg, split in [(torch.bfloat16, SMALL, True), (torch.bfloat16, GGUF, True)]:       # fp16: blank/NaN; split: 13 GB > one T4
    try:
        t1 = time.time(); torch.cuda.reset_peak_memory_stats(); say('editor try', dt, gg, 'split' if split else '')
        tr = QwenImageTransformer2DModel.from_single_file(f'https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/blob/main/{gg}',
              quantization_config=GGUFQuantizationConfig(compute_dtype=dt), torch_dtype=dt, config=REPO, subfolder='transformer')
        if split and torch.cuda.device_count() > 1: split_gpus(tr); say('transformer split over 2 GPUs', gm())
        vae = AutoencoderKLQwenImage.from_pretrained(REPO, subfolder='vae', torch_dtype=dt)
        pipe = QwenImageEditPlusPipeline.from_pretrained(REPO, transformer=tr, vae=vae, text_encoder=None, torch_dtype=dt,
                                                         scheduler=FlowMatchEulerDiscreteScheduler.from_config(LIGHT))
        try:
            pipe.load_lora_weights('lightx2v/Qwen-Image-Edit-2511-Lightning', weight_name=LORA); REP['lora'] = LORA; steps = STEPS
            for n_, q_ in pipe.transformer.named_parameters():                                # LoRA ships bf16; compute may be fp16
                if 'lora' in n_ and q_.dtype != dt: q_.data = q_.data.to(dt)
            say('LoRA loaded', gm())
        except Exception as ex: say('LoRA failed:', type(ex).__name__, str(ex)[:300]); REP['lora'] = f'failed: {str(ex)[:200]}'; steps = 12
        if split and torch.cuda.device_count() > 1: pipe.vae.to('cuda:0')                  # transformer already placed
        else: pipe.to('cuda:0')
        REP['load_s'] = round(time.time() - t1); say('editor loaded', REP['load_s'], 's', gm())
        for c in cases:
            e, m = emb[c['name']]; t2 = time.time()
            with torch.inference_mode():
                out = pipe(image=[c['render']] + c['refs'], prompt_embeds=e.to('cuda:0', dt), prompt_embeds_mask=None if m is None else m.to('cuda:0'),
                           true_cfg_scale=1.0, num_inference_steps=steps, height=HO, width=WO, generator=torch.Generator('cuda').manual_seed(0)).images[0]
            a = np.asarray(out, np.float32)
            if not np.isfinite(a).all() or a.std() < 2: raise RuntimeError(f'blank/NaN output with {dt} (std {a.std():.2f})')
            out = out.resize((WO, HO)); out.save(f"{O}/{c['name']}_qwen.jpg", quality=92); c['qwen'] = out
            r = dict(name=c['name'], seconds=round(time.time() - t2, 1))
            if c['gt'] is not None: r.update(before=metrics(c['render'], c['gt']), after=metrics(out, c['gt']))
            REP['cases'].append(r); say(json.dumps(r), gm()); save_report()
        REP.update(edit_dtype=str(dt), gguf=gg, steps=steps, gpu_peak_gb=round(torch.cuda.max_memory_allocated(0) / 1e9, 1)); done = True; break
    except Exception as ex:
        say('editor failed with', dt, gg, type(ex).__name__, str(ex)[:400]); REP[f'editor_error_{dt}_{gg}'] = str(ex)[:400]; REP['cases'] = []
        for c in cases: c.pop('qwen', None)
    finally:
        for x in ('pipe', 'tr', 'vae'): globals().pop(x, None)
        gc.collect(); torch.cuda.empty_cache()
REP['ok'] = done; REP['minutes'] = round((time.time() - T0) / 60, 1); save_report()

# 4 sheet: refs (small) | render | Qwen | real (held-out only)
rows = []
for c in cases:
    refs = np.concatenate([np.asarray(r.resize((WO // 2, HO // 2))) for r in c['refs']], 0)
    tiles = [refs, np.asarray(c['render']), np.asarray(c.get('qwen', Image.new('RGB', (WO, HO))))] + ([np.asarray(c['gt'])] if c['gt'] else [np.zeros((HO, WO, 3), np.uint8)])
    rows.append(np.concatenate(tiles, 1))
Image.fromarray(np.concatenate(rows, 0)).save(f'{O}/qwen_sheet.jpg', quality=88)
say('QWEN DONE', json.dumps(REP)[:2000])
