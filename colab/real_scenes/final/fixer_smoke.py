import base64, gc, glob, io, json, os, shutil, subprocess, sys, time
"""Kaggle side of the NVIDIA Fixer smoke test (v3/fixer_gh.py): fix the rendered tour views in the input dataset with Fixer
(github.com/nv-tlabs/Fixer, code Apache-2.0; weights hf nvidia/Fixer, NVIDIA Open Model License, commercial use allowed).
Fixer is made for NVIDIA's Cosmos container (Ampere+, transformer_engine, megatron); here it runs from the plain pip package
with small stand-ins for those two (final/fixer_shim, sent in JOB_CFG) and the attention in plain PyTorch, on a T4.
Writes fixed/<name>.png and fixer_report.json (precision that worked, seconds per view, GPU)."""
cfg = json.loads(os.environ.get('JOB_CFG', '{}')); T0 = time.time(); W = '/kaggle/working'; X = '/tmp/fx'; os.makedirs(X, exist_ok=True)
log = open(f'{W}/fixer.log', 'a')

def say(*a):
    m = ' '.join(str(x) for x in a); print(f'[{(time.time() - T0) / 60:5.1f} min]', m, flush=True); log.write(m + '\n'); log.flush()

def sh(c): r = subprocess.run(c, shell=True, executable='/bin/bash', capture_output=True, text=True); say(c[:90], '->', r.returncode, (r.stdout + r.stderr)[-600:]); return r.returncode

rep = dict(gpu=subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', shell=True, capture_output=True, text=True).stdout.strip())
try:
    import tarfile; tarfile.open(fileobj=io.BytesIO(base64.b64decode(cfg['shim']))).extractall(f'{X}/shim')
    sh('pip install -q --no-deps cosmos-predict2==1.0.9')
    sh('pip install -q attrs better-profanity boto3 einops ftfy fvcore hydra-core "imageio[ffmpeg]" iopath loguru mediapy nltk nvidia-ml-py '
       'omegaconf peft pycocotools pynvml pyyaml safetensors scikit-image sentencepiece termcolor webdataset natsort lpips qwen-vl-utils "transformers==4.51.3" "diffusers==0.33.1"')
    sh(f'git clone -q --depth 1 https://github.com/nv-tlabs/Fixer {X}/Fixer && sed -i "s#/work/models#{X}/models#g" {X}/Fixer/src/*.py')
    from huggingface_hub import snapshot_download
    snapshot_download('nvidia/Fixer', local_dir=f'{X}/models'); say('weights', os.listdir(f'{X}/models'))
    sys.path[:0] = [f'{X}/shim', f'{X}/Fixer/src']
    import numpy as np, torch
    from PIL import Image
    _load = torch.load; torch.load = lambda *a, **k: _load(*a, **{**k, 'weights_only': False})       # NVIDIA's own checkpoint
    from unittest.mock import MagicMock
    STUB = ('qwen_vl_utils', 'decord', 'retinaface', 'modelscope', 'better_profanity', 'mediapy', 'webdataset', 'boto3', 'natten')
    for m_ in ('imaginaire.models.vlm_qwen', 'imaginaire.networks.qwen2_5_vl'):    # Qwen text encoder (needs flash-attn 2): never used
        sys.modules[m_] = MagicMock(__path__=[], __spec__=None)
    for _ in range(25):                          # parts never used here (text encoder, guardrails, data loaders): stand-ins if missing
        try: import cosmos_predict2.pipelines.text2image as t2i; break
        except ModuleNotFoundError as e:
            if not (e.name or '').split('.')[0] in STUB: raise
            say('stand-in for', e.name); sys.modules[e.name] = MagicMock(__path__=[], __spec__=None)
            for k in [k for k in sys.modules if k.startswith(('cosmos_predict2', 'imaginaire'))]: sys.modules.pop(k)
    FC = t2i.Text2ImagePipeline.__dict__['from_config']; orig = FC.__func__                    # a staticmethod or classmethod
    torch.set_grad_enabled(False)
    ins = sorted(glob.glob('/kaggle/input/**/*.png', recursive=True)); say(len(ins), 'views'); assert ins, 'no input views under /kaggle/input'
    for prec in cfg.get('precisions', ['bfloat16', 'float16', 'float32']):
        DT = getattr(torch, prec)
        try:
            t2i.Text2ImagePipeline.from_config = (staticmethod(lambda *a, **k: orig(*a, **{**k, 'torch_dtype': DT})) if isinstance(FC, staticmethod)
                                                  else classmethod(lambda cls, *a, **k: orig(cls, *a, **{**k, 'torch_dtype': DT})))
            import pix2pix_turbo_nocond_cosmos_base_faster_tokenizer as fx
            fx.config.precision = prec
            m = fx.Pix2Pix_Turbo(pretrained_path=f'{X}/models/pretrained/pretrained_fixer.pkl', timestep=int(cfg.get('timestep', 250)),
                                 batch_size=1).to(device='cuda', dtype=DT); m.set_eval()
            from contextlib import nullcontext                            # the tokenizer keeps its own bf16 dtype/autocast: follow DT
            for mod in m.modules():
                if isinstance(getattr(mod, 'dtype', None), torch.dtype):
                    try: mod.dtype = DT
                    except Exception: pass
                if getattr(mod, 'is_amp', False) and hasattr(mod, 'context'):
                    mod.context = nullcontext() if prec == 'float32' else torch.amp.autocast('cuda', dtype=DT)
            if hasattr(m.unet, 'tensor_kwargs'): m.unet.tensor_kwargs['dtype'] = DT; m.unet.precision = DT
            out = f'{W}/fixed_{prec}'; os.makedirs(out, exist_ok=True); secs = []
            for p in ins:
                im = Image.open(p).convert('RGB'); size = im.size
                x = (torch.tensor(np.asarray(im.resize((1024, 576), Image.BILINEAR)), dtype=torch.float32).permute(2, 0, 1)[None] / 127.5 - 1).to('cuda', DT)
                torch.cuda.synchronize(); t = time.time()
                with torch.autocast('cuda', dtype=DT, enabled=prec != 'float32'):
                    y = m(x)
                torch.cuda.synchronize(); secs.append(time.time() - t)
                y = ((y.float()[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).byte().cpu().numpy()
                assert np.isfinite(y).all() and y.std() > 2, f'{prec}: blank/invalid output'
                Image.fromarray(y).resize(size, Image.BILINEAR).save(f'{out}/{os.path.basename(p)}')
            rep[prec] = dict(ok=True, sec_per_view=round(float(np.median(secs[1:] or secs)), 3), first_sec=round(secs[0], 2),
                             mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
            say(prec, rep[prec])
            if cfg.get('first_only', True): break
        except Exception as e:
            import traceback; rep[prec] = dict(ok=False, error=f'{type(e).__name__}: {str(e)[:400]}'); say(prec, 'failed:', traceback.format_exc()[-1500:])
        finally:
            for k in ('m', 'x', 'y'): globals().pop(k, None)
            sys.modules.pop('pix2pix_turbo_nocond_cosmos_base_faster_tokenizer', None); gc.collect(); torch.cuda.empty_cache()
except Exception as e:
    import traceback; rep['error'] = f'{type(e).__name__}: {str(e)[:400]}'; say('setup failed:', traceback.format_exc()[-2000:])
rep['minutes'] = round((time.time() - T0) / 60, 1); json.dump(rep, open(f'{W}/fixer_report.json', 'w'), indent=1); say('report', rep)
shutil.rmtree(f'{X}/models', ignore_errors=True)
