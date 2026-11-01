"""NVIDIA Fixer (github.com/nv-tlabs/Fixer, code Apache-2.0; weights hf nvidia/Fixer, NVIDIA Open Model License, commercial use
allowed) outside NVIDIA's Cosmos container, on a Kaggle T4, as worked out by the smoke test (final/fixer_smoke.py, 30 Oct):
cosmos-predict2 1.0.9 from pip without its deps, stand-ins for transformer_engine/megatron (final/fixer_shim) and for the unused
text encoder, bf16 (2.9 s per 1024x576 view, 2.4 GB).
  fixer = load(work_dir, shim_dir, device)   ->   fixer(rgb uint8 HxWx3) -> uint8 HxWx3 (same size)"""
import os, subprocess, sys


def setup(X, say=print):
    def sh(c):
        r = subprocess.run(c, shell=True, executable='/bin/bash', capture_output=True, text=True); say(c[:80], '->', r.returncode, (r.stdout + r.stderr)[-300:])
    sh('pip install -q --no-deps cosmos-predict2==1.0.9')
    sh('pip install -q attrs better-profanity boto3 einops ftfy fvcore hydra-core "imageio[ffmpeg]" iopath loguru mediapy nltk nvidia-ml-py '
       'omegaconf peft pycocotools pynvml pyyaml safetensors scikit-image sentencepiece termcolor webdataset natsort lpips qwen-vl-utils "transformers==4.51.3" "diffusers==0.33.1"')
    sh(f'git clone -q --depth 1 https://github.com/nv-tlabs/Fixer {X}/Fixer && sed -i "s#/work/models#{X}/models#g" {X}/Fixer/src/*.py')
    from huggingface_hub import snapshot_download
    snapshot_download('nvidia/Fixer', local_dir=f'{X}/models', allow_patterns=['base/*', 'pretrained/*'])


def load(X, shim, device='cuda', prec='bfloat16', timestep=250, say=print):
    sys.path[:0] = [shim, f'{X}/Fixer/src']
    import numpy as np, torch
    from contextlib import nullcontext
    from unittest.mock import MagicMock
    from PIL import Image
    _load = torch.load; torch.load = lambda *a, **k: _load(*a, **{**k, 'weights_only': False})       # NVIDIA's own checkpoint
    for m_ in ('imaginaire.models.vlm_qwen', 'imaginaire.networks.qwen2_5_vl'):    # Qwen text encoder (needs flash-attn 2): never used
        sys.modules[m_] = MagicMock(__path__=[], __spec__=None)
    STUB = ('qwen_vl_utils', 'decord', 'retinaface', 'modelscope', 'better_profanity', 'mediapy', 'webdataset', 'boto3', 'natten')
    for _ in range(25):
        try: import cosmos_predict2.pipelines.text2image as t2i; break
        except ModuleNotFoundError as e:
            if (e.name or '').split('.')[0] not in STUB: raise
            sys.modules[e.name] = MagicMock(__path__=[], __spec__=None)
            for k in [k for k in sys.modules if k.startswith(('cosmos_predict2', 'imaginaire'))]: sys.modules.pop(k)
    DT = getattr(torch, prec); FC = t2i.Text2ImagePipeline.__dict__['from_config']; orig = FC.__func__
    t2i.Text2ImagePipeline.from_config = (staticmethod(lambda *a, **k: orig(*a, **{**k, 'torch_dtype': DT})) if isinstance(FC, staticmethod)
                                          else classmethod(lambda cls, *a, **k: orig(cls, *a, **{**k, 'torch_dtype': DT})))
    with torch.cuda.device(device):                                               # Fixer's code says plain "cuda": its own GPU
        import pix2pix_turbo_nocond_cosmos_base_faster_tokenizer as fx
        fx.config.precision = prec
        m = fx.Pix2Pix_Turbo(pretrained_path=f'{X}/models/pretrained/pretrained_fixer.pkl', timestep=timestep, batch_size=1).to(device=device, dtype=DT)
    m.set_eval(); torch.load = _load
    for mod in m.modules():                                       # the tokenizer keeps its own bf16 dtype/autocast: follow DT
        if isinstance(getattr(mod, 'dtype', None), torch.dtype):
            try: mod.dtype = DT
            except Exception: pass
        if getattr(mod, 'is_amp', False) and hasattr(mod, 'context'):
            mod.context = nullcontext() if prec == 'float32' else torch.amp.autocast('cuda', dtype=DT)
    if hasattr(m.unet, 'tensor_kwargs'): m.unet.tensor_kwargs.update(dtype=DT, device=device); m.unet.precision = DT
    say('Fixer ready on', device, prec)

    @torch.no_grad()
    def fix(rgb):
        h, w = rgb.shape[:2]
        x = (torch.tensor(np.asarray(Image.fromarray(rgb).resize((1024, 576), Image.BILINEAR)), dtype=torch.float32).permute(2, 0, 1)[None] / 127.5 - 1).to(device, DT)
        with torch.cuda.device(device), torch.autocast('cuda', dtype=DT, enabled=prec != 'float32'):
            y = m(x)
        y = ((y.float()[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).byte().cpu().numpy()
        return np.asarray(Image.fromarray(y).resize((w, h), Image.BILINEAR))
    return fix
