import gc, glob, json, os, shutil, subprocess, sys, time
"""Kaggle job anasrazy/namma-objects: the object layer (final/objects3d.py) on a finished run. Inputs (v3/objects_gh.py): the
dataset namma-objects-in (the run's model.ply + sparse.zip) and the run's own frames dataset (never other images); the run's level
matrix and metres-per-unit come in JOB_CFG. Out: objects.json, splat_object.npy, objects_sheet.jpg, crops/."""
T0 = time.time(); W = '/kaggle/temp/ob'; O = '/kaggle/working'; os.makedirs(W, exist_ok=True)
CFG = json.loads(os.environ.get('JOB_CFG', '{}'))
os.environ['HF_HOME'] = f'{W}/hf'; os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def sh(c, check=False):
    print(f'\n[{time.time()-T0:5.0f}s] $ {c[:160]}', flush=True); r = subprocess.call(c, shell=True, executable='/bin/bash')
    if check and r: raise RuntimeError(f'failed: {c[:80]}')
    return r

def say(*a): print(f'[{time.time()-T0:5.0f}s]', *a, flush=True)

sh(f'git clone -q --depth 1 -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA {W}/repo', check=True)
F = f'{W}/repo/colab/real_scenes/final'; EX = f'{W}/gsplat/examples'; sys.path.insert(0, F)
sh(f'git clone -q --depth 1 -b v1.5.3 https://github.com/nerfstudio-project/gsplat {W}/gsplat && touch {EX}/datasets/__init__.py; '
   f'echo "torch==$(python -c "import torch; print(torch.__version__)")" > {W}/pin.txt; '
   f'pip install -q -c {W}/pin.txt ninja gsplat==1.5.3 plyfile open3d scipy "imageio[ffmpeg]" accelerate 2>&1 | tail -3; '
   f'SAM2_BUILD_CUDA=0 pip install -q -c {W}/pin.txt "git+https://github.com/facebookresearch/sam2.git" 2>&1 | tail -3; '
   'pip install -q --no-deps "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e"; '
   'sed -i "s/np.uint64(-1)/np.uint64(np.iinfo(np.uint64).max)/" $(python -c "import importlib.util as u; print(u.find_spec(\'pycolmap\').submodule_search_locations[0])")/scene_manager.py')
tgz = glob.glob('/kaggle/input/**/gsplat_ext.tgz', recursive=True)                          # namma-prep's compiled gsplat
if tgz:
    sh(f'mkdir -p ~/.cache && tar xzf {tgz[0]} -C ~/.cache')
    sh('find $(python -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec(\'gsplat\').origin))") -exec touch -d "2000-01-01" {} +')
sh(f'python {F}/gsplat_warm.py', check=True)

import frames
D = f'{W}/data'; ply = glob.glob('/kaggle/input/**/namma-objects-in/**/model.ply', recursive=True); assert ply, 'model.ply not attached'
zs = glob.glob('/kaggle/input/**/namma-objects-in/**/sparse.zip', recursive=True)
if zs: shutil.unpack_archive(zs[0], D)
else: shutil.copytree(os.path.dirname(glob.glob('/kaggle/input/**/namma-objects-in/**/cameras.bin', recursive=True)[0]), f'{D}/sparse/0')
FDS = CFG.get('frames_dataset', 'namma-dl3dv').split('/')[-1]
for z in glob.glob(f'/kaggle/input/**/{FDS}/**/frames.zip', recursive=True): shutil.unpack_archive(z, f'{W}/fr')
cnt = {}
for f in [g for g in glob.glob('/kaggle/input/**/*', recursive=True) if f'/{FDS}/' in g] + glob.glob(f'{W}/fr/**/*', recursive=True):
    if f.lower().endswith(('.jpg', '.jpeg', '.png')) and '/sparse' not in f: cnt[os.path.dirname(f)] = cnt.get(os.path.dirname(f), 0) + 1
assert cnt, f'no frames found in the attached dataset {FDS}'
say('frames', frames.from_images(max(cnt, key=cnt.get), f'{D}/images', int(CFG.get('nframes', 420)), int(CFG.get('long', 1440))))
json.dump(dict(matrix=CFG['level']), open(f'{W}/level.json', 'w'))
r = sh(f'cd {EX} && PYTHONPATH={EX}:{F}:$PYTHONPATH python {F}/objects3d.py {D} {ply[0]} {W}/level.json {CFG["metres_per_unit"]} {O} '
       f'{int(CFG.get("frames", 40))}')
say('objects3d exit', r, 'after', round((time.time() - T0) / 60, 1), 'min'); sys.exit(r)
