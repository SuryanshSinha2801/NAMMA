"""Whole-home test video for the app pipeline (.github/workflows/demo-video.yml, settings in v3/DEMO_VIDEO.json): a multi-room
walk-through captured for radiance fields (Zip-NeRF's homes: handheld photos taken while walking through every room, in order),
joined into one video.mp4 and filed in the app inbox as a new job (status 'uploaded', preset from the settings). Like a phone upload,
it starts when its trigger v3/inbox/<job>.json is pushed. The job id is printed (no links here)."""
import glob, json, os, secrets, subprocess, sys, time
V = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, V)
cfg = json.load(open(f'{V}/DEMO_VIDEO.json'))
if not cfg.get('n'): sys.exit(0)
from huggingface_hub import snapshot_download
import imageio_ffmpeg, inbox
d = snapshot_download(cfg['src'], repo_type='dataset', allow_patterns=[f"{cfg['scene']}/images/*"], local_dir='src')
ims = sorted(glob.glob(f"{d}/{cfg['scene']}/images/*")); print(len(ims), 'photos', flush=True)
open('list.txt', 'w').write(''.join(f"file '{os.path.abspath(p)}'\nduration {1 / cfg['fps']}\n" for p in ims))
L = int(cfg.get('long', 1920))
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', 'list.txt', '-vf',
                f"scale='if(gt(iw,ih),{L},-2)':'if(gt(iw,ih),-2,{L})',fps={cfg['fps']},format=yuv420p", '-c:v', 'libx264', '-crf', '16', 'video.mp4'], check=True)
api, repo = inbox.hub(); job = time.strftime('%Y%m%d-%H%M%S', time.gmtime()) + '-' + secrets.token_hex(2)
api.upload_file(path_or_fileobj='video.mp4', path_in_repo=f'{job}/video.mp4', repo_id=repo, repo_type='dataset')
api.upload_file(path_or_fileobj=json.dumps(dict(name=f"demo {cfg['scene']}", phone='Zip-NeRF capture (photos joined)', preset=cfg.get('preset', 'best'),
                photos=len(ims), fps=cfg['fps'])).encode(), path_in_repo=f'{job}/meta.json', repo_id=repo, repo_type='dataset')
inbox.set_status(job, 'uploaded', note='demo whole-home video')
print('JOB', job, round(os.path.getsize('video.mp4') / 1e6), 'MB', flush=True)
