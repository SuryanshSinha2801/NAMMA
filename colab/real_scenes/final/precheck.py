"""Whole-house pre-check (plan of 30 Oct; run.py PREP_ONLY=1): BEFORE any splat training, prove the run will finish within
Kaggle's limits at run-4 quality. Runs on Kaggle after frames + cameras, in the training Python (gsplat ready).
  1 rooms (rooms.py split) -> chunks of run-4 size (<= MAX_CHUNK own frames; + frames from other rooms that see into them)
  2 GPU benchmarks, 700 steps each, splats grown to the planned count within ~500 steps (x1.05 every 5 steps):
      A  whole house as ONE model on both T4s (gsplat's Grendel mode: half the splats per GPU, 2 views a step), packed + visible_adam
      B  one T4 at run 4's 1.5M cap (one chunk)
  3 gate: decode, cameras in one piece, memory, time per Kaggle session (<= 9 h) and per week (<= 27 h incl. this pre-check)
Run 4 (the reference): 411 frames (359 train views), 30k steps = 83.6 steps per train view, 1.5M splats for ~37 m2,
5445 s of training = 0.181 s/step, 2.2 GB GPU memory.
  python precheck.py <gsplat examples> <data dir (images/ + sparse/0)> <work dir> <out dir> <timings json> [fused-ssim shim dir]"""
import json, math, os, re, subprocess, sys, threading, time
import numpy as np

EX = D = W = O = PP = ''; TIM = {}                                                          # set by setup() (run.py imports bench)
F = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, F)


def setup(ex, d, w, o, tim=None, pp=''):
    global EX, D, W, O, TIM, PP
    EX, D, W, O, TIM, PP = ex, d, w, o, tim or {}, pp
R4 = dict(frames=411, train_views=359, steps=30000, splats=1_500_000, area_m2=37.0, s_per_step=0.1815, mem_gb=2.2)
STEPS_PER_VIEW = R4['steps'] / R4['train_views']
MAX_CHUNK, SESSION_H, WEEK_H, MEM_GB, MARGIN, SETUP_H = 450, 9.0, 27.0, 13.0, 1.2, 0.5
T0 = time.time()


def say(*a): print(f'[precheck {time.time() - T0:6.0f}s]', *a, flush=True)


def chunks_from_rooms(rooms, order):
    """Rooms in walk order, grouped while their own frames fit MAX_CHUNK; each chunk also trains on the frames of other rooms
    that look into it (rooms.py 'extra'). A single room larger than MAX_CHUNK is its own chunk (flagged)."""
    out, cur = [], []
    own = lambda rs: sum(len(rooms[i]['cameras']) for i in rs)
    for i in order:
        if cur and own(cur + [i]) > MAX_CHUNK: out.append(cur); cur = []
        cur.append(i)
    if cur: out.append(cur)
    res = []
    for rs in out:
        cams = sorted({c for i in rs for c in rooms[i]['cameras']}); extra = sorted({c for i in rs for c in rooms[i]['extra']} - set(cams))
        res.append(dict(rooms=[rooms[i]['id'] for i in rs], own_frames=len(cams), frames=cams + extra, n_frames=len(cams) + len(extra),
                        area_m2=round(sum(rooms[i]['area_m2'] for i in rs), 1), too_big=len(cams) > MAX_CHUNK))
    return res


DEPTH = '--depth-loss'                                                                         # as run.py: off when a frame has no SfM points


def bench(name, gpus, cap, extra='', steps=700):
    """700 steps of run 4's trainer settings, splats grown to `cap` (per GPU) by step ~500: s/step over the last 200 steps
    (at full size), peak memory per GPU (nvidia-smi every 2 s), splat count reached."""
    res = f'{W}/bench_{name}'; log = f'{res}.log'; big = 10 ** 7
    cmd = (f'cd {EX} && CUDA_VISIBLE_DEVICES={gpus} PYTHONPATH={PP}:$PYTHONPATH python simple_trainer.py mcmc --data-dir {D} --data-factor 1 '
           f'--result-dir {res} --max-steps {steps} --eval-steps {big} --save-steps {big} --ply-steps {big} --disable-viewer '
           f'--use-bilateral-grid {DEPTH} --strategy.cap-max {cap} --strategy.refine-start-iter 20 --strategy.refine-every 5 '
           f'--strategy.refine-stop-iter 500 {extra}')
    say(name, cmd); peak = {}; marks = []; stop = threading.Event()

    def watch():
        while not stop.is_set():
            r = subprocess.run('nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits', shell=True, capture_output=True, text=True)
            for ln in r.stdout.strip().splitlines():
                g, m = [int(x) for x in ln.split(',')]; peak[g] = max(peak.get(g, 0), m)
            try:
                s = re.findall(rf'(\d+)/{steps} \[', open(log, errors='ignore').read()[-4000:])
                if s: marks.append((time.time(), int(s[-1])))
            except FileNotFoundError: pass
            time.sleep(2)

    th = threading.Thread(target=watch, daemon=True); th.start(); t = time.time()
    p = subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=open(log, 'w'), stderr=subprocess.STDOUT)
    try: rc = p.wait(timeout=40 * 60)
    except subprocess.TimeoutExpired: p.kill(); rc = 'timeout (40 min)'
    stop.set(); th.join()
    txt = open(log, errors='ignore').read(); n = [int(x) for x in re.findall(r'Now having (\d+) GSs', txt)]
    a = [m for m in marks if m[1] >= 500]; sps = (a[-1][0] - a[0][0]) / (a[-1][1] - a[0][1]) if len(a) > 1 and a[-1][1] > a[0][1] else None
    out = dict(ok=rc == 0, exit=rc, minutes=round((time.time() - t) / 60, 1), s_per_step=round(sps, 4) if sps else None,
               peak_mem_gb={g: round(m / 1024, 2) for g, m in sorted(peak.items())}, splats_reached_per_gpu=max(n) if n else None, cap_per_gpu=cap)
    if rc != 0: out['log_tail'] = [x for x in txt.replace('\r', '\n').split('\n') if x.strip()][-15:]
    say(name, json.dumps(out)[:600]); return out


def main():
    global DEPTH
    rep = json.load(open(f'{W}/sfm/sfm_report.json')); P = dict(timings=TIM, sfm=rep, reference_run4=R4)
    if rep.get('frames_without_points'): DEPTH = ''
    import rooms
    names, C, R, Pts, obs = rooms.load(f'{D}/sparse/0')
    meta, rms, order, (reg, wallm, c2, lo) = rooms.split(names, C, R, Pts, obs)
    json.dump(dict(meta, order=order, rooms=rms), open(f'{O}/rooms.json', 'w')); rooms.draw(f'{O}/rooms.png', reg, wallm, c2, lo, rms)
    ch = chunks_from_rooms(rms, order)
    for c in ch: c['frame_names'] = [names[i] for i in c.pop('frames')]
    json.dump(ch, open(f'{O}/chunks.json', 'w'))
    area = sum(r['area_m2'] for r in rms); reg_n = rep['registered']; train_views = reg_n - math.ceil(reg_n / 8)   # every 8th held out
    splats = int(max(R4['splats'] * area / R4['area_m2'], R4['splats'] * reg_n / R4['frames']))                 # run-4 density
    P['house'] = dict(rooms=len(rms), area_m2=round(area, 1), registered=reg_n, train_views=train_views, planned_splats=splats,
                      chunks=[{k: v for k, v in c.items() if k != 'frame_names'} for c in ch])
    say('house', json.dumps(P['house'])[:800])

    import torch; ngpu = torch.cuda.device_count()
    A = bench('A_one_model_2gpu', '0,1', math.ceil(splats / 2), '--packed --visible-adam') if ngpu >= 2 else dict(ok=False, exit='one GPU only')
    if not A['ok'] and ngpu >= 2:                                                              # visible_adam with MCMC may not mix
        A = dict(bench('A_one_model_2gpu_packed', '0,1', math.ceil(splats / 2), '--packed'), note='without visible_adam')
    B = bench('B_chunk_1gpu', '0', R4['splats'])
    P['bench'] = dict(A=A, B=B)

    # gate
    g = {}; planned = TIM.get('frames_planned') or reg_n
    g['decode'] = dict(ok=TIM.get('frames', 0) >= 0.95 * planned, frames=TIM.get('frames'), planned=planned)
    g['cameras'] = dict(ok=reg_n >= 0.9 * rep['frames'], registered=reg_n, frames=rep['frames'], models=rep.get('models'))
    view_steps = STEPS_PER_VIEW * train_views                                                     # run-4 steps per train view
    if A['ok'] and A['s_per_step']:
        steps = math.ceil(view_steps / 2); h = steps * A['s_per_step'] * MARGIN / 3600 + SETUP_H; mem = max(A['peak_mem_gb'].values())
        g['A_one_model'] = dict(ok=h <= SESSION_H and mem <= MEM_GB and (A['splats_reached_per_gpu'] or 0) >= 0.9 * A['cap_per_gpu'],
                                steps=steps, hours=round(h, 2), peak_mem_gb=mem, sessions=1)
    else: g['A_one_model'] = dict(ok=False, reason='benchmark did not run (see bench.A)')
    if B['ok'] and B['s_per_step']:
        jobs = sorted(((STEPS_PER_VIEW * (c['n_frames'] - math.ceil(c['n_frames'] / 8)) * B['s_per_step'] * MARGIN / 3600) for c in ch), reverse=True)
        sessions = []                                                                              # 2 GPUs a session, <= SESSION_H
        for j in jobs:
            for s in sessions:
                k = min(range(2), key=lambda q: s[q])
                if s[k] + j <= SESSION_H - SETUP_H: s[k] += j; break
            else: sessions.append([j, 0.0])
        hrs = sum(max(s) + SETUP_H for s in sessions); mem = max(B['peak_mem_gb'].values())
        g['B_chunks'] = dict(ok=all(not c['too_big'] for c in ch) and max(jobs) <= SESSION_H - SETUP_H and mem <= MEM_GB,
                             chunks=len(ch), sessions=len(sessions), hours=round(hrs, 2), longest_chunk_h=round(max(jobs), 2), peak_mem_gb=mem)
    else: g['B_chunks'] = dict(ok=False, reason='benchmark did not run (see bench.B)')
    prep_h = (time.time() - T0) / 3600 + TIM.get('hours_before_precheck', 0)
    pick = 'A_one_model' if g['A_one_model']['ok'] else 'B_chunks' if g['B_chunks']['ok'] else None
    week = prep_h + (g[pick]['hours'] if pick else 0)
    g['week'] = dict(ok=pick is not None and week <= WEEK_H, hours_incl_precheck=round(week, 2), limit=WEEK_H)
    P['gate'] = g; P['decision'] = pick if (g['decode']['ok'] and g['cameras']['ok'] and g['week']['ok']) else 'STOP'
    P['minutes'] = round((time.time() - T0) / 60, 1)
    json.dump(P, open(f'{O}/precheck.json', 'w'), indent=1, default=str)
    say('GATE', json.dumps(g)); say('DECISION', P['decision'])


if __name__ == '__main__':
    setup(*sys.argv[1:5], json.loads(sys.argv[5]), sys.argv[6] if len(sys.argv) > 6 else '')
    main()
