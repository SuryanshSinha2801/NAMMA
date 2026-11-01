> **Superseded**: the current hand-off is `docs/HANDOFF_V3.md` (v3 pipeline, 26 Oct). This file is kept for history.

# NAMMA: hand-off from the cloud session (2026-10-24 → 25)

Read with: `docs/HANDOFF.md` (Mac session, 24 Oct), `docs/VIDEO_METHOD_PLAN.md` (plan), `colab/real_scenes/final/NOTES.md`
(measured findings, newest first). Branch: **`claude/wonderful-cori-i3fn41`** (based on `claude/project-thread-l70z4u`).

## 1. Goal and owner rules (unchanged)
Phone video → photoreal, walkable, true-to-room 3D (real estate). Generation may only fill gaps, never replace or invent what was
filmed, and generated content must be flagged. Free GPUs only (Kaggle T4). No long per-scene training "that never gets photoreal".
Owner: few tokens/time, no questions, act on the plan, measure with proof. Owner sleeps between jobs: **give an alarm time (IST)
with every job**; use `run_and_alarm.sh` so the Mac rings when a job ends.

## 2. Where things are
| What | Where |
|---|---|
| One-shot Kaggle job: video → frames → DA3 → SfM cameras → fused cloud → floor plan/nav → viewer → splat fits | `colab/real_scenes/final/run.py` (kernel `anasrazy/namma-final`, `kernel-metadata.json` in the same folder) |
| Depth Anything 3 chunked inference + Sim3 merge of chunks | `final/chunk_align.py` (+ `test_chunk_align.py`) |
| SfM cameras (hloc SuperPoint+LightGlue, sequential 10 + NetVLAD 15 pairs, one PINHOLE camera), runs in a py3.10 venv | `final/sfm_hloc.py` |
| DA3 depth put on the SfM cameras (metric scale, per-frame scale fix) | `final/sfm_align.py` |
| Point cloud fusion + floor/walkable grid/A* + camera-based up | `final/fuse_nav.py`, `experiments/nav_from_mesh/nav_from_mesh.py` |
| gsplat trainer (Default/MCMC, SH 0-3, depth loss, camera refinement option, held-out eval, diagnostics, y-up PLY with SH) | `final/train_splat.py` |
| Self-contained WebGL walk viewer (collision from walk grid, minimap) | `final/build_viewer.py` → `model/viewer.html` |
| 360° panoramas from the REAL cloud, holes kept as holes + mask + "real share" | `final/make_pano.py` (CPU, run on the Mac) |
| Push job, keep Mac awake, download, ring until Ctrl-C | `final/run_and_alarm.sh` |
| Matrix-3D job (paused, see §4) | `colab/real_scenes/matrix3d/kaggle/run.py` (utils3d pin added, not yet run) |

Run (Mac): `cd ~/NAMMA && rm -rf src && git clone -q -b claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA src && bash src/colab/real_scenes/final/run_and_alarm.sh src/colab/real_scenes/final anasrazy/namma-final ~/NAMMA/final_out`
Output: `~/NAMMA/final_out/namma-final.log` + `model/` (cloud.ply/npz, walk.npz, nav.json, floorplan.png, viewer.html, splat*.ply,
*_heldout*.jpg, *_step0.jpg, *_final.jpg). Old output is moved to `final_out_prev_HHMM`.

## 3. What ran (all on the 22 Oct 4K room video, Kaggle T4), measured
| Run | Setup | Result |
|---|---|---|
| CPU nav test | collision meshes in `walk_demo` | lobby 15.2 m², icl 14.4 m² walkable; phone_room failed (0.99) |
| final v1 | DA3 on 8 frames (60 OOMs a T4; 15 still OOM after the first OOM until memory was freed outside `except`) | ran end to end in 247 s |
| final v2 | DA3 chunked 8 views/3 shared, Sim3 merge, 60 frames | all 60 merged; owner screenshot: recognisable room as a grainy point cloud |
| splat v1 | 60 frames, 3000 steps, 332k splats | train-view 25.1 dB; owner: "better than points, still blurry, broken" |
| A/B 200 frames | A plain / B + camera refinement + depth loss | A train 19.3, held-out 17.2; B train 24.7, cameras moved **7.6 cm / 2.6°** → DA3 chained cameras drift |
| SfM run | hloc on 200 frames | **195/200 registered, 1.07 px**; DA3 per-frame scale wandered **33-47%** vs SfM |
| SfM vs DA3 cams (both with refinement) | held-out after aligning held-out cams | 22.1 vs 22.4 dB; raw 17.1 both → refinement co-drifts with the scene |
| V1 SfM, no refinement | 841k splats | held-out 18.8 dB / SSIM 0.78, but **held-out renders are fog** (owner images): PSNR hides it on white walls |
| V2 2M MCMC + SH3 | twice | broken (15.3, then 12.6 dB) — bugs: lr scaled by point extent, init opacity, densify before optimizer step (all fixed) |
| floor plan | every run | walkable collapsed to 0.1-0.6 m²: "up" from surface normals picked a wall (25-88° off); now from camera x axes |
| Matrix-3D | `namma-matrix3d` | pano + 33 GB checkpoints OK; MoGe failed (`utils3d.numpy.icosahedron` gone in utils3d master) |
| Panorama | SDXL-filled pano from 22 DA3 frames | **owner: invented a yellow door and an AC (it's a fan); not the room.** Do not feed to Matrix-3D. |

## 4. Decisions and lessons
1. Cameras must come from SfM (hloc SP+LG works on this video). DA3 only for dense metric depth, aligned per frame to SfM.
2. Don't refine cameras on top of good SfM cameras without anchoring (they co-drift with the scene).
3. Always look at held-out renders; PSNR on white-wall rooms is misleading.
4. Floor "up" from the phone cameras' x axes (no roll), not from normals or mean image-up (pitch bias).
5. Generation: follow the video's method (`docs/VIDEO_METHOD_PLAN.md`) but on our real reconstruction: geometry first, render
   drone paths, re-project REAL 4K frames, generate only the remaining holes with a video model, flag generated pixels, owner checks
   any generated pano/video against the real video. The SDXL pano and Matrix-3D on it are paused.
6. Splat count matters (owner: a 2M-splat "berry" scene looks real) but MCMC must first train correctly.

## 5. Running now (started 25 Oct ~midday IST, ~75 min)
`run.py` at the latest commit: 600 frames (SfM on all, DA3 on every 3rd), camera-x-axis up, optimizer-before-densify,
W1 = 600 views plain (`splat.ply`), W2 = + strong depth anchor that never fades (`splat_depth.ply`). Diagnostics per variant:
`*_step0.jpg` (init cloud seen from 2 train + 2 held-out cameras: geometry/camera check), `*_final.jpg`, log line with splat size
percentiles, % splats > 20 cm, mean opacity, held-out depth error vs the depth prior.
How to read it: step0 looks like the room → geometry right, training makes the fog (floaters/oversized splats: look at size stats).
step0 does not look like the room → bug before training (alignment/intrinsics).

## 6. Next steps (in order)
0. Pushed AFTER the running job started (so it is in the NEXT run): our own opacity reset every 3000 steps (gsplat 1.5.3's never
   fires: precedence bug), 30 min training cap for W1/W2. If fog persists after that, run splatfacto on the same SfM cameras as a check.
1. Read the run above; fix whatever step0/final/stats point at. Get a plain fit whose held-out renders show the fan/window/door.
2. Then 1-2M splats (MCMC) + SH3 for reflections; then train at 2016 px (E6).
3. Drone paths + real-frame re-projection (E4 in the plan), then hole-only video fill (E5), flagged.
4. Floor plan/collision from the fixed geometry; viewer with "not filmed" overlay; per-room later.
5. Techfest R1 (31 Oct): abstract PDF, hosted viewer, 3-5 min video, repo with sample data — not started.

## 7. Gotchas solved in this session
- Kaggle input path is `/kaggle/input/datasets/<user>/<dataset>/...` (find files recursively).
- Kaggle Python 3.13.15: DA3 needs `--no-deps --ignore-requires-python`; xformers wheel missing (not needed); DA3 pulls pycolmap/evo.
- DA3 OOM retries only work if CUDA memory is freed outside the `except` block.
- gsplat has no wheel for torch 2.11/cu128: `pip install gsplat` JIT-builds in ~5 min on first rasterization (exclude from timing).
- hloc/LightGlue: separate `uv venv -p 3.10` with torch 2.1.2 cu118, recursive hloc clone; pycolmap 4.2.1 (`cam_from_world` may be a method).
- Open3D TSDF returned empty meshes in the cloud container; fusion uses back-projected points instead.
- One failed Kaggle start (status ERROR in 1 min, empty log, quota fine): just re-push.
- Cloud container: no GPU, no Hugging Face, no Kaggle; PyPI and GitHub raw work. Only the Mac drives Kaggle.
