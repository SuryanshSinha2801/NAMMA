> **Superseded**: the current hand-off is `docs/HANDOFF_V3.md` (v3 pipeline, 26 Oct). This file is kept for history.

# NAMMA Space: hand-off for parallel work (2026-10-24)

## Target
Phone video (no LiDAR, no special hardware) of a real indoor space becomes a **photoreal, walkable 3D model**:
clean from every angle, including views off the path the phone was carried along, with collision, floor plan and
navigation, produced fast and without hand work. Real-estate use, so the model must be **true to the real room**:
generation may only fill gaps, never replace what was filmed. Later: multi-room and multi-floor.

## Where everything is

| What | Where |
|---|---|
| Code (all work below) | github.com/RazyAnas/NAMMA, branch **`claude/project-thread-l70z4u`** |
| 4K test video (22 Oct, 3 min 45 s, 2160x3840, 30 fps, camera-only recording) | Mac: `/Users/mariyafatima/NAMMA/namma-camera-2026-10-22T10-13-20.zip` (`video.h264` inside, 1.2 GB) |
| Same video as MP4 on Kaggle (private dataset) | `anasrazy/namma-room-2026-10-22` → `/kaggle/input/namma-room-2026-10-22/video.mp4` |
| 30 s lobby clip used for the tests | repo `colab/real_scenes/videos/lobby_clip30.mp4` (+ 180 frames in `colab/real_scenes/frames/lobby_clip30`) |
| Room splat, Nerfstudio 30k (soft, see results) | Mac: `/Users/mariyafatima/NAMMA/room_2026-10-22_splat.ply` (= `~/Downloads/results (1).zip`) |
| Kaggle account | `anasrazy`; CLI token on the Mac at `~/.kaggle/access_token` (`kaggle kernels push/status/output`) |
| Recorder web app | repo `app/static/record/` (published copy: repo `RazyAnas/namma-space-pages`, edited by another session) |
| Render server job (Modal / Colab / laptop) | repo `app/pipeline/stages/phone.py` (`final_plan`, task `splats.final`) |
| Research notes | repo `docs/SPLATTOUR_RESEARCH.md`, `docs/WHY_LOBBY_FAILED.md` |
| Result pages | Nerfstudio runs: claude.ai/artifact/6Rj1BWscw6KqFbmqpTG2zr · walkable lobby: claude.ai/artifact/1fcs8dFCoa97yhuGjJYJVT |

## What is proven (measured)

| Run | Input | Held-out PSNR / SSIM / LPIPS | Note |
|---|---|---|---|
| Nerfstudio defaults | whole lobby, 1 frame/s | 21.0 / 0.887 / 0.48 | fog, floaters |
| Same, sharp 30 s clip, sequential matching | 101/180 frames placed | 22.8 / 0.904 / 0.28 | |
| Same clip, exhaustive SuperPoint+LightGlue | 173/180 | 24.95 / 0.914 / 0.28 | |
| Full pipeline, 30k steps | 173/180 | **26.9 / 0.923 / 0.24** | best so far; doors/fire-hose near photo |
| 4K room video, render-server mode | 574/578 cameras | (no hold-out) | geometry right, surfaces mush: only 317k splats for 3+ min of walking |
| **Depth Anything 3 (feed-forward, no training)** | 22 frames at 540p, HF Space | | **61 s**, clean coherent room, softer texture |

Lessons: sharp dense frames are the biggest lever; SuperPoint+LightGlue solves what SIFT/ARCore could not; deleting
"floaters" after training lowers quality; plain white walls fail with every trainer; off-path views break beyond
~1-3% of the path length; one model for a whole floor is too thin (Splat Tour trains one splat per room).

## Plan (each step is independent enough to run in parallel)

1. **Feed-forward 3D, no training** (in progress): WorldMirror 2.0 (Tencent HY-World 2.0, `tencent/HY-World-2.0`) and
   Depth Anything 3 on the 4K room frames: cameras + depth + Gaussians in one pass. Also MapAnything (Meta) for metric
   scale. Output: a clean, gap-free base model in minutes.
2. **Fill only the gaps with generation**: render views off the walked path from step 1, let a video/image model fill
   just the holes (NVIDIA Difix3D+, or Skywork Matrix-3D's Wan 2.1 LoRA), guided by the real geometry.
3. **Keep filmed pixels true**: re-project the real 4K frames onto those views (the "8K re-projection" trick from the
   Matrix-3D ComfyUI video) so everything filmed stays exact.
4. **Short splat fit** (5-10 min, not 2 h) on real + filled views, starting from step 1. Then splat → mesh/TSDF for collision,
   floor plan and walkable area (like Splat Tour's navigability grid).
5. **Per room**: split the walk into rooms (camera position + time), one model per room plus an overview, streamed by
   room in the viewer (PlayCanvas, SOG format), as Splat Tour does. Multi-floor = same per floor.
6. **Viewer**: walk limit around the filmed path, room streaming, 360° backdrop, keep the capture's aspect.

## How to run things

- Kaggle background job (no notebook babysitting): a folder with `run.py` + `kernel-metadata.json`
  (`enable_gpu`, `enable_internet`, `machine_shape: NvidiaTeslaT4`, `dataset_sources: ["anasrazy/namma-room-2026-10-22"]`),
  then `kaggle kernels push -p <folder>`, `kaggle kernels status anasrazy/<name>`, `kaggle kernels output anasrazy/<name> -p <dir>`.
  Max 2 GPU jobs at once; stop extra sessions at kaggle.com → View Active Events.
- Full Nerfstudio pipeline: `colab/real_scenes/scripts/full_pipeline.sh` (env: `VIDEO`/`FRAMES`, `FPS`, `STEPS`, `FINAL=1`,
  `METHOD`, `TRAIN_ARGS`, `DATA_ARGS`, `WORK`, `OUT`). Kernel templates: `colab/real_scenes/kaggle_full/`.
- Local fast checks without a GPU: `colab/real_scenes/scripts/splat_cpu.py` (CPU splat renderer, 0.5 s/view, matches gsplat).
- Feed-forward demo without any GPU: Hugging Face Space `depth-anything/depth-anything-3` via `gradio_client`
  (`/handle_uploads` then `/gradio_demo` with `infer_gs=True`, `num_max_points` ≤ 100000).

## Status at 2026-10-24 17:15 UTC (latest; read this first)

Direction from the owner: **no long per-scene splat training** (it never gets photoreal). Build the room with feed-forward
3D + generative world models (Matrix-3D style), free GPUs only (Kaggle / Colab, no paid GPUs), quick results.

| Job / test | State | Where |
|---|---|---|
| `namma-room-render-mcmc` (Nerfstudio splatfacto-big) | **stopped on purpose**, don't restart | — |
| `namma-worldmirror2` (Tencent WorldMirror 2.0) | **dropped**: v5 "No available kernel" (attention), v6 died with an empty log (out of memory on T4). Depth Anything 3 does the same job in 61 s | repo `colab/real_scenes/kaggle_worldmirror/` |
| Difix3D+ repair test (NVIDIA, HF Space `georgeniu/difix`, `/restore_image`, ~70 s/image, no training) | done: fixes soft/smeared renders into a believable room, cannot rescue garbage, so it needs a decent base | repo `colab/real_scenes/matrix3d/results/difix_compare.jpg`, `difix_best.png` |
| **360° panorama of the real room** | done, looks good: Depth Anything 3 points rendered as an equirectangular image from the camera-path centre (55% unseen) + SDXL inpainting of only the unseen pixels, real pixels pasted back | repo `colab/real_scenes/matrix3d/results/pano_raw.jpg` (real only), `pano_img.jpg` (filled) |
| **`anasrazy/namma-matrix3d` v3** (Skywork Matrix-3D, 5B 720p, low-VRAM, 2×T4) | **running** since 17:15 UTC, ~1 h: pano → 360° walk-through video `pano_video.mp4` | kaggle.com/code/anasrazy/namma-matrix3d; job source repo `colab/real_scenes/matrix3d/kaggle/` |

Matrix-3D facts:
- Repo github.com/SkyworkAI/Matrix-3D. Steps: (1) image → panorama (uses FLUX.1-dev, gated, 24 GB: **skipped**, we make the pano
  from the real room instead), (2) `torchrun code/panoramic_image_to_video.py --inout_dir=<dir with pano_img.jpg + prompt.txt>
  --resolution=720 --use_5b_model --enable_vram_management`, (3) `code/panoramic_video_to_3DScene.py --inout_dir=... --resolution=720`
  (optimisation-based, ~10 GB VRAM; needs nvdiffrast, simple-knn, ODGS, pytorch3d builds: **not done yet**).
- Weights are gated on Hugging Face. The 5B LoRA (243 MB) is on the Mac at `/Users/mariyafatima/NAMMA/pano_video_gen_720p_5b.safetensors`
  and on Kaggle as private dataset `anasrazy/namma-matrix3d-5b`. Base model `Wan-AI/Wan2.2-TI2V-5B` and `Ruicheng/moge-vitl` are public.
- Don't run the repo's `download_checkpoints.py`: it pulls the 14B and LRM models (tens of GB, Kaggle disk runs out).
- The 5B LoRA path in the code is `checkpoints/Wan-AI/wan-lora/` (hyphen), the download script uses `wan_lora` (underscore).

## Matrix-3D job log (latest first)
- v5 (17:41 UTC): all installs and imports OK, models downloaded (Wan2.2-TI2V-5B is 33 GB on disk, fits). Failed in MoGe:
  `AttributeError: module 'utils3d.numpy' has no attribute 'icosahedron'`. **Fix next:** pin `utils3d` to the older commit that
  MoGe v1 was written for (the latest utils3d renamed things) in `colab/real_scenes/matrix3d/kaggle/run.py`, push v6.
- v4: missing pyrender (fixed: all imports now pre-checked). v3: DiffSynth install broke on pkg_resources (fixed). v1/v2: clone into existing folder (fixed).
- Local working copy of the job: scratchpad `km3d/` (same as repo `colab/real_scenes/matrix3d/kaggle/` plus the v4/v5 fixes).

## Next steps (in order)
1. When Matrix-3D v3 finishes: `kaggle kernels output anasrazy/namma-matrix3d -p <dir>`, look at `pano_video.mp4`. If it failed,
   read the log, fix `run.py`, push again (`kaggle kernels push -p colab/real_scenes/matrix3d/kaggle`).
2. Add step 3 (video → 3D Gaussian scene `generated_3dgs_opt.ply`) to the same job, then view it in `colab/real_scenes/scripts/web_walk`.
3. Better panorama input: more Depth Anything 3 frames (only 22 so far) so less of it is invented; one panorama per room.
4. Merge per-room results with real camera poses (multi-room, later multi-floor). Difix3D+ with a real reference frame as final clean-up.

## Gotchas already solved (don't repeat them)
- Kaggle's Python is 3.13 and uv targets it: create a 3.10/3.11 venv and pass `--python <venv>/bin/python` to every `uv pip install`.
- Nerfstudio 1.1.5: needs `setuptools<70` (pkg_resources), `apt-get install colmap` (ns-process-data checks it), hloc from a
  `--recursive` clone (SuperPoint submodule), prebuilt `gsplat==1.4.0+pt21cu118`; it has **no MCMC strategy** (use `splatfacto-big`).
- DN-Splatter on gsplat 1.4 needs `scripts/patch_dn_splatter.py`; its ZoeDepth script breaks on torch 2.1 (use `scripts/mono_depth_prior.py`).
- WorldMirror 2.0 on a T4: torch 2.7.1 cu128, install requirements one by one, FlashAttention does not run on T4 → stand-in using
  PyTorch SDPA in fp16 with the memory-efficient kernel (the math fallback needs 13 GB and OOMs).
- Web viewers: never crop the splat to the camera-path box (cuts far walls), keep the capture's portrait aspect, artifacts
  serve only web file types (ship .ply bytes as `.wasm`).
- Recorder: AR removed (poses drifted/reset, camera capped); 4K30 camera mode; exposure/colour locked by the user in a
  preview; focus automatic (tap to focus, hold or button to lock, never auto-lock: it blurred).
