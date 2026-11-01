# NAMMA hand-off: v3 pipeline (current, written 26 Oct 2026 ~00:30 IST)

**Start here.** This replaces `docs/HANDOFF.md` (24 Oct) and `docs/HANDOFF_CLOUD_2026-10-25.md` (25 Oct) as the entry point; they
describe the older custom pipeline and are kept for history. Measured findings, newest last: `colab/real_scenes/final/NOTES.md`.
Repo `github.com/RazyAnas/NAMMA` (public), branch **`claude/wonderful-cori-i3fn41`** (all work below is there, not on `main`).

## 1. Goal and owner rules
- **Free and open-source tools only, nothing paid** (owner, 26 Oct): no paid APIs (Gemini/OpenAI image APIs ruled out), free GPUs (Kaggle).
  Personal project, not commercial (owner, 26 Oct): research / non-commercial licences are fine (Difix3D+, Dr. Splat code,
  NC depth/pose models). Still free only: no paid APIs, no paid GPUs. Licences that forbid showing the results publicly (Lyra 2.0) stay out.
- **Goal**: phone video of a real room in → photoreal, walkable 3D model of *that* room out (real estate). Generation may only
  fill gaps, never replace or invent what was filmed, and generated content is flagged. Free GPUs only (Kaggle T4 x2).
  One command: video in, best possible 3D model out. Owner wants "extreme photorealism" (like Marble / Tencent HY-World) and
  quick proofs. The YouTube method (Mickmumpitz / Matrix-3D: geometry → drone views → re-project real pixels → video model fills
  only the rest → retrain) is planned as step 6 (below).
- **Owner**: Raiyan Razi (they/them), GitHub `RazyAnas`, Kaggle `anasrazy`, Hugging Face `anasrazy`. Time zone IST.
- **Rules** (keep them):
  - Don't ask questions, act. Short answers, plain words.
  - Check every result yourself (QA sheet, metrics) before handing anything over. Never hand over unchecked models.
  - Measure with proof. Keep measured numbers and claims apart. Say plainly when something failed.
  - Give an expected finish time (IST) with every job.
  - Only one AI drives Kaggle at a time.
  - Tokens and keys: never in git or chat. They live in GitHub Actions secrets and Kaggle Secrets.
  - Public / passive data only; never leaked data; never log in as anyone else.
  - The owner's Mac disk is nearly full: download only small files (QA sheet, metrics, `.splat`), never the whole output.

## 2. Why v3 (decided 25 Oct after namma-all failed)
Three days of custom code produced one decent partial model (corridor: 162 of 596 frames, 38.1 dB held-out, 38% needle
splats, no windows). Root causes: (1) the owner's 22 Oct capture (fast turns, blank walls, look-alike corridor ends,
auto-exposure) broke the camera solve into disagreeing passes; (2) custom trainer + repair layers (pass alignment, video bridges,
IBR checks, Wan filling) compensated for a bad capture instead of using good capture + standard tools; (3) namma-all (ARKitScenes
video) died because Depth Anything 3 ran out of GPU memory on landscape frames (5/300 merged) and every later stage needed depth.
**v3 = what everyone with good results uses**: good capture → SfM cameras (hloc) → gsplat's own `simple_trainer.py` → QA.
The custom pieces stay in the repo as options (`final/train_splat.py`, `align_passes.py`, `ibr*.py`, `fill_holes.py`, `densify.py`,
photo viewer) but are out of the default path.

## 3. The v3 pipeline (one Kaggle job: `colab/real_scenes/v3/run.py`, kernel `anasrazy/namma-v3`, 2x T4)
| Step | What | Code | Measured time |
|---|---|---|---|
| 1 frames | video: sharpest frame per time window (variance of Laplacian); folder: evenly spaced subset; long side ≤ 1280 | `final/frames.py` | < 1 min |
| 2 cameras | hloc SuperPoint+LightGlue, pairs = next 10 frames + NetVLAD top 15, COLMAP mapper, one shared SIMPLE_RADIAL camera → `sfm_report.json` | `final/sfm_hloc.py` (py3.10 uv venv, torch 2.1.2 cu118) | 411 frames: ~11 min incl. venv install |
| 3 splats | gsplat v1.5.3 `examples/simple_trainer.py mcmc`, bilateral grid (phone auto-exposure), SfM depth loss, held-out eval every 8th frame, `--save-ply`. GPU 1: 7k-step preview (cap 1M splats); GPU 0 at the same time: 30k-step full model (cap 1.5M) | upstream trainer, system Python 3.13 | preview ~15 min, full ~60 min (estimate, not yet measured) |
| 4 QA | `qa_sheet.jpg` (row 1: 3 held-out frames real ∣ model; row 2: two views ~50 cm beside the walk, one from outside above, one top-down), `metrics.json` (held-out PSNR/SSIM/LPIPS, exposure-fair `cc_` versions, needle fraction, off-path coverage, `pass`), levelled `model.splat` (floor at y=0, for SuperSplat, SH0 only), `model.ply` (full SH3), `walkthrough.mp4` (trainer fly-through, colour ∣ depth) | `final/v3_qa.py`, `final/to_splat.py` | ~1 min |

**Pass** = exposure-fair held-out PSNR ≥ 28 dB, needles < 10%, and the QA sheet shows the whole room (no missing walls / fog /
floaters in the off-path, outside and top-down views), model levelled.

Kaggle output layout: `/kaggle/working/{source.json, sfm_report.json, v3_summary.json, preview/, full/}`, each model folder:
`qa_sheet.jpg metrics.json nav.json model.splat model.ply walkthrough.mp4 train.log`.

`SOURCE` (set by `gh_run.py`, or edit the default in `run.py`): `kaggle:<dataset name>` (video, frame folder or .zip in that
attached dataset) · `kaggle` (first .mp4/.mov under /kaggle/input) · `dl3dv:<hash>|dl3dv:auto` (needs Kaggle secret `HF_TOKEN`
attached in the Kaggle editor) · a path. Other env knobs: `NFRAMES` (default 300), `LONG` (1280), `STEPS` (30000), `PREVIEW` (7000).

## 4. How runs are started and collected (GitHub Actions; no Mac needed)
The Claude cloud container **cannot reach kaggle.com** (proxy 403). GitHub's runners can, so:
- Workflow `.github/workflows/namma-v3.yml` runs `colab/real_scenes/v3/gh_run.py` on every push to the branch that changes
  **`colab/real_scenes/v3/RUN.json`**. (`workflow_dispatch` only works once the file is on `main`.)
- **To start a run**: edit `RUN.json`: bump `n`, set `source`, make sure `"collect": false` (or remove it), push.
  `source`: `dl3dv:<hash>` / `dl3dv:auto` (runner downloads the DL3DV-Benchmark scene with the HF token, converts to JPG, uploads
  it once as private Kaggle dataset `anasrazy/namma-dl3dv-<hash[:10]>`) or `kaggle:<owner>/<dataset>` (a video/frames dataset
  already on Kaggle, e.g. the owner's re-filmed room). `env` overrides run.py knobs.
- **`"collect": true`** = push nothing; wait for the Kaggle version already running and collect its outputs (use when a GitHub run
  was cancelled: cancelling GitHub does NOT stop the Kaggle job). **Set it back to false before the next real run.**
- gh_run then pushes the Kaggle job (run.py with the settings prepended), polls `kaggle kernels status` every minute, downloads all
  outputs, prints sfm_report / v3_summary / metrics / log tail into the GitHub log, and uploads everything (except the preview .ply)
  to the private HF dataset **`anasrazy/namma-v3-out/<YYYYMMDD-HHMM UTC of the GitHub run>/`**. Exits red if the Kaggle job
  failed or made no full model.
- **Secrets** (repo Settings → Secrets and variables → Actions; already set by the owner): `KAGGLE_TOKEN` (contents of
  `~/.kaggle/access_token`; `KAGGLE_API_TOKEN` also accepted) and `HUGGINGFACE_WRITE_TOKEN` (`HF_TOKEN` also accepted). The
  owner has clicked "Agree" on `DL3DV/DL3DV-Benchmark` with that HF account.
- **Watching**: run status is public JSON: `curl -s https://api.github.com/repos/RazyAnas/NAMMA/actions/runs/<run id>`
  (`status`, `conclusion`). Job logs are only downloadable after the job ends (GitHub MCP `get_job_logs`). Results: HF connector
  `hf_fs` (authenticated as anasrazy): `cat hf://datasets/anasrazy/namma-v3-out/<run>/full/metrics.json`,
  `attach .../full/qa_sheet.jpg` to look at the sheet. Owner opens `model.splat` (or `model.ply`) by drag-and-drop in SuperSplat
  (superspl.at/editor), downloaded from the HF dataset page while logged in.
- Old way (owner's Mac, still works for kernels that need no secrets): `bash colab/real_scenes/final/run_and_alarm.sh <kernel
  folder> anasrazy/<kernel> <out dir> '<file regex>'` (pushes, waits, downloads matching files, rings).

## 5. State right now (00:30 IST, 26 Oct)
- **Test scene**: DL3DV-Benchmark #44 `389a460ca1995e0658e85fe8e6b520b4e88b370cd6710dfe728b1564bba31aee` (Hotels-Motels, bounded,
  non-reflective, non-transparent): a hotel room filmed handheld on a phone, 411 frames 960x540 (bed, sofa, window, bathroom door,
  a mirror, a dog walks through). Chosen by looking at its frames. On Kaggle as `anasrazy/namma-dl3dv-389a460ca1`.
- **Run 1** (Kaggle version 1): cameras **411/411 registered, 1 model, 1.15 px reprojection, k1 = -0.0007, 13,964 points**.
  gsplat CUDA build reused from namma-prep (ready in 1 s), fused-ssim compiled, 2 GPUs. Both trainers died on
  `No module named 'datasets.colmap'` (Kaggle's Hugging Face `datasets` package hides gsplat's `examples/datasets`, which has no
  `__init__.py`). Fixed: run.py touches `examples/datasets/__init__.py`, imports the trainer once and pip-installs anything missing.
- **Run 2** (Kaggle version 2, pushed 22:46 IST, finished ~00:04 IST; its GitHub runner was cancelled by hand at 00:03 IST, the
  outputs were fetched by collect run `40221983672`). **Results: `hf://datasets/anasrazy/namma-v3-out/20261025-1835/`**
  (https://huggingface.co/datasets/anasrazy/namma-v3-out/tree/main/20261025-1835): cameras 411/411 again (2 models in the folder,
  411 in the main one, 1.15 px).
  | model | steps / time | splats | held-out PSNR raw / exposure-fair | SSIM | LPIPS |
  |---|---|---|---|---|---|
  | preview (GPU 1) | 7k / 10 min | 317k | 24.54 / 26.76 dB | 0.908 | 0.207 |
  | **full (GPU 0)** | 30k / 68 min (~6.3 it/s on a T4 while the preview shared the CPU) | 1.5M (cap) | **25.71 / 29.50 dB** | **0.939 / 0.945** | **0.132 / 0.120** |
  QA sheet checked by eye: held-out renders are near-identical to the real frames (bed, mirror, bathroom, sofa, the dog), and the
  two views 50 cm beside the walk are clean and photoreal. **First genuinely photoreal model.** `pass` was false only because
  of two QA-tool faults, both fixed after the run (not yet re-run): (1) the needle count flagged flat discs (longest/shortest > 10),
  76%; now needle = longest/middle > 10, flat discs are reported separately; (2) the outside and top-down overview shots looked
  into the shell of big splats that models the view through the window and showed only blur; they now render a dollhouse (room box
  from the SfM points, ceiling cut). Files: `full/model.splat` (48 MB, levelled, SH0), `full/model.ply` (354 MB, SH3),
  `full/walkthrough.mp4`, `full/qa_sheet.jpg`, `full/metrics.json`; same for `preview/` (no .ply on HF).
- `RUN.json` was reset to `n: 0, collect: false` after collecting (pushing RUN.json always starts the workflow; n = 0 exits at once).
- No reminders or jobs are pending (the old session's check-in was deleted once run 2 was collected).

## 6. Next steps (in order)
1. Show the owner run 2 (`full/model.splat` in SuperSplat). Optional: re-run the same scene (RUN.json `n` bump, `collect` false)
   to get the fixed QA sheet (dollhouse views, true needle count) before claiming "whole room OK"; the off-path / outside quality is
   still unproven. If smears or floaters show up: MCMC `--scale-reg` up / cap down, or more steps.
2. **Owner's own room**: owner re-films following `docs/CAPTURE_GUIDE.md` (landscape, locked exposure/focus, slow turns, one loop,
   1-2 min), uploads the video as a Kaggle dataset, then RUN.json `source: "kaggle:anasrazy/<dataset>"`, bump `n`, push.
3. gh_run: check whether the Kaggle dataset already exists *before* downloading DL3DV (wastes ~8 min now); optionally upload a
   progress file to HF while waiting (logs are invisible until the GitHub job ends).
4. **Camera fallback** when SfM breaks a video into several models (`sfm_report.models > 1` or registered < 85%): VGGT-Long
   (`facebook/VGGT-1B`, CC BY-NC) poses → triangulate + bundle adjust with the hloc matches (new `final/vggt_poses.py`).
5. **Step 6, the YouTube method** (only after the base model passes): virtual "drone" paths 30-100 cm off the walk, render from the
   model, holes = low alpha; fill holes with real pixels re-projected from all frames first (`final/ibr.py` `synth(..., kmax=40)`),
   Wan 2.1 VACE 1.3B fills only what is left (`final/fill_holes.py`, flagged), add these views to training at weight 0.3 with
   unfilmed pixels masked, +5k steps (new `final/drone_views.py`). Measured on the old data: Wan filled 0.8% of pixels for 20 GPU
   minutes, so real coverage matters more than generation.
6. Viewer integration (trained model between photo spots, collision from the floor plan: `final/fuse_nav.py`, `photo_viewer.*`),
   Techfest R1 on **31 Oct** (`docs/TECHFEST_R1.md`: numbers, screen recording, abstract, hosted viewer).
7. Old bugs (only if those paths are used again): DA3 OOM on landscape frames in `final/run.py`; a failed depth merge must not kill
   the camera solve (`sfm_align.py`); clear errors on 0 frames in `ibr_check.py` / `align_passes.py`.

## 7. Gotchas (each one cost a run or an hour)
- **Two Pythons on Kaggle**: hloc needs the official `pycolmap` (py3.10 venv at `/kaggle/temp/v3/sfmenv`); simple_trainer needs
  the `rmbrualla/pycolmap` fork (system Python). The fork overflows on numpy 2 at import (`np.uint64(-1)`): run.py sed-patches
  `scene_manager.py`, locating it with `importlib.util.find_spec` (importing it would fail).
- gsplat 1.5.3 from PyPI compiles CUDA on first use (~5 min). namma-prep's output (`gsplat_ext.tgz`, attached via
  `kernel_sources`) is restored into `~/.cache` and the gsplat sources are back-dated so ninja reuses it; `final/gsplat_warm.py`
  compiles/loads once before two GPUs start.
- fused-ssim is compiled with `TORCH_CUDA_ARCH_LIST=7.5`; if it fails, `final/fused_ssim_shim/` (pure torch) goes on PYTHONPATH.
- simple_trainer's depth loss crashes (KeyError) if any image has no SfM points; sfm_report counts `frames_without_points` and
  run.py drops `--depth-loss` then.
- Checkpoints are `ckpts/ckpt_{steps-1}_rank0.pt`, PLY `ply/point_cloud_{steps-1}.ply`, stats `stats/val_step{steps-1:04d}.json`.
  With the bilateral grid, `psnr` is raw and `cc_psnr` is after colour matching (exposure-fair; compare this with old numbers).
- The trainer's world is normalised (Parser `normalize=True`); v3_qa uses the same Parser, so cameras/points match the PLY.
  `nav.json` has `flipped: false` (simple_trainer PLYs are not y/z flipped; old train_splat PLYs were).
- Trained in classic mode (not `--antialiased`) so SuperSplat shows exactly what was measured. `.splat` keeps only SH0 colour;
  `.ply` keeps view-dependent SH3.
- After `kaggle kernels push`, status can still show the previous version's `complete`; gh_run waits to see running/queued first
  (or 15 min).
- Kaggle mounts datasets at `/kaggle/input/datasets/<owner>/<name>/` (new layout) or `/kaggle/input/<name>/`: always glob.
- DL3DV-Benchmark meta labels: `environment` bounded/unbounded, `ref_degree_binary` (non)reflection, `trans_degree_binary`
  (non)transparent, `scene` category (no "indoor"/"room" words). Frames: `<hash>/nerfstudio/images_4/frame_00001.png` (960x540).
- kaggle CLI 2.2.5 authenticates with the `KAGGLE_API_TOKEN` env var alone.

## 8. Earlier results worth knowing (details in NOTES.md)
- Lobby/room tests with Nerfstudio and our trainer: fog and floaters whenever cameras were wrong; camera quality was the lever.
- Corridor model (namma-model, 25 Oct): 162 frames of one consistent pass group, **38.1 dB / SSIM 0.957 held-out** (custom trainer,
  15 min), 37.8% needles; MCMC variant 34.2 dB with 0.1% needles. Levelled .splat opened by the owner: no windows, smears.
- Wan 2.1 VACE hole filling: honest only for small gaps (0.8% of pixels at 30 cm off path); in big holes it invented objects.
- `docs/CAPTURE_GUIDE.md` is the owner's filming guide; `docs/VIDEO_METHOD_PLAN.md`, `docs/PHOTOREAL_PLAN.md` older plans;
  `docs/TODO_2026-10-26.md` the to-do written before v3 (its "one-command job" item is now v3).
- Status page: `dash/status.json` (+ `dash/index.html`, `dash/watch.sh`).

## 9. Work since 00:30 IST, 26 Oct (keep this section current)
- **Run 3 (00:16-02:07 IST, results `hf://datasets/anasrazy/namma-v3-out/20261025-1846/`)**: same hotel room.
  full 1.5M splats: **29.55 dB exposure-fair (25.70 raw), SSIM 0.946, LPIPS 0.13**, needle fraction (true metric) 15.8%, flat 50%;
  big 3M splats: 29.39 dB, needles 17.8% -> more splats do not help, keep 1.5M. Fixed QA sheet: held-out and off-path views
  photoreal; the dollhouse views (room crop) show the whole room recognisably but with spiky smears around the edges (the
  needles). Fix in the tour: make_tour build drops splats with longest > 8x middle axis AND longest > 1 % of the room before
  compressing. Next training run could raise MCMC `--scale-reg` (default 0.01) to stop them forming.
- **Tour publishing**: collect runs (RUN.json `collect: true`, n bump) build the tour for the latest Kaggle output. Kaggle CLI
  `datasets download` of our private dataset returned 404 on the runner, so for DL3DV sources gh_run fetches the frames straight
  from HF (dl3dv.fetch). First real tour build: GitHub run `40232064223` (n 6) -> private Space `anasrazy/namma-tour`.
  SOG compression of 1.5M splats on the runner's CPU is slow (tens of minutes).
- **SplatTour viewer UI (screenshots in `site_look/20261025-1957/`)**: top toggle Panoramas | Splats | Staging, floor-plan
  minimap bottom-left (top-down glowing point density), room-thumbnail strip, play/walk buttons. Their splats also smear when the
  camera goes off the walk (close to a wall). Our tour now has the same minimap (live dot, clickable spots) + photo strip.
- **run.py / gh_run.py**: `VARIANTS` env = JSON list `[name, steps, cap(, '2dgs')]`, first two run on the two GPUs, the rest
  queue; gh_run skips the DL3DV download when the Kaggle dataset already exists; HF upload skips `model.ply` except `full/`.
- **Geometry ("mould") branch, built + CPU-tested, not yet run on a GPU**: `final/mesh_tsdf.py` renders median depth from a
  2DGS checkpoint (gsplat `examples/simple_trainer_2dgs.py`, `--normal-loss --dist-loss`) or expected depth from the 3DGS
  model, fuses all cameras into a TSDF on the GPU, marching cubes (scikit-image), drops loose pieces, colours vertices from the
  real undistorted photos, applies `full/level.json` (written by `to_splat.level`, p' = R p + t) so the mesh lines up with
  `full/model.splat`. Outputs `mesh/{mesh.ply, mesh.glb, mesh_sheet.jpg, mesh.json}`. run.py builds it at the end of every run
  (from a `'2dgs'` variant if present, else from `full`). Synthetic test: 0.4 voxel median error, 97% faces toward cameras.
  Planned run 4: VARIANTS `[["full",30000,1500000],["geo",30000,0,"2dgs"]]`.
- **Owner wants results "exactly like splattour.com, not messy output".** Their own pages (read via the screenshot tool):
  capture = one uncut video per floor, phone upright, 0.5x ultra-wide, per room 3 passes (lap along walls, reverse lap, both
  diagonals), ~6 min for a 60 m2 flat; processing = frames 1-2 fps, COLMAP, gsplat training 20-90 min, SOG/SPZ compression;
  product = per-room splats in one shared frame streamed by room, floor plan + room thumbnails, 360 panoramas from real frames,
  floor markers to move. Sample tours: splattour.com/en/t/Terrassa_Apartment, /Manresa_Premium, /Airbnb_Santa_Teresa (they load
  `cdn.splattour.com/configs/<tour>.json`, `<room>.sog`, `thumbs/floorplan-*.jpg`, `thumbs/rooms/*.jpg`). Our splat quality is
  already in their class (hotel room); what we lack is capture discipline, room cropping and a tour viewer.
- **Screenshot tool**: `scripts/site_look/look.py` + `.github/workflows/site-look.yml` (push a change to
  `scripts/site_look/URLS.txt`; lines `tour <url>` are walked through). Screenshots land in
  `hf://datasets/anasrazy/namma-v3-out/site_look/<time>/` (view with hf_fs `attach`). WebGL on the runner is software-rendered:
  screenshots of 3D viewers need long timeouts (fixed: 240 s, 1100x700 viewport).
- **Tour (built, tested locally, not yet run on a real model)**: `final/make_tour.py` on the official **SuperSplat Viewer**
  (`@playcanvas/supersplat-viewer@1.37.0`, its static `public/` site) + **`@playcanvas/splat-transform@3.10.1`** (SOG works on
  CPU with `-g cpu`; `--filter-floaters` needs a GPU, not used). `plan` (needs gsplat examples Parser + torch + the frames):
  cameras in room coordinates (level matrix, floor y = 0), room box from SfM points (3-97 %, +12 %), start view = camera nearest
  the room middle, `animTracks` = 120-key guided walk along the smoothed real camera path (startMode `animTrack`; keyframe times
  are FRAME NUMBERS, ints), 6 hotspots; settings pass the viewer's `validateSettings(..., {limits: true})`. `build`:
  splat-transform `-r E -t t -B box -N` -> `scene.sog`, `collision.glb` = mesh if present, index.html bootstrap JSON.
  **Coordinates (measured)**: the viewer rotates every splat file 180 deg about z (F = diag(-1,-1,1)); splat-transform applies
  -r/-t in that flipped space (file' = F E F p + F t, Euler 'xyz' extrinsic). So file = F (R p + t) with E = R F, t as is, and
  settings/box are plain room coordinates. Orbit mode auto-rotates (start view drifts); the guided walk and fly/walk modes don't.
  `window.app` (PlayCanvas app) is exposed in the standalone page for probing. Local test: tour view at a real camera matches
  the photo (orientation, not mirrored).
  **gh_run.py** builds the tour after every run (CPU torch + pycolmap fork + gsplat examples + npm packages on the runner; frames
  from the run's `images/` (run.py saves them since run 4) or rebuilt from the scene's Kaggle dataset) and publishes it to the
  **private static HF Space `anasrazy/namma-tour`** (folder per GitHub run + index page; owner opens
  https://huggingface.co/spaces/anasrazy/namma-tour). The site-look tool sends the HF token only to `*.hf.space`, so it can
  screenshot the private tour for checking. For run 3 (started before this existed): after it finishes, push RUN.json with
  `collect: true` (n bump) -> collect run builds + publishes the tour.
- Network facts for the cloud container: kaggle.com, huggingface.co, splattour.com, unpkg/jsdelivr and Azure blob (workflow
  artifacts) are blocked; github.com, raw.githubusercontent.com, api.github.com (public JSON) and registry.npmjs.org work.
- **Capture guide rewritten** (`docs/CAPTURE_GUIDE.md`): SplatTour's method (upright, 0.5x ultra-wide, 3 passes per room, one
  uncut video per floor, floor out of frame) + our 22 Oct lessons; upload as a private Kaggle dataset, source `kaggle:anasrazy/<name>`.
- **Tour publishing now (26 Oct, replaces the private-Space note above)**: public static Space `anasrazy/namma-tour`, each tour
  in a folder `<RUN>-<20 random hex>` (owner's choice: private static Spaces cannot serve their files to a browser). The link
  is never printed (GitHub logs are public): it is in the private `namma-v3-out/<RUN>/tour_url.txt`. Code:
  `v3/tour_publish.py` (used by gh_run.py after a run). **Republish any run's built tour** without Kaggle: set `run` in
  `v3/PUBLISH.json` and push -> `.github/workflows/tour-publish.yml` (own concurrency group, ~5 min) publishes, then
  screenshots it with `scripts/site_look/look.py` in `TOUR_RUN` mode (link read from the dataset, folder redacted in logs) ->
  `site_look/<time>/`. Bug fixed: gh_run.py published only if `scene.sog` existed, so tour v2 (compressed PLY) was built but
  not published by the collect run; run 4 started before the fix, so republish it with PUBLISH.json when it ends.
- **Tour v2 (run 3 model, results `20261026-0309`, published 09:20 IST)**: metric scale 1.848 m/unit (median phone height
  1.35 m), room box 7.1 x 2.6 x 7.0 m, room-box collision, walk mode on (viewer's 5 m minimum patched to 2), SH1 colour as
  `scene.compressed.ply` (1.39M splats, 35 MB), 15,794 smear splats removed, sharpness 0.2, 6 spots, 68 s guided walk.
  **Checked on screenshots (site_look/20261026-0349)**: loads, guided walk start is photoreal, Walk Mode active with WASD.
  Still bad: streaky smears on the floor, ceiling and anything off the filmed path (needs run 4 detail + Difix3D+); the
  floor-plan minimap (SfM points top-down) is mostly black with sparse dots, hard to read (TODO: render a top-down image of
  the splat instead).
- **Run 4** (GitHub run 40262727449, started 08:40 IST): `images_2` 1920x1080 -> 1440 px, 420 frames, full 3DGS 30k + 2DGS geo
  20k -> TSDF mesh -> mesh collision; ETA ~12:00 IST.
- **Owner's YouTube link (2t-PLeenqqA) judged**: it shows the **Aholo Viewer** (Manycore, MIT): LOD chunk streaming of huge
  splat scenes + collision tools. A viewer/delivery upgrade, not reconstruction quality; its demos look sharp because of
  drone/scanner captures. Useful later for whole-house tours; our PlayCanvas viewer already has LOD streaming and voxel
  collision. Quality levers stay: training resolution, capture coverage, Difix3D+/GSFixer, SH colour.
- **Run 4 (results `20261026-0327`, owner: "excellent")**: 1440 px from images_2, 411/411 frames, held-out 29.22 dB exposure-fair
  (run 3: 29.55 at 960 px; not directly comparable), SSIM 0.934, needles 12.7 % (run 3: 15.8 %), 2DGS mesh 3M faces with holes
  (59 % voxels observed). Tour collision = room box + mesh furniture cut to the room, simplified to 150k faces (2.8 MB,
  `make_tour.collision`). Published via PUBLISH.json (floorplan + collision). site_look now logs the camera position per shot.
- **Run 5** (GitHub run 40278290892, started 11:42 IST): run 4 settings, variants `full` vs `prior` (room prior A/B).
- **Qwen smoke test ready, not started** (`final/qwen_smoke.py`, `v3/qwen_gh.py`, `.github/workflows/qwen-smoke.yml`, start by
  setting `n` > 0 in `v3/QWEN.json`, only when no namma-v3 run is training): renders 2 held-out + 3 off-path views of run 4,
  Qwen-Image-Edit-2511 with [render + 2 nearest real frames], text encoder 4-bit (freed), transformer GGUF Q4_K_M (fallback Q3_K_M,
  then bf16), Lightning 4-step LoRA; report = s/image, GPU GB, PSNR/SSIM/LPIPS before/after on held-out; sheet to
  `namma-v3-out/qwen_smoke/<time>/`. Owner rule: free/open only (Qwen weights Apache 2.0; unsloth GGUF = quantised Apache).
- **More frames (owner's idea, prepared)**: DL3DV gives only ~411 frames of this scene (~2-3 per second of video); we already use
  all of them. New source `dl3dv-video:<hash>` in gh_run.py: the scene's original phone video (DL3DV/DL3DV-ALL-video,
  `7K/389a460ca1.../video.mp4`, 1.6 GB) -> Kaggle dataset `namma-dl3dv-<hash10>-video` -> run.py `frames.from_video` (sharpest
  frame per time window; ffmpeg H.264 fallback if OpenCV cannot decode it). Planned **run 6** after the Qwen test: NFRAMES ~900,
  LONG 1440, STEPS 40000, variants per run 5's winner. Expected: better detail near the path and during fast turns; it does not
  add viewpoints the camera never reached (off-path smears need the room prior / AI fix).
- **Run 5 (results `20261026-0610`, room prior A/B, same frames/settings as run 4)**: `full` 29.07 dB exposure-fair / SSIM 0.925 /
  needles 13 %; `prior` (final/room_prior.py) 29.00 dB / 0.925 / **needles 0.0 %** (pass). QA sheets look the same; the prior
  costs nothing measurable and removes needle splats -> use `prior` from run 6 on. Mesh from 3DGS: 74 % voxels observed.
  Tour published (gh_run publish fix works): link in `20261026-0610/tour_url.txt`.
- **Hybrid "solid room" demo (`final/hybrid_demo.py`, hybrid-demo.yml, results `hybrid_demo/20261026-0906`)**: room shell from
  the TSDF mesh footprint + furniture mesh, both textured from real training photos (Open3D ray-cast visibility), as flat
  opaque surface splats. Synthetic room: fine (29-35 dB). **Run 4 room: worse than the splat model, stopped**: held-out
  13.6 / 20.2 / 22.5 dB vs splat 25.1 / 26.0 / 28.2 (QA JPEG); wavy textures (TSDF surface noise), black holes (mesh holes),
  floor only 2.3 % textured (bits of the TSDF floor above the shell floor block it), outline 78 corners. Lesson: with phone
  video only, mesh geometry is not accurate enough to carry the photos; splats stay the base. Possible later use: shell planes
  UNDER the trained splats only where splats are unreliable (not replacing them).
- **Owner rule (26 Oct): never mix the hall and the hotel room.** The hall is a separate capture/project (images of it sit in
  the `namma-prep` Kaggle kernel output, attached to jobs only for the gsplat build cache). Jobs on a run must take frames only
  from that run's own frames dataset; `qwen_smoke.py` now filters by dataset name and checks frames == cameras (the first Qwen
  test silently picked the hall images as "reference photos").
- **Qwen smoke test, attempt 2 (`qwen_smoke/20261026-0929`)**: Q4_K_M (13.2 GB) out of memory on one T4 (fp16 and bf16);
  Q3_K_M fp16 gave blank/NaN images; Lightning LoRA failed (Kaggle's torchao 0.10 too old). Attempt 3: torchao removed, bf16,
  Q4_K_M split by hand over both T4s (`split_gpus`), Q3_K_M bf16 on one T4 as fallback.
- **Qwen attempt 3 (`qwen_smoke/20261026-1010`)**: LoRA loads now (torchao removed), split over 2 T4s works, but attention
  needed a 5.1 GB block (3 inputs x ~4000 tokens at 1 MP each; bf16 on a T4 falls back to plain attention). Attempt 4: each
  input image at 512^2 for the VAE (`qp.VAE_IMAGE_SIZE`), output still 1024x576, expandable CUDA segments.
- **Qwen smoke test RESULT (attempt 5, `qwen_smoke/20261026-1102`)**: ran with Q3_K_M bf16 split over both T4s + Lightning
  4-step LoRA, inputs at 512^2, output 1024x576: **~4 min per image**, GPU peak 14.1 GB. Held-out frames: exposure-fair
  PSNR 28.03 -> 16.09 and 26.36 -> 15.84 dB, SSIM 0.92 -> 0.48, LPIPS 0.16 -> 0.56. Outputs look like sharp photos, but
  the editor re-frames the view (camera drift) and copies objects from the reference photos (the dog twice, wall art twice).
  **Not usable for training views** (would teach the model shifted walls and duplicates). Next candidate: FreeFix (MIT,
  per-pixel confidence mask keeps confident pixels fixed), Difix3D+ for the research demo only (non-commercial licence).
- **FreeFix RESULT (attempt 3, `freefix/20261026-1235`, Kaggle T4, 18.6 min)**: FreeFix's own loop (certainty from the
  splat Hessian -> masked SDXL-refiner img2img, fp16-safe VAE `madebyollin/sdxl-vae-fp16-fix` -> refine 400-800 steps) on
  8 off-path views of run 4, held-out frames never trained on. Held-out (16 frames, FreeFix renderer): PSNR 25.84 -> 27.11,
  but SSIM 0.925 -> 0.902 and LPIPS 0.203 -> 0.249 (worse): colours closer, fine texture softer (sofa fabric, floor). The
  SDXL views remove some smear but also invent/smooth detail (toilet, door). Tour of the refined model published
  (`freefix/20261026-1235/tour_url.txt`, screenshots `site_look/20261026-1312`): smears by the mirror/bathroom and on the
  floor are still there. **Not a fix in this form**; not merged into the pipeline.
- **SAM 3D Objects demo** (`v3/sam3d_gh.py`, `.github/workflows/sam3d-demo.yml`, `v3/SAM3D.json`; CPU runner + free
  ZeroGPU Space `dev-bjoern/sam3d-objects-mcp`, no Kaggle): torchvision Faster R-CNN finds sofa/bed/chair/TV/... in the
  hotel-room frames, picks each object's best frame, SAM 3 + SAM 3D Objects (Meta, SAM License) rebuild each in full 3D
  (GLB), viewer page published like a tour (`sam3d/<time>/tour_url.txt`). The object half of the Blender-style pipeline.
  **First result (`sam3d/20261026-1307`)**: detector picked bed/sofa/chair/TV/suitcase frames correctly (sofa frame_00043, bed
  frame_00127); sofa rebuilt in 82 s: clean solid shape with the back tufting, legs and the unseen back, but untextured grey
  (this Space turns texture baking off) and proportions only approximate. Free ZeroGPU quota (~4.5 min/day) allowed only that
  one object (bed/chair/TV refused: quota). More objects need the quota to reset (24 h) or our own GPU (SAM 3D Objects asks
  for 32 GB VRAM; a T4 has 16 GB).
- **Difix3D+ test** (`final/difix_test.py`, `v3/difix_gh.py`, `.github/workflows/difix-test.yml`, `v3/DIFIX.json`): NVIDIA's
  own gsplat progressive-update trainer started from run 4's splats; ~210 off-path targets (train cameras +-50 cm sideways),
  fixed by single-step Difix (`nvidia/difix_ref`, guided by the nearest real frame) every 2000 steps for 12000 steps, trained at
  0.3 weight beside the real frames. Held-out frames never trained on (Difix's own split trains on every 8th frame: patched)
  and measured exposure-fair before/after. Caveat: Difix was trained on 112 of DL3DV-Benchmark's 140 scenes and run 4's room
  is one of them, so a good result here can be flattered; the owner's home is the fair test. Results: `difix/<time>/`.
- **Object layer plan (approved 26 Oct, after run 6)**: copy Hyper3D WorldGen's good idea for free on a real video. The splat
  stays the look; each object gets an id/label/embedding (segment-everything masks voted onto splats + 3D clusters, so unknown
  objects are found too) and a proxy mesh (SAM 3D / Hunyuan3D fitted to its real points, measured shape as fallback) for
  collision, clicking, search and Blender export. Whole house = room by room. Details: `docs/TOUR_QUALITY_RESEARCH.md` 9.
  Order: Difix v2 -> owner's home video (priority) -> run 6 -> object layer demo.
- **Difix3D+ results.**
  - **v1** (`difix/20261026-1850`, 4 h on a T4: 9 s per Difix view). Held-out frames, exposure-fair: 29.51 -> 30.31 dB, but
    LPIPS 0.189 -> 0.195 (worse). The owner saw blotchy walls: the fine-tune ran without run 4's exposure correction, and Difix
    re-shades each view a little differently.
  - **v2** (`difix/20261026-1925`, 87 min): bilateral grid on real frames only, and a smear-only mask (colour-matched
    difference > 0.08). Held-out: exposure-fair PSNR 29.51 -> 30.31, SSIM 0.932 -> 0.937, LPIPS 0.189 -> 0.183, all better.
    Walls are no longer blotchy (sheet row 5). Smears by the mirror/bathroom and on the floor are only slightly reduced in the
    tour (screenshots `site_look/20261026-2113`). A small, safe gain, not the fix for off-path smears.
  - Caveat: Difix may have seen this DL3DV room in training.
- **Difix dropped (owner, 26 Oct: "not worth it").** Code, config and workflow removed from the repo (`difix_test.py`,
  `difix_gh.py`, `DIFIX.json`, `difix-test.yml`; in git history if ever needed). Its results stay above. The off-path smears are
  now tackled at training time instead (run 6: room prior + depth/surface guide), not by AI-fixed views.
- **Object layer v2 (27 Oct, `objects/20261027-0443`, 14.5 min on a T4)**: 17 objects. Better than v1: window went to the room;
  3 duplicates merged; bag (CLIP 0.97), AC, suitcase, coat, picture frame right. Worse than v1: the "bed" merge swallowed the
  dresser + TV (a lamp crop was renamed "bed" by CLIP 0.58, then merged on the shared word "bed"); sofa and dresser lost their
  names; 17 "couch" (3.25 m) and 11 "bed" are mixed wall pieces; side table -> "trash can", wall art -> "cushion", floor lamp ->
  "lampshade". Verdict: not better than v1 overall. Fixes for v3: merge only on look (CLIP) + overlap, never on a name; CLIP may
  only replace a generic VLM answer; give the VLM the measured size; split mixed objects. Tour labels / click / search are in
  `make_tour.py` (tested with a stub viewer), used by the next objects run (or `collect_only`).
- **Run 6 stopped (owner, 27 Oct: no Kaggle GPU hours left this week)**. RUN.json n=12 is prepared (original video, 900
  frames, 1920 px, 3M splats, 50k steps; A/B `full` room prior vs `geo` room prior + DA3 guide). The attempt had already failed
  in 30 s: gh_run pushed the kernel before the new video dataset `namma-dl3dv-389a460ca1-video` was ready ("not valid dataset
  sources"; ffprobe missing on the runner). Before rerunning: wait for the dataset to be ready, then bump n. Consider the
  `mcmc` strategy first (docs/KMYI_NOTES.md).
- **Kwang Moo Yi's posts** (owner: "all the important tweets related to our problems"): `docs/KMYI_NOTES.md`, about 55 posts
  grouped by our problems, with a ranked list of what to try (1: gsplat's `mcmc` strategy, his own 3DGS-MCMC).
