# Findings that must not be lost (newest first)

## 2026-10-25 (3): gsplat opacity reset never runs (confirmed in gsplat 1.5.3 source; found by the Mac session)
`DefaultStrategy.step_post_backward`: `if step % self.reset_every == 0 & step > 0:` is always False (`&` binds before `==`), so
the 3DGS opacity reset that clears floaters never happens. Every gsplat fit so far ran without it: a strong candidate for the
held-out fog. `train_splat.py` now resets opacities to 0.01 every 3000 steps while densifying. Training cap raised to 30 min for
the 600-view variants (the only capped run so far was A at step 10028; S1/D/V1 finished 15000 steps in 470-520 s).
Sanity check still worth one run: Nerfstudio splatfacto on the same SfM cameras (`scripts/ns_env.sh`), lobby gave 25-27 dB.

## 2026-10-25 (2): V1/V2 run
- SfM again 195/200, 1.14 px. DA3/SfM per-frame scale spread 47%.
- V1 (SfM cameras, no refinement): held-out 18.79 dB raw, SSIM 0.780, 841k splats. Owner-visible held-out renders are FOG: fan,
  window, door gone. PSNR on white-wall rooms hides this; always look at the held-out images.
- V2 (2M MCMC + SH3, fixed lr/opacity): 12.55 dB, train 16.0. Still broken. Found: we ran densify/relocate BEFORE the optimizer
  step; gsplat's trainer does it after. Fixed. MCMC paused until a plain fit is right.
- Floor up from mean image-up: biased by camera pitch when walking mostly one way (synthetic: 25 deg). Now from the camera x axes
  (always horizontal for a phone without roll): 0.4 deg in the same test.
- Next: 600 frames (SfM on all, DA3 on every 3rd), W1 vs W2 (stronger depth anchor), step-0 and final renders + splat stats.

## 2026-10-25: SfM cameras run (200 frames, Kaggle T4, 69 min)
- hloc SuperPoint+LightGlue (sequential 10 + NetVLAD 15 pairs): **195/200 registered, 8425 points, reprojection error 1.07 px**.
- Chunked DA3 depth scale wandered **32.8% between frames** (std of log DA3/SfM ratio): DA3 alone cannot give consistent geometry
  for a long walk; use it only for dense depth after per-frame alignment to SfM (`sfm_align.py`).
- Held-out (frames 3,11,19,..., 25 views) with camera refinement on, after aligning held-out cameras:
  SfM cameras 22.12 dB / SSIM 0.841 (722k splats); DA3 cameras 22.42 / 0.836 (645k). Raw cameras: 17.1 dB for both.
  -> camera refinement co-drifts with the scene (moved even SfM cameras ~10 cm / 2.8 deg). Turned off for SfM cameras.
- 2M MCMC + SH3 variant: 15.3 dB (train 16.3), clearly broken training, NOT evidence against more splats. Suspected cause: means
  learning rate scaled by point-cloud extent (~3x the camera extent) + init opacity 0.5. Fixed (camera extent, opacity 0.1).
- Floor "up" from surface normals was 25 deg off (SfM run) and 88 deg off on a synthetic room (it picked a wall): walkable area
  collapsed to 0.1-0.5 m2. Now taken from the upright phone cameras; synthetic walkable 4.5 -> 14.0 m2, robust to a 25 deg tilt.

## 2026-10-24: the SDXL-filled panorama invents the room (owner-verified)
`matrix3d/kaggle/run.py` builds `pano_img.jpg` from 22 Depth Anything 3 frames; ~55% of the sphere is unseen and SDXL inpaints it.
The owner (who was in the room) says the result shows a **yellow door that does not exist** and a **vent that is really a fan**; it does
not match the video. Earlier notes calling its contents "real" were reported, never verified. Treat that panorama as hallucinated.
- Do NOT feed it to Matrix-3D / Wan: the video model spreads the invention to the whole walk-through.
- Rule from the owner: generation may only fill gaps, never replace what was filmed, and must stay identifiable as generated.
- Safer order: (1) best splat from real frames (`final/run.py`, A/B with held-out PSNR), (2) find holes by rendering off-path views,
  (3) repair renders with a model that edits the render (Difix3D+) instead of inventing a whole panorama, (4) keep a per-pixel/
  per-Gaussian "generated" flag so the viewer can show what was not filmed.
- If a panorama is still wanted: render it from the 200-frame cloud only, leave the unseen pixels as an explicit mask, and have a human
  check it against the video before it goes into any generator.

## Measured so far (real video, 3:45, 4K, Kaggle T4)
- DA3 chunked (8 views, 3 shared, Sim3 merge): all frames placed; 60-frame splat looked better than the point cloud but blurry and broken
  (owner screenshot). Train-view PSNR 25.1 dB (training views, not a quality claim). A/B with 200 frames + camera correction + depth anchor
  is the next measurement (held-out PSNR/SSIM printed by `train_splat.py`).

## 2026-10-25: verified-camera list
- From the all-frame real-pixel check (median 34.3 dB): keep frames whose rebuild is >= 20 dB -> `good_frames.json`, 507/596.
  Dropped stretches 60-63, 89-108, 323-330, 444-447, 526-549 (corridor look-alikes linked by NetVLAD).
- Walk-order-only SfM (`namma-resfm`) was worse: 372/600 registered in 5 models, then OOM in sfm_align (lazy npz reads, fixed).
  So: keep the original SfM, drop the bad cameras. `ibr.py/ibr_check.py --keep`, training via env `KEEP_FRAMES`.
- The list is only valid for the current namma-prep cameras; re-run ibr_check if prep is re-run.
- Run 2 (507 kept): held-out real-pixel rebuild 27.0 -> **30.4 dB, SSIM 0.90**; camera-check median 34.9 dB. But 28 kept frames
  now fail (64-100, 274-294, 318-332, 439-443, 525-552): a misplaced stretch rebuilds itself fine and hides until its worst
  frames go. Frame 75 still shows a door where the photo has the window.
- Bug fixed in ibr.warp: np.round (half-to-even) left every other pixel empty when warped positions were near integers
  (synthetic test: coverage 0.30 -> 1.0). Real runs were less affected, but off-path renders had speckle from it.
- New `ibr_check --prune`: cut the walk where the camera jumps (step > 6x median and > 10 cm), drop short pieces whole, then
  drop < 20 dB and re-score the frames that used them, until clean. Synthetic test with 8 planted misplaced frames: drops exactly those 8.
- Run 3 (`--prune 6` from all 596): walk cut at 36 camera jumps; 319 frames kept, none below 20 dB, median 35.5 dB;
  held-out **30.7 dB / SSIM 0.91**, off-path 30 cm coverage 94%. Held-out images now show the right door/window everywhere.
  Cost: 46% of frames dropped (start 0-51, 60-93, 217-233, 268-294, 317-332, 354-384, 439-452, 489-502, 522-558, 570-599).
  Next: re-place dropped frames against the verified cameras (hloc localisation) instead of losing them.
- `photo_viewer.py` + `photo_viewer.html`: browser walkthrough from real photos (depth mesh per verified frame, nearest frame
  drawn first, others fill only its holes via stencil, depth edges cut). Synthetic box room: at a filmed pose the view matches
  the photo (26.5-26.9 dB on a hard 8-px random texture at 2x upscale); off-pose shows holes as holes.
- Owner, viewer: photos from different angles pop up in the wrong places (left wall appears when going right). The neighbour
  check only compares frames with their own walk-neighbours, so whole passes placed wrongly relative to each other pass it.
  Camera heights of the kept groups differ by up to ~0.75 m (not plausible for a hand-held walk). New `ibr_check --cross`:
  rebuild each frame only from OTHER walk pieces; synthetic test with one piece shifted 36 cm: 13.6 dB vs ~28 dB for correct pieces.
  Viewer now uses one walk pass at a time by default (M mixes passes).
- Cross-pass check on the 319 frames: only 52-59 (28 dB) and 453-488 (26 dB) agree with other passes; the rest 11-18 dB.
  Worst views: the real photo shows the corridor door, the other passes put the window there. The global SfM merged look-alike
  places (symmetric corridor / repeated grilled windows) = the "doppelganger" problem (Doppelgangers++, CVPR 2025, arXiv 2412.05826).
- `align_passes.py`: keep each pass rigid, hypotheses from SIFT + aligned depth (RANSAC similarity) plus the current placement,
  judge each by rebuilding pass A only from pass B's photos, join along best links >= 20 dB. Synthetic: one pass off by 25 deg /
  60 cm -> back within 0.4 cm, cross-pass 13.2 -> 28.5 dB.
- namma-dense: DA3 on the 216 verified frames without depth, 215 scaled from neighbours (1 skipped), viewer 318 photos, 8 min T4.
- align_passes run 1 (103 depth frames): only 333-353 joined 125-216 (20.7 dB, identity) and its cross rebuild still looks wrong
  (bin/pillar misplaced) -> 20 dB too permissive. Bugs: links used one direction only (385-438 <- 333-353 at 27.9 dB was ignored).
  Strong pairs seen: 94-121<-52-59 29.2, 385-438<-333-353 27.9, 52-59<-234-267 27.6, 453-488<-385-438 27.5, 453-488<-125-216 25.7.
  Fixed: both directions, MIN_DB 23, spanning forest = groups (nothing dropped), viewer mixes only within a group.
- align_passes run 2 (318 depth frames): 5 groups [125-216, 333-353, 385-438, 453-488] [52-59, 94-121, 234-267] + 3 singles;
  every accepted link was the current placement (moved 0.0 m): matches between passes never produced a better placement.
  Cross check was also unfair (it mixed groups); now only within a group.
- `--bridge`: follow the video across each gap between consecutive passes (SIFT + depth + PnP, chained over the dropped frames
  that have depth). Frames seconds apart cannot be look-alike confusions. Bridges compete with matches in the same judge; a bridge
  with no visible overlap to judge is trusted (the walk is continuous). Synthetic: 60 cm error found by the bridge; a pass with
  no overlap joined within 2 cm.
- align_passes run 3: all 9 video bridges failed at their first step; groups unchanged; within-group cross check only 15 dB
  for 125-216 / 385-438 (the judge's 4 test frames near matches said 25-28: too local).
  Fixes: (1) frames are 567x1008 but depth 280x504 (different aspect): lift/PnP used one scale for both axes -> per-axis now
  (synthetic with that aspect: matches failed with the old code, bridges 7 cm off); (2) PSNR after per-channel gain matching
  for cross-pass comparisons (auto-exposure differs between passes); (3) judge also tests 4 frames spread over the whole pass;
  (4) looser bridge (ratio 0.8, 15 PnP inliers, 5 frames ahead) and it prints why a step fails.
- align_passes run 4 (aspect fix): bridges 59->94 (0.01 m), 121->125 (1.00 m, judged 38.5 dB), 316->333 (2.14 m), 353->385
  (0.04 m) worked; 216->234 (2 matches), 267->295 / 521->559 (0 PnP inliers), 488->503 (0 matches) failed; 438->453 blew up
  (4e13 m). Cross check: 94-121 9 -> 36.8 dB, frames 125-137 at 37-39 dB. But 'already placed' look-alike links still joined
  passes that disagree with the rest of the group (125-216 vs 385-438 ~16 dB).
  Run 5: bridge sanity (step <= 0.6 m, total <= 5 m), bridges tried first, every join verified against all passes already in the
  group (>= 21 dB; a bridge may join with no overlap). Synthetic: a drifted bridge rejected (19.7), the right link taken.
- align_passes run 5 (bridges first + whole-group check): cross-pass median 15.7 -> **28.2 dB**, frames < 18 dB 142 -> 27.
  Groups: [52-59, 94-121, 125-216, 453-488] (162 frames; 94-121 at 38.4, 125-145 at 34-39) and [295-316, 333-353, 385-438]
  (94 frames, joined by video bridges; 385-438 at 27.9); wrong joins rejected at 13-17 dB. Worst views: the same wall rebuilt
  smaller/farther -> scale differs between passes (bridges are rigid). Run 6: trimmed similarity ICP on every candidate (kept only
  if the whole-group check improves) and whole-group merging. Synthetic: ICP never replaces a better placement (24.3 vs 30.3 kept).
- PROOF 4 run 1 (Wan 2.1 VACE 1.3B, fp16 T4, 49 frames 832x480, 25 steps = 14 min; one pass 453-488 as source, 50 cm off the
  path): holes 18% (max 40%); honesty strip (real plain wall hidden) 35.7 dB; but in the big hole behind the pillar Wan invented a
  glass door, a screen-like object and a small figure (owner: "fake, random stuff"). Rule from now on: real pixels from every
  aligned pass first; the model fills only small gaps (thin slivers / < 0.6% patches); large unfilmed regions are never invented
  (shown 'not filmed', flagged blue); the viewer limits how far off the path one can go (sweep measures holes vs offset).
  Bug fixed: ibr.synth image cache shared across render scales (keyed by size now).
- align run 6 (ICP + group merge): group 0 = 52-59, 94-121, 125-216, 453-488 (162 frames), cross-pass median **30.8 dB**
  (453-488 28.2 -> 30.8 with ICP, scale x0.996, 20 cm); ICP made 94-121 worse (39.2 -> 32.9, not used). Other passes agree with
  group 0 only at 13-20 dB (merges rejected). Frames 172-199 of 125-216 still ~17 dB (inside one pass).
- namma-model: splats trained ONLY on group 0 (cameras that agree), dense depth with real colours, fused init cloud, floor plan,
  photo viewer (mixes all photos by default when there is one group), .splat export (to_splat.py, 32 B/splat).
- 25 Oct evening, owner's viewer data debugged here (namma_debug.zip from align6): group 0 = 162 photos, own-pass rebuild 38.3 dB,
  other-pass rebuild 31.2 dB on 146 photos; only 198-201 marginal (19-22 dB). Group 1 (385-438) disagrees inside (16.4 dB): the
  1.19 m jump at 401/403 (pieces() now also splits at camera jumps). The 'random popping' was coverage, not cameras: neighbouring
  photos drawn as floating cut-out cards with dark gaps. Fix: fused 3D points of the group (viewer_to_splat --viewer,
  cloud_<g>.bin) drawn behind the photos where no photo drew (stencil): views 30-60 cm off the path and turned 40 deg now show the
  whole room continuously. Instant splat (no training) of group 0: 1.79M splats, 57 MB, owner opened it in SuperSplat.
- **namma-model (25 Oct): first good trained model.** Group 0 (162 frames, cameras agree at ~31 dB). A (default densify, SH3,
  depth 0.05, 15k steps / 15 min): **held-out 38.1 dB, SSIM 0.957**, 900k splats, but 37.8% splats stretched > 10x (smears).
  B (MCMC 1.5M, aniso 0.1): 34.2 dB / 0.946, needles 0.1%, only 7.6k steps in 15 min, mean opacity 0.02. Earlier splats on
  disagreeing cameras: 18.8 dB (fog). The camera fix was the missing piece. Floor 21.6 m2 seen, 5.6 m2 walkable.
  Model was tilted 7.9 deg (SfM world not gravity-aligned): to_splat.py now levels with nav.json (floor at y = 0).
- PROOF 4 run 2 (all aligned passes, small gaps only): holes 10.3% at 30 cm, generated 0.8% (slivers), 9.5% shown 'not filmed'
  (nothing invented in the composite), honesty strip 37.8 dB; Wan 49 frames = 19.5 min on a T4. BUG found: sweep gave identical
  holes for one pass vs all 162 photos (2.9/7.8/13/18.5/24.1%): ibr.synth only used the 6 nearest photos, always the same pass.
  Fix (kmax): keep adding photos (any aligned pass) while they still fill holes. Reproduced on the owner's data here: holes at
  30 cm 13.1 -> 7.0%, at 50 cm 24.3 -> 13.7% (real pixels; checked visually: the dark doorway behind the pillar, not misplaced).
  Verdict: video-model filling is honest now but adds little (0.8% of pixels for 20 GPU-minutes); real coverage matters more.
- **26 Oct (night of 25th): namma-all failed, v3 planned and built.** namma-all (ARKitScenes iPad video): Depth Anything 3 ran
  out of GPU memory on landscape frames, merged 5 of 300, every later stage failed on missing depth; SfM placed 168/300 frames in
  2 models. Decision (approved plan): stop compensating with custom code; the default path is now the standard stack: frames ->
  hloc SuperPoint+LightGlue (COLMAP model, SIMPLE_RADIAL) -> gsplat's own examples/simple_trainer.py (MCMC, bilateral grid for
  auto-exposure, SfM depth loss, held-out eval every 8th frame) -> v3_qa.py (QA sheet: held-out real | model, 50 cm off the walk,
  outside, top-down; metrics.json with pass = exposure-fair held-out PSNR >= 28 dB and < 10% needles) -> levelled .splat.
  Job: colab/real_scenes/v3 (preview 7k steps on GPU 1 while the full 30k model trains on GPU 0). Trained classic (not
  antialiased) so SuperSplat shows exactly what was measured. Checked here on CPU: trainer deps import on Python 3.13 + numpy 2
  (pycolmap fork needs np.uint64(-1) patched), Parser reads hloc-style COLMAP binaries incl. SIMPLE_RADIAL undistortion, the
  trainer CLI accepts every flag, depth-loss loader works, QA renders match the real frames (stub renderer), levelled floor y = 0,
  sharpest-frame picker skips blurred frames. Not checkable here: GPU training, fused-ssim build (pure-torch shim as fallback).
- **v3 run 1 (25 Oct 22:05 IST, from GitHub Actions; DL3DV-Benchmark #44 hotel room, handheld phone, 411 frames 960x540):**
  cameras **411/411 registered, 1 model, 1.15 px reprojection error** (hloc SuperPoint+LightGlue, SIMPLE_RADIAL k1 = -0.0007),
  11 min on a T4. gsplat build reused from namma-prep (ready in 1 s), fused-ssim compiled. Training never started: Kaggle's
  preinstalled Hugging Face `datasets` package hid gsplat's examples/datasets folder (namespace package, no __init__.py)
  -> "No module named 'datasets.colmap'". Fixed (touch __init__.py; the job now imports the trainer once and installs whatever
  is missing before training). Kaggle is driven from GitHub Actions now (.github/workflows/namma-v3.yml): this container
  cannot reach kaggle.com.
- **v3 run 2 (25-26 Oct, DL3DV hotel room, 411 handheld phone frames): first genuinely photoreal model.** Cameras 411/411,
  1.15 px. Full model (gsplat simple_trainer MCMC, bilateral grid, depth loss, 30k steps, 68 min on a T4, 1.5M splats):
  **held-out 29.50 dB exposure-fair (25.71 raw), SSIM 0.945, LPIPS 0.120** on 52 held-out frames; preview (7k, 10 min) 26.76 dB.
  Checked by eye: held-out renders near-identical to the real frames, views 50 cm off the walk clean. QA-tool faults found and
  fixed (not yet re-run): needle metric counted flat discs (76%); outside/top-down shots looked into the shell of big splats that
  models the view through the window -> dollhouse crop now. Results: HF dataset anasrazy/namma-v3-out/20261025-1835.
  Conclusion: with a good handheld capture the standard stack works; the owner's own re-filmed room is next.
