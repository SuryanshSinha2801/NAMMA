# Tour quality: owner's problems -> causes -> fixes (26 Oct 2026)

Owner on the first tour (DL3DV hotel room, run 3, tour `20261025-2336-...`): **no walk mode; details could be sharper; smears and
blur at some viewpoints; colours could be better.** Labels: **[M]** measured in this project, **[Q]** quoted from a paper,
**[C]** read in code. Earlier measurements: `docs/OFF_PATH_RESEARCH.md` (19 Oct), the YouTube lessons: `docs/VIDEO_METHOD_PLAN.md`.

## 1. No walk mode
- **Cause [C]**: the SuperSplat Viewer (`@playcanvas/supersplat-viewer@1.37.0`, `dist/viewer.js`) only allows walk mode with a
  collision model AND a scene footprint of at least `WALK_MIN_HORIZONTAL_RANGE = 5` (metres) in both x and z; eyes at
  `eyeHeight = 1.3`, capsule radius 0.2 / height 1.5, `hoverHeight = 0.2`. Run 3's tour had no collision file, and our model is in
  SfM units (room ~3.8 x 3.8 x 1.4 units), not metres.
- **Fix**: (a) metric scale from the camera path: scale so the median phone height above the floor is 1.35 m (handheld height;
  then the viewer's 1.3 m eye height walks exactly where the video was filmed = where the model is best); (b) collision: room
  box (floor + walls aligned to the room's main axes) now, the TSDF mesh (`final/mesh_tsdf.py`) from run 4 on; (c) lower the
  5 m minimum in the viewer file we ship (bedrooms are smaller than 5 x 5 m).

## 2. Smears and blur at some viewpoints (off the filmed path)
- **Measured [M]**: off the path quality drops fast (phone room: ~34 dB on the path, 15-17 dB 18 cm away). Post-hoc cleanups
  of a finished splat cost on-path quality and do not fix off-path views. Run 3: 15.8% needle splats (long in one direction);
  the tour filter removed 15,794 big ones.
- **What fixes it, best evidence first**
  1. **Stay near the filmed path** (what SplatTour does: guided walk, start spots, room crop) and walk at the filmed height
     (fix 1a). Free.
  2. **Capture coverage** (new `docs/CAPTURE_GUIDE.md`: 3 passes per room, ultra-wide). Biggest single lever [M][Q].
  3. **Generative clean-up of novel views, distilled back into the splat**: **Difix3D+** (NVIDIA, CVPR 2025,
     [paper](https://arxiv.org/abs/2503.01774), weights `nvidia/difix_ref`): render views beside the path, clean them with a
     one-step diffusion model (nearest real frame as reference), add them as training views, fine-tune. Measured here on
     19 Oct: +0.37 dB / SSIM +0.021 on our held-out stretch (best splat so far) [M]; paper: +0.8 dB DL3DV, large perceptual
     gains [Q]. Successor: **GSFixer** (ICML 2026, [paper](https://arxiv.org/abs/2508.09667), code github.com/gvclab/GSFixer):
     reference-guided VIDEO diffusion (consistent across frames), reports beating Difix3D+ on DL3DV [Q]; heavier (CogVideoX).
     **"Filling the Unseen"** (arXiv 2609.13262, Sep 2026): quality-aware mask down-weights generated content [Q].
  4. **Indoor floater suppression with depth priors**: **TIDI-GS** (Jan 2026, [paper](https://arxiv.org/abs/2601.09291)),
     StableGS, DN-Splatter, 2DGS/PGSR [Q]. Our gsplat run already uses SfM depth loss; monocular depth priors are the next step.
  5. **The YouTube (Mickmumpitz / Matrix-3D) lesson**: make the unknown explicit (render beside the path, holes = low alpha),
     fill with REAL re-projected pixels first (`final/ibr.py synth(kmax=40)`), generate only what is left with a video model,
     train again; "send in more drones" where it looks bad. = Difix3D+/GSFixer pattern with our real-pixel step first.

## 3. Details could be sharper
- **Cause**: DL3DV `images_4` frames are 960 x 540; a full-screen browser shows ~2x that, so every splat is magnified [M: run 3
  trained at 960 px]. The YouTube lesson: "real high-res pixels wherever they exist" was their biggest jump [Q].
- **Fix**: train from `images_2` (1920 x 1080) at 1600 px (owner's phone video: 4K source -> 1600-1920 px); both T4s
  data-parallel (`simple_trainer` multi-GPU, `--steps-scaler 0.5`) to keep the time near 1.5 h; viewer sharpness post effect
  (amount 0.2, range 0-1) as a light display-side boost.

## 4. Colours could be better
- **Cause**: the first tour shipped base colour only (SOG with `-H 0`), because SOG's spherical-harmonics k-means on CPU takes
  hours [M: 150k splats SH1 208 s, SH3 > 300 s, SH3 with one k-means pass > 5 min]. That drops the view-dependent part of the
  colour (sheen on floors, sheets, glass). Exposure: the bilateral grid trains each frame's exposure separately, so the base
  model has the average exposure [C].
- **Fix**: ship PlayCanvas compressed PLY with SH band 1 (25 bytes/splat, ~37 MB for 1.5M, converts in seconds on CPU [M]);
  later full SH3 SOG compressed on the Kaggle GPU (splat-transform's WebGPU k-means under Vulkan [Q], needs the NVIDIA Vulkan
  driver in the Kaggle container, unverified). No colour grading by default (true to the room).

## 5. Order
1. Tour v2 on run 3 (no retraining, ~15 min): metric scale, collision (room box), walk mode allowed, SH1 colour, sharpness.
2. Run 4: 1600 px from `images_2`, two GPUs, mesh at the end (collision with furniture), same QA.
3. Difix3D+ stage after training (`scripts/gsplat_difix.py` adapted to simple_trainer checkpoints): views 30-60 cm beside the
   path at walk height, real-pixel re-projection first, Difix for the rest, fine-tune; compare QA off-path before/after.
4. GSFixer if Difix3D+ leaves visible smears and the T4 fits it.

Sources: [Difix3D+](https://arxiv.org/abs/2503.01774), [GSFixer](https://arxiv.org/abs/2508.09667),
[TIDI-GS](https://arxiv.org/abs/2601.09291), [Filling the Unseen](https://arxiv.org/abs/2609.13262),
[EVPGS](https://arxiv.org/abs/2503.21816), [Robust Gaussian Splatting](https://arxiv.org/abs/2404.04211),
[splat-transform SOG speed-up](https://radiancefields.com/playcanvas-splat-transform-v3.9.0-speeds-up-sog-compression),
[SOG format](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/).

## 6. Round 2 (26 Oct, after tour v2): what is still bad and what fixes it
Tour v2 checked on screenshots: walk mode works, the guided walk is photoreal, but **streaky smears on the floor, ceiling and
everything off the filmed path**, and the **floor-plan minimap is unreadable** (black with sparse dots).

**What the newest evidence says**
- **Off-path quality is a coverage problem first [Q]**: "Mind the Gap" (arXiv 2607.01556, Jul 2026): standard held-out tests
  (every 8th frame) only measure near-path interpolation; real off-path views are **3-12 dB worse**, the gap tracks the angle to
  the nearest training view, and **loss-side regularisation gives only marginal gains**. Our own 19 Oct numbers agree (34 dB on
  the path, 15-17 dB 18 cm away). So: more filming coverage and generated views fix off-path; regularisers fix shapes.
- **Floor/ceiling streaks = textureless planes + few views [Q]**: indoor 3DGS papers all add a geometric prior there:
  DN-Splatter (WACV 2025, depth + normal priors), GaussianRoom, **PlanarGS** (Oct 2025, MIT code, GroundedSAM + DUSt3R planes,
  1 h on an RTX 3090), **UncertainGS** (Feb 2026, Manhattan-world normals with uncertainty). The streaks themselves are needle
  splats (effective rank 1): **effective-rank regulariser** (Hyung et al., NeurIPS 2024) pushes them to discs.
- **Generative clean-up, licences checked [C]**: Difix3D+ (measured +0.37 dB here) is **NVIDIA licence = non-commercial only**,
  and it was **fine-tuned on DL3DV**, so on our DL3DV test scene it will look better than on the owner's own videos.
  **FreeFix** (Jan 2026, **MIT code** github.com/hyzhou404/FreeFix): no fine-tuning, any image diffusion model (SDXL or Flux),
  interleaved render -> fix -> retrain per view with a per-pixel confidence mask; reports beating Difix3D+ on LLFF/Mip-NeRF 360.
  Heavier (SDXL on a T4 fits in fp16; Flux does not). GSFixer: video diffusion, heaviest.
- Monocular priors if we go further: **MoGe-2** (MIT code, metric depth + normals), Depth Anything V2 **Small** (Apache; Base/
  Large are non-commercial), DSINE (restrictive Imperial licence).

**Done now (code ready, measured on synthetic data; real numbers come from run 5)**
1. **Room prior** (`final/room_prior.py`, variant kind `prior`): needle penalty (longest/middle > 6, = effective-rank idea in our
   QA's terms) + floor/ceiling planes (floor height from SfM points, up from the cameras; splats within ~5 cm of a plane are
   turned flat to it). No extra network. Synthetic room: needles 9.4 % -> 0 %, floor-splat tilt 61 deg -> 0 deg. Patched into
   gsplat's simple_trainer by run.py, off unless the variant asks for it, so run 5 trains `full` and `prior` side by side.
2. **Floor plan** (`make_tour.floorplan`): the model seen from above, ceiling and wall tops cut off, each pixel the highest splat
   in its own colour, walls dark on top; redrawable without a GPU (`tour-publish.yml` with `"floorplan": true`).

**Next, in order**: run 5 = run 4's settings + `prior` variant (A/B on the same frames, QA off-path views + needle %);
Difix3D+ stage for the research demo (licence!) and FreeFix + SDXL as the commercial-safe path; MoGe-2 depth/normal loss if the
cheap prior is not enough; the owner's own re-filmed room (coverage beats every loss term).

Sources: [Mind the Gap](https://arxiv.org/abs/2607.01556), [FreeFix](https://arxiv.org/abs/2601.20857),
[PlanarGS](https://arxiv.org/abs/2510.23930), [UncertainGS](https://www.sciencedirect.com/science/article/abs/pii/S0925231225028085),
[Effective rank](https://arxiv.org/abs/2406.11672), [DN-Splatter](https://arxiv.org/abs/2403.17822),
[MoGe-2](https://arxiv.org/abs/2507.02546).

## 7. "Use a ChatGPT/Gemini-class image AI to fix the smears" (owner's idea, 26 Oct)
**Problem [Q]**: an image editor fixes each picture on its own; two neighbouring views get two different "fixes", and training on
both makes the splat blurry again (FreeFix, 3DGS-Enhancer, GSFixer all name multi-view consistency as the main difficulty).
**What makes it work (combined from Difix3D+, FreeFix, GSFixer, ExploreGS)**:
1. Give the editor the **nearest real photos as references** (it copies real detail instead of inventing).
2. Only let it change the **unreliable pixels** (confidence mask from the render; real reprojected pixels kept), composite back.
3. Fix views **one at a time along a path and retrain in between** (each fixed view constrains the next = consistency), with
   low weight and a per-view colour correction (FreeFix).
4. Place the extra views where the walker goes but the video did not (ExploreGS: information-gain camera placement, 2025).
**Editors**:
- **Qwen-Image-Edit-2511** (Alibaba, **Apache 2.0, open weights**, ChatGPT/Gemini-class editing): takes several input images
  (render + real reference photos), new-viewpoint ability built in, 4-step Lightning LoRA. 20B parameters: on Kaggle's T4 only
  as a 4-bit/GGUF build; quality and speed on a T4 **unverified** (community reports of artefacts at very low bits) -> smoke test.
- ~~Gemini / OpenAI image APIs~~: **ruled out by the owner (26 Oct): free and open-source only, nothing paid.**
- Difix3D+ (baseline we measured, non-commercial), FreeFix (MIT, SDXL/Flux, no fine-tuning), GSFixer (video model, heavy),
  ArtifactWorld (Apr 2026, video model trained on 107k artefact clips; code "to be released").
**Proof**: hold out a contiguous stretch of the video (not every 8th frame: "Mind the Gap" shows that only tests the path) and
measure that stretch before/after; plus the off-path QA views.
**Plan (`final/fix_views.py`, after run 4)**: render ~60 views 30-60 cm beside the path at walk height -> mask -> editor with 2
nearest real photos -> composite -> 3 rounds of fine-tuning (weight 0.3, colour affine) -> QA on the held-out stretch.
Smoke test first: 5 views through Qwen-Image-Edit-2511 on a T4 (time per image, does it keep the layout). If it does not fit or is
too slow: FreeFix with SDXL (open), then Difix3D+ (free, but its licence is non-commercial: research demo only).

Sources: [FreeFix](https://arxiv.org/abs/2601.20857), [GSFixer](https://github.com/GVCLab/GSFixer),
[ExploreGS](https://arxiv.org/abs/2508.06014), [ArtifactWorld](https://arxiv.org/abs/2604.12251),
[FixingGS](https://arxiv.org/abs/2509.18759), [3DGS-Enhancer](https://nips.cc/virtual/2024/poster/95333),
[GSFix3D](https://arxiv.org/abs/2508.14717), [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511),
[Lightning LoRA](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning).

## 8. "A Marble-like product: upload photos/video, get a full photoreal room" (owner, 26 Oct)

**What Marble does** (World Labs, architecture unpublished): text / 1 image / panorama / 2-8 photos / short video in,
Gaussian-splat world out (+ mesh, .spz export, Chisel editing). Generative: it invents what the input does not show.

**What we measured so far** (all on run 4, hotel room): per-image AI fixes do not work for us. Qwen-Image-Edit drifts the
view and copies objects (28 -> 16 dB). FreeFix (masked SDXL, per image) raises held-out PSNR 25.84 -> 27.11 but makes texture
softer (SSIM 0.925 -> 0.902, LPIPS 0.203 -> 0.249) and leaves the smears. SAM 3D Objects gives clean solid furniture, but grey
and approximate. Common cause: each image is fixed on its own, so the fixes disagree and the splat averages them into blur.

**Merge that fits the evidence and the rules (free/open, T4)**: keep the trained splat as the truth where the video saw the
room, and let a *video* model fill only what it did not see, many frames at once so the fill is consistent:
1. **Poses + geometry**: hloc SfM (now). Fallback and fast preview for "ordinary photos": WorldMirror 2.0 (Tencent HY-World 2.0,
   1.2B, feed-forward camera + depth + 3DGS in one pass, accepts camera/depth priors; free Tencent licence, not EU/UK/KR, >1M
   monthly users needs Tencent's OK).
2. **Photoreal core**: gsplat MCMC on the real frames (run 4/5) -> run 6 (video frames, 1920 px, ~3M splats). Candidate kernel
   upgrade: Deformable Beta Splatting (SIGGRAPH 2025, Apache-2.0, authors report better than 3DGS-MCMC with 45% of the params).
3. **Generative fill ("the Marble part")**: render the splat along walk paths through the bad spots; mask = pixels no real photo
   supports (low alpha / FreeFix-style certainty); **Wan2.1-VACE-1.3B** (Apache-2.0, ~8 GB, 480p, video inpainting with masks)
   fills the masked pixels across the whole 81-frame clip at once; retrain with real frames (full weight) + filled frames
   (masked pixels only, low weight). Real pixels are never changed, so held-out frames must not get worse (same proof as before).
4. **Solid objects** (optional, Round 2): SAM 3D Objects for collision/search, not for looks.
5. **Product**: free Gradio Space upload form -> private HF dataset -> GitHub Action -> Kaggle T4 job -> tour link. Not instant
   like Marble (H100s): ~2-3 h per room, Kaggle's 30 GPU h/week = ~10 rooms/week.

**Rejected**: Lyra 2.0 (NVIDIA internal-research licence: outputs may not be shown publicly), Difix3D+ (non-commercial),
GaussFusion (CVPR 2026, no code yet), ArtifactWorld (gated weights, licence unknown), GSFixer (CogVideoX-5B, licence unstated,
tested on H100), WorldStereo 2.0 (17B, too big for a T4).

Sources: [Marble overview](https://help.scenario.com/articles/4774055514-marble-by-world-labs-the-essentials),
[HY-World 2.0](https://huggingface.co/tencent/HY-World-2.0), [Lyra 2.0 card](https://huggingface.co/nvidia/Lyra-2.0),
[Wan2.1-VACE-1.3B](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B), [Beta Splatting](https://github.com/kristofe/beta-splatting),
[GaussFusion](https://arxiv.org/abs/2603.25053v2), [ArtifactWorld](https://arxiv.org/pdf/2604.12251),
[GSFixer](https://arxiv.org/pdf/2508.09667), [worldsplat](https://github.com/timscheuerai/worldsplat).

### 8b. Re-ranked after "personal project, not commercial" (owner, 26 Oct): non-commercial licences allowed

**Update 26 Oct: Difix3D+ tested twice on run 4 and dropped by the owner ("not worth it"): v2 was safe (held-out slightly
better, walls clean) but barely touched the off-path smears, at 9 s per view on a T4. AI view-fixing is off the plan.**

Compute is still the limit (free Kaggle T4, 16 GB each). Best fixer that fits, in order:
1. **Difix3D+** (NVIDIA, CVPR 2025 oral, research licence): single-step SD-Turbo fixer, ~1 s per view on a T4, made for exactly
   this (artifacts in novel views of 3DGS), code built on gsplat. Paper on DL3DV novel trajectories: 3DGS 17.18 -> 17.99 dB,
   LPIPS 0.384 -> 0.293, FID 107 -> 41. Running now on run 4 (`difix/<time>/`). Caveat: trained on DL3DV-Benchmark scenes,
   run 4's room may be among them; the owner's home is the fair test.
2. **GSFixer** (ICML 2026, CogVideoX-5B video fixer + VGGT features, weights released): beats Difix3D+ and GenFusion on most
   DL3DV-Res scenes in its paper; video model = consistent across views. 5B on a T4 is slow (offload); next if Difix is not enough.
3. **ArtiFixer3D+** (NVIDIA, SIGGRAPH 2026, Wan2.1-14B based, ~17B params, research licence): claims the best published results;
   too big for a free T4 (needs a 40-80 GB GPU). Out until we have such a GPU for free.
4. **Fixer** (NVIDIA, Cosmos 0.6B, Nov 2025, open model licence): Difix's successor, but trained on driving data (3DGUT); Turing
   (T4) not in its tested list. Possible A/B against Difix later.
Also now allowed: Dr. Splat's own code for Round 2 search (Inria NC), MASt3R / VGGT for hard captures, Depth Anything V2-L and
DSINE normals as depth/normal priors (DN-Splatter-style) against wall and floor smears.
Still out: Lyra 2.0 (licence forbids showing results publicly; also H100-class).

Sources: [Difix3D+ paper](https://arxiv.org/pdf/2503.01774), [Difix3D code](https://github.com/nv-tlabs/Difix3D),
[difix_ref card](https://huggingface.co/nvidia/difix_ref), [Fixer card](https://huggingface.co/nvidia/Fixer),
[GSFixer](https://github.com/gvclab/GSFixer), [ArtiFixer paper](https://arxiv.org/pdf/2603.00492),
[ArtiFixer card](https://huggingface.co/nvidia/ArtiFixer), [DN-Splatter](https://arxiv.org/html/2403.17822v3),
[TIDI-GS](https://arxiv.org/pdf/2601.09291).

## 9. Hyper3D WorldGen (owner's video, 26 Oct) -> NAMMA object layer

WorldGen (Deemos, Aug/Sep 2026; built on CAST, SIGGRAPH 2025 best paper): one image -> each object a complete Rodin mesh
(automatic or pin selection, regenerate/edit, "pose correction", optional physics for USDZ) + a 3DGS background; exports to
Blender (script + proxy object), Unity, Unreal. Paid (Rodin credits, 2-week trial, no public WorldGen API), CAST has no official
code, and from one photo the background and hidden sides are invented, so it is a reference, not a tool for a true-to-room tour.

What we copy for free (approved plan, after run 6): keep the real splat for looks and add an object layer. Objects are found
without needing names (segment-everything masks voted onto splats + 3D clusters off the floor/wall planes), named best-effort by a
vision-language model (else "object N" + photo) with an appearance embedding for search. A proxy mesh per object (SAM 3D /
Hunyuan3D fitted by ICP to the object's real points and snapped to the floor; the measured TSDF/hull shape if the fit is off by
more than 5 cm) is used for collision, clicking, moving and hidden sides, never drawn over the real splats. Export: background
.ply + per-object .ply/.glb + labels JSON (Blender via KIRI 3DGS Render, Apache). A whole house works room by room: one camera
solve, rooms trained separately with door overlap (2 at a time on Kaggle's 2 T4s, ~8-10 GPU h per 5-room house), one file per
room streamed in the viewer.

Sources: [WorldGen on X](https://x.com/DeemosTech/status/2094440163246256523),
[launch article](https://finance.yahoo.com/technology/ai/articles/hyper3d-launches-worldgen-turn-single-120600509.html),
[Rodin pricing](https://hyper3d.ai/pricing), [CAST](https://arxiv.org/abs/2502.12894v2),
[KIRI 3DGS Render](https://radiancefields.com/kiri-engine-ships-3dgs-render-5.0-for-blender).
