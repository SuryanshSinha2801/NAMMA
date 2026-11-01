# PLAN A — "Clay room": zero smears, straight solid walls, furniture as real 3D objects (true to the filmed room)

## Context
The owner wants **literally zero smears or blurs**: the room like a Blender scene, with exact, straight, solid walls (clay
style) and the furniture as proper 3D objects. Gaussian splats can't give that: every splat model we trained (runs 3–5)
smears off the filmed path, and today's tests showed this is not floating junk that could be pruned away:
- CPU pruning: 1.4k of 1.5M splats removed, held-out score unchanged.
- The smudges are in the surfaces themselves.

**Constraints:**
- Free only.
- Kaggle GPU hours are used up until the weekly reset.
- Free CPU: GitHub runners.
- Free GPU: Hugging Face ZeroGPU Spaces, 5 min/day on a free account.

## Evidence gathered (sources)
- **Owner's earlier sources:**
  - **Hyper3D WorldGen** transcript: objects as separate meshes, background separate, "pose correction" onto the room.
  - **Mickmumpitz / Matrix-3D** video: geometry first, then re-project real pixels, generate only what's missing.
  - **Kwang Moo Yi:**
    - Lyra 2.0 — "we still have need for explicit 3D";
    - DepR — depth + instance segmentation + per-object shape model gives a scene layout;
    - Semantic Foam — object decomposition;
    - MP-SfM — depth priors in camera solving.
- **Room shell, straight planes:**
  - [PlanarSplatting (CVPR 2025, Apache-2.0, code ant-research/PlanarSplatting)](https://github.com/ant-research/PlanarSplatting): indoor surfaces as exact planes in ~3 min; runs on COLMAP data; needs CUDA.
  - [SpatialLM (NeurIPS 2025, models on HF)](https://huggingface.co/manycore-research/SpatialLM1.1-Qwen-0.5B): point cloud in, walls/doors/windows + object boxes out. Needs CUDA, and its v1.1 encoder needs flash-attn, which a T4 doesn't run.
  - [PlanarGS (NeurIPS 2025)](https://arxiv.org/abs/2510.23930): planar priors for indoor splats; no code found.
- **Whole-room digital twins:**
  - [DRAWER (CVPR 2025, code xiahongchi/DRAWER)](https://github.com/xiahongchi/DRAWER): phone video → mesh room + articulated drawers for Unreal. Heavy: SDFStudio + Isaac Sim + GPT-4o, multi-GPU-hour.
  - MIT kitchen-from-video notes say the room shell and thin parts are the hard part.
- **Opaque mesh rendering instead of splats:**
  - [Triangle Splatting+ (code)](https://trianglesplatting2.github.io/trianglesplatting2/) and MeshSplatting (CVPR 2026): opaque game-engine meshes.
  - [MILo (SIGGRAPH Asia 2025, Anttwo/MILo)](https://github.com/Anttwo/MILo).
  - All are photometric: blank walls still come out lumpy, and they need hours on CUDA.
- **Furniture as 3D objects from photos:**
  - [TRELLIS.2 (MIT)](https://huggingface.co/spaces/microsoft/TRELLIS.2): free HF Space; the code needs a 24 GB GPU, so not a T4.
  - [Hunyuan3D-2.1 Space](https://huggingface.co/spaces/tencent/Hunyuan3D-2.1): shape fits a T4; texturing needs 21 GB.
  - SAM 3D Objects: SAM licence; already called from `v3/sam3d_gh.py` through a free ZeroGPU Space.
- **Our own proof:**
  - `final/hybrid_demo.py` (26 Oct) built a solid room from the TSDF mesh and scored 13.6–22.5 dB vs splats 25–28.
  - It failed because the geometry came from the noisy TSDF: wavy textures, holes, an outline with 78 corners, and TSDF floor
    bits blocking the floor texture. Its texture step `colour()` itself worked.

## The trial (charge: "this gives zero smears, straight solid walls, real 3D objects")
| Case | Prosecution (why it fails) | Defence | Verdict |
|---|---|---|---|
| A. Keep splats, fix in training (MCMC, depth guide, more frames) | Splats always blur where the camera didn't go; "zero" is impossible by design. Needs GPU we don't have | Best photo look on the filmed path | **Rejected as the main deliverable**; stays as an optional "photo mode" later |
| B. Dense mesh from splats/TSDF + photo textures | Already tried: wavy, holey, 13.6 dB | Free, CPU | **Rejected** (our own proof) |
| C. Opaque-mesh optimisers (Triangle Splatting+, MeshSplatting, MILo) | Geometry still photometric: textureless walls come out lumpy, not straight; hours of CUDA | No floaters, game-ready | **Rejected for walls**; may help small objects later |
| D. Generative worlds (Marble, HY-World 2.0, Lyra 2.0) | Invents the room (not true to the owner's home), needs big GPUs, Lyra's licence forbids public display | Clean and pretty | **Rejected** (owner rule: never invent what was filmed) |
| E. DRAWER end-to-end | Multi-day setup, Isaac Sim, GPT-4o (paid), multi-GPU-hour | Exactly the idea | **Rejected as a tool**; adopt its idea (mesh + real appearance) |
| **F. Structured "clay" room**: walls/floor/ceiling as exact planes from measured geometry; furniture as solid objects (first measured blocks, then generated meshes fitted to the measured size and place); every visible surface textured from the real photos | Generated furniture is an approximation (hidden backs invented). Small clutter is simplified. PSNR may stay below the splats near furniture. Free GPU for generation is ~5 min/day until Kaggle resets | Walls, floor and ceiling are most of every view: exact planes give perfectly straight, sharp, smear-free surfaces by construction. It fixes each cause of B's failure: planes instead of TSDF, Manhattan outline instead of 78 corners, no TSDF occluders. Sizes and positions are measured, not invented. Textures are real pixels. Exports to Blender. The shell runs on CPU today | **Accepted** |

**Ruling:** build F in three phases.
- Phase 1 needs no GPU and delivers the clay room now.
- Phase 2 swaps blocks for real-looking furniture as free GPU allows.
- Phase 3 is optional precision (PlanarSplatting) after Kaggle resets.
- Honesty clause: report held-out PSNR/SSIM against the splat model, plus pictures, and say plainly where F looks worse.

## Ideal outcome (definition of done)
One link opens the room in the browser. The same room is one `.glb` file that opens in Blender.

| Part | Ideal result | How it is judged |
|---|---|---|
| Walls, floor, ceiling | Perfectly flat; straight edges; square corners where the room is square; solid 10 cm walls (dollhouse view looks like a clean model) | Clay render by eye; corner count = real corners (hotel room: 6–8, not 78); flatness 0 mm by construction |
| Size | True to the room: wall lengths and ceiling height within ±3 cm (owner can check with a tape) | Measured vs the splat model's box; for the owner's home, vs 2–3 tape measurements |
| Surfaces | Real photo texture: paint, skirting, switches, wall art, floor pattern sharp; **zero smears, zero floaters from any viewpoint**, including off-path and from above | Off-path + dollhouse renders; texels never seen < 5 % of visible wall area (flagged, filled) |
| Doors / windows | Real openings in the walls; window shows the outside photo flat on the glass | By eye in clay and textured views |
| Furniture | Every big item (bed, sofa, dresser + TV, chair, tables, lamps) is a solid 3D object in the right place and size (±5 cm), real-photo texture on the sides the camera saw, plausible hidden sides | Silhouette overlap with the real frames ≥ 0.6 IoU; size vs measured splats |
| Small clutter | Simple solid shapes or left out (bags, clothes) | By eye |
| Viewer | Orbit + walk mode with collision (can't walk through walls or furniture); clay / textured toggle; loads on a phone (target < 50 MB) | Screenshots on the runner; file size |
| Blender | Each part a separate named object (`Wall_1`, `Floor`, `Bed` …) with UVs and textures | Headless Blender import + one render on the runner |
| Honesty numbers | Held-out PSNR/SSIM of the textured model next to the splat model; on wall/floor pixels at least equal to splats | Report + side-by-side sheet |
| Round 1 | The same pipeline runs on the owner's home video (due 31 Oct) | Same checks |

## Work order (each step must pass its gate before the next starts)
| # | Step | Delivers | Gate | Compute |
|---|---|---|---|---|
| 1 | Shell geometry: floor, ceiling, Manhattan walls, openings | Clay room shell (`shell.glb`) + top-view plan picture | Synthetic L-room → exact 6 walls; hotel room: corner count + size match | CPU runner |
| 2 | Shell textures from real photos | Textured shell | Synthetic planted texture recovered > 30 dB; real: no seams/smears by eye, unseen < 5 % | CPU runner |
| 3 | Furniture blocks (clay) + their real-photo textures | Full clay room + textured room | Blocks inside measured extents; textured off-path views smear-free | CPU runner |
| 4 | Viewer + GLB + Blender check + publish | **First link for the owner** (clay + textured) | Screenshots, Blender render, PSNR sheet vs splats | CPU runner |
| 5 | Furniture → generated 3D meshes fitted into the blocks | Real-looking objects | IoU ≥ 0.6 per object, else the block stays | Free HF Spaces (3–5 objects/day) or Kaggle after reset |
| 6 | Owner's home video through steps 1–5 | Round 1 model | Same gates + tape measurements | Kaggle (camera solve) + CPU |
| 7 | PlanarSplatting refinement of planes | Finer planes (table tops, fronts) | Plane offset vs step 1 reported | Kaggle T4 after reset |

## Plan

### Phase 1 — clay room on CPU (GitHub runner, ~1 h): steps 1–4
New `colab/real_scenes/final/clay_room.py`; runner `v3/clay_gh.py` + `.github/workflows/clay-room.yml` + `v3/CLAY.json` (same
pattern as `v3/fidelity_gh.py`). Input: run 4 (`20261026-0327`): model.ply, sparse, frames, level.json, mesh.glb, and the
per-splat object ids from `objects/20261027-0642/splat_object.npy` (used only to know which splats are furniture).
1. **Shell.**
   - Floor and ceiling heights: `room_prior.planes` (levelled, y up).
   - Walls: RANSAC vertical planes on the non-furniture opaque splats (`objects_refine.wall_planes`).
   - Snap the planes to the 2 dominant directions (Manhattan) and intersect them into a closed floor polygon with few corners.
   - Extrude to the ceiling height.
   - Openings (door, window, bathroom doorway): wall cells where many splats/SfM points lie well behind the plane → cut a hole.
   - Solid walls are 10 cm thick boxes, so the outside of the room also looks solid in the dollhouse view.
2. **Furniture blocks.**
   - For each furniture id: an oriented box, yaw from the room axes, fitted to its splats (2–98 % extents), resting on the floor.
   - Tables, bed and sofa: a 2–3 piece split by height layers (top slab + base), so they read as shapes, not crates.
3. **Real-photo textures.**
   - Every plane and block face gets a texture atlas at 1 cm/texel. It's coloured from the training frames by reusing
     `hybrid_demo.colour()`: best frontal, close, visible view; visibility by Open3D ray casting against the new clean shell +
     blocks only.
   - Exposure-matched blending across 3 views to avoid seams.
   - Texels no photo saw are filled from neighbours and flagged.
4. **Export.**
   - `room.glb`: textured, plus a clay version without textures. trimesh, opens in Blender.
   - A three.js viewer page: orbit + walk with the mesh as collision, plus a clay/textured toggle. Published like the tours
     via `v3/tour_publish.publish()`; link only in the private dataset.
5. **Measure.**
   - Render the GLB at the held-out frames and the off-path cameras (Open3D ray-cast renderer as in `hybrid_demo.render`).
   - PSNR/SSIM vs real photos, next to the splat model (`cpu_render.py`, `tour_fidelity.py` cameras).
   - Geometry: wall flatness (0 by construction), room size vs the splat model's box, corner count.
   - Sheet: real | splat | clay-textured | clay.

### Phase 2 — furniture as real 3D objects (free GPU): step 5
- For bed, sofa, dresser + TV, armchair, side tables, lamps:
  - best crop from the frames;
  - mesh from the free **TRELLIS.2 / SAM 3D Objects Spaces** (gradio_client like `v3/sam3d_gh.py`; ~3–5 objects/day on the
    free quota), or **Hunyuan3D-2.1 shape on Kaggle** in one job after the weekly reset.
- Fit each mesh into its Phase-1 block:
  - scale to the measured extents;
  - yaw by silhouette overlap in 2–3 frames;
  - snap to the floor.
- Re-texture the visible sides with real photos (`colour()`); the generated texture stays only on hidden sides.
- If a generated mesh fits badly (silhouette IoU < 0.6), keep the block.

### Phase 3 — owner's home (step 6) and optional precision (step 7, after Kaggle reset)
Step 6: when the owner's home video arrives, solve the cameras on Kaggle (or the GitHub CPU runner if short), then steps 1–5
unchanged.
Step 7:
PlanarSplatting on run 4's COLMAP data (T4) to refine the wall/floor planes and catch smaller planar surfaces (table tops,
dresser fronts); compare plane positions with Phase 1.

## Reuse
- `final/hybrid_demo.py`: `colour()`, `scene_of()`, `cast()`, `render()`.
- `final/objects_refine.py`: `wall_planes()`, `on_planes()`, `load()`.
- `final/room_prior.py`: `planes()`, `camera_up()`.
- `final/v3_qa.py`: `dollhouse()`.
- `final/make_tour.py`: coordinates (level matrix × metres per unit), `plan()`.
- `final/cpu_render.py` + `final/tour_fidelity.py`: `cameras()` for the comparison.
- `v3/tour_publish.py`: `publish()`.
- `v3/sam3d_gh.py`: ZeroGPU client.

## Verification
- Synthetic test before pushing:
  - an L-shaped room + 2 boxes as points → 6 walls, 6 corners, both blocks;
  - a planted photo texture is recovered on a wall (PSNR > 30 dB on the synthetic).
- Real run (checked by eye before sending):
  - walls perfectly straight in the clay render, corner count reported;
  - textured view without smears from all off-path cameras;
  - held-out PSNR/SSIM next to the splat model, stated honestly;
  - GLB opens in Blender (headless Blender import on the runner, one render);
  - viewer screenshots via `scripts/site_look/look.py`.


# PLAN B — "NAMMA World": any ordinary photo (or video) in -> a generated 3D world out (Marble-like, our own pipeline)
Goal (owner, 27 Oct): upload any photo/video and get a high-quality 3D world that looks 90-95 % like the input (not measured-exact
like Plan A). Built from what we learnt; free; a small example in < 5 min.
- **v0 (now, CPU, ~1 min per photo)** — "walk-in photo":
  - metric depth from Depth Anything V2 (indoor metric);
  - the photo becomes a textured 3D surface, cut at depth jumps so nothing stretches (no smears);
  - LaMa (generative inpainting, CPU) fills what is hidden behind objects as a second layer, and outpaints a border so you can
    look around a little;
  - GLB + three.js viewer.
  - Measured honestly: render at the input view (≈ the photo) and at the real neighbouring video frames (true novel views)
    → PSNR / SSIM ("similarity %").
- **v1 (free GPU: Kaggle after reset or HF Spaces)** — 360° world:
  - generative outpainting of the photo to a full panorama;
  - panorama depth → full room;
  - SDXL/Flux inpainting instead of LaMa for the hidden parts;
  - several frames of a video fused (VGGT/DA3) before generating the rest.
- **v2** — objects as generated meshes (TRELLIS.2 / SAM 3D, as Plan A step 5) inside the generated world.
