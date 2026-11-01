# What Kwang Moo Yi (@kwangmoo_yi) shares, and what it means for NAMMA (27 Oct 2026)

Kwang Moo Yi is an associate professor at UBC (computer vision lab). He co-authored 3DGS-MCMC (the "mcmc" strategy inside
gsplat), StochasticSplats, Radiant Foam and Semantic Foam. Nearly every day he posts one paper with a one-line take.

How this was collected: x.com itself cannot be opened from this container, and logged-out X shows no timeline. The posts below
were found through web-search indexes of his public posts (about 55 posts, Jan 2025 to Sep 2026). It is a sample, not his whole
feed. Each line gives the paper, his take (quoted or close to it), and the post link.

## A. Directly about our problems (floaters, smudged walls, smears, densification)
| Paper | His take | Post |
|---|---|---|
| **3DGS as Markov Chain Monte Carlo** (Kheradmand ... Yi, NeurIPS'24, his own) | Gaussians are samples; split/clone/prune is replaced by *relocating* dead (low-opacity / tiny) Gaussians plus noise on positions; fixed Gaussian budget | [repo](https://github.com/ubc-vision/3dgs-mcmc) |
| FeatureGS (Jäger et al.) | "Aligns Gaussian shapes with how neighbouring Gaussians are arranged via eigenvalues. Byebye floaters!" | [post](https://x.com/kwangmoo_yi/status/1885061091837043093) |
| FastGS (Ren, Wen et al.) | "I like simple ideas: consider multiple views when you prune/clone": fewer Gaussians, 100 s training | [post](https://x.com/kwangmoo_yi/status/1986864275374416143) |
| MoE3D (Wang et al.) | Flying pixels in DPT-type depth models (DA2/DA3) come from the convolutional head; MoE "sort of bilateral filtering" | [post](https://x.com/kwangmoo_yi/status/2009707804848992628) |
| StochasticSplats (his own, ICCV'25) | Sorting-free stochastic rasterisation: no popping, fast, more samples per pixel = better quality | [repo](https://github.com/ubc-vision/stochasticsplats) |
| Radiant Foam (his lab) | Volumetric ray tracing with a Voronoi "foam" mesh, faster than rasterising splats | [post](https://x.com/kwangmoo_yi/status/1886836311899955656) |
| Dual Covariance GS-SLAM (Tan et al.) | Geometry uncertainty and appearance differ; separate covariances help | [post](https://x.com/kwangmoo_yi/status/2103560947487175138) |

## B. Cameras / SfM (our hall video lost tracking on blank walls and fast turns)
| Paper | His take | Post |
|---|---|---|
| MP-SfM (Pataki et al.) | "Depth estimators work great. Time to integrate them into SfM pipelines." | [post](https://x.com/kwangmoo_yi/status/1917282752065593512) |
| Marginalized Bundle Adjustment (Zhu et al.) | Camera poses from monocular depth; often better for SfM, not yet best on phototourism | [post](https://x.com/kwangmoo_yi/status/2027105682470969441) |
| RoMa v2 (Edstedt et al.) | 1.7x faster dense matching; still beats feed-forward 3D methods by far outdoors | [post](https://x.com/kwangmoo_yi/status/1991643996905697504) |
| Can these views be one scene? (Paul, Kaushik et al.) | 3D-consistency metrics built on "modern" models do NOT match humans; COLMAP-based ones do. "You can never escape COLMAP" | [post](https://x.com/kwangmoo_yi/status/2057158432835387645) |
| GGPT (Chen et al.) | Feed-forward -> RoMa -> sparse BA for accurate SfM, then point-transformer refinement | [post](https://x.com/kwangmoo_yi/status/2034060366821659039) |
| Long-Tail Internet Photo Recon (Li et al.) | Many feed-forward estimators are COLMAP distillations; data prep matters | [post](https://x.com/kwangmoo_yi/status/2049306266359767226) |
| VGGT (Wang et al.) | DINO + camera tokens, alternating frame/global attention -> pose, point maps | [post](https://x.com/kwangmoo_yi/status/1901707903301746840) |
| VGGT-SLAM / 2.0 (Maggio, Carlone) | VGGT + SLAM, SL(4) for projective ambiguity; layer 22 gives correspondences for overlap tests | [v1](https://x.com/kwangmoo_yi/status/1925317715969155382), [v2](https://x.com/kwangmoo_yi/status/2016576178430353841) |
| SAIL-Recon (Deng, Li et al.) | Large-scale feed-forward SfM on VGGT with anchor images | [post](https://x.com/kwangmoo_yi/status/1960476458553631176) |
| MapAnything (Keetha et al.) | Multi-modal in/out feed-forward metric 3D; beats the others | [post](https://x.com/kwangmoo_yi/status/1968754722409533698) |
| MUSt3R (Cabon et al.) | Multi-view DUSt3R with memory | [post](https://x.com/kwangmoo_yi/status/1899899766949240996) |
| CAM3R (Guruprasad et al.) | Camera model encoded as spherical harmonics: one model for any lens | [post](https://x.com/kwangmoo_yi/status/2036984133977858414) |
| AllTracker (Harley et al.) | Dense point tracking of all pixels in real time, high resolution | [post](https://x.com/kwangmoo_yi/status/1936113081366855934) |

## C. Fixing / filling what the camera did not see
| Paper | His take | Post |
|---|---|---|
| FlowR (Fischer et al.) | "Render, then fix with a multi-view diffusion model. Like Difix3D+, but multi-view." | [post](https://x.com/kwangmoo_yi/status/1907882889435558315) |
| M2SVid (Shvetsova et al.) | Depth -> splat to the new view -> inpaint only dis-occluded parts: "only generate what you don't know" | [post](https://x.com/kwangmoo_yi/status/1926041097316229429) |
| Lyra 2.0 (NVIDIA) | Explicit 3D memory + 3D foundation models: "we still have need for explicit 3D" | [post](https://x.com/kwangmoo_yi/status/2044973417842311533) |
| Seedance 1.0 / Veo 3 | "Once you see the 3D inconsistencies in them, you'll get dizzy like me" | [post](https://x.com/kwangmoo_yi/status/1934713279722738034) |
| RLGF (Yan, Han et al.) | Video "world models" make many geometric errors, even the best; RL with geometry to fix | [post](https://x.com/kwangmoo_yi/status/1970563517855211805) |
| CamCtrl3D (Popov et al.) | Video models (SVD) are a great 3D prior with camera control | [post](https://x.com/kwangmoo_yi/status/1878890026597257423) |
| Geometric Consistency Refinement (Bengtson et al.) | Change the initial noise of novel-view diffusion with epipolar geometry | [post](https://x.com/kwangmoo_yi/status/1911906318744756639) |
| MV-RAG (Dayani et al.) | Retrieval-augmented multi-view diffusion: "many things look alike" | [post](https://x.com/kwangmoo_yi/status/1960062930449486248) |
| Noise selection papers (Kim & Kim; Li et al.) | "Initial noise matters": pick seeds by attention | [post](https://x.com/kwangmoo_yi/status/1927099206281441635) |

## D. Objects / scene decomposition (our object layer)
| Paper | His take / key point | Link |
|---|---|---|
| **Semantic Foam** (his lab, CVPR'26) | Per-cell semantic features on Radiant Foam's Voronoi cells; spatial regularisation fixes artefacts from occlusion and inconsistent masks across views; equal or better than Gaussian Grouping / SAGA; clean non-convex object masks | [paper](https://arxiv.org/abs/2604.26262) |
| DepR (Zhao, Zhang et al.) | Depth + instance segmentation + SDF diffusion -> scene layout | [post](https://x.com/kwangmoo_yi/status/1950980460018016427) |
| Can3Tok (Gao et al.) | Normalise scenes before tokenising 3DGS | [post](https://x.com/kwangmoo_yi/status/1952779060003913794) |
| Dense Semantic Matching with VGGT prior (Yang et al.) | A foundation model made for matching beats "any" foundation model | [post](https://x.com/kwangmoo_yi/status/1971646997360005538) |

## E. Feed-forward splats (instant previews, e.g. an upload page)
SparseSplat ([post](https://x.com/kwangmoo_yi/status/2042804312364192091)): splats from 3D points and anchors, not one per pixel.
TokenGS ([post](https://x.com/kwangmoo_yi/status/2045300417014485154)): DETR-style tokens, with test-time adaptation.
Both still give lower quality than an optimised splat.

## What this means for NAMMA (ranked; all free)
1. **Switch training to his own MCMC strategy** (`simple_trainer.py mcmc` in gsplat 1.5.3, `--strategy.cap-max` = our
   splat budget). It addresses our floaters/smears at the root: no clone/split blow-up and dead splats are recycled. It is a
   flag change, keeping our room prior / geo prior patches. A/B it against `default` on run 4's frames when GPU hours come
   back (Kaggle's weekly GPU quota resets each week).
2. **Plain walls: FeatureGS-style shape alignment** (flat Gaussians aligned with their neighbours). This is a small loss term,
   like room_prior.py. Together with our geo prior (DA3 depth/normals) it targets the smudged walls. MoE3D warns that DPT
   depth has flying pixels at edges; geo_prior.py already skips depth edges.
3. **Multi-view pruning (FastGS idea)**: prune/clone only what several training views agree on. Fewer splats and faster
   training, which matters now that GPU hours are short.
4. **Cameras for the owner's home**: if COLMAP/hloc loses track on blank walls, add monocular-depth-aided SfM (MP-SfM,
   Marginalized BA) or RoMa v2 matching. Keep a COLMAP-based check as the judge (his point: modern-model metrics do not agree
   with people; COLMAP does).
5. **Generative fill**: his takes confirm the owner's call. Video/diffusion models are geometrically inconsistent. If ever,
   generate only what was never seen (M2SVid), multi-view (FlowR), and keep explicit 3D (Lyra 2.0).
6. **Object layer v3**: the Semantic Foam / Gaussian Grouping lesson is that object ids must be regularised in 3D (neighbours
   agree), not only voted per frame. Our v2's mixed objects and wrong merges come from per-frame votes. Next: a 3D
   neighbour-smoothing step on the splat ids (CPU, no GPU), merge only on look + overlap, plus the naming fixes in
   HANDOFF_V3.md.
