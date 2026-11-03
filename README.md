# NAMMA

**NAMMA** is a reproducible pipeline for transforming a well-captured indoor video into a photorealistic Gaussian-splat scene, reviewed geometry, and a web-ready tour. The production path runs on free Kaggle GPUs and is orchestrated by GitHub Actions; the repository also contains a separate laptop-first FastAPI prototype.

> NAMMA reconstructs what the camera observed. Generated content is not part of the default reconstruction path, and unobserved space must not be presented as real.

## Start here

For the production v3 pipeline, follow [test/README.md](test/README.md). For filming guidance, read [docs/CAPTURE_GUIDE.md](docs/CAPTURE_GUIDE.md). The operational hand-off, findings, quality thresholds, and known limitations live in [docs/HANDOFF_V3.md](docs/HANDOFF_V3.md).

## Production pipeline

```text
Indoor video or frame set
        |
        v
GitHub Actions ── stages private input through the Kaggle API
        |
        v
Kaggle GPU kernel ── frames → hloc/COLMAP cameras → 3DGS + 2DGS → QA
        |
        v
Private Hugging Face dataset ── models, metrics, QA sheet, mesh, tour assets
```

The GitHub runner invokes `colab/real_scenes/v3/gh_run.py`, which prepares the input, creates or reuses a private Kaggle dataset, runs the GPU kernel, collects results, and uploads the output set to Hugging Face. The GPU job uses sharp-frame selection, hloc with SuperPoint and LightGlue for camera recovery, COLMAP mapping, gsplat training, and measured visual QA.

## Key capabilities

- **Camera-first reconstruction** — validates registration before training; poor camera poses produce poor splats.
- **Photorealistic appearance** — trains Gaussian splats with exposure-aware optimization and held-out image evaluation.
- **Geometry candidate** — trains a 2DGS variant and produces mesh/collision-oriented artifacts when that stage completes.
- **Evidence-based QA** — outputs held-out render comparisons, PSNR, SSIM, LPIPS, splat-shape checks, off-path views, and walkthrough media.
- **Private cloud workflow** — source videos, Kaggle staging, and Hugging Face outputs remain private when configured with repository secrets.
- **Local experimentation** — retains the FastAPI, SQLite, COLMAP, viewer, live-capture, and navigation prototype for laptop-based development.

## Repository guide

| Path | Purpose |
| --- | --- |
| [colab/real_scenes/v3](colab/real_scenes/v3) | Production GitHub-to-Kaggle orchestration and GPU reconstruction entry points |
| [.github/workflows](.github/workflows) | Production, sample, diagnostic, tour, and support workflows |
| [test](test) | Isolated sample configuration and end-to-end reproduction guide |
| [docs](docs) | Capture guidance, hand-offs, research, quality notes, and project plans |
| [app](app) | Local FastAPI prototype and project/job services |
| [viewer](viewer) | Vite, Three.js, and Gaussian-splat browser viewer |
| [pipeline_cpu](pipeline_cpu) | CPU-oriented utilities and pipeline experiments |
| [research](research) and [notebooks](notebooks) | Research references and exploratory work |

## Run the production sample

The sample follows the same cloud path as a production run but uses an isolated configuration at [test/RUN.sample.json](test/RUN.sample.json). It stages the included lobby video as a private Kaggle dataset, uses a Kaggle GPU, and uploads results to the configured private Hugging Face dataset.

Before running it, configure these repository secrets in GitHub Actions:

| Secret | Purpose |
| --- | --- |
| `KAGGLE_API_TOKEN` or `KAGGLE_TOKEN` | Kaggle API authentication for private dataset and kernel operations |
| `HF_TOKEN` or `HUGGINGFACE_WRITE_TOKEN` | Hugging Face write access for output publication |

Workflow triggers can be branch-pinned in each repository. Before starting a run, inspect the `on.push.branches` setting in the relevant workflow, then use that branch for push-triggered runs or select **Run workflow** in GitHub Actions. Do not assume that pushing `main` automatically starts a reconstruction.

```bash
git clone <this-repository-url>
cd NAMMA

# Review and intentionally bump the sample run counter before committing.
# See test/README.md for the complete, credential-safe workflow.
```

Never commit credentials, access tokens, or private video content. The sample documentation explains how to configure secrets interactively and monitor a run without exposing secret values.

## Capture quality matters

For reliable room reconstruction, record a single uncut video with a clean lens, stable exposure/focus, slow walking motion, overlapping passes, and coverage of each room and doorway. Avoid rapid pans, blank/reflective surfaces, moving people, and dramatic lighting changes. See the full [capture guide](docs/CAPTURE_GUIDE.md) before recording.

## Outputs and quality gates

A run can publish a timestamped result directory containing:

```text
source.json          resolved input information
sfm_report.json      registered-frame, camera, and sparse-point report
full/model.ply       full Gaussian-splat model
full/model.splat     viewer-oriented model, when produced
full/metrics.json    held-out quality metrics
full/qa_sheet.jpg    visual QA comparison sheet
mesh/mesh.glb        geometry artifact, when 2DGS geometry completes
tour/                optional web-tour assets
```

Treat the QA sheet and metrics as release evidence, not decoration. NAMMA’s v3 hand-off defines the current pass criteria, including exposure-fair held-out quality, splat-shape checks, and human inspection of off-path views. A completed job is not automatically a validated scene.

## Local FastAPI prototype

The local service is intentionally separate from the cloud v3 route. It supports video upload, frame extraction, COLMAP reconstruction, quality-gated Gaussian training, live phone capture, collision candidates, reviewed navigation preparation, and the included viewer.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env

Set-Location viewer
npm install
npm run build
Set-Location ..

.\run.ps1
```

Open `http://127.0.0.1:8000/docs` for the API and `http://127.0.0.1:8000/viewer/` for the local viewer. This prototype is not the production Kaggle-backed v3 pipeline.

## Validation

Run the local regression suite after changing the prototype:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\verify_environment.py
```

## License and use

This repository currently has no license file. Do not assume redistribution rights for bundled sample media, third-party models, or generated outputs. Check the applicable source-data and model licences before publishing results.
