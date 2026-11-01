# NAMMA Round 1 — setup, sample data and reconstruction

This folder is the reproducibility package for the Techfest Namma Space Round 1 repository deliverable. It contains a sample indoor video, a safe sample run configuration and the steps needed to reproduce the v3 reconstruction pipeline.

## 1. Repository and pipeline

- Repository: https://github.com/RazyAnas/NAMMA
- Branch used for this pipeline: `claude/wonderful-cori-i3fn41`
- Sample input: [`sample_data/lobby_clip30.mp4`](sample_data/lobby_clip30.mp4)
- Sample configuration: [`RUN.sample.json`](RUN.sample.json)
- Capture procedure: [`docs/CAPTURE_GUIDE.md`](../docs/CAPTURE_GUIDE.md)
- Video-to-tour pipeline: [`colab/real_scenes/v3/run.py`](../colab/real_scenes/v3/run.py)
- GitHub runner: [`colab/real_scenes/v3/gh_run.py`](../colab/real_scenes/v3/gh_run.py)
- Workflow: [`.github/workflows/namma-v3.yml`](../.github/workflows/namma-v3.yml)

The production v3 path is:

`phone video → sharp-frame extraction → camera recovery (hloc + COLMAP) → 3DGS appearance model + 2DGS geometry model → mesh/collision export → QA and web-tour publishing`

The training pipeline runs on Kaggle, not on a normal laptop. GitHub Actions prepares the Kaggle job, attaches the input dataset, waits for the run, collects the outputs and uploads the results to Hugging Face when a write token is configured.

## 2. What is included in the sample folder?

`test/sample_data/lobby_clip30.mp4` is a short lobby video already maintained in this repository. The same Git blob is also present at `colab/real_scenes/videos/lobby_clip30.mp4`; it is exposed here as a Round 1 test input so judges do not have to search through the research folders.

This is a **pipeline smoke-test input**, not the team's own-home competition capture and not the source of the whole-house results in the technical abstract. Use your own indoor video, recorded according to the capture guide, for the primary Round 1 demonstration.

The repository does not attach a separate licence statement to this MP4. Do not label it as an openly licensed dataset; confirm the recording's reuse rights before redistributing it outside this repository.

## 3. Local checks (no GPU training)

These commands check out the code, install the local Python dependencies, run the automated tests and extract a sample set of sharp frames. They do not run the full 3DGS/2DGS training job.

### macOS / Linux

```bash
git clone --branch claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA.git
cd NAMMA

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Run the automated API / pipeline unit tests.
python -m pytest -q tests

# Extract 120 sharp frames, resized to a 960-pixel long side.
python colab/real_scenes/final/frames.py test/sample_data/lobby_clip30.mp4 /tmp/namma-sample-frames 120 960
```

### Windows PowerShell

```powershell
git clone --branch claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA.git
Set-Location NAMMA

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m pytest -q tests

python colab/real_scenes/final/frames.py test/sample_data/lobby_clip30.mp4 "$env:TEMP\namma-sample-frames" 120 960
```

## 4. Full sample reconstruction on Kaggle

The cloud reconstruction uses a Kaggle GPU kernel. A Kaggle account with NVIDIA T4 GPU access and internet enabled is required. The GitHub Actions runner needs the credentials below; **never put tokens in a committed JSON file**.

### 4.1 Create the Kaggle input dataset

1. Open Kaggle → Datasets → New Dataset.
2. Upload `test/sample_data/lobby_clip30.mp4` at the dataset root.
3. Set the dataset visibility to **Private**.
4. Use a simple slug such as `namma-test-sample`. Your dataset ID will look like `<your-kaggle-username>/namma-test-sample`.

The sample configuration refers to that dataset through `source`. The Kaggle upload is a one-time preparation step; subsequent runs can reuse the same private dataset.

### 4.2 Configure GitHub Actions secrets

In the repository, open **Settings → Secrets and variables → Actions** and add:

| Secret | Purpose |
|---|---|
| `KAGGLE_API_TOKEN` | Kaggle API token for the account that owns or can read the private sample dataset. The existing runner also accepts `KAGGLE_TOKEN`. |
| `HF_TOKEN` | Hugging Face write token used to store run outputs in `<HF username>/namma-v3-out`. The runner also accepts `HUGGINGFACE_WRITE_TOKEN`. |

The Kaggle token is obtained from Kaggle account settings → API. The Hugging Face token needs write permission to the account's results dataset. If the Hugging Face token is omitted, the run may remain available through Kaggle output, but the private Hugging Face results upload and hosted-tour publishing will not complete.

The Kaggle kernel metadata is already configured by the repository for a private script kernel with GPU and internet enabled. The Kaggle account still needs to be eligible for GPU sessions and have remaining quota.

### 4.3 Prepare the sample run configuration

1. Copy [`RUN.sample.json`](RUN.sample.json) to `colab/real_scenes/v3/RUN.json`.
2. Replace `<YOUR_KAGGLE_USERNAME>` in `source` with the owner part of the Kaggle dataset ID.
3. Change `n` from `0` to a positive run counter (for example, `1`). A value of `0` is deliberately a no-op.
4. Keep the sample kernel name `namma-r1-sample`; it avoids overwriting the main `namma-v3` production kernel.
5. Save the current `RUN.json` first and do not start the sample while another Kaggle job is using the same account/GPU quota.

Example `source` value:

```json
"source": "kaggle:anasrazy/namma-test-sample"
```

The supplied quick configuration requests 120 frames at 960 pixels and short 3,000-step 3DGS and 2DGS runs. This keeps the sample run smaller than a full-quality house reconstruction. It is intended to demonstrate that the pipeline can ingest the video, recover cameras, train the appearance and geometry models, and export reconstruction outputs; it is **not** a quality benchmark.

### 4.4 Start the reconstruction

Commit and push the updated `RUN.json` to the configured branch. The existing `namma-v3` workflow watches this exact file.

```bash
git add colab/real_scenes/v3/RUN.json
git commit -m "Run NAMMA Round 1 sample reconstruction"
git push origin claude/wonderful-cori-i3fn41
```

Then open the repository's **Actions** tab and select the `namma-v3 on Kaggle` run. The workflow:
1. validates Kaggle GPU availability;
2. prepares and pushes the Kaggle kernel with the sample dataset attached;
3. monitors the kernel until it completes (longer jobs use collection jobs);
4. downloads the output files and uploads them to the private Hugging Face results dataset when `HF_TOKEN` is configured;
5. attempts to build and publish a web tour when the required model and camera outputs are present.

## 5. Expected outputs

A successful run creates a timestamped folder under:

`https://huggingface.co/datasets/<HF_USERNAME>/namma-v3-out/tree/main/<RUN_ID>/`

The main outputs include:

| Output | What it contains |
|---|---|
| `sfm_report.json` | Number of extracted and registered frames, camera-model count, point count and reprojection information. |
| `full/model.ply` | Full-colour 3D Gaussian Splatting model. |
| `full/model.splat` | Viewer-oriented splat export when produced by the QA/export stage. |
| `full/metrics.json` | Held-out image-quality and model diagnostics. |
| `full/qa_sheet.jpg` | Real-versus-rendered and off-path visual checks. |
| `mesh/mesh.glb` and `mesh/mesh.json` | Geometry mesh and its metadata when the geometry stage completes. |
| `tour/` | Static web-tour files when the publish preparation succeeds. |

A run can complete model training but fail the optional tour-publishing step. Check the GitHub Actions log and the output folder rather than assuming every listed artifact exists.

## 6. Reconstruct your own home

For a competition-representative capture, replace the sample clip with your own continuous video of your home or another accessible indoor space. Follow [`docs/CAPTURE_GUIDE.md`](../docs/CAPTURE_GUIDE.md): landscape ultra-wide capture, locked exposure/focus, slow turns, overlapping wall passes, diagonals and both directions through every doorway.

Upload that video as a private Kaggle dataset and change only the `source` value in the sample configuration to `kaggle:<username>/<your-home-dataset>`. Increase `NFRAMES`, `LONG`, training steps and splat budget only after the short sample run works.

## 7. Important reproducibility notes

- `test/RUN.sample.json` is a template. It has `n: 0` and cannot start a Kaggle run until intentionally enabled.
- `tests/` is the automated test suite. `test/` is the Round 1 sample-data and reconstruction guide.
- The local FastAPI prototype and the v3 Kaggle runner are separate execution paths; use the v3 runner above to reproduce the hosted Round 1 3DGS/2DGS result.
- The short sample configuration is a smoke test, not the high-quality house recipe used for the main demonstration.
- Do not commit `RUN.json` with real tokens, Kaggle credentials, Hugging Face tokens or other secrets.
- The exact output metrics depend on the supplied video, registration quality, GPU runtime and training settings; no metric is promised by this sample package.
