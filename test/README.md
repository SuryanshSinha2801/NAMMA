# NAMMA v3: reproduce the production reconstruction pipeline

This is the actual production route used by NAMMA. It does not depend on a mobile application and it does not ask you to run the reconstruction on your laptop.

A terminal command updates a run configuration in GitHub. GitHub Actions uses repository secrets to stage the video through the Kaggle API, push a GPU script kernel, collect the model outputs and upload them to a private Hugging Face dataset.

## 1. End-to-end pipeline

~~~text
Terminal (Git + GitHub CLI)
  | push test/RUN.sample.json
  v
GitHub Actions
  | reads GitHub Secrets: KAGGLE_API_TOKEN + HF_TOKEN
  | runs colab/real_scenes/v3/gh_run.py
  | uploads repository sample as a private Kaggle Dataset
  | creates Kaggle kernel metadata and pushes script
  v
Kaggle NVIDIA T4 kernel
  | runs colab/real_scenes/v3/run.py
  | samples frames -> solves cameras (hloc + COLMAP)
  | trains 3DGS appearance + 2DGS geometry
  | exports QA, metrics and mesh
  v
GitHub runner collects Kaggle output
  | uploads results with Hugging Face API
  v
Private Hugging Face dataset: HF_USER/namma-v3-out/RUN_ID
~~~

The sample workflow calls the same GitHub runner and Kaggle training script as production. Its only difference is that it passes test/RUN.sample.json through the RUN_JSON environment variable, so it never overwrites the production colab/real_scenes/v3/RUN.json.

### Source files

| File | Function |
|---|---|
| .github/workflows/namma-v3.yml | Production GitHub Actions workflow |
| .github/workflows/namma-sample.yml | Isolated sample workflow |
| colab/real_scenes/v3/gh_run.py | GitHub-side input staging, Kaggle orchestration and HF output upload |
| colab/real_scenes/v3/kaggle_io.py | Kaggle account, dataset and kernel helpers |
| colab/real_scenes/v3/run.py | Actual GPU reconstruction script |
| colab/real_scenes/v3/kernel-metadata.json | Kaggle script kernel settings |
| colab/real_scenes/final/frames.py | Video/frame sampling |
| test/RUN.sample.json | Separate, non-production sample configuration |

## 2. Requirements

- Git and a terminal.
- GitHub CLI (gh), signed in to an account that can push to RazyAnas/NAMMA and manage Actions secrets.
- Kaggle account with API access and available GPU quota.
- Hugging Face account with a write access token.
- GitHub Actions enabled for the repository.

Your local laptop does not need an NVIDIA GPU. The training kernel requests the Kaggle NVIDIA T4 machine shape configured in the repository.

## 3. Clone the configured branch

The current workflows listen to the branch below. Use it for this version of the repository.

~~~bash
git clone --branch claude/wonderful-cori-i3fn41 https://github.com/RazyAnas/NAMMA.git
cd NAMMA
~~~

Install GitHub CLI if needed:

~~~bash
# macOS
brew install gh

# Windows
winget install --id GitHub.cli
~~~

Authenticate and verify access:

~~~bash
gh auth login
gh auth status
gh repo view RazyAnas/NAMMA
~~~

## 4. Configure credentials once, from the terminal

The workflow reads repository secrets. It never reads credentials committed in JSON or source files.

### 4.1 Kaggle API token

Create an API token at https://www.kaggle.com/settings/api. Save the token file at the path supported by the Kaggle CLI, usually ~/.kaggle/access_token.

Add it to GitHub without printing its contents:

~~~bash
gh secret set KAGGLE_API_TOKEN --repo RazyAnas/NAMMA < ~/.kaggle/access_token
~~~

The current Kaggle CLI supports the KAGGLE_API_TOKEN environment variable and the ~/.kaggle/access_token file. The production runner reads KAGGLE_API_TOKEN from GitHub Actions.

### 4.2 Hugging Face write token

Create a User Access Token with the write role at https://huggingface.co/settings/tokens.

Set it interactively:

~~~bash
gh secret set HF_TOKEN --repo RazyAnas/NAMMA
~~~

Paste the token when prompted. Write access is needed to create/update the private namma-v3-out dataset and upload the output models and metrics.

### 4.3 Verify secret names

~~~bash
gh secret list --repo RazyAnas/NAMMA
~~~

The workflow accepts these compatible names:

| Primary name | Alias |
|---|---|
| KAGGLE_API_TOKEN | KAGGLE_TOKEN |
| HF_TOKEN | HUGGINGFACE_WRITE_TOKEN |

At least one secret from each row must exist. The command lists names only, not secret values.

## 5. Run the included sample

Input video: test/sample_data/lobby_clip30.mp4

Sample config: test/RUN.sample.json

The sample source is:

~~~json
"source": "github:test/sample_data/lobby_clip30.mp4"
~~~

The GitHub runner handles the upload automatically:

1. It resolves the path inside the checked-out repository.
2. It copies the video into a temporary input folder.
3. It creates or versions a private Kaggle Dataset using KAGGLE_API_TOKEN.
4. It waits until Kaggle reports that dataset is ready.
5. It attaches that dataset to the isolated namma-r1-sample-v2 script kernel. The first slug returned a Kaggle 409 Conflict, so the reproducibility config uses a fresh kernel slug.
6. It pushes the kernel and runs the same run.py used by production.
7. It collects the kernel output and uploads it to Hugging Face using HF_TOKEN.

You do not need to create or upload the sample Kaggle dataset manually.

The first sample run is triggered automatically when the sample configuration and sample workflow are committed. To run it again, increase the n field and push the config:

~~~bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("test/RUN.sample.json")
cfg = json.loads(path.read_text())
cfg["n"] = int(cfg.get("n", 0)) + 1
path.write_text(json.dumps(cfg, indent=2) + "\n")
print("Sample run number:", cfg["n"])
PY

git add test/RUN.sample.json
git commit -m "Run NAMMA v3 sample reconstruction"
git push origin claude/wonderful-cori-i3fn41
~~~

This workflow passes RUN_JSON=test/RUN.sample.json to gh_run.py. The production RUN.json remains unchanged.

## 6. Monitor the run from your terminal

List recent sample runs:

~~~bash
gh run list --repo RazyAnas/NAMMA --workflow "NAMMA v3 sample reconstruction" --limit 5
~~~

Watch a run, replacing RUN_ID with the ID printed above:

~~~bash
gh run watch RUN_ID --repo RazyAnas/NAMMA
~~~

Read the full log after completion:

~~~bash
gh run view RUN_ID --repo RazyAnas/NAMMA --log
~~~

The log reports dataset staging, Kaggle kernel push/status, number of extracted and registered frames, GPU detection, training variants, output collection and Hugging Face upload.

## 7. Output location

Results are uploaded to the private Hugging Face dataset:

~~~text
https://huggingface.co/datasets/HF_USER/namma-v3-out/tree/main/RUN_ID
~~~

The timestamped run folder can contain:

| Output | Meaning |
|---|---|
| github_run.json | Input source and GitHub run metadata |
| source.json | Resolved input video |
| sfm_report.json | Extracted/registered frames, cameras and sparse points |
| full/model.ply | Full-colour Gaussian model |
| full/model.splat | Viewer-oriented splat when produced |
| full/metrics.json | Model evaluation metrics |
| full/qa_sheet.jpg | Visual QA sheet when produced |
| mesh/mesh.glb | Geometry mesh when 2DGS geometry completes |
| mesh/mesh.json | Geometry metadata |
| tour/ | Web viewer files when tour build succeeds |

Training may succeed even if the optional web-tour step fails. Always inspect the actual output files and logs.

## 8. Use a different indoor video

The production runner supports these input source formats:

**A. A video or frame dataset already on Kaggle**

~~~json
"source": "kaggle:KAGGLE_USERNAME/DATASET_SLUG"
~~~

The runner attaches the dataset to the new script kernel. It can contain a video, a frame directory or a ZIP of frames.

**B. A video stored in a Hugging Face dataset**

~~~json
"source": "hf:HF_USERNAME/DATASET_NAME:PATH_TO_VIDEO"
~~~

The GitHub runner downloads the video with HF_TOKEN and stages it as a private Kaggle dataset with the Kaggle API. This works for large private phone videos, subject to GitHub runner disk limits.

For a home capture, follow docs/CAPTURE_GUIDE.md. Keep the video private unless you have permission to redistribute it. Change the source in the run configuration and push it to start a new run.

## 9. Configuration reference

| Field | Purpose |
|---|---|
| n | Run counter; 0 means no run, positive means execute |
| collect | true collects a job already running on Kaggle instead of pushing a new version |
| source | github:, kaggle:, hf:, dl3dv: or reuse: input |
| kernel | Kaggle script-kernel slug |
| kaggle_dataset | Dataset slug used by the GitHub sample input adapter |
| env | Runtime settings injected into run.py (NFRAMES, LONG, STEPS, VARIANTS, etc.) |

The sample requests 120 frames, a 960-pixel long side and two short 3,000-step training variants: 3DGS appearance and 2DGS geometry. This is a smoke test, not a high-quality reconstruction benchmark.

## 10. Reproducibility and safety

- Never commit Kaggle or Hugging Face tokens.
- The sample uses a separate Kaggle dataset and kernel; it does not overwrite the production kernel.
- The sample workflow reads its own config using RUN_JSON; production RUN.json is not modified.
- The run depends on Kaggle GPU eligibility/quota, valid tokens, input-video quality and successful camera registration.
- The sample lobby video is already tracked elsewhere in this repository and is exposed under test/sample_data. No separate open licence statement is attached to the recording; confirm redistribution rights before publishing it outside this repository.
