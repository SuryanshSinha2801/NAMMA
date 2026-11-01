# NAMMA — phone video to walkable indoor 3D scenes

## Techfest Round 1: reproducible reconstruction package

The competition-facing v3 pipeline converts a handheld indoor video into a 3D Gaussian Splatting model, a geometry/collision mesh and a hosted web tour. It runs the training stage on a Kaggle NVIDIA T4 through GitHub Actions; the local FastAPI application documented later in this README is a separate prototype/API path.

**Start here:** [Round 1 setup and reconstruction guide](test/README.md)

The guide includes:
- a sample video under [test/sample_data/](test/sample_data/);
- a non-production run configuration at [test/RUN.sample.json](test/RUN.sample.json);
- local frame-extraction and unit-test commands;
- the exact GitHub Actions + Kaggle steps for a short sample reconstruction;
- output locations and limitations of the smoke-test run.

For the project's capture procedure, see [docs/CAPTURE_GUIDE.md](docs/CAPTURE_GUIDE.md). The v3 training runner is [colab/real_scenes/v3/run.py](colab/real_scenes/v3/run.py), orchestrated by [gh_run.py](colab/real_scenes/v3/gh_run.py) and [.github/workflows/namma-v3.yml](.github/workflows/namma-v3.yml).

## Local FastAPI prototype

The repository also contains a laptop-oriented FastAPI prototype with project/job records, video upload, frame extraction, tracking, quality checks and stage APIs. The sections below describe that local prototype; they are not a substitute for the Kaggle-backed v3 reconstruction instructions above.

