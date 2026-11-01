# NAMMA — reproducible indoor video-to-3D reconstruction

## Techfest Round 1: production pipeline reproduction

**Start here: [test/README.md — reproduce the production pipeline](test/README.md)**

NAMMA's actual production route is terminal → GitHub Actions → Kaggle API/GPU script kernel → Hugging Face Hub. The local machine prepares credentials and pushes a run configuration; all reconstruction training happens on Kaggle. GitHub Actions collects the result and publishes it to a private Hugging Face dataset.

The sample workflow uses the same production scripts as the main workflow:
- GitHub orchestration: [colab/real_scenes/v3/gh_run.py](colab/real_scenes/v3/gh_run.py)
- Kaggle helpers: [colab/real_scenes/v3/kaggle_io.py](colab/real_scenes/v3/kaggle_io.py)
- GPU reconstruction: [colab/real_scenes/v3/run.py](colab/real_scenes/v3/run.py)
- Production workflow: [.github/workflows/namma-v3.yml](.github/workflows/namma-v3.yml)
- Sample workflow: [.github/workflows/namma-sample.yml](.github/workflows/namma-sample.yml)

The sample configuration is separate from the production RUN.json. The capture procedure is documented in [docs/CAPTURE_GUIDE.md](docs/CAPTURE_GUIDE.md).

## Local prototype

A separate laptop-oriented FastAPI prototype is also present in this repository. It is not the production Kaggle-backed v3 reconstruction path. Follow the terminal-to-GitHub-to-Kaggle-to-Hugging-Face reproduction guide above for Round 1.
