# Video in, tour link out (28 Oct 2026)

Any room video becomes a 3D tour with no code edits. Use the NAMMA Capture app, or put a video in the inbox by hand.

## The path
1. **Inbox**: private HF dataset `<hf user>/namma-inbox`, one folder per job (`YYYYMMDD-HHMMSS-xxxx`, UTC):
   - `video.json` + `parts/000…`: the app's upload, cut into 32 MB parts so it can resume. `video.json` gives the name, size,
     sha256 and part list. A plain `video.mp4` works too.
   - `meta.json`: name, phone, camera settings, passes, cue counts, `preset`.
   - `status.json`: `uploaded → training → publishing → done | failed`, plus `run` (results folder), `link` (tour) and `reason`.
2. **Trigger**: a push to `colab/real_scenes/v3/inbox/<job>.json` on branch `claude/wonderful-cori-i3fn41` starts
   `.github/workflows/inbox.yml`. The trigger file holds only the job id. GitHub starts workflows on non-default branches only from
   pushes, so the app pushes this file with the owner's GitHub token.
3. **`v3/inbox.py pick`**: runs one Kaggle job at a time.
   - A job in `training` whose Kaggle job has ended: collect it.
   - Any of our Kaggle jobs still running: wait.
   - Otherwise: start the oldest `uploaded` job.
4. **`v3/gh_run.py`**, source `hf:<inbox>:<job>/video.json`:
   - joins the parts and checks the sha256;
   - makes a Kaggle dataset `namma-in-<job>` and waits until it is ready;
   - Kaggle kernel `namma-job`: `v3/run.py`, with the settings of the job's `PIPELINE.json` preset;
   - results go to `namma-v3-out/<run>`; the tour (SH1) is built and published;
   - `status.json` gets `done` and the link.
5. **Runs longer than 6 h** (one GitHub job): gh_run exits cleanly while Kaggle still trains, and the workflow's `collect1` /
   `collect2` jobs pick it up. Together they cover 18 h, more than Kaggle's 12 h limit.
6. **The app** reads `status.json` and shows the link. When a job waits and no inbox run is going, the app pushes the trigger again,
   at most every 20 min. This covers a GPU that was busy, a lost run, and waits while another run (e.g. run 6) holds Kaggle.

## Presets (`v3/PIPELINE.json`)
- `best` (default): run 4's recipe with more detail.
  - 900 sharpest frames, 1920 px, 40k steps, up to 3M splats;
  - gsplat MCMC + bilateral grid + depth loss;
  - mesh from the 3DGS model; tour with SH1.
  - It is updated to run 6's winner once run 6 is checked.
- `quick`: 150 frames, 960 px, 3k steps. Only for testing the path.

## By hand
1. Upload `<job>/video.mp4` and a `status.json` of `{"stage": "uploaded"}` to the inbox (`meta.json` is optional).
2. Push `colab/real_scenes/v3/inbox/<job>.json`.

## Privacy
- Links appear only in private files (`status.json` and the results dataset's `tour_url.txt`), never in the public GitHub logs.
- The workflow artefacts hold only json and log files.
- Tokens:
  - GitHub secrets for the runners;
  - typed into the app once, encrypted with the phone's keystore, never in git.
