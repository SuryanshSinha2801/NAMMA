Trigger files for .github/workflows/inbox.yml: the NAMMA Capture app pushes `<job id>.json` here (job id only, no links, no tokens)
after it has uploaded the video to the private HF dataset `<hf user>/namma-inbox`. Any push here makes the workflow look at the
inbox (v3/inbox.py pick). See docs/VIDEO_TO_TOUR.md.
