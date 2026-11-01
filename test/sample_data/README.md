# Sample data

## lobby_clip30.mp4

- Input file: `lobby_clip30.mp4`
- Size: 33,960,786 bytes (about 34 MB).
- Purpose: short smoke-test input for the v3 video-to-reconstruction pipeline.
- Expected processing: sharp-frame extraction → camera recovery → short 3DGS and 2DGS training → mesh and QA outputs.
- Not a Round 1 home-capture substitute. Use your own home video, recorded according to `docs/CAPTURE_GUIDE.md`, for the competition's own-space demonstration.

The same video is already tracked at `colab/real_scenes/videos/lobby_clip30.mp4`; this path exposes the existing Git blob inside the dedicated `test/sample_data/` folder.

No standalone licence or release metadata is attached to this MP4. Confirm the capture's rights before redistributing it beyond this repository.
