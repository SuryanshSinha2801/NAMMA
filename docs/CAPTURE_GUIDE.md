# How to film a room for NAMMA (v3, 26 Oct; 28 Oct: landscape, fast shutter, walk-don't-pan; the NAMMA Capture app does all of this itself)

SplatTour's published method (their clean tours come from it) plus what broke on our 22 Oct video. A 2-3 room flat is about
3 minutes of video, a big room about 2 minutes, a bathroom about 1 minute; up to 10 minutes per floor.

## Before you press record
| Do this | Why |
|---|---|
| **Phone sideways (landscape), slightly angled, 0.5x ultra-wide lens** (0.6x on some Android) | More of the room in every frame: tracking is far more robust; wide lenses forgive shake |
| **Clean the lens** | A smudge makes every frame soft |
| **Fast shutter, 1/500 s or faster; accept ISO noise** | Splats rebuild exactly what they see: blur spreads through the whole model, noise barely matters. Rolling-shutter wobble also drops |
| **Lock exposure and focus** (iPhone: long-press until "AE/AF LOCK"; Android: Pro mode) | 22 Oct: the same white wall was brighter/darker on different passes and that broke the camera check |
| **HDR off, "super steady"/action stabilisation off, 4K at 30 fps, highest bitrate (HEVC)** | HDR and heavy stabilisation change the image between frames; matching gets worse |
| **Lights on, doors as they should be shown, nobody walking through** | Moving things and people break matching and leave ghosts |
| **Plain walls: tape up a poster or put a chair/plant near them for the recording** | Blank walls have nothing to match; 22 Oct lost track exactly there (frames 216, 488) |

## In every room: three passes, one take
1. **Lap along the walls**, about 1 m from them, phone at chest height pointing straight ahead at eye level.
2. **The same lap the other way round.**
3. **Loops around the big furniture** (bed, table): walk around each one.
4. **Corner to corner**: cross the room along both diagonals.
5. **A little past the edge**: film slightly beyond where the tour should end, so the edges hold up.

Detail comes from seeing every surface from more than one place. Walk through doorways through the middle, looking ahead, without
stopping in the frame; walk hallways on the way.

## While walking
| Do this | Why |
|---|---|
| **Walk, don't pan ("zoom with your feet")**: parallax comes only from moving the phone through space; walk loops around big furniture |
| **Walk slowly, in a line; never orbit around one point** | Slow, steady motion is what lets every frame be placed |
| **To change direction: stop, turn slowly (3 s for a quarter turn), keep walking** | A ~35 deg jump between frames breaks matching; 22 Oct: 4 of 9 gaps came from fast turns |
| **Keep the floor out of the frame**: aim at eye level | Repetitive tiles filling the frame is the most common reason a room fails |
| **Pass look-alike places slowly with something unique in view** (a switch board, a bin, a sticker) | 22 Oct: both corridor ends looked alike and the solver put a window where a door is |
| **One video per floor, never cut**: start at the entrance, stop after the last room | Rooms are linked by the walk between them; a cut loses that link |
| **Every doorway twice: once walking in, once walking out, slowly** | Rooms are trained one by one and joined at the doors; both directions give the overlap that joins them |
| **Same lights the whole time; don't switch lights between rooms** | Brightness jumps between rooms show up as seams and blotchy walls |
| **Show big furniture from two sides** (walk past the sofa, bed, table so both ends are seen) | Each object gets its own solid shape for walking into, clicking and search; two sides make it fit |
| **Mirrors: pass them sideways, don't stand facing one** | The model builds a fake room behind a mirror; a short, side-on view keeps it small |

## Send it
Upload the video to Kaggle as a new **private** dataset (kaggle.com -> Datasets -> New dataset -> drop the video; any name, e.g.
`namma-room-2026-10-26`) and tell Claude the name. Claude starts the run (`colab/real_scenes/v3/RUN.json`, source
`kaggle:anasrazy/<name>`) and sends the tour link after checking it.
