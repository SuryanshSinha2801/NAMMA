"""Room priors for gsplat's simple_trainer, against the two smears the owner saw in tour v2 (streaks on the floor/ceiling,
long needle splats off the filmed path). Cheap, no extra model: v3/run.py patches two lines into simple_trainer.py and turns it on
with ROOM_PRIOR=1 (variant kind 'prior'), so the same run can train with and without it and v3_qa measures both.
  needle   splats whose longest axis is > R x the middle one are pushed back (discs stay free: walls, floors are thin discs).
           = the effective-rank idea (Hyung et al., NeurIPS 2024) in the form our QA measures (needle = longest/middle > 10).
  planes   splats within a thin band of the floor (and the ceiling, if SfM saw it) are turned flat to it: shortest axis along
           "up". A Manhattan / planar prior (UncertainGS 2026, PlanarGS 2025) without their segmentation and depth networks:
           the floor height comes from the SfM points, "up" from the cameras (phones are held with ~no roll).
Weights are guesses until measured (ROOM_PRIOR_NEEDLE, ROOM_PRIOR_PLANE); the loss starts after ROOM_PRIOR_START steps."""
import os, numpy as np, torch


def camera_up(c2w):
    """Most-perpendicular direction to all camera x axes, signed like image-up (same as v3_qa.camera_up)."""
    x = c2w[:, :3, 0]; u = np.linalg.eigh(x.T @ x)[1][:, 0]
    return u if u @ (-c2w[:, :3, 1].mean(0)) > 0 else -u


def planes(c2w, pts):
    """Floor height, ceiling height or None, band half-width and up, in the trainer's (normalised) world."""
    up = camera_up(c2w); h = pts @ up; cam = float(np.median(c2w[:, :3, 3] @ up))
    floor = float(np.percentile(h, 2)); band = 0.04 * (cam - floor)                  # ~5 cm with the phone at ~1.35 m
    top = float(np.percentile(h, 99.5)); ceil = None
    if top > cam + 0.4 * (cam - floor) and np.mean(np.abs(h - top) < band) > 0.002:   # enough SfM points up there to be a ceiling
        ceil = top
    return floor, ceil, band, up


def quat_axes(q):
    """[N,4] wxyz -> [N,3,3] rotation matrices (columns = the splat's axes)."""
    w, x, y, z = torch.nn.functional.normalize(q, dim=-1).unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


class RoomPrior:
    def __init__(self, parser, device='cuda'):
        e = os.environ.get
        self.start = int(e('ROOM_PRIOR_START', 1000)); self.wn = float(e('ROOM_PRIOR_NEEDLE', 0.01)); self.wp = float(e('ROOM_PRIOR_PLANE', 0.05))
        self.r = float(e('ROOM_PRIOR_RATIO', 6.0))
        f, c, self.band, up = planes(parser.camtoworlds, parser.points)
        self.h = torch.tensor([f] + ([c] if c is not None else []), device=device, dtype=torch.float32)
        self.up = torch.tensor(up, device=device, dtype=torch.float32)
        print(f'ROOM PRIOR floor {f:.3f} ceiling {c} band {self.band:.4f} needle w {self.wn} ratio {self.r} plane w {self.wp} from step {self.start}', flush=True)

    def __call__(self, splats, step):
        if step < self.start: return 0.0
        s, i = torch.sort(splats['scales'], dim=1)                                       # log scales, ascending
        loss = self.wn * torch.relu(s[:, 2] - s[:, 1] - np.log(self.r)).mean()           # log(longest / middle) above log R
        d = (splats['means'] @ self.up)[:, None] - self.h[None]                           # height above each plane
        near = (d.abs() < self.band).any(1)
        if near.any():
            ax = quat_axes(splats['quats'][near]); k = i[near, 0]                         # shortest axis of each splat near a plane
            n = ax[torch.arange(len(k), device=k.device), :, k]
            loss = loss + self.wp * (1 - (n @ self.up).abs()).mean()
        return loss


def patch(trainer_py):
    """Insert the prior into gsplat's examples/simple_trainer.py (idempotent; does nothing unless ROOM_PRIOR=1 at run time)."""
    s = open(trainer_py).read()
    if 'room_prior' in s: return True
    a = '        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale\n'
    b = '            loss.backward()\n'
    if s.count(a) != 1 or s.count(b) != 1: return False
    s = s.replace(a, a + "        self.room_prior = None\n        if os.environ.get('ROOM_PRIOR'):\n"
                         "            import room_prior; self.room_prior = room_prior.RoomPrior(self.parser, self.device)\n")
    s = s.replace(b, '            if self.room_prior is not None: loss = loss + self.room_prior(self.splats, step)\n' + b)
    if 'import os' not in s: s = 'import os\n' + s
    open(trainer_py, 'w').write(s); return True
