"""Pure-PyTorch stand-in for the fused-ssim CUDA package, used only if `pip install fused-ssim` fails to compile.
Same call as gsplat's simple_trainer: fused_ssim(img1, img2, padding="valid") -> mean SSIM (11x11 Gaussian window, sigma 1.5)."""
import torch
import torch.nn.functional as F

def _window(c, device, dtype):
    g = torch.exp(-((torch.arange(11, device=device, dtype=dtype) - 5) ** 2) / (2 * 1.5 ** 2)); g = g / g.sum()
    return (g[:, None] @ g[None, :]).expand(c, 1, 11, 11).contiguous()

def fused_ssim(img1, img2, padding="same", train=True):
    c = img1.shape[1]; w = _window(c, img1.device, img1.dtype); p = 5 if padding == "same" else 0
    mu1, mu2 = F.conv2d(img1, w, padding=p, groups=c), F.conv2d(img2, w, padding=p, groups=c)
    s11 = F.conv2d(img1 * img1, w, padding=p, groups=c) - mu1 ** 2; s22 = F.conv2d(img2 * img2, w, padding=p, groups=c) - mu2 ** 2
    s12 = F.conv2d(img1 * img2, w, padding=p, groups=c) - mu1 * mu2; C1, C2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s11 + s22 + C2))).mean()
