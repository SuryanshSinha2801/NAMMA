import torch
from . import attention, distributed, module


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6, **kw):
        super().__init__(); self.eps = eps; self.weight = torch.nn.Parameter(torch.ones(dim))

    def reset_parameters(self): torch.nn.init.ones_(self.weight)

    def forward(self, x):
        xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight
