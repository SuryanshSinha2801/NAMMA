import torch, torch.nn.functional as F


class DotProductAttention(torch.nn.Module):
    """TE's DotProductAttention (no mask, no dropout) on PyTorch's scaled_dot_product_attention: q/k/v [B, S, H, D] ('bshd',
    or more middle dims) or [S, B, H, D] ('sbhd') -> [..., H*D] like TE."""
    def __init__(self, num_attention_heads=None, kv_channels=None, qkv_format='bshd', **kw):
        super().__init__(); self.fmt = qkv_format

    def forward(self, q, k, v, *a, **kw):
        if self.fmt == 'sbhd': q, k, v = (t.transpose(0, 1) for t in (q, k, v))
        shp = q.shape; f = lambda t: t.reshape(t.shape[0], -1, t.shape[-2], t.shape[-1]).transpose(1, 2)      # [B, H, S, D]
        o = F.scaled_dot_product_attention(f(q), f(k), f(v)).transpose(1, 2)                                    # [B, S, H, D]
        o = o.reshape(*shp[:-2], shp[-2] * shp[-1])
        return o.transpose(0, 1) if self.fmt == 'sbhd' else o


def apply_rotary_pos_emb(t, freqs, tensor_format='sbhd', fused=False, **kw):
    """TE semantics (non-interleaved): freqs [S,1,1,D] angles; rotate first D dims of t as two halves."""
    if tensor_format == 'bshd': freqs = freqs.transpose(0, 1)                               # [S,1,1,D] -> [1,S,1,D]
    d = freqs.shape[-1]; tr, rest = t[..., :d], t[..., d:]
    cos, sin = freqs.cos().to(t.dtype), freqs.sin().to(t.dtype)
    x1, x2 = tr.chunk(2, -1); rot = torch.cat((-x2, x1), -1)
    return torch.cat((tr * cos + rot * sin, rest), -1)
