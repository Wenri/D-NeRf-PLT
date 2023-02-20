from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from scipy.spatial import ConvexHull

from palette.Additive_mixing_layers_extraction import Hull_Simplification_determined_version


def sort_palette(rgbs, palette_rgb, bg=None, K=1e8):
    N = rgbs.shape[0]
    if N > K:
        rgbs = rgbs[torch.randperm(N)[:K]]
    dist = np.linalg.norm(rearrange(rgbs, 'N C -> N 1 C') - rearrange(palette_rgb, 'P C -> 1 P C'), axis=-1)
    dist = np.bincount(np.argmin(dist, axis=-1))
    if bg is not None:
        idx = np.argmin(np.linalg.norm(palette_rgb - bg, axis=-1))
        dist[idx] = np.iinfo(dist.dtype).max
    dist = np.argsort(dist)

    # bg = np.ones(3) if dataset.white_bg else np.zeros(3)
    # palette_rgb = [tuple(a.tolist()) for a in palette_rgb[dist.cpu().numpy()] if not np.allclose(a, bg)]
    # palette_rgb.append(tuple(bg.tolist()))

    palette_rgb = palette_rgb.tolist()
    return [(*palette_rgb[i.item()],) for i in dist]


def build_palette(filepath, rgbs, white_bg=True, simplify=True):
    filepath = Path(filepath)
    rgbs = torch.as_tensor(rgbs)
    if white_bg:
        fg = torch.lt(rgbs, 1.).any(dim=-1)
        rgbs = rgbs[fg]
    bg = np.zeros((3,), dtype=np.float32)
    rgbs = rgbs.to(device='cpu', dtype=torch.double).numpy()
    hull = ConvexHull(rgbs)
    hull_vertices = hull.points[hull.vertices]
    palette = Hull_Simplification_determined_version(
        rgbs, filepath.stem + "-convexhull_vertices") if simplify else hull_vertices
    return sort_palette(rgbs, palette, bg=bg), hull.points[hull.vertices]


def pick_palette(palette, root=None):
    from tkcolorpicker import askcolor
    if root is None:
        import tkinter as tk
        import tkinter.ttk as ttk
        root = tk.Tk()
        style = ttk.Style(root)
        style.theme_use('clam')
    palette = (np.asarray(palette) * 255).astype(np.uint8)
    palette = [askcolor(p.tolist(), root)[0] for p in palette]
    palette = np.asarray(palette) / 255.
    return [tuple(p) for p in palette.tolist()]


class PLTRender(torch.nn.Module):
    def __init__(self, palette=None, hullVertices=None):
        super().__init__()
        self.n_opaque = len(palette) - 1
        self.n_dim = 3 + self.n_opaque
        palette = torch.as_tensor(palette, dtype=torch.float32)
        # torch.nn.init.uniform_(palette)
        self.register_buffer('palette', palette.T, persistent=False)

    def rgb_from_palette_rev(self, logits):
        opaque = torch.sigmoid(logits)
        log_opq = F.logsigmoid(logits)
        log_wa = torch.cumsum(F.logsigmoid(torch.neg(logits)), dim=-1)
        w_0 = opaque[..., :1]
        w_a = torch.exp(log_wa[..., :-1] + log_opq[..., 1:])
        w_last = torch.exp(log_wa[..., -1:])
        bary_coord = torch.cat((w_0, w_a, w_last), dim=-1)
        # bary_coord guarantee sum .eq. 1
        # assert torch.allclose(bary_coord.sum(dim=-1), torch.ones(()), atol=1e-3)
        return F.linear(bary_coord, self.palette), opaque

    def forward(self, raw):
        logits = raw[..., :self.n_opaque]
        rgb, opaque = self.rgb_from_palette_rev(logits)
        return torch.cat((rgb, raw[..., self.n_opaque:], opaque), dim=-1)
