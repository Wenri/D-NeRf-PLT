import torch
import torch.nn.functional as F


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
