import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class RealNVPBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.scale_net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2),
            nn.Tanh()  # 限制scale输出范围
        )
        self.translate_net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2)
        )

    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=1)
        if not reverse:
            s = self.scale_net(x1)
            t = self.translate_net(x1)
            y2 = x2 * torch.exp(s) + t
            y = torch.cat([x1, y2], dim=1)
            log_det_jacobian = s.sum(dim=1)
            return y, log_det_jacobian
        else:
            s = self.scale_net(x1)
            t = self.translate_net(x1)
            y2 = (x2 - t) * torch.exp(-s)
            y = torch.cat([x1, y2], dim=1)
            log_det_jacobian = -s.sum(dim=1)
            return y, log_det_jacobian

class RealNVP(nn.Module):
    def __init__(self, dim, hidden_dim, num_coupling_layers=6):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_coupling_layers = num_coupling_layers
        self.layers = nn.ModuleList([RealNVPBlock(dim, hidden_dim) for _ in range(num_coupling_layers)])

    def forward(self, x, reverse=False):
        log_det_jacobian_total = 0
        if not reverse:
            for i, layer in enumerate(self.layers):
                x, log_det_jacobian = layer(x, reverse=False)
                log_det_jacobian_total += log_det_jacobian
                x = x.flip(dims=[1])  # 交替翻转维度，增加表达能力
        else:
            for i, layer in reversed(list(enumerate(self.layers))):
                x = x.flip(dims=[1])
                x, log_det_jacobian = layer(x, reverse=True)
                log_det_jacobian_total += log_det_jacobian
        return x, log_det_jacobian_total