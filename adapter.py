# adapter.py

import torch
import torch.nn as nn


class PEClipAdapter(torch.nn.Module):
    """
    Residual MLP adapter: y = x + alpha * MLP(LN(x))
    """
    def __init__(self, dim: int, hidden_ratio: int = 4, alpha: float = 0.5, dropout: float = 0.0):
        super().__init__()
        hidden = max(1, dim // max(1, hidden_ratio))
        self.ln = torch.nn.LayerNorm(dim)
        self.fc1 = torch.nn.Linear(dim, hidden, bias=False)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(hidden, dim, bias=False)
        self.alpha = alpha
        # init
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.ln(x)
        z = self.fc2(self.drop(self.act(self.fc1(z))))
        return x + self.alpha * z



class ResidualAdapter(torch.nn.Module):
    def __init__(self, dim: int, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, dim, bias=False),
            torch.nn.GELU(),
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, dim, bias=False),
        )
    def forward(self, x):
        # x: (..., C)
        return x + self.alpha * self.net(x)


class WeightAdapter(torch.nn.Module):
    def __init__(self, c_in: int, reduction: int = 4, scalar: float = 5.0):
        super().__init__()
        hidden = max(1, c_in // reduction)
        self.scalar = float(scalar)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(c_in, hidden, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, c_in, bias=False),
            torch.nn.ReLU(inplace=True),
        )
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.scalar * x
        g = torch.sigmoid(self.fc(z))
        y = g * z
        return y
    


def load_pe_adapter(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "state_dict" in ckpt:
        state = ckpt["state_dict"]
        dim = ckpt.get("dim", None)
        alpha = ckpt.get("alpha", 0.5)
        hidden_ratio = ckpt.get("hidden_ratio", 4)
        dropout = ckpt.get("dropout", 0.0)
        
    else:
        state = ckpt
        dim = None
        alpha = 0.2
        hidden_ratio = 4
        dropout = 0.0
    adapter = PEClipAdapter(
        dim=dim,
        hidden_ratio=hidden_ratio,
        alpha=alpha,
        dropout=dropout
    ).to(device)
    adapter.load_state_dict(state, strict=True)
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)
    return adapter
