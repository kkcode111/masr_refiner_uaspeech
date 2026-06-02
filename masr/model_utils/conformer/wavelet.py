import math
from typing import Optional

import torch
from torch import nn


class TemporalHaarWaveletResidual(nn.Module):
    """Conservative Haar wavelet residual branch over encoder time states."""

    def __init__(
            self,
            size: int,
            alpha_init: float = -4.0,
            gate_init: float = -2.0,
            alpha_max: float = 1.0,
            dropout_rate: float = 0.1):
        super().__init__()
        self.size = size
        self.alpha_max = float(alpha_max)

        self.pre_norm = nn.LayerNorm(size)
        self.low_adapter = nn.Sequential(
            nn.Linear(size, size * 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(size * 2, size),
        )
        self.high_adapter = nn.Sequential(
            nn.Linear(size, size * 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(size * 2, size),
        )
        self.out_norm = nn.LayerNorm(size)
        self.out_proj = nn.Linear(size, size)
        self.dropout = nn.Dropout(dropout_rate)

        self.gate_norm = nn.LayerNorm(size)
        self.gate_proj = nn.Linear(size, size)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, gate_init)

        self.alpha_param = nn.Parameter(torch.tensor(float(alpha_init)))

        self._last_gate_mean: Optional[torch.Tensor] = None
        self._last_residual_l2: Optional[torch.Tensor] = None
        self._last_delta_mean: Optional[torch.Tensor] = None
        self._last_delta_std: Optional[torch.Tensor] = None
        self._last_low_energy: Optional[torch.Tensor] = None
        self._last_high_energy: Optional[torch.Tensor] = None

    def _apply_mask(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return x
        if mask.dim() == 3:
            mask = mask.transpose(1, 2)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")
        return x * mask.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        B, T, D = x.shape
        if D != self.size:
            raise ValueError(f"Expected hidden size {self.size}, got {D}")

        x_norm = self._apply_mask(self.pre_norm(x), mask)
        if T % 2 == 1:
            pad = x_norm.new_zeros(B, 1, D)
            x_pair = torch.cat([x_norm, pad], dim=1)
        else:
            x_pair = x_norm

        even = x_pair[:, 0::2, :]
        odd = x_pair[:, 1::2, :]
        scale = 1.0 / math.sqrt(2.0)
        low = (even + odd) * scale
        high = (even - odd) * scale

        low_delta = self.low_adapter(low)
        high_delta = self.high_adapter(high)
        even_delta = (low_delta + high_delta) * scale
        odd_delta = (low_delta - high_delta) * scale

        delta_pair = x_pair.new_zeros(B, x_pair.size(1), D)
        delta_pair[:, 0::2, :] = even_delta
        delta_pair[:, 1::2, :] = odd_delta
        wavelet_delta = delta_pair[:, :T, :]
        wavelet_delta = self.out_proj(self.out_norm(wavelet_delta))
        wavelet_delta = self._apply_mask(wavelet_delta, mask)

        gate = torch.sigmoid(self.gate_proj(self.gate_norm(residual)))
        gate = self._apply_mask(gate, mask)
        alpha = torch.sigmoid(self.alpha_param) * self.alpha_max
        gated_delta = alpha * gate * self.dropout(wavelet_delta)
        gated_delta = self._apply_mask(gated_delta, mask)

        valid_delta = self._apply_mask(wavelet_delta.detach(), mask)
        self._last_gate_mean = gate.detach().mean()
        self._last_residual_l2 = gated_delta.detach().pow(2).mean().sqrt()
        self._last_delta_mean = valid_delta.mean()
        self._last_delta_std = valid_delta.std(unbiased=False)
        self._last_low_energy = low.detach().pow(2).mean()
        self._last_high_energy = high.detach().pow(2).mean()

        return residual + gated_delta

    def get_alpha_value(self):
        return (torch.sigmoid(self.alpha_param.detach()) * self.alpha_max).item()

    def get_gate_mean(self):
        return None if self._last_gate_mean is None else self._last_gate_mean.item()

    def get_residual_l2(self):
        return None if self._last_residual_l2 is None else self._last_residual_l2.item()

    def get_wavelet_delta_mean(self):
        return None if self._last_delta_mean is None else self._last_delta_mean.item()

    def get_wavelet_delta_std(self):
        return None if self._last_delta_std is None else self._last_delta_std.item()

    def get_low_energy(self):
        return None if self._last_low_energy is None else self._last_low_energy.item()

    def get_high_energy(self):
        return None if self._last_high_energy is None else self._last_high_energy.item()


WaveletResidualEncoder = TemporalHaarWaveletResidual
