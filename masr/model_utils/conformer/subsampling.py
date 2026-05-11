from typing import Tuple

import torch
from torch import nn

__all__ = [
    "LinearNoSubsampling", "Conv2dSubsampling4", "Conv2dSubsampling6",
    "Conv2dSubsampling8", "ParallelSpectralBranching4", "GlobalParallelSpectralBranching"
]


class BaseSubsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: int, size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class LinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling."""

    def __init__(self,
                 idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: nn.Module):
        """Construct an linear object.
        Args:
            idim (int): Input dimension.
            odim (int): Output dimension.
            dropout_rate (float): Dropout rate.
            pos_enc_class (PositionalEncoding): position encoding class
        """
        super().__init__()
        self.out = nn.Sequential(nn.Linear(idim, odim),
                                 nn.LayerNorm(odim, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.ReLU(), )
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, offset: int = 0
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
            offset (int): position encoding offset.
        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: positional encoding
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .
        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length)."""

    def __init__(self,
                 idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: nn.Module):
        """Construct an Conv2dSubsampling4 object.

        Args:
            idim (int): Input dimension.
            odim (int): Output dimension.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, odim, 3, 2),
                                  nn.ReLU(),
                                  nn.Conv2d(odim, odim, 3, 2),
                                  nn.ReLU(), )
        self.out = nn.Sequential(nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, offset: int = 0
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
            offset (int): position encoding offset.
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: positional encoding
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = self.out(x.transpose(1, 2).reshape([b, -1, c * f]))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]


class Conv2dSubsampling6(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/6 length)."""

    def __init__(self,
                 idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: nn.Module):
        """Construct an Conv2dSubsampling6 object.

        Args:
            idim (int): Input dimension.
            odim (int): Output dimension.
            dropout_rate (float): Dropout rate.
            pos_enc (PositionalEncoding): Custom position encoding layer.
        """
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, odim, 3, 2),
                                  nn.ReLU(),
                                  nn.Conv2d(odim, odim, 5, 3),
                                  nn.ReLU(), )
        self.linear = nn.Linear(odim * (((idim - 1) // 2 - 2) // 3), odim)
        self.pos_enc = pos_enc_class
        # 10 = (3 - 1) * 1 + (5 - 1) * 2
        self.subsampling_rate = 6
        self.right_context = 10

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, offset: int = 0
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
            offset (int): position encoding offset.
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            torch.Tensor: positional encoding
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = self.linear(x.transpose(1, 2).reshape([b, -1, c * f]))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-4:3]


class Conv2dSubsampling8(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/8 length)."""

    def __init__(self,
                 idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: nn.Module):
        """Construct an Conv2dSubsampling8 object.

        Args:
            idim (int): Input dimension.
            odim (int): Output dimension.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, odim, 3, 2),
                                  nn.ReLU(),
                                  nn.Conv2d(odim, odim, 3, 2),
                                  nn.ReLU(),
                                  nn.Conv2d(odim, odim, 3, 2),
                                  nn.ReLU(), )
        self.linear = nn.Linear(odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim)
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 8
        # 14 = (3 - 1) * 1 + (3 - 1) * 2 + (3 - 1) * 4
        self.right_context = 14

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, offset: int = 0
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
            offset (int): position encoding offset.
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: positional encoding
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = self.linear(x.transpose(1, 2).reshape([b, -1, c * f]))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]


class FrequencySELayer(nn.Module):
    """频率维度挤压-激励模块 (F-SE)"""

    def __init__(self, channels, freq_dim, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [Batch, Channels, Time, Freq]
        y = x.mean(dim=2, keepdim=True)  # 时间轴全局平均池化
        y = y.permute(0, 2, 3, 1)  # -> [B, 1, F, C]
        y = self.fc(y)  # 计算频率权重
        y = y.permute(0, 3, 1, 2)  # -> [B, C, 1, F]
        return x * y


class ParallelSpectralBranching4(BaseSubsampling):
    """并行频谱分支前端 (PSB4)"""

    def __init__(self, idim=80, odim=256, dropout_rate=0.1,
                 pos_enc_class=None, branch_channels: int = None,
                 alpha_init: float = 0.0):
        super().__init__()
        self.odim = odim
        branch_channels = odim // 2

        # 1. 低频专家 (0-40 bins) - 大卷积核
        self.low_branch = nn.Sequential(
            nn.Conv2d(1, branch_channels, kernel_size=(3, 5),
                      stride=(2, 2), padding=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(branch_channels, branch_channels,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
        )
        self.low_se = FrequencySELayer(branch_channels, 10)

        # 2. 中频专家 (30-70 bins)
        self.mid_branch = nn.Sequential(
            nn.Conv2d(1, branch_channels, kernel_size=3,
                      stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(branch_channels, branch_channels,
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.mid_se = FrequencySELayer(branch_channels, 10)

        # 3. 高频专家 (60-80 bins)
        self.high_branch = nn.Sequential(
            nn.Conv2d(1, branch_channels, kernel_size=3,
                      stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(branch_channels, branch_channels,
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.high_se = FrequencySELayer(branch_channels, 5)

        # 4. 融合投影 (10+10+5=25 bins)
        self.out = nn.Sequential(
            nn.Linear(branch_channels * 25, odim),
            nn.LayerNorm(odim),
            nn.Dropout(dropout_rate)
        )
        self.pos_enc = pos_enc_class  # 传入位置编码实例
        self.subsampling_rate = 4
        self.right_context = 6

    def forward(self, x, x_mask, offset=0):
        # x: [B, T, 80]
        x_low = x[:, :, 0:40].unsqueeze(1)
        x_mid = x[:, :, 30:70].unsqueeze(1)
        x_high = x[:, :, 60:80].unsqueeze(1)

        # 分支处理
        out_low = self.low_se(self.low_branch(x_low))
        out_mid = self.mid_se(self.mid_branch(x_mid))
        out_high = self.high_se(self.high_branch(x_high))

        # 对齐与拼接
        b, c, t, f = out_low.shape
        out_mid = out_mid[:, :, :t, :]
        out_high = out_high[:, :, :t, :]
        combined = torch.cat([out_low, out_mid, out_high],
                             dim=-1)

        # 展平与投影
        x_final = combined.transpose(1, 2).reshape(b, t, -1)
        x_final = self.out(x_final)

        # 掩码对齐 (4倍下采样)
        updated_mask = x_mask[:, :, :-2:2][:, :, :-2:2]
        if updated_mask.size(2) != x_final.size(1):
            x_final = x_final[:, :updated_mask.size(2), :]

        if self.pos_enc:
            x_final, pos_emb = self.pos_enc(x_final, offset)
            return x_final, pos_emb, updated_mask
        return x_final, None, updated_mask


class GlobalParallelSpectralBranching(BaseSubsampling):
    """全局+并行频谱分支前端 (GPSB)

    保底并联版本：conv2d + alpha * gpsb_delta。
    alpha 初始化为 0，保证初始行为接近基线 Conv2dSubsampling4。
    """

    def __init__(self, idim=80, odim=256, dropout_rate=0.1,
                 pos_enc_class=None, branch_channels: int = None,
                 alpha_init: float = 0.0):
        super().__init__()
        self.odim = odim
        # 保留基线主干，命名为 conv/out 以兼容旧 checkpoint 键名
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU(),
        )
        self.out = nn.Sequential(nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))

        # 降低增量分支容量，减少过拟合风险；允许从配置传入固定值做网格搜索
        if branch_channels is None:
            branch_channels = max(odim // 8, 16)
        else:
            branch_channels = max(int(branch_channels), 16)

        # 1. 全局专家 (0-80 bins)
        self.global_branch = nn.Sequential(
            nn.Conv2d(1, branch_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.global_se = FrequencySELayer(branch_channels, 20)

        # 2. 低频专家 (0-40 bins)
        self.low_branch = nn.Sequential(
            nn.Conv2d(1, branch_channels, kernel_size=(3, 5), stride=(2, 2), padding=(1, 2)),
            nn.SiLU(),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.SiLU(),
        )
        self.low_se = FrequencySELayer(branch_channels, 10)

        # 3. 中频专家 (30-70 bins)
        self.mid_branch = nn.Sequential(
            nn.Conv2d(1, branch_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.mid_se = FrequencySELayer(branch_channels, 10)

        # 4. 高频专家 (60-80 bins)
        self.high_branch = nn.Sequential(
            nn.Conv2d(1, branch_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.high_se = FrequencySELayer(branch_channels, 5)

        # 5. 增量分支投影 (20+10+10+5 = 45 bins)
        self.delta_out = nn.Sequential(
            nn.Linear(branch_channels * 45, odim),
            nn.LayerNorm(odim),
            nn.Dropout(dropout_rate)
        )
        self.fuse_norm = nn.LayerNorm(odim)
        self.alpha = nn.Parameter(torch.tensor([float(alpha_init)]))
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 4
        self.right_context = 6

    def forward(self, x, x_mask, offset=0):
        # x: [B, T, 80]
        x_4d = x.unsqueeze(1)
        base = self.conv(x_4d)
        b, c, t, f = base.shape
        base = self.out(base.transpose(1, 2).reshape(b, -1, c * f))

        x_global = x_4d
        x_low = x[:, :, 0:40].unsqueeze(1)
        x_mid = x[:, :, 30:70].unsqueeze(1)
        x_high = x[:, :, 60:80].unsqueeze(1)

        # 增量分支处理
        out_global = self.global_se(self.global_branch(x_global))
        out_low = self.low_se(self.low_branch(x_low))
        out_mid = self.mid_se(self.mid_branch(x_mid))
        out_high = self.high_se(self.high_branch(x_high))

        t_delta = min(out_global.size(2), out_low.size(2), out_mid.size(2), out_high.size(2))
        out_global = out_global[:, :, :t_delta, :]
        out_low = out_low[:, :, :t_delta, :]
        out_mid = out_mid[:, :, :t_delta, :]
        out_high = out_high[:, :, :t_delta, :]
        combined = torch.cat([out_global, out_low, out_mid, out_high], dim=-1)
        delta = combined.transpose(1, 2).reshape(b, t_delta, -1)
        delta = self.delta_out(delta)

        # 保底并联融合：初始 alpha=0，先复现基线，再学习增量收益
        t_final = min(base.size(1), delta.size(1))
        x_final = base[:, :t_final, :] + self.alpha * delta[:, :t_final, :]
        x_final = self.fuse_norm(x_final)

        # 掩码对齐
        updated_mask = x_mask[:, :, :-2:2][:, :, :-2:2]
        if updated_mask.size(2) != x_final.size(1):
            x_final = x_final[:, :updated_mask.size(2), :]

        if self.pos_enc:
            x_final, pos_emb = self.pos_enc(x_final, offset)
            return x_final, pos_emb, updated_mask
        return x_final, None, updated_mask
