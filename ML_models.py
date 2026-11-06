"""Single TBDT model used by every paper training configuration."""

from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F


TBDT_MODEL_NAME = "CNN1D_TBDT"


def _validate_kernel_size(kernel_size: int) -> int:
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, Integral):
        raise ValueError(f"kernel_size must be an odd positive integer, got {kernel_size!r}")
    kernel_size = int(kernel_size)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be an odd positive integer, got {kernel_size}")
    return kernel_size


class _ChannelDropout1D(nn.Module):
    """Apply channel-wise dropout to a (batch, channel, time) tensor."""

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = float(p)
        self.inplace = bool(inplace)
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"dropout probability must be in [0, 1], got {p}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        if x.ndim != 3:
            raise ValueError(
                "ChannelDropout1D expects (batch, channels, time), "
                f"got {tuple(x.shape)}"
            )
        if self.p == 1.0:
            return x.zero_() if self.inplace else torch.zeros_like(x)
        keep_probability = 1.0 - self.p
        mask = x.new_empty((x.shape[0], x.shape[1], 1)).bernoulli_(keep_probability)
        mask = mask.div_(keep_probability)
        if self.inplace:
            return x.mul_(mask)
        return x * mask


class _ResidualConvBlock1D(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        gn_groups: int = 8,
        dropout2d_p: float = 0.10,
    ):
        super().__init__()
        padding = int(kernel_size) // 2
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=1,
            bias=False,
        )
        self.gn1 = nn.GroupNorm(num_groups=gn_groups, num_channels=channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=1,
            bias=False,
        )
        self.gn2 = nn.GroupNorm(num_groups=gn_groups, num_channels=channels)
        self.drop2d = _ChannelDropout1D(p=dropout2d_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        output = F.relu(self.gn1(self.conv1(x)))
        output = self.drop2d(output)
        output = self.gn2(self.conv2(output))
        output = F.relu(output + identity)
        return self.drop2d(output)


class _GroupPreservingResidualConvBlock1D(nn.Module):
    """Grouped block used to preserve the model's seeded initialization sequence."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        vg_groups: int = 8,
        gn_groups: int = 8,
        dropout2d_p: float = 0.10,
    ):
        super().__init__()
        padding = int(kernel_size) // 2
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=vg_groups,
            bias=False,
        )
        self.gn1 = nn.GroupNorm(num_groups=gn_groups, num_channels=channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=vg_groups,
            bias=False,
        )
        self.gn2 = nn.GroupNorm(num_groups=gn_groups, num_channels=channels)
        self.drop2d = _ChannelDropout1D(p=dropout2d_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        output = F.relu(self.gn1(self.conv1(x)))
        output = self.drop2d(output)
        output = self.gn2(self.conv2(output))
        output = F.relu(output + identity)
        return self.drop2d(output)


class CNN1D_TBDT(nn.Module):
    """TBDT with average pooling, a fixed-scale cosine head, and late mixing."""

    def __init__(
        self,
        dim_v: int = 96,
        hidden_size: int = 60,
        num_classes: int = 8,
        num_layers: int = 1,
        dropout_p: float = 0.65,
        dropout2d_p: float = 0.05,
        vg_groups: int = 12,
        kernel_size: int = 5,
        gn_groups: int = 12,
        cosine_scale: float = 16.0,
        vg_group_dropout_p: float = 0.0,
    ):
        super().__init__()
        dim_v = int(dim_v)
        hidden_size = int(hidden_size)
        num_classes = int(num_classes)
        num_layers = int(num_layers)
        vg_groups = int(vg_groups)
        gn_groups = int(gn_groups)
        kernel_size = _validate_kernel_size(kernel_size)

        if dim_v <= 0 or hidden_size <= 0 or num_classes <= 0 or num_layers <= 0:
            raise ValueError("dim_v, hidden_size, num_classes, and num_layers must be positive")
        if vg_groups <= 0 or dim_v % vg_groups != 0 or hidden_size % vg_groups != 0:
            raise ValueError(
                "dim_v and hidden_size must both be divisible by vg_groups; "
                f"got dim_v={dim_v}, hidden_size={hidden_size}, vg_groups={vg_groups}"
            )
        if gn_groups <= 0 or hidden_size % gn_groups != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by gn_groups={gn_groups}"
            )
        if float(cosine_scale) <= 0.0:
            raise ValueError(f"cosine_scale must be positive, got {cosine_scale}")
        if not 0.0 <= float(vg_group_dropout_p) < 1.0:
            raise ValueError(
                f"vg_group_dropout_p must be in [0, 1), got {vg_group_dropout_p}"
            )

        self.dim_v = dim_v
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vg_groups = vg_groups
        self.vg_group_dropout_p = float(vg_group_dropout_p)

        self.stem = nn.Conv1d(
            dim_v,
            hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=vg_groups,
            bias=False,
        )
        self.stem_gn = nn.GroupNorm(num_groups=gn_groups, num_channels=hidden_size)
        self.stem_drop = _ChannelDropout1D(p=dropout2d_p)
        # Initialize these temporary grouped blocks before replacing them below.
        # This fixes the random-number sequence used for reproducible weights.
        self.blocks = nn.ModuleList(
            [
                _GroupPreservingResidualConvBlock1D(
                    channels=hidden_size,
                    kernel_size=kernel_size,
                    vg_groups=vg_groups,
                    gn_groups=gn_groups,
                    dropout2d_p=dropout2d_p,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.mix = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, bias=False)
        self.mix_gn = nn.GroupNorm(num_groups=gn_groups, num_channels=hidden_size)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_p)
        self.feature_dim = hidden_size
        self.class_weight = nn.Parameter(torch.empty(num_classes, self.feature_dim))
        nn.init.normal_(self.class_weight, mean=0.0, std=0.01)
        self.register_buffer(
            "_fixed_cosine_scale",
            torch.tensor(float(cosine_scale), dtype=torch.float32),
        )

        self.blocks = nn.ModuleList(
            [
                _ResidualConvBlock1D(
                    channels=hidden_size,
                    kernel_size=kernel_size,
                    gn_groups=gn_groups,
                    dropout2d_p=dropout2d_p,
                )
                for _ in range(self.num_layers)
            ]
        )

    def _apply_vg_group_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.vg_group_dropout_p <= 0.0:
            return x
        batch_size, _, time_length = x.shape
        channels_per_group = self.hidden_size // self.vg_groups
        grouped = x.reshape(
            batch_size,
            self.vg_groups,
            channels_per_group,
            time_length,
        )
        keep_mask = (
            torch.rand(batch_size, self.vg_groups, 1, 1, device=x.device)
            >= self.vg_group_dropout_p
        ).to(dtype=x.dtype)
        grouped = grouped * keep_mask / (1.0 - self.vg_group_dropout_p)
        return grouped.reshape(batch_size, self.hidden_size, time_length)

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.dim_v:
            raise ValueError(
                f"CNN1D_TBDT expects (batch, time, {self.dim_v}), got {tuple(x.shape)}"
            )
        output = x.transpose(1, 2)
        output = F.relu(self.stem_gn(self.stem(output)))
        output = self.stem_drop(output)
        output = self._apply_vg_group_dropout(output)
        for block in self.blocks:
            output = block(output)
            output = self._apply_vg_group_dropout(output)
        return F.relu(self.mix_gn(self.mix(output)))

    def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.avg_pool(x).squeeze(-1)
        return self.dropout(features)

    def _classify(self, features: torch.Tensor) -> torch.Tensor:
        normalized_features = F.normalize(features, dim=1)
        normalized_weights = F.normalize(self.class_weight, dim=1)
        return self._fixed_cosine_scale * F.linear(normalized_features, normalized_weights)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self._pool_features(self._forward_backbone(x))
        logits = self._classify(features)
        if return_features:
            return logits, features
        return logits, {
            "classifier_head": "cosine",
            "features": features.detach(),
        }

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._pool_features(self._forward_backbone(x))


def build_model(model_name: str = TBDT_MODEL_NAME, **kwargs) -> CNN1D_TBDT:
    """Construct the repository's only supported model."""
    resolved_name = str(model_name or TBDT_MODEL_NAME).strip()
    if resolved_name != TBDT_MODEL_NAME:
        raise ValueError(
            f"Unknown model_name {resolved_name!r}; only {TBDT_MODEL_NAME!r} is supported"
        )
    return CNN1D_TBDT(**kwargs)


__all__ = ["CNN1D_TBDT", "TBDT_MODEL_NAME", "build_model"]

