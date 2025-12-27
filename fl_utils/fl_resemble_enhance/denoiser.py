"""
Denoiser model - extracted from resemble_enhance without deepspeed dependency.
"""

import logging
import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)


def _normalize(x: Tensor) -> Tensor:
    return x / (x.abs().max(dim=-1, keepdim=True).values + 1e-7)


class PreactResBlock(nn.Sequential):
    def __init__(self, dim):
        super().__init__(
            nn.GroupNorm(dim // 16, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(dim // 16, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x):
        return x + super().forward(x)


class UNetBlock(nn.Module):
    def __init__(self, input_dim, output_dim=None, scale_factor=1.0):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.pre_conv = nn.Conv2d(input_dim, output_dim, 3, padding=1)
        self.res_block1 = PreactResBlock(output_dim)
        self.res_block2 = PreactResBlock(output_dim)
        self.downsample = self.upsample = nn.Identity()
        if scale_factor > 1:
            self.upsample = nn.Upsample(scale_factor=scale_factor)
        elif scale_factor < 1:
            self.downsample = nn.Upsample(scale_factor=scale_factor)

    def forward(self, x, h=None):
        x = self.upsample(x)
        if h is not None:
            assert x.shape == h.shape, f"{x.shape} != {h.shape}"
            x = x + h
        x = self.pre_conv(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.downsample(x), x


class UNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=16, num_blocks=4, num_middle_blocks=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_proj = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.encoder_blocks = nn.ModuleList(
            [
                UNetBlock(input_dim=hidden_dim * 2**i, output_dim=hidden_dim * 2 ** (i + 1), scale_factor=0.5)
                for i in range(num_blocks)
            ]
        )
        self.middle_blocks = nn.ModuleList(
            [UNetBlock(input_dim=hidden_dim * 2**num_blocks) for _ in range(num_middle_blocks)]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                UNetBlock(input_dim=hidden_dim * 2 ** (i + 1), output_dim=hidden_dim * 2**i, scale_factor=2)
                for i in reversed(range(num_blocks))
            ]
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_dim, 1),
        )

    @property
    def scale_factor(self):
        return 2 ** len(self.encoder_blocks)

    def pad_to_fit(self, x):
        hpad = (self.scale_factor - x.shape[2] % self.scale_factor) % self.scale_factor
        wpad = (self.scale_factor - x.shape[3] % self.scale_factor) % self.scale_factor
        return F.pad(x, (0, wpad, 0, hpad))

    def forward(self, x):
        shape = x.shape
        x = self.pad_to_fit(x)
        x = self.input_proj(x)

        s_list = []
        for block in self.encoder_blocks:
            x, s = block(x)
            s_list.append(s)

        for block in self.middle_blocks:
            x, _ = block(x)

        for block, s in zip(self.decoder_blocks, reversed(s_list)):
            x, _ = block(x, s)

        x = self.head(x)
        x = x[..., : shape[2], : shape[3]]
        return x


class Denoiser(nn.Module):
    @property
    def stft_cfg(self) -> dict:
        hop_size = self.hp.hop_size
        return dict(hop_length=hop_size, n_fft=hop_size * 4, win_length=hop_size * 4)

    @property
    def n_fft(self):
        return self.stft_cfg["n_fft"]

    @property
    def eps(self):
        return 1e-7

    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.net = UNet(input_dim=3, output_dim=3)

        # Import MelSpectrogram from the package (safe, no deepspeed chain)
        from resemble_enhance.melspec import MelSpectrogram
        self.mel_fn = MelSpectrogram(hp)

        self.dummy: Tensor
        self.register_buffer("dummy", torch.zeros(1), persistent=False)

    def to_mel(self, x: Tensor, drop_last=True):
        if drop_last:
            return self.mel_fn(x)[..., :-1]
        return self.mel_fn(x)

    def _stft(self, x):
        dtype = x.dtype
        device = x.device

        if x.is_mps:
            x = x.cpu()

        window = torch.hann_window(self.stft_cfg["win_length"], device=x.device)
        s = torch.stft(x.float(), **self.stft_cfg, window=window, return_complex=True)
        s = s[..., :-1]

        mag = s.abs()
        phi = s.angle()
        cos = phi.cos()
        sin = phi.sin()

        mag = mag.to(dtype=dtype, device=device)
        cos = cos.to(dtype=dtype, device=device)
        sin = sin.to(dtype=dtype, device=device)

        return mag, cos, sin

    def _istft(self, mag: Tensor, cos: Tensor, sin: Tensor):
        device = mag.device
        dtype = mag.dtype

        if mag.is_mps:
            mag = mag.cpu()
            cos = cos.cpu()
            sin = sin.cpu()

        real = mag * cos
        imag = mag * sin
        s = torch.complex(real, imag)

        if s.isnan().any():
            logger.warning("NaN detected in ISTFT input.")

        s = F.pad(s, (0, 1), "replicate")

        window = torch.hann_window(self.stft_cfg["win_length"], device=s.device)
        x = torch.istft(s, **self.stft_cfg, window=window, return_complex=False)

        if x.isnan().any():
            logger.warning("NaN detected in ISTFT output, set to zero.")
            x = torch.where(x.isnan(), torch.zeros_like(x), x)

        x = x.to(dtype=dtype, device=device)
        return x

    def _magphase(self, real, imag):
        mag = (real.pow(2) + imag.pow(2) + self.eps).sqrt()
        cos = real / mag
        sin = imag / mag
        return mag, cos, sin

    def _predict(self, mag: Tensor, cos: Tensor, sin: Tensor):
        x = torch.stack([mag, cos, sin], dim=1)
        mag_mask, real, imag = self.net(x).unbind(1)
        mag_mask = mag_mask.sigmoid()
        real = real.tanh()
        imag = imag.tanh()
        _, cos_res, sin_res = self._magphase(real, imag)
        return mag_mask, sin_res, cos_res

    def _separate(self, mag, cos, sin, mag_mask, cos_res, sin_res):
        sep_mag = F.relu(mag * mag_mask)
        sep_cos = cos * cos_res - sin * sin_res
        sep_sin = sin * cos_res + cos * sin_res
        return sep_mag, sep_cos, sep_sin

    def forward(self, x: Tensor, y: Tensor = None):
        assert x.dim() == 2, f"Expected (b t), got {x.size()}"
        x = x.to(self.dummy)
        x = _normalize(x)

        if y is not None:
            assert y.dim() == 2, f"Expected (b t), got {y.size()}"
            y = y.to(self.dummy)
            y = _normalize(y)

        mag, cos, sin = self._stft(x)
        mag_mask, sin_res, cos_res = self._predict(mag, cos, sin)
        sep_mag, sep_cos, sep_sin = self._separate(mag, cos, sin, mag_mask, cos_res, sin_res)

        o = self._istft(sep_mag, sep_cos, sep_sin)

        npad = x.shape[-1] - o.shape[-1]
        o = F.pad(o, (0, npad))

        if y is not None:
            self.losses = dict(l1=F.l1_loss(o, y))

        return o
