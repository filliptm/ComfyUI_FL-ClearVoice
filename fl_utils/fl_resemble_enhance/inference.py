"""
Resemble-Enhance inference - standalone version without deepspeed dependency.

This module provides denoise() and enhance() functions that work without
importing the train.py files that depend on deepspeed.
"""

import logging
import os
import sys
import importlib.util
from functools import cache
from pathlib import Path

import torch

# Handle both package-style and direct imports
try:
    from .hparams import EnhancerHParams, DenoiserHParams
except ImportError:
    # Fallback for when loaded outside package context
    if "fl_resemble_enhance.hparams" in sys.modules:
        hparams_module = sys.modules["fl_resemble_enhance.hparams"]
        EnhancerHParams = hparams_module.EnhancerHParams
        DenoiserHParams = hparams_module.DenoiserHParams
    else:
        from hparams import EnhancerHParams, DenoiserHParams

logger = logging.getLogger(__name__)


def _get_cv_paths_module():
    """Get the cv_paths module, loading it if necessary."""
    cv_paths = sys.modules.get('cv_paths')
    if cv_paths is None:
        # Fallback: load directly if running standalone
        current_dir = Path(__file__).parent.parent  # fl_utils directory
        spec = importlib.util.spec_from_file_location("cv_paths", current_dir / "paths.py")
        cv_paths = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cv_paths)
    return cv_paths


def _get_download_utils_module():
    """Get the download_utils module, loading it if necessary."""
    download_utils = sys.modules.get('cv_download_utils')
    if download_utils is None:
        # Fallback: load directly if running standalone
        current_dir = Path(__file__).parent.parent  # fl_utils directory
        spec = importlib.util.spec_from_file_location("cv_download_utils", current_dir / "download_utils.py")
        download_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(download_utils)
    return download_utils


# Model download location - use centralized clear_voice directory
def get_models_dir():
    """Get the models directory for Resemble-Enhance."""
    cv_paths = _get_cv_paths_module()
    return cv_paths.get_resemble_enhance_dir()

REPO_ID = "ResembleAI/resemble-enhance"

# Files needed for Resemble-Enhance
RESEMBLE_ENHANCE_FILES = [
    "enhancer_stage2/hparams.yaml",
    "enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt",
    "denoiser/hparams.yaml",
    "denoiser/ds/G/default/mp_rank_00_model_states.pt",
]


def download_models():
    """Download the Resemble-Enhance models from HuggingFace with progress bars."""
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    run_dir = models_dir / "enhancer_stage2"

    # Check if already downloaded
    required_file = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
    if required_file.exists():
        logger.info(f"[FL ClearVoice] Resemble-Enhance models already downloaded at {run_dir}")
        return run_dir

    print(f"[FL ClearVoice] Downloading Resemble-Enhance models to {models_dir}...")

    try:
        # Use our custom download utility with progress bars
        download_utils = _get_download_utils_module()
        download_utils.download_with_progress(
            repo_id=REPO_ID,
            filenames=RESEMBLE_ENHANCE_FILES,
            local_dir=models_dir,
            prefix="[FL ClearVoice]"
        )
    except (ImportError, Exception) as e:
        # Fallback to standard huggingface_hub if our utility fails
        logger.warning("[FL ClearVoice] Custom download utility not available, using standard download")
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise RuntimeError(
                "huggingface_hub not installed. Please install with: pip install huggingface_hub"
            )

        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
        )

    if not run_dir.exists():
        raise RuntimeError(
            f"Model directory not found at {run_dir}. "
            "Download may have failed."
        )

    print(f"[FL ClearVoice] Resemble-Enhance models ready at {run_dir}")
    return run_dir


def _load_enhancer_model(run_dir, device):
    """
    Load enhancer model using our local standalone implementation.
    This bypasses the deepspeed import chain entirely.
    """
    # Handle both package-style and direct imports
    try:
        from .enhancer import Enhancer
    except ImportError:
        if "fl_resemble_enhance.enhancer" in sys.modules:
            Enhancer = sys.modules["fl_resemble_enhance.enhancer"].Enhancer
        else:
            from enhancer import Enhancer

    hp = EnhancerHParams.load(run_dir)
    enhancer = Enhancer(hp)

    path = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
    state_dict = torch.load(path, map_location="cpu")["module"]
    enhancer.load_state_dict(state_dict)
    enhancer.eval()
    enhancer.to(device)

    return enhancer


@cache
def load_enhancer(run_dir, device):
    """
    Load the enhancer model. Downloads if necessary.

    Args:
        run_dir: Path to the model directory, or None to auto-download
        device: Device to load model on

    Returns:
        Enhancer model
    """
    if run_dir is None:
        run_dir = download_models()

    return _load_enhancer_model(run_dir, device)


def clear_cache():
    """Clear the model cache to force reload."""
    load_enhancer.cache_clear()


def _inference(model, dwav, sr, device, chunk_seconds: float = 30.0, overlap_seconds: float = 1.0):
    """
    Run inference on audio using the model.
    Adapted from resemble_enhance.inference.inference
    """
    import time
    import torch.nn.functional as F
    from torch.nn.utils.parametrize import remove_parametrizations
    from torchaudio.functional import resample
    from tqdm import trange

    # Remove weight norm for inference
    for _, module in model.named_modules():
        try:
            remove_parametrizations(module, "weight")
        except Exception:
            pass

    hp = model.hp

    # Resample to model's expected rate
    dwav = resample(
        dwav,
        orig_freq=sr,
        new_freq=hp.wav_rate,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )

    sr = hp.wav_rate

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    chunk_length = int(sr * chunk_seconds)
    overlap_length = int(sr * overlap_seconds)
    hop_length = chunk_length - overlap_length

    chunks = []
    for start in trange(0, dwav.shape[-1], hop_length, desc="Processing chunks"):
        chunk = _inference_chunk(model, dwav[start:start + chunk_length], sr, device)
        chunks.append(chunk)

    hwav = _merge_chunks(chunks, chunk_length, hop_length, sr=sr, length=dwav.shape[-1])

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_time = time.perf_counter() - start_time
    logger.info(f"Elapsed time: {elapsed_time:.3f} s, {hwav.shape[-1] / elapsed_time / 1000:.3f} kHz")

    return hwav, sr


@torch.inference_mode()
def _inference_chunk(model, dwav, sr, device, npad=441):
    """Process a single chunk of audio."""
    import torch.nn.functional as F

    assert model.hp.wav_rate == sr, f"Expected {model.hp.wav_rate} Hz, got {sr} Hz"

    length = dwav.shape[-1]
    abs_max = dwav.abs().max().clamp(min=1e-7)

    assert dwav.dim() == 1, f"Expected 1D waveform, got {dwav.dim()}D"
    dwav = dwav.to(device)
    dwav = dwav / abs_max  # Normalize
    dwav = F.pad(dwav, (0, npad))
    hwav = model(dwav[None])[0].cpu()  # (T,)
    hwav = hwav[:length]  # Trim padding
    hwav = hwav * abs_max  # Unnormalize

    return hwav


def _merge_chunks(chunks, chunk_length, hop_length, sr=44100, length=None):
    """Merge overlapping audio chunks."""
    import torch.nn.functional as F
    from torchaudio.transforms import MelSpectrogram

    def compute_corr(x, y):
        return torch.fft.ifft(torch.fft.fft(x) * torch.fft.fft(y).conj()).abs()

    def compute_offset(chunk1, chunk2, sr=44100):
        hop_len = sr // 200
        win_len = hop_len * 4
        n_fft = 2 ** (win_len - 1).bit_length()

        mel_fn = MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=win_len,
            hop_length=hop_len,
            n_mels=80,
            f_min=0.0,
            f_max=sr // 2,
        )

        spec1 = mel_fn(chunk1).log1p()
        spec2 = mel_fn(chunk2).log1p()

        corr = compute_corr(spec1, spec2)
        corr = corr.mean(dim=0)

        argmax = corr.argmax().item()

        if argmax > len(corr) // 2:
            argmax -= len(corr)

        offset = -argmax * hop_len
        return offset

    signal_length = (len(chunks) - 1) * hop_length + chunk_length
    overlap_length = chunk_length - hop_length
    signal = torch.zeros(signal_length, device=chunks[0].device)

    fadein = torch.linspace(0, 1, overlap_length, device=chunks[0].device)
    fadein = torch.cat([fadein, torch.ones(hop_length, device=chunks[0].device)])
    fadeout = torch.linspace(1, 0, overlap_length, device=chunks[0].device)
    fadeout = torch.cat([torch.ones(hop_length, device=chunks[0].device), fadeout])

    for i, chunk in enumerate(chunks):
        start = i * hop_length
        end = start + chunk_length

        if len(chunk) < chunk_length:
            chunk = F.pad(chunk, (0, chunk_length - len(chunk)))

        if i > 0:
            pre_region = chunks[i - 1][-overlap_length:]
            cur_region = chunk[:overlap_length]
            offset = compute_offset(pre_region, cur_region, sr=sr)
            start -= offset
            end -= offset

        if i == 0:
            chunk = chunk * fadeout
        elif i == len(chunks) - 1:
            chunk = chunk * fadein
        else:
            chunk = chunk * fadein * fadeout

        signal[start:end] += chunk[:len(signal[start:end])]

    signal = signal[:length]
    return signal


@torch.inference_mode()
def denoise(dwav, sr, device, run_dir=None):
    """
    Denoise audio using the Resemble-Enhance denoiser.

    Args:
        dwav: Input waveform tensor (1D)
        sr: Sample rate
        device: Device to run on
        run_dir: Optional path to model directory

    Returns:
        Tuple of (denoised_waveform, sample_rate)
    """
    enhancer = load_enhancer(run_dir, device)
    return _inference(model=enhancer.denoiser, dwav=dwav, sr=sr, device=device)


@torch.inference_mode()
def enhance(dwav, sr, device, nfe=32, solver="midpoint", lambd=0.5, tau=0.5, run_dir=None):
    """
    Enhance audio using the full Resemble-Enhance pipeline.

    Args:
        dwav: Input waveform tensor (1D)
        sr: Sample rate
        device: Device to run on
        nfe: Number of function evaluations (1-128)
        solver: Solver method ('midpoint', 'rk4', 'euler')
        lambd: Denoiser strength (0-1)
        tau: Prior temperature (0-1)
        run_dir: Optional path to model directory

    Returns:
        Tuple of (enhanced_waveform, sample_rate)
    """
    assert 0 < nfe <= 128, f"nfe must be in (0, 128], got {nfe}"
    assert solver in ("midpoint", "rk4", "euler"), f"solver must be in ('midpoint', 'rk4', 'euler'), got {solver}"
    assert 0 <= lambd <= 1, f"lambd must be in [0, 1], got {lambd}"
    assert 0 <= tau <= 1, f"tau must be in [0, 1], got {tau}"

    enhancer = load_enhancer(run_dir, device)
    enhancer.configurate_(nfe=nfe, solver=solver, lambd=lambd, tau=tau)
    return _inference(model=enhancer, dwav=dwav, sr=sr, device=device)
