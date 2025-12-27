"""
Audio utility functions for FL-ClearVoice
Handles conversion between ComfyUI AUDIO format and tensors/numpy arrays
"""

import torch
import numpy as np
from typing import Tuple, Optional

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


def tensor_to_comfyui_audio(waveform: torch.Tensor, sample_rate: int) -> dict:
    """
    Convert a waveform tensor to ComfyUI AUDIO format.

    Args:
        waveform: Audio tensor, can be:
            - [samples] - mono, no batch
            - [channels, samples] - stereo, no batch
            - [batch, channels, samples] - full format
        sample_rate: Sample rate in Hz

    Returns:
        dict with 'waveform' and 'sample_rate'
    """
    # Ensure CPU
    if waveform.device != torch.device('cpu'):
        waveform = waveform.cpu()

    # Normalize to [batch, channels, samples]
    if waveform.ndim == 1:
        # [samples] -> [1, 1, samples]
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2:
        # [channels, samples] -> [1, channels, samples]
        waveform = waveform.unsqueeze(0)

    return {
        "waveform": waveform,
        "sample_rate": sample_rate
    }


def comfyui_audio_to_tensor(audio: dict) -> Tuple[torch.Tensor, int]:
    """
    Extract waveform tensor and sample rate from ComfyUI AUDIO format.

    Args:
        audio: ComfyUI AUDIO dict with 'waveform' and 'sample_rate'

    Returns:
        Tuple of (waveform tensor [batch, channels, samples], sample_rate)
    """
    waveform = audio['waveform']
    sample_rate = audio['sample_rate']

    # Ensure 3D
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)

    return waveform, sample_rate


def resample_audio(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """
    Resample audio to target sample rate.

    Args:
        waveform: Audio tensor [batch, channels, samples] or [channels, samples]
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled waveform tensor
    """
    if orig_sr == target_sr:
        return waveform

    # Remember original shape
    needs_batch_dim = waveform.ndim == 2
    if needs_batch_dim:
        waveform = waveform.unsqueeze(0)

    batch, channels, samples = waveform.shape

    # Use torchaudio if available (fastest)
    if HAS_TORCHAUDIO:
        # Process each batch item
        resampled = []
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        for b in range(batch):
            resampled.append(resampler(waveform[b]))
        waveform = torch.stack(resampled, dim=0)
    elif HAS_LIBROSA:
        # Fallback to librosa
        resampled = []
        for b in range(batch):
            batch_resampled = []
            for c in range(channels):
                audio_np = waveform[b, c].numpy()
                resampled_np = librosa.resample(audio_np, orig_sr=orig_sr, target_sr=target_sr)
                batch_resampled.append(torch.from_numpy(resampled_np))
            resampled.append(torch.stack(batch_resampled, dim=0))
        waveform = torch.stack(resampled, dim=0)
    else:
        raise RuntimeError("No resampling library available. Install torchaudio or librosa.")

    if needs_batch_dim:
        waveform = waveform.squeeze(0)

    return waveform


def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    Convert stereo/multichannel audio to mono by averaging channels.

    Args:
        waveform: Audio tensor [batch, channels, samples] or [channels, samples]

    Returns:
        Mono waveform tensor
    """
    if waveform.ndim == 2:
        # [channels, samples]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.ndim == 3:
        # [batch, channels, samples]
        if waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)

    return waveform


def get_audio_duration(audio: dict) -> float:
    """
    Get duration of audio in seconds.

    Args:
        audio: ComfyUI AUDIO dict

    Returns:
        Duration in seconds
    """
    waveform = audio['waveform']
    sample_rate = audio['sample_rate']
    return waveform.shape[-1] / sample_rate


def tensor_to_numpy_for_clearvoice(waveform: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to NumPy array in ClearVoice's expected format.

    ClearVoice expects: [batch, samples] as float32

    Args:
        waveform: Audio tensor [batch, channels, samples] or [channels, samples] or [samples]

    Returns:
        NumPy array shaped [1, samples]
    """
    # Ensure CPU
    if waveform.device != torch.device('cpu'):
        waveform = waveform.cpu()

    # Flatten to 1D samples
    if waveform.ndim == 3:
        # [batch, channels, samples] -> [samples]
        waveform = waveform.squeeze(0).squeeze(0)
    elif waveform.ndim == 2:
        # [channels, samples] -> [samples]
        waveform = waveform.squeeze(0)

    # Convert to numpy and reshape to [1, samples]
    audio_np = waveform.numpy().astype(np.float32)
    return np.reshape(audio_np, [1, audio_np.shape[0]])


def numpy_to_tensor_from_clearvoice(audio_np: np.ndarray) -> torch.Tensor:
    """
    Convert ClearVoice output NumPy array back to PyTorch tensor.

    ClearVoice returns: [batch, samples] for enhancement/SR

    Args:
        audio_np: NumPy array from ClearVoice

    Returns:
        PyTorch tensor shaped [channels, samples]
    """
    # Handle different output shapes
    if audio_np.ndim == 2:
        # [batch, samples] -> [samples]
        audio_np = audio_np[0, :]

    # Convert to tensor and add channel dimension
    waveform = torch.from_numpy(audio_np.astype(np.float32))

    # Add channel dimension: [samples] -> [1, samples]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    return waveform
