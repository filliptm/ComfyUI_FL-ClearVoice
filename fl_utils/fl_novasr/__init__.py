"""
FL NovaSR - Standalone NovaSR integration for FL ClearVoice
Ultra-lightweight audio super-resolution (16kHz -> 48kHz)

Based on NovaSR by Yatharth Sharma (https://github.com/ysharma3501/NovaSR)
Modified for integration with FL ClearVoice node pack.
"""

import torch
import os
from .speechsr import SynthesizerTrn


class FastSR:
    """
    Fast audio super-resolution model.

    Upsamples 16kHz audio to 48kHz using a lightweight neural network.

    Args:
        ckpt_path: Path to the model checkpoint file (required)
        device: Device to run inference on ('cuda', 'mps', 'cpu', or None for auto-detect)
    """

    def __init__(self, ckpt_path=None, device=None):
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        # Model hyperparameters (fixed for NovaSR)
        self.hps = {
            "train": {
                "segment_size": 9600
            },
            "data": {
                "hop_length": 320,
                "n_mel_channels": 128
            },
            "model": {
                "resblock": "0",
                "resblock_kernel_sizes": [11],
                "resblock_dilation_sizes": [[1, 3, 5]],
                "upsample_initial_channel": 32,
            }
        }

        # Require checkpoint path (download handled externally)
        if ckpt_path is None:
            raise ValueError(
                "ckpt_path is required. Model download is handled by FL ClearVoice."
            )

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Load model
        self.model = self._load_model(ckpt_path).half().eval()

    def _load_model(self, ckpt_path):
        """Load model from checkpoint."""
        model = SynthesizerTrn(
            self.hps['data']['n_mel_channels'],
            self.hps['train']['segment_size'] // self.hps['data']['hop_length'],
            **self.hps['model']
        ).to(self.device)

        checkpoint_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        model.dec.remove_weight_norm()
        model.load_state_dict(checkpoint_dict, strict=True)
        model.eval()

        return model

    def load_audio(self, audio_file):
        """
        Load audio file and prepare for inference.

        Args:
            audio_file: Path to audio file

        Returns:
            Tensor of shape [1, 1, samples] at 16kHz
        """
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa is required for audio loading: pip install librosa")

        y, sr = librosa.load(audio_file, sr=16000)
        lowres_wav = torch.from_numpy(y).unsqueeze(0).half().unsqueeze(1).to(self.device)
        return lowres_wav

    def infer(self, lowres_wav):
        """
        Run super-resolution inference.

        Args:
            lowres_wav: Input tensor of shape [batch, 1, samples] at 16kHz (float16)

        Returns:
            Output tensor of shape [1, samples*3] at 48kHz (float16)
            Note: Returns [channel, samples] format matching original NovaSR
        """
        with torch.no_grad():
            new_wav = self.model(lowres_wav)

        # Original NovaSR does squeeze(0) to remove batch dim
        # Output: [batch, 1, samples] -> [1, samples] (channel, samples)
        return new_wav.squeeze(0)
