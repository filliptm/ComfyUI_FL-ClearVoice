"""
FL ClearVoice Process Node
Unified audio processing - automatically adapts to model type and backend
"""

import torch
import numpy as np
import tempfile
import os
from typing import Tuple
import sys

# Import from pre-loaded modules (loaded by __init__.py)
# These use unique names to avoid conflicts with other FL node packs
if "cv_audio_utils" in sys.modules:
    cv_audio_utils = sys.modules["cv_audio_utils"]
    comfyui_audio_to_tensor = cv_audio_utils.comfyui_audio_to_tensor
    tensor_to_comfyui_audio = cv_audio_utils.tensor_to_comfyui_audio
    resample_audio = cv_audio_utils.resample_audio
    ensure_mono = cv_audio_utils.ensure_mono
    get_audio_duration = cv_audio_utils.get_audio_duration
    tensor_to_numpy_for_clearvoice = cv_audio_utils.tensor_to_numpy_for_clearvoice
    numpy_to_tensor_from_clearvoice = cv_audio_utils.numpy_to_tensor_from_clearvoice

    cv_model_manager = sys.modules["cv_model_manager"]
    MODEL_SAMPLE_RATES = cv_model_manager.MODEL_SAMPLE_RATES
    SR_MODELS = cv_model_manager.SR_MODELS
    RESTORATION_MODELS = cv_model_manager.RESTORATION_MODELS
    DENOISE_ONLY_MODELS = cv_model_manager.DENOISE_ONLY_MODELS
    MODEL_BACKEND = cv_model_manager.MODEL_BACKEND
else:
    # Fallback for standalone testing
    import importlib.util
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    spec = importlib.util.spec_from_file_location(
        "cv_audio_utils",
        os.path.join(current_dir, "fl_utils", "audio_utils.py")
    )
    cv_audio_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cv_audio_utils)
    comfyui_audio_to_tensor = cv_audio_utils.comfyui_audio_to_tensor
    tensor_to_comfyui_audio = cv_audio_utils.tensor_to_comfyui_audio
    resample_audio = cv_audio_utils.resample_audio
    ensure_mono = cv_audio_utils.ensure_mono
    get_audio_duration = cv_audio_utils.get_audio_duration
    tensor_to_numpy_for_clearvoice = cv_audio_utils.tensor_to_numpy_for_clearvoice
    numpy_to_tensor_from_clearvoice = cv_audio_utils.numpy_to_tensor_from_clearvoice

    spec2 = importlib.util.spec_from_file_location(
        "cv_model_manager",
        os.path.join(current_dir, "fl_utils", "model_manager.py")
    )
    cv_model_manager = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(cv_model_manager)
    MODEL_SAMPLE_RATES = cv_model_manager.MODEL_SAMPLE_RATES
    SR_MODELS = cv_model_manager.SR_MODELS
    RESTORATION_MODELS = cv_model_manager.RESTORATION_MODELS
    DENOISE_ONLY_MODELS = cv_model_manager.DENOISE_ONLY_MODELS
    MODEL_BACKEND = cv_model_manager.MODEL_BACKEND


class FL_ClearVoice_Process:
    """
    Process audio using various speech enhancement models.

    Automatically detects the model type and applies the appropriate processing:
    - ClearVoice models: Enhancement and super-resolution
    - Resemble-Enhance: Denoising and full enhancement
    - VoiceFixer: All-in-one restoration
    """

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "message")
    FUNCTION = "process_audio"
    CATEGORY = "🎵FL ClearVoice"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("CLEARVOICE_MODEL", {
                    "description": "Model from Model Loader"
                }),
                "audio": ("AUDIO", {
                    "description": "Input audio to process"
                }),
            },
        }

    def _process_clearvoice(self, model_info: dict, waveform: torch.Tensor, orig_sr: int) -> Tuple[torch.Tensor, int]:
        """Process audio using ClearVoice backend."""
        cv = model_info["model"]
        model_name = model_info["model_name"]
        input_sr = model_info.get("input_sample_rate")
        output_sr = model_info["output_sample_rate"]
        is_sr_model = model_name in SR_MODELS

        # Handle sample rate conversion
        if is_sr_model:
            # SR model expects 48kHz input
            if orig_sr != 48000:
                print(f"[FL ClearVoice] Resampling {orig_sr}Hz -> 48000Hz for SR processing...")
                waveform = resample_audio(waveform, orig_sr, 48000)
        else:
            # SE model: resample to model's expected input rate
            target_input_sr = input_sr if input_sr else output_sr
            if orig_sr != target_input_sr:
                print(f"[FL ClearVoice] Resampling {orig_sr}Hz -> {target_input_sr}Hz...")
                waveform = resample_audio(waveform, orig_sr, target_input_sr)

        # Convert to mono
        original_channels = waveform.shape[1]
        if original_channels > 1:
            print(f"[FL ClearVoice] Converting {original_channels} channels to mono...")
            waveform = ensure_mono(waveform)

        # Convert to NumPy for ClearVoice
        audio_np = tensor_to_numpy_for_clearvoice(waveform)

        print(f"[FL ClearVoice] Processing with ClearVoice...")
        output_np = cv(audio_np, online_write=False)

        # Convert back to tensor
        processed_waveform = numpy_to_tensor_from_clearvoice(output_np)

        # Restore stereo if needed
        if original_channels > 1:
            processed_waveform = processed_waveform.repeat(original_channels, 1)

        return processed_waveform, output_sr

    def _process_resemble_enhance(self, model_info: dict, waveform: torch.Tensor, orig_sr: int) -> Tuple[torch.Tensor, int]:
        """Process audio using Resemble-Enhance backend."""
        model_name = model_info["model_name"]
        denoise_func = model_info["denoise_func"]
        enhance_func = model_info["enhance_func"]
        device = model_info["device"]
        output_sr = model_info["output_sample_rate"]  # 44100

        # Resample to 44.1kHz if needed
        if orig_sr != 44100:
            print(f"[FL ClearVoice] Resampling {orig_sr}Hz -> 44100Hz for Resemble-Enhance...")
            waveform = resample_audio(waveform, orig_sr, 44100)

        # Convert to mono
        original_channels = waveform.shape[1]
        if original_channels > 1:
            print(f"[FL ClearVoice] Converting {original_channels} channels to mono...")
            waveform = ensure_mono(waveform)

        # Prepare tensor for Resemble-Enhance (expects [samples] 1D tensor)
        # waveform is [batch, channels, samples] -> get [samples]
        audio_tensor = waveform.squeeze(0).squeeze(0)  # [samples]

        print(f"[FL ClearVoice] Processing with Resemble-Enhance ({model_name})...")

        if model_name == "Resemble_Denoise":
            # Denoise only
            processed, new_sr = denoise_func(audio_tensor, 44100, device)
        else:
            # Full enhancement (denoise + enhance)
            processed, new_sr = enhance_func(audio_tensor, 44100, device, nfe=32, solver="midpoint", lambd=0.5, tau=0.5)

        # Convert back to proper shape [channels, samples]
        if processed.ndim == 1:
            processed_waveform = processed.unsqueeze(0)  # [1, samples]
        else:
            processed_waveform = processed

        # Restore stereo if needed
        if original_channels > 1:
            processed_waveform = processed_waveform.repeat(original_channels, 1)

        return processed_waveform.cpu(), new_sr

    def _process_voicefixer(self, model_info: dict, waveform: torch.Tensor, orig_sr: int) -> Tuple[torch.Tensor, int]:
        """Process audio using VoiceFixer backend."""
        import soundfile as sf

        vf = model_info["model"]
        use_cuda = model_info["use_cuda"]
        output_sr = model_info["output_sample_rate"]  # 44100

        # VoiceFixer works with file I/O, so we need temp files
        # Convert to mono first
        original_channels = waveform.shape[1]
        if original_channels > 1:
            print(f"[FL ClearVoice] Converting {original_channels} channels to mono...")
            waveform = ensure_mono(waveform)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            tmp_input_path = tmp_in.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            tmp_output_path = tmp_out.name

        try:
            # Save input audio using soundfile (torchaudio.save requires torchcodec in newer versions)
            # waveform is [batch, channels, samples] -> [channels, samples] -> [samples, channels] for soundfile
            audio_to_save = waveform.squeeze(0).cpu().numpy()
            # soundfile expects [samples, channels] format
            if audio_to_save.ndim == 1:
                audio_to_save = audio_to_save  # Already 1D for mono
            else:
                audio_to_save = audio_to_save.T  # [channels, samples] -> [samples, channels]
            sf.write(tmp_input_path, audio_to_save, orig_sr)

            print(f"[FL ClearVoice] Processing with VoiceFixer...")
            vf.restore(
                input=tmp_input_path,
                output=tmp_output_path,
                cuda=use_cuda,
                mode=0  # Default mode
            )

            # Load output audio
            processed_audio, processed_sr = sf.read(tmp_output_path)

            # Convert to tensor [channels, samples]
            if processed_audio.ndim == 1:
                processed_waveform = torch.from_numpy(processed_audio).float().unsqueeze(0)
            else:
                processed_waveform = torch.from_numpy(processed_audio.T).float()

            # Restore stereo if needed
            if original_channels > 1:
                processed_waveform = processed_waveform.repeat(original_channels, 1)

            return processed_waveform, processed_sr

        finally:
            # Cleanup temp files
            if os.path.exists(tmp_input_path):
                os.unlink(tmp_input_path)
            if os.path.exists(tmp_output_path):
                os.unlink(tmp_output_path)

    def process_audio(
        self,
        model: dict,
        audio: dict,
    ) -> Tuple[dict, str]:
        """
        Process audio using the loaded model.

        Automatically adapts to the model backend and type.

        Args:
            model: Model info dict from Model Loader
            audio: Input ComfyUI AUDIO

        Returns:
            Tuple of (processed AUDIO, status message)
        """
        model_name = model["model_name"]
        backend = model.get("backend", MODEL_BACKEND.get(model_name, "clearvoice"))
        output_sr = model["output_sample_rate"]

        print(f"\n{'='*60}")
        print(f"[FL ClearVoice] Processing audio...")
        print(f"[FL ClearVoice] Model: {model_name}")
        print(f"[FL ClearVoice] Backend: {backend}")
        print(f"{'='*60}\n")

        try:
            # Get input audio
            waveform, orig_sr = comfyui_audio_to_tensor(audio)
            input_duration = get_audio_duration(audio)

            print(f"[FL ClearVoice] Input: {orig_sr}Hz, {input_duration:.2f}s")
            print(f"[FL ClearVoice] Output target: {output_sr}Hz")

            # Route to appropriate backend
            if backend == "clearvoice":
                processed_waveform, final_sr = self._process_clearvoice(model, waveform, orig_sr)
            elif backend == "resemble_enhance":
                processed_waveform, final_sr = self._process_resemble_enhance(model, waveform, orig_sr)
            elif backend == "voicefixer":
                processed_waveform, final_sr = self._process_voicefixer(model, waveform, orig_sr)
            else:
                raise ValueError(f"Unknown backend: {backend}")

            # Convert to ComfyUI format
            output_audio = tensor_to_comfyui_audio(processed_waveform, final_sr)
            output_duration = processed_waveform.shape[-1] / final_sr

            # Generate message based on model type
            if model_name in SR_MODELS:
                mode = "Super-resolved"
            elif model_name in RESTORATION_MODELS:
                mode = "Restored"
            elif model_name in DENOISE_ONLY_MODELS:
                mode = "Denoised"
            else:
                mode = "Enhanced"

            message = (
                f"{mode} with {model_name} | "
                f"{orig_sr}Hz -> {final_sr}Hz | "
                f"Duration: {output_duration:.2f}s"
            )

            print(f"\n{'='*60}")
            print(f"[FL ClearVoice] Success!")
            print(f"[FL ClearVoice] {message}")
            print(f"{'='*60}\n")

            return (output_audio, message)

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"[FL ClearVoice] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")

            # Return empty audio on error
            empty_audio = tensor_to_comfyui_audio(
                torch.zeros(1, 1, 48000),
                48000
            )
            return (empty_audio, f"Error: {str(e)}")


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FL_ClearVoice_Process": FL_ClearVoice_Process,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_ClearVoice_Process": "FL ClearVoice Process",
}
