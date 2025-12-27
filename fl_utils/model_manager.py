"""
Model manager for FL-ClearVoice
Handles ClearVoice model loading and caching
"""

import torch
import numpy as np
from typing import Dict, Any, Optional


def _patch_clearvoice_sr_decode():
    """
    Monkey-patch ClearVoice's SR decode function to fix the squeeze bug.

    The original code does `outputs = generator_output.squeeze()` which removes
    ALL dimensions of size 1, making a batch-1 tensor 1D. Then it tries to
    index with [batch_idx,:] which fails.

    This patch replaces the problematic function with a fixed version.
    """
    try:
        from clearvoice.utils import decode_batch
        from clearvoice.utils.decode_batch import get_mel, bandwidth_sub

        def fixed_decode_one_audio_mossformer2_sr_48k(model, device, inputs, args):
            """Fixed version that handles batch_size=1 correctly."""
            b, input_len = inputs.shape

            if input_len > args.sampling_rate * args.one_time_decode_length:
                # Long audio - use sliding window (original logic, but with fix)
                window = int(args.sampling_rate * args.decode_window)
                stride = int(window * 0.75)
                t = inputs.shape[1]

                if t < window:
                    inputs = np.concatenate([inputs, np.zeros((b, window - t))], 1)
                elif t < window + stride:
                    padding = window + stride - t
                    inputs = np.concatenate([inputs, np.zeros((b, padding))], 1)
                else:
                    if (t - window) % stride != 0:
                        padding = t - (t - window) // stride * stride
                        inputs = np.concatenate([inputs, np.zeros((b, padding))], 1)

                audio = torch.from_numpy(inputs).type(torch.FloatTensor)
                t = audio.shape[1]
                outputs = torch.from_numpy(np.zeros((b, t)))
                give_up_length = (window - stride) // 2
                current_idx = 0

                while current_idx + window <= t:
                    audio_segment = audio[:, current_idx:current_idx + window]

                    for batch_idx in range(b):
                        mel_segment = get_mel(audio_segment[batch_idx:batch_idx+1, :], args)
                        if batch_idx == 0:
                            mel_segment_b = mel_segment
                        else:
                            mel_segment_b = torch.cat([mel_segment_b, mel_segment], dim=0)

                    mossformer_output_segment = model[0](mel_segment_b.to(device))
                    generator_output_segment = model[1](mossformer_output_segment)
                    generator_output_segment = torch.squeeze(generator_output_segment, 1)
                    offset = audio_segment.shape[1] - generator_output_segment.shape[1]

                    if current_idx == 0:
                        outputs[:, current_idx:current_idx + window - give_up_length] = \
                            generator_output_segment[:, :-give_up_length + offset]
                    else:
                        generator_output_segment = generator_output_segment[:, -window:]
                        outputs[:, current_idx + give_up_length:current_idx + window - give_up_length] = \
                            generator_output_segment[:, give_up_length:-give_up_length + offset]

                    current_idx += stride
            else:
                # Short audio - process at once
                audio = torch.from_numpy(inputs).type(torch.FloatTensor)
                for batch_idx in range(b):
                    mel_input = get_mel(audio[batch_idx:batch_idx+1, :], args)
                    if batch_idx == 0:
                        mel_input_b = mel_input
                    else:
                        mel_input_b = torch.cat([mel_input_b, mel_input], dim=0)

                mossformer_output = model[0](mel_input_b.to(device))
                generator_output = model[1](mossformer_output)

                # FIX: Use squeeze(1) instead of squeeze() to only remove dim 1
                # This preserves the batch dimension
                outputs = generator_output.squeeze(1)

            outputs_pred = outputs.detach().cpu().numpy()

            # FIX: Ensure outputs_pred is 2D even for batch_size=1
            if outputs_pred.ndim == 1:
                outputs_pred = outputs_pred.reshape(1, -1)

            for batch_idx in range(b):
                if batch_idx == 0:
                    outputs_rep = bandwidth_sub(inputs[batch_idx, :], outputs_pred[batch_idx, :])
                    outputs_rep = np.reshape(outputs_rep, [1, outputs_rep.shape[0]])
                else:
                    output_tmp = bandwidth_sub(inputs[batch_idx, :], outputs_pred[batch_idx, :])
                    output_tmp = np.reshape(output_tmp, [1, output_tmp.shape[0]])
                    outputs_rep = np.concatenate((outputs_rep, output_tmp), axis=0)

            return outputs_rep[:, :input_len]

        # Apply the patch
        decode_batch.decode_one_audio_mossformer2_sr_48k = fixed_decode_one_audio_mossformer2_sr_48k
        print("[FL ClearVoice] Applied SR decode bug fix patch")
        return True

    except Exception as e:
        print(f"[FL ClearVoice] Warning: Could not patch SR decode: {e}")
        return False


# Apply patch on module load
_SR_PATCH_APPLIED = _patch_clearvoice_sr_decode()

# Model configuration
# NOTE: ClearVoice internally uses these tasks:
# - speech_enhancement: for SE models (denoising)
# - speech_super_resolution: for SR models (bandwidth extension)
# - speech_separation: for SS models
# - target_speaker_extraction: for TSE models
TASK_MODELS = {
    "speech_enhancement": ["MossFormer2_SE_48K", "FRCRN_SE_16K", "MossFormerGAN_SE_16K"],
    "speech_super_resolution": ["MossFormer2_SR_48K"],
}

# Model libraries/backends
MODEL_BACKEND = {
    # ClearVoice models
    "MossFormer2_SE_48K": "clearvoice",
    "FRCRN_SE_16K": "clearvoice",
    "MossFormerGAN_SE_16K": "clearvoice",
    "MossFormer2_SR_48K": "clearvoice",
    # Resemble-Enhance models
    "Resemble_Enhance": "resemble_enhance",
    "Resemble_Denoise": "resemble_enhance",
    # VoiceFixer models
    "VoiceFixer": "voicefixer",
}

# All available models flattened
ALL_MODELS = [
    # ClearVoice
    "MossFormer2_SE_48K",
    "FRCRN_SE_16K",
    "MossFormerGAN_SE_16K",
    "MossFormer2_SR_48K",
    # Resemble-Enhance
    "Resemble_Enhance",
    "Resemble_Denoise",
    # VoiceFixer
    "VoiceFixer",
]

# Sample rates for each model
MODEL_SAMPLE_RATES = {
    "MossFormer2_SE_48K": {"input": 48000, "output": 48000},
    "FRCRN_SE_16K": {"input": 16000, "output": 16000},
    "MossFormerGAN_SE_16K": {"input": 16000, "output": 16000},
    "MossFormer2_SR_48K": {"input": None, "output": 48000},
    # Resemble-Enhance works at 44.1kHz
    "Resemble_Enhance": {"input": 44100, "output": 44100},
    "Resemble_Denoise": {"input": 44100, "output": 44100},
    # VoiceFixer outputs 44.1kHz
    "VoiceFixer": {"input": None, "output": 44100},
}

# Models that do super-resolution (bandwidth extension)
SR_MODELS = ["MossFormer2_SR_48K"]

# Models that do full restoration (denoise + enhance + bandwidth extension)
RESTORATION_MODELS = ["Resemble_Enhance", "VoiceFixer"]

# Models that only denoise
DENOISE_ONLY_MODELS = ["Resemble_Denoise"]

# Global model cache
_MODEL_CACHE: Dict[str, Any] = {}


def get_clearvoice_model(
    task: str,
    model_name: str,
    force_reload: bool = False
) -> Dict[str, Any]:
    """
    Get or load a ClearVoice model.

    Args:
        task: Task type ('speech_enhancement' or 'super_resolution')
        model_name: Name of the model to load
        force_reload: Force reload even if cached

    Returns:
        Dict with model info including 'model', 'task', 'model_name', 'sample_rates'
    """
    global _MODEL_CACHE

    # Auto-correct task based on model if mismatched
    correct_task = None
    for t, models in TASK_MODELS.items():
        if model_name in models:
            correct_task = t
            break

    if correct_task is None:
        raise ValueError(f"Unknown model: {model_name}. Available: {ALL_MODELS}")

    if correct_task != task:
        print(f"[FL ClearVoice] Auto-correcting task: {task} -> {correct_task}")
        task = correct_task

    cache_key = f"{task}_{model_name}"

    # Return cached if available and valid
    if not force_reload and cache_key in _MODEL_CACHE:
        cached = _MODEL_CACHE[cache_key]
        # Verify cached model is still valid
        if cached.get("model") is not None:
            cv = cached["model"]
            if hasattr(cv, 'models') and cv.models and cv.models[0] is not None:
                print(f"[FL ClearVoice] Using cached model: {model_name}")
                return cached
        # Invalid cache, reload
        print(f"[FL ClearVoice] Cached model invalid, reloading...")
        del _MODEL_CACHE[cache_key]

    # Validate task and model
    if task not in TASK_MODELS:
        raise ValueError(f"Unknown task: {task}. Available: {list(TASK_MODELS.keys())}")

    if model_name not in TASK_MODELS[task]:
        raise ValueError(
            f"Model {model_name} not available for task {task}. "
            f"Available: {TASK_MODELS[task]}"
        )

    print(f"[FL ClearVoice] Loading model: {model_name} for task: {task}")

    # Import ClearVoice
    try:
        from clearvoice import ClearVoice
    except ImportError:
        raise RuntimeError(
            "ClearVoice not installed. Please install with: pip install clearvoice"
        )

    # Check if we need to force CPU for MPS compatibility
    # MossFormer2_SR_48K has convolutions > 65536 channels which MPS doesn't support
    force_cpu_for_sr = False
    if model_name in SR_MODELS and torch.backends.mps.is_available():
        force_cpu_for_sr = True
        print(f"[FL ClearVoice] MPS detected - SR model will use CPU (MPS conv1d limit)")

    # Create ClearVoice instance
    cv = ClearVoice(task=task, model_names=[model_name])

    # Verify model loaded correctly
    if not hasattr(cv, 'models') or cv.models is None or len(cv.models) == 0 or cv.models[0] is None:
        raise RuntimeError(
            f"ClearVoice model {model_name} failed to load. "
            "The model download may have failed. Check your internet connection."
        )

    # Move SR model to CPU if on MPS to avoid conv1d channel limit
    if force_cpu_for_sr and cv.models[0] is not None:
        print(f"[FL ClearVoice] Moving SR model to CPU...")
        cv.models[0].device = torch.device('cpu')
        if hasattr(cv.models[0], 'model') and cv.models[0].model is not None:
            # Move the actual model weights to CPU
            for i, m in enumerate(cv.models[0].model):
                if m is not None:
                    cv.models[0].model[i] = m.to('cpu')
        print(f"[FL ClearVoice] SR model moved to CPU")

    print(f"[FL ClearVoice] Model object: {cv.models[0]}")

    # Get sample rate info
    sample_rates = MODEL_SAMPLE_RATES.get(model_name, {"input": 48000, "output": 48000})

    # Create model info dict
    model_info = {
        "model": cv,
        "task": task,
        "model_name": model_name,
        "input_sample_rate": sample_rates["input"],
        "output_sample_rate": sample_rates["output"],
    }

    # Cache it
    _MODEL_CACHE[cache_key] = model_info

    print(f"[FL ClearVoice] Model loaded successfully!")
    print(f"[FL ClearVoice] Input SR: {sample_rates['input'] or 'flexible'}, Output SR: {sample_rates['output']}")

    return model_info


def get_resemble_enhance_model(
    model_name: str,
    force_reload: bool = False
) -> Dict[str, Any]:
    """
    Get or load a Resemble-Enhance model.

    Args:
        model_name: 'Resemble_Enhance' or 'Resemble_Denoise'
        force_reload: Force reload even if cached

    Returns:
        Dict with model info
    """
    global _MODEL_CACHE

    cache_key = f"resemble_{model_name}"

    # Return cached if available
    if not force_reload and cache_key in _MODEL_CACHE:
        print(f"[FL ClearVoice] Using cached model: {model_name}")
        return _MODEL_CACHE[cache_key]

    print(f"[FL ClearVoice] Loading Resemble-Enhance model: {model_name}")

    # Import from our local standalone module (bypasses deepspeed dependency)
    # Named fl_resemble_enhance to avoid conflict with the original package
    try:
        import sys
        import os

        # Get the path to fl_utils directory and add it to sys.path temporarily
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        # Import from our renamed local module
        from fl_resemble_enhance import denoise, enhance
    except Exception as e:
        # Fallback error message
        raise RuntimeError(
            f"Failed to load Resemble-Enhance inference module: {e}\n"
            "Make sure the resemble-enhance package components are installed."
        )

    # Determine device
    # NOTE: Resemble-Enhance has convolutions > 65536 channels which MPS doesn't support
    # So we force CPU on Mac to avoid the "Output channels > 65536 not supported" error
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "cpu"  # Force CPU on MPS due to conv1d channel limit
        print(f"[FL ClearVoice] MPS detected - Resemble-Enhance will use CPU (MPS conv1d channel limit)")
    else:
        device = "cpu"

    print(f"[FL ClearVoice] Resemble-Enhance will use device: {device}")

    # Get sample rate info
    sample_rates = MODEL_SAMPLE_RATES.get(model_name, {"input": 44100, "output": 44100})

    # Create model info dict
    # For Resemble-Enhance, we store the functions rather than a model object
    model_info = {
        "model": None,  # No model object, we use functions directly
        "backend": "resemble_enhance",
        "model_name": model_name,
        "denoise_func": denoise,
        "enhance_func": enhance,
        "device": device,
        "input_sample_rate": sample_rates["input"],
        "output_sample_rate": sample_rates["output"],
    }

    # Cache it
    _MODEL_CACHE[cache_key] = model_info

    print(f"[FL ClearVoice] Resemble-Enhance loaded successfully!")
    return model_info


def get_voicefixer_model(
    force_reload: bool = False
) -> Dict[str, Any]:
    """
    Get or load VoiceFixer model.

    Args:
        force_reload: Force reload even if cached

    Returns:
        Dict with model info
    """
    global _MODEL_CACHE

    cache_key = "voicefixer"

    # Return cached if available
    if not force_reload and cache_key in _MODEL_CACHE:
        cached = _MODEL_CACHE[cache_key]
        if cached.get("model") is not None:
            print(f"[FL ClearVoice] Using cached VoiceFixer model")
            return cached

    print(f"[FL ClearVoice] Loading VoiceFixer model...")

    # Import VoiceFixer
    try:
        from voicefixer import VoiceFixer
    except ImportError:
        raise RuntimeError(
            "voicefixer not installed. Please install with: pip install voicefixer"
        )

    # Determine if CUDA is available
    use_cuda = torch.cuda.is_available()
    # Note: VoiceFixer doesn't support MPS natively

    print(f"[FL ClearVoice] VoiceFixer will use CUDA: {use_cuda}")

    # Create VoiceFixer instance
    vf = VoiceFixer()

    # Get sample rate info
    sample_rates = MODEL_SAMPLE_RATES.get("VoiceFixer", {"input": None, "output": 44100})

    # Create model info dict
    model_info = {
        "model": vf,
        "backend": "voicefixer",
        "model_name": "VoiceFixer",
        "use_cuda": use_cuda,
        "input_sample_rate": sample_rates["input"],
        "output_sample_rate": sample_rates["output"],
    }

    # Cache it
    _MODEL_CACHE[cache_key] = model_info

    print(f"[FL ClearVoice] VoiceFixer loaded successfully!")
    return model_info


def get_model(
    model_name: str,
    force_reload: bool = False
) -> Dict[str, Any]:
    """
    Universal model loader - routes to the appropriate backend.

    Args:
        model_name: Name of the model to load
        force_reload: Force reload even if cached

    Returns:
        Dict with model info
    """
    if model_name not in ALL_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {ALL_MODELS}")

    backend = MODEL_BACKEND.get(model_name, "clearvoice")

    if backend == "clearvoice":
        # Determine task for ClearVoice models
        task = None
        for t, models in TASK_MODELS.items():
            if model_name in models:
                task = t
                break
        return get_clearvoice_model(task=task, model_name=model_name, force_reload=force_reload)

    elif backend == "resemble_enhance":
        return get_resemble_enhance_model(model_name=model_name, force_reload=force_reload)

    elif backend == "voicefixer":
        return get_voicefixer_model(force_reload=force_reload)

    else:
        raise ValueError(f"Unknown backend: {backend} for model: {model_name}")


def clear_cache():
    """Clear the model cache to free memory."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()

    # Force garbage collection
    import gc
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[FL ClearVoice] Model cache cleared")


def get_model_for_task(task: str) -> list:
    """Get available models for a task."""
    return TASK_MODELS.get(task, [])


def get_all_enhancement_models() -> list:
    """Get all speech enhancement models."""
    return TASK_MODELS.get("speech_enhancement", [])


def get_all_super_resolution_models() -> list:
    """Get all super-resolution models."""
    return TASK_MODELS.get("super_resolution", [])
