"""
Model manager for FL-ClearVoice
Handles ClearVoice model loading and caching

All models download to: ComfyUI/models/clear_voice/
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Import from cv_paths module (loaded by __init__.py before this module)
# This avoids relative import issues when loaded via import_module_from_path
cv_paths = sys.modules.get('cv_paths')
if cv_paths is None:
    # Fallback: direct import if running standalone
    from pathlib import Path as _Path
    _current_dir = _Path(__file__).parent
    import importlib.util
    _spec = importlib.util.spec_from_file_location("cv_paths", _current_dir / "paths.py")
    cv_paths = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(cv_paths)

get_clearvoice_models_dir = cv_paths.get_clearvoice_models_dir
get_clearvoice_backend_dir = cv_paths.get_clearvoice_backend_dir
get_voicefixer_dir = cv_paths.get_voicefixer_dir


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


def _patch_clearvoice_download():
    """
    Monkey-patch ClearVoice's download_model method to use our progress bar utility.

    The original method uses huggingface_hub.snapshot_download which shows
    "Fetching X files" without detailed per-file progress. We replace it with
    our download utility that shows progress bars for each file.
    """
    try:
        import clearvoice.networks as networks

        # Get original method (for fallback)
        original_download = networks.SpeechModel.download_model

        def patched_download_model(self, model_name):
            """Download ClearVoice model with progress bars."""
            checkpoint_dir = self.args.checkpoint_dir

            # Try our download utility first
            try:
                # Import download utility - use absolute import path
                import importlib.util
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                download_utils_path = os.path.join(current_dir, "download_utils.py")

                if os.path.exists(download_utils_path):
                    spec = importlib.util.spec_from_file_location("download_utils", download_utils_path)
                    download_utils = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(download_utils)

                    from huggingface_hub import HfApi

                    repo_id = f'alibabasglab/{model_name}'
                    local_dir = Path(checkpoint_dir)
                    local_dir.mkdir(parents=True, exist_ok=True)

                    # Get list of files in the repo
                    api = HfApi()
                    try:
                        repo_info = api.repo_info(repo_id=repo_id, files_metadata=True)
                        filenames = [f.rfilename for f in repo_info.siblings if f.rfilename]
                    except Exception as e:
                        print(f"[FL ClearVoice] Could not fetch file list: {e}")
                        # Fallback to original download
                        return original_download(self, model_name)

                    print(f"[FL ClearVoice] Downloading {model_name} model files...")

                    downloader = download_utils.MultiFileDownloader(prefix="[FL ClearVoice]")
                    downloader.download_hf_files(
                        repo_id=repo_id,
                        filenames=filenames,
                        local_dir=local_dir
                    )

                    return True
                else:
                    return original_download(self, model_name)

            except Exception as e:
                print(f"[FL ClearVoice] Progress download failed: {e}, using standard download...")
                return original_download(self, model_name)

        # Apply the patch
        networks.SpeechModel.download_model = patched_download_model
        return True

    except Exception as e:
        print(f"[FL ClearVoice] Warning: Could not patch download method: {e}")
        return False


# Apply download patch on module load
_DOWNLOAD_PATCH_APPLIED = _patch_clearvoice_download()


def _patch_clearvoice_checkpoint_dir():
    """
    Monkey-patch ClearVoice's network_wrapper to use our centralized model directory.

    ClearVoice downloads models to `checkpoint_dir` which defaults to relative paths
    like 'checkpoints/MossFormer2_SE_48K'. We patch it to use our centralized location:
    ComfyUI/models/clear_voice/clearvoice/{model_name}/

    The key insight is that the original __call__ method:
    1. Calls load_args_* (which sets self.args.checkpoint_dir to default path)
    2. Then instantiates the network class (which downloads/loads using checkpoint_dir)

    We need to intercept BETWEEN steps 1 and 2 to override checkpoint_dir.
    We do this by patching the load_args_* methods to apply our override after they run.
    """
    try:
        from clearvoice.network_wrapper import network_wrapper

        # Store references to original load_args methods
        original_load_args_se = network_wrapper.load_args_se
        original_load_args_ss = network_wrapper.load_args_ss
        original_load_args_sr = network_wrapper.load_args_sr
        original_load_args_tse = network_wrapper.load_args_tse

        def _override_checkpoint_dir(self):
            """Override checkpoint_dir to our centralized location after args are loaded."""
            if hasattr(self, 'model_name') and hasattr(self, 'args'):
                our_checkpoint_dir = get_clearvoice_backend_dir(self.model_name)
                self.args.checkpoint_dir = str(our_checkpoint_dir)

        def patched_load_args_se(self):
            result = original_load_args_se(self)
            _override_checkpoint_dir(self)
            return result

        def patched_load_args_ss(self):
            result = original_load_args_ss(self)
            _override_checkpoint_dir(self)
            return result

        def patched_load_args_sr(self):
            result = original_load_args_sr(self)
            _override_checkpoint_dir(self)
            return result

        def patched_load_args_tse(self):
            result = original_load_args_tse(self)
            _override_checkpoint_dir(self)
            return result

        # Apply the patches
        network_wrapper.load_args_se = patched_load_args_se
        network_wrapper.load_args_ss = patched_load_args_ss
        network_wrapper.load_args_sr = patched_load_args_sr
        network_wrapper.load_args_tse = patched_load_args_tse

        print(f"[FL ClearVoice] Model download path: {get_clearvoice_models_dir()}")
        return True

    except Exception as e:
        print(f"[FL ClearVoice] Warning: Could not patch checkpoint dir: {e}")
        return False


# Apply checkpoint dir patch on module load
_CHECKPOINT_PATCH_APPLIED = _patch_clearvoice_checkpoint_dir()

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
    # We use importlib to properly load the module and its dependencies
    try:
        import importlib.util
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        fl_resemble_dir = os.path.join(current_dir, "fl_resemble_enhance")

        # First, manually load the submodules that inference.py depends on
        # This ensures relative imports work by having all modules in sys.modules

        # Load hparams module
        hparams_path = os.path.join(fl_resemble_dir, "hparams.py")
        hparams_spec = importlib.util.spec_from_file_location(
            "fl_resemble_enhance.hparams", hparams_path,
            submodule_search_locations=[fl_resemble_dir]
        )
        hparams_module = importlib.util.module_from_spec(hparams_spec)
        sys.modules["fl_resemble_enhance.hparams"] = hparams_module
        hparams_spec.loader.exec_module(hparams_module)

        # Load denoiser module
        denoiser_path = os.path.join(fl_resemble_dir, "denoiser.py")
        denoiser_spec = importlib.util.spec_from_file_location(
            "fl_resemble_enhance.denoiser", denoiser_path,
            submodule_search_locations=[fl_resemble_dir]
        )
        denoiser_module = importlib.util.module_from_spec(denoiser_spec)
        sys.modules["fl_resemble_enhance.denoiser"] = denoiser_module
        denoiser_spec.loader.exec_module(denoiser_module)

        # Load enhancer module
        enhancer_path = os.path.join(fl_resemble_dir, "enhancer.py")
        enhancer_spec = importlib.util.spec_from_file_location(
            "fl_resemble_enhance.enhancer", enhancer_path,
            submodule_search_locations=[fl_resemble_dir]
        )
        enhancer_module = importlib.util.module_from_spec(enhancer_spec)
        sys.modules["fl_resemble_enhance.enhancer"] = enhancer_module
        enhancer_spec.loader.exec_module(enhancer_module)

        # Load inference module
        inference_path = os.path.join(fl_resemble_dir, "inference.py")
        inference_spec = importlib.util.spec_from_file_location(
            "fl_resemble_enhance.inference", inference_path,
            submodule_search_locations=[fl_resemble_dir]
        )
        inference_module = importlib.util.module_from_spec(inference_spec)
        sys.modules["fl_resemble_enhance.inference"] = inference_module
        inference_spec.loader.exec_module(inference_module)

        # Now create the package module and set it up
        package_spec = importlib.util.spec_from_file_location(
            "fl_resemble_enhance",
            os.path.join(fl_resemble_dir, "__init__.py"),
            submodule_search_locations=[fl_resemble_dir]
        )
        fl_resemble_enhance = importlib.util.module_from_spec(package_spec)
        sys.modules["fl_resemble_enhance"] = fl_resemble_enhance
        package_spec.loader.exec_module(fl_resemble_enhance)

        # Get the functions
        denoise = fl_resemble_enhance.denoise
        enhance = fl_resemble_enhance.enhance

    except Exception as e:
        import traceback
        traceback.print_exc()
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


def _setup_voicefixer_path():
    """
    Set up VoiceFixer to use our centralized model directory.

    VoiceFixer hardcodes ~/.cache/voicefixer/ as its model path.
    We download models to our location with a progress bar, then create a symlink.
    """
    import os

    our_vf_dir = get_voicefixer_dir()
    expected_cache_dir = Path.home() / ".cache" / "voicefixer"
    expected_ckpt = expected_cache_dir / "analysis_module" / "checkpoints" / "vf.ckpt"
    our_ckpt_dir = our_vf_dir / "analysis_module" / "checkpoints"
    our_ckpt = our_ckpt_dir / "vf.ckpt"

    voicefixer_url = "https://zenodo.org/record/5600188/files/vf.ckpt?download=1"

    # If model exists in our location but not in expected location, create symlink
    if our_ckpt.exists() and not expected_ckpt.exists():
        print(f"[FL ClearVoice] VoiceFixer model found at {our_vf_dir}")
        expected_cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Create symlink from ~/.cache/voicefixer -> our_vf_dir
            if expected_cache_dir.is_symlink():
                expected_cache_dir.unlink()
            elif expected_cache_dir.exists():
                # Directory exists but isn't a symlink - leave it alone
                return
            expected_cache_dir.symlink_to(our_vf_dir)
        except Exception as e:
            print(f"[FL ClearVoice] Could not create symlink: {e}")

    # If model doesn't exist anywhere, download with progress bar
    if not our_ckpt.exists() and not expected_ckpt.exists():
        print(f"[FL ClearVoice] Downloading VoiceFixer model to: {our_vf_dir}")
        # Create our directory structure
        our_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Download with progress bar
        try:
            import importlib.util
            current_dir = os.path.dirname(os.path.abspath(__file__))
            download_utils_path = os.path.join(current_dir, "download_utils.py")

            if os.path.exists(download_utils_path):
                spec = importlib.util.spec_from_file_location("download_utils", download_utils_path)
                download_utils = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(download_utils)

                download_utils.download_url_with_progress(
                    url=voicefixer_url,
                    local_path=our_ckpt,
                    description="vf.ckpt (VoiceFixer)",
                    prefix="[FL ClearVoice]"
                )
            else:
                # Fallback to urllib
                print("[FL ClearVoice] Downloading VoiceFixer model (no progress available)...")
                import urllib.request
                urllib.request.urlretrieve(voicefixer_url, str(our_ckpt))
                print(f"[FL ClearVoice] VoiceFixer model downloaded")

        except Exception as e:
            print(f"[FL ClearVoice] Download with progress failed: {e}")
            # Fallback to urllib
            import urllib.request
            print("[FL ClearVoice] Downloading VoiceFixer model...")
            urllib.request.urlretrieve(voicefixer_url, str(our_ckpt))
            print(f"[FL ClearVoice] VoiceFixer model downloaded")

        # Create symlink so VoiceFixer finds the model
        expected_cache_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            if expected_cache_dir.is_symlink():
                expected_cache_dir.unlink()
            if not expected_cache_dir.exists():
                expected_cache_dir.symlink_to(our_vf_dir)
        except Exception as e:
            print(f"[FL ClearVoice] Could not create symlink: {e}")


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

    # Setup our centralized path before importing VoiceFixer
    _setup_voicefixer_path()

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
