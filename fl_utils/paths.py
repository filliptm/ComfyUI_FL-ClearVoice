"""
Centralized path management for FL ClearVoice models.
All models download to: ComfyUI/models/clear_voice/
"""

from pathlib import Path
import os


def get_clearvoice_models_dir() -> Path:
    """
    Get the centralized models directory for all FL ClearVoice models.

    Returns:
        Path to ComfyUI/models/clear_voice/
    """
    # Navigate: fl_utils -> ComfyUI_FL-ClearVoice -> custom_nodes -> ComfyUI
    current_dir = Path(__file__).parent
    comfyui_root = current_dir.parent.parent.parent

    models_dir = comfyui_root / "models" / "clear_voice"

    # Verify we're in a valid ComfyUI structure
    if not (comfyui_root / "custom_nodes").exists():
        # Fallback: use a local directory if not in ComfyUI structure
        models_dir = current_dir.parent / "models"

    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_resemble_enhance_dir() -> Path:
    """Get the directory for Resemble-Enhance models."""
    path = get_clearvoice_models_dir() / "resemble_enhance"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_clearvoice_backend_dir(model_name: str) -> Path:
    """Get the directory for a specific ClearVoice model."""
    path = get_clearvoice_models_dir() / "clearvoice" / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_voicefixer_dir() -> Path:
    """Get the directory for VoiceFixer models."""
    path = get_clearvoice_models_dir() / "voicefixer"
    path.mkdir(parents=True, exist_ok=True)
    return path
