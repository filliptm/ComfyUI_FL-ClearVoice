"""
FL ClearVoice - Speech Enhancement, Restoration & Super-Resolution for ComfyUI

Combines multiple state-of-the-art audio processing models:
- ClearVoice (Alibaba): https://github.com/modelscope/ClearerVoice-Studio
- Resemble-Enhance (Resemble AI): https://github.com/resemble-ai/resemble-enhance
- VoiceFixer: https://github.com/haoheliu/voicefixer
"""

import sys
import os
import importlib.util

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

def import_module_from_path(module_name, file_path):
    """Import a module from an explicit file path to avoid naming conflicts."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import utilities first (needed by nodes)
# Load paths module first (required by model_manager and fl_resemble_enhance)
cv_paths = import_module_from_path(
    "cv_paths",
    os.path.join(current_dir, "fl_utils", "paths.py")
)
cv_download_utils = import_module_from_path(
    "cv_download_utils",
    os.path.join(current_dir, "fl_utils", "download_utils.py")
)
cv_audio_utils = import_module_from_path(
    "cv_audio_utils",
    os.path.join(current_dir, "fl_utils", "audio_utils.py")
)
cv_model_manager = import_module_from_path(
    "cv_model_manager",
    os.path.join(current_dir, "fl_utils", "model_manager.py")
)

# Import nodes using explicit paths to avoid conflict with other FL node packs
cv_model_loader = import_module_from_path(
    "cv_model_loader",
    os.path.join(current_dir, "fl_nodes", "model_loader.py")
)
cv_process = import_module_from_path(
    "cv_process",
    os.path.join(current_dir, "fl_nodes", "process.py")
)

# Extract node classes
FL_ClearVoice_ModelLoader = cv_model_loader.FL_ClearVoice_ModelLoader
FL_ClearVoice_Process = cv_process.FL_ClearVoice_Process

# Node class mappings - This is how ComfyUI discovers nodes
NODE_CLASS_MAPPINGS = {
    "FL_ClearVoice_ModelLoader": FL_ClearVoice_ModelLoader,
    "FL_ClearVoice_Process": FL_ClearVoice_Process,
}

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_ClearVoice_ModelLoader": "FL ClearVoice Model Loader",
    "FL_ClearVoice_Process": "FL ClearVoice Process",
}

# ASCII art banner
ascii_art = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ███████╗██╗       ██████╗██╗     ███████╗ █████╗ ██████╗                    ║
║   ██╔════╝██║      ██╔════╝██║     ██╔════╝██╔══██╗██╔══██╗                   ║
║   █████╗  ██║      ██║     ██║     █████╗  ███████║██████╔╝                   ║
║   ██╔══╝  ██║      ██║     ██║     ██╔══╝  ██╔══██║██╔══██╗                   ║
║   ██║     ███████╗ ╚██████╗███████╗███████╗██║  ██║██║  ██║                   ║
║   ╚═╝     ╚══════╝  ╚═════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝                   ║
║                                                                               ║
║   ██╗   ██╗ ██████╗ ██╗ ██████╗███████╗                                       ║
║   ██║   ██║██╔═══██╗██║██╔════╝██╔════╝                                       ║
║   ██║   ██║██║   ██║██║██║     █████╗                                         ║
║   ╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝                                         ║
║    ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗                                       ║
║     ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝                                       ║
║                                                                               ║
║              Speech Enhancement & Super-Resolution for ComfyUI                ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

print(ascii_art)
print("=" * 80)
print("FL ClearVoice Custom Nodes Loaded")
print("Version: 2.0.0")
print("Nodes: FL_ClearVoice_ModelLoader, FL_ClearVoice_Process")
print("")
print("ClearVoice Models:")
print("  - MossFormer2_SE_48K, FRCRN_SE_16K, MossFormerGAN_SE_16K, MossFormer2_SR_48K")
print("Resemble-Enhance Models:")
print("  - Resemble_Enhance (full restoration), Resemble_Denoise (denoise only)")
print("VoiceFixer Models:")
print("  - VoiceFixer (all-in-one restoration)")
print("=" * 80)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
