"""
Resemble-Enhance inference module - standalone version without deepspeed dependency.

This module extracts only the inference components from the resemble-enhance package,
bypassing the train.py files that import deepspeed.
"""

import sys

# Handle both package-style and direct imports
try:
    from .inference import denoise, enhance, load_enhancer, clear_cache
except ImportError:
    # Fallback for when loaded outside package context
    if "fl_resemble_enhance.inference" in sys.modules:
        inference_module = sys.modules["fl_resemble_enhance.inference"]
        denoise = inference_module.denoise
        enhance = inference_module.enhance
        load_enhancer = inference_module.load_enhancer
        clear_cache = inference_module.clear_cache
    else:
        from inference import denoise, enhance, load_enhancer, clear_cache

__all__ = ["denoise", "enhance", "load_enhancer", "clear_cache"]
