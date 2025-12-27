"""
FL ClearVoice Model Loader Node
Loads speech enhancement, restoration, and super-resolution models
"""

from typing import Tuple
import sys

# Import from pre-loaded modules (loaded by __init__.py)
# These use unique names to avoid conflicts with other FL node packs
if "cv_model_manager" in sys.modules:
    cv_model_manager = sys.modules["cv_model_manager"]
    get_model = cv_model_manager.get_model
    ALL_MODELS = cv_model_manager.ALL_MODELS
    MODEL_BACKEND = cv_model_manager.MODEL_BACKEND
else:
    # Fallback for standalone testing
    import os
    import importlib.util
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "cv_model_manager",
        os.path.join(current_dir, "fl_utils", "model_manager.py")
    )
    cv_model_manager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cv_model_manager)
    get_model = cv_model_manager.get_model
    ALL_MODELS = cv_model_manager.ALL_MODELS
    MODEL_BACKEND = cv_model_manager.MODEL_BACKEND


class FL_ClearVoice_ModelLoader:
    """
    Loads a model for speech enhancement, restoration, or super-resolution.

    Available models:
    - MossFormer2_SE_48K: Best quality 48kHz enhancement (ClearVoice)
    - FRCRN_SE_16K: Fast 16kHz enhancement (ClearVoice)
    - MossFormerGAN_SE_16K: GAN-based 16kHz enhancement (ClearVoice)
    - MossFormer2_SR_48K: Super-resolution to 48kHz (ClearVoice)
    - Resemble_Enhance: Full restoration - denoise + enhance (Resemble AI)
    - Resemble_Denoise: Denoising only (Resemble AI)
    - VoiceFixer: All-in-one restoration - noise, reverb, clipping, bandwidth (VoiceFixer)
    """

    RETURN_TYPES = ("CLEARVOICE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "🎵FL ClearVoice/Loaders"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (ALL_MODELS, {
                    "default": "MossFormer2_SE_48K",
                    "description": "Model to load. MossFormer2_SE_48K=best quality, FRCRN_SE_16K=fast, MossFormer2_SR_48K=super-resolution"
                }),
            },
            "optional": {
                "force_reload": ("BOOLEAN", {
                    "default": False,
                    "description": "Force reload model even if cached"
                }),
            }
        }

    def load_model(
        self,
        model: str = "MossFormer2_SE_48K",
        force_reload: bool = False
    ) -> Tuple[dict]:
        """
        Load the model.

        Args:
            model: Model name
            force_reload: Force reload from disk

        Returns:
            Tuple containing model info dict
        """
        backend = MODEL_BACKEND.get(model, "unknown")

        print(f"\n{'='*60}")
        print(f"[FL ClearVoice ModelLoader] Loading model...")
        print(f"[FL ClearVoice ModelLoader] Model: {model}")
        print(f"[FL ClearVoice ModelLoader] Backend: {backend}")
        print(f"{'='*60}\n")

        try:
            # Get or load model using universal loader
            model_info = get_model(
                model_name=model,
                force_reload=force_reload
            )

            print(f"\n{'='*60}")
            print(f"[FL ClearVoice ModelLoader] Model loaded successfully!")
            print(f"[FL ClearVoice ModelLoader] Model: {model_info['model_name']}")
            in_sr = model_info.get('input_sample_rate')
            print(f"[FL ClearVoice ModelLoader] Input SR: {in_sr if in_sr else 'flexible'}")
            print(f"[FL ClearVoice ModelLoader] Output SR: {model_info['output_sample_rate']}Hz")
            print(f"{'='*60}\n")

            return (model_info,)

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"[FL ClearVoice ModelLoader] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            raise


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FL_ClearVoice_ModelLoader": FL_ClearVoice_ModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_ClearVoice_ModelLoader": "FL ClearVoice Model Loader",
}
