"""
HParams for Resemble-Enhance models - uses the original classes directly.
The hparams modules don't have deepspeed dependencies.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Import the original hparams classes from resemble_enhance
# These don't have deepspeed dependencies - only the train.py files do
from resemble_enhance.hparams import HParams as BaseHParams
from resemble_enhance.enhancer.hparams import HParams as EnhancerHParams
from resemble_enhance.denoiser.hparams import HParams as DenoiserHParams

__all__ = ["BaseHParams", "EnhancerHParams", "DenoiserHParams"]
